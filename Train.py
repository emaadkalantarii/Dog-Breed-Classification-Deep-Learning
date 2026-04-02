"""
train.py
--------
Training pipeline for dog breed classification using ResNet50 transfer learning.

Training configuration is defined in TrainingConfig. The Trainer class handles
the full training loop including:
    - Model and optimiser setup
    - Mixed precision training (AMP)
    - Gradient clipping
    - Epoch-level validation
    - Best-model checkpointing
    - Training history logging
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataset import create_dataloaders
from model import ModelFactory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters and paths for a training run."""

    # Data
    data_dir:         str = "processed_data"
    batch_size:       int = 64
    num_workers:      int = 4

    # Model
    model_name:               str  = "resnet50"
    freeze_feature_extractor: bool = True

    # Optimiser
    learning_rate:  float = 1e-3
    weight_decay:   float = 1e-2

    # Scheduler
    num_epochs:     int   = 50
    scheduler_t_max: int  = 50    # T_max for CosineAnnealingLR

    # Regularisation
    gradient_clipping: bool  = True
    max_grad_norm:     float = 1.0
    mixed_precision:   bool  = True

    # Checkpointing
    checkpoint_dir:    str = "checkpoints"
    experiment_name:   str = "dog_breed_resnet50"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates the full training and validation loop.

    Args:
        config: TrainingConfig instance with all hyperparameters.
        device: torch.device to run training on (auto-detected if None).
    """

    def __init__(self, config: TrainingConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.best_val_accuracy = 0.0
        self.history: Dict = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        # Mixed precision scaler
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if config.mixed_precision and self.device.type == "cuda"
            else None
        )

        self.checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialised | device={self.device}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Loads class mappings, builds model, optimiser, scheduler, and dataloaders."""
        mapping_path = Path(self.config.data_dir) / "class_mappings.json"
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"class_mappings.json not found at {mapping_path}. "
                "Run preprocessing.py first."
            )
        with open(mapping_path, "r") as f:
            class_mappings = json.load(f)

        num_classes = class_mappings["num_classes"]
        logger.info(f"Number of classes: {num_classes}")

        # Model
        factory     = ModelFactory(
            num_classes=num_classes,
            model_name=self.config.model_name,
            pretrained=True,
            freeze_feature_extractor=self.config.freeze_feature_extractor,
        )
        self.model = factory.create_model().to(self.device)
        factory.get_model_summary(self.model)

        # Loss, optimiser, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.scheduler_t_max,
        )

        # DataLoaders
        self.loaders = create_dataloaders(
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        logger.info(
            f"Setup complete | lr={self.config.learning_rate}, "
            f"batch_size={self.config.batch_size}, "
            f"epochs={self.config.num_epochs}"
        )

    # ------------------------------------------------------------------
    # Epoch runner
    # ------------------------------------------------------------------

    def _run_epoch(self, phase: str, epoch: int) -> Tuple[float, float]:
        """Runs one training or validation epoch."""
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[phase]

        running_loss      = 0.0
        running_corrects  = 0
        total_samples     = 0

        pbar = tqdm(loader, desc=f"{phase.capitalize():5s} [{epoch + 1}/{self.config.num_epochs}]")

        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if is_train:
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss    = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    if self.config.gradient_clipping:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss    = self.criterion(outputs, labels)
                    loss.backward()
                    if self.config.gradient_clipping:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)
                            loss    = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(inputs)
                        loss    = self.criterion(outputs, labels)

            _, preds          = torch.max(outputs, 1)
            running_loss     += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples    += inputs.size(0)

            pbar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{running_corrects / total_samples:.4f}",
            )

        epoch_loss = running_loss / total_samples
        epoch_acc  = running_corrects / total_samples
        logger.info(f"{phase.capitalize():5s} | loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        return epoch_loss, epoch_acc

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool) -> None:
        """Saves the latest checkpoint and, if improved, the best-model checkpoint."""
        state = {
            "epoch":               epoch + 1,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_accuracy":   self.best_val_accuracy,
            "history":             self.history,
        }

        torch.save(state, self.checkpoint_dir / "latest_checkpoint.pth")

        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pth")
            logger.info(f"Best model saved — val_acc={val_acc:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """
        Runs the full training loop for config.num_epochs epochs.

        Returns:
            history: Dict with train/val loss and accuracy per epoch.
        """
        logger.info(
            f"Starting training | {self.config.num_epochs} epochs, "
            f"device={self.device}"
        )

        for epoch in range(self.config.num_epochs):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch("train", epoch)
            val_loss,   val_acc   = self._run_epoch("val",   epoch)

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            is_best = val_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_acc

            self._save_checkpoint(epoch, val_acc, is_best)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} "
                f"| {elapsed:.0f}s "
                f"| best_val_acc={self.best_val_accuracy:.4f}"
            )

        logger.info(
            f"Training complete. Best validation accuracy: "
            f"{self.best_val_accuracy:.4f}"
        )
        return self.history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  Dog Breed Classification — Training Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")

    config = TrainingConfig(
        data_dir                 = "processed_data",
        batch_size               = 64,
        num_workers              = 4,
        model_name               = "resnet50",
        freeze_feature_extractor = True,   # Set False to fine-tune all layers
        learning_rate            = 1e-3,
        weight_decay             = 1e-2,
        num_epochs               = 50,
        scheduler_t_max          = 50,
        gradient_clipping        = True,
        max_grad_norm            = 1.0,
        mixed_precision          = True,
        checkpoint_dir           = "checkpoints",
        experiment_name          = "dog_breed_resnet50",
    )

    trainer = Trainer(config, device)
    trainer.setup()
    history = trainer.train()

    print("\n" + "=" * 60)
    print(f"  Best validation accuracy : {trainer.best_val_accuracy:.4f}")
    print(f"  Checkpoints saved to     : {trainer.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
