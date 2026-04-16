"""
evaluate.py
-----------
Model evaluation and metrics reporting for dog breed classification.

Loads the best model checkpoint, runs inference on the test set, and reports:
    - Overall accuracy (top-1, top-3, top-5)
    - Macro precision, recall, and F1-score
    - Per-class classification report
    - Confusion matrix (saved as PNG)
    - Training / validation curves (saved as PNG)

All outputs are saved to the evaluation_results/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from dataset import create_dataloaders
from model import ModelFactory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Loads a trained model checkpoint and evaluates it on the test set.

    Args:
        checkpoint_path:      Path to the best_model.pth checkpoint file.
        data_dir:             Root directory of processed data.
        output_dir:           Directory to save evaluation outputs.
        batch_size:           Batch size for inference.
        num_workers:          DataLoader workers.
        device:               Inference device (auto-detected if None).
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str            = "processed_data",
        output_dir: str          = "evaluation_results",
        batch_size: int          = 64,
        num_workers: int         = 4,
        device: Optional[str]   = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir        = Path(data_dir)
        self.output_dir      = Path(output_dir)
        self.batch_size      = batch_size
        self.num_workers     = num_workers

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluator | device={self.device}")
        logger.info(f"Checkpoint: {self.checkpoint_path}")

        self._load_class_mappings()
        self._load_model()
        self._load_dataloaders()

    def _load_class_mappings(self) -> None:
        mapping_path = self.data_dir / "class_mappings.json"
        with open(mapping_path, "r") as f:
            mappings = json.load(f)
        self.class_to_idx: Dict[str, int] = mappings["class_to_idx"]
        self.idx_to_class: Dict[int, str] = {
            int(k): v for k, v in mappings["idx_to_class"].items()
        }
        self.num_classes: int = mappings["num_classes"]
        logger.info(f"Class mappings loaded: {self.num_classes} classes")

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        factory    = ModelFactory(num_classes=self.num_classes, model_name="resnet50",
                                  pretrained=False, freeze_feature_extractor=False)
        self.model = factory.create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.history = checkpoint.get("history", {})
        logger.info(
            f"Model loaded | best_val_acc="
            f"{checkpoint.get('best_val_accuracy', 'N/A')}"
        )

    def _load_dataloaders(self) -> None:
        self.loaders = create_dataloaders(
            data_dir     = str(self.data_dir),
            batch_size   = self.batch_size,
            num_workers  = self.num_workers,
            pin_memory   = self.device.type == "cuda",
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs inference on the test set.

        Returns:
            all_labels:  Ground-truth class indices (N,).
            all_preds:   Top-1 predicted class indices (N,).
            all_probs:   Softmax probability distributions (N, num_classes).
        """
        all_labels, all_preds, all_probs = [], [], []

        logger.info("Running inference on test set...")
        with torch.no_grad():
            for inputs, labels in tqdm(self.loaders["test"], desc="Evaluating"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs   = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        return (
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        labels: np.ndarray,
        preds:  np.ndarray,
        probs:  np.ndarray,
    ) -> Dict:
        """Computes and logs all evaluation metrics."""

        top1  = accuracy_score(labels, preds)
        top3  = top_k_accuracy_score(labels, probs, k=3)
        top5  = top_k_accuracy_score(labels, probs, k=5)
        prec  = precision_score(labels, preds, average="macro", zero_division=0)
        rec   = recall_score(labels, preds, average="macro", zero_division=0)
        f1    = f1_score(labels, preds, average="macro", zero_division=0)

        metrics = {
            "top1_accuracy":  round(top1, 4),
            "top3_accuracy":  round(top3, 4),
            "top5_accuracy":  round(top5, 4),
            "macro_precision": round(prec, 4),
            "macro_recall":   round(rec, 4),
            "macro_f1":       round(f1, 4),
        }

        logger.info("=" * 50)
        logger.info("  Test Set Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"  Top-1 Accuracy  : {top1:.4f} ({top1 * 100:.2f}%)")
        logger.info(f"  Top-3 Accuracy  : {top3:.4f} ({top3 * 100:.2f}%)")
        logger.info(f"  Top-5 Accuracy  : {top5:.4f} ({top5 * 100:.2f}%)")
        logger.info(f"  Macro Precision : {prec:.4f}")
        logger.info(f"  Macro Recall    : {rec:.4f}")
        logger.info(f"  Macro F1-Score  : {f1:.4f}")
        logger.info("=" * 50)

        # Save metrics JSON
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Per-class report
        class_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        report = classification_report(
            labels, preds, target_names=class_names, zero_division=0
        )
        with open(self.output_dir / "classification_report.txt", "w") as f:
            f.write(report)

        logger.info("Metrics saved to evaluation_results/")
        return metrics

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_training_curves(self) -> None:
        """Saves training and validation accuracy/loss curves."""
        if not self.history:
            logger.warning("No training history found in checkpoint — skipping curves.")
            return

        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History — Dog Breed Classification (ResNet50)", fontsize=14)

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss",   color="steelblue")
        axes[0].plot(epochs, self.history["val_loss"],   label="Val Loss",     color="tomato", linestyle="--")
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        best_val = max(self.history["val_acc"])
        axes[1].plot(epochs, self.history["train_acc"], label="Train Accuracy", color="steelblue")
        axes[1].plot(epochs, self.history["val_acc"],   label="Val Accuracy",   color="tomato", linestyle="--")
        axes[1].axhline(y=best_val, color="green", linestyle=":", alpha=0.7,
                        label=f"Best Val Acc: {best_val:.4f}")
        axes[1].set_title("Accuracy over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = self.output_dir / "training_curves.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Training curves saved to {out_path}")

    def plot_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray,
                               top_n: int = 30) -> None:
        """
        Saves a confusion matrix for the top_n most frequent classes.

        Args:
            labels: Ground-truth class indices.
            preds:  Predicted class indices.
            top_n:  Number of most frequent classes to display.
        """
        from collections import Counter
        top_classes = [cls for cls, _ in Counter(labels).most_common(top_n)]
        mask        = np.isin(labels, top_classes)
        sub_labels  = labels[mask]
        sub_preds   = preds[mask]

        cm         = confusion_matrix(sub_labels, sub_preds, labels=top_classes)
        class_names = [self.idx_to_class[c] for c in top_classes]

        fig, ax = plt.subplots(figsize=(18, 16))
        sns.heatmap(
            cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax
        )
        ax.set_title(f"Confusion Matrix (Top {top_n} Classes)", fontsize=14)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)

        plt.tight_layout()
        out_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved to {out_path}")

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict:
        """
        Runs the complete evaluation pipeline:
        inference → metrics → confusion matrix → training curves.

        Returns:
            metrics dict.
        """
        labels, preds, probs = self.run_inference()
        metrics              = self.compute_metrics(labels, preds, probs)
        self.plot_training_curves()
        self.plot_confusion_matrix(labels, preds)
        return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  Dog Breed Classification — Model Evaluation")
    print("=" * 60)

    checkpoint_path = "checkpoints/dog_breed_resnet50/best_model.pth"

    evaluator = Evaluator(
        checkpoint_path = checkpoint_path,
        data_dir        = "processed_data",
        output_dir      = "evaluation_results",
        batch_size      = 64,
        num_workers     = 4,
    )

    metrics = evaluator.evaluate()

    print("\n" + "=" * 60)
    print(f"  Top-1 Accuracy : {metrics['top1_accuracy'] * 100:.2f}%")
    print(f"  Top-3 Accuracy : {metrics['top3_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy : {metrics['top5_accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score : {metrics['macro_f1'] * 100:.2f}%")
    print(f"  Results saved  : evaluation_results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
