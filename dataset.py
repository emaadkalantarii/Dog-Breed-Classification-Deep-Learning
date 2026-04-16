"""
dataset.py
----------
PyTorch Dataset and DataLoader factory for dog breed classification.

Classes:
    DogBreedDataset: Loads preprocessed images from the processed_data/
                     directory structure and applies the provided transforms.

Functions:
    create_dataloaders: Instantiates train, val, and test DataLoaders
                        ready for use in the training loop.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from augmentation import get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DogBreedDataset(Dataset):
    """
    PyTorch Dataset for the preprocessed Stanford Dogs Dataset.

    Expects the following directory layout under data_dir/split/:
        processed_data/
            train/
                n02085620-Chihuahua/
                    img_001.jpg
                    ...
                n02085782-Japanese_spaniel/
                    ...
            val/
                ...
            test/
                ...
            class_mappings.json

    Args:
        data_dir:            Root directory of preprocessed data.
        split:               One of 'train', 'val', or 'test'.
        transform:           Albumentations transform pipeline.
        class_mapping_path:  Path to the class_mappings.json file.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        class_mapping_path: str = "processed_data/class_mappings.json",
    ):
        self.data_dir  = Path(data_dir)
        self.split     = split
        self.split_dir = self.data_dir / split
        self.transform = transform

        # Load class mappings
        mapping_file = Path(class_mapping_path)
        if not mapping_file.exists():
            raise FileNotFoundError(f"Class mapping file not found: {mapping_file}")
        with open(mapping_file, "r") as f:
            mappings = json.load(f)

        self.class_to_idx: Dict[str, int] = mappings["class_to_idx"]
        self.idx_to_class: Dict[int, str] = {
            int(k): v for k, v in mappings["idx_to_class"].items()
        }
        self.num_classes: int = mappings["num_classes"]

        self.samples: List[Tuple[Path, int]] = self._load_samples()
        logger.info(
            f"Loaded {split} split: {len(self.samples):,} samples, "
            f"{self.num_classes} classes"
        )

    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Scans the split directory and collects (image_path, class_idx) pairs."""
        samples = []
        if not self.split_dir.exists():
            logger.warning(f"Split directory not found: {self.split_dir}")
            return samples

        for breed_folder in self.split_dir.iterdir():
            if not breed_folder.is_dir():
                continue
            breed_name = breed_folder.name
            if breed_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[breed_name]
            for img_path in breed_folder.glob("*.jpg"):
                samples.append((img_path, class_idx))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str = "processed_data",
    batch_size: int = 64,
    num_workers: int = 4,
    class_mapping_path: str = "processed_data/class_mappings.json",
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Creates train, val, and test DataLoaders.

    Training DataLoader uses on-the-fly augmentation via get_train_transforms().
    Validation and test DataLoaders use deterministic transforms via get_val_transforms().

    Args:
        data_dir:           Root directory of preprocessed data.
        batch_size:         Number of samples per batch.
        num_workers:        Number of parallel data loading workers.
        class_mapping_path: Path to class_mappings.json.
        pin_memory:         Pin memory for faster GPU transfer (set False if using CPU).

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a DataLoader.
    """
    datasets = {
        "train": DogBreedDataset(
            data_dir, split="train",
            transform=get_train_transforms(),
            class_mapping_path=class_mapping_path,
        ),
        "val": DogBreedDataset(
            data_dir, split="val",
            transform=get_val_transforms(),
            class_mapping_path=class_mapping_path,
        ),
        "test": DogBreedDataset(
            data_dir, split="test",
            transform=get_val_transforms(),
            class_mapping_path=class_mapping_path,
        ),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    for split, loader in loaders.items():
        logger.info(
            f"{split:5s} DataLoader: {len(loader.dataset):,} samples, "
            f"{len(loader):,} batches (batch_size={batch_size})"
        )

    return loaders
