"""
preprocessing.py
----------------
Data exploration and preprocessing pipeline for the Stanford Dogs Dataset.

Steps performed:
    1. Explores dataset directory structure and validates image/annotation alignment.
    2. Parses XML-format annotation files to extract breed names and bounding boxes.
    3. Crops images to bounding boxes, resizes to 224x224, and saves to processed_data/.
    4. Performs a stratified 70/20/10 train/val/test split.
    5. Saves class mappings and split metadata as JSON for downstream use.
"""

import os
import json
import shutil
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from functools import partial

import cv2
import numpy as np
import pandas as pd
import xmltodict
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE      = 224
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.20
TEST_RATIO      = 0.10
RANDOM_SEED     = 42
NUM_WORKERS     = max(1, mp.cpu_count() - 1)

IMAGES_DIR      = "data/Images"
ANNOTATIONS_DIR = "data/Annotation"
OUTPUT_DIR      = "processed_data"


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def explore_dataset(images_dir: str, annotations_dir: str) -> pd.DataFrame:
    """
    Validates directory structure and counts images per breed.

    Args:
        images_dir:      Path to the images root directory.
        annotations_dir: Path to the annotations root directory.

    Returns:
        DataFrame with columns [breed, image_count, annotation_count].
    """
    images_path      = Path(images_dir)
    annotations_path = Path(annotations_dir)

    image_breeds      = {d.name for d in images_path.iterdir() if d.is_dir()}
    annotation_breeds = {d.name for d in annotations_path.iterdir() if d.is_dir()}
    common_breeds     = sorted(image_breeds & annotation_breeds)

    print(f"Breed folders in images      : {len(image_breeds)}")
    print(f"Breed folders in annotations : {len(annotation_breeds)}")
    print(f"Matched breed folders        : {len(common_breeds)}")

    missing_ann = image_breeds - annotation_breeds
    missing_img = annotation_breeds - image_breeds
    if missing_ann:
        print(f"[WARN] Breeds missing annotations: {missing_ann}")
    if missing_img:
        print(f"[WARN] Breeds missing images: {missing_img}")

    stats = []
    for breed in common_breeds:
        img_count = len(list((images_path / breed).glob("*.jpg")))
        ann_count = len([
            f for f in (annotations_path / breed).iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])
        stats.append({"breed": breed, "image_count": img_count, "annotation_count": ann_count})

    df = pd.DataFrame(stats)
    total = df["image_count"].sum()
    cv    = df["image_count"].std() / df["image_count"].mean()

    print(f"\nTotal images          : {total:,}")
    print(f"Avg images per breed  : {df['image_count'].mean():.1f}")
    print(f"Min / Max per breed   : {df['image_count'].min()} / {df['image_count'].max()}")
    print(f"Class balance (CV)    : {cv:.3f}  {'(balanced)' if cv < 0.3 else '(imbalanced)'}")

    return df


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotation(annotation_path: Path) -> dict:
    """
    Parses a single XML-format annotation file (may have no extension).

    Args:
        annotation_path: Path to the annotation file.

    Returns:
        Dict with keys: breed_name, xmin, ymin, xmax, ymax.
        Returns None if parsing fails.
    """
    try:
        with open(annotation_path, "r") as f:
            content = f.read()

        data       = xmltodict.parse(content)
        annotation = data.get("annotation", {})

        # Extract breed name
        breed_name = annotation.get("object", {})
        if isinstance(breed_name, list):
            breed_name = breed_name[0]
        breed_name = breed_name.get("name", "unknown")

        # Extract bounding box
        bndbox = annotation.get("object", {})
        if isinstance(bndbox, list):
            bndbox = bndbox[0]
        bndbox = bndbox.get("bndbox", {})

        return {
            "breed_name": breed_name,
            "xmin": int(bndbox.get("xmin", 0)),
            "ymin": int(bndbox.get("ymin", 0)),
            "xmax": int(bndbox.get("xmax", 0)),
            "ymax": int(bndbox.get("ymax", 0)),
        }

    except Exception:
        return None


def _parse_breed_annotations(breed: str, annotations_dir: str) -> list:
    """Worker function: parses all annotations for one breed."""
    results = []
    breed_path = Path(annotations_dir) / breed

    for ann_file in breed_path.iterdir():
        if not ann_file.is_file() or ann_file.name.startswith("."):
            continue

        parsed = parse_annotation(ann_file)
        if parsed:
            parsed["filename"] = ann_file.name
            parsed["breed_folder"] = breed
            results.append(parsed)

    return results


def build_annotations_dataframe(images_dir: str, annotations_dir: str) -> pd.DataFrame:
    """
    Parses all annotation files in parallel and returns a unified DataFrame.

    Args:
        images_dir:      Path to images root (used to derive breed list).
        annotations_dir: Path to annotations root.

    Returns:
        DataFrame with one row per image containing breed and bbox info.
    """
    breeds = sorted(d.name for d in Path(images_dir).iterdir() if d.is_dir())

    print(f"Parsing annotations for {len(breeds)} breeds using {NUM_WORKERS} workers...")

    worker = partial(_parse_breed_annotations, annotations_dir=annotations_dir)
    with mp.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(worker, breeds), total=len(breeds)))

    records = [item for sublist in results for item in sublist]
    df = pd.DataFrame(records)
    print(f"Total annotations parsed: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def crop_and_resize(image_path: Path, xmin: int, ymin: int,
                    xmax: int, ymax: int, output_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Crops an image to a bounding box and resizes to output_size x output_size.

    Args:
        image_path:  Path to the source image.
        xmin, ymin, xmax, ymax: Bounding box coordinates.
        output_size: Target width and height in pixels.

    Returns:
        Processed image as a numpy array (H x W x 3), or None on failure.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(xmin + 1, min(xmax, w))
    ymax = max(ymin + 1, min(ymax, h))

    cropped = img[ymin:ymax, xmin:xmax]
    resized = cv2.resize(cropped, (output_size, output_size),
                         interpolation=cv2.INTER_LANCZOS4)
    return resized


def _process_single_image(row: dict, images_dir: str, output_dir: str) -> bool:
    """Worker function: crops, resizes, and saves one image."""
    try:
        breed_folder = row["breed_folder"]
        filename     = row["filename"]
        split        = row["split"]

        img_path = Path(images_dir) / breed_folder / (filename + ".jpg")
        if not img_path.exists():
            # Try without extension (some files already have it in filename)
            img_path = Path(images_dir) / breed_folder / filename

        processed = crop_and_resize(
            img_path,
            row["xmin"], row["ymin"],
            row["xmax"], row["ymax"],
        )
        if processed is None:
            return False

        out_dir = Path(output_dir) / split / breed_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (filename + ".jpg")
        cv2.imwrite(str(out_path), processed)
        return True

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dataset splitting and saving
# ---------------------------------------------------------------------------

def split_dataset(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a stratified 70/20/10 train/val/test split.

    Args:
        annotations_df: DataFrame with a 'breed_folder' column.

    Returns:
        Same DataFrame with an added 'split' column.
    """
    df = annotations_df.copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["breed_folder"],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["breed_folder"],
        random_state=RANDOM_SEED,
    )

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df["split"] = "train"
    val_df["split"]   = "val"
    test_df["split"]  = "test"

    result = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    print(f"Train : {len(train_df):,} images")
    print(f"Val   : {len(val_df):,} images")
    print(f"Test  : {len(test_df):,} images")
    print(f"Total : {len(result):,} images")

    return result


def save_class_mappings(annotations_df: pd.DataFrame, output_dir: str) -> dict:
    """
    Saves class-to-index mappings as class_mappings.json.

    Args:
        annotations_df: DataFrame containing 'breed_folder' column.
        output_dir:     Directory to save the JSON file.

    Returns:
        The class mappings dict.
    """
    breeds      = sorted(annotations_df["breed_folder"].unique())
    class_to_idx = {breed: idx for idx, breed in enumerate(breeds)}
    idx_to_class = {str(idx): breed for breed, idx in class_to_idx.items()}

    mappings = {
        "num_classes": len(breeds),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "class_mappings.json", "w") as f:
        json.dump(mappings, f, indent=4)

    print(f"Class mappings saved: {len(breeds)} classes")
    return mappings


def save_split_info(annotations_df: pd.DataFrame, output_dir: str) -> None:
    """Saves split counts and ratios as split_info.json."""
    split_counts = annotations_df["split"].value_counts().to_dict()
    total        = len(annotations_df)

    split_info = {
        "total_images": total,
        "splits": {
            split: {
                "count": count,
                "ratio": round(count / total, 4),
            }
            for split, count in split_counts.items()
        },
        "random_seed": RANDOM_SEED,
    }

    with open(Path(output_dir) / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=4)

    print("Split info saved to split_info.json")


def process_and_save_images(annotations_df: pd.DataFrame,
                             images_dir: str,
                             output_dir: str) -> None:
    """
    Processes all images in parallel: crops to bbox, resizes, saves.

    Args:
        annotations_df: DataFrame with split assignments and bbox coords.
        images_dir:     Root directory of raw images.
        output_dir:     Root directory for processed output.
    """
    rows = annotations_df.to_dict("records")
    worker = partial(_process_single_image,
                     images_dir=images_dir,
                     output_dir=output_dir)

    print(f"Processing {len(rows):,} images using {NUM_WORKERS} workers...")
    with mp.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(worker, rows), total=len(rows)))

    success = sum(results)
    print(f"Successfully processed: {success:,} / {len(rows):,} images")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  Dog Breed Classification — Data Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Explore dataset
    print("\n--- Step 1: Dataset Exploration ---")
    explore_dataset(IMAGES_DIR, ANNOTATIONS_DIR)

    # Step 2: Parse annotations
    print("\n--- Step 2: Parsing Annotations ---")
    annotations_df = build_annotations_dataframe(IMAGES_DIR, ANNOTATIONS_DIR)

    # Step 3: Stratified split
    print("\n--- Step 3: Splitting Dataset ---")
    annotations_df = split_dataset(annotations_df)

    # Step 4: Save metadata
    print("\n--- Step 4: Saving Metadata ---")
    save_class_mappings(annotations_df, OUTPUT_DIR)
    save_split_info(annotations_df, OUTPUT_DIR)

    # Step 5: Process and save images
    print("\n--- Step 5: Processing Images (crop + resize) ---")
    process_and_save_images(annotations_df, IMAGES_DIR, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  Preprocessing complete.")
    print(f"  Output saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
