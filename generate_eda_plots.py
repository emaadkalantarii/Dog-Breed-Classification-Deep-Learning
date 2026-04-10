"""
generate_eda_plots.py
---------------------
Generates high-resolution EDA visualisations for the Stanford Dogs Dataset:
    1. assets/class_balance.png      — boxplot of images-per-breed distribution
    2. assets/image_dimensions.png   — scatter plot of raw image width vs height

Run this script once after downloading the dataset:

    python generate_eda_plots.py

Outputs are saved at 200 DPI, suitable for GitHub README display.
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

IMAGES_DIR  = "data/Images"
OUTPUT_DIR  = "assets"
SAMPLE_SIZE = 300   # Number of images sampled for the dimensions scatter plot
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_images_per_breed(images_dir: str) -> dict:
    """Returns {breed_folder_name: image_count} for all breed folders."""
    return {
        folder.name: len(list(folder.glob("*.jpg")))
        for folder in sorted(Path(images_dir).iterdir())
        if folder.is_dir()
    }


def sample_image_dimensions(images_dir: str, sample_size: int) -> list:
    """
    Randomly samples images from the dataset and returns
    a list of (width, height) tuples without loading full tensors.
    """
    random.seed(RANDOM_SEED)
    all_paths = list(Path(images_dir).rglob("*.jpg"))
    sampled   = random.sample(all_paths, min(sample_size, len(all_paths)))

    dims = []
    for path in sampled:
        try:
            with Image.open(path) as img:
                dims.append(img.size)   # (width, height)
        except Exception:
            continue
    return dims


# ---------------------------------------------------------------------------
# Plot 1: Class balance boxplot
# ---------------------------------------------------------------------------

def plot_class_balance(counts: dict, output_path: str) -> None:
    """
    Saves a boxplot of the images-per-breed distribution.
    Annotates the Coefficient of Variation (CV) in the title.
    """
    values = list(counts.values())
    cv     = np.std(values) / np.mean(values)

    fig, ax = plt.subplots(figsize=(5, 6))

    ax.boxplot(
        values,
        patch_artist=True,
        medianprops  =dict(color="#e07b39", linewidth=2),
        boxprops     =dict(facecolor="#cde8f5", color="#2c5f8a"),
        whiskerprops =dict(color="#2c5f8a"),
        capprops     =dict(color="#2c5f8a"),
        flierprops   =dict(marker="o", color="#2c5f8a", alpha=0.5, markersize=4),
    )

    ax.set_title(f"Class Balance  (CV: {cv:.3f})", fontsize=13, pad=10)
    ax.set_ylabel("Images per Breed", fontsize=11)
    ax.set_xticks([])
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Side annotation with key statistics
    stats_text = (
        f"Breeds : {len(values)}\n"
        f"Total  : {sum(values):,}\n"
        f"Median : {int(np.median(values))}\n"
        f"Mean   : {np.mean(values):.0f}\n"
        f"Min    : {min(values)}\n"
        f"Max    : {max(values)}"
    )
    ax.text(
        1.30, np.median(values), stats_text,
        transform=ax.get_yaxis_transform(),
        fontsize=8.5, va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Image dimensions scatter plot
# ---------------------------------------------------------------------------

def plot_image_dimensions(dimensions: list, output_path: str) -> None:
    """
    Saves a scatter plot of raw image width vs height.
    Point colour encodes image size in megapixels.
    """
    widths  = [d[0] for d in dimensions]
    heights = [d[1] for d in dimensions]
    mp      = [w * h / 1e6 for w, h in dimensions]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sc = ax.scatter(
        widths, heights, c=mp,
        cmap="viridis", s=40, alpha=0.65, edgecolors="none"
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Megapixels", fontsize=10)

    ax.set_title(
        f"Raw Image Dimensions  (n={len(dimensions)} sampled)",
        fontsize=13, pad=10
    )
    ax.set_xlabel("Width (pixels)", fontsize=11)
    ax.set_ylabel("Height (pixels)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Counting images per breed...")
    counts = count_images_per_breed(IMAGES_DIR)
    print(f"  {len(counts)} breeds | {sum(counts.values()):,} total images")

    print("Sampling image dimensions...")
    dims = sample_image_dimensions(IMAGES_DIR, SAMPLE_SIZE)
    print(f"  Sampled {len(dims)} images")

    print("Generating plots...")
    plot_class_balance(counts, f"{OUTPUT_DIR}/class_balance.png")
    plot_image_dimensions(dims, f"{OUTPUT_DIR}/image_dimensions.png")

    print("\nDone — both plots saved to assets/")


if __name__ == "__main__":
    main()
