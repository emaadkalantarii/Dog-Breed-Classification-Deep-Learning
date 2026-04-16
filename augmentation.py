"""
augmentation.py
---------------
Data augmentation transforms for dog breed classification.

Provides two transform pipelines:
    - get_train_transforms(): On-the-fly augmentation applied during training.
    - get_val_transforms():   Deterministic transforms for validation and test.

All transforms follow the ImageNet normalisation convention, as the model
(ResNet50) was pre-trained on ImageNet.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE     = 224
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def get_train_transforms() -> A.Compose:
    """
    Returns the training augmentation pipeline.

    Transforms are applied on-the-fly to each image as it is loaded,
    so the model sees a different variation of every image each epoch.
    This approach maximises data diversity without pre-storing augmented
    copies and is the primary defence against overfitting.

    Pipeline overview:
        - RandomResizedCrop:   Simulates varied framing and zoom levels.
        - HorizontalFlip:      Dogs appear facing both directions.
        - ShiftScaleRotate:    Modest geometric perturbation.
        - ColorJitter:         Handles lighting and colour variation.
        - HueSaturationValue:  Coat colour robustness.
        - RGBShift:            Camera sensor colour cast variation.
        - GaussNoise:          Simulates image noise.
        - GaussianBlur:        Simulates slight defocus.
        - CoarseDropout:       Occlusion regularisation.
        - RandomErasing:       Prevents over-reliance on specific regions.

    Returns:
        An Albumentations Compose pipeline that outputs a normalised tensor.
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),

        # Colour augmentations
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3,
        ),
        A.RGBShift(
            r_shift_limit=10,
            g_shift_limit=10,
            b_shift_limit=10,
            p=0.2,
        ),

        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Occlusion / erasing
        A.CoarseDropout(
            max_holes=8,
            max_height=24,
            max_width=24,
            fill_value=0,
            p=0.15,
        ),

        # Normalise and convert to tensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Returns the deterministic validation / test transform pipeline.

    No random augmentations are applied — only resizing, centre-cropping,
    and normalisation — to ensure reproducible and unbiased evaluation.

    Returns:
        An Albumentations Compose pipeline that outputs a normalised tensor.
    """
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
