"""
model.py
--------
Model factory for dog breed classification using transfer learning.

Supports ResNet50 and EfficientNet-B0. In both cases, pretrained ImageNet
weights are loaded and the final classification head is replaced with a
new Linear layer sized for the target number of classes.

When freeze_feature_extractor=True, only the classification head is trained
in the first phase; the backbone can be unfrozen for fine-tuning afterwards.
"""

import logging

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision.models.resnet import ResNet50_Weights

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """
    Creates and configures CNN models for dog breed classification.

    Args:
        num_classes:              Number of output classes (dog breeds).
        model_name:               Architecture to use: 'resnet50' or 'efficientnet_b0'.
        pretrained:               Load ImageNet pretrained weights.
        freeze_feature_extractor: Freeze backbone; only the classifier head is trained.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_feature_extractor: bool = True,
    ):
        self.num_classes             = num_classes
        self.model_name              = model_name.lower()
        self.pretrained              = pretrained
        self.freeze_feature_extractor = freeze_feature_extractor

        logger.info(
            f"ModelFactory | model={model_name}, num_classes={num_classes}, "
            f"pretrained={pretrained}, freeze_backbone={freeze_feature_extractor}"
        )

    def create_model(self) -> nn.Module:
        """
        Builds the model, optionally freezes the backbone, and replaces
        the classification head.

        Returns:
            Configured nn.Module ready for training.

        Raises:
            ValueError: If model_name is not supported.
        """
        if self.model_name == "resnet50":
            model = self._build_resnet50()
        elif self.model_name == "efficientnet_b0":
            model = self._build_efficientnet_b0()
        else:
            raise ValueError(
                f"Unsupported model: '{self.model_name}'. "
                "Choose 'resnet50' or 'efficientnet_b0'."
            )

        return model

    def _build_resnet50(self) -> nn.Module:
        weights = ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        model   = models.resnet50(weights=weights)

        if self.freeze_feature_extractor:
            for param in model.parameters():
                param.requires_grad = False

        # Replace classification head
        num_features = model.fc.in_features
        model.fc     = nn.Linear(num_features, self.num_classes)

        # Ensure the new head is always trainable
        for param in model.fc.parameters():
            param.requires_grad = True

        logger.info(f"ResNet50 head: in_features={num_features}, out_features={self.num_classes}")
        return model

    def _build_efficientnet_b0(self) -> nn.Module:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if self.pretrained else None
        model   = models.efficientnet_b0(weights=weights)

        if self.freeze_feature_extractor:
            for param in model.features.parameters():
                param.requires_grad = False

        # Replace classification head (EfficientNet: Sequential(Dropout, Linear))
        num_features         = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, self.num_classes)

        for param in model.classifier.parameters():
            param.requires_grad = True

        logger.info(f"EfficientNet-B0 head: in_features={num_features}, out_features={self.num_classes}")
        return model

    def get_model_summary(self, model: nn.Module) -> None:
        """Logs total and trainable parameter counts."""
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Total parameters     : {total:,}")
        logger.info(f"Trainable parameters : {trainable:,}")
        logger.info(f"Frozen parameters    : {total - trainable:,}")
