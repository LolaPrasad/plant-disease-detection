"""
Model factory for EfficientNetB0, MobileNetV2, and ResNet50.

All models:
  - Load ImageNet pretrained weights from torchvision
  - Replace the final classifier with a task-specific head (15 classes)
  - Expose a `grad_cam_target_layer` attribute for Grad-CAM hooks

Design decision: we fine-tune end-to-end from epoch 0 (no frozen backbone).
This gives the best downstream accuracy and is standard practice when the
source dataset (PlantVillage) is large enough.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 15


# ── EfficientNetB0 ────────────────────────────────────────────────────────────

class EfficientNetB0(nn.Module):
    """
    EfficientNetB0 with a dropout + linear classification head.

    Grad-CAM target: last convolutional block before average pool
    → model.features[-1]  (the final MBConv block, outputs 1280 channels)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features        # convolutional backbone
        self.avgpool  = base.avgpool         # AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        # features[-1] is the last MBConv block (output: 1280 channels)
        return self.features[-1]


# ── MobileNetV2 ───────────────────────────────────────────────────────────────

class MobileNetV2(nn.Module):
    """
    MobileNetV2 with a dropout + linear classification head.

    Grad-CAM target: features[-1] — the final Conv-BN-ReLU6 block
    (output: 1280 channels, 7×7 at 224px input)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2) -> None:
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        return self.features[-1]


# ── ResNet50 ──────────────────────────────────────────────────────────────────

class ResNet50(nn.Module):
    """
    ResNet50 with a dropout + linear classification head replacing the
    default fully-connected layer.

    Grad-CAM target: layer4 — the final residual block group
    (output: 2048 channels, 7×7 at 224px input)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Keep all layers except the final fc
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool  # AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        return self.layer4


# ── Factory ───────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "efficientnetb0": EfficientNetB0,
    "mobilenetv2":    MobileNetV2,
    "resnet50":       ResNet50,
}


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """
    Build and return a model from the merged config dict.

    Parameters
    ----------
    cfg : dict
        Merged config (base + model-specific). Must contain cfg['model']['name'].

    Returns
    -------
    nn.Module
        Model with ImageNet weights and task-specific classification head.
    """
    model_cfg = cfg.get("model", {})
    name      = model_cfg.get("name", "").lower().replace("-", "").replace("_", "")
    num_cls   = cfg["classes"]["num_classes"]
    dropout   = model_cfg.get("dropout", 0.3)

    # Normalise aliases
    _aliases = {
        "efficientnetb0": "efficientnetb0",
        "efficientnet":   "efficientnetb0",
        "mobilenetv2":    "mobilenetv2",
        "mobilenet":      "mobilenetv2",
        "resnet50":       "resnet50",
        "resnet":         "resnet50",
    }
    key = _aliases.get(name)
    if key is None:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {list(_REGISTRY.keys())}"
        )

    model_cls = _REGISTRY[key]
    return model_cls(num_classes=num_cls, dropout=dropout)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load a saved checkpoint into model (in-place).

    Returns the full checkpoint dict (contains epoch, metrics, etc.).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return ckpt


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
