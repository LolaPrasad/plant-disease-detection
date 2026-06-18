"""
Grad-CAM heatmap generation for EfficientNetB0, MobileNetV2, and ResNet50.

Uses the pytorch-grad-cam library which handles hook registration, gradient
accumulation, and CAM computation cleanly for all three architectures.

Design:
  - One GradCAMGenerator per model instance — holds the CAM object and target layer.
  - generate_heatmap() returns a numpy overlay ready for matplotlib / cv2 saving.
  - generate_batch() processes a list of (image_tensor, true_label, pred_label)
    tuples and saves PNGs with overlays automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from datasets.plantvillage import CANONICAL_CLASSES

# ImageNet normalisation constants (used to denormalise for display)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor → HWC float32 in [0, 1]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
    img = img * _STD + _MEAN                         # undo normalisation
    return np.clip(img, 0.0, 1.0).astype(np.float32)


class GradCAMGenerator:
    """
    Wraps pytorch-grad-cam for a single model.

    Parameters
    ----------
    model      : trained nn.Module with a `grad_cam_target_layer` attribute
    device     : torch.device
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device

        target_layer = model.grad_cam_target_layer
        # pytorch-grad-cam expects a list of target layers
        self._cam = GradCAM(
            model=model,
            target_layers=[target_layer],
        )

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a Grad-CAM heatmap for a single image.

        Parameters
        ----------
        image_tensor : (C, H, W) tensor, normalised
        target_class : class index to explain; if None, uses the predicted class

        Returns
        -------
        cam_image : (H, W, 3) uint8 overlay (heatmap blended onto original image)
        raw_cam   : (H, W) float32 grayscale CAM in [0, 1]
        """
        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        targets = [ClassifierOutputTarget(target_class)] \
                  if target_class is not None else None

        raw_cam = self._cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]  # (H, W) float32

        rgb_img   = _denormalise(image_tensor)
        cam_image = show_cam_on_image(rgb_img, raw_cam, use_rgb=True)

        return cam_image, raw_cam

    def __del__(self):
        # Clean up hooks registered by pytorch-grad-cam
        try:
            self._cam.__exit__(None, None, None)
        except Exception:
            pass


def save_gradcam_image(
    cam_image: np.ndarray,
    original_rgb: np.ndarray,
    save_path: Path,
    title: str = "",
) -> None:
    """
    Save a side-by-side (original | Grad-CAM overlay) PNG at 300 DPI.

    Parameters
    ----------
    cam_image    : (H, W, 3) uint8 Grad-CAM overlay
    original_rgb : (H, W, 3) float32 [0,1] original image
    save_path    : output file path
    title        : figure suptitle
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(cam_image)
    axes[1].set_title("Grad-CAM", fontsize=9)
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=8, wrap=True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_gradcam_for_samples(
    generator: GradCAMGenerator,
    samples: list[dict],
    save_dir: Path,
    model_name: str,
    max_per_category: int = 5,
) -> list[dict]:
    """
    Generate and save Grad-CAM images for a curated list of samples.

    Parameters
    ----------
    generator        : GradCAMGenerator instance
    samples          : list of dicts with keys:
                         'image_tensor' (C,H,W),
                         'true_label'  int,
                         'pred_label'  int,
                         'category'    str  — 'correct_pv', 'correct_pd', 'wrong_pd'
                         'sample_idx'  int
    save_dir         : root output directory
    model_name       : used for subdirectory naming
    max_per_category : cap per category to avoid huge output sets

    Returns
    -------
    list of dicts describing each saved file
    """
    from collections import defaultdict

    saved: list[dict] = []
    category_counts: dict[str, int] = defaultdict(int)

    for s in samples:
        cat = s["category"]
        if category_counts[cat] >= max_per_category:
            continue

        true_cls = CANONICAL_CLASSES[s["true_label"]]
        pred_cls = CANONICAL_CLASSES[s["pred_label"]]
        correct  = s["true_label"] == s["pred_label"]

        cam_img, raw_cam = generator.generate_heatmap(
            s["image_tensor"],
            target_class=s["pred_label"],
        )
        original_rgb = _denormalise(s["image_tensor"])

        status = "correct" if correct else "wrong"
        fname  = (
            f"{cat}_{status}_idx{s['sample_idx']}"
            f"_true-{true_cls[:20]}_pred-{pred_cls[:20]}.png"
        ).replace(" ", "_").replace("/", "-")

        out_path = save_dir / model_name / cat / fname
        title = (
            f"True: {true_cls}\nPred: {pred_cls} "
            f"({'✓' if correct else '✗'})"
        )
        save_gradcam_image(cam_img, original_rgb, out_path, title=title)

        saved.append({
            "model":      model_name,
            "category":   cat,
            "sample_idx": s["sample_idx"],
            "true_class": true_cls,
            "pred_class": pred_cls,
            "correct":    correct,
            "file":       str(out_path),
        })
        category_counts[cat] += 1

    return saved
