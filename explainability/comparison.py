"""
Side-by-side Grad-CAM comparisons across models and domains.

Produces a grid figure:
    rows = selected sample images
    cols = [Original | EfficientNetB0 CAM | MobileNetV2 CAM | ResNet50 CAM]

One figure per category (correct_pv, correct_pd, wrong_pd).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from explainability.gradcam import GradCAMGenerator, _denormalise
from datasets.plantvillage import CANONICAL_CLASSES

MODEL_LABELS = {
    "efficientnetb0": "EfficientNetB0",
    "mobilenetv2":    "MobileNetV2",
    "resnet50":       "ResNet50",
}


def generate_comparison_grid(
    generators: dict[str, GradCAMGenerator],
    samples: list[dict],
    save_path: Path,
    category: str,
    max_rows: int = 5,
) -> None:
    """
    Build an (n_samples × (1 + n_models)) grid figure.

    Parameters
    ----------
    generators : dict mapping model_name → GradCAMGenerator
    samples    : list of sample dicts (must have same image_tensor / labels)
    save_path  : output PNG path
    category   : used in the figure title
    max_rows   : cap to prevent huge figures
    """
    samples = samples[:max_rows]
    n_rows  = len(samples)
    n_cols  = 1 + len(generators)   # original + one per model

    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.8),
        squeeze=False,
    )

    model_names = list(generators.keys())

    for row, s in enumerate(samples):
        img_tensor = s["image_tensor"]
        true_cls   = CANONICAL_CLASSES[s["true_label"]]
        pred_cls   = CANONICAL_CLASSES[s["pred_label"]]
        correct    = s["true_label"] == s["pred_label"]

        # Column 0: original image
        original_rgb = _denormalise(img_tensor)
        ax = axes[row][0]
        ax.imshow(original_rgb)
        ax.set_ylabel(
            f"True: {true_cls[:22]}\nPred: {pred_cls[:22]}\n"
            f"{'✓' if correct else '✗'}",
            fontsize=6, rotation=0, labelpad=80, va="center",
        )
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title("Original", fontsize=8)

        # Columns 1…: Grad-CAM per model
        for col, model_name in enumerate(model_names, start=1):
            cam_img, _ = generators[model_name].generate_heatmap(
                img_tensor,
                target_class=s["pred_label"],
            )
            ax = axes[row][col]
            ax.imshow(cam_img)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=8)

    category_titles = {
        "correct_pv": "Grad-CAM: Correct PlantVillage Predictions",
        "correct_pd": "Grad-CAM: Correct PlantDoc Predictions (Cross-Domain)",
        "wrong_pd":   "Grad-CAM: Misclassified PlantDoc Images (Domain Shift Failures)",
    }
    fig.suptitle(category_titles.get(category, category),
                 fontsize=10, fontweight="bold", y=1.01)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
