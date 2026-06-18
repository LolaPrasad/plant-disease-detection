"""
Select representative samples for Grad-CAM visualisation.

Three categories (matching the research brief):
  1. correct_pv  — correctly predicted PlantVillage images
  2. correct_pd  — correctly predicted PlantDoc images
  3. wrong_pd    — misclassified PlantDoc images

Selects diverse samples across classes to maximise visual coverage.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def select_samples(
    loader: DataLoader,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    category: str,
    n_per_class: int = 1,
    max_total: int = 30,
    seed: int = 42,
) -> list[dict]:
    """
    Pull image tensors for samples matching the selection criterion.

    Parameters
    ----------
    loader      : DataLoader for the dataset (shuffle=False)
    y_true      : ground-truth labels (from run_inference)
    y_pred      : predicted labels    (from run_inference)
    category    : 'correct_pv', 'correct_pd', or 'wrong_pd'
    n_per_class : how many samples to take per class
    max_total   : hard cap on total samples returned

    Returns
    -------
    list of dicts ready for generate_gradcam_for_samples()
    """
    rng = np.random.default_rng(seed)

    # Which indices match the criterion
    if category in ("correct_pv", "correct_pd"):
        mask = (y_true == y_pred)
    elif category == "wrong_pd":
        mask = (y_true != y_pred)
    else:
        raise ValueError(f"Unknown category: {category}")

    matching_indices = np.where(mask)[0]

    # Stratify: pick up to n_per_class per true class
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx in matching_indices:
        by_class[int(y_true[idx])].append(int(idx))

    chosen_indices: list[int] = []
    for cls_indices in by_class.values():
        rng.shuffle(cls_indices)
        chosen_indices.extend(cls_indices[:n_per_class])

    rng.shuffle(chosen_indices)
    chosen_indices = chosen_indices[:max_total]
    chosen_set = set(chosen_indices)

    # Collect tensors by iterating the loader once
    samples: list[dict] = []
    global_idx = 0
    for images, labels in loader:
        for i in range(images.size(0)):
            if global_idx in chosen_set:
                samples.append({
                    "image_tensor": images[i].cpu(),
                    "true_label":   int(y_true[global_idx]),
                    "pred_label":   int(y_pred[global_idx]),
                    "category":     category,
                    "sample_idx":   global_idx,
                })
            global_idx += 1

    return samples
