"""Confusion matrix plots with short class-name labels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from datasets.plantvillage import CANONICAL_CLASSES
from visualization.style import MODEL_LABELS, apply_ieee_style, save_figure

# Short display names to fit on axes
_SHORT = {
    "Pepper__bell___Bacterial_spot":                      "Pep.BactSpot",
    "Pepper__bell___healthy":                             "Pep.Healthy",
    "Potato___Early_blight":                              "Pot.EarlyBlight",
    "Potato___Late_blight":                               "Pot.LateBlight",
    "Potato___healthy":                                   "Pot.Healthy",
    "Tomato___Bacterial_spot":                            "Tom.BactSpot",
    "Tomato___Early_blight":                              "Tom.EarlyBlight",
    "Tomato___Late_blight":                               "Tom.LateBlight",
    "Tomato___Leaf_Mold":                                 "Tom.LeafMold",
    "Tomato___Septoria_leaf_spot":                        "Tom.Septoria",
    "Tomato___Spider_mites Two-spotted_spider_mite":      "Tom.SpiderMites",
    "Tomato___Target_Spot":                               "Tom.TargetSpot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":             "Tom.YLCV",
    "Tomato___Tomato_mosaic_virus":                       "Tom.MosaicV",
    "Tomato___healthy":                                   "Tom.Healthy",
}

# Handle the slightly different names that appear in result files
_SHORT_ALT = {k.replace("___", "_").replace(" ", "_"): v for k, v in _SHORT.items()}


def _short(name: str) -> str:
    return _SHORT.get(name) or _SHORT_ALT.get(name) or name[:14]


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    model_name: str,
    dataset_name: str,
    save_path: Path,
    normalise: bool = True,
) -> None:
    """
    Heatmap confusion matrix.

    Parameters
    ----------
    cm           : raw confusion matrix (list of lists)
    labels       : class names corresponding to rows/cols
    normalise    : if True, show row-normalised values (recall per class)
    """
    apply_ieee_style()
    mat = np.array(cm, dtype=float)

    if normalise:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid divide-by-zero for absent classes
        mat = mat / row_sums

    short_labels = [_short(l) for l in labels]
    n = len(short_labels)
    fig_size = max(6.0, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    sns.heatmap(
        mat, ax=ax,
        annot=(n <= 15),           # annotate only if legible
        fmt=".2f" if normalise else "d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.65},
        vmin=0, vmax=1 if normalise else None,
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    norm_str = "Normalised " if normalise else ""
    ax.set_title(
        f"{norm_str}Confusion Matrix — "
        f"{MODEL_LABELS.get(model_name, model_name)} on "
        f"{dataset_name.title()}",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_all_confusion_matrices(
    metrics_dir: Path,
    figures_dir: Path,
    model_names: list[str] | None = None,
    datasets: list[str] | None = None,
) -> None:
    if model_names is None:
        model_names = ["efficientnetb0", "mobilenetv2", "resnet50"]
    if datasets is None:
        datasets = ["plantvillage", "plantdoc"]

    cm_dir = figures_dir / "confusion_matrices"

    for model in model_names:
        for dataset in datasets:
            path = metrics_dir / f"{model}_{dataset}_metrics.json"
            if not path.exists():
                print(f"  [skip] {model}_{dataset}_metrics.json not found")
                continue
            with open(path) as f:
                m = json.load(f)

            cm     = m.get("confusion_matrix", [])
            labels = m.get("confusion_matrix_labels", CANONICAL_CLASSES)
            if not cm:
                continue

            save_path = cm_dir / f"cm_{model}_{dataset}"
            plot_confusion_matrix(cm, labels, model, dataset, save_path)
            print(f"  Saved: cm_{model}_{dataset}.png")
