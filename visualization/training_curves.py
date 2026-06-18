"""Training curve plots: loss and accuracy per epoch for one or all models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from visualization.style import MODEL_COLORS, MODEL_LABELS, apply_ieee_style, save_figure


def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: Path,
) -> None:
    """
    Two-panel figure: training/val loss (left) and training/val accuracy (right).

    Parameters
    ----------
    history    : dict with keys train_loss, val_loss, train_acc, val_acc
    model_name : used in title and filename
    save_path  : output path (PNG + PDF saved)
    """
    apply_ieee_style()
    color = MODEL_COLORS.get(model_name, "#333333")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=color,     linestyle="-",  label="Train")
    ax.plot(epochs, history["val_loss"],   color=color,     linestyle="--", label="Val",  alpha=0.75)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss"); ax.legend()
    ax.set_xticks(list(epochs))

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color=color,     linestyle="-",  label="Train")
    ax.plot(epochs, history["val_acc"],   color=color,     linestyle="--", label="Val",  alpha=0.75)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy"); ax.legend()
    ax.set_ylim(0, 1.05); ax.set_xticks(list(epochs))

    fig.suptitle(f"Training Curves — {MODEL_LABELS.get(model_name, model_name)}",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_all_training_curves(
    metrics_dir: Path,
    figures_dir: Path,
    model_names: list[str] | None = None,
) -> None:
    """Load history JSONs and plot curves for every available model."""
    if model_names is None:
        model_names = ["efficientnetb0", "mobilenetv2", "resnet50"]

    for name in model_names:
        hist_path = metrics_dir / f"{name}_training_history.json"
        if not hist_path.exists():
            print(f"  [skip] No training history for {name}")
            continue
        with open(hist_path) as f:
            history = json.load(f)
        save_path = figures_dir / f"training_curves_{name}"
        plot_training_curves(history, name, save_path)
        print(f"  Saved training curves: {save_path.name}.png")
