"""
Shared matplotlib style for all publication figures.

IEEE two-column style: 3.5" single-column, 7.16" double-column, 300 DPI.
Colourblind-safe palette (Wong 2011).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Wong (2011) colourblind-safe palette ──────────────────────────────────────
PALETTE = {
    "blue":        "#0072B2",
    "orange":      "#E69F00",
    "green":       "#009E73",
    "red":         "#D55E00",
    "purple":      "#CC79A7",
    "sky":         "#56B4E9",
    "yellow":      "#F0E442",
    "black":       "#000000",
}

# Per-model colours (consistent across all figures)
MODEL_COLORS = {
    "efficientnetb0": PALETTE["blue"],
    "mobilenetv2":    PALETTE["orange"],
    "resnet50":       PALETTE["green"],
}

MODEL_LABELS = {
    "efficientnetb0": "EfficientNetB0",
    "mobilenetv2":    "MobileNetV2",
    "resnet50":       "ResNet50",
}

# Dataset colours
DATASET_COLORS = {
    "plantvillage": PALETTE["blue"],
    "plantdoc":     PALETTE["red"],
}


def apply_ieee_style() -> None:
    """Apply IEEE-compatible rcParams globally."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "pdf.fonttype":      42,   # embed fonts — required by IEEE
        "ps.fonttype":       42,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
    })


def save_figure(fig: plt.Figure, path, tight: bool = True) -> None:
    """Save as both PNG (300 DPI) and PDF."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight") if tight else {}
    fig.savefig(path.with_suffix(".png"), dpi=300, **kw)
    fig.savefig(path.with_suffix(".pdf"), **kw)
    plt.close(fig)
