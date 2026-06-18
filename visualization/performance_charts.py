"""
Performance comparison bar charts and domain shift plots.

Figures produced:
  1. Grouped bar chart: PV vs PD accuracy/F1 per model
  2. Accuracy drop (absolute + relative) per model
  3. Per-class F1 heatmap across models and domains
  4. Per-class degradation ranked horizontal bar chart
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

from visualization.style import (
    DATASET_COLORS, MODEL_COLORS, MODEL_LABELS, PALETTE,
    apply_ieee_style, save_figure,
)

MODELS = ["efficientnetb0", "mobilenetv2", "resnet50"]

# Legacy src/outputs uses "efficientnet_b0" — map to canonical name
_LEGACY_NAME = {
    "efficientnet_b0": "efficientnetb0",
    "efficientnetb0":  "efficientnetb0",
    "mobilenetv2":     "mobilenetv2",
    "resnet50":        "resnet50",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_domain_shift_csv(metrics_dir: Path) -> list[dict]:
    path = metrics_dir / "domain_shift_summary.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _load_per_class_gap(metrics_dir: Path, model: str) -> dict | None:
    """Try both naming conventions (new pipeline and legacy src/outputs)."""
    # Legacy uses efficientnet_b0; new pipeline uses efficientnetb0
    legacy_name = "efficientnet_b0" if model == "efficientnetb0" else model
    candidates = [
        metrics_dir / f"per_class_gap_{model}.csv",
        metrics_dir.parent.parent / "src" / "outputs" / "domain_shift"
        / f"per_class_gap_{legacy_name}.csv",
    ]
    for p in candidates:
        if p.exists():
            data = {}
            with open(p) as f:
                for row in csv.DictReader(f):
                    data[row["class"]] = {
                        "pv":  float(row["plantvillage_f1"]),
                        "pd":  float(row["plantdoc_f1"]),
                        "gap": float(row["gap"]),
                    }
            return data
    return None


def _load_shift_rows(metrics_dir: Path) -> list[dict]:
    """Load from new pipeline CSV; fall back to legacy src/outputs CSV."""
    rows = _load_domain_shift_csv(metrics_dir)
    if rows:
        return rows
    legacy = (metrics_dir.parent.parent / "src" / "outputs"
              / "domain_shift" / "domain_gap_summary.csv")
    if legacy.exists():
        with open(legacy) as f:
            raw = list(csv.DictReader(f))
        # Normalise model names to canonical form
        for row in raw:
            row["model"] = _LEGACY_NAME.get(row["model"], row["model"])
        return raw
    return []


# ── Figure 1: grouped bar chart ───────────────────────────────────────────────

def plot_domain_gap_bars(metrics_dir: Path, figures_dir: Path) -> None:
    apply_ieee_style()
    rows = _load_shift_rows(metrics_dir)
    if not rows:
        print("  [skip] domain_shift_summary.csv not found"); return

    # Accept both column naming conventions
    def _get(row, *keys):
        for k in keys:
            if k in row:
                return float(row[k])
        return 0.0

    models  = [r["model"] for r in rows if r["model"] in MODELS]
    rows_d  = {r["model"]: r for r in rows}

    x = np.arange(len(models)); w = 0.28
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=False)

    for ax, (pv_key, pd_key, gap_key), title in zip(
        axes,
        [("pv_accuracy",  "plantdoc_accuracy",  "gap_accuracy"),
         ("pv_f1_macro",  "plantdoc_f1_macro",   "gap_f1_macro")],
        ["Accuracy", "Macro F1-Score"],
    ):
        pv_vals  = [_get(rows_d[m], pv_key)  for m in models]
        pd_vals  = [_get(rows_d[m], pd_key)  for m in models]
        gap_vals = [_get(rows_d[m], gap_key) for m in models]

        b1 = ax.bar(x - w, pv_vals,  w, label="PlantVillage", color=DATASET_COLORS["plantvillage"], zorder=3)
        b2 = ax.bar(x,     pd_vals,  w, label="PlantDoc",     color=DATASET_COLORS["plantdoc"],     zorder=3)
        b3 = ax.bar(x + w, gap_vals, w, label="Gap ↓",        color=PALETTE["purple"],              zorder=3, alpha=0.85)

        for bars in [b1, b2, b3]:
            for rect in bars:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.008,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

        ax.set_title(title); ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=15, ha="right")
        ax.set_ylim(0, 1.22)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Performance Under Domain Shift: PlantVillage → PlantDoc",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, figures_dir / "fig_domain_gap_bars")
    print("  Saved: fig_domain_gap_bars.png")


# ── Figure 2: accuracy drop chart ────────────────────────────────────────────

def plot_accuracy_drop(metrics_dir: Path, figures_dir: Path) -> None:
    apply_ieee_style()
    rows = _load_shift_rows(metrics_dir)
    if not rows:
        print("  [skip] domain_shift_summary.csv not found"); return

    rows_d = {r["model"]: r for r in rows if r["model"] in MODELS}
    models = [m for m in MODELS if m in rows_d]

    def _f(row, *keys):
        for k in keys:
            if k in row: return float(row[k])
        return 0.0

    abs_drops = [_f(rows_d[m], "abs_drop_accuracy", "gap_accuracy")          for m in models]
    rel_drops = [_f(rows_d[m], "rel_drop_accuracy_%", "gap_accuracy") * (
                 100 if _f(rows_d[m], "rel_drop_accuracy_%") < 1 else 1)      for m in models]

    # Recompute rel drop properly
    rel_drops = []
    for m in models:
        pv  = _f(rows_d[m], "pv_accuracy")
        gap = _f(rows_d[m], "abs_drop_accuracy", "gap_accuracy")
        rel_drops.append((gap / pv * 100) if pv > 0 else 0.0)

    x = np.arange(len(models)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(5.0, 3.5))
    ax2 = ax1.twinx()

    colors = [MODEL_COLORS.get(m, "#333") for m in models]
    b1 = ax1.bar(x - w / 2, abs_drops, w, color=colors, alpha=0.85, label="Absolute drop", zorder=3)
    b2 = ax2.bar(x + w / 2, rel_drops, w, color=colors, alpha=0.50, label="Relative drop (%)", zorder=3,
                 hatch="//")

    for rect, v in zip(b1, abs_drops):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for rect, v in zip(b2, rel_drops):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    ax1.set_ylabel("Absolute Accuracy Drop"); ax2.set_ylabel("Relative Drop (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax1.set_ylim(0, max(abs_drops) * 1.25)
    ax2.set_ylim(0, max(rel_drops) * 1.25)

    handles = [
        Patch(color="#555", alpha=0.85, label="Absolute drop"),
        Patch(color="#555", alpha=0.50, hatch="//", label="Relative drop (%)"),
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=7)
    ax1.set_title("Accuracy Drop: PlantVillage → PlantDoc",
                  fontsize=10, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, figures_dir / "fig_accuracy_drop")
    print("  Saved: fig_accuracy_drop.png")


# ── Figure 3: per-class F1 heatmap ───────────────────────────────────────────

def plot_perclass_f1_heatmap(metrics_dir: Path, figures_dir: Path) -> None:
    import seaborn as sns
    apply_ieee_style()

    all_data: dict[str, dict] = {}
    for m in MODELS:
        d = _load_per_class_gap(metrics_dir, m)
        if d:
            all_data[m] = d

    if not all_data:
        print("  [skip] No per_class_gap CSVs found"); return

    classes    = sorted(next(iter(all_data.values())).keys())
    short_cls  = [c.replace("Tomato_", "Tom.").replace("Potato__", "Pot.")
                   .replace("Pepper__bell___", "Pep.").replace("_", " ")
                  for c in classes]
    col_labels, matrix = [], []

    for m in MODELS:
        if m not in all_data: continue
        col_labels += [f"{MODEL_LABELS[m]}\n(PV)", f"{MODEL_LABELS[m]}\n(PD)"]

    for cls in classes:
        row = []
        for m in MODELS:
            if m not in all_data: continue
            row += [all_data[m][cls]["pv"], all_data[m][cls]["pd"]]
        matrix.append(row)

    mat = np.array(matrix)
    n_r, n_c = mat.shape
    fig, ax = plt.subplots(figsize=(n_c * 1.0 + 1.5, n_r * 0.48 + 1.5))

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_c)); ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(range(n_r)); ax.set_yticklabels(short_cls,  fontsize=7.5)

    for r in range(n_r):
        for c in range(n_c):
            v = mat[r, c]
            color = "white" if v < 0.45 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=color)

    # Vertical separators between model groups
    present = [m for m in MODELS if m in all_data]
    for i in range(1, len(present)):
        ax.axvline(i * 2 - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("F1-Score", fontsize=8)
    ax.set_title("Per-Class F1: PlantVillage (PV) vs PlantDoc (PD)",
                 fontsize=10, fontweight="bold", pad=8)
    plt.tight_layout()
    save_figure(fig, figures_dir / "fig_perclass_f1_heatmap")
    print("  Saved: fig_perclass_f1_heatmap.png")


# ── Figure 4: per-class degradation ranking ──────────────────────────────────

def plot_class_degradation_ranking(metrics_dir: Path, figures_dir: Path) -> None:
    apply_ieee_style()

    # Use the best-performing model on PlantDoc (ResNet50 by our results)
    for target in ["resnet50", "efficientnetb0", "mobilenetv2"]:
        data = _load_per_class_gap(metrics_dir, target)
        if data:
            break
    else:
        print("  [skip] No per_class_gap CSV found"); return

    items   = sorted(data.items(), key=lambda x: -x[1]["gap"])
    classes = [c.replace("Tomato_", "Tom.").replace("Potato__", "Pot.")
                .replace("Pepper__bell___", "Pep.").replace("_", " ")
               for c, _ in items]
    gaps    = [v["gap"] for _, v in items]
    pd_f1s  = [v["pd"]  for _, v in items]

    n  = len(classes)
    fig, ax = plt.subplots(figsize=(5.5, max(3.2, n * 0.40)))
    colors = [PALETTE["red"] if g > 0.15 else PALETTE["green"] for g in gaps]

    bars = ax.barh(range(n), gaps, color=colors, height=0.65, zorder=3)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("F1-Score Drop (PlantVillage → PlantDoc)")
    ax.set_title(
        f"Per-Class Performance Degradation\n({MODEL_LABELS.get(target, target)})",
        fontsize=10, fontweight="bold",
    )

    for bar, pd_f1 in zip(bars, pd_f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"PD={pd_f1:.2f}", va="center", fontsize=6.5)

    ax.legend(handles=[
        Patch(color=PALETTE["red"],   label="Large drop (>0.15)"),
        Patch(color=PALETTE["green"], label="Moderate drop (≤0.15)"),
    ], fontsize=7.5, loc="lower right")

    plt.tight_layout()
    save_figure(fig, figures_dir / "fig_class_degradation_ranking")
    print("  Saved: fig_class_degradation_ranking.png")


# ── Figure 5: model performance radar ────────────────────────────────────────

def plot_performance_radar(metrics_dir: Path, figures_dir: Path) -> None:
    """Spider/radar chart comparing models on 4 metrics on PlantDoc."""
    apply_ieee_style()

    metrics_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    n_metrics = len(metrics_keys)

    model_scores: dict[str, list[float]] = {}
    for m in MODELS:
        legacy_name = "efficientnet_b0" if m == "efficientnetb0" else m
        candidates = [
            metrics_dir / f"{m}_plantdoc_metrics.json",
            metrics_dir.parent.parent / "src" / "outputs" / "domain_shift"
            / f"report_{legacy_name}_plantdoc.json",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        with open(path) as f:
            d = json.load(f)
        # New pipeline: top-level keys; legacy: nested under 'macro avg'
        macro = d.get("macro avg", {})
        scores = [
            d.get("accuracy", macro.get("precision", 0.0)),   # accuracy not in legacy
            macro.get("precision", d.get("precision_macro", 0.0)),
            macro.get("recall",    d.get("recall_macro",    0.0)),
            macro.get("f1-score",  d.get("f1_macro",        0.0)),
        ]
        # For legacy files, read accuracy from micro avg if available
        if "micro avg" in d and "accuracy" not in d:
            scores[0] = d["micro avg"].get("precision", scores[0])
        model_scores[m] = scores

    if not model_scores:
        print("  [skip] No plantdoc metrics found for radar chart"); return

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(4.0, 4.0),
                           subplot_kw=dict(polar=True))

    for model, scores in model_scores.items():
        vals = scores + scores[:1]
        ax.plot(angles, vals, color=MODEL_COLORS.get(model, "#333"),
                linewidth=1.5, label=MODEL_LABELS.get(model, model))
        ax.fill(angles, vals, color=MODEL_COLORS.get(model, "#333"), alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6.5)
    ax.set_title("PlantDoc Performance Comparison",
                 fontsize=9, fontweight="bold", pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5)

    plt.tight_layout()
    save_figure(fig, figures_dir / "fig_performance_radar")
    print("  Saved: fig_performance_radar.png")
