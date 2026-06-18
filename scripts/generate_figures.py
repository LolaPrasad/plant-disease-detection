"""
Generate all publication-quality figures from stored metrics.

Reads from: outputs/metrics/  (or legacy src/outputs/domain_shift/)
Writes to:  outputs/figures/

Usage:
    python scripts/generate_figures.py               # all figures
    python scripts/generate_figures.py --only curves # just training curves
    python scripts/generate_figures.py --only cm     # just confusion matrices
    python scripts/generate_figures.py --only domain # just domain shift charts

Available --only values: curves, cm, domain, all (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.logging_utils import get_logger
from visualization.confusion_matrices import plot_all_confusion_matrices
from visualization.performance_charts import (
    plot_accuracy_drop,
    plot_class_degradation_ranking,
    plot_domain_gap_bars,
    plot_perclass_f1_heatmap,
    plot_performance_radar,
)
from visualization.training_curves import plot_all_training_curves

logger = get_logger("generate_figures")


def resolve(cfg: dict) -> tuple[Path, Path]:
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    if not metrics_dir.is_absolute():
        metrics_dir = ROOT / metrics_dir
    if not figures_dir.is_absolute():
        figures_dir = ROOT / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, figures_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication figures")
    p.add_argument("--only", default="all",
                   choices=["all", "curves", "cm", "domain"],
                   help="Which figure group to generate")
    p.add_argument("--config-dir", default=str(ROOT / "configs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(None, args.config_dir)
    metrics_dir, figures_dir = resolve(cfg)

    logger.info(f"Metrics dir : {metrics_dir}")
    logger.info(f"Figures dir : {figures_dir}")

    only = args.only

    # ── Training curves ────────────────────────────────────────────────────
    if only in ("all", "curves"):
        logger.info("\n[Training Curves]")
        plot_all_training_curves(metrics_dir, figures_dir)

    # ── Confusion matrices ─────────────────────────────────────────────────
    if only in ("all", "cm"):
        logger.info("\n[Confusion Matrices]")
        plot_all_confusion_matrices(metrics_dir, figures_dir)

    # ── Domain shift figures ───────────────────────────────────────────────
    if only in ("all", "domain"):
        logger.info("\n[Domain Shift Charts]")
        plot_domain_gap_bars(metrics_dir, figures_dir)
        plot_accuracy_drop(metrics_dir, figures_dir)
        plot_perclass_f1_heatmap(metrics_dir, figures_dir)
        plot_class_degradation_ranking(metrics_dir, figures_dir)
        plot_performance_radar(metrics_dir, figures_dir)

    # ── List all outputs ───────────────────────────────────────────────────
    all_figs = sorted(figures_dir.rglob("*.png"))
    logger.info(f"\nTotal figures: {len(all_figs)}")
    for f in all_figs:
        logger.info(f"  {f.relative_to(figures_dir)}")


if __name__ == "__main__":
    main()
