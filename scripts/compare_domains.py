"""
Aggregate results from all three models and produce:

  1. outputs/metrics/publication_tables.csv   — LaTeX-ready performance table
  2. outputs/metrics/domain_shift_summary.csv — per-model domain shift numbers
  3. outputs/metrics/statistical_tests.csv    — McNemar pairwise comparisons

Reads the pre-computed *_metrics.json files written by evaluate.py.

Usage:
    python scripts/compare_domains.py
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np

from evaluation.metrics import mcnemar_test, domain_shift_summary, bootstrap_accuracy_ci
from utils.config import load_config
from utils.logging_utils import get_logger

logger = get_logger("compare_domains")

MODELS  = ["efficientnetb0", "mobilenetv2", "resnet50"]
DATASETS = ["plantvillage", "plantdoc"]

MODEL_DISPLAY = {
    "efficientnetb0": "EfficientNetB0",
    "mobilenetv2":    "MobileNetV2",
    "resnet50":       "ResNet50",
}


def load_metrics(metrics_dir: Path, model: str, dataset: str) -> dict | None:
    path = metrics_dir / f"{model}_{dataset}_metrics.json"
    if not path.exists():
        logger.warning(f"Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions(preds_dir: Path, model: str, dataset: str) \
        -> tuple[np.ndarray, np.ndarray] | None:
    """Load y_true and y_pred from the saved predictions CSV."""
    import csv as _csv
    path = preds_dir / f"{model}_{dataset}_predictions.csv"
    if not path.exists():
        return None
    y_true, y_pred = [], []
    with open(path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["true_label"]))
            y_pred.append(int(row["pred_label"]))
    return np.array(y_true), np.array(y_pred)


def main() -> None:
    cfg_base   = load_config(None, ROOT / "configs")
    metrics_dir = ROOT / Path(cfg_base["paths"]["metrics_dir"])
    preds_dir   = ROOT / Path(cfg_base["paths"]["predictions_dir"])

    # Resolve if relative
    if not metrics_dir.is_absolute():
        metrics_dir = ROOT / metrics_dir
    if not preds_dir.is_absolute():
        preds_dir = ROOT / preds_dir

    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all metrics ────────────────────────────────────────────────
    all_metrics: dict[str, dict[str, dict]] = {}  # model → dataset → metrics
    for model in MODELS:
        all_metrics[model] = {}
        for dataset in DATASETS:
            m = load_metrics(metrics_dir, model, dataset)
            if m:
                all_metrics[model][dataset] = m

    available = [m for m in MODELS if "plantvillage" in all_metrics.get(m, {})]
    if not available:
        logger.error(
            "No evaluation results found. "
            "Run evaluate.py for each model first:\n"
            "  python scripts/evaluate.py --model efficientnetb0 --dataset both\n"
            "  python scripts/evaluate.py --model mobilenetv2    --dataset both\n"
            "  python scripts/evaluate.py --model resnet50       --dataset both"
        )
        sys.exit(1)

    # ── 1. Publication table ───────────────────────────────────────────────
    pub_path = metrics_dir / "publication_tables.csv"
    with open(pub_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Model", "Dataset",
            "Accuracy", "Accuracy_CI_low", "Accuracy_CI_high",
            "Precision_macro", "Recall_macro", "F1_macro",
            "N_samples",
        ])
        for model in MODELS:
            for dataset in DATASETS:
                m = all_metrics.get(model, {}).get(dataset)
                if m is None:
                    continue
                ci = m.get("accuracy_ci_95", [None, None])
                w.writerow([
                    MODEL_DISPLAY[model],
                    dataset.title(),
                    f"{m['accuracy']:.4f}",
                    f"{ci[0]:.4f}" if ci[0] is not None else "",
                    f"{ci[1]:.4f}" if ci[1] is not None else "",
                    f"{m['precision_macro']:.4f}",
                    f"{m['recall_macro']:.4f}",
                    f"{m['f1_macro']:.4f}",
                    m["n_samples"],
                ])
    logger.info(f"Publication table saved: {pub_path}")

    # ── 2. Domain shift summary ────────────────────────────────────────────
    ds_path = metrics_dir / "domain_shift_summary.csv"
    shift_rows = []
    with open(ds_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "pv_accuracy", "pd_accuracy",
            "abs_drop_accuracy", "rel_drop_accuracy_%",
            "pv_f1_macro", "pd_f1_macro",
            "abs_drop_f1", "rel_drop_f1_%",
        ])
        for model in MODELS:
            pv = all_metrics.get(model, {}).get("plantvillage")
            pd = all_metrics.get(model, {}).get("plantdoc")
            if pv is None or pd is None:
                continue
            s = domain_shift_summary(pv, pd, model)
            shift_rows.append(s)
            w.writerow([
                model,
                f"{s['pv_accuracy']:.4f}", f"{s['pd_accuracy']:.4f}",
                f"{s['abs_drop_accuracy']:.4f}", f"{s['rel_drop_accuracy_%']:.2f}",
                f"{s['pv_f1_macro']:.4f}", f"{s['pd_f1_macro']:.4f}",
                f"{s['abs_drop_f1']:.4f}", f"{s['rel_drop_f1_%']:.2f}",
            ])
    logger.info(f"Domain shift summary saved: {ds_path}")

    # ── 3. McNemar pairwise tests ──────────────────────────────────────────
    stat_path = metrics_dir / "statistical_tests.csv"
    with open(stat_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model_A", "model_B", "dataset",
            "mcnemar_statistic", "p_value", "significant (p<0.05)",
            "n_A_right_B_wrong", "n_A_wrong_B_right",
        ])
        for dataset in DATASETS:
            for model_a, model_b in combinations(MODELS, 2):
                pa = load_predictions(preds_dir, model_a, dataset)
                pb = load_predictions(preds_dir, model_b, dataset)
                if pa is None or pb is None:
                    continue
                y_true_a, y_pred_a = pa
                y_true_b, y_pred_b = pb
                # Ensure same samples (both datasets evaluated identically)
                if not np.array_equal(y_true_a, y_true_b):
                    logger.warning(
                        f"y_true mismatch for {model_a} vs {model_b} on {dataset}"
                    )
                    continue
                result = mcnemar_test(y_true_a, y_pred_a, y_pred_b)
                w.writerow([
                    MODEL_DISPLAY[model_a], MODEL_DISPLAY[model_b], dataset.title(),
                    f"{result['statistic']:.4f}",
                    f"{result['p_value']:.4f}",
                    "Yes" if result["significant"] else "No",
                    result["n_a_right_b_wrong"],
                    result["n_a_wrong_b_right"],
                ])
    logger.info(f"Statistical tests saved: {stat_path}")

    # ── Console summary ────────────────────────────────────────────────────
    if shift_rows:
        logger.info("\n=== Domain Shift Summary ===")
        logger.info(f"{'Model':<18} {'PV Acc':>8} {'PD Acc':>8} "
                    f"{'Abs Drop':>10} {'Rel Drop%':>10}")
        logger.info("-" * 58)
        for s in shift_rows:
            logger.info(
                f"{MODEL_DISPLAY[s['model']]:<18} "
                f"{s['pv_accuracy']:>8.4f} {s['pd_accuracy']:>8.4f} "
                f"{s['abs_drop_accuracy']:>10.4f} {s['rel_drop_accuracy_%']:>9.1f}%"
            )

        best = min(shift_rows, key=lambda x: x["rel_drop_accuracy_%"])
        logger.info(
            f"\nMost robust model: {MODEL_DISPLAY[best['model']]} "
            f"(relative drop: {best['rel_drop_accuracy_%']:.1f}%)"
        )


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Aggregate evaluation results from all models and produce "
            "publication_tables.csv, domain_shift_summary.csv, and statistical_tests.csv."
        )
    )
    p.add_argument("--config-dir", default=str(ROOT / "configs"),
                   help="Directory containing YAML configs")
    return p.parse_args()


if __name__ == "__main__":
    _parse_args()   # validates args; main() reads config independently
    main()
