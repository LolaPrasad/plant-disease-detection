"""
Evaluator: runs inference on a DataLoader and collects predictions.

Returns raw arrays (y_true, y_pred, y_prob) so metrics can be computed
separately — clean separation of inference from metric computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging_utils import get_logger

logger = get_logger("evaluator")


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over all batches in `loader`.

    Returns
    -------
    y_true : (N,) int array of ground-truth class indices
    y_pred : (N,) int array of predicted class indices
    y_prob : (N, C) float array of softmax probabilities
    """
    model.eval()

    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc="inference", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.append(probs)

    return (
        np.array(all_true, dtype=np.int64),
        np.array(all_pred, dtype=np.int64),
        np.vstack(all_prob).astype(np.float32),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    dataset_name: str,
    cfg: dict[str, Any],
    save_predictions: bool = True,
) -> dict:
    """
    Full evaluation pipeline for one model on one dataset.

    1. Runs inference
    2. Computes all metrics
    3. Saves predictions CSV and metrics JSON
    4. Returns metrics dict

    Parameters
    ----------
    model        : trained nn.Module
    loader       : DataLoader for the evaluation set
    device       : torch.device
    model_name   : e.g. 'efficientnetb0'
    dataset_name : 'plantvillage' or 'plantdoc'
    cfg          : merged config dict
    save_predictions : whether to write per-image CSV
    """
    import csv

    from datasets.plantvillage import CANONICAL_CLASSES
    from evaluation.metrics import compute_metrics, bootstrap_accuracy_ci

    logger.info(f"Evaluating {model_name} on {dataset_name} ...")

    y_true, y_pred, y_prob = run_inference(model, loader, device)

    metrics = compute_metrics(y_true, y_pred)
    ci_low, ci_high = bootstrap_accuracy_ci(y_true, y_pred)
    metrics["accuracy_ci_95"] = [ci_low, ci_high]

    logger.info(
        f"  accuracy={metrics['accuracy']:.4f} "
        f"({ci_low:.4f}–{ci_high:.4f} 95% CI) | "
        f"f1_macro={metrics['f1_macro']:.4f} | "
        f"n={metrics['n_samples']}"
    )

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_name}_{dataset_name}_metrics.json"

    # confusion matrix and report are verbose — save but don't log
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    save_metrics = {k: v for k, v in metrics.items()
                    if k not in ("classification_report", "_y_true", "_y_pred", "_y_prob")}
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2, cls=_NumpyEncoder)

    # Save classification report as plain text
    report_path = metrics_dir / f"{model_name}_{dataset_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(metrics["classification_report"])

    # ── Save predictions CSV ───────────────────────────────────────────────
    if save_predictions:
        preds_dir = Path(cfg["paths"]["predictions_dir"])
        preds_dir.mkdir(parents=True, exist_ok=True)
        preds_path = preds_dir / f"{model_name}_{dataset_name}_predictions.csv"

        with open(preds_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["sample_idx", "true_label", "true_class",
                 "pred_label", "pred_class", "correct",
                 "confidence"] + [f"prob_{c}" for c in CANONICAL_CLASSES]
            )
            for i, (true, pred, prob) in enumerate(zip(y_true, y_pred, y_prob)):
                writer.writerow([
                    i,
                    int(true), CANONICAL_CLASSES[int(true)],
                    int(pred), CANONICAL_CLASSES[int(pred)],
                    int(true == pred),
                    float(prob[pred]),
                    *[float(p) for p in prob],
                ])

    logger.info(f"  Metrics saved: {metrics_path}")
    if save_predictions:
        logger.info(f"  Predictions saved: {preds_path}")

    # Return metrics plus raw arrays for downstream use (Grad-CAM, plots)
    metrics["_y_true"] = y_true
    metrics["_y_pred"] = y_pred
    metrics["_y_prob"] = y_prob

    return metrics
