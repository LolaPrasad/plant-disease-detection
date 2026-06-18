"""
Classification metrics for in-domain and cross-domain evaluation.

All functions accept numpy arrays of ground-truth and predicted labels.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from datasets.plantvillage import CANONICAL_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute the full classification metric suite.

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_per_class, recall_per_class, f1_per_class,
        confusion_matrix, classification_report
    """
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics — only for classes present in y_true
    labels_present = sorted(set(y_true))
    prec_pc   = precision_score(y_true, y_pred, average=None,
                                labels=labels_present, zero_division=0)
    recall_pc = recall_score(y_true, y_pred, average=None,
                             labels=labels_present, zero_division=0)
    f1_pc     = f1_score(y_true, y_pred, average=None,
                         labels=labels_present, zero_division=0)

    per_class_names = [CANONICAL_CLASSES[i] for i in labels_present]

    cm     = confusion_matrix(y_true, y_pred, labels=labels_present)
    report = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=per_class_names,
        zero_division=0,
    )

    return {
        "accuracy":            float(acc),
        "precision_macro":     float(prec),
        "recall_macro":        float(recall),
        "f1_macro":            float(f1),
        "precision_per_class": {n: float(v) for n, v in zip(per_class_names, prec_pc)},
        "recall_per_class":    {n: float(v) for n, v in zip(per_class_names, recall_pc)},
        "f1_per_class":        {n: float(v) for n, v in zip(per_class_names, f1_pc)},
        "confusion_matrix":    cm.tolist(),
        "confusion_matrix_labels": per_class_names,
        "classification_report": report,
        "n_samples":           int(len(y_true)),
        "labels_present":      labels_present,
    }


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for accuracy.

    Returns
    -------
    (lower, upper) bounds of the CI.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    correct = (y_true == y_pred).astype(float)
    boot_accs = [
        correct[rng.integers(0, n, size=n)].mean()
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_accs, alpha * 100))
    upper = float(np.percentile(boot_accs, (1 - alpha) * 100))
    return lower, upper


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """
    McNemar's test comparing two classifiers on the same test set.

    Parameters
    ----------
    y_true   : ground-truth labels
    preds_a  : predictions from model A
    preds_b  : predictions from model B

    Returns
    -------
    dict with keys: statistic, p_value, significant (p < 0.05),
                    n_both_wrong, n_a_right_b_wrong, n_a_wrong_b_right, n_both_right
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # McNemar contingency table
    n_both_right       = int(( correct_a &  correct_b).sum())
    n_a_right_b_wrong  = int(( correct_a & ~correct_b).sum())
    n_a_wrong_b_right  = int((~correct_a &  correct_b).sum())
    n_both_wrong       = int((~correct_a & ~correct_b).sum())

    b = n_a_right_b_wrong
    c = n_a_wrong_b_right

    # Use continuity-corrected McNemar (Edwards' correction) when b+c < 25
    if b + c == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value   = float(1 - stats.chi2.cdf(statistic, df=1))

    return {
        "statistic":          float(statistic),
        "p_value":            float(p_value),
        "significant":        p_value < 0.05,
        "n_both_right":       n_both_right,
        "n_a_right_b_wrong":  n_a_right_b_wrong,
        "n_a_wrong_b_right":  n_a_wrong_b_right,
        "n_both_wrong":       n_both_wrong,
    }


def domain_shift_summary(
    pv_metrics:  dict,
    pd_metrics:  dict,
    model_name:  str,
) -> dict:
    """
    Compute absolute and relative performance drops between domains.

    Performance Drop (%) = (PV_acc - PD_acc) / PV_acc × 100
    """
    pv_acc = pv_metrics["accuracy"]
    pd_acc = pd_metrics["accuracy"]
    pv_f1  = pv_metrics["f1_macro"]
    pd_f1  = pd_metrics["f1_macro"]

    abs_drop_acc = pv_acc - pd_acc
    rel_drop_acc = (abs_drop_acc / pv_acc * 100) if pv_acc > 0 else 0.0
    abs_drop_f1  = pv_f1 - pd_f1
    rel_drop_f1  = (abs_drop_f1 / pv_f1 * 100) if pv_f1 > 0 else 0.0

    return {
        "model":               model_name,
        "pv_accuracy":         pv_acc,
        "pd_accuracy":         pd_acc,
        "abs_drop_accuracy":   abs_drop_acc,
        "rel_drop_accuracy_%": rel_drop_acc,
        "pv_f1_macro":         pv_f1,
        "pd_f1_macro":         pd_f1,
        "abs_drop_f1":         abs_drop_f1,
        "rel_drop_f1_%":       rel_drop_f1,
    }
