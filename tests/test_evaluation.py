"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    bootstrap_accuracy_ci,
    compute_metrics,
    domain_shift_summary,
    mcnemar_test,
)


def _make_perfect(n: int = 50, n_classes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    y = np.tile(np.arange(n_classes), n // n_classes + 1)[:n]
    return y, y.copy()


def _make_random(n: int = 100, n_classes: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, n)
    y_pred = rng.integers(0, n_classes, n)
    return y_true, y_pred


class TestComputeMetrics:
    def test_perfect_accuracy(self):
        y_true, y_pred = _make_perfect()
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)

    def test_keys_present(self):
        y_true, y_pred = _make_random()
        m = compute_metrics(y_true, y_pred)
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "confusion_matrix", "n_samples"):
            assert key in m

    def test_n_samples_correct(self):
        y_true, y_pred = _make_random(n=73)
        m = compute_metrics(y_true, y_pred)
        assert m["n_samples"] == 73

    def test_per_class_keys(self):
        y_true, y_pred = _make_random()
        m = compute_metrics(y_true, y_pred)
        assert isinstance(m["f1_per_class"], dict)
        assert len(m["f1_per_class"]) <= 15


class TestBootstrapCI:
    def test_ci_bounds_ordered(self):
        y_true, y_pred = _make_random()
        lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=200)
        assert lo <= hi

    def test_ci_within_0_1(self):
        y_true, y_pred = _make_random()
        lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=200)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_perfect_ci_is_one(self):
        y_true, y_pred = _make_perfect(n=50)
        lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=200)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)


class TestMcNemar:
    def test_identical_models_not_significant(self):
        y_true, y_pred = _make_random()
        result = mcnemar_test(y_true, y_pred, y_pred.copy())
        assert not result["significant"]
        assert result["statistic"] == pytest.approx(0.0)

    def test_result_keys(self):
        y_true, y_pred_a = _make_random(seed=1)
        _, y_pred_b      = _make_random(seed=2)
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        for key in ("statistic", "p_value", "significant",
                    "n_both_right", "n_a_right_b_wrong"):
            assert key in result

    def test_p_value_in_range(self):
        y_true, y_pred_a = _make_random(n=200, seed=10)
        _, y_pred_b      = _make_random(n=200, seed=11)
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert 0.0 <= result["p_value"] <= 1.0


class TestDomainShift:
    def test_zero_drop_same_metrics(self):
        m = {"accuracy": 0.85, "f1_macro": 0.84}
        s = domain_shift_summary(m, m, "test_model")
        assert s["abs_drop_accuracy"] == pytest.approx(0.0)
        assert s["rel_drop_accuracy_%"] == pytest.approx(0.0)

    def test_drop_values_correct(self):
        pv = {"accuracy": 0.90, "f1_macro": 0.88}
        pd = {"accuracy": 0.60, "f1_macro": 0.55}
        s  = domain_shift_summary(pv, pd, "test_model")
        assert s["abs_drop_accuracy"] == pytest.approx(0.30, abs=1e-6)
        assert s["rel_drop_accuracy_%"] == pytest.approx(33.333, abs=0.01)
        assert s["abs_drop_f1"] == pytest.approx(0.33, abs=1e-6)
