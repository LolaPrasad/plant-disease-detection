"""Unit tests for the Grad-CAM explainability module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluator import run_inference
from explainability.gradcam import GradCAMGenerator, _denormalise
from explainability.sample_selector import select_samples
from models.model_factory import EfficientNetB0, MobileNetV2


@pytest.fixture(scope="module")
def tiny_loader():
    x = torch.randn(15, 3, 224, 224)
    y = torch.arange(15)
    return DataLoader(TensorDataset(x, y), batch_size=5)


@pytest.fixture(scope="module")
def eff_model():
    return EfficientNetB0(num_classes=15)


@pytest.fixture(scope="module")
def inference_results(eff_model, tiny_loader):
    device = torch.device("cpu")
    y_true, y_pred, y_prob = run_inference(eff_model, tiny_loader, device)
    return y_true, y_pred, y_prob


class TestDenormalise:
    def test_output_shape(self):
        t = torch.randn(3, 224, 224)
        out = _denormalise(t)
        assert out.shape == (224, 224, 3)

    def test_range(self):
        t = torch.randn(3, 224, 224)
        out = _denormalise(t)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestGradCAMGenerator:
    def test_heatmap_shape(self, eff_model):
        gen = GradCAMGenerator(eff_model, torch.device("cpu"))
        t = torch.randn(3, 224, 224)
        cam_img, raw_cam = gen.generate_heatmap(t, target_class=0)
        assert cam_img.shape == (224, 224, 3)
        assert raw_cam.shape == (224, 224)

    def test_raw_cam_range(self, eff_model):
        gen = GradCAMGenerator(eff_model, torch.device("cpu"))
        t = torch.randn(3, 224, 224)
        _, raw_cam = gen.generate_heatmap(t, target_class=3)
        assert raw_cam.min() >= 0.0
        assert raw_cam.max() <= 1.0 + 1e-5   # allow tiny fp error

    def test_no_target_class(self, eff_model):
        """Should default to predicted class without error."""
        gen = GradCAMGenerator(eff_model, torch.device("cpu"))
        t = torch.randn(3, 224, 224)
        cam_img, _ = gen.generate_heatmap(t, target_class=None)
        assert cam_img.shape == (224, 224, 3)


class TestSampleSelector:
    def test_correct_category(self, tiny_loader, inference_results):
        y_true, y_pred, _ = inference_results
        samples = select_samples(
            tiny_loader, y_true, y_pred,
            category="correct_pv", n_per_class=1, max_total=10,
        )
        for s in samples:
            assert s["true_label"] == s["pred_label"]

    def test_wrong_category(self, tiny_loader, inference_results):
        y_true, y_pred, _ = inference_results
        samples = select_samples(
            tiny_loader, y_true, y_pred,
            category="wrong_pd", n_per_class=1, max_total=10,
        )
        for s in samples:
            assert s["true_label"] != s["pred_label"]

    def test_max_total_respected(self, tiny_loader, inference_results):
        y_true, y_pred, _ = inference_results
        samples = select_samples(
            tiny_loader, y_true, y_pred,
            category="wrong_pd", n_per_class=1, max_total=2,
        )
        assert len(samples) <= 2

    def test_sample_has_required_keys(self, tiny_loader, inference_results):
        y_true, y_pred, _ = inference_results
        samples = select_samples(
            tiny_loader, y_true, y_pred,
            category="correct_pv", n_per_class=1, max_total=5,
        )
        if samples:
            for key in ("image_tensor", "true_label", "pred_label",
                        "category", "sample_idx"):
                assert key in samples[0]


class TestGradCAMSave:
    def test_saves_png(self, eff_model, tiny_loader, inference_results):
        y_true, y_pred, _ = inference_results
        device = torch.device("cpu")
        gen = GradCAMGenerator(eff_model, device)

        samples = select_samples(
            tiny_loader, y_true, y_pred,
            category="wrong_pd", n_per_class=1, max_total=2,
        )

        from explainability.gradcam import generate_gradcam_for_samples
        with tempfile.TemporaryDirectory() as d:
            saved = generate_gradcam_for_samples(
                gen, samples[:1],
                save_dir=Path(d),
                model_name="efficientnetb0",
                max_per_category=1,
            )
            assert len(saved) == 1
            assert Path(saved[0]["file"]).exists()
            assert Path(saved[0]["file"]).suffix == ".png"
