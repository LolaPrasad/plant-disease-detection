"""Unit tests for model initialisation, forward pass, and checkpointing."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model_factory import (
    EfficientNetB0,
    MobileNetV2,
    ResNet50,
    build_model,
    count_parameters,
)
from models.checkpoint import save_checkpoint, load_best_checkpoint
from utils.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(params=["efficientnetb0", "mobilenetv2", "resnet50"])
def model_and_cfg(request):
    cfg   = load_config(request.param, CONFIGS_DIR)
    model = build_model(cfg)
    return model, cfg, request.param


class TestModelForwardPass:
    def test_output_shape(self, model_and_cfg):
        model, cfg, _ = model_and_cfg
        x   = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == torch.Size([2, 15])

    def test_output_finite(self, model_and_cfg):
        model, cfg, _ = model_and_cfg
        x   = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_grad_cam_layer_exists(self, model_and_cfg):
        model, _, _ = model_and_cfg
        layer = model.grad_cam_target_layer
        assert isinstance(layer, torch.nn.Module)


class TestParameterCounts:
    def test_efficientnetb0_params(self):
        m = EfficientNetB0()
        p = count_parameters(m)
        # Should be ~4M
        assert 3_000_000 < p["total"] < 6_000_000

    def test_mobilenetv2_params(self):
        m = MobileNetV2()
        p = count_parameters(m)
        # Should be ~2.2M
        assert 1_500_000 < p["total"] < 4_000_000

    def test_resnet50_params(self):
        m = ResNet50()
        p = count_parameters(m)
        # Should be ~23.5M
        assert 20_000_000 < p["total"] < 30_000_000

    def test_all_params_trainable_by_default(self, model_and_cfg):
        model, _, _ = model_and_cfg
        p = count_parameters(model)
        assert p["trainable"] == p["total"]


class TestCheckpoint:
    def test_save_and_load_best(self, model_and_cfg):
        model, cfg, name = model_and_cfg
        opt   = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max")

        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(
                model, opt, sched,
                epoch=3, val_accuracy=0.87, val_loss=0.4,
                model_name=name, cfg=cfg,
                checkpoint_dir=d, is_best=True,
            )
            ckpt = load_best_checkpoint(model, name, d, torch.device("cpu"))

        assert ckpt["epoch"] == 3
        assert ckpt["val_accuracy"] == pytest.approx(0.87)
        assert ckpt["model_name"] == name

    def test_missing_checkpoint_raises(self):
        model = EfficientNetB0()
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                load_best_checkpoint(model, "efficientnetb0", d, torch.device("cpu"))

    def test_last_checkpoint_written(self, model_and_cfg):
        model, cfg, name = model_and_cfg
        opt   = torch.optim.Adam(model.parameters())
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max")

        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(
                model, opt, sched,
                epoch=1, val_accuracy=0.5, val_loss=1.0,
                model_name=name, cfg=cfg,
                checkpoint_dir=d, is_best=False,
            )
            assert (Path(d) / f"{name}_last.pt").exists()
            assert not (Path(d) / f"{name}_best.pt").exists()


class TestBuildModelAliases:
    @pytest.mark.parametrize("alias,expected_cls", [
        ("efficientnetb0", EfficientNetB0),
        ("mobilenetv2",    MobileNetV2),
        ("resnet50",       ResNet50),
    ])
    def test_alias_resolves(self, alias, expected_cls):
        cfg = load_config(alias, CONFIGS_DIR)
        m   = build_model(cfg)
        assert isinstance(m, expected_cls)

    def test_unknown_model_raises(self):
        cfg = load_config("efficientnetb0", CONFIGS_DIR)
        cfg["model"]["name"] = "vgg16"
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(cfg)
