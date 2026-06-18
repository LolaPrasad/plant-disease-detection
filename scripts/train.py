"""
Train a model on PlantVillage.

Usage:
    python scripts/train.py --model efficientnetb0
    python scripts/train.py --model mobilenetv2
    python scripts/train.py --model resnet50
    python scripts/train.py --model resnet50 --epochs 20 --batch-size 16

All hyperparameters default to configs/<model>.yaml; CLI flags override them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from datasets.loaders import get_plantvillage_loaders
from models.model_factory import build_model
from training.trainer import train
from utils.config import load_config
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a plant disease classifier")
    p.add_argument("--model",       required=True,
                   choices=["efficientnetb0", "mobilenetv2", "resnet50"],
                   help="Architecture to train")
    p.add_argument("--epochs",      type=int,   default=None,
                   help="Override number of epochs from config")
    p.add_argument("--batch-size",  type=int,   default=None,
                   help="Override batch size from config")
    p.add_argument("--lr",          type=float, default=None,
                   help="Override learning rate from config")
    p.add_argument("--config-dir",  type=str,   default=str(ROOT / "configs"),
                   help="Directory containing YAML configs")
    p.add_argument("--no-amp",      action="store_true",
                   help="Disable mixed-precision training")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.model, args.config_dir)

    # CLI overrides
    if args.epochs     is not None: cfg["training"]["epochs"]        = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"]    = args.batch_size
    if args.lr         is not None: cfg["training"]["learning_rate"] = args.lr
    if args.no_amp:                 cfg["training"]["mixed_precision"] = False

    # Paths are relative to project root
    for key in cfg["paths"]:
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            cfg["paths"][key] = str(ROOT / p)

    set_seed(cfg["seed"])

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_plantvillage_loaders(cfg)
    logger.info(
        f"PlantVillage — train: {len(train_loader.dataset)} "
        f"val: {len(val_loader.dataset)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    logger.info(
        f"Model: {args.model} | "
        f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M"
    )

    # Save the exact config used for this run
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = metrics_dir / f"{args.model}_config.json"
    with open(config_snapshot, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info(f"Config snapshot: {config_snapshot}")

    # ── Train ─────────────────────────────────────────────────────────────────
    history = train(
        model=model,
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    best_val_acc = max(history["val_acc"])
    logger.info(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    logger.info(
        f"Best checkpoint: {cfg['paths']['checkpoints_dir']}/{args.model}_best.pt"
    )


if __name__ == "__main__":
    main()
