"""
Evaluate a trained model on PlantVillage (in-domain) or PlantDoc (cross-domain).

Usage:
    # In-domain evaluation
    python scripts/evaluate.py --model efficientnetb0 --dataset plantvillage

    # Cross-domain evaluation
    python scripts/evaluate.py --model efficientnetb0 --dataset plantdoc

    # Both at once
    python scripts/evaluate.py --model resnet50 --dataset both

Outputs (in outputs/metrics/ and outputs/predictions/):
    <model>_<dataset>_metrics.json
    <model>_<dataset>_report.txt
    <model>_<dataset>_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from datasets.loaders import get_plantdoc_loader, get_plantvillage_loaders
from evaluation.evaluator import evaluate_model
from models.checkpoint import load_best_checkpoint
from models.model_factory import build_model
from utils.config import load_config
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger("evaluate")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_paths(cfg: dict) -> dict:
    for key in cfg["paths"]:
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            cfg["paths"][key] = str(ROOT / p)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate plant disease classifiers")
    p.add_argument("--model",   required=True,
                   choices=["efficientnetb0", "mobilenetv2", "resnet50"])
    p.add_argument("--dataset", required=True,
                   choices=["plantvillage", "plantdoc", "both"],
                   help="Which dataset to evaluate on")
    p.add_argument("--split",   default="test",
                   help="Which split to use for PlantVillage (default: test)")
    p.add_argument("--config-dir", default=str(ROOT / "configs"))
    p.add_argument("--no-save-preds", action="store_true",
                   help="Skip saving per-image prediction CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg    = resolve_paths(load_config(args.model, args.config_dir))
    device = get_device()
    set_seed(cfg["seed"])

    logger.info(f"Model: {args.model} | Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    try:
        ckpt = load_best_checkpoint(
            model, args.model, cfg["paths"]["checkpoints_dir"], device
        )
        logger.info(
            f"Loaded checkpoint — epoch {ckpt['epoch']}, "
            f"val_acc={ckpt['val_accuracy']:.4f}"
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    results: dict[str, dict] = {}

    # ── PlantVillage evaluation ─────────────────────────────────────────────
    if args.dataset in ("plantvillage", "both"):
        _, _, test_loader = get_plantvillage_loaders(cfg)
        logger.info(
            f"PlantVillage {args.split} set: "
            f"{len(test_loader.dataset)} images"
        )
        pv_metrics = evaluate_model(
            model, test_loader, device,
            model_name=args.model,
            dataset_name="plantvillage",
            cfg=cfg,
            save_predictions=not args.no_save_preds,
        )
        results["plantvillage"] = pv_metrics

    # ── PlantDoc evaluation ─────────────────────────────────────────────────
    if args.dataset in ("plantdoc", "both"):
        pd_loader = get_plantdoc_loader(cfg, split="test")
        logger.info(f"PlantDoc test set: {len(pd_loader.dataset)} images")
        pd_metrics = evaluate_model(
            model, pd_loader, device,
            model_name=args.model,
            dataset_name="plantdoc",
            cfg=cfg,
            save_predictions=not args.no_save_preds,
        )
        results["plantdoc"] = pd_metrics

    # ── Domain shift summary ────────────────────────────────────────────────
    if "plantvillage" in results and "plantdoc" in results:
        from evaluation.metrics import domain_shift_summary
        summary = domain_shift_summary(
            results["plantvillage"],
            results["plantdoc"],
            model_name=args.model,
        )
        logger.info("\n=== Domain Shift Summary ===")
        logger.info(
            f"  PlantVillage acc : {summary['pv_accuracy']:.4f}\n"
            f"  PlantDoc acc     : {summary['pd_accuracy']:.4f}\n"
            f"  Absolute drop    : {summary['abs_drop_accuracy']:.4f}\n"
            f"  Relative drop    : {summary['rel_drop_accuracy_%']:.1f}%\n"
            f"  PV F1 macro      : {summary['pv_f1_macro']:.4f}\n"
            f"  PD F1 macro      : {summary['pd_f1_macro']:.4f}\n"
            f"  F1 abs drop      : {summary['abs_drop_f1']:.4f}"
        )
        summary_path = Path(cfg["paths"]["metrics_dir"]) / \
                       f"{args.model}_domain_shift.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Domain shift summary saved: {summary_path}")


if __name__ == "__main__":
    main()
