"""
Generate Grad-CAM visualisations for a trained model.

Produces:
  1. Per-sample PNGs in outputs/gradcam/<model>/<category>/
  2. Multi-model comparison grids in outputs/gradcam/comparisons/
  3. outputs/gradcam/gradcam_manifest.csv   — catalogue of all saved files

Usage:
    python scripts/generate_gradcam.py --model efficientnetb0
    python scripts/generate_gradcam.py --model all          # all three models
    python scripts/generate_gradcam.py --model all --compare  # + grid comparisons

The comparison grids require checkpoints for all three models to exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from datasets.loaders import get_plantdoc_loader, get_plantvillage_loaders
from evaluation.evaluator import run_inference
from explainability.gradcam import GradCAMGenerator, generate_gradcam_for_samples
from explainability.sample_selector import select_samples
from models.checkpoint import load_best_checkpoint
from models.model_factory import build_model
from utils.config import load_config
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger("generate_gradcam")

ALL_MODELS = ["efficientnetb0", "mobilenetv2", "resnet50"]


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


def process_one_model(
    model_name: str,
    device: torch.device,
    gradcam_dir: Path,
    n_per_class: int,
) -> tuple[dict, list[dict]]:
    """
    Run Grad-CAM for a single model on both datasets.
    Returns (generators_entry, all_saved_records).
    """
    cfg = resolve_paths(load_config(model_name, ROOT / "configs"))
    set_seed(cfg["seed"])

    model = build_model(cfg).to(device)
    try:
        ckpt = load_best_checkpoint(
            model, model_name, cfg["paths"]["checkpoints_dir"], device
        )
        logger.info(
            f"[{model_name}] Loaded checkpoint epoch={ckpt['epoch']} "
            f"val_acc={ckpt['val_accuracy']:.4f}"
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return {}, []

    generator = GradCAMGenerator(model, device)

    # ── PlantVillage: correct predictions ──────────────────────────────────
    _, _, pv_test_loader = get_plantvillage_loaders(cfg)
    logger.info(f"[{model_name}] Running inference on PlantVillage test ...")
    pv_true, pv_pred, _ = run_inference(model, pv_test_loader, device)

    pv_correct_samples = select_samples(
        pv_test_loader, pv_true, pv_pred,
        category="correct_pv", n_per_class=n_per_class, max_total=30,
    )
    logger.info(
        f"[{model_name}] PlantVillage: "
        f"{(pv_true==pv_pred).sum()}/{len(pv_true)} correct — "
        f"selected {len(pv_correct_samples)} for Grad-CAM"
    )

    # ── PlantDoc: correct + wrong predictions ──────────────────────────────
    pd_loader = get_plantdoc_loader(cfg, split="test")
    logger.info(f"[{model_name}] Running inference on PlantDoc test ...")
    pd_true, pd_pred, _ = run_inference(model, pd_loader, device)

    pd_correct_samples = select_samples(
        pd_loader, pd_true, pd_pred,
        category="correct_pd", n_per_class=n_per_class, max_total=20,
    )
    pd_wrong_samples = select_samples(
        pd_loader, pd_true, pd_pred,
        category="wrong_pd", n_per_class=n_per_class, max_total=20,
    )
    logger.info(
        f"[{model_name}] PlantDoc: "
        f"correct={len(pd_correct_samples)} wrong={len(pd_wrong_samples)}"
    )

    # ── Generate and save heatmaps ─────────────────────────────────────────
    all_samples = pv_correct_samples + pd_correct_samples + pd_wrong_samples
    saved = generate_gradcam_for_samples(
        generator, all_samples,
        save_dir=gradcam_dir,
        model_name=model_name,
        max_per_category=5,
    )
    logger.info(f"[{model_name}] Saved {len(saved)} Grad-CAM images")

    return (
        {
            "model_name": model_name,
            "generator":  generator,
            "pv_correct_samples": pv_correct_samples,
            "pd_correct_samples": pd_correct_samples,
            "pd_wrong_samples":   pd_wrong_samples,
        },
        saved,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Grad-CAM visualisations")
    p.add_argument("--model", default="efficientnetb0",
                   help="Model name or 'all' for all three models")
    p.add_argument("--n-per-class", type=int, default=1,
                   help="Samples per class per category (default: 1)")
    p.add_argument("--compare", action="store_true",
                   help="Also generate multi-model comparison grids")
    p.add_argument("--config-dir", default=str(ROOT / "configs"))
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = get_device()

    cfg_base    = resolve_paths(load_config(None, args.config_dir))
    gradcam_dir = Path(cfg_base["paths"]["gradcam_dir"])

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    logger.info(f"Models: {models_to_run} | Device: {device}")

    all_saved: list[dict] = []
    generators_data: list[dict] = {}

    for model_name in models_to_run:
        gen_data, saved = process_one_model(
            model_name, device, gradcam_dir, args.n_per_class
        )
        all_saved.extend(saved)
        if gen_data:
            generators_data[model_name] = gen_data

    # ── Multi-model comparison grids ───────────────────────────────────────
    if args.compare and len(generators_data) > 1:
        from explainability.comparison import generate_comparison_grid

        generators = {
            name: data["generator"]
            for name, data in generators_data.items()
        }
        comp_dir = gradcam_dir / "comparisons"

        logger.info("Generating multi-model comparison grids ...")

        # Use the first model's samples as the reference (same images, all models)
        ref_model = list(generators_data.keys())[0]
        ref_data  = generators_data[ref_model]

        for cat, samples in [
            ("correct_pv", ref_data["pv_correct_samples"]),
            ("correct_pd", ref_data["pd_correct_samples"]),
            ("wrong_pd",   ref_data["pd_wrong_samples"]),
        ]:
            if not samples:
                logger.warning(f"No samples for category {cat}, skipping grid")
                continue
            save_path = comp_dir / f"comparison_{cat}.png"
            generate_comparison_grid(
                generators, samples,
                save_path=save_path,
                category=cat,
                max_rows=5,
            )

    # ── Save manifest ──────────────────────────────────────────────────────
    if all_saved:
        manifest_path = gradcam_dir / "gradcam_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_saved[0].keys())
            writer.writeheader()
            writer.writerows(all_saved)
        logger.info(f"Manifest saved: {manifest_path} ({len(all_saved)} entries)")


if __name__ == "__main__":
    main()
