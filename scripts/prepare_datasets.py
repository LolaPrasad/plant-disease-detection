"""
Prepare datasets: split PlantVillage raw images into train/val/test directories.

This script is idempotent — re-running it skips already-split classes.

Expected input:
    src/data/plantvillage_raw/
        PlantVillage/
            color/
                <ClassName>/
                    *.jpg

Output:
    src/data/PlantVillage_processed/
        train/<ClassName>/<images>
        val/<ClassName>/<images>
        test/<ClassName>/<images>

Split ratios: 70% train / 15% val / 15% test (stratified, reproducible).

Usage:
    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --train-ratio 0.7 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logging_utils import get_logger
from utils.reproducibility import set_seed
from datasets.plantvillage import CANONICAL_CLASSES, _normalise_folder_name

logger = get_logger("prepare_datasets")

RAW_DIR = ROOT / "src" / "data" / "plantvillage_raw"
OUT_DIR = ROOT / "src" / "data" / "PlantVillage_processed"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_raw_class_dirs(raw_dir: Path) -> dict[str, Path]:
    """Walk raw_dir and find one directory per canonical class."""
    found: dict[str, Path] = {}
    for d in raw_dir.rglob("*"):
        if not d.is_dir():
            continue
        canonical = _normalise_folder_name(d.name)
        if canonical in {c for c in CANONICAL_CLASSES} and canonical not in found:
            found[canonical] = d
    return found


def split_class(
    class_dir: Path,
    canonical_name: str,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, int]:
    """Copy images for one class into train/val/test splits."""
    images = sorted([p for p in class_dir.iterdir() if p.suffix in IMG_EXTENSIONS])
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    counts: dict[str, int] = {}
    for split_name, imgs in splits.items():
        dest_dir = out_dir / split_name / canonical_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            dest = dest_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        counts[split_name] = len(imgs)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split PlantVillage into train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--raw-dir",     type=str,   default=str(RAW_DIR))
    parser.add_argument("--out-dir",     type=str,   default=str(OUT_DIR))
    args = parser.parse_args()

    set_seed(args.seed)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        logger.error(
            f"Raw directory not found: {raw_dir}\n"
            "  Run: python scripts/download_data.py first."
        )
        sys.exit(1)

    logger.info(f"Searching for class directories in {raw_dir} ...")
    class_dirs = find_raw_class_dirs(raw_dir)

    if not class_dirs:
        logger.error("No class directories found. Check the raw data structure.")
        sys.exit(1)

    logger.info(f"Found {len(class_dirs)}/{len(CANONICAL_CLASSES)} classes")
    missing = [c for c in CANONICAL_CLASSES if c not in class_dirs]
    if missing:
        logger.warning(f"Missing classes: {missing}")

    summary: dict[str, dict[str, int]] = {}
    for canonical, src_dir in sorted(class_dirs.items()):
        counts = split_class(
            src_dir, canonical, out_dir,
            args.train_ratio, args.val_ratio, args.seed
        )
        summary[canonical] = counts
        logger.info(
            f"  {canonical}: "
            f"train={counts['train']} val={counts['val']} test={counts['test']}"
        )

    # Save split manifest for reproducibility
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 4),
            "class_counts": summary,
        }, f, indent=2)

    total = {s: sum(v[s] for v in summary.values()) for s in ["train", "val", "test"]}
    logger.info(f"\nTotal: train={total['train']} val={total['val']} test={total['test']}")
    logger.info(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
