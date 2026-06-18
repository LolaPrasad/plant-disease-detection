"""
Download and prepare PlantVillage and PlantDoc datasets.

PlantDoc is already present at src/data/PlantDoc-Dataset.
PlantVillage must be downloaded from Kaggle.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --skip-plantdoc
    python scripts/download_data.py --plantvillage-only

Requirements:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logging_utils import get_logger

logger = get_logger("download_data")

PLANTVILLAGE_KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
PLANTVILLAGE_DEST = ROOT / "src" / "data" / "plantvillage_raw"
PLANTVILLAGE_PROCESSED = ROOT / "src" / "data" / "PlantVillage_processed"

PLANTDOC_DIR = ROOT / "src" / "data" / "PlantDoc-Dataset"


def check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_file = kaggle_json.exists()
    if not has_env and not has_file:
        logger.error(
            "Kaggle credentials not found.\n"
            "  Option 1: export KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy\n"
            "  Option 2: place credentials at ~/.kaggle/kaggle.json\n"
            "  Get your API key at: https://www.kaggle.com/account"
        )
        return False
    return True


def download_plantvillage() -> None:
    """Download PlantVillage dataset via Kaggle API."""
    if not check_kaggle_credentials():
        sys.exit(1)

    try:
        import kaggle  # noqa: F401
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    PLANTVILLAGE_DEST.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading PlantVillage from Kaggle → {PLANTVILLAGE_DEST}")

    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", PLANTVILLAGE_KAGGLE_DATASET,
         "-p", str(PLANTVILLAGE_DEST), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"Kaggle download failed:\n{result.stderr}")
        sys.exit(1)

    logger.info("PlantVillage download complete.")


def verify_plantdoc() -> bool:
    if not PLANTDOC_DIR.exists():
        logger.warning(f"PlantDoc not found at {PLANTDOC_DIR}")
        return False
    n = sum(1 for _ in PLANTDOC_DIR.rglob("*.jpg")) + \
        sum(1 for _ in PLANTDOC_DIR.rglob("*.JPG")) + \
        sum(1 for _ in PLANTDOC_DIR.rglob("*.png"))
    logger.info(f"PlantDoc found at {PLANTDOC_DIR} ({n} images)")
    return True


def verify_plantvillage_processed() -> bool:
    if not PLANTVILLAGE_PROCESSED.exists():
        return False
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = PLANTVILLAGE_PROCESSED / split
        if not split_dir.exists():
            return False
        n = sum(1 for _ in split_dir.rglob("*.jpg")) + \
            sum(1 for _ in split_dir.rglob("*.JPG")) + \
            sum(1 for _ in split_dir.rglob("*.png")) + \
            sum(1 for _ in split_dir.rglob("*.PNG"))
        if n == 0:
            logger.warning(
                f"PlantVillage split '{split}' exists but contains no images.\n"
                f"  The folder structure is present (empty placeholder dirs).\n"
                f"  Download the dataset with: python scripts/download_data.py"
            )
            return False
        logger.info(f"  {split}: {n} images")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download plant disease datasets")
    parser.add_argument("--skip-plantdoc", action="store_true",
                        help="Skip PlantDoc verification (already present)")
    parser.add_argument("--plantvillage-only", action="store_true",
                        help="Only download PlantVillage")
    args = parser.parse_args()

    logger.info("=== Dataset Download / Verification ===")

    # PlantDoc verification
    if not args.plantvillage_only:
        logger.info("\n[PlantDoc]")
        if verify_plantdoc():
            logger.info("  ✓ PlantDoc ready")
        else:
            logger.warning(
                "  PlantDoc not found. Clone it from:\n"
                "  git clone https://github.com/pratikkayal/PlantDoc-Dataset"
                f"  into {PLANTDOC_DIR.parent}"
            )

    # PlantVillage verification / download
    logger.info("\n[PlantVillage]")
    if verify_plantvillage_processed():
        logger.info("  ✓ PlantVillage processed splits ready")
    else:
        logger.info("  PlantVillage images not found in processed splits.")
        logger.info(
            "\n  To download, run:\n"
            "    python scripts/download_data.py --plantvillage-only\n"
            "  Then run:\n"
            "    python scripts/prepare_datasets.py\n\n"
            "  Or manually download from Kaggle:\n"
            "    https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset\n"
            "  and place images in src/data/plantvillage_raw/\n"
            "  then run: python scripts/prepare_datasets.py"
        )


if __name__ == "__main__":
    main()
