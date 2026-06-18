"""
PlantVillage dataset loader.

The on-disk structure is:
    <root>/
        train/<class_folder>/<image>
        val/<class_folder>/<image>
        test/<class_folder>/<image>

Folder names on disk may differ slightly from the canonical class names in
base.yaml (e.g. underscores vs double-underscores, Tomato__Tomato_YellowLeaf…).
This module normalises folder names to the 15 canonical classes at load time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

# Canonical class list (index = class id used everywhere in this project)
CANONICAL_CLASSES: list[str] = [
    "Pepper__bell___Bacterial_spot",           # 0
    "Pepper__bell___healthy",                  # 1
    "Potato___Early_blight",                   # 2
    "Potato___Late_blight",                    # 3
    "Potato___healthy",                        # 4
    "Tomato___Bacterial_spot",                 # 5
    "Tomato___Early_blight",                   # 6
    "Tomato___Late_blight",                    # 7
    "Tomato___Leaf_Mold",                      # 8
    "Tomato___Septoria_leaf_spot",             # 9
    "Tomato___Spider_mites Two-spotted_spider_mite",  # 10
    "Tomato___Target_Spot",                    # 11
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",  # 12
    "Tomato___Tomato_mosaic_virus",            # 13
    "Tomato___healthy",                        # 14
]

_CANONICAL_IDX: dict[str, int] = {c: i for i, c in enumerate(CANONICAL_CLASSES)}


def _normalise_folder_name(name: str) -> str:
    """
    Map on-disk folder variants to canonical class names.

    Handles known quirks in the PlantVillage_processed directory:
      - 'Tomato_healthy'            → 'Tomato___healthy'
      - 'Tomato_Early_blight'       → 'Tomato___Early_blight'
      - 'Tomato__Tomato_YellowLeaf__Curl_Virus' → 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
      - 'Tomato__Tomato_mosaic_virus' → 'Tomato___Tomato_mosaic_virus'
      - 'Tomato__Target_Spot'       → 'Tomato___Target_Spot'
      - 'Tomato_Leaf_Mold'          → 'Tomato___Leaf_Mold'
      - 'Tomato_Spider_mites_Two_spotted_spider_mite'
                                    → 'Tomato___Spider_mites Two-spotted_spider_mite'
      - 'Tomato_Septoria_leaf_spot' → 'Tomato___Septoria_leaf_spot'
      - 'Tomato_Bacterial_spot'     → 'Tomato___Bacterial_spot'
      - 'Tomato_Late_blight'        → 'Tomato___Late_blight'
    """
    # Already canonical
    if name in _CANONICAL_IDX:
        return name

    # Hard-coded known aliases (on-disk → canonical)
    _ALIASES: dict[str, str] = {
        "Tomato_healthy": "Tomato___healthy",
        "Tomato_Early_blight": "Tomato___Early_blight",
        "Tomato_Late_blight": "Tomato___Late_blight",
        "Tomato_Bacterial_spot": "Tomato___Bacterial_spot",
        "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
        "Tomato_Leaf_Mold": "Tomato___Leaf_Mold",
        "Tomato_Spider_mites_Two_spotted_spider_mite":
            "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato__Target_Spot": "Tomato___Target_Spot",
        "Tomato__Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
        "Tomato__Tomato_YellowLeaf__Curl_Virus":
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        # Potato variants
        "Potato___Early_blight": "Potato___Early_blight",
        "Potato___Late_blight": "Potato___Late_blight",
        "Potato___healthy": "Potato___healthy",
    }

    if name in _ALIASES:
        return _ALIASES[name]

    # Fallback: try stripping repeated underscores and matching
    normalised = re.sub(r"_+", "_", name)
    for canonical in CANONICAL_CLASSES:
        if re.sub(r"_+", "_", canonical) == normalised:
            return canonical

    return name  # unchanged — caller checks membership


class PlantVillageDataset(Dataset):
    """
    PyTorch Dataset for the pre-split PlantVillage directory.

    Parameters
    ----------
    root : str | Path
        Path to the split directory, e.g. 'src/data/PlantVillage_processed'.
    split : str
        One of 'train', 'val', 'test'.
    transform : callable | None
        torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples: list[tuple[Path, int]] = []
        self.skipped_folders: list[str] = []

        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue
            canonical = _normalise_folder_name(folder.name)
            if canonical not in _CANONICAL_IDX:
                self.skipped_folders.append(folder.name)
                continue
            label = _CANONICAL_IDX[canonical]
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                "Check that the path is correct and images are present."
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------
    @property
    def class_names(self) -> list[str]:
        return CANONICAL_CLASSES

    @property
    def num_classes(self) -> int:
        return len(CANONICAL_CLASSES)

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {c: 0 for c in CANONICAL_CLASSES}
        for _, label in self.samples:
            counts[CANONICAL_CLASSES[label]] += 1
        return counts
