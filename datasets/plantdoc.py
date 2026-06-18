"""
PlantDoc dataset loader with automatic class mapping to PlantVillage labels.

PlantDoc folder names (on disk) differ from PlantVillage canonical names.
This module:
  1. Maps PlantDoc folder names to PlantVillage class indices where a match exists.
  2. Silently skips folders with no PlantVillage equivalent (e.g. Apple, Blueberry).
  3. Records every mapping and exclusion decision in a log accessible after init.

PlantDoc on-disk structure:
    <root>/
        train/<class_folder>/<image>
        test/<class_folder>/<image>

The 'test' split is used as the cross-domain evaluation set;
'train' is used only if explicitly requested (we do not train on PlantDoc).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from datasets.plantvillage import CANONICAL_CLASSES, _CANONICAL_IDX

# ------------------------------------------------------------------
# PlantDoc folder name → PlantVillage canonical class name
# Folders not listed here are excluded with a logged reason.
# ------------------------------------------------------------------
PLANTDOC_TO_PLANTVILLAGE: dict[str, str] = {
    # Bell pepper
    "Bell_pepper leaf":       "Pepper__bell___healthy",
    "Bell_pepper leaf spot":  "Pepper__bell___Bacterial_spot",

    # Potato
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight":  "Potato___Late_blight",
    # NOTE: "Potato leaf" (healthy) exists in PlantDoc but PlantVillage
    # uses 'Potato___healthy' — map it.
    "Potato leaf":              "Potato___healthy",

    # Tomato
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato Early blight leaf":   "Tomato___Early_blight",
    "Tomato leaf late blight":    "Tomato___Late_blight",
    "Tomato mold leaf":           "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot":  "Tomato___Septoria_leaf_spot",
    # PlantDoc has no Spider mites class → excluded
    "Tomato leaf":                "Tomato___healthy",
    "Tomato leaf mosaic virus":   "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":   "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    # PlantDoc has no Target Spot class → excluded
}

# Folders explicitly excluded and why (for documentation/logging)
EXCLUDED_PLANTDOC_CLASSES: dict[str, str] = {
    "Apple Scab Leaf":            "no PlantVillage equivalent (apple not in PV-15)",
    "Apple leaf":                 "no PlantVillage equivalent (apple not in PV-15)",
    "Apple rust leaf":            "no PlantVillage equivalent (apple not in PV-15)",
    "Blueberry leaf":             "no PlantVillage equivalent (blueberry not in PV-15)",
    "Cherry leaf":                "no PlantVillage equivalent (cherry not in PV-15)",
    "Corn Gray leaf spot":        "no PlantVillage equivalent (corn not in PV-15)",
    "Corn leaf blight":           "no PlantVillage equivalent (corn not in PV-15)",
    "Corn rust leaf":             "no PlantVillage equivalent (corn not in PV-15)",
    "Peach leaf":                 "no PlantVillage equivalent (peach not in PV-15)",
    "Raspberry leaf":             "no PlantVillage equivalent (raspberry not in PV-15)",
    "Soyabean leaf":              "no PlantVillage equivalent (soybean not in PV-15)",
    "Squash Powdery mildew leaf": "no PlantVillage equivalent (squash not in PV-15)",
    "Strawberry leaf":            "no PlantVillage equivalent (strawberry not in PV-15)",
    "grape leaf":                 "no PlantVillage equivalent (grape not in PV-15)",
    "grape leaf black rot":       "no PlantVillage equivalent (grape not in PV-15)",
}


class PlantDocDataset(Dataset):
    """
    PyTorch Dataset for PlantDoc images mapped to PlantVillage class indices.

    Parameters
    ----------
    root : str | Path
        Path to the PlantDoc root directory (contains 'train' and 'test' folders).
    split : str
        'train' or 'test'. For cross-domain evaluation use 'test'.
    transform : callable | None
        torchvision transform applied to each PIL image.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples: list[tuple[Path, int]] = []
        self.mapping_log: list[dict] = []   # one entry per folder

        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_name = folder.name

            if folder_name in PLANTDOC_TO_PLANTVILLAGE:
                canonical = PLANTDOC_TO_PLANTVILLAGE[folder_name]
                label = _CANONICAL_IDX[canonical]
                images = [
                    p for p in sorted(folder.iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                ]
                for img_path in images:
                    self.samples.append((img_path, label))

                self.mapping_log.append({
                    "plantdoc_folder": folder_name,
                    "mapped_to": canonical,
                    "class_id": label,
                    "n_images": len(images),
                    "status": "mapped",
                })

            elif folder_name in EXCLUDED_PLANTDOC_CLASSES:
                self.mapping_log.append({
                    "plantdoc_folder": folder_name,
                    "mapped_to": None,
                    "class_id": None,
                    "n_images": sum(
                        1 for p in folder.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    ),
                    "status": "excluded",
                    "reason": EXCLUDED_PLANTDOC_CLASSES[folder_name],
                })

            else:
                # Unexpected folder — log it for review
                self.mapping_log.append({
                    "plantdoc_folder": folder_name,
                    "mapped_to": None,
                    "class_id": None,
                    "n_images": 0,
                    "status": "unknown",
                    "reason": "folder not found in mapping table — excluded",
                })

        if not self.samples:
            raise RuntimeError(
                f"No mapped images found in {split_dir}. "
                "Check PLANTDOC_TO_PLANTVILLAGE mapping."
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
        return {k: v for k, v in counts.items() if v > 0}

    def print_mapping_summary(self) -> None:
        print("\n=== PlantDoc → PlantVillage Mapping Summary ===")
        mapped   = [r for r in self.mapping_log if r["status"] == "mapped"]
        excluded = [r for r in self.mapping_log if r["status"] == "excluded"]
        unknown  = [r for r in self.mapping_log if r["status"] == "unknown"]

        print(f"\nMapped ({len(mapped)} folders, {len(self.samples)} images):")
        for r in sorted(mapped, key=lambda x: x["mapped_to"]):
            print(f"  '{r['plantdoc_folder']}' → '{r['mapped_to']}' ({r['n_images']} imgs)")

        print(f"\nExcluded ({len(excluded)} folders):")
        for r in sorted(excluded, key=lambda x: x["plantdoc_folder"]):
            print(f"  '{r['plantdoc_folder']}': {r.get('reason', '')}")

        if unknown:
            print(f"\nUnknown ({len(unknown)} folders — review needed):")
            for r in unknown:
                print(f"  '{r['plantdoc_folder']}'")
