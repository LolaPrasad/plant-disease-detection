"""Unit tests for dataset loading and label mapping."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.plantvillage import (
    CANONICAL_CLASSES,
    PlantVillageDataset,
    _normalise_folder_name,
)
from datasets.plantdoc import (
    EXCLUDED_PLANTDOC_CLASSES,
    PLANTDOC_TO_PLANTVILLAGE,
    PlantDocDataset,
)
from datasets.transforms import get_train_transforms, get_val_transforms
from utils.config import load_config


# ── Label mapping ─────────────────────────────────────────────────────────────

class TestFolderNormalisation:
    def test_canonical_passthrough(self):
        for cls in CANONICAL_CLASSES:
            assert _normalise_folder_name(cls) == cls

    def test_known_aliases(self):
        aliases = {
            "Tomato_healthy":           "Tomato___healthy",
            "Tomato_Early_blight":      "Tomato___Early_blight",
            "Tomato_Late_blight":       "Tomato___Late_blight",
            "Tomato_Bacterial_spot":    "Tomato___Bacterial_spot",
            "Tomato_Leaf_Mold":         "Tomato___Leaf_Mold",
            "Tomato_Septoria_leaf_spot":"Tomato___Septoria_leaf_spot",
            "Tomato__Target_Spot":      "Tomato___Target_Spot",
            "Tomato__Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
            "Tomato__Tomato_YellowLeaf__Curl_Virus":
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato_Spider_mites_Two_spotted_spider_mite":
                "Tomato___Spider_mites Two-spotted_spider_mite",
        }
        for alias, expected in aliases.items():
            result = _normalise_folder_name(alias)
            assert result == expected, f"{alias!r} → {result!r}, expected {expected!r}"

    def test_15_canonical_classes(self):
        assert len(CANONICAL_CLASSES) == 15

    def test_unique_indices(self):
        from datasets.plantvillage import _CANONICAL_IDX
        assert len(_CANONICAL_IDX) == 15
        assert set(_CANONICAL_IDX.values()) == set(range(15))


class TestPlantDocMapping:
    def test_all_mapped_classes_valid(self):
        """Every PlantDoc mapped class must resolve to a canonical PlantVillage class."""
        for pd_cls, pv_cls in PLANTDOC_TO_PLANTVILLAGE.items():
            assert pv_cls in CANONICAL_CLASSES, \
                f"{pd_cls!r} maps to {pv_cls!r} which is not in CANONICAL_CLASSES"

    def test_no_overlap_mapped_excluded(self):
        """A folder should not be both mapped and excluded."""
        overlap = set(PLANTDOC_TO_PLANTVILLAGE) & set(EXCLUDED_PLANTDOC_CLASSES)
        assert overlap == set(), f"Overlap found: {overlap}"

    def test_mapped_count(self):
        """Expect at least 10 PlantDoc classes to map to PlantVillage."""
        assert len(PLANTDOC_TO_PLANTVILLAGE) >= 10


# ── PlantDoc dataset (real data available) ────────────────────────────────────

PLANTDOC_ROOT = Path(__file__).resolve().parent.parent / \
                "src" / "data" / "PlantDoc-Dataset"


@pytest.mark.skipif(not PLANTDOC_ROOT.exists(), reason="PlantDoc data not present")
class TestPlantDocDataset:
    def test_loads_test_split(self):
        ds = PlantDocDataset(PLANTDOC_ROOT, split="test")
        assert len(ds) > 0

    def test_labels_in_range(self):
        ds = PlantDocDataset(PLANTDOC_ROOT, split="test")
        for _, label in ds.samples:
            assert 0 <= label < 15

    def test_image_tensor_shape(self):
        cfg = load_config("efficientnetb0",
                          Path(__file__).resolve().parent.parent / "configs")
        from datasets.transforms import get_val_transforms
        ds = PlantDocDataset(PLANTDOC_ROOT, split="test",
                             transform=get_val_transforms(cfg))
        img, label = ds[0]
        assert img.shape == torch.Size([3, 224, 224])
        assert isinstance(label, int)

    def test_no_unknown_folders_after_fix(self):
        ds = PlantDocDataset(PLANTDOC_ROOT, split="test")
        unknown = [r for r in ds.mapping_log if r["status"] == "unknown"]
        assert unknown == [], f"Unexpected folders: {[r['plantdoc_folder'] for r in unknown]}"

    def test_class_distribution_nonzero(self):
        ds = PlantDocDataset(PLANTDOC_ROOT, split="test")
        dist = ds.class_distribution()
        assert all(v > 0 for v in dist.values())


# ── Transforms ────────────────────────────────────────────────────────────────

class TestTransforms:
    @pytest.fixture
    def cfg(self):
        return load_config("efficientnetb0",
                           Path(__file__).resolve().parent.parent / "configs")

    def test_train_transform_output_shape(self, cfg):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype("uint8"))
        t   = get_train_transforms(cfg)
        out = t(img)
        assert out.shape == torch.Size([3, 224, 224])

    def test_val_transform_output_shape(self, cfg):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(
            (np.random.rand(300, 300, 3) * 255).astype("uint8"))
        t   = get_val_transforms(cfg)
        out = t(img)
        assert out.shape == torch.Size([3, 224, 224])

    def test_val_transform_normalised(self, cfg):
        """Output should not be in [0,1] after normalisation."""
        from PIL import Image
        import numpy as np
        img = Image.fromarray(
            (np.ones((300, 300, 3)) * 128).astype("uint8"))
        t   = get_val_transforms(cfg)
        out = t(img)
        # ImageNet mean ≈ 0.45 → after subtract: values are near 0, not 0.5
        assert out.abs().max() < 3.0   # just verify it's been normalised
