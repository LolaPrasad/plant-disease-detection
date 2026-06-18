"""
DataLoader factory — returns train/val/test loaders from config.

Call get_plantvillage_loaders() or get_plantdoc_loader() from training/evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from datasets.plantvillage import PlantVillageDataset
from datasets.plantdoc import PlantDocDataset
from datasets.transforms import get_train_transforms, get_val_transforms


def get_plantvillage_loaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) for PlantVillage.

    Reads dataset root from cfg['paths']['plantvillage_dir'].
    """
    root = Path(cfg["paths"]["plantvillage_dir"])
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    pin = cfg["training"]["pin_memory"]

    train_ds = PlantVillageDataset(root, "train", transform=get_train_transforms(cfg))
    val_ds   = PlantVillageDataset(root, "val",   transform=get_val_transforms(cfg))
    test_ds  = PlantVillageDataset(root, "test",  transform=get_val_transforms(cfg))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    return train_loader, val_loader, test_loader


def get_plantdoc_loader(
    cfg: dict[str, Any],
    split: str = "test",
) -> DataLoader:
    """
    Returns a DataLoader for PlantDoc (cross-domain evaluation).

    Reads dataset root from cfg['paths']['plantdoc_dir'].
    """
    root = Path(cfg["paths"]["plantdoc_dir"])
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    pin = cfg["training"]["pin_memory"]

    ds = PlantDocDataset(root, split=split, transform=get_val_transforms(cfg))
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      num_workers=nw, pin_memory=pin)
