"""
Checkpoint save/load helpers used by the training loop.

Checkpoint format (dict saved with torch.save):
    {
        'epoch':             int,
        'model_name':        str,
        'model_state_dict':  OrderedDict,
        'optimizer_state_dict': ...,
        'scheduler_state_dict': ...,
        'val_accuracy':      float,
        'val_loss':          float,
        'config':            dict,
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_accuracy: float,
    val_loss: float,
    model_name: str,
    cfg: dict[str, Any],
    checkpoint_dir: str | Path,
    is_best: bool = False,
) -> Path:
    """
    Save a training checkpoint.

    Always writes <model_name>_last.pt.
    If is_best=True, also writes <model_name>_best.pt (symlink-free copy).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":                epoch,
        "model_name":           model_name,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
        "val_accuracy":         val_accuracy,
        "val_loss":             val_loss,
        "config":               cfg,
    }

    last_path = checkpoint_dir / f"{model_name}_last.pt"
    torch.save(state, last_path)

    if is_best:
        best_path = checkpoint_dir / f"{model_name}_best.pt"
        torch.save(state, best_path)
        return best_path

    return last_path


def load_best_checkpoint(
    model: nn.Module,
    model_name: str,
    checkpoint_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load the best checkpoint for a model. Raises FileNotFoundError if missing."""
    path = Path(checkpoint_dir) / f"{model_name}_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No best checkpoint found at {path}. "
            "Run training first: python scripts/train.py --model {model_name}"
        )
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
