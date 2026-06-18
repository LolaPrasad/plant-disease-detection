"""
Core training loop.

Features:
  - Mixed-precision (torch.amp) — halves GPU memory, ~1.5× faster
  - ReduceLROnPlateau scheduler monitoring val accuracy
  - Early stopping to prevent overfitting (patience from config)
  - TensorBoard logging of loss, accuracy, and learning rate per epoch
  - Best-model checkpoint saved whenever val accuracy improves
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast as _autocast
    import contextlib

    @contextlib.contextmanager
    def _safe_autocast(device_type: str, enabled: bool):
        if enabled and device_type == "cuda":
            with _autocast(device_type=device_type, enabled=True):
                yield
        else:
            yield

except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore
    import contextlib

    @contextlib.contextmanager
    def _safe_autocast(device_type: str, enabled: bool):  # type: ignore
        yield
from torch.utils.data import DataLoader
import subprocess as _subprocess, sys as _sys
def _check_tensorboard() -> bool:
    """Check tensorboard availability without crashing the main process."""
    result = _subprocess.run(
        [_sys.executable, "-c", "from torch.utils.tensorboard import SummaryWriter"],
        capture_output=True, timeout=10,
    )
    return result.returncode == 0

_TB_AVAILABLE: bool = _check_tensorboard()
SummaryWriter = None  # type: ignore
if _TB_AVAILABLE:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore  # noqa: F811
    except Exception:
        _TB_AVAILABLE = False
from tqdm import tqdm

from models.checkpoint import save_checkpoint
from utils.logging_utils import get_logger

logger = get_logger("trainer")


class EarlyStopping:
    """Stop training when val accuracy stops improving for `patience` epochs."""

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_val_acc: float = 0.0
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.counter = 0
            return True   # improved — save this as best
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False   # no improvement


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """Run one pass over `loader`. Returns (avg_loss, accuracy)."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    use_autocast = scaler is not None

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False,
                                   desc="train" if is_train else "val "):
            images, labels = images.to(device, non_blocking=True), \
                             labels.to(device, non_blocking=True)

            with _safe_autocast(device_type=amp_device, enabled=use_autocast):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds      = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, list[float]]:
    """
    Full training run. Returns history dict with per-epoch metrics.

    Parameters
    ----------
    model       : nn.Module   — model to train (already on device)
    model_name  : str         — used for checkpoint filenames and TB run name
    train_loader, val_loader  — DataLoaders
    cfg         : dict        — merged config
    device      : torch.device

    Returns
    -------
    history : dict with keys
        'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'
    """
    tcfg   = cfg["training"]
    epochs  = tcfg["epochs"]
    lr      = tcfg["learning_rate"]
    wd      = tcfg.get("weight_decay", 1e-5)
    patience= tcfg["early_stopping_patience"]
    use_amp = tcfg.get("mixed_precision", True) and device.type == "cuda"

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    log_dir  = Path(cfg["paths"].get("logs_dir", "outputs/logs")) / model_name

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=tcfg["scheduler"]["mode"],
        factor=tcfg["scheduler"]["factor"],
        patience=tcfg["scheduler"]["patience"],
        min_lr=tcfg["scheduler"]["min_lr"],
    )
    criterion  = nn.CrossEntropyLoss()
    scaler     = GradScaler("cuda") if use_amp else None
    early_stop = EarlyStopping(patience=patience)
    if _TB_AVAILABLE and SummaryWriter is not None:
        try:
            writer: Any = SummaryWriter(log_dir=str(log_dir))
        except Exception:
            writer = None
    else:
        writer = None

    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }

    logger.info(f"Training {model_name} for up to {epochs} epochs "
                f"| AMP={use_amp} | device={device}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, is_train=True)
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, None, device, is_train=False)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # TensorBoard (guarded — writer may be None if tensorboard crashed on import)
        if writer is not None:
            writer.add_scalars("loss",     {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("accuracy", {"train": train_acc,  "val": val_acc},  epoch)
            writer.add_scalar ("lr",       current_lr,                             epoch)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        # Checkpoint + early stopping
        is_best = early_stop.step(val_acc)
        save_checkpoint(
            model, optimizer, scheduler,
            epoch=epoch,
            val_accuracy=val_acc,
            val_loss=val_loss,
            model_name=model_name,
            cfg=cfg,
            checkpoint_dir=ckpt_dir,
            is_best=is_best,
        )

        if early_stop.should_stop:
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(best val_acc={early_stop.best_val_acc:.4f})"
            )
            break

    if writer is not None:
        writer.close()

    # Persist history to JSON for later analysis
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / f"{model_name}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {history_path}")

    return history
