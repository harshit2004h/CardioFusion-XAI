"""Reusable PyTorch training loop for CardioFusion-XAI."""
from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = "min", min_delta: float = 0.0):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class Trainer:
    """General trainer supporting tensor, dict and tuple batches.

    A model may return:
      - Tensor
      - dict containing a ``logits`` key
      - dict of named logits

    For the biomarker masked multitask case, a custom ``loss_fn`` should
    consume ``(outputs, batch)`` and return either a scalar or
    ``(scalar, component_losses)``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Any, Any], Any],
        device: str | torch.device = "auto",
        scheduler: Optional[Any] = None,
        grad_clip_norm: Optional[float] = 1.0,
        amp: bool = True,
        checkpoint_dir: str | os.PathLike[str] = "artifacts/checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.device = self._resolve_device(device)
        self.amp_enabled = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _move(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move(v) for v in obj)
        return obj

    def _forward_loss(self, batch: Any):
        batch = self._move(batch)
        if isinstance(batch, dict):
            model_input = batch.get("x", batch.get("inputs", batch.get("signal", batch.get("video", batch))))
        else:
            model_input = batch[0]
        outputs = self.model(model_input)
        result = self.loss_fn(outputs, batch)
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, {}

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        total_loss, n = 0.0, 0
        parts_acc: dict[str, float] = {}
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
                loss, parts = self._forward_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_n = self._batch_size(batch)
            total_loss += float(loss.detach()) * batch_n
            n += batch_n
            for key, value in parts.items():
                parts_acc[key] = parts_acc.get(key, 0.0) + float(value) * batch_n

        metrics = {"loss": total_loss / max(n, 1)}
        metrics.update({k: v / max(n, 1) for k, v in parts_acc.items()})
        return metrics

    @staticmethod
    def _batch_size(batch: Any) -> int:
        if isinstance(batch, dict):
            for key in ("x", "inputs", "signal", "video"):
                if key in batch and torch.is_tensor(batch[key]):
                    return int(batch[key].shape[0])
            for value in batch.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        if isinstance(batch, (list, tuple)) and batch and torch.is_tensor(batch[0]):
            return int(batch[0].shape[0])
        return 1

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        self.model.eval()
        total_loss, n = 0.0, 0
        parts_acc: dict[str, float] = {}
        for batch in loader:
            loss, parts = self._forward_loss(batch)
            batch_n = self._batch_size(batch)
            total_loss += float(loss) * batch_n
            n += batch_n
            for key, value in parts.items():
                parts_acc[key] = parts_acc.get(key, 0.0) + float(value) * batch_n
        metrics = {"loss": total_loss / max(n, 1)}
        metrics.update({k: v / max(n, 1) for k, v in parts_acc.items()})
        return metrics

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 30,
        early_stopping: Optional[EarlyStopping] = None,
        monitor: str = "loss",
        save_name: str = "model.pt",
    ) -> list[dict[str, float]]:
        history = []
        best_value = math.inf
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            row = {f"train_{k}": v for k, v in train_metrics.items()}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                row.update({f"val_{k}": v for k, v in val_metrics.items()})
                current = val_metrics.get(monitor, val_metrics["loss"])
                if current < best_value:
                    best_value = current
                    self.save_checkpoint(self.checkpoint_dir / save_name, epoch, row)
                if early_stopping and early_stopping.step(current):
                    break
            else:
                self.save_checkpoint(self.checkpoint_dir / save_name, epoch, row)
            if self.scheduler is not None:
                # ReduceLROnPlateau expects a metric; other schedulers do not.
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(row.get(f"val_{monitor}", row["train_loss"]))
                else:
                    self.scheduler.step()
            history.append(row)
            print(f"epoch={epoch:03d} " + " ".join(f"{k}={v:.5f}" for k, v in row.items()))
        return history

    def save_checkpoint(self, path: str | os.PathLike[str], epoch: int, metrics: dict[str, float]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": copy.deepcopy(metrics),
            },
            path,
        )
