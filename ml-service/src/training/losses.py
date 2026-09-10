"""Loss functions used by CardioFusion-XAI training.

Important:
- Biomarker tasks are supervised with per-sample masks because the four
  clinical datasets do not share the same label set.
- BCEWithLogitsLoss is used for binary/multi-label heads, with optional
  per-head pos_weight.
- LET_IS / mortality-cause is multiclass and uses CrossEntropyLoss.
- Focal loss is optional for very rare MI-complication heads.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch
from torch import Tensor, nn


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: Optional[Tensor] = None, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits = logits.float()
        targets = targets.float()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if mask is not None:
            mask = mask.to(device=loss.device, dtype=loss.dtype)
            while mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            denom = mask.expand_as(loss).sum().clamp_min(1.0)
            return loss.sum() / denom
        return loss.mean() if self.reduction == "mean" else loss.sum()


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weight: Optional[Tensor] = None):
        super().__init__()
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float())
        else:
            self.class_weight = None

    def forward(self, logits: Tensor, targets: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits = logits.float()
        targets = targets.long()
        loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weight,
            reduction="none",
        )
        if mask is not None:
            mask = mask.to(device=loss.device, dtype=loss.dtype).reshape(-1)
            loss = loss.reshape(-1) * mask
            return loss.sum() / mask.sum().clamp_min(1.0)
        return loss.mean()


class FocalLoss(nn.Module):
    """Binary focal loss on logits, optionally masked."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.alpha = alpha
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits = logits.float()
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = focal * alpha_t
        if mask is not None:
            mask = mask.to(device=focal.device, dtype=focal.dtype)
            while mask.ndim < focal.ndim:
                mask = mask.unsqueeze(-1)
            focal = focal * mask
            return focal.sum() / mask.expand_as(focal).sum().clamp_min(1.0)
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """Weighted sum of named task losses.

    Expected model output:
        logits = {"task_a": Tensor, "task_b": Tensor, ...}

    Expected targets:
        targets = {"task_a": Tensor, "task_b": Tensor, ...}

    Expected masks:
        masks = {"task_a": Tensor, "task_b": Tensor, ...}
    """

    def __init__(self, task_losses: Mapping[str, nn.Module], weights: Optional[Mapping[str, float]] = None):
        super().__init__()
        self.task_losses = nn.ModuleDict(dict(task_losses))
        self.weights = dict(weights or {name: 1.0 for name in task_losses})

    def forward(
        self,
        logits: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        masks: Optional[Mapping[str, Tensor]] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        masks = masks or {}
        total = None
        parts = {}
        for name, loss_fn in self.task_losses.items():
            if name not in logits or name not in targets:
                continue
            value = loss_fn(logits[name], targets[name], masks.get(name))
            weighted = value * float(self.weights.get(name, 1.0))
            total = weighted if total is None else total + weighted
            parts[name] = value.detach()
        if total is None:
            raise RuntimeError("No overlapping tasks were found between logits and targets.")
        return total, parts
