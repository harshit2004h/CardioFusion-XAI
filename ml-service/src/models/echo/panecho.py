"""PanEcho video model for the CardioFusion-XAI echo expert.

PanEcho is a view-agnostic, multi-task echocardiography model:

    16 frames x (3, 224, 224)  ->  ConvNeXt-Tiny per frame (768-d)
                               ->  sinusoidal time encoding
                               ->  4-layer temporal Transformer (8 heads)
                               ->  mean pooling over time (768-d video embedding)
                               ->  one linear head per task

The module/attribute names below intentionally mirror the official release so
the published ``panecho.pt`` state dict loads with ``strict`` key checking:

    encoder.encoder.model.*        ConvNeXt-Tiny image encoder
    encoder.transformer.*          temporal Transformer
    encoder.time_encoder.pe        fixed positional encoding (recomputed, not loaded)
    <task>_head.1.{weight,bias}    task heads (index 0 is Dropout)

Reference: Holste G. et al. "Complete AI-Enabled Echocardiography
Interpretation With Multitask Deep Learning." JAMA (2025).
Pretrained weights: CC BY-NC-SA 4.0 (non-commercial use only).
"""
from __future__ import annotations

import math
import os
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .panecho_tasks import (
    BINARY,
    MULTICLASS,
    REGRESSION,
    PanEchoTask,
    resolve_tasks,
)

PANECHO_CLIP_LEN = 16
PANECHO_IMAGE_SIZE = 224
PANECHO_RESIZE_TO = 256
PANECHO_EMBED_DIM = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_POSITIONAL_KEY = "encoder.time_encoder.pe"


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class SinusoidalTimeEncoding(nn.Module):
    """Fixed sin/cos encoding of frame position, added to frame embeddings."""

    def __init__(self, d_model: int, max_len: int = PANECHO_CLIP_LEN):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # Stored as a frozen Parameter (not a buffer) to keep the official key name.
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Clip has {x.size(1)} frames but the model was built for at most "
                f"{self.pe.size(1)}. Rebuild with clip_len={x.size(1)}."
            )
        return x + self.pe[:, : x.size(1), :]


class ConvNeXtFrameEncoder(nn.Module):
    """ConvNeXt-Tiny 2D encoder applied to every frame independently."""

    def __init__(self, imagenet_init: bool = False):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if imagenet_init else None
        self.model = convnext_tiny(weights=weights)
        self.n_features = self.model.classifier[-1].in_features  # 768
        self.model.classifier[-1] = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PanEchoBackbone(nn.Module):
    """Frame encoder + temporal Transformer. Returns a (batch, 768) video embedding."""

    def __init__(
        self,
        clip_len: int = PANECHO_CLIP_LEN,
        n_heads: int = 8,
        n_layers: int = 4,
        transformer_dropout: float = 0.0,
        imagenet_init: bool = False,
    ):
        super().__init__()
        self.encoder = ConvNeXtFrameEncoder(imagenet_init=imagenet_init)
        dim = self.encoder.n_features
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim,
            dropout=transformer_dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.time_encoder = SinusoidalTimeEncoding(dim, max_len=clip_len)
        self.clip_len = clip_len
        self.embed_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected video input [batch, 3, frames, height, width], got {tuple(x.shape)}."
            )
        b, c, t, h, w = x.shape
        # NOTE: this is a plain reshape of [B, C, T, H, W], *not* a permute to
        # [B, T, C, H, W]. It is exactly what the official PanEcho code does, and the
        # released weights were trained with it, so the encoder sees groups of 3
        # consecutive planes. Changing it to a permute silently degrades accuracy.
        frames = x.reshape(b * t, c, h, w)
        embeddings = self.encoder(frames).reshape(b, t, self.embed_dim)
        embeddings = self.time_encoder(embeddings)
        features = self.transformer(embeddings)
        return features.mean(dim=1)


class PanEchoModel(nn.Module):
    """PanEcho backbone with one linear head per selected task.

    ``forward`` returns ``{task_name: tensor}``:
      * regression           -> raw value, shape [B, 1]
      * binary               -> P(positive_label), shape [B, 1]  (when activations=True)
      * multi-class          -> softmax over class_names, shape [B, K]
    With ``activations=False`` every head returns raw logits (use this for training).
    """

    def __init__(
        self,
        tasks: str | Iterable[str] = "all",
        clip_len: int = PANECHO_CLIP_LEN,
        fc_dropout: float = 0.25,
        activations: bool = True,
        imagenet_init: bool = False,
    ):
        super().__init__()
        self.encoder = PanEchoBackbone(clip_len=clip_len, imagenet_init=imagenet_init)
        self.tasks: tuple[PanEchoTask, ...] = resolve_tasks(tasks if isinstance(tasks, str) else list(tasks))
        self.activations = activations
        self.clip_len = clip_len
        for task in self.tasks:
            head = nn.Sequential(nn.Dropout(p=fc_dropout), nn.Linear(self.encoder.embed_dim, task.num_outputs))
            if task.task_type != MULTICLASS:
                with torch.no_grad():
                    head[-1].bias.fill_(task.mean)
            self.add_module(f"{task.name}_head", head)

    @property
    def task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def head(self, task_name: str) -> nn.Module:
        return getattr(self, f"{task_name}_head")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_heads(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for task in self.tasks:
            out = self.head(task.name)(features)
            if self.activations:
                if task.task_type == BINARY:
                    out = torch.sigmoid(out)
                    if task.invert_sigmoid:
                        out = 1.0 - out
                elif task.task_type == MULTICLASS:
                    out = torch.softmax(out, dim=1)
            outputs[task.name] = out
        return outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_heads(self.forward_features(x))

    @torch.inference_mode()
    def predict_clips(self, clips: torch.Tensor) -> dict[str, torch.Tensor]:
        """Average activated outputs over several clips of the *same* video.

        ``clips`` has shape [num_clips, 3, T, H, W]; returns tensors with batch dim 1.
        This mirrors PanEcho's multi-clip test-time evaluation.
        """
        if clips.ndim == 4:
            clips = clips.unsqueeze(0)
        was_activated = self.activations
        self.activations = True
        try:
            outputs = self.forward(clips)
        finally:
            self.activations = was_activated
        return {name: value.mean(dim=0, keepdim=True) for name, value in outputs.items()}


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------


def _is_torch_zip(path: Path) -> bool:
    """A modern torch.save file is itself a zip with an ``*/data.pkl`` member."""
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("data.pkl") for name in archive.namelist())
    except zipfile.BadZipFile:
        return False


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    """Accept ``panecho.pt`` or a downloaded ``panecho.pt.zip`` wrapper.

    If ``path`` is an ordinary zip that *contains* a ``.pt`` file, the inner file is
    extracted next to the archive (once) and that path is returned.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PanEcho checkpoint not found: {path}")
    if not zipfile.is_zipfile(path) or _is_torch_zip(path):
        return path

    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist() if m.endswith((".pt", ".pth")) and not m.startswith("__MACOSX")]
        if not members:
            raise ValueError(f"{path} is a zip archive but contains no .pt/.pth checkpoint.")
        member = members[0]
        target = path.parent / Path(member).name
        if not target.exists():
            # Extract to a temp file first so an interrupted extraction never leaves a broken checkpoint.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".partial") as tmp:
                with archive.open(member) as source:
                    while chunk := source.read(16 * 1024 * 1024):
                        tmp.write(chunk)
            os.chmod(tmp.name, 0o644)
            Path(tmp.name).replace(target)
    return target


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Return the model weights from any of the supported checkpoint layouts."""
    if isinstance(payload, Mapping):
        for key in ("weights", "model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Could not find a state dict in the PanEcho checkpoint.")
    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        state[key] = value
    return state


def heads_in_state_dict(state: Mapping[str, Any]) -> list[str]:
    """Task names whose heads are present in a state dict (original order kept)."""
    present = {key[: -len("_head.1.weight")] for key in state if key.endswith("_head.1.weight")}
    from .panecho_tasks import ALL_TASK_NAMES

    return [name for name in ALL_TASK_NAMES if name in present]


def load_panecho_payload(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = resolve_checkpoint_path(path)
    # weights_only=True: the official checkpoint is only tensors + optimizer state,
    # so it loads without executing arbitrary pickled code.
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        payload = {"weights": payload}
    payload["_resolved_path"] = str(resolved)
    return payload


def load_panecho(
    checkpoint_path: str | os.PathLike[str],
    tasks: str | Iterable[str] = "auto",
    clip_len: int = PANECHO_CLIP_LEN,
    device: str | torch.device = "cpu",
    activations: bool = True,
    eval_mode: bool = True,
) -> PanEchoModel:
    """Build PanEcho and load pretrained (or project fine-tuned) weights.

    ``tasks='auto'`` keeps every head stored in the checkpoint (40 heads for the official
    release, fewer for a checkpoint fine-tuned on EchoNet by this project).
    """
    payload = load_panecho_payload(checkpoint_path)
    state = extract_state_dict(payload)

    available = heads_in_state_dict(state)
    if tasks == "auto":
        selected = available
    else:
        selected = list(resolve_tasks(tasks if isinstance(tasks, str) else list(tasks)))
        selected = [t.name for t in selected]
        missing_heads = [name for name in selected if name not in available]
        if missing_heads:
            raise KeyError(f"Checkpoint has no head for task(s): {missing_heads}")
    if not selected:
        raise ValueError("The checkpoint contains no PanEcho task heads.")

    model = PanEchoModel(tasks=selected, clip_len=clip_len, activations=activations)

    def head_task(key: str) -> str | None:
        return key.split("_head.", 1)[0] if "_head." in key else None

    # Drop the fixed positional table (recomputed for any clip_len) and unselected heads.
    filtered = {
        key: value
        for key, value in state.items()
        if key != _POSITIONAL_KEY and head_task(key) in (None, *selected)
    }
    result = model.load_state_dict(filtered, strict=False)
    missing = [key for key in result.missing_keys if key != _POSITIONAL_KEY]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            "PanEcho checkpoint does not match the expected architecture. "
            f"missing={missing[:10]} unexpected={list(result.unexpected_keys)[:10]}"
        )

    # Small, non-tensor info (source, licence, fine-tuning tasks, ...) for status reporting.
    model.checkpoint_metadata = {
        key: value for key, value in payload.items() if not isinstance(value, (Mapping, torch.Tensor))
    }
    model.checkpoint_metadata["fine_tuned"] = "panecho_tasks" in payload or "model_state_dict" in payload
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def build_panecho_for_finetuning(
    checkpoint_path: str | os.PathLike[str] | None,
    tasks: Iterable[str] = ("EF", "LVEDV", "LVESV"),
    clip_len: int = PANECHO_CLIP_LEN,
    fc_dropout: float = 0.25,
) -> PanEchoModel:
    """PanEcho with raw-logit outputs for fine-tuning on EchoNet-Dynamic.

    With a checkpoint, the backbone and the selected heads are warm-started from
    PanEcho. Without one, ConvNeXt starts from ImageNet weights (ablation baseline).
    """
    tasks = list(tasks)
    if checkpoint_path is None:
        return PanEchoModel(tasks=tasks, clip_len=clip_len, fc_dropout=fc_dropout, activations=False, imagenet_init=True)
    model = load_panecho(checkpoint_path, tasks=tasks, clip_len=clip_len, activations=False, eval_mode=False)
    for task_name in tasks:
        model.head(task_name)[0].p = fc_dropout
    return model


# ---------------------------------------------------------------------------
# Output decoding
# ---------------------------------------------------------------------------


def decode_outputs(outputs: Mapping[str, torch.Tensor], index: int = 0) -> dict[str, dict[str, Any]]:
    """Turn activated model outputs for one sample into JSON-friendly values."""
    from .panecho_tasks import TASKS_BY_NAME

    decoded: dict[str, dict[str, Any]] = {}
    for name, tensor in outputs.items():
        task = TASKS_BY_NAME[name]
        values = tensor[index].detach().float().cpu().reshape(-1)
        if task.task_type == REGRESSION:
            value = float(values[0])
            if name == "GLS":
                value = -value  # PanEcho predicts |GLS|
            decoded[name] = {"type": "regression", "value": value, "unit": task.unit, "description": task.description}
        elif task.task_type == BINARY:
            decoded[name] = {
                "type": "binary",
                "positive_label": task.positive_label,
                "probability": float(values[0]),
                "description": task.description,
            }
        else:
            probabilities = {label: float(p) for label, p in zip(task.class_names, values.tolist(), strict=True)}
            decoded[name] = {
                "type": "multiclass",
                "predicted_label": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
                "description": task.description,
            }
    return decoded


__all__ = [
    "PANECHO_CLIP_LEN",
    "PANECHO_IMAGE_SIZE",
    "PANECHO_RESIZE_TO",
    "PANECHO_EMBED_DIM",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PanEchoBackbone",
    "PanEchoModel",
    "load_panecho",
    "load_panecho_payload",
    "build_panecho_for_finetuning",
    "extract_state_dict",
    "heads_in_state_dict",
    "resolve_checkpoint_path",
    "decode_outputs",
]
