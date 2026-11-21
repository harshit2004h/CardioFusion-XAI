"""Clip sampling and pixel preprocessing that match PanEcho's training pipeline.

PanEcho contract (official EchoNet-Dynamic loader):
  * 16 *consecutive* frames at the video's native frame rate (no stride)
  * every frame resized to 256 x 256 (cv2.INTER_AREA)
  * centre crop to 224 x 224 for evaluation (random crop + light augmentation for training)
  * scaled to [0, 1] and ImageNet-normalised
  * channel order left exactly as OpenCV decodes it (BGR); echo is almost entirely
    greyscale, but keeping BGR reproduces the pretrained input distribution
  * videos shorter than 16 frames repeat their last frame ("last frame carried forward")
  * at test time several clips from the same video are averaged

The output tensor layout is [3, T, 224, 224] per clip, i.e. what
``PanEchoModel`` expects after batching: [B, 3, T, 224, 224].
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.models.echo.panecho import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PANECHO_CLIP_LEN,
    PANECHO_IMAGE_SIZE,
    PANECHO_RESIZE_TO,
)


def clip_starts(
    total_frames: int,
    clip_len: int = PANECHO_CLIP_LEN,
    num_clips: int = 1,
    random: bool = False,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Start frame of each clip.

    Evaluation (``random=False``) spreads clips evenly over the video so results are
    reproducible; training draws random starts like the original loader.
    """
    if total_frames <= 0:
        raise ValueError("Video has no frames.")
    max_start = max(total_frames - clip_len, 0)
    if max_start == 0:
        return [0] * num_clips
    if random:
        rng = rng or np.random.default_rng()
        return [int(s) for s in rng.integers(0, max_start + 1, size=num_clips)]
    if num_clips == 1:
        return [max_start // 2]
    return [int(round(s)) for s in np.linspace(0, max_start, num_clips)]


def clip_indices(start: int, total_frames: int, clip_len: int = PANECHO_CLIP_LEN) -> np.ndarray:
    """Consecutive frame indices with the last frame repeated for short videos."""
    return np.minimum(start + np.arange(clip_len), total_frames - 1).astype(np.int64)


def to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    """Coerce a decoded frame (grey, BGR or BGRA; any dtype) to HxWx3 uint8 BGR."""
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        lo, hi = float(frame.min()), float(frame.max())
        frame = ((frame - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(frame)
        frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"Unsupported frame shape {frame.shape}.")


def resize_frame(frame: np.ndarray, size: int = PANECHO_RESIZE_TO) -> np.ndarray:
    frame = to_bgr_uint8(frame)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


class PanEchoClipTransform:
    """uint8 frames [T, H, W, 3] (already 256x256) -> normalised float tensor [3, T, 224, 224]."""

    def __init__(
        self,
        train: bool = False,
        crop_size: int = PANECHO_IMAGE_SIZE,
        hflip_p: float = 0.0,
        rotation_degrees: float = 15.0,
        zoom_out_p: float = 0.5,
    ):
        # Project config disables horizontal flips for echo (it swaps LV/RV sides).
        # PanEcho itself used p=0.5; pass hflip_p=0.5 to reproduce that exactly.
        if train:
            ops: list = [
                v2.RandomZoomOut(fill=0, side_range=(1.0, 1.2), p=zoom_out_p),
                v2.RandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
                v2.RandomHorizontalFlip(p=hflip_p),
                v2.RandomRotation(degrees=(-rotation_degrees, rotation_degrees)),
            ]
        else:
            ops = [v2.CenterCrop(size=(crop_size, crop_size))]
        ops += [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
        self.transform = v2.Compose(ops)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames [T, H, W, 3], got {frames.shape}.")
        video = tv_tensors.Video(torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2))
        video = self.transform(video)  # [T, 3, 224, 224]
        return video.permute(1, 0, 2, 3).contiguous()  # [3, T, 224, 224]


def build_clips(
    frames: np.ndarray,
    clip_len: int = PANECHO_CLIP_LEN,
    num_clips: int = 4,
    train: bool = False,
    transform: PanEchoClipTransform | None = None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """All decoded (256x256) frames of one video -> [num_clips, 3, clip_len, 224, 224]."""
    if frames.shape[0] == 0:
        raise ValueError("Video yielded no frames.")
    transform = transform or PanEchoClipTransform(train=train)
    clips = []
    for start in clip_starts(frames.shape[0], clip_len, num_clips, random=train, rng=rng):
        clips.append(transform(frames[clip_indices(start, frames.shape[0], clip_len)]))
    return torch.stack(clips, dim=0)


__all__ = [
    "clip_starts",
    "clip_indices",
    "to_bgr_uint8",
    "resize_frame",
    "PanEchoClipTransform",
    "build_clips",
]
