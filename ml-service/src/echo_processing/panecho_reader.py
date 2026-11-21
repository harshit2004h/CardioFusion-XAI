"""Decode echo uploads (AVI / MP4 / multi-frame DICOM) into PanEcho input clips.

The existing ``video_reader.sample_video`` produces the R(2+1)D contract
(32 frames, stride 4, 112x112, [0,1] floats). PanEcho needs a different
contract (16 consecutive frames, 224x224, ImageNet-normalised), so it has its
own reader instead of re-using that output.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.echo.panecho import PANECHO_CLIP_LEN, PANECHO_RESIZE_TO
from src.preprocessing.echo.panecho_transforms import build_clips, resize_frame

# ~512 frames x 256 x 256 x 3 bytes = ~100 MB worst case per request.
DEFAULT_MAX_FRAMES = 512


def decode_video(
    path: str | Path,
    keep: Iterable[int] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resize_to: int = PANECHO_RESIZE_TO,
) -> tuple[dict[int, np.ndarray] | np.ndarray, dict[str, float | int]]:
    """Sequentially decode a video with OpenCV (no seeking; seeking is unreliable for AVI/MP4).

    * ``keep=None``  -> returns an array [N, resize_to, resize_to, 3] (uint8, BGR) of up to
      ``max_frames`` frames.
    * ``keep=[...]`` -> returns ``{index: frame}`` for just those indices; decoding stops after
      the largest requested index, so only the needed frames are resized.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not decode the supplied echo video.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    wanted = sorted(set(int(i) for i in keep)) if keep is not None else None
    last_wanted = wanted[-1] if wanted else None
    wanted_set = set(wanted) if wanted is not None else None

    collected: list[np.ndarray] = []
    selected: dict[int, np.ndarray] = {}
    index = 0
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if wanted_set is None:
                collected.append(resize_frame(frame, resize_to))
                if len(collected) >= max_frames:
                    break
            else:
                if index in wanted_set:
                    selected[index] = resize_frame(frame, resize_to)
                if index >= last_wanted:
                    break
            index += 1
    finally:
        capture.release()

    metadata = {
        "fps": fps,
        "reported_frames": reported,
        "decoded_frames": decoded,
        "source_width": width,
        "source_height": height,
    }
    if wanted_set is None:
        if not collected:
            raise ValueError("Echo video contains no readable frames.")
        return np.stack(collected, axis=0), metadata
    if not selected:
        raise ValueError("Echo video contains no readable frames.")
    return selected, metadata


def decode_dicom(path: str | Path, max_frames: int = DEFAULT_MAX_FRAMES, resize_to: int = PANECHO_RESIZE_TO) -> tuple[np.ndarray, dict]:
    """Multi-frame ultrasound DICOM -> [N, 256, 256, 3] uint8 BGR (same layout as ``decode_video``)."""
    from src.ingestion.dicom import DicomProcessingError, read_dataset

    dataset = read_dataset(Path(path))
    try:
        pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:  # pragma: no cover - depends on codec plugins
        raise DicomProcessingError(f"Could not decode DICOM Echo pixels: {exc}") from exc

    frames_attr = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    photometric = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))
    if pixels.ndim == 2:  # single greyscale frame
        pixels = pixels[None, ...]
    elif pixels.ndim == 3 and frames_attr == 1 and pixels.shape[-1] in (3, 4):  # single colour frame
        pixels = pixels[None, ...]
    if pixels.ndim not in (3, 4) or pixels.shape[0] < 1:
        raise DicomProcessingError("DICOM Echo does not contain a readable frame sequence.")
    if photometric == "MONOCHROME1":  # inverted greyscale
        pixels = pixels.max() - pixels

    # Scale the whole cine loop together (per-frame scaling would flicker).
    pixels = pixels[:max_frames]
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.float32)
        lo, hi = float(pixels.min()), float(pixels.max())
        pixels = ((pixels - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(pixels)).astype(np.uint8)

    frames = []
    for frame in pixels:
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # pydicom gives RGB; PanEcho saw BGR
        frames.append(resize_frame(frame, resize_to))

    fps = 0.0
    for attr in ("RecommendedDisplayFrameRate", "CineRate"):
        if getattr(dataset, attr, None):
            fps = float(getattr(dataset, attr))
            break
    else:
        frame_time = getattr(dataset, "FrameTime", None)  # milliseconds
        if frame_time:
            fps = 1000.0 / float(frame_time)
    return np.stack(frames, axis=0), {
        "fps": fps,
        "reported_frames": frames_attr,
        "decoded_frames": len(frames),
        "source_width": int(getattr(dataset, "Columns", 0) or 0),
        "source_height": int(getattr(dataset, "Rows", 0) or 0),
    }


def read_echo_for_panecho(
    path: str | Path,
    file_format: str | None = None,
    clip_len: int = PANECHO_CLIP_LEN,
    num_clips: int = 4,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Upload on disk -> ([num_clips, 3, clip_len, 224, 224] tensor, metadata)."""
    path = Path(path)
    file_format = (file_format or path.suffix.lstrip(".")).lower()
    if file_format in {"dcm", "dicom"}:
        frames, metadata = decode_dicom(path, max_frames=max_frames)
        label = "DICOM"
    else:
        frames, metadata = decode_video(path, max_frames=max_frames)
        label = file_format.upper()
    clips = build_clips(frames, clip_len=clip_len, num_clips=num_clips, train=False)
    metadata = {
        **metadata,
        "format": label,
        "clip_len": clip_len,
        "num_clips": num_clips,
        "frames_used": int(min(frames.shape[0], clip_len * num_clips)),
        "short_video_padded": bool(frames.shape[0] < clip_len),
    }
    return clips, metadata


__all__ = ["decode_video", "decode_dicom", "read_echo_for_panecho", "DEFAULT_MAX_FRAMES"]
