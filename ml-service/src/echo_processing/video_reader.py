from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_video(
    path: Path,
    *,
    frames: int = 32,
    stride: int = 4,
    size: int = 112,
    format_label: str | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not decode the supplied echo video.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        capture.release()
        raise ValueError("Echo video contains no readable frames.")
    indices = np.linspace(0, total - 1, frames, dtype=np.int64) if total < frames * stride else np.arange(frames) * stride
    sampled = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        sampled.append(np.repeat(gray[..., None], 3, axis=2))
    capture.release()
    if len(sampled) < frames:
        raise ValueError(f"Echo video yielded {len(sampled)} frames; {frames} are required.")
    array = np.asarray(sampled, dtype=np.float32) / 255.0
    return array, {"format": format_label or path.suffix.lstrip(".").upper(), "frames_used": frames, "fps": fps}