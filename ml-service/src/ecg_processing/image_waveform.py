from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ECGImageError(ValueError):
    """The ECG image cannot be converted into a trustworthy waveform."""


def _trace_panel(panel: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) if panel.ndim == 3 else panel
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = gray < 100
    height, width = mask.shape
    margin_y = max(2, int(height * 0.08))
    mask[:margin_y] = False
    mask[-margin_y:] = False
    mask[:, :2] = False
    mask[:, -2:] = False

    trace = np.full(width, np.nan, dtype=np.float32)
    previous = height / 2.0
    for x in range(width):
        candidates = np.flatnonzero(mask[:, x])
        if candidates.size == 0:
            continue
        candidate = float(candidates[np.argmin(np.abs(candidates - previous))])
        if abs(candidate - previous) <= height * 0.35 or x == 0:
            trace[x] = candidate
            previous = candidate

    valid = np.isfinite(trace)
    coverage = float(valid.mean())
    if coverage < 0.20:
        raise ECGImageError("ECG lead trace coverage is too low.")
    positions = np.arange(width)
    trace = np.interp(positions, positions[valid], trace[valid])
    trace = cv2.GaussianBlur(trace.reshape(1, -1), (1, 5), 0).reshape(-1)
    centered = trace - np.median(trace)
    amplitude = np.percentile(np.abs(centered), 99)
    if amplitude < 1.0:
        raise ECGImageError("ECG lead trace has no measurable waveform variation.")
    return centered.astype(np.float32), coverage


def digitize_ecg_image(
    path: Path,
    *,
    target_samples: int = 1000,
    target_rate: int = 100,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ECGImageError("OpenCV could not decode the ECG image.")
    height, width = image.shape[:2]
    if width < 400 or height < 300:
        raise ECGImageError("ECG image is too small for 12-lead digitization.")

    # Most printed 12-lead ECGs use three rows and four columns of lead panels.
    row_edges = np.linspace(0, height, 4, dtype=int)
    col_edges = np.linspace(0, width, 5, dtype=int)
    leads: list[np.ndarray] = []
    coverages: list[float] = []
    for row in range(3):
        for col in range(4):
            top, bottom = row_edges[row], row_edges[row + 1]
            left, right = col_edges[col], col_edges[col + 1]
            panel = image[top:bottom, left:right]
            trace, coverage = _trace_panel(panel)
            source_time = np.linspace(0.0, 1.0, trace.size, endpoint=False)
            target_time = np.arange(target_samples, dtype=np.float32) / target_samples
            leads.append(np.interp(target_time, source_time, trace).astype(np.float32))
            coverages.append(coverage)

    waveform = np.stack(leads, axis=0)
    quality_score = float(np.clip(np.mean(coverages), 0.0, 1.0))
    if quality_score < 0.35:
        raise ECGImageError("ECG waveform quality is insufficient for model inference.")
    return waveform, {
        "waveform_available": True,
        "leads": 12,
        "duration_seconds": 10.0,
        "sampling_rate": target_rate,
        "quality_score": quality_score,
        "digitization": "opencv_3x4_panel_trace",
    }