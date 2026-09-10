"""Normalization of processed ECG waveforms."""

from __future__ import annotations

import numpy as np


def zscore_normalize(
    signal: np.ndarray,
    *,
    axis: int = -1,
    eps: float = 1e-8,
    clip: float | None = 5.0,
) -> np.ndarray:
    """Normalize each lead independently along the sample axis."""
    x = np.asarray(signal, dtype=np.float32)

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)

    y = (x - mean) / np.maximum(std, eps)

    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive.")
        y = np.clip(y, -clip, clip)

    return y.astype(np.float32)


def robust_normalize(
    signal: np.ndarray,
    *,
    axis: int = -1,
    eps: float = 1e-8,
    clip: float | None = 8.0,
) -> np.ndarray:
    """Median/IQR normalization, useful when a waveform contains outliers."""
    x = np.asarray(signal, dtype=np.float32)

    median = np.median(x, axis=axis, keepdims=True)
    q25 = np.percentile(x, 25, axis=axis, keepdims=True)
    q75 = np.percentile(x, 75, axis=axis, keepdims=True)
    iqr = np.maximum(q75 - q25, eps)

    y = (x - median) / iqr

    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive.")
        y = np.clip(y, -clip, clip)

    return y.astype(np.float32)


def normalize_ecg(
    signal: np.ndarray,
    *,
    method: str = "zscore",
    axis: int = -1,
    clip: float | None = 5.0,
) -> np.ndarray:
    """Select the ECG normalization method."""
    if method == "zscore":
        return zscore_normalize(
            signal,
            axis=axis,
            clip=clip,
        )

    if method == "robust":
        return robust_normalize(
            signal,
            axis=axis,
            clip=clip,
        )

    raise ValueError(
        "method must be 'zscore' or 'robust'."
    )
