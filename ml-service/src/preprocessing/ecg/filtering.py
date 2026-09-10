"""ECG signal filtering for PTB-XL."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)

    if x.ndim not in {1, 2}:
        raise ValueError("ECG must be 1D or 2D.")

    if x.shape[-1] < 32:
        raise ValueError("ECG is too short for stable zero-phase filtering.")

    if not np.isfinite(x).all():
        raise ValueError(
            "ECG contains NaN/Inf. Clean or interpolate it before filtering."
        )

    return x


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    *,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    x = _validate_signal(signal)

    if fs <= 0:
        raise ValueError("fs must be positive.")

    nyquist = fs / 2.0

    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            f"Require 0 < lowcut < highcut < Nyquist ({nyquist:.2f} Hz)."
        )

    b, a = butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype="band",
    )

    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def notch_filter(
    signal: np.ndarray,
    fs: float,
    *,
    frequency: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Remove power-line interference; 50 Hz is the default for this project."""
    x = _validate_signal(signal)

    if fs <= 0:
        raise ValueError("fs must be positive.")

    if frequency <= 0 or frequency >= fs / 2:
        raise ValueError("Notch frequency must be below Nyquist.")

    if quality_factor <= 0:
        raise ValueError("quality_factor must be positive.")

    b, a = iirnotch(
        w0=frequency,
        Q=quality_factor,
        fs=fs,
    )

    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def filter_ecg(
    signal: np.ndarray,
    fs: float,
    *,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    notch_frequency: float | None = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Apply notch filtering followed by band-pass filtering."""
    out = np.asarray(signal, dtype=np.float32)

    if notch_frequency is not None:
        out = notch_filter(
            out,
            fs,
            frequency=notch_frequency,
        )

    return bandpass_filter(
        out,
        fs,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )
