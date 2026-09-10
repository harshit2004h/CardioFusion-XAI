"""PTB-XL ECG resampling.

The project stores processed ECGs at 100 Hz. PTB-XL records can be supplied
at their source sampling rate and are resampled with scipy.signal.resample_poly.
Internal tensor convention is (12, samples).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def ensure_leads_first(
    signal: np.ndarray,
    *,
    n_leads: int = 12,
    leads_first: bool | None = None,
) -> np.ndarray:
    """Return ECG as (leads, samples)."""
    x = np.asarray(signal, dtype=np.float32)

    if x.ndim == 1:
        return x[None, :]

    if x.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D ECG, got shape {x.shape}."
        )

    if leads_first is True:
        if x.shape[0] != n_leads:
            raise ValueError(
                f"leads_first=True but first dimension is {x.shape[0]}."
            )
        return x

    if leads_first is False:
        if x.shape[1] != n_leads:
            raise ValueError(
                f"leads_first=False but second dimension is {x.shape[1]}."
            )
        return x.T

    if x.shape[0] == n_leads:
        return x

    if x.shape[1] == n_leads:
        return x.T

    raise ValueError(
        f"Cannot infer lead orientation from {x.shape}. "
        f"Expected ({n_leads}, samples) or (samples, {n_leads})."
    )


def resample_ecg(
    signal: np.ndarray,
    original_fs: float,
    target_fs: float = 100.0,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Polyphase-resample an ECG to target_fs."""
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling frequencies must be positive.")

    x = np.asarray(signal, dtype=np.float32)
    if x.size == 0:
        raise ValueError("ECG signal is empty.")

    ratio = Fraction(
        float(target_fs) / float(original_fs)
    ).limit_denominator(1000)

    y = resample_poly(
        x,
        ratio.numerator,
        ratio.denominator,
        axis=axis,
    )

    return np.asarray(y, dtype=np.float32)


def resample_to_project_rate(
    signal: np.ndarray,
    original_fs: float,
    *,
    target_fs: float = 100.0,
) -> np.ndarray:
    """Convenience wrapper used by the PTB-XL preprocessing pipeline."""
    return resample_ecg(
        signal,
        original_fs,
        target_fs,
        axis=-1,
    )
