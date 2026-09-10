"""Training-only ECG augmentation.

Never call augment_ecg() for validation, test, calibration or deterministic
user inference. The project uses augmentation only to improve training
robustness of the PTB-XL ECG branch.
"""

from __future__ import annotations

import numpy as np


def _generator(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def add_gaussian_noise(
    signal: np.ndarray,
    *,
    std: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    if std < 0:
        raise ValueError("std must be non-negative.")

    x = np.asarray(signal, dtype=np.float32)
    noise = _generator(seed).normal(
        0.0,
        std,
        size=x.shape,
    ).astype(np.float32)

    return x + noise


def add_baseline_wander(
    signal: np.ndarray,
    *,
    amplitude: float = 0.02,
    frequency_hz: float = 0.30,
    fs: float = 100.0,
    seed: int | None = None,
) -> np.ndarray:
    """Add a low-frequency baseline-wander component."""
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative.")
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    x = np.asarray(signal, dtype=np.float32)

    n_samples = x.shape[-1]
    time = np.arange(
        n_samples,
        dtype=np.float32,
    ) / fs

    phase = _generator(seed).uniform(
        0.0,
        2.0 * np.pi,
    )

    wander = (
        amplitude
        * np.sin(
            2.0 * np.pi * frequency_hz * time + phase
        )
    ).astype(np.float32)

    return x + wander


def random_time_shift(
    signal: np.ndarray,
    *,
    max_fraction: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Circularly shift an ECG by a small number of samples."""
    if not 0 <= max_fraction < 0.5:
        raise ValueError(
            "max_fraction must be in [0, 0.5)."
        )

    x = np.asarray(signal, dtype=np.float32)

    max_shift = int(
        round(x.shape[-1] * max_fraction)
    )

    shift = int(
        _generator(seed).integers(
            -max_shift,
            max_shift + 1,
        )
    )

    return np.roll(
        x,
        shift=shift,
        axis=-1,
    )


def random_amplitude_scale(
    signal: np.ndarray,
    *,
    low: float = 0.95,
    high: float = 1.05,
    seed: int | None = None,
) -> np.ndarray:
    if low <= 0 or high < low:
        raise ValueError(
            "Require 0 < low <= high."
        )

    scale = float(
        _generator(seed).uniform(low, high)
    )

    return (
        np.asarray(signal, dtype=np.float32) * scale
    ).astype(np.float32)


def random_lead_dropout(
    signal: np.ndarray,
    *,
    probability: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Randomly zero complete leads for robustness training."""
    if not 0 <= probability <= 1:
        raise ValueError(
            "probability must be in [0, 1]."
        )

    x = np.asarray(signal, dtype=np.float32).copy()

    if x.ndim != 2:
        raise ValueError(
            "Lead dropout requires shape (leads, samples)."
        )

    rng = _generator(seed)
    mask = rng.random(x.shape[0]) < probability
    x[mask] = 0.0

    return x


def augment_ecg(
    signal: np.ndarray,
    *,
    fs: float = 100.0,
    noise_std: float = 0.01,
    baseline_amplitude: float = 0.02,
    baseline_frequency_hz: float = 0.30,
    shift_fraction: float = 0.02,
    amplitude_range: tuple[float, float] = (0.95, 1.05),
    lead_dropout_probability: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply mild ECG augmentations in a reproducible sequence."""
    rng = _generator(seed)
    seeds = rng.integers(
        0,
        2**32 - 1,
        size=5,
        dtype=np.uint64,
    )

    out = add_gaussian_noise(
        signal,
        std=noise_std,
        seed=int(seeds[0]),
    )

    out = add_baseline_wander(
        out,
        amplitude=baseline_amplitude,
        frequency_hz=baseline_frequency_hz,
        fs=fs,
        seed=int(seeds[1]),
    )

    out = random_time_shift(
        out,
        max_fraction=shift_fraction,
        seed=int(seeds[2]),
    )

    out = random_amplitude_scale(
        out,
        low=amplitude_range[0],
        high=amplitude_range[1],
        seed=int(seeds[3]),
    )

    if lead_dropout_probability > 0:
        out = random_lead_dropout(
            out,
            probability=lead_dropout_probability,
            seed=int(seeds[4]),
        )

    return np.asarray(out, dtype=np.float32)
