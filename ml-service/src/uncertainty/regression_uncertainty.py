from __future__ import annotations

import math

import numpy as np


def fit_absolute_residual_quantile(y_true, y_pred, alpha: float = 0.10) -> dict[str, float | int | str]:
    """Fit a split-conformal absolute-residual quantile on a held-out calibration set."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have equal lengths.")
    valid = np.isfinite(y) & np.isfinite(p)
    residuals = np.abs(y[valid] - p[valid])
    if residuals.size < 20:
        raise ValueError("At least 20 finite calibration residuals are required.")
    level = min(1.0, math.ceil((residuals.size + 1) * (1.0 - alpha)) / residuals.size)
    quantile = float(np.quantile(residuals, level, method="higher"))
    return {"method": "split_conformal_absolute_residual_quantile", "alpha": float(alpha), "nominal_coverage": float(1.0 - alpha), "n_calibration": int(residuals.size), "quantile_level": float(level), "absolute_error_quantile": quantile}


def conformal_interval_from_quantile(prediction: float, quantile: float, lower_bound: float | None = None, upper_bound: float | None = None) -> tuple[float, float]:
    values = np.asarray([prediction, quantile], dtype=np.float64)
    if not np.isfinite(values).all() or quantile < 0:
        raise ValueError("prediction must be finite and quantile must be finite/non-negative.")
    lower, upper = float(prediction - quantile), float(prediction + quantile)
    if lower_bound is not None:
        lower = max(float(lower_bound), lower)
    if upper_bound is not None:
        upper = min(float(upper_bound), upper)
    return lower, upper
