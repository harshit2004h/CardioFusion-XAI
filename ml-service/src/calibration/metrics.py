from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.size == 0 or y_true.size != probabilities.size:
        raise ValueError("y_true and probabilities must be non-empty and equal length.")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        upper_inclusive = index == n_bins - 1
        mask = (probabilities >= bins[index]) & (
            probabilities <= bins[index + 1] if upper_inclusive else probabilities < bins[index + 1]
        )
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(error)


def calibration_metrics(y_true, raw_probabilities, calibrated_probabilities):
    return {
        "raw_brier": float(brier_score_loss(y_true, raw_probabilities)),
        "calibrated_brier": float(brier_score_loss(y_true, calibrated_probabilities)),
        "raw_log_loss": float(log_loss(y_true, raw_probabilities)),
        "calibrated_log_loss": float(log_loss(y_true, calibrated_probabilities)),
        "raw_ece": expected_calibration_error(y_true, raw_probabilities),
        "calibrated_ece": expected_calibration_error(y_true, calibrated_probabilities),
    }