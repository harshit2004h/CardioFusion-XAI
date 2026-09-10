from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins: int = 10,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (probabilities >= bins[i])
                & (probabilities <= bins[i + 1])
            )
        else:
            mask = (
                (probabilities >= bins[i])
                & (probabilities < bins[i + 1])
            )

        if not np.any(mask):
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()

        ece += (
            np.sum(mask)
            / len(y_true)
        ) * abs(
            confidence - accuracy
        )

    return float(ece)


def calibration_metrics(
    y_true,
    raw_probabilities,
    calibrated_probabilities,
):
    return {
        "raw_brier": float(
            brier_score_loss(
                y_true,
                raw_probabilities,
            )
        ),
        "calibrated_brier": float(
            brier_score_loss(
                y_true,
                calibrated_probabilities,
            )
        ),
        "raw_log_loss": float(
            log_loss(
                y_true,
                raw_probabilities,
            )
        ),
        "calibrated_log_loss": float(
            log_loss(
                y_true,
                calibrated_probabilities,
            )
        ),
        "raw_ece": expected_calibration_error(
            y_true,
            raw_probabilities,
        ),
        "calibrated_ece": expected_calibration_error(
            y_true,
            calibrated_probabilities,
        ),
    }