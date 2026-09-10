from __future__ import annotations

import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """
    Binary Platt scaling.

    Input:
        raw model probabilities in [0, 1]

    Output:
        calibrated probabilities in [0, 1]
    """

    def __init__(
        self,
        regularization: float = 1.0,
    ):
        self.regularization = regularization
        self.model = LogisticRegression(
            C=regularization,
            solver="lbfgs",
        )

    @staticmethod
    def _clip(
        probabilities,
        eps=1e-6,
    ):
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        return np.clip(
            probabilities,
            eps,
            1.0 - eps,
        )

    @staticmethod
    def _logit(
        probabilities,
    ):
        probabilities = PlattCalibrator._clip(
            probabilities
        )

        return np.log(
            probabilities
            / (1.0 - probabilities)
        )

    def fit(
        self,
        probabilities,
        y_true,
    ):
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        y_true = np.asarray(
            y_true,
            dtype=np.int32,
        )

        valid = (
            np.isfinite(probabilities)
            & np.isfinite(y_true)
        )

        probabilities = probabilities[valid]
        y_true = y_true[valid]

        if len(probabilities) < 10:
            raise ValueError(
                "Not enough samples for calibration."
            )

        if len(np.unique(y_true)) < 2:
            raise ValueError(
                "Calibration requires both "
                "positive and negative classes."
            )

        logits = self._logit(
            probabilities
        ).reshape(-1, 1)

        self.model.fit(
            logits,
            y_true,
        )

        return self

    def predict_proba(
        self,
        probabilities,
    ):
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        logits = self._logit(
            probabilities
        ).reshape(-1, 1)

        return self.model.predict_proba(
            logits
        )[:, 1]

    def save(
        self,
        path,
    ):
        with open(
            path,
            "wb",
        ) as f:
            pickle.dump(
                self,
                f,
            )

    @classmethod
    def load(
        cls,
        path,
    ):
        with open(
            path,
            "rb",
        ) as f:
            return pickle.load(f)