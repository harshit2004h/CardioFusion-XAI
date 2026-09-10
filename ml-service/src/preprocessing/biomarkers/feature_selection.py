"""Leakage-safe feature selection for biomarker models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)


class BiomarkerFeatureSelector:
    """Remove zero-variance features and optionally select top-K features.

    The selector is supervised, so it MUST be fitted using training X and y
    only. For the project's masked multi-task model, selection should be
    performed per task or disabled for the shared feature space.
    """

    def __init__(
        self,
        *,
        k: int | str = "all",
        task: str = "classification",
        score_method: str = "mutual_info",
        variance_threshold: float = 0.0,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression.")
        if score_method not in {"mutual_info", "f"}:
            raise ValueError("score_method must be mutual_info or f.")

        self.k = k
        self.task = task
        self.score_method = score_method
        self.variance_threshold = variance_threshold

        self.variance = VarianceThreshold(variance_threshold)
        self.selector: SelectKBest | None = None

        self.input_columns_: list[str] = []
        self.variance_columns_: list[str] = []
        self.selected_columns_: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y) -> "BiomarkerFeatureSelector":
        if X.empty:
            raise ValueError("X is empty.")

        self.input_columns_ = list(X.columns)

        self.variance.fit(X)
        variance_mask = self.variance.get_support()

        self.variance_columns_ = [
            column
            for column, keep in zip(
                self.input_columns_,
                variance_mask,
            )
            if keep
        ]

        if not self.variance_columns_:
            raise ValueError(
                "All features were removed by VarianceThreshold."
            )

        if self.task == "classification":
            score_func = (
                mutual_info_classif
                if self.score_method == "mutual_info"
                else f_classif
            )
        else:
            score_func = (
                mutual_info_regression
                if self.score_method == "mutual_info"
                else f_regression
            )

        self.selector = SelectKBest(
            score_func=score_func,
            k=self.k,
        )
        self.selector.fit(X[self.variance_columns_], y)

        mask = self.selector.get_support()
        self.selected_columns_ = [
            column
            for column, keep in zip(
                self.variance_columns_,
                mask,
            )
            if keep
        ]

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.selector is None:
            raise RuntimeError("Call fit() before transform().")

        missing = [
            column
            for column in self.input_columns_
            if column not in X.columns
        ]
        if missing:
            raise ValueError(f"Missing columns during transform: {missing}")

        values = self.selector.transform(
            X[self.variance_columns_]
        )

        return pd.DataFrame(
            values.astype(np.float32),
            columns=self.selected_columns_,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
