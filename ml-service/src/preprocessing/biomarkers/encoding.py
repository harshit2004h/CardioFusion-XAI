"""Categorical encoding for biomarker/clinical features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class BiomarkerEncoder:
    """One-hot encode categorical features while preserving numeric features.

    handle_unknown='ignore' is required because a user-uploaded report may
    contain a category not seen during training.
    """

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "BiomarkerEncoder":
        self.categorical_columns_ = list(
            df.select_dtypes(
                include=["object", "string", "category", "bool"]
            ).columns
        )
        self.numeric_columns_ = [
            c for c in df.columns
            if c not in self.categorical_columns_
        ]

        if self.categorical_columns_:
            self.encoder.fit(df[self.categorical_columns_])
            encoded_names = list(
                self.encoder.get_feature_names_out(
                    self.categorical_columns_
                )
            )
        else:
            encoded_names = []

        self.feature_names_ = self.numeric_columns_ + encoded_names
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        expected = self.numeric_columns_ + self.categorical_columns_
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns during transform: {missing}")

        numeric = df[self.numeric_columns_].to_numpy(dtype=np.float32)

        if self.categorical_columns_:
            encoded = self.encoder.transform(
                df[self.categorical_columns_]
            ).astype(np.float32)

            values = np.hstack([numeric, encoded])
        else:
            values = numeric

        return pd.DataFrame(
            values,
            columns=self.feature_names_,
            index=df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
