"""Training-fitted scaling for biomarker features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class BiomarkerScaler:
    """StandardScaler or RobustScaler fitted only on training data."""

    def __init__(self, method: str = "standard") -> None:
        if method not in {"standard", "robust"}:
            raise ValueError("method must be 'standard' or 'robust'.")

        self.method = method
        self.scaler = (
            StandardScaler()
            if method == "standard"
            else RobustScaler()
        )
        self.columns_: list[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "BiomarkerScaler":
        if df.empty:
            raise ValueError("Cannot fit scaler on an empty DataFrame.")

        self.columns_ = list(df.columns)
        self.scaler.fit(df[self.columns_].astype(np.float32))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        missing = [c for c in self.columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns during transform: {missing}")

        values = self.scaler.transform(
            df[self.columns_].astype(np.float32)
        )

        return pd.DataFrame(
            values.astype(np.float32),
            columns=self.columns_,
            index=df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def scale_features(
    train_df: pd.DataFrame,
    transform_dfs: list[pd.DataFrame] | None = None,
    *,
    method: str = "standard",
) -> tuple[pd.DataFrame, list[pd.DataFrame], BiomarkerScaler]:
    scaler = BiomarkerScaler(method=method).fit(train_df)
    train = scaler.transform(train_df)
    others = [
        scaler.transform(df)
        for df in (transform_dfs or [])
    ]
    return train, others, scaler
