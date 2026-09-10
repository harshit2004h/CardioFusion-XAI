"""Leakage-safe missing-value imputation for biomarker data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class BiomarkerImputer:
    """Separate numeric/categorical imputers.

    Fit ONLY on the training split. Reuse the fitted object for validation,
    test and inference. MI-Complications should be inspected for extreme
    missingness before this class is applied.
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        categorical_fill_value: str = "missing",
    ) -> None:
        if numeric_strategy not in {"mean", "median", "most_frequent"}:
            raise ValueError(
                "numeric_strategy must be mean, median or most_frequent."
            )
        if categorical_strategy not in {"most_frequent", "constant"}:
            raise ValueError(
                "categorical_strategy must be most_frequent or constant."
            )

        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value

        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self._fitted = False

        self.numeric_imputer = SimpleImputer(
            strategy=numeric_strategy,
            keep_empty_features=True,
        )
        self.categorical_imputer = SimpleImputer(
            strategy=categorical_strategy,
            fill_value=categorical_fill_value,
            keep_empty_features=True,
        )

    def fit(self, df: pd.DataFrame) -> "BiomarkerImputer":
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        self.numeric_columns_ = list(
            df.select_dtypes(include=[np.number]).columns
        )
        self.categorical_columns_ = [
            column for column in df.columns
            if column not in self.numeric_columns_
        ]

        if self.numeric_columns_:
            self.numeric_imputer.fit(df[self.numeric_columns_])

        if self.categorical_columns_:
            self.categorical_imputer.fit(df[self.categorical_columns_])

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        expected = self.numeric_columns_ + self.categorical_columns_
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns during transform: {missing}")

        pieces = []

        if self.numeric_columns_:
            values = self.numeric_imputer.transform(df[self.numeric_columns_])
            pieces.append(
                pd.DataFrame(
                    values,
                    columns=self.numeric_columns_,
                    index=df.index,
                )
            )

        if self.categorical_columns_:
            values = self.categorical_imputer.transform(
                df[self.categorical_columns_]
            )
            pieces.append(
                pd.DataFrame(
                    values,
                    columns=self.categorical_columns_,
                    index=df.index,
                )
            )

        if not pieces:
            return pd.DataFrame(index=df.index)

        return pd.concat(pieces, axis=1)[expected]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def fit_source_imputer(
    train_df: pd.DataFrame,
    transform_dfs: Iterable[pd.DataFrame] = (),
) -> tuple[pd.DataFrame, list[pd.DataFrame], BiomarkerImputer]:
    """Convenience function for one source dataset at a time."""
    imputer = BiomarkerImputer().fit(train_df)
    train = imputer.transform(train_df)
    others = [imputer.transform(df) for df in transform_dfs]
    return train, others, imputer
