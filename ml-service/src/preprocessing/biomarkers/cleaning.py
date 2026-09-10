"""Conservative cleaning for Zheen, UCI-HF, MI-Complications and Framingham.

Important project rule:
    Cleaning is source-aware but does not perform imputation, encoding,
    scaling, feature selection, or target construction.

The Hugging Face heart-failure-prediction dataset is intentionally not
included: the project design uses four biomarker sources:
Zheen, UCI Heart Failure, MI-Complications and Framingham.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "none", "null",
    "missing", "unknown", "not available", "not_applicable",
    "not applicable", "?", "--",
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to stable snake_case names."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out = df.copy()
    new_columns = []
    for column in out.columns:
        name = str(column).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        new_columns.append(name)

    out.columns = new_columns
    return out


def replace_missing_tokens(
    df: pd.DataFrame,
    missing_tokens: Iterable[str] = DEFAULT_MISSING_TOKENS,
) -> pd.DataFrame:
    """Replace textual missing-value tokens with NaN."""
    out = df.copy()
    tokens = {str(token).strip().lower() for token in missing_tokens}

    for column in out.columns:
        if (
            pd.api.types.is_object_dtype(out[column])
            or pd.api.types.is_string_dtype(out[column])
            or pd.api.types.is_categorical_dtype(out[column])
        ):
            normalized = out[column].astype("string").str.strip().str.lower()
            out.loc[normalized.isin(tokens), column] = np.nan

    return out


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    min_conversion_ratio: float = 0.90,
) -> pd.DataFrame:
    """Convert columns to numeric only when conversion is sufficiently safe."""
    if not 0 < min_conversion_ratio <= 1:
        raise ValueError("min_conversion_ratio must be in (0, 1].")

    out = df.copy()
    candidates = list(columns) if columns is not None else list(out.columns)

    for column in candidates:
        if column not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            continue

        converted = pd.to_numeric(out[column], errors="coerce")
        original_non_null = out[column].notna().sum()

        if original_non_null == 0:
            continue

        ratio = converted.notna().sum() / original_non_null
        if ratio >= min_conversion_ratio:
            out[column] = converted

    return out


def clean_biomarkers(
    df: pd.DataFrame,
    *,
    numeric_columns: Iterable[str] | None = None,
    missing_tokens: Iterable[str] = DEFAULT_MISSING_TOKENS,
    drop_duplicate_rows: bool = True,
    drop_all_null_columns: bool = True,
) -> pd.DataFrame:
    """Apply safe, dataset-independent cleaning.

    No target column is inferred or removed here. Target/leakage handling is
    deliberately delegated to leakage_guard.py.
    """
    out = standardize_column_names(df)
    out = replace_missing_tokens(out, missing_tokens)
    out = coerce_numeric_columns(out, numeric_columns)

    out = out.replace([np.inf, -np.inf], np.nan)

    if drop_duplicate_rows:
        out = out.drop_duplicates().reset_index(drop=True)

    if drop_all_null_columns:
        out = out.dropna(axis=1, how="all")

    return out


def apply_explicit_ranges(
    df: pd.DataFrame,
    ranges: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Apply explicitly supplied domain ranges.

    This function never invents medical reference ranges. The caller must
    provide the limits appropriate to the source and measurement.
    """
    out = df.copy()

    for column, bounds in ranges.items():
        if column not in out.columns:
            continue

        lower, upper = bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid range for {column}: {bounds}")

        out[column] = pd.to_numeric(out[column], errors="coerce")
        out[column] = out[column].clip(lower=lower, upper=upper)

    return out
