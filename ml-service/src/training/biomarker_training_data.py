"""Load and canonicalize the four independent clinical/biomarker datasets.

All transformations that estimate distributional parameters (median imputation
and scaling) are fitted later on the training split only. This module performs
column mapping, target identification and deterministic feature selection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BiomarkerSourceData:
    source: str
    features: pd.DataFrame
    targets: dict[str, pd.Series]
    target_specs: dict[str, dict[str, Any]]
    class_values: dict[str, list[Any]]


def normalise_column(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [normalise_column(name) for name in result.columns]
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Column normalization produced duplicate columns: {duplicates}")
    return result


def _find_csv(root: Path, source: str, preferred_names: tuple[str, ...]) -> Path:
    source_dir = root / source
    for name in preferred_names:
        candidate = source_dir / name
        if candidate.is_file():
            return candidate
    matches = sorted(source_dir.glob("*.csv")) if source_dir.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No CSV was found for {source!r}. Expected one of {preferred_names} under {source_dir}."
        )
    raise FileNotFoundError(
        f"Multiple CSV files were found for {source!r} under {source_dir}; "
        f"specify the canonical filename. Candidates: {[path.name for path in matches]}"
    )


def _numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        mapped = series.astype(str).str.strip().str.lower().map({
            "positive": 1, "yes": 1, "true": 1, "1": 1,
            "negative": 0, "no": 0, "false": 0, "0": 0,
        })
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.where(numeric.notna(), mapped)
    return pd.to_numeric(series, errors="coerce")


def _binary_target(series: pd.Series, name: str) -> pd.Series:
    values = _numeric_series(series)
    unique = set(values.dropna().unique().tolist())
    if not unique or not unique.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target {name!r} must be binary 0/1; observed values: {sorted(map(str, unique))[:20]}")
    return values.astype(float)


def _canonicalize_zheen(frame: pd.DataFrame) -> BiomarkerSourceData:
    df = normalise_frame(frame)
    aliases = {
        "age": ("age",), "gender": ("gender", "sex"),
        "heart_rate": ("heart_rate", "heartrate", "impluse", "pulse"),
        "systolic_bp": ("systolic_blood_pressure", "systolic_bp", "pressurehight", "sysbp"),
        "diastolic_bp": ("diastolic_blood_pressure", "diastolic_bp", "pressurelow", "diabp"),
        "blood_glucose": ("blood_sugar", "blood_glucose", "glucose", "sugar"),
        "ck_mb": ("ck_mb", "ckmb", "kcm"), "troponin": ("troponin", "troponin_i"),
    }
    target_col = next((name for name in ("result", "target", "heart_attack") if name in df.columns), None)
    if target_col is None:
        raise ValueError("Zheen CSV must contain one of: Result, target, heart_attack.")
    mapped: dict[str, pd.Series] = {}
    for canonical, candidates in aliases.items():
        col = next((name for name in candidates if name in df.columns), None)
        if col is None:
            raise ValueError(f"Zheen CSV is missing required feature {canonical!r}; accepted columns: {candidates}.")
        mapped[canonical] = df[col]
    # Gender may be encoded as Male/Female or 0/1.
    gender = mapped["gender"].astype(str).str.strip().str.lower().map({"male": 1, "m": 1, "female": 0, "f": 0})
    raw_gender = _numeric_series(mapped["gender"])
    mapped["gender"] = raw_gender.where(raw_gender.notna(), gender)
    features = pd.DataFrame(mapped).apply(pd.to_numeric, errors="coerce")
    if features["heart_rate"].quantile(0.99) > 300:
        features["heart_rate"] = features["heart_rate"].clip(upper=300)
    target = _binary_target(df[target_col], "acute_mi")
    return BiomarkerSourceData("zheen", features, {"acute_mi": target}, {"acute_mi": {"type": "binary"}}, {})


def _canonicalize_uci(frame: pd.DataFrame) -> BiomarkerSourceData:
    df = normalise_frame(frame)
    target_col = "death_event"
    if target_col not in df.columns:
        raise ValueError("UCI Heart Failure CSV must contain DEATH_EVENT.")
    feature_names = [
        "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
        "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking",
    ]
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"UCI Heart Failure CSV is missing required features: {missing}")
    features = df[feature_names].apply(pd.to_numeric, errors="coerce")
    target = _binary_target(df[target_col], "hf_mortality")
    return BiomarkerSourceData("uci_heart_failure", features, {"hf_mortality": target}, {"hf_mortality": {"type": "binary"}}, {})


def _canonicalize_framingham(frame: pd.DataFrame) -> BiomarkerSourceData:
    df = normalise_frame(frame)
    target_col = "tenyearchd"
    if target_col not in df.columns:
        raise ValueError("Framingham CSV must contain TenYearCHD.")
    aliases = {
        "sex": ("male", "sex"), "age": ("age",), "education": ("education",),
        "current_smoker": ("currentsmoker", "current_smoker"),
        "cigarettes_per_day": ("cigsperday", "cigarettes_per_day"),
        "bp_medication": ("bpmeds", "bp_medication"),
        "prevalent_stroke": ("prevalentstroke", "prevalent_stroke"),
        "prevalent_hypertension": ("prevalenthyp", "prevalent_hypertension"),
        "diabetes": ("diabetes",), "total_cholesterol": ("totchol", "total_cholesterol"),
        "systolic_bp": ("sysbp", "systolic_bp"), "diastolic_bp": ("diabp", "diastolic_bp"),
        "bmi": ("bmi",), "heart_rate": ("heartrate", "heart_rate"), "glucose": ("glucose",),
    }
    mapped: dict[str, pd.Series] = {}
    for canonical, candidates in aliases.items():
        col = next((name for name in candidates if name in df.columns), None)
        if col is None:
            raise ValueError(f"Framingham CSV is missing required feature {canonical!r}; accepted columns: {candidates}.")
        mapped[canonical] = pd.to_numeric(df[col], errors="coerce")
    features = pd.DataFrame(mapped)
    target = _binary_target(df[target_col], "ten_year_chd_risk")
    return BiomarkerSourceData("framingham", features, {"ten_year_chd_risk": target}, {"ten_year_chd_risk": {"type": "binary"}}, {})


def _canonicalize_mi(frame: pd.DataFrame) -> BiomarkerSourceData:
    df = normalise_frame(frame)
    known_targets = {
        "fibr_preds": "fibr_preds", "preds_tah": "preds_tah", "jelud_tah": "jelud_tah",
        "fibr_jelud": "fibr_jelud", "a_v_blok": "a_v_blok", "o_l_post": "o_l_post",
        "otec_lanc": "otec_lanc", "razriv": "razriv", "dressler": "dressler",
        "zsn": "zsn", "rec_im": "rec_im",
        "p_im_sten": "p_im_sten", "let_is": "let_is",
    }
    target_columns = {task: column for task, column in known_targets.items() if column in df.columns}
    if not target_columns:
        raise ValueError("MI Complications CSV contains none of the configured outcome columns.")
    id_columns = {"id", "unnamed_0", "record_id"}
    excluded = set(target_columns.values()) | id_columns
    raw_features = df.drop(columns=[column for column in excluded if column in df.columns]).apply(pd.to_numeric, errors="coerce")
    # Drop features with more than half their entries missing, as specified in project data review.
    raw_features = raw_features.loc[:, raw_features.isna().mean() <= 0.50]
    raw_features = raw_features.dropna(axis=1, how="all")
    if raw_features.empty:
        raise ValueError("No usable numeric MI Complications predictors remain after missingness filtering.")
    features = raw_features.astype(float)
    targets: dict[str, pd.Series] = {}
    specs: dict[str, dict[str, Any]] = {}
    class_values: dict[str, list[Any]] = {}
    for task, column in target_columns.items():
        values = _numeric_series(df[column])
        observed = sorted(values.dropna().unique().tolist())
        if len(observed) == 2 and set(observed).issubset({0, 1, 0.0, 1.0}):
            targets[task] = values.astype(float)
            specs[task] = {"type": "binary"}
        elif task == "let_is" and len(observed) >= 2:
            # Preserve the source-defined cause codes as a categorical endpoint.
            mapping = {value: index for index, value in enumerate(observed)}
            encoded = values.map(mapping).astype(float)
            targets[task] = encoded
            specs[task] = {"type": "multiclass", "num_classes": len(observed)}
            class_values[task] = observed
        # Non-binary complication targets are deliberately omitted rather than silently binarized.
    if not targets:
        raise ValueError("No valid binary complication or multiclass LET_IS target was found.")
    return BiomarkerSourceData("mi_complications", features, targets, specs, class_values)


def load_biomarker_source(data_root: str | Path, source: str) -> BiomarkerSourceData:
    root = Path(data_root).resolve()
    source = source.strip().lower()
    file_map = {
        "zheen": ("Medicaldataset.csv", "heart_attack.csv", "zheen.csv"),
        "uci_heart_failure": ("Heart_failure_clinical_records_dataset.csv", "heart_failure_clinical_records_dataset.csv"),
        "framingham": ("framingham.csv",),
        "mi_complications": ("Myocardial infarction complications Database.csv", "mi_complications.csv"),
    }
    if source not in file_map:
        raise ValueError(f"Unknown biomarker source: {source!r}")
    path = _find_csv(root, source, file_map[source])
    frame = pd.read_csv(path, na_values=["?", "NA", "nan", ""])
    if frame.empty:
        raise ValueError(f"Dataset file is empty: {path}")
    loaders = {
        "zheen": _canonicalize_zheen,
        "uci_heart_failure": _canonicalize_uci,
        "framingham": _canonicalize_framingham,
        "mi_complications": _canonicalize_mi,
    }
    result = loaders[source](frame)
    result.features = result.features.replace([np.inf, -np.inf], np.nan)
    if result.features.shape[0] != len(next(iter(result.targets.values()))):
        raise RuntimeError(f"Feature/target row count mismatch in {source}.")
    return result
