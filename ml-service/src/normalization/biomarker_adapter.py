from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.extraction.schemas import DocumentExtraction


@dataclass(frozen=True)
class BiomarkerInput:
    source: str
    values: np.ndarray
    feature_names: tuple[str, ...]
    task: str
    missing_features: tuple[str, ...]


SOURCE_SCHEMAS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "zheen": (
        ("age", "gender", "heart_rate", "systolic_bp", "diastolic_bp", "blood_glucose", "ck_mb", "troponin"),
        "acute_mi",
        "zheen_acute_mi",
    ),
    "uci_heart_failure": (
        ("age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking"),
        "hf_mortality",
        "uci_heart_failure_hf_mortality",
    ),
    "framingham": (
        ("sex", "age", "education", "current_smoker", "cigarettes_per_day", "bp_medication", "prevalent_stroke", "prevalent_hypertension", "diabetes", "total_cholesterol", "systolic_bp", "diastolic_bp", "bmi", "heart_rate", "glucose"),
        "ten_year_chd_risk",
        "framingham_ten_year_chd_risk",
    ),
}


def select_biomarker_input(extraction: DocumentExtraction) -> BiomarkerInput | None:
    values: dict[str, float] = {}
    for test in extraction.tests:
        if test.canonical_candidate and test.value is not None:
            values[test.canonical_candidate] = float(test.value)
    candidates: list[tuple[int, str, tuple[str, ...], str, str]] = []
    for source, (features, task, calibration_task) in SOURCE_SCHEMAS.items():
        present = sum(feature in values for feature in features)
        candidates.append((present, source, features, task, calibration_task))
    present, source, features, task, _ = max(candidates, key=lambda item: item[0])
    missing = tuple(feature for feature in features if feature not in values)
    if present < len(features):
        return BiomarkerInput(source, np.empty((0,), dtype=np.float32), features, task, missing)
    return BiomarkerInput(source, np.asarray([values[feature] for feature in features], dtype=np.float32), features, task, ())