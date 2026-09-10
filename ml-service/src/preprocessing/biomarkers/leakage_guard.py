"""Leakage protection for the project's four biomarker datasets.

Project-specific rules from the design document:
- Hugging Face heart-failure-prediction is NOT used.
- UCI Heart Failure 'time' is follow-up time and must not be used to predict
  DEATH_EVENT.
- Target columns must never enter X.
- Patient/record identifiers are excluded by default.
- Dataset-specific forbidden columns can be supplied explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


DATASET_TARGETS = {
    "zheen": {"heart_attack", "heart_attack_prediction", "output"},
    "uci_heart_failure": {"death_event"},
    "mi_complications": {
        "fibr_preds", "preds_tah", "jelud_tah", "fibr_jelud",
        "a_v_blok", "otek_lanc", "rec_im", "let_is",
        "razriv", "dressler", "zsn", "p_im_sten",
    },
    "framingham": {"tenyearchd"},
}

DATASET_FORBIDDEN = {
    # Follow-up duration directly encodes survival information.
    "uci_heart_failure": {"time"},
    "zheen": set(),
    "mi_complications": set(),
    "framingham": set(),
}

COMMON_ID_COLUMNS = {
    "id", "patient_id", "patientid", "subject_id", "subjectid",
    "record_id", "recordid", "encounter_id", "encounterid",
}


@dataclass
class LeakageGuard:
    """Detect/remove columns that should not enter a model."""

    dataset_name: str | None = None
    target_columns: set[str] = field(default_factory=set)
    forbidden_columns: set[str] = field(default_factory=set)
    allow_id_columns: bool = False

    def __post_init__(self) -> None:
        if self.dataset_name:
            self.dataset_name = self.dataset_name.strip().lower()

        self.target_columns = {
            str(c).strip().lower()
            for c in self.target_columns
        }
        self.forbidden_columns = {
            str(c).strip().lower()
            for c in self.forbidden_columns
        }

    def forbidden(self, columns: Iterable[str]) -> set[str]:
        original = {str(c) for c in columns}
        normalized = {c.strip().lower() for c in original}

        blocked = set(self.forbidden_columns)

        if self.dataset_name in DATASET_FORBIDDEN:
            blocked.update(
                DATASET_FORBIDDEN[self.dataset_name]
            )

        if self.dataset_name in DATASET_TARGETS:
            blocked.update(
                normalized.intersection(
                    DATASET_TARGETS[self.dataset_name]
                )
            )

        blocked.update(normalized.intersection(self.target_columns))

        if not self.allow_id_columns:
            blocked.update(
                normalized.intersection(COMMON_ID_COLUMNS)
            )

        # Return the actual column spellings.
        return {
            column
            for column in original
            if column.strip().lower() in blocked
        }

    def validate(self, feature_columns: Iterable[str]) -> None:
        blocked = self.forbidden(feature_columns)
        if blocked:
            raise ValueError(
                "Potential leakage columns detected: "
                + ", ".join(sorted(blocked))
            )

    def remove_forbidden(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        blocked = self.forbidden(df.columns)
        return df.drop(columns=list(blocked), errors="ignore")


def make_training_features(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    target_columns: Iterable[str] = (),
    extra_forbidden_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Return X after applying the project's leakage rules."""
    guard = LeakageGuard(
        dataset_name=dataset_name,
        target_columns=set(target_columns),
        forbidden_columns=set(extra_forbidden_columns),
    )
    guard.validate(df.columns)
    return guard.remove_forbidden(df)


def assert_no_leakage(
    feature_columns: Iterable[str],
    *,
    dataset_name: str | None = None,
    target_columns: Iterable[str] = (),
    forbidden_columns: Iterable[str] = (),
) -> None:
    guard = LeakageGuard(
        dataset_name=dataset_name,
        target_columns=set(target_columns),
        forbidden_columns=set(forbidden_columns),
    )
    guard.validate(feature_columns)
