"""Describe cross-source relationships without pretending that outputs are paired.

The project datasets do not share patient identifiers or a common outcome label.
Accordingly this module never compares, averages, multiplies, or votes over values
from different modalities. It only classifies the relationship declared in the
registry and creates a transparent evidence description.
"""
from __future__ import annotations

from typing import Any


VALID_ROLES = {
    "direct_dataset_prediction",
    "direct_measurement",
    "related_independent_finding",
    "contextual_finding",
}


class EvidenceCompatibility:
    """Validate and describe evidence roles from the disease registry."""

    @staticmethod
    def validate_source(source: dict[str, Any]) -> None:
        modality = source.get("modality")
        dataset = source.get("dataset")
        output = source.get("output")
        role = source.get("role")
        if modality not in {"biomarkers", "ecg", "echo"}:
            raise ValueError(f"Unsupported modality in evidence registry: {modality!r}")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("Every evidence source must declare a dataset.")
        if not isinstance(output, str) or not output.strip():
            raise ValueError("Every evidence source must declare an output name.")
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported evidence role: {role!r}")

    @staticmethod
    def describe(source: dict[str, Any], value: Any) -> dict[str, Any]:
        return {
            "modality": source["modality"],
            "dataset": source["dataset"],
            "output": source["output"],
            "role": source["role"],
            "output_type": source.get("output_type", "binary_probability"),
            "value": value,
            "interpretation": source.get("interpretation"),
            "population_scope": source.get("population_scope"),
            "component": source.get("component"),
            "horizon": source.get("horizon"),
        }

    @staticmethod
    def cross_modal_policy() -> str:
        return (
            "Independent findings are displayed side by side. No cross-modal "
            "probability comparison or combined probability is calculated because "
            "the source datasets are unpaired and use different endpoint definitions."
        )
