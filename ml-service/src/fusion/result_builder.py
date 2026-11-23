"""Stable output schema for disease-specific evidence assessments."""
from __future__ import annotations

from typing import Any


class ResultBuilder:
    @staticmethod
    def base(disease_id: str, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "disease": disease_id,
            "display_name": config.get("display_name", disease_id),
            "category": config.get("category"),
            "status": "not_assessed",
            # risk is populated only for a calibrated binary probability.
            "risk": None,
            "risk_type": None,
            "model_score": None,
            "calibration_status": "not_available",
            "confidence": "NOT_ESTIMATED",
            "detected_from": [],
            "evidence": [],
            "related_evidence": [],
            "component_findings": [],
            "measurements": {},
            "derived_findings": [],
            "missing_modalities": [],
            "unavailable_sources": [],
            "modality_processing": {},
            "recommendations": [],
            "population_scope": config.get("target_definition"),
            "message": None,
        }
