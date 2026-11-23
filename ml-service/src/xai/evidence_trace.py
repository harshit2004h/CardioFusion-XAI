from __future__ import annotations

from typing import Any


def build_evidence_trace(disease_result: dict[str, Any]) -> dict[str, Any]:
    """Explain the deterministic registry/orchestration decisions for one endpoint."""
    evidence = disease_result.get("evidence", [])
    related = disease_result.get("related_evidence", [])
    rejected = disease_result.get("rejected_evidence", [])
    return {
        "disease": disease_result.get("disease"),
        "status": disease_result.get("status"),
        "direct_evidence_sources": [
            {"modality": item.get("modality"), "dataset": item.get("dataset"), "output": item.get("output"), "calibration_status": item.get("calibration_status")}
            for item in evidence
        ],
        "related_independent_findings": [
            {"modality": item.get("modality"), "dataset": item.get("dataset"), "output": item.get("output"), "interpretation": item.get("interpretation")}
            for item in related
        ],
        "rejected_or_unavailable_sources": rejected,
        "missing_modalities": list(disease_result.get("missing_modalities", [])),
        "recommendations": list(disease_result.get("recommendations", [])),
        "probability_policy": "No probability is combined across unpaired datasets. A percentage is exposed only for a calibrated model output with the same endpoint definition.",
    }
