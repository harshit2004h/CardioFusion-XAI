from pathlib import Path

from src.fusion.registry_loader import load_disease_registry
from src.fusion.staged_fusion import StagedFusionEngine


REGISTRY = Path(__file__).parents[1] / "config" / "disease_registry.yaml"


def engine() -> StagedFusionEngine:
    return StagedFusionEngine(load_disease_registry(REGISTRY))


def test_nested_registry_iterates_diseases_only():
    result = engine().evaluate_all({"biomarkers": {"acute_mi": 0.89}})
    assert "acute_mi" in result
    assert "settings" not in result
    assert "diseases" not in result


def test_missing_primary_returns_null_risk():
    result = engine().evaluate_disease("atrial_fibrillation", {"biomarkers": {}})
    assert result["risk"] is None
    assert result["status"] == "primary_missing"


def test_agreement_preserves_primary_probability():
    result = engine().evaluate_disease(
        "acute_mi",
        {"biomarkers": {"acute_mi": 0.89}, "ecg": {"MI": 0.82}},
    )
    assert result["risk"] == 0.89
    assert result["status"] == "agreement"


def test_disagreement_does_not_average_probabilities():
    result = engine().evaluate_disease(
        "acute_mi",
        {"biomarkers": {"acute_mi": 0.93}, "ecg": {"MI": 0.28}},
    )
    assert result["risk"] == 0.93
    assert result["status"] == "disagreement"
    assert result["confidence"] == "LOW"


def test_unsupported_disease_returns_null_risk():
    result = engine().evaluate_disease("vt_vf", {})
    assert result["risk"] is None
    assert result["status"] == "unsupported"
