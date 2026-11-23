from pathlib import Path

from src.fusion.registry_loader import load_disease_registry
from src.fusion.evidence_fusion import EvidenceFusion
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


def test_related_ecg_signal_is_not_treated_as_same_endpoint():
    result = engine().evaluate_disease(
        "acute_mi",
        {
            "biomarkers": {"acute_mi": 0.89, "_metadata": {"targets": {"acute_mi": {"calibrated": True}}}},
            "ecg": {"MI": 0.82, "_metadata": {"targets": {"MI": {"calibrated": True}}}},
        },
    )
    assert result["risk"] == 0.89
    assert result["risk_calibrated"] is True
    assert result["status"] == "primary_only"
    assert result["supporting_evidence"]["ecg"]["probability"] == 0.82
    assert result["supporting_evidence"]["ecg"]["role"] == "independent_ecg_label_signal"


def test_disagreement_does_not_average_probabilities():
    result = engine().evaluate_disease(
        "acute_mi",
        {
            "biomarkers": {"acute_mi": 0.93, "_metadata": {"targets": {"acute_mi": {"calibrated": True}}}},
            "ecg": {"MI": 0.28, "_metadata": {"targets": {"MI": {"calibrated": True}}}},
        },
    )
    assert result["risk"] == 0.93
    assert result["status"] == "primary_only"
    assert result["confidence"] == "LOW"


def test_agreement_uses_decision_threshold_not_probability_distance():
    test_engine = StagedFusionEngine({"diseases": {"test": {
        "primary_modality": "biomarkers", "primary_output": "primary",
        "confirmatory_modalities": ["ecg"], "confirmatory": {"ecg": {"output": "secondary", "same_endpoint": True}},
        "threshold": {"positive_probability": 0.5},
    }}})
    result = test_engine.evaluate_disease(
        "test",
        {
            "biomarkers": {"primary": 0.90, "_metadata": {"targets": {"primary": {"calibrated": True}}}},
            "ecg": {"secondary": 0.41, "_metadata": {"targets": {"secondary": {"calibrated": True}}}},
        },
    )
    assert result["status"] == "disagreement"
    assert result["risk"] == 0.90


def test_uncalibrated_or_unverified_score_is_not_reported_as_risk():
    result = engine().evaluate_disease("acute_mi", {"biomarkers": {"acute_mi": 0.89}})
    assert result["risk"] is None
    assert result["model_score"] == 0.89
    assert result["status"] == "primary_calibration_unverified"
    assert result["calibration_status"] == "not_verified"


def test_uncalibrated_confirmatory_score_does_not_claim_agreement():
    test_engine = StagedFusionEngine({"diseases": {"test": {
        "primary_modality": "biomarkers", "primary_output": "primary",
        "confirmatory_modalities": ["ecg"], "confirmatory": {"ecg": {"output": "secondary", "same_endpoint": True}},
        "threshold": {"positive_probability": 0.5},
    }}})
    result = test_engine.evaluate_disease(
        "test",
        {
            "biomarkers": {"primary": 0.89, "_metadata": {"targets": {"primary": {"calibrated": True}}}},
            "ecg": {"secondary": 0.82, "_metadata": {"targets": {"secondary": {"calibrated": False}}}},
        },
    )
    assert result["risk"] == 0.89
    assert result["status"] == "inconclusive"


def test_ef_measurement_and_flags_are_not_probabilities():
    result = engine().evaluate_disease("lv_dysfunction", {"echo": {"EF": 45.0}})
    assert result["risk"] is None
    assert result["measurement"] == {"name": "EF", "value": 45.0, "unit": "%"}
    assert result["finding"]["present"] is True
    assert result["finding"]["threshold_assessment"] == "point_estimate_only"
    assert result["finding"]["clinical_diagnosis"] is False


def test_echo_conformal_interval_is_attached_to_ef_measurement():
    result = engine().evaluate_disease(
        "reduced_lvef",
        {"echo": {"EF": 45.0, "EF_interval_lower": 38.0, "EF_interval_upper": 52.0,
                  "_metadata": {"ef_interval_nominal_coverage": 0.9}}},
    )
    assert result["risk"] is None
    assert result["measurement"]["interval"] == {"lower": 38.0, "upper": 52.0, "nominal_coverage": 0.9}


def test_ef_threshold_flag_is_indeterminate_when_interval_crosses_cutoff():
    result = engine().evaluate_disease(
        "lv_dysfunction",
        {"echo": {"EF": 45.0, "EF_interval_lower": 38.0, "EF_interval_upper": 52.0}},
    )
    assert result["risk"] is None
    assert result["finding"]["present"] is None
    assert result["finding"]["point_estimate_flag"] is True
    assert result["finding"]["threshold_assessment"] == "interval_crosses_threshold"


def test_grouped_labels_are_not_combined_into_a_probability():
    result = engine().evaluate_disease(
        "lbbb_rbbb",
        {"ecg": {"CLBBB": 0.8, "ILBBB": 0.1, "CRBBB": 0.2, "IRBBB": 0.3,
                 "_metadata": {"targets": {name: {"calibrated": True} for name in ("CLBBB", "ILBBB", "CRBBB", "IRBBB")}}}},
    )
    assert result["risk"] is None
    assert result["component_scores"]["CLBBB"] == 0.8
    assert result["finding"]["present"] is True


def test_invalid_probability_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        engine().evaluate_disease("acute_mi", {"biomarkers": {"acute_mi": float("nan")}})


def test_legacy_fusion_helper_requires_calibration_and_same_endpoint():
    fusion = EvidenceFusion()
    result = fusion.combine(0.8, "HIGH", 0.9)
    assert result["risk"] is None
    assert result["model_score"] == 0.8
    assert result["confidence"] == "LOW"
    assert result["status"] == "inconclusive"

    result = fusion.combine(
        0.8, "LOW", 0.9, primary_calibrated=True,
        confirmatory_calibrated=True, same_endpoint=True,
    )
    assert result["risk"] == 0.8
    assert result["status"] == "agreement"


def test_unsupported_disease_returns_null_risk():
    result = engine().evaluate_disease("vt_vf", {})
    assert result["risk"] is None
    assert result["status"] == "unsupported"


def test_processed_modality_without_output_is_not_told_to_upload_again():
    result = engine().evaluate_disease("av_block", {"ecg": {"CD": 0.66}})
    assert result["status"] == "output_not_modeled"
    assert result["recommendations"] == []
    assert result["primary_modality"] == "ecg"
