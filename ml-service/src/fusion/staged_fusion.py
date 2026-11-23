"""Disease-aware evidence routing without unvalidated probability pooling."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class StagedFusionEngine:
    """Keep each disease's configured primary evidence as its only risk estimate.

    Other modalities are exposed as supporting evidence. The engine deliberately
    does not average scores, multiply likelihoods, or infer a patient-level
    confidence from cohort AUCs or simple cross-modal agreement.
    """

    def __init__(self, registry: dict):
        if "diseases" in registry:
            self.registry = registry["diseases"]
            self.settings = registry.get("settings", {})
        else:
            self.registry = registry
            self.settings = {}

    @staticmethod
    def _probability(value: Any, *, name: str) -> float:
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1].")
        return probability

    @staticmethod
    def _threshold(config: Mapping[str, Any], default: float = 0.5) -> float:
        threshold = config.get("threshold", {})
        if isinstance(threshold, Mapping):
            value = threshold.get("positive_probability", default)
        else:
            value = default
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Decision thresholds must be finite and within [0, 1].")
        return value

    @staticmethod
    def _target_metadata(predictions: Mapping[str, Any], target: str) -> Mapping[str, Any]:
        metadata = predictions.get("_metadata", {})
        targets = metadata.get("targets", {}) if isinstance(metadata, Mapping) else {}
        detail = targets.get(target, {}) if isinstance(targets, Mapping) else {}
        return detail if isinstance(detail, Mapping) else {}

    @staticmethod
    def _model_name(predictions: Mapping[str, Any]) -> str | None:
        metadata = predictions.get("_metadata", {})
        name = metadata.get("model") if isinstance(metadata, Mapping) else None
        return str(name) if name else None

    @classmethod
    def _primary_evidence(cls, config: Mapping[str, Any], predictions: Mapping[str, Any]) -> dict[str, Any]:
        """Return a probability only when the configured output is a probability."""
        output = config.get("primary_output")

        if isinstance(output, Mapping):
            output_type = output.get("type")
            field = output.get("field", "EF")
            value = predictions.get(field)
            if value is None:
                return {"risk": None}
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Primary measurement {field!r} must be finite.")
            if field == "EF" and not 0.0 <= value <= 100.0:
                raise ValueError("EF measurements must be within [0, 100].")
            measurement = {"name": field, "value": value, "unit": "%" if field == "EF" else None}
            lower = predictions.get(f"{field}_interval_lower")
            upper = predictions.get(f"{field}_interval_upper")
            if lower is not None or upper is not None:
                if lower is None or upper is None:
                    raise ValueError(f"Both lower and upper {field} interval limits are required.")
                lower = float(lower)
                upper = float(upper)
                if not math.isfinite(lower) or not math.isfinite(upper) or not 0.0 <= lower <= value <= upper <= 100.0:
                    raise ValueError(f"{field} interval must contain the measurement and stay within [0, 100].")
                metadata = predictions.get("_metadata", {})
                coverage = metadata.get("ef_interval_nominal_coverage") if isinstance(metadata, Mapping) else None
                measurement["interval"] = {"lower": lower, "upper": upper, "nominal_coverage": coverage}

            if output_type == "continuous":
                return {"risk": None, "measurement": measurement}
            if output_type == "derived_from_ef":
                threshold = float(output["threshold_ef"])
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
                    raise ValueError("EF thresholds must be finite and within [0, 100].")
                finding = value < threshold
                threshold_assessment = "point_estimate_only"
                point_estimate_flag = finding
                if "interval" in measurement:
                    interval_lower = measurement["interval"]["lower"]
                    interval_upper = measurement["interval"]["upper"]
                    if interval_upper < threshold:
                        threshold_assessment = "interval_below_threshold"
                    elif interval_lower >= threshold:
                        threshold_assessment = "interval_at_or_above_threshold"
                    else:
                        threshold_assessment = "interval_crosses_threshold"
                        finding = None
                return {
                    "risk": None,
                    "measurement": measurement,
                    "finding": {
                        "present": finding,
                        "point_estimate_flag": point_estimate_flag,
                        "threshold_assessment": threshold_assessment,
                        "kind": "threshold_flag",
                        "rule": output.get("rule", f"{field} < {threshold}"),
                        "threshold": threshold,
                        "clinical_diagnosis": False,
                    },
                }
            raise ValueError(f"Unsupported primary output type: {output_type}")

        if output is not None:
            value = predictions.get(output)
            if value is None:
                return {"risk": None}
            probability = cls._probability(value, name=f"{output} probability")
            metadata = cls._target_metadata(predictions, str(output))
            if metadata.get("calibrated") is not True:
                return {
                    "risk": None,
                    "model_score": probability,
                    "calibration_status": "uncalibrated" if metadata.get("calibrated") is False else "not_verified",
                    "model_name": cls._model_name(predictions),
                }
            return {
                "risk": probability,
                "risk_calibrated": metadata.get("calibrated"),
                "calibration_status": "platt" if metadata.get("calibrated") is True else "not_specified",
                "model_name": cls._model_name(predictions),
            }

        grouped = config.get("primary_outputs") or config.get("primary_output_group") or []
        if grouped:
            components = {name: cls._probability(predictions[name], name=f"{name} probability")
                          for name in grouped if predictions.get(name) is not None}
            if not components:
                return {"risk": None}
            uncalibrated = [name for name in components if cls._target_metadata(predictions, name).get("calibrated") is not True]
            threshold = cls._threshold(config)
            result = {"risk": None, "component_scores": components, "component_threshold": threshold}
            if uncalibrated:
                result["uncalibrated_components"] = uncalibrated
                result["calibration_status"] = "uncalibrated" if any(cls._target_metadata(predictions, name).get("calibrated") is False for name in uncalibrated) else "not_verified"
            else:
                # This is a per-label threshold flag, never the probability of their union.
                result["finding"] = {
                    "present": any(p >= threshold for p in components.values()),
                    "kind": "component_threshold_flag",
                    "rule": "at least one component reaches its configured threshold",
                    "threshold": threshold,
                    "clinical_diagnosis": False,
                }
            return result
        return {"risk": None}

    @classmethod
    def _supporting_evidence(
        cls,
        config: Mapping[str, Any],
        predictions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Expose related model outputs separately, without comparing or pooling them."""
        evidence: dict[str, dict[str, Any]] = {}
        for modality, support in (config.get("supporting", {}) or {}).items():
            if not isinstance(support, Mapping):
                continue
            output = support.get("output")
            value = predictions.get(modality, {}).get(output) if output else None
            if value is None:
                continue
            probability = cls._probability(value, name=f"{modality} supporting probability")
            metadata = cls._target_metadata(predictions.get(modality, {}), str(output))
            item = {
                "dataset": support.get("dataset"),
                "output": output,
                "role": support.get("role", "independent_supporting_signal"),
                "interpretation": support.get("interpretation"),
                "calibration_status": "platt" if metadata.get("calibrated") is True else (
                    "uncalibrated" if metadata.get("calibrated") is False else "not_verified"
                ),
            }
            item["probability" if metadata.get("calibrated") is True else "model_score"] = probability
            evidence[modality] = item
        return evidence

    @staticmethod
    def _empty_result(disease: str, config: Mapping[str, Any], status: str) -> dict[str, Any]:
        return {
            "disease": disease,
            "display_name": config.get("display_name", disease),
            "risk": None,
            "confidence": "LOW",
            "confidence_basis": "patient_level_uncertainty_not_estimated",
            "risk_calibrated": None,
            "population_scope": config.get("target_definition", config.get("primary_dataset")),
            "support_level": config.get("status", "unspecified"),
            "primary_modality": config.get("primary_modality"),
            "status": status,
            "detected_from": [],
            "recommendations": [],
        }

    def evaluate_disease(self, disease: str, predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        if disease not in self.registry:
            raise KeyError(f"Unknown disease: {disease}")
        config = self.registry[disease]
        supporting_evidence = self._supporting_evidence(config, predictions)

        if config.get("status") == "unsupported":
            result = self._empty_result(disease, config, "unsupported")
            result["message"] = config.get("reason", "Unsupported disease.")
            return result

        primary_modality = config.get("primary_modality")
        primary_predictions = predictions.get(primary_modality, {})
        evidence = self._primary_evidence(config, primary_predictions)
        if evidence.get("risk") is None and not evidence.get("measurement") and not evidence.get("component_scores"):
            if evidence.get("model_score") is not None:
                result = self._empty_result(disease, config, "uncalibrated_primary")
                if evidence.get("calibration_status") == "not_verified":
                    result["status"] = "primary_calibration_unverified"
                result["model_score"] = evidence["model_score"]
                result["calibration_status"] = evidence["calibration_status"]
                result["model_name"] = evidence.get("model_name")
                result["detected_from"] = [primary_modality]
                if supporting_evidence:
                    result["supporting_evidence"] = supporting_evidence
                result["message"] = (
                    "A model score is available, but no matching calibration artifact is loaded; it is not reported as patient risk."
                    if evidence["calibration_status"] == "uncalibrated"
                    else "A model score is available, but its calibration status was not supplied; it is not reported as patient risk."
                )
                return result
            if primary_modality in predictions:
                # The modality was uploaded and processed, but the loaded model does not
                # produce this condition's output; uploading it again would not help.
                result = self._empty_result(disease, config, "output_not_modeled")
                result["message"] = f"The loaded {primary_modality} model does not produce an output for this condition."
                return result
            result = self._empty_result(disease, config, "primary_missing")
            result["recommendations"] = [f"Upload {primary_modality}."]
            if supporting_evidence:
                result["supporting_evidence"] = supporting_evidence
            return result

        result = self._empty_result(disease, config, "primary_only")
        result["risk"] = evidence.get("risk")
        if evidence.get("risk") is not None:
            result["risk_calibrated"] = evidence.get("risk_calibrated")
            result["calibration_status"] = evidence.get("calibration_status")
            result["decision_threshold"] = self._threshold(config)
            if evidence.get("model_name"):
                result["model_name"] = evidence["model_name"]
        result["detected_from"] = [primary_modality]
        if supporting_evidence:
            result["supporting_evidence"] = supporting_evidence
        result.update({key: value for key, value in evidence.items() if key != "risk"})

        # A measured EF or a rule over multiple labels is not a calibrated disease
        # probability, so never compare it with confirmatory probabilities.
        primary_probability = evidence.get("risk")
        if primary_probability is None:
            return result

        primary_threshold = self._threshold(config)
        confirmation_states: list[bool | None] = []
        for modality in config.get("confirmatory_modalities", []):
            confirmation_config = config.get("confirmatory", {}).get(modality, {})
            output = confirmation_config.get("output")
            confirmatory_predictions = predictions.get(modality, {})
            value = confirmatory_predictions.get(output) if output else None
            if value is None:
                result["recommendations"].append(f"Upload {modality} for supporting evidence.")
                continue

            probability = self._probability(value, name=f"{modality} confirmatory probability")
            metadata = self._target_metadata(confirmatory_predictions, str(output))
            if confirmation_config.get("same_endpoint") is not True:
                result.setdefault("supporting_evidence", {})[modality] = {
                    "output": output,
                    "model_score": probability,
                    "calibration_status": "platt" if metadata.get("calibrated") is True else (
                        "uncalibrated" if metadata.get("calibrated") is False else "not_verified"
                    ),
                    "interpretation": "Separate endpoint; no threshold agreement is claimed.",
                }
                continue
            result.setdefault("confirmatory_evidence", {})[modality] = probability
            if metadata.get("calibrated") is not True:
                confirmation_states.append(None)
                continue
            confirmation_threshold = self._threshold(confirmation_config, primary_threshold)
            confirmation_states.append((primary_probability >= primary_threshold) == (probability >= confirmation_threshold))

        if confirmation_states:
            if any(state is None for state in confirmation_states):
                result["status"] = "inconclusive"
                result["confirmation_message"] = "At least one supporting score is uncalibrated, so threshold agreement was not assessed."
            else:
                result["status"] = "agreement" if all(confirmation_states) else "disagreement"
                result["confirmation_message"] = (
                    "Supporting modality decisions agree at their configured thresholds."
                    if all(confirmation_states)
                    else "Supporting modalities differ at their configured thresholds; the primary risk is unchanged."
                )
        return result

    def diseases(self) -> tuple[str, ...]:
        """Return only configured disease identifiers."""
        return tuple(self.registry)

    def evaluate_all(self, predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        return {disease: self.evaluate_disease(disease, predictions) for disease in self.diseases()}
