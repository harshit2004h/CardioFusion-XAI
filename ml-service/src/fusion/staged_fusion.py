from .disagreement_handler import (
    DisagreementHandler,
)


class StagedFusionEngine:

    def __init__(
        self,
        registry: dict,
    ):
        if "diseases" in registry:
            self.registry = registry["diseases"]
            self.settings = registry.get("settings", {})
        else:
            self.registry = registry
            self.settings = {}

    def _base_confidence(
        self,
        auc,
    ):
        if auc is None:
            return "LOW"

        if auc >= 0.90:
            return "HIGH"

        if auc >= 0.75:
            return "MODERATE"

        return "LOW"

    @staticmethod
    def _primary_probability(
        config,
        modality_predictions,
    ):
        primary_output = config.get("primary_output")

        if isinstance(primary_output, dict):
            output_type = primary_output.get("type")

            if output_type == "derived_from_ef":
                field = primary_output.get("field", "EF")
                ef_value = modality_predictions.get(field)

                if ef_value is None:
                    return None

                return float(
                    float(ef_value)
                    < float(primary_output["threshold_ef"])
                )

            if output_type == "continuous":
                field = primary_output.get("field")
                value = modality_predictions.get(field)
                return None if value is None else float(value)

            raise ValueError(
                f"Unsupported primary output type: {output_type}"
            )

        if primary_output is not None:
            value = modality_predictions.get(primary_output)
            return None if value is None else float(value)

        for output_name in config.get("primary_outputs", []):
            value = modality_predictions.get(output_name)
            if value is not None:
                return float(value)

        for output_name in config.get("primary_output_group", []):
            value = modality_predictions.get(output_name)
            if value is not None:
                return max(
                    float(modality_predictions.get(name, 0.0))
                    for name in config["primary_output_group"]
                )

        return None

    def diseases(self):
        """Return only configured disease identifiers."""
        return tuple(self.registry)

    def evaluate_disease(
        self,
        disease,
        predictions,
    ):
        if disease not in self.registry:
            raise KeyError(
                f"Unknown disease: {disease}"
            )

        config = self.registry[disease]

        # ------------------------------------
        # Unsupported
        # ------------------------------------

        if config.get("status") == "unsupported":

            return {
                "disease": disease,
                "display_name": config.get(
                    "display_name",
                    disease
                ),
                "risk": None,
                "confidence": "LOW",
                "status": "unsupported",
                "detected_from": [],
                "recommendations": [],
                "message": config.get(
                    "reason",
                    "Unsupported disease.",
                ),
            }

        primary_modality = config.get(
            "primary_modality"
        )

        # ------------------------------------
        # Primary prediction
        # ------------------------------------

        modality_predictions = predictions.get(
            primary_modality,
            {}
        )

        primary_probability = self._primary_probability(
            config,
            modality_predictions,
        )

        if primary_probability is None:

            return {
                "disease": disease,
                "display_name": config.get(
                    "display_name",
                    disease
                ),
                "risk": None,
                "confidence": "LOW",
                "status": "primary_missing",
                "detected_from": [],
                "recommendations": [
                    f"Upload {primary_modality}."
                ],
            }

        auc = config.get(
            "primary_auc"
        )

        confidence = (
            self._base_confidence(auc)
        )

        result = {
            "disease": disease,
            "display_name": config.get(
                "display_name",
                disease
            ),
            "risk": float(
                primary_probability
            ),
            "confidence": confidence,
            "status": "primary_only",
            "detected_from": [
                primary_modality
            ],
            "recommendations": [],
        }

        # ------------------------------------
        # Confirmation
        # ------------------------------------

        confirmatory_modalities = config.get(
            "confirmatory_modalities",
            []
        )

        for modality in confirmatory_modalities:

            confirmatory_config = (
                config.get(
                    "confirmatory",
                    {}
                ).get(
                    modality,
                    {}
                )
            )

            confirmatory_output = (
                confirmatory_config.get(
                    "output"
                )
            )

            confirmatory_predictions = (
                predictions.get(
                    modality,
                    {}
                )
            )

            confirmatory_probability = None

            if confirmatory_output:

                confirmatory_probability = (
                    confirmatory_predictions.get(
                        confirmatory_output
                    )
                )

            if confirmatory_probability is None:
                result["recommendations"].append(
                    f"Upload {modality} "
                    "for confirmation."
                )
                continue

            threshold = self.settings.get(
                "agreement", {}
            ).get(
                "threshold",
                0.50,
            )
            threshold = config.get(
                "agreement_threshold",
                threshold,
            )

            result.setdefault(
                "confirmatory_evidence",
                {}
            )

            result[
                "confirmatory_evidence"
            ][modality] = float(
                confirmatory_probability
            )

            if abs(
                float(primary_probability)
                - float(confirmatory_probability)
            ) <= float(threshold):

                result["status"] = (
                    "agreement"
                )

                result["confidence"] = (
                    DisagreementHandler.upgrade(
                        confidence
                    )
                )

                result[
                    "confirmation_message"
                ] = (
                    f"{modality} provides "
                    "supporting evidence."
                )

            else:

                result["status"] = (
                    "disagreement"
                )

                result["confidence"] = (
                    DisagreementHandler.cap_moderate(
                        confidence
                    )
                )

                result[
                    "confirmation_message"
                ] = (
                    f"Primary evidence and "
                    f"{modality} disagree."
                )

        return result

    def evaluate_all(
        self,
        predictions,
    ):
        return {
            disease: self.evaluate_disease(
                disease,
                predictions,
            )
            for disease in self.diseases()
        }