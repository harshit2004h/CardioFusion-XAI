import math


class EvidenceFusion:
    """
    Rule-based evidence fusion.

    Important:
        This class does NOT average or multiply
        probabilities.
    """

    def combine(
        self,
        primary_probability: float,
        primary_confidence: str,
        confirmatory_probability: float | None = None,
        agreement_threshold: float = 0.5,
        *,
        primary_calibrated: bool = False,
        confirmatory_calibrated: bool = False,
        same_endpoint: bool = False,
    ) -> dict:

        primary_probability = float(primary_probability)
        agreement_threshold = float(agreement_threshold)
        if not math.isfinite(primary_probability) or not 0.0 <= primary_probability <= 1.0:
            raise ValueError("primary_probability must be finite and within [0, 1].")
        if not math.isfinite(agreement_threshold) or not 0.0 <= agreement_threshold <= 1.0:
            raise ValueError("agreement_threshold must be finite and within [0, 1].")

        primary_is_calibrated = primary_calibrated is True
        confirmatory_is_calibrated = confirmatory_calibrated is True
        result = {
            "risk": primary_probability if primary_is_calibrated else None,
            "model_score": None if primary_is_calibrated else primary_probability,
            "risk_calibrated": primary_is_calibrated,
            "confidence": primary_confidence if primary_is_calibrated else "LOW",
            "status": "primary_only" if primary_is_calibrated else "uncalibrated_primary",
            "evidence": {
                "primary": primary_probability
            },
        }

        if (
            confirmatory_probability
            is None
        ):
            return result

        confirmatory_probability = float(confirmatory_probability)
        if not math.isfinite(confirmatory_probability) or not 0.0 <= confirmatory_probability <= 1.0:
            raise ValueError("Probabilities must be within [0, 1].")

        primary_positive = primary_probability >= agreement_threshold
        confirmatory_positive = confirmatory_probability >= agreement_threshold

        result["evidence"][
            "confirmatory"
        ] = float(
            confirmatory_probability
        )
        result["evidence"]["agreement_threshold"] = agreement_threshold

        if same_endpoint is not True or not primary_is_calibrated or not confirmatory_is_calibrated:
            result["status"] = "inconclusive"
            result["evidence"]["comparison"] = "not assessed: endpoint identity or calibration is unverified"
            return result

        if (
            primary_positive
            == confirmatory_positive
        ):

            result["status"] = "agreement"

        else:

            result["status"] = "disagreement"

        return result
