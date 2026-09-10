from .disagreement_handler import (
    DisagreementHandler,
)


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
    ) -> dict:

        result = {
            "risk": float(
                primary_probability
            ),
            "confidence": primary_confidence,
            "status": "primary_only",
            "evidence": {
                "primary": float(
                    primary_probability
                )
            },
        }

        if (
            confirmatory_probability
            is None
        ):
            return result

        primary_positive = (
            primary_probability
            >= agreement_threshold
        )

        confirmatory_positive = (
            confirmatory_probability
            >= agreement_threshold
        )

        result["evidence"][
            "confirmatory"
        ] = float(
            confirmatory_probability
        )

        if (
            primary_positive
            == confirmatory_positive
        ):

            result["confidence"] = (
                DisagreementHandler.upgrade(
                    primary_confidence
                )
            )

            result["status"] = "agreement"

        else:

            result["confidence"] = (
                DisagreementHandler.cap_moderate(
                    primary_confidence
                )
            )

            result["status"] = "disagreement"

        return result