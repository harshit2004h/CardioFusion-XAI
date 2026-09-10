class DisagreementHandler:
    """
    Handles agreement/disagreement between
    primary and confirmatory modalities.
    """

    CONFIDENCE_ORDER = {
        "LOW": 0,
        "MODERATE": 1,
        "HIGH": 2,
    }

    @classmethod
    def upgrade(
        cls,
        confidence: str,
    ) -> str:
        level = cls.CONFIDENCE_ORDER.get(
            confidence,
            0,
        )

        level = min(
            level + 1,
            2,
        )

        reverse = {
            0: "LOW",
            1: "MODERATE",
            2: "HIGH",
        }

        return reverse[level]

    @classmethod
    def cap_moderate(
        cls,
        confidence: str,
    ) -> str:
        if confidence == "HIGH":
            return "MODERATE"

        return confidence

    @staticmethod
    def agrees(
        probability: float,
        threshold: float = 0.5,
    ) -> bool:
        return probability >= threshold