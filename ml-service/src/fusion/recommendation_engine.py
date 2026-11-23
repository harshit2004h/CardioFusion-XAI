"""Disease-specific next-input recommendations."""
from __future__ import annotations

from typing import Any


class RecommendationEngine:
    """Return registry-authored recommendations for absent direct evidence."""

    def recommend(
        self,
        disease_config: dict[str, Any],
        available_modalities: set[str],
        present_evidence_modalities: set[str],
    ) -> list[str]:
        recommendations: list[str] = []
        recommendation_map = disease_config.get("missing_recommendations", {})
        for modality, message in recommendation_map.items():
            if modality not in available_modalities and modality not in present_evidence_modalities:
                recommendations.append(str(message))
        return recommendations
