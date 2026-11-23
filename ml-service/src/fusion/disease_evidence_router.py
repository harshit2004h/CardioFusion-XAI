"""Route model outputs to the disease definitions that can use them."""
from __future__ import annotations

from typing import Any

from .evidence_compatibility import EvidenceCompatibility


class DiseaseEvidenceRouter:
    def __init__(self, diseases: dict[str, Any]) -> None:
        if not isinstance(diseases, dict):
            raise TypeError("Disease registry section must be a mapping.")
        self.diseases = diseases
        for disease_id, config in diseases.items():
            if not isinstance(config, dict):
                raise ValueError(f"Registry entry {disease_id!r} must be a mapping.")
            for source in config.get("evidence_sources", []):
                EvidenceCompatibility.validate_source(source)

    def sources_for(self, disease_id: str) -> list[dict[str, Any]]:
        if disease_id not in self.diseases:
            raise KeyError(f"Unknown disease: {disease_id}")
        return list(self.diseases[disease_id].get("evidence_sources", []))
