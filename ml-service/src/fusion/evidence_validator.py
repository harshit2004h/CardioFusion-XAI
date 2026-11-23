"""Validation helpers for modality outputs and population restrictions."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class EvidenceValidationError(ValueError):
    """Raised when a model output cannot be interpreted safely."""


class EvidenceValidator:
    MODALITIES = {"biomarkers", "ecg", "echo"}

    @staticmethod
    def numeric(value: Any, *, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise EvidenceValidationError(f"{name} must be numeric.") from exc
        if not math.isfinite(number):
            raise EvidenceValidationError(f"{name} must be finite.")
        return number

    @classmethod
    def probability(cls, value: Any, *, name: str) -> float:
        number = cls.numeric(value, name=name)
        if not 0.0 <= number <= 1.0:
            raise EvidenceValidationError(f"{name} must be within [0, 1].")
        return number

    @classmethod
    def measurement(
        cls,
        value: Any,
        *,
        name: str,
        valid_range: list[float] | tuple[float, float] | None = None,
    ) -> float:
        number = cls.numeric(value, name=name)
        if valid_range is not None:
            lower, upper = map(float, valid_range)
            if not lower <= number <= upper:
                raise EvidenceValidationError(
                    f"{name} must be within [{lower}, {upper}]."
                )
        return number

    @staticmethod
    def context_matches(required_context: Mapping[str, Any], supplied_context: Mapping[str, Any]) -> bool:
        """Require explicitly supplied context; unknown is not treated as true."""
        return all(supplied_context.get(key) is expected for key, expected in required_context.items())
