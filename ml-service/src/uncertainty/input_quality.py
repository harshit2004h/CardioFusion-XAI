from __future__ import annotations

import numpy as np


def assess_numeric_input_quality(values, expected_features: int | None = None) -> dict[str, float | int | str]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if expected_features is not None and expected_features <= 0:
        raise ValueError("expected_features must be positive.")
    finite = np.isfinite(array)
    total = int(array.size)
    finite_count = int(finite.sum())
    missing_count = total - finite_count
    if expected_features is not None and total != expected_features:
        status = "feature_count_mismatch"
    elif total == 0:
        status = "empty_input"
    elif missing_count:
        status = "contains_non_finite_values"
    else:
        status = "complete_numeric_input"
    return {
        "status": status,
        "provided_features": total,
        "finite_features": finite_count,
        "non_finite_features": missing_count,
        "finite_fraction": float(finite_count / total) if total else 0.0,
    }
