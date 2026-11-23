from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def explain_tabular_prediction(
    predict_function: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    sample: np.ndarray,
    feature_names: Sequence[str],
    max_evals: int | None = None,
) -> dict:
    """Explain one selected scalar output using model-agnostic SHAP.

    `predict_function` must return one scalar per row for a single dataset/task.
    Background rows must come from that same dataset's training partition.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install the explainability dependencies from requirements.txt.") from exc
    background = np.asarray(background, dtype=np.float32)
    sample = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    if background.ndim != 2 or background.shape[0] < 2:
        raise ValueError("background must contain at least two rows and be 2D.")
    if sample.shape[1] != background.shape[1] or len(feature_names) != sample.shape[1]:
        raise ValueError("Feature dimensions/names must agree across background and sample.")
    if not np.isfinite(background).all() or not np.isfinite(sample).all():
        raise ValueError("SHAP inputs must be finite after fitted preprocessing.")
    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(predict_function, masker, algorithm="permutation", feature_names=list(feature_names))
    budget = max_evals or (2 * sample.shape[1] + 1)
    values = explainer(sample, max_evals=budget)
    contribution = np.asarray(values.values).reshape(-1)
    base_value = float(np.asarray(values.base_values).reshape(-1)[0])
    prediction = float(np.asarray(predict_function(sample)).reshape(-1)[0])
    ordered = np.argsort(np.abs(contribution))[::-1]
    return {
        "method": "SHAP_PermutationExplainer",
        "output": prediction,
        "base_value": base_value,
        "additivity_residual": float(prediction - (base_value + contribution.sum())),
        "feature_contributions": [
            {"feature": str(feature_names[index]), "shap_value": float(contribution[index])}
            for index in ordered
        ],
        "background_size": int(background.shape[0]),
        "warning": "SHAP values explain this model output relative to the selected same-dataset background; they are not causal effects.",
    }
