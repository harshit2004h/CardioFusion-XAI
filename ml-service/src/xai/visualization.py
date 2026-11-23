from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_feature_contribution_plot(contributions: Sequence[dict], output_path: str | Path, top_n: int = 15) -> Path:
    """Save a horizontal SHAP contribution plot; caller supplies signed SHAP values."""
    import matplotlib.pyplot as plt

    rows = list(contributions)[:top_n]
    if not rows:
        raise ValueError("No feature contributions were supplied.")
    labels = [str(item["feature"]) for item in reversed(rows)]
    values = [float(item["shap_value"]) for item in reversed(rows)]
    if not np.isfinite(values).all():
        raise ValueError("Feature contributions must be finite.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, max(3, 0.35 * len(rows))))
    axis.barh(labels, values)
    axis.axvline(0.0, linewidth=0.8)
    axis.set_xlabel("SHAP value (model-output contribution)")
    axis.set_title("Feature contributions for this prediction")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path
