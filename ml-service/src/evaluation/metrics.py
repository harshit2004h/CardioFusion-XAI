from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]
    if not y_true.size:
        return {name: np.nan for name in ("auroc", "auprc", "f1", "precision", "recall")}
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_prob)) if np.unique(y_true).size > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def multiclass_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def regression_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Calculate standard regression metrics for ejection-fraction predictions."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if y_true.size == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else np.nan,
    }


def ef_threshold_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Evaluate EF threshold classifications at 50% and 40%."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    metrics = {}

    for threshold in (50, 40):
        true_labels = y_true < threshold
        predicted_labels = y_pred < threshold
        metrics[f"EF_<{threshold}_Accuracy"] = float(
            np.mean(true_labels == predicted_labels)
        ) if y_true.size else np.nan

    return metrics


__all__ = [
    "binary_metrics",
    "multiclass_metrics",
    "regression_metrics",
    "ef_threshold_metrics",
]

def multilabel_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    per_label_auc = []

    for i in range(y_true.shape[1]):
        unique_values = np.unique(
            y_true[:, i]
        )

        if len(unique_values) < 2:
            per_label_auc.append(np.nan)
        else:
            per_label_auc.append(
                roc_auc_score(
                    y_true[:, i],
                    y_prob[:, i],
                )
            )

    valid_auc = [
        x
        for x in per_label_auc
        if not np.isnan(x)
    ]

    macro_auc = (
        float(np.mean(valid_auc))
        if valid_auc
        else np.nan
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_auprc = average_precision_score(
        y_true,
        y_prob,
        average="macro",
    )

    return {
        "macro_AUROC": float(macro_auc),
        "macro_AUPRC": float(macro_auprc),
        "macro_F1": float(macro_f1),
        "per_label_AUROC": per_label_auc,
    }


def binary_confusion_matrix(
    y_true,
    y_prob,
    threshold=0.5,
):
    y_pred = (
        np.asarray(y_prob) >= threshold
    ).astype(int)

    return confusion_matrix(
        np.asarray(y_true),
        y_pred,
    )