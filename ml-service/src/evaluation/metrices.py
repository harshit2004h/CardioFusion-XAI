from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


def binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calculate metrics for one binary prediction task.

    Parameters
    ----------
    y_true:
        True binary labels.

    y_prob:
        Predicted probabilities.

    threshold:
        Probability threshold used for F1/precision/recall.

    Returns
    -------
    dict
        AUROC, AUPRC, F1, precision and recall.
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_prob)
    )

    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
        }

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    if len(np.unique(y_true)) >= 2:
        auroc = roc_auc_score(
            y_true,
            y_prob
        )

        auprc = average_precision_score(
            y_true,
            y_prob
        )
    else:
        auroc = np.nan
        auprc = np.nan

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


def multiclass_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate metrics for a multiclass task.
    """

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_pred = np.asarray(
        y_pred
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def confusion_matrix_binary(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Return binary confusion matrix.
    """

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_prob = np.asarray(
        y_prob
    ).astype(float)

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    return confusion_matrix(
        y_true,
        y_pred
    )