"""Small, reusable biomarker-training utilities imported by notebooks and tests.

This module deliberately contains no dataset-training entry point, CLI or epoch loop.
The full experiment workflow lives in Jupyter notebooks.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import torch
from src.training.biomarker_training_data import BiomarkerSourceData
from src.training.losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MultiTaskLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss


def split_indices(data: BiomarkerSourceData, seed: int = 42):
    n_rows = len(data.features)
    if n_rows < 30:
        raise ValueError(f"{data.source} has only {n_rows} rows; at least 30 are required for train/validation/test splitting.")
    stratify = None
    for target in data.targets.values():
        values = pd.to_numeric(target, errors="coerce")
        if values.isna().any():
            continue
        counts = values.value_counts()
        if len(counts) >= 2 and counts.min() >= 8:
            stratify = values.to_numpy()
            break
    indices = np.arange(n_rows)
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=stratify)
    temp_stratify = None
    if stratify is not None:
        temp_values = pd.Series(stratify[temp_idx])
        counts = temp_values.value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            temp_stratify = temp_values
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed + 1, stratify=temp_stratify)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def make_loss(target_specs: dict[str, dict[str, Any]], train_targets: dict[str, np.ndarray]) -> MultiTaskLoss:
    losses = {}
    for task, spec in target_specs.items():
        if spec["type"] == "binary":
            target = np.asarray(train_targets[f"{task}__target"], dtype=float).reshape(-1)
            mask = np.asarray(train_targets[f"{task}__mask"], dtype=float).reshape(-1) > 0
            observed = target[mask]
            positives = int((observed == 1).sum()); negatives = int((observed == 0).sum())
            pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32) if positives and negatives else None
            losses[task] = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
        elif spec["type"] == "multiclass":
            losses[task] = MaskedCrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported target type: {spec['type']}")
    return MultiTaskLoss(losses)


def multiclass_test_metrics(frame: pd.DataFrame, task: str) -> dict[str, Any]:
    subset = frame[(frame["target"] == task) & (frame["output_type"] == "multiclass_class_probability")].copy()
    if subset.empty:
        return {"status": "no_test_rows"}
    grouped = subset.sort_values(["row_index", "class_index"]).groupby("row_index")
    y_true, distributions = [], []
    for _, group in grouped:
        labels = group["class_index"].to_numpy(dtype=int)
        probabilities = group["prediction"].to_numpy(dtype=float)
        if len(labels) < 2 or not np.array_equal(labels, np.arange(len(labels))) or not np.isfinite(probabilities).all():
            continue
        total = probabilities.sum()
        if total <= 0 or not np.isclose(total, 1.0, atol=1e-4):
            continue
        probabilities = probabilities / total
        y_true.append(int(group["true"].iloc[0])); distributions.append(probabilities)
    if not y_true:
        return {"status": "no_complete_multiclass_distributions"}
    probs = np.asarray(distributions, dtype=np.float64)
    pred = probs.argmax(axis=1)
    return {"status": "computed", "n_test": len(y_true),
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y_true, probs, labels=np.arange(probs.shape[1])))}
