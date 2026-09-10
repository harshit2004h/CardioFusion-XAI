from __future__ import annotations

import numpy as np
import torch

from .metrics import (
    binary_metrics,
    multiclass_metrics,
)


def collect_predictions(
    model,
    loader,
    source: str,
    device: torch.device,
    multiclass_targets: set[str] | None = None,
):
    """
    Run inference over one source-specific DataLoader.

    Returns:
        {
            target_name: {
                "true": np.ndarray,
                "prob": np.ndarray
            }
        }

    For multiclass targets:
        "prob" contains predicted class indices.
    """

    if multiclass_targets is None:
        multiclass_targets = set()

    model.eval()

    predictions = {}
    truths = {}

    with torch.no_grad():

        for x, targets in loader:

            x = x.to(
                device,
                non_blocking=True
            )

            outputs = model(
                x,
                source
            )

            for target, y in targets.items():

                if target not in outputs:
                    continue

                logits = outputs[target]

                if target in multiclass_targets:

                    valid = (
                        y >= 0
                    )

                    if not valid.any():
                        continue

                    pred = (
                        logits[valid]
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                    )

                    true = (
                        y[valid]
                        .cpu()
                        .numpy()
                    )

                else:

                    y_np = (
                        y.cpu()
                        .numpy()
                    )

                    valid = ~np.isnan(
                        y_np
                    )

                    if not valid.any():
                        continue

                    pred = (
                        torch.sigmoid(
                            logits[:, 0]
                        )
                        .cpu()
                        .numpy()
                    )

                    pred = pred[valid]
                    true = y_np[valid]

                predictions.setdefault(
                    target,
                    []
                )

                truths.setdefault(
                    target,
                    []
                )

                predictions[target].extend(
                    pred.tolist()
                )

                truths[target].extend(
                    true.tolist()
                )

    return {
        target: {
            "true": np.asarray(
                truths[target]
            ),
            "pred": np.asarray(
                predictions[target]
            ),
        }
        for target in predictions
    }


def evaluate_predictions(
    prediction_dict,
    multiclass_targets: set[str] | None = None,
    thresholds: dict[str, float] | None = None,
):
    """
    Calculate metrics from collected predictions.
    """

    if multiclass_targets is None:
        multiclass_targets = set()

    if thresholds is None:
        thresholds = {}

    results = {}

    for target, values in prediction_dict.items():

        y_true = values["true"]
        y_pred = values["pred"]

        if target in multiclass_targets:

            results[target] = (
                multiclass_metrics(
                    y_true,
                    y_pred
                )
            )

        else:

            results[target] = (
                binary_metrics(
                    y_true,
                    y_pred,
                    threshold=thresholds.get(
                        target,
                        0.5,
                    ),
                )
            )

    return results




import numpy as np
import torch


@torch.no_grad()
def collect_echo_predictions(
    model,
    loader,
    device,
):
    model.eval()

    y_true = []
    y_pred = []
    video_paths = []

    for batch in loader:
        videos = batch["video"].to(
            device,
            non_blocking=True,
        )

        targets = batch["ef"]

        outputs = model(videos)

        y_true.extend(
            targets.cpu().numpy().tolist()
        )

        y_pred.extend(
            outputs.cpu().numpy().reshape(-1).tolist()
        )

        video_paths.extend(
            batch["video_path"]
        )

    return {
        "y_true": np.asarray(
            y_true,
            dtype=np.float32,
        ),
        "y_pred": np.asarray(
            y_pred,
            dtype=np.float32,
        ),
        "video_paths": video_paths,
    }


@torch.no_grad()
def collect_ecg_predictions(
    model,
    loader,
    device,
):
    model.eval()

    diagnostic_true = []
    diagnostic_prob = []

    rhythm_true = []
    rhythm_prob = []

    record_paths = []

    for batch in loader:
        signal = batch["signal"].to(
            device,
            non_blocking=True,
        )

        outputs = model(signal)

        diagnostic_logits = (
            outputs["diagnostic"]
        )

        rhythm_logits = (
            outputs["rhythm"]
        )

        diagnostic_probability = (
            torch.sigmoid(
                diagnostic_logits
            )
        )

        rhythm_probability = (
            torch.sigmoid(
                rhythm_logits
            )
        )

        diagnostic_true.append(
            batch["diagnostic"]
            .cpu()
            .numpy()
        )

        diagnostic_prob.append(
            diagnostic_probability
            .cpu()
            .numpy()
        )

        rhythm_true.append(
            batch["rhythm"]
            .cpu()
            .numpy()
        )

        rhythm_prob.append(
            rhythm_probability
            .cpu()
            .numpy()
        )

        if "record_path" in batch:
            record_paths.extend(batch["record_path"])
        else:
            record_paths.extend(
                [str(len(record_paths) + index) for index in range(signal.size(0))]
            )

    return {
        "diagnostic_true": np.concatenate(
            diagnostic_true,
            axis=0,
        ),
        "diagnostic_prob": np.concatenate(
            diagnostic_prob,
            axis=0,
        ),
        "rhythm_true": np.concatenate(
            rhythm_true,
            axis=0,
        ),
        "rhythm_prob": np.concatenate(
            rhythm_prob,
            axis=0,
        ),
        "record_paths": record_paths,
    }