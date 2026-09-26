"""
Train the PTB-XL ECG multi-label classifier.

Supported ECG models:
    - xresnet1d
    - cnn_bilstm_attention

Project targets:
    Diagnostic superclasses:
        NORM, MI, STTC, CD, HYP

    Rhythm:
        project-defined rhythm targets from the PTB-XL manifest

Expected model output:

    {
        "diagnostic": diagnostic_logits,
        "rhythm": rhythm_logits,
        ...
    }

or:

    {
        "logits": {
            "diagnostic": diagnostic_logits,
            "rhythm": rhythm_logits,
        }
    }

Important:
    PTB-XL is treated as a multi-label problem.
    Therefore BCEWithLogits-based losses are used.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# Existing project training components.
from .losses import (
    MaskedBCEWithLogitsLoss,
    MultiTaskLoss,
)

from .trainer import (
    EarlyStopping,
    Trainer,
    seed_everything,
)


# ============================================================
# Factory loader
# ============================================================

def load_factory(
    spec: str,
):
    """
    Load a function or callable from:

        module.path:function_name
    """

    module_name, separator, attribute = (
        spec.rpartition(":")
    )

    if not separator:

        raise ValueError(
            "Expected MODULE:FUNCTION, "
            f"got {spec!r}"
        )

    module = importlib.import_module(
        module_name
    )

    return getattr(
        module,
        attribute
    )


# ============================================================
# Argument parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train a PTB-XL ECG "
            "multi-label model."
        )
    )

    # --------------------------------------------------------
    # Data factory
    # --------------------------------------------------------

    parser.add_argument(
        "--factory",
        required=True,
        help=(
            "MODULE:FUNCTION for "
            "building train/val/test loaders."
        ),
    )

    # --------------------------------------------------------
    # Model factory
    # --------------------------------------------------------

    parser.add_argument(
        "--model-factory",
        required=True,
        help=(
            "MODULE:FUNCTION for "
            "building the ECG model."
        ),
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    parser.add_argument(
        "--data-root",
        required=True,
        help="PTB-XL project/data root.",
    )

    parser.add_argument(
        "--artifact-dir",
        default="artifacts/metrics/ecg",
        help="Directory for training artifacts.",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    # --------------------------------------------------------
    # Model name
    # --------------------------------------------------------

    parser.add_argument(
        "--model-name",
        default="cnn_bilstm_attention",
        choices=[
            "xresnet1d",
            "cnn_bilstm_attention",
        ],
    )

    return parser.parse_args()


# ============================================================
# Model output normalization
# ============================================================

def extract_logits(
    outputs: Any,
):
    """
    Normalize different model return formats.

    Supported:

        {
            "diagnostic": ...,
            "rhythm": ...
        }

    or:

        {
            "logits": {
                "diagnostic": ...,
                "rhythm": ...
            }
        }

    Returns:

        {
            "diagnostic": ...,
            "rhythm": ...
        }
    """

    if not isinstance(
        outputs,
        dict,
    ):

        raise TypeError(
            "ECG model output must "
            "be a dictionary."
        )

    # ---------------------------------------------
    # Nested logits dictionary
    # ---------------------------------------------

    if "logits" in outputs:

        logits = outputs["logits"]

    else:

        logits = outputs

    if not isinstance(
        logits,
        dict,
    ):

        raise TypeError(
            "Expected logits to be "
            "a dictionary."
        )

    if "diagnostic" not in logits:

        raise KeyError(
            "Model output is missing "
            "'diagnostic' logits."
        )

    if "rhythm" not in logits:

        raise KeyError(
            "Model output is missing "
            "'rhythm' logits."
        )

    return {
        "diagnostic":
            logits["diagnostic"],

        "rhythm":
            logits["rhythm"],
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed_everything(
        args.seed
    )

    # --------------------------------------------------------
    # Load loader factory
    # --------------------------------------------------------

    build_loaders = load_factory(
        args.factory
    )

    # --------------------------------------------------------
    # Load model factory
    # --------------------------------------------------------

    build_model = load_factory(
        args.model_factory
    )

    # --------------------------------------------------------
    # Build datasets/loaders
    # --------------------------------------------------------

    loaders = build_loaders(

        data_root=args.data_root,

        batch_size=args.batch_size,

        seed=args.seed,
    )

    required_loader_keys = [
        "train",
    ]

    for key in required_loader_keys:

        if key not in loaders:

            raise KeyError(
                f"Loader dictionary is missing "
                f"required key: {key}"
            )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    diagnostic_targets = loaders.get(
        "diagnostic_targets",
        ["NORM", "MI", "STTC", "CD", "HYP"],
    )

    rhythm_targets = loaders.get(
        "rhythm_targets",
        [],
    )

    if not rhythm_targets:
        raise ValueError(
            "No rhythm targets were found in the PTB-XL manifest. "
            "Check the processed ECG manifest columns."
        )

    model = build_model(

        diagnostic_classes=
            diagnostic_targets,

        rhythm_classes=
            len(rhythm_targets),
    )

    # --------------------------------------------------------
    # Print model
    # --------------------------------------------------------

    print("=" * 80)

    print(
        "PTB-XL ECG TRAINING"
    )

    print("=" * 80)

    print(
        "Model:",
        args.model_name
    )

    print(
        "Epochs:",
        args.epochs
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Learning rate:",
        args.lr
    )

    print(
        "Weight decay:",
        args.weight_decay
    )

    print()

    print(model)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print()

    print(
        f"Total parameters    : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------
    #
    # If the model factory provides:
    #
    #     model.diagnostic_pos_weight
    #     model.rhythm_pos_weight
    #
    # they will be used.
    #
    # Otherwise, BCE uses equal weights.
    # --------------------------------------------------------

    diagnostic_pw = getattr(
        model,
        "diagnostic_pos_weight",
        None,
    )

    rhythm_pw = getattr(
        model,
        "rhythm_pos_weight",
        None,
    )

    # --------------------------------------------------------
    # Multi-task loss
    # --------------------------------------------------------

    criterion = MultiTaskLoss({

        "diagnostic":
            MaskedBCEWithLogitsLoss(
                diagnostic_pw
            ),

        "rhythm":
            MaskedBCEWithLogitsLoss(
                rhythm_pw
            ),
    })

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=3,

            min_lr=1e-7,
        )
    )

    # --------------------------------------------------------
    # Loss wrapper
    # --------------------------------------------------------

    def loss_fn(
        outputs,
        batch,
    ):

        logits = extract_logits(
            outputs
        )

        return criterion(

            logits,

            batch["targets"],

            batch.get(
                "masks"
            ),
        )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        loss_fn=loss_fn,

        device=args.device,

        scheduler=scheduler,

        checkpoint_dir=(
            Path(
                args.artifact_dir
            )
            / "checkpoints"
        ),
    )

    # --------------------------------------------------------
    # Checkpoint filename
    # --------------------------------------------------------

    if args.model_name == (
        "cnn_bilstm_attention"
    ):

        save_name = (
            "ptbxl_"
            "cnn_bilstm_attention.pt"
        )

    else:

        save_name = (
            "ptbxl_xresnet1d.pt"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    trainer.fit(

        loaders["train"],

        loaders.get(
            "val"
        ),

        epochs=args.epochs,

        early_stopping=(
            EarlyStopping(
                args.patience,
                mode="min",
            )
        ),

        save_name=save_name,
    )


if __name__ == "__main__":

    main()