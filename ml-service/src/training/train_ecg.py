"""Train the PTB-XL ECG multi-label classifier.

Target heads required by the project design:
- diagnostic superclasses: NORM, MI, STTC, CD, HYP
- rhythm labels: 12 rhythm targets

The model should expose ``diagnostic_logits`` and ``rhythm_logits`` (or return
them inside a ``logits`` dictionary). PTB-XL is not used to create a general-
population VT/VF head because that label is not a suitable target for this
dataset in the project design.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from .losses import MaskedBCEWithLogitsLoss, MultiTaskLoss
from .trainer import EarlyStopping, Trainer, seed_everything


def load_factory(spec: str):
    module_name, _, attr = spec.rpartition(":")
    if not module_name:
        raise ValueError(f"Expected MODULE:FUNCTION, got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--factory", default="src.data.ecg_dataset:build_ecg_dataloaders")
    p.add_argument("--model-factory", default="src.models.ecg:build_ecg_model")
    p.add_argument("--data-root", required=True)
    p.add_argument("--artifact-dir", default="artifacts/ecg")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    build_loaders = load_factory(args.factory)
    build_model = load_factory(args.model_factory)
    loaders = build_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = build_model(
        diagnostic_classes=["NORM", "MI", "STTC", "CD", "HYP"],
        rhythm_classes=12,
    )

    # Per-head pos_weight should be computed from the training fold and passed
    # by the model or loader metadata; defaults here keep the script runnable.
    diagnostic_pw = getattr(model, "diagnostic_pos_weight", None)
    rhythm_pw = getattr(model, "rhythm_pos_weight", None)
    criterion = MultiTaskLoss({
        "diagnostic": MaskedBCEWithLogitsLoss(diagnostic_pw),
        "rhythm": MaskedBCEWithLogitsLoss(rhythm_pw),
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lambda outputs, batch: criterion(
            outputs.get("logits", outputs),
            batch["targets"],
            batch.get("masks"),
        ),
        device=args.device,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        ),
        checkpoint_dir=Path(args.artifact_dir) / "checkpoints",
    )

    trainer.fit(
        loaders["train"],
        loaders.get("val"),
        epochs=args.epochs,
        early_stopping=EarlyStopping(args.patience, mode="min"),
        save_name="ptbxl_xresnet1d101.pt",
    )


if __name__ == "__main__":
    main()
