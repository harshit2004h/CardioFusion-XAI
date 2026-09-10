"""Train the shared biomarker FT-Transformer with masked multi-task labels.

The four source datasets remain schema-specific. The dataset/model layer is
expected to provide:
    build_biomarker_dataloaders(...)
    build_biomarker_model(...)
The loaders should return dictionaries with:
    x / inputs
    targets: {task_name: tensor}
    masks:   {task_name: tensor}

A Framingham SMOTE implementation, if used, must be applied only to the
training fold before the DataLoader is constructed.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch

from .losses import FocalLoss, MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, MultiTaskLoss
from .trainer import EarlyStopping, Trainer, seed_everything


def load_factory(spec: str, name: str):
    module_name, _, attr = spec.rpartition(":")
    if not module_name:
        raise ValueError(f"Expected MODULE:FUNCTION, got {spec!r}")
    fn = getattr(importlib.import_module(module_name), attr)
    return fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--factory", required=True)
    p.add_argument("--model-factory", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--artifact-dir", default="artifacts/biomarkers")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    build_loaders = load_factory(args.factory, "build_biomarker_dataloaders")
    build_model = load_factory(args.model_factory, "build_biomarker_model")

    loaders = build_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if not isinstance(loaders, dict) or "train" not in loaders:
        raise ValueError("Biomarker loader factory must return {'train': ..., 'val': ...}.")

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Task definitions are intentionally read from the model when available.
    # Example names: acute_mi, hf_mortality, ten_year_chd, complication_*,
    # let_is. This avoids inventing labels that a dataset does not contain.
    task_losses = {}
    for task_name, spec in getattr(model, "task_specs", {}).items():
        kind = spec.get("kind", "binary")
        if kind == "multiclass":
            task_losses[task_name] = MaskedCrossEntropyLoss(spec.get("class_weight"))
        elif spec.get("focal", False):
            task_losses[task_name] = FocalLoss(
                gamma=spec.get("gamma", 2.0),
                pos_weight=spec.get("pos_weight"),
            )
        else:
            task_losses[task_name] = MaskedBCEWithLogitsLoss(spec.get("pos_weight"))

    if not task_losses:
        raise ValueError("Model must expose non-empty task_specs for masked biomarker training.")

    criterion = MultiTaskLoss(task_losses)
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
        monitor="loss",
        save_name="biomarker_ft_transformer.pt",
    )


if __name__ == "__main__":
    main()
