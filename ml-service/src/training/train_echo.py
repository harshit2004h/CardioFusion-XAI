"""Train the EchoNet-Dynamic R(2+1)D-18 LVEF regressor.

Project preprocessing contract:
- sample 32 frames at a stride of 4
- resize to 112x112
- grayscale duplicated to 3 channels
- predict EF% with one regression head
- derive LV systolic dysfunction (<50%) and reduced-EF/HFrEF (<40%)
  after prediction; do not train separate LVH/RVH heads from EchoNet.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn

from .trainer import EarlyStopping, Trainer, seed_everything


def load_factory(spec: str):
    module_name, _, attr = spec.rpartition(":")
    if not module_name:
        raise ValueError(f"Expected MODULE:FUNCTION, got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


class EFRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs, batch):
        if isinstance(outputs, dict):
            pred = outputs.get("ef", outputs.get("lvef", outputs.get("logits")))
        else:
            pred = outputs
        target = batch["ef"].float().to(pred.device).reshape_as(pred)
        return self.loss(pred, target)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--factory", default="src.data.echo_dataset:build_echo_dataloaders")
    p.add_argument("--model-factory", default="src.models.echo:build_echo_model")
    p.add_argument("--data-root", required=True)
    p.add_argument("--artifact-dir", default="artifacts/echo")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
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
        num_frames=32,
        frame_stride=4,
        image_size=112,
    )
    model = build_model(
        pretrained=True,
        in_channels=3,
        output_dim=1,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=EFRegressionLoss(),
        device=args.device,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        ),
        checkpoint_dir=Path(args.artifact_dir) / "checkpoints",
        grad_clip_norm=1.0,
    )

    trainer.fit(
        loaders["train"],
        loaders.get("val"),
        epochs=args.epochs,
        early_stopping=EarlyStopping(args.patience, mode="min"),
        monitor="loss",
        save_name="echonet_r2plus1d18_lvef.pt",
    )


if __name__ == "__main__":
    main()
