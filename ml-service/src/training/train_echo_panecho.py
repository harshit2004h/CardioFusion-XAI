"""Fine-tune pretrained PanEcho on EchoNet-Dynamic (EF, optionally EDV/ESV).

This mirrors the EchoNet-Dynamic fine-tuning experiment from the PanEcho paper:
the whole PanEcho video backbone is warm-started from ``panecho.pt`` and the
EF / LVEDV / LVESV heads are trained on EchoNet labels with MSE.

Run from ml-service/:

    python -m src.training.train_echo_panecho \
        --checkpoint checkpoints/echo/panecho/pretrained/v1/model.pt \
        --data-root data/raw/echo/echonet \
        --epochs 30 --batch-size 4

Outputs
  artifacts/03_training/echo/echonet/panecho_finetuning/v1/checkpoints/ (Trainer checkpoints)
  checkpoints/echo/echonet/panecho_finetuning/v1/model.pt
  artifacts/04_evaluation/echo/echonet/panecho_finetuning/v1/ (test metrics and predictions)

Tips
  * GPU strongly recommended. batch 4 x 16 frames x 224^2 needs ~10 GB with AMP.
    Lower --batch-size if you run out of memory.
  * --freeze-backbone-epochs N trains only the heads for the first N epochs
    (useful on small GPUs or as a quick linear-probe baseline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.datasets.panecho_echonet_dataset import build_panecho_loaders, load_echonet_records
from src.evaluation.evaluate_echo_panecho import predict_loader, summarize
from src.models.echo.panecho import (
    PANECHO_CLIP_LEN,
    build_panecho_for_finetuning,
    load_panecho,
)

from .trainer import EarlyStopping, Trainer, seed_everything

TARGET_KEYS = {"EF": "ef", "LVEDV": "edv", "LVESV": "esv"}


class EchoNetMultiTargetLoss(nn.Module):
    """Masked MSE per head, each divided by its training-set variance.

    Without scaling, EDV (variance ~ 2000 mL^2) would dominate EF (~ 150 %^2).
    Returns (total, {"loss_EF": ..., ...}) as the project Trainer expects.
    """

    def __init__(self, variances: dict[str, float]):
        super().__init__()
        self.variances = variances

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        total = None
        parts: dict[str, torch.Tensor] = {}
        for head, variance in self.variances.items():
            pred = outputs[head].float().reshape(-1)
            target = batch[TARGET_KEYS[head]].float().to(pred.device).reshape(-1)
            mask = torch.isfinite(target)
            if not mask.any():
                continue
            mse = torch.mean((pred[mask] - target[mask]) ** 2)
            parts[f"mse_{head}"] = mse.detach()
            term = mse / variance
            total = term if total is None else total + term
        if total is None:
            raise ValueError("Batch has no finite targets.")
        return total, parts


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = trainable
    model.encoder.time_encoder.pe.requires_grad = False  # always fixed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/echo/panecho/pretrained/v1/model.pt",
                   help="Pretrained PanEcho weights (panecho.pt or panecho.pt.zip)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Ablation: same architecture from ImageNet ConvNeXt, no PanEcho weights")
    p.add_argument("--data-root", default="data/raw/echo/echonet")
    p.add_argument("--manifest", default=None, help="Project manifest CSV instead of FileList.csv")
    p.add_argument("--targets", default="EF,LVEDV,LVESV",
                   help="Comma list of heads to train: EF, LVEDV, LVESV")
    p.add_argument("--artifact-dir", default="artifacts/03_training/echo/echonet/panecho_finetuning/v1")
    p.add_argument("--output", default="checkpoints/echo/echonet/panecho_finetuning/v1/model.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-scale", type=float, default=0.1,
                   help="Backbone LR = lr x this (heads use lr)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--fc-dropout", type=float, default=0.0)
    p.add_argument("--freeze-backbone-epochs", type=int, default=0)
    p.add_argument("--augment", action="store_true", help="Random zoom-out/crop/rotation (PanEcho-style)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-num-clips", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--resume", default=None,
                   help="Resume from an intermediate PanEcho .pt checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in TARGET_KEYS]
    if unknown:
        raise SystemExit(f"Unknown target(s) {unknown}; choose from {list(TARGET_KEYS)}")

    records = load_echonet_records(args.data_root if not args.manifest else None, manifest=args.manifest, project_root=Path.cwd())
    train_records = records[records["split"] == "train"]
    column = {"EF": "EF", "LVEDV": "EDV", "LVESV": "ESV"}
    variances = {}
    for head in targets:
        values = train_records[column[head]].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2 or values.var() < 1e-6:
            print(f"[warn] not enough variation in training labels for {head}; dropping it")
            continue
        variances[head] = float(values.var())
    if "EF" not in variances:
        raise SystemExit("EF labels are required.")
    targets = list(variances)
    print(f"Training heads: {targets}  (train videos: {len(train_records)})")

    loaders = build_panecho_loaders(
        data_root=args.data_root if not args.manifest else None,
        manifest=args.manifest,
        project_root=Path.cwd(),
        batch_size=args.batch_size,
        seed=args.seed,
        eval_num_clips=1,  # fast single-clip validation during training
        num_workers=args.num_workers,
        augment=args.augment,
    )

    model = build_panecho_for_finetuning(
        None if args.no_pretrained else args.checkpoint,
        tasks=targets,
        fc_dropout=args.fc_dropout,
    )

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and not n.endswith("time_encoder.pe")]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        if "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        print(f"Resumed PanEcho checkpoint: {args.resume}")

    artifact_dir = Path(args.artifact_dir)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=EchoNetMultiTargetLoss(variances),
        device=args.device,
        scheduler=scheduler,
        checkpoint_dir=artifact_dir / "checkpoints",
        grad_clip_norm=1.0,
    )

    history: list[dict] = []
    ckpt_dir = artifact_dir / "checkpoints"
    stage_files: list[Path] = []
    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        print(f"Stage 1: heads only for {args.freeze_backbone_epochs} epoch(s)")
        history += trainer.fit(loaders["train"], loaders.get("val"), epochs=args.freeze_backbone_epochs,
                               monitor="loss", save_name="panecho_echonet_stage1.pt")
        stage_files.append(ckpt_dir / "panecho_echonet_stage1.pt")
        set_backbone_trainable(model, True)
    remaining = args.epochs - args.freeze_backbone_epochs
    if remaining > 0:
        print(f"Stage 2: full fine-tuning for up to {remaining} epoch(s)")
        history += trainer.fit(loaders["train"], loaders.get("val"), epochs=remaining,
                               early_stopping=EarlyStopping(args.patience, mode="min"),
                               monitor="loss", save_name="panecho_echonet_stage2.pt")
        stage_files.append(ckpt_dir / "panecho_echonet_stage2.pt")
    (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Each stage keeps its own best-by-validation checkpoint; take the better of the two.
    candidates = [torch.load(f, map_location="cpu", weights_only=True) for f in stage_files if f.exists()]
    if not candidates:
        raise SystemExit("Training produced no checkpoint.")
    best = min(candidates, key=lambda c: c.get("metrics", {}).get("val_loss", c.get("metrics", {}).get("train_loss", np.inf)))

    # Re-save the best weights in the inference format the API loads.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best["model_state_dict"],
            "panecho_tasks": targets,
            "clip_len": PANECHO_CLIP_LEN,
            "source": "PanEcho (Holste et al., JAMA 2025) fine-tuned on EchoNet-Dynamic by CardioFusion-XAI",
            "pretrained_init": not args.no_pretrained,
            "epoch": best.get("epoch"),
            "val_metrics": best.get("metrics", {}),
            "license": "PanEcho weights: CC BY-NC-SA 4.0 (non-commercial)",
        },
        output,
    )
    print(f"Inference checkpoint written to {output}")

    if args.skip_test or "test" not in loaders:
        return
    device = trainer.device
    eval_model = load_panecho(output, tasks="auto", device=device)
    test_loaders = build_panecho_loaders(
        data_root=args.data_root if not args.manifest else None,
        manifest=args.manifest,
        project_root=Path.cwd(),
        batch_size=max(1, args.batch_size // 2),
        seed=args.seed,
        eval_num_clips=args.eval_num_clips,
        num_workers=args.num_workers,
    )
    predictions = predict_loader(eval_model, test_loaders["test"], device)
    metrics = summarize(predictions)
    out_dir = Path("artifacts/04_evaluation/echo/echonet/panecho_finetuning/v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({"checkpoint": str(output), "split": "test", **metrics}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
