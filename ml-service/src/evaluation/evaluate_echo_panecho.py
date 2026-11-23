"""Evaluate PanEcho on EchoNet-Dynamic (zero-shot or after project fine-tuning).

Examples (run from ml-service/):

    # Zero-shot: the published PanEcho EF head, no EchoNet training at all
    python -m src.evaluation.evaluate_echo_panecho \
        --checkpoint checkpoints/echo/panecho/pretrained/v1/model.pt \
        --data-root data/raw/echo/echonet --split test --tag zero_shot

    # Fine-tuned checkpoint produced by src.training.train_echo_panecho
    python -m src.evaluation.evaluate_echo_panecho \
        --checkpoint checkpoints/echo/echonet/panecho_finetuning/v1/model.pt \
        --data-root data/raw/echo/echonet --split test --tag finetuned

Outputs go to ``artifacts/04_evaluation/echo/echonet/panecho/<tag>/``:
  predictions.csv, metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.panecho_echonet_dataset import PanEchoEchoNetDataset, load_echonet_records
from src.evaluation.metrics import binary_metrics, ef_threshold_metrics, regression_metrics
from src.models.echo.panecho import PanEchoModel, load_panecho

TARGETS = {"EF": "ef", "LVEDV": "edv", "LVESV": "esv"}


@torch.inference_mode()
def predict_loader(model: PanEchoModel, loader: DataLoader, device: torch.device, amp: bool = True) -> pd.DataFrame:
    """Run the model over a loader; multi-clip items ([B, K, 3, T, H, W]) are averaged per video."""
    model.eval()
    was_activated = model.activations
    model.activations = True
    rows: list[dict] = []
    heads = [name for name in TARGETS if name in model.task_names]
    try:
        for batch in tqdm(loader, desc="PanEcho inference", leave=False):
            video = batch["video"].to(device, non_blocking=True)
            multi_clip = video.ndim == 6
            if multi_clip:
                b, k = video.shape[:2]
                video = video.flatten(0, 1)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                outputs = model(video)
            for i in range(len(batch["video_path"])):
                row = {"video_path": batch["video_path"][i]}
                for head in heads:
                    values = outputs[head].float().reshape(-1)
                    pred = values.view(b, k)[i].mean() if multi_clip else values[i]
                    row[f"{head}_pred"] = float(pred)
                    row[f"{head}_true"] = float(batch[TARGETS[head]][i])
                rows.append(row)
    finally:
        model.activations = was_activated
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {"n_videos": int(len(predictions))}
    for head in TARGETS:
        if f"{head}_pred" not in predictions:
            continue
        y_true, y_pred = predictions[f"{head}_true"].to_numpy(), predictions[f"{head}_pred"].to_numpy()
        if np.isfinite(y_true).sum() == 0:
            continue
        metrics.update({f"{head}_{k}": v for k, v in regression_metrics(y_true, y_pred).items()})
    if "EF_pred" in predictions:
        y_true, y_pred = predictions["EF_true"].to_numpy(), predictions["EF_pred"].to_numpy()
        metrics.update(ef_threshold_metrics(y_true, y_pred))
        # Classification view used by the registry: lower predicted EF = higher risk.
        for threshold in (50, 40):
            # Monotone in -EF (AUROC uses only the ranking); score >= 0.5 exactly when predicted
            # EF <= threshold, so F1/precision/recall match the registry EF cut-off rule.
            score = 1.0 / (1.0 + np.exp((y_pred - threshold) / 5.0))
            for name, value in binary_metrics((y_true < threshold).astype(float), score).items():
                metrics[f"EF_<{threshold}_{name}"] = value
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/echo/panecho/pretrained/v1/model.pt")
    p.add_argument("--data-root", default="data/raw/echo/echonet")
    p.add_argument("--manifest", default=None, help="Use the project manifest CSV instead of FileList.csv")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-clips", type=int, default=4, help="Clips averaged per video (PanEcho used 4)")
    p.add_argument("--batch-size", type=int, default=2, help="Videos per batch (each has --num-clips clips)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N videos (quick check)")
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default=None, help="Sub-folder name for outputs")
    p.add_argument("--output-root", default="artifacts/04_evaluation/echo/echonet/panecho")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    model = load_panecho(args.checkpoint, tasks="auto", device=device)
    keep = [name for name in TARGETS if name in model.task_names]
    if not keep:
        raise SystemExit("Checkpoint has no EF/LVEDV/LVESV heads to evaluate.")
    # Only run the heads we can score (much faster than all 40).
    for task in list(model.tasks):
        if task.name not in keep:
            delattr(model, f"{task.name}_head")
    model.tasks = tuple(t for t in model.tasks if t.name in keep)

    records = load_echonet_records(args.data_root if not args.manifest else None, manifest=args.manifest, project_root=Path.cwd())
    dataset = PanEchoEchoNetDataset(records, split=args.split, num_clips=args.num_clips, augment=False)
    if args.limit:
        dataset.records = dataset.records.head(args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    predictions = predict_loader(model, loader, device)
    metrics = summarize(predictions)

    tag = args.tag or Path(args.checkpoint).stem
    out_dir = Path(args.output_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "split": args.split, "num_clips": args.num_clips, **metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
