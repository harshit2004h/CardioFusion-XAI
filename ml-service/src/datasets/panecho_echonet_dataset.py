"""EchoNet-Dynamic dataset in PanEcho's input format (16 x 3 x 224 x 224 clips).

Works with either source the project already has:

1. The raw EchoNet release: ``data/raw/echo/echonet/FileList.csv`` + ``Videos/*.avi``
   (columns FileName, EF, EDV, ESV, NumberOfFrames, Split).
2. The project manifest written by ``notebooks/02_preprocessing/03_echonet_dynamic_preprocessing.ipynb``:
   ``data/processed/echo/echonet/echonet_manifest.csv`` (columns video_path, EF, split, ...).

Targets returned per item (NaN when unavailable):
  ef  -> PanEcho ``EF`` head   (%)
  edv -> PanEcho ``LVEDV`` head (mL)
  esv -> PanEcho ``LVESV`` head (mL)
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.echo_processing.panecho_reader import decode_video
from src.models.echo.panecho import PANECHO_CLIP_LEN
from src.preprocessing.echo.panecho_transforms import (
    PanEchoClipTransform,
    clip_indices,
    clip_starts,
)

_SPLIT_ALIASES = {"train": "train", "val": "val", "validation": "val", "valid": "val", "test": "test"}


def _first_column(frame: pd.DataFrame, *names: str) -> str | None:
    lowered = {c.lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def load_echonet_records(
    data_root: str | Path | None = None,
    manifest: str | Path | pd.DataFrame | None = None,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Normalise FileList.csv or the project manifest to:
    ``video_path, EF, EDV, ESV, num_frames, split``.
    """
    project_root = Path(project_root).resolve() if project_root else None

    if manifest is not None:
        records = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        path_col = _first_column(records, "video_path", "path")
        if path_col is None:
            raise ValueError("Manifest needs a 'video_path' column.")
        paths = records[path_col].astype(str).map(Path)
        if project_root is not None:
            paths = paths.map(lambda p: p if p.is_absolute() else project_root / p)
    else:
        if data_root is None:
            raise ValueError("Pass data_root (folder with FileList.csv and Videos/) or manifest.")
        data_root = Path(data_root)
        records = pd.read_csv(data_root / "FileList.csv")
        name_col = _first_column(records, "FileName", "Filename")
        if name_col is None:
            raise ValueError("FileList.csv has no FileName column.")
        videos_dir = data_root / "Videos"

        def video_path(name: str) -> Path:
            name = str(name).strip()
            return videos_dir / (name if Path(name).suffix else f"{name}.avi")

        paths = records[name_col].map(video_path)

    out = pd.DataFrame({"video_path": paths.map(str)})
    for target, candidates in {"EF": ("EF",), "EDV": ("EDV",), "ESV": ("ESV",)}.items():
        col = _first_column(records, *candidates)
        out[target] = pd.to_numeric(records[col], errors="coerce") if col else np.nan
    frames_col = _first_column(records, "NumberOfFrames", "frame_count", "num_frames")
    out["num_frames"] = pd.to_numeric(records[frames_col], errors="coerce") if frames_col else np.nan
    split_col = _first_column(records, "split", "Split")
    out["split"] = records[split_col].astype(str).str.strip().str.lower().map(_SPLIT_ALIASES) if split_col else "train"

    out = out[out["EF"].between(0, 100)].reset_index(drop=True)
    return out


class PanEchoEchoNetDataset(Dataset):
    """One item = one EchoNet video.

    ``video`` is [3, T, 224, 224] when ``num_clips == 1`` (training / quick validation)
    and [num_clips, 3, T, 224, 224] otherwise (multi-clip test-time averaging).
    """

    def __init__(
        self,
        records: pd.DataFrame,
        split: str | None = "train",
        clip_len: int = PANECHO_CLIP_LEN,
        num_clips: int = 1,
        augment: bool = False,
        hflip_p: float = 0.0,
        seed: int | None = None,
        random_clips: bool | None = None,
    ):
        if split is not None:
            split = _SPLIT_ALIASES.get(split.lower(), split.lower())
            records = records[records["split"] == split]
        self.records = records.reset_index(drop=True)
        if self.records.empty:
            raise ValueError(f"No EchoNet records for split={split!r}.")
        self.split = split
        self.clip_len = clip_len
        self.num_clips = num_clips
        # PanEcho draws a random clip start for every training item; evaluation is deterministic.
        self.random_clips = (split == "train") if random_clips is None else random_clips
        self.transform = PanEchoClipTransform(train=augment, hflip_p=hflip_p)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def _rng(self, idx: int) -> np.random.Generator | None:
        if not self.random_clips:
            return None
        # Distinct stream per worker/item; reproducible if seed is set.
        info = torch.utils.data.get_worker_info()
        base = self.seed if self.seed is not None else int(torch.initial_seed() % 2**31)
        return np.random.default_rng([base, idx, info.id if info else 0, int(torch.randint(0, 2**31, (1,)))])

    def _total_frames(self, row: pd.Series) -> int:
        n = row.get("num_frames")
        if n is not None and np.isfinite(n) and n > 0:
            return int(n)
        import cv2

        capture = cv2.VideoCapture(row["video_path"])
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records.iloc[idx]
        total = self._total_frames(row)
        if total <= 0:
            raise RuntimeError(f"Could not determine frame count: {row['video_path']}")

        starts = clip_starts(total, self.clip_len, self.num_clips, random=self.random_clips, rng=self._rng(idx))
        per_clip = [clip_indices(s, total, self.clip_len) for s in starts]
        frames_by_index, _ = decode_video(row["video_path"], keep=np.concatenate(per_clip))

        # FileList frame counts can overstate what OpenCV decodes; carry the last decoded frame forward.
        available = sorted(frames_by_index)

        def frame_at(i: int) -> np.ndarray:
            if i in frames_by_index:
                return frames_by_index[i]
            return frames_by_index[available[max(bisect_right(available, i) - 1, 0)]]

        clips = [self.transform(np.stack([frame_at(int(i)) for i in indices], axis=0)) for indices in per_clip]
        video = clips[0] if self.num_clips == 1 else torch.stack(clips, dim=0)

        return {
            "video": video,
            "ef": torch.tensor(float(row["EF"]), dtype=torch.float32),
            "edv": torch.tensor(float(row["EDV"]), dtype=torch.float32),
            "esv": torch.tensor(float(row["ESV"]), dtype=torch.float32),
            "video_path": str(row["video_path"]),
        }


def build_panecho_loaders(
    data_root: str | Path | None = None,
    batch_size: int = 8,
    seed: int = 42,
    manifest: str | Path | None = None,
    project_root: str | Path | None = None,
    clip_len: int = PANECHO_CLIP_LEN,
    eval_num_clips: int = 1,
    num_workers: int = 2,
    augment: bool = False,
    hflip_p: float = 0.0,
    **_ignored: Any,
) -> dict[str, DataLoader]:
    """Train/val/test loaders. Extra kwargs (e.g. num_frames/frame_stride from the
    R(2+1)D script) are ignored because PanEcho has a fixed input contract."""
    records = load_echonet_records(data_root=data_root, manifest=manifest, project_root=project_root)
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        if not (records["split"] == split).any():
            continue
        train = split == "train"
        dataset = PanEchoEchoNetDataset(
            records,
            split=split,
            clip_len=clip_len,
            num_clips=1 if train else eval_num_clips,
            augment=train and augment,
            hflip_p=hflip_p,
            seed=seed if train else None,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if train or eval_num_clips == 1 else max(1, batch_size // eval_num_clips),
            shuffle=train,
            drop_last=train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=generator if train else None,
        )
    return loaders


__all__ = ["load_echonet_records", "PanEchoEchoNetDataset", "build_panecho_loaders"]
