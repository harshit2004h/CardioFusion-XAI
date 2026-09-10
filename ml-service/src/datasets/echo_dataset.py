from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EchoNetDataset(Dataset):
    def __init__(
        self,
        manifest,
        split="test",
        num_frames=32,
        frame_stride=4,
        image_size=112,
        project_root=None,
    ):
        if isinstance(manifest, (str, Path)):
            manifest = pd.read_csv(manifest)

        self.manifest = manifest.copy()

        if "split" in self.manifest.columns:
            self.manifest = self.manifest[
                self.manifest["split"].astype(str).str.lower() == split.lower()
            ].reset_index(drop=True)

        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.split = split
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else None
        )

        required = ["video_path", "EF"]

        missing = [
            col for col in required
            if col not in self.manifest.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required manifest columns: {missing}"
            )

    def __len__(self):
        return len(self.manifest)

    def _load_video(self, video_path):
        video_path = Path(video_path)

        if not video_path.is_absolute() and self.project_root is not None:
            video_path = self.project_root / video_path

        video_path = video_path.resolve()
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(
                f"Video contains no frames: {video_path}"
            )

        required_frames = (
            self.num_frames * self.frame_stride
        )

        if total_frames >= required_frames:
            max_start = total_frames - required_frames
            if self.split.lower() == "train":
                start = np.random.randint(0, max_start + 1)
            else:
                start = max_start // 2

            indices = start + np.arange(self.num_frames) * self.frame_stride

        else:
            indices = np.linspace(
                0,
                total_frames - 1,
                self.num_frames,
                dtype=np.int64,
            )

        frames = []

        for index in indices:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index),
            )

            success, frame = cap.read()

            if not success:
                cap.release()
                raise RuntimeError(
                    f"Could not read frame {index} "
                    f"from {video_path}"
                )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            frame = cv2.resize(
                frame,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )

            frame = (
                frame.astype(np.float32)
                / 255.0
            )

            frame = np.stack(
                [frame, frame, frame],
                axis=0,
            )

            frames.append(frame)

        cap.release()

        video = np.stack(
            frames,
            axis=1,
        )

        return video
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        video_path = Path(row["video_path"])

        if not video_path.is_absolute() and self.project_root is not None:
            video_path = self.project_root / video_path

        video_path = video_path.resolve()

        video = self._load_video(video_path)

        video = torch.tensor(
            video,
            dtype=torch.float32,
        )

        ef = torch.tensor(
            float(row["EF"]),
            dtype=torch.float32,
        )

        return {
            "video": video,
            "ef": ef,
            "video_path": str(video_path),
        }