"""
PTB-XL ECG Dataset.

Supports:
1. Existing in-memory NumPy arrays.
2. Lazy loading of preprocessed .npy files from the PTB-XL manifest.

Expected preprocessing output:
    signal shape = (1000, 12)

Returned PyTorch tensor:
    signal shape = (12, 1000)

Targets:
    diagnostic -> NORM, MI, STTC, CD, HYP
    rhythm     -> project-defined rhythm labels
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PTBXLDataset(Dataset):
    """
    Dataset for preprocessed PTB-XL ECG records.

    Backward-compatible constructor:

        PTBXLDataset(
            signals,
            labels_df,
            superclass_cols,
            rhythm_cols,
            record_paths=None,
        )

    For lazy loading:

        PTBXLDataset(
            signals=None,
            labels_df=labels_df,
            superclass_cols=...,
            rhythm_cols=...,
            record_paths=processed_paths,
            project_root=PROJECT_ROOT,
        )
    """

    def __init__(
        self,
        signals: Optional[np.ndarray],
        labels_df,
        superclass_cols: Sequence[str],
        rhythm_cols: Sequence[str],
        record_paths=None,
        project_root=None,
        signal_length: int = 1000,
        num_leads: int = 12,
    ):
        self.signals = signals

        self.labels_df = (
            labels_df.reset_index(drop=True)
        )

        self.superclass_cols = list(
            superclass_cols
        )

        self.rhythm_cols = list(
            rhythm_cols
        )

        self.signal_length = int(
            signal_length
        )

        self.num_leads = int(
            num_leads
        )

        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else None
        )

        # -------------------------------------------------
        # Record paths
        # -------------------------------------------------

        if record_paths is not None:

            self.record_paths = list(
                record_paths
            )

        elif signals is not None:

            self.record_paths = [
                str(index)
                for index in range(
                    len(signals)
                )
            ]

        else:

            raise ValueError(
                "Either signals or record_paths "
                "must be provided."
            )

        # -------------------------------------------------
        # Length checks
        # -------------------------------------------------

        if signals is not None:

            if len(signals) != len(
                self.labels_df
            ):

                raise ValueError(
                    "signals and labels_df must "
                    "have the same length."
                )

        if len(self.record_paths) != len(
            self.labels_df
        ):

            raise ValueError(
                "record_paths and labels_df "
                "must have the same length."
            )

        # -------------------------------------------------
        # Validate target columns
        # -------------------------------------------------

        missing_superclass = [
            column
            for column in self.superclass_cols
            if column not in self.labels_df.columns
        ]

        missing_rhythm = [
            column
            for column in self.rhythm_cols
            if column not in self.labels_df.columns
        ]

        if missing_superclass:

            raise ValueError(
                "Missing diagnostic target columns: "
                f"{missing_superclass}"
            )

        if missing_rhythm:

            raise ValueError(
                "Missing rhythm target columns: "
                f"{missing_rhythm}"
            )

        # -------------------------------------------------
        # Cache labels as float32 matrices
        # -------------------------------------------------

        self.superclass_targets = (
            self.labels_df[
                self.superclass_cols
            ]
            .values
            .astype(
                np.float32
            )
        )

        if self.rhythm_cols:

            self.rhythm_targets = (
                self.labels_df[
                    self.rhythm_cols
                ]
                .values
                .astype(
                    np.float32
                )
            )

        else:

            self.rhythm_targets = np.empty(
                (
                    len(self.labels_df),
                    0,
                ),
                dtype=np.float32,
            )

    def __len__(self):

        return len(
            self.labels_df
        )

    def _load_signal(
        self,
        idx: int,
    ) -> np.ndarray:

        # -------------------------------------------------
        # In-memory mode
        # -------------------------------------------------

        if self.signals is not None:

            signal = np.asarray(
                self.signals[idx]
            )

        # -------------------------------------------------
        # Lazy .npy mode
        # -------------------------------------------------

        else:

            if self.project_root is None:

                raise ValueError(
                    "project_root is required "
                    "when using lazy record_paths."
                )

            path = (
                self.project_root
                / self.record_paths[idx]
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Processed ECG file not found:\n"
                    f"{path}"
                )

            signal = np.load(
                path,
                allow_pickle=False,
            )

        # -------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------

        expected_shape = (
            self.signal_length,
            self.num_leads,
        )

        if signal.shape != expected_shape:

            raise ValueError(
                f"Unexpected ECG shape "
                f"{signal.shape}. "
                f"Expected {expected_shape}."
            )

        if not np.isfinite(
            signal
        ).all():

            raise ValueError(
                "ECG contains NaN or Inf values."
            )

        if signal.dtype != np.float32:

            signal = signal.astype(
                np.float32
            )

        return signal

    def __getitem__(
        self,
        idx: int,
    ):

        signal = self._load_signal(
            idx
        )

        # ---------------------------------------------
        # Preprocessed:
        #     [time, leads]
        #
        # Conv1D:
        #     [channels, time]
        #
        # [1000, 12] -> [12, 1000]
        # ---------------------------------------------

        signal_tensor = (
            torch.from_numpy(
                signal
            )
            .transpose(
                0,
                1
            )
            .contiguous()
        )

        diagnostic_tensor = (
            torch.from_numpy(
                self.superclass_targets[
                    idx
                ]
            )
        )

        rhythm_tensor = (
            torch.from_numpy(
                self.rhythm_targets[
                    idx
                ]
            )
        )

        return {
            "signal":
                signal_tensor,

            "diagnostic":
                diagnostic_tensor,

            "rhythm":
                rhythm_tensor,

            "record_path":
                self.record_paths[idx],
        }