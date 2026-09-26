"""
PTB-XL DataLoader factory.

Reads the manifest produced by the existing
PTB-XL preprocessing pipeline and lazily loads
the processed .npy ECG arrays.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from .ecg_dataset import PTBXLDataset


DIAGNOSTIC_LABELS = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


def build_ptbxl_loaders(
    data_root,
    batch_size=32,
    seed=42,
    preprocessing_version=(
        "100hz_record_zscore"
    ),
    num_workers=0,
    pin_memory=True,
):
    """
    Build PTB-XL train/validation/test loaders.

    data_root should be the project root:

        ml-service/

    Expected structure:

        data/
          processed/
            ecg/
              ptbxl/
                <preprocessing_version>/
                  manifests/
                    ptbxl_manifest.csv
                  waveforms/
                    *.npy
    """

    project_root = Path(
        data_root
    ).resolve()

    processed_dir = (
        project_root
        / "data"
        / "processed"
        / "ecg"
        / "ptbxl"
        / preprocessing_version
    )

    manifest_path = (
        processed_dir
        / "manifests"
        / "ptbxl_manifest.csv"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"PTB-XL manifest not found:\n"
            f"{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    diagnostic_targets = [
        column
        for column in DIAGNOSTIC_LABELS
        if column in manifest.columns
    ]

    if len(diagnostic_targets) != 5:

        raise ValueError(
            "Expected all five PTB-XL "
            "diagnostic targets:\n"
            f"{DIAGNOSTIC_LABELS}\n"
            f"Found: {diagnostic_targets}"
        )

    rhythm_targets = [
        column
        for column in manifest.columns
        if column.startswith(
            "RHYTHM_"
        )
    ]

    required_columns = [
        "processed_path",
        "patient_id",
        "ecg_id",
        "split",
    ]

    missing = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing:

        raise ValueError(
            f"Missing manifest columns: {missing}"
        )

    # -------------------------------------------------------
    # Create split DataFrames
    # -------------------------------------------------------

    train_df = (
        manifest[
            manifest["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        manifest[
            manifest["split"] == "val"
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        manifest[
            manifest["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    # -------------------------------------------------------
    # Patient-leakage check
    # -------------------------------------------------------

    train_patients = set(
        train_df["patient_id"]
    )

    val_patients = set(
        val_df["patient_id"]
    )

    test_patients = set(
        test_df["patient_id"]
    )

    if train_patients & val_patients:

        raise RuntimeError(
            "Patient leakage detected "
            "between train and validation."
        )

    if train_patients & test_patients:

        raise RuntimeError(
            "Patient leakage detected "
            "between train and test."
        )

    if val_patients & test_patients:

        raise RuntimeError(
            "Patient leakage detected "
            "between validation and test."
        )

    # -------------------------------------------------------
    # Dataset helper
    # -------------------------------------------------------

    def make_dataset(
        dataframe,
    ):

        return PTBXLDataset(

            signals=None,

            labels_df=dataframe,

            superclass_cols=
                diagnostic_targets,

            rhythm_cols=
                rhythm_targets,

            record_paths=
                dataframe[
                    "processed_path"
                ].tolist(),

            project_root=
                project_root,

            signal_length=1000,

            num_leads=12,
        )

    # -------------------------------------------------------
    # Datasets
    # -------------------------------------------------------

    train_dataset = make_dataset(
        train_df
    )

    val_dataset = make_dataset(
        val_df
    )

    test_dataset = make_dataset(
        test_df
    )

    # -------------------------------------------------------
    # Loaders
    # -------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=pin_memory,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory,
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=pin_memory,
    )

    return {

        "train":
            train_loader,

        "val":
            val_loader,

        "test":
            test_loader,

        "train_df":
            train_df,

        "val_df":
            val_df,

        "test_df":
            test_df,

        "diagnostic_targets":
            diagnostic_targets,

        "rhythm_targets":
            rhythm_targets,
    }