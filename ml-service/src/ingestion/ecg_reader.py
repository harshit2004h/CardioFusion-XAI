"""
PTB-XL ECG data reader.

This module handles raw ECG ingestion from the PTB-XL dataset.

Responsibilities:
    - Locate PTB-XL files.
    - Load PTB-XL metadata.
    - Load SCP diagnostic statements.
    - Read individual ECG waveform files.

Preprocessing and model-specific transformations are handled
by the preprocessing pipeline, not this module.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import PTBXL_DIR


# ============================================================
# PTB-XL READER
# ============================================================

class PTBXLECGReader:
    """
    Reader for the PTB-XL ECG dataset.
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the PTB-XL reader.

        Args:
            dataset_dir:
                Path to the PTB-XL dataset directory.
                If omitted, the project's default PTB-XL
                directory is used.
        """

        if dataset_dir is None:
            dataset_dir = PTBXL_DIR

        self.dataset_dir = Path(dataset_dir)

        self.database_file = (
            self.dataset_dir / "ptbxl_database.csv"
        )

        self.scp_file = (
            self.dataset_dir / "scp_statements.csv"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_dataset(self) -> None:
        """
        Validate that the essential PTB-XL files exist.

        Raises:
            FileNotFoundError:
                If required files are missing.
        """

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"PTB-XL directory does not exist: "
                f"{self.dataset_dir}"
            )

        if not self.database_file.exists():
            raise FileNotFoundError(
                f"PTB-XL database file not found: "
                f"{self.database_file}"
            )

        if not self.scp_file.exists():
            raise FileNotFoundError(
                f"PTB-XL SCP statements file not found: "
                f"{self.scp_file}"
            )

    # ========================================================
    # METADATA
    # ========================================================

    def load_database(
        self,
    ) -> pd.DataFrame:
        """
        Load ptbxl_database.csv.

        Returns:
            Pandas DataFrame containing PTB-XL metadata.
        """

        if not self.database_file.exists():
            raise FileNotFoundError(
                f"PTB-XL database file not found: "
                f"{self.database_file}"
            )

        dataframe = pd.read_csv(
            self.database_file,
            index_col="ecg_id",
        )

        return dataframe

    def load_scp_statements(
        self,
    ) -> pd.DataFrame:
        """
        Load scp_statements.csv.

        Returns:
            Pandas DataFrame containing SCP statements.
        """

        if not self.scp_file.exists():
            raise FileNotFoundError(
                f"SCP statements file not found: "
                f"{self.scp_file}"
            )

        dataframe = pd.read_csv(
            self.scp_file,
            index_col=0,
        )

        return dataframe

    # ========================================================
    # RECORD PATH
    # ========================================================

    def get_record_path(
        self,
        filename: str,
    ) -> Path:
        """
        Resolve a PTB-XL waveform filename.

        PTB-XL metadata commonly stores paths such as:

            records100/00000/00001_lr

        or:

            records500/00000/00001_hr

        The reader also accepts paths with an extension.

        Args:
            filename:
                Relative waveform path.

        Returns:
            Resolved waveform path.

        Raises:
            FileNotFoundError:
                If the waveform cannot be located.
        """

        filename = str(filename)

        relative_path = Path(filename)

        direct_path = (
            self.dataset_dir / relative_path
        )

        if direct_path.exists():
            return direct_path

        # Try WFDB's normal extension.
        wfdb_path = Path(
            str(direct_path) + ".hea"
        )

        if wfdb_path.exists():
            return direct_path

        # Try .dat.
        dat_path = Path(
            str(direct_path) + ".dat"
        )

        if dat_path.exists():
            return direct_path

        raise FileNotFoundError(
            f"ECG record could not be found: "
            f"{filename}"
        )

    # ========================================================
    # ECG READING
    # ========================================================

    def read_record(
        self,
        record_path: str | Path,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Read an ECG waveform using WFDB.

        Args:
            record_path:
                Relative or absolute path to the ECG record.

        Returns:
            Tuple containing:

                signal:
                    NumPy array with shape
                    [samples, leads].

                metadata:
                    WFDB record metadata.
        """

        try:
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "The 'wfdb' package is required to read "
                "PTB-XL ECG records. Install it with "
                "'pip install wfdb'."
            ) from exc

        record_path = Path(record_path)

        if not record_path.is_absolute():
            record_path = (
                self.dataset_dir / record_path
            )

        record_path_without_extension = Path(
            str(record_path)
        )

        if record_path_without_extension.suffix:
            record_path_without_extension = (
                record_path_without_extension.with_suffix("")
            )

        if not Path(
            str(record_path_without_extension) + ".hea"
        ).exists():

            raise FileNotFoundError(
                f"ECG header file not found for record: "
                f"{record_path_without_extension}"
            )

        record = wfdb.rdrecord(
            str(record_path_without_extension)
        )

        signal = np.asarray(
            record.p_signal,
            dtype=np.float32,
        )

        metadata: dict[str, Any] = {
            "record_name": record.record_name,
            "sampling_frequency": record.fs,
            "num_samples": record.sig_len,
            "num_leads": record.n_sig,
            "lead_names": record.sig_name,
        }

        return signal, metadata

    # ========================================================
    # RECORD FROM DATABASE
    # ========================================================

    def read_ecg_by_id(
        self,
        ecg_id: int,
        sampling_rate: int = 100,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Read an ECG using its PTB-XL ECG ID.

        Args:
            ecg_id:
                PTB-XL ECG identifier.

            sampling_rate:
                Sampling-rate version to use.

                100 -> records100
                500 -> records500

        Returns:
            ECG signal and metadata.
        """

        if sampling_rate not in {100, 500}:
            raise ValueError(
                "sampling_rate must be either 100 or 500."
            )

        database = self.load_database()

        if ecg_id not in database.index:
            raise KeyError(
                f"ECG ID {ecg_id} was not found "
                "in ptbxl_database.csv."
            )

        row = database.loc[ecg_id]

        if sampling_rate == 100:
            column_name = "filename_lr"
        else:
            column_name = "filename_hr"

        if column_name not in database.columns:
            raise KeyError(
                f"Column '{column_name}' not found "
                "in PTB-XL metadata."
            )

        record_filename = row[column_name]

        signal, metadata = self.read_record(
            record_filename
        )

        metadata["ecg_id"] = int(ecg_id)

        return signal, metadata

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    def summary(self) -> dict[str, Any]:
        """
        Return a basic summary of the PTB-XL dataset.
        """

        database = self.load_database()

        return {
            "dataset": "PTB-XL",
            "directory": str(self.dataset_dir),
            "num_records": len(database),
            "columns": list(database.columns),
        }