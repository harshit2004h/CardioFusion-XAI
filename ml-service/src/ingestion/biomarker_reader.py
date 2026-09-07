"""
Clinical biomarker dataset reader.

Handles ingestion of tabular clinical datasets used by
CardioFusion-XAI, including:

    - ZHEEN
    - UCI Heart Failure Clinical Records

Preprocessing such as imputation, scaling, feature selection,
and train/validation/test splitting is handled elsewhere.
"""

from pathlib import Path

import pandas as pd

from src.utils.paths import (
    UCI_HEART_FAILURE_DIR,
    ZHEEN_DIR,
)


# ============================================================
# BIOMARKER READER
# ============================================================

class BiomarkerReader:
    """
    Reader for tabular clinical biomarker datasets.
    """

    def __init__(
        self,
        zheen_dir: str | Path | None = None,
        uci_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the biomarker reader.

        Args:
            zheen_dir:
                Directory containing the ZHEEN dataset.

            uci_dir:
                Directory containing the UCI Heart Failure
                dataset.
        """

        self.zheen_dir = Path(
            zheen_dir
            if zheen_dir is not None
            else ZHEEN_DIR
        )

        self.uci_dir = Path(
            uci_dir
            if uci_dir is not None
            else UCI_HEART_FAILURE_DIR
        )

    # ========================================================
    # CSV DISCOVERY
    # ========================================================

    @staticmethod
    def find_csv_files(
        directory: Path,
    ) -> list[Path]:
        """
        Find CSV files recursively in a directory.

        Args:
            directory:
                Dataset directory.

        Returns:
            Sorted list of CSV paths.
        """

        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: "
                f"{directory}"
            )

        return sorted(
            directory.rglob("*.csv")
        )

    # ========================================================
    # GENERIC CSV READER
    # ========================================================

    @staticmethod
    def read_csv(
        csv_path: str | Path,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Read a CSV file into a DataFrame.

        Args:
            csv_path:
                CSV file path.

            **kwargs:
                Additional arguments passed to pandas.read_csv.

        Returns:
            Pandas DataFrame.
        """

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: "
                f"{csv_path}"
            )

        return pd.read_csv(
            csv_path,
            **kwargs,
        )

    # ========================================================
    # ZHEEN
    # ========================================================

    def load_zheen(
        self,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """
        Load the ZHEEN dataset.

        Args:
            filename:
                Optional CSV filename.

                If omitted, the function searches the ZHEEN
                directory for CSV files.

        Returns:
            ZHEEN DataFrame.

        Raises:
            FileNotFoundError:
                If no suitable CSV file is found.
        """

        if filename is not None:

            csv_path = (
                self.zheen_dir / filename
            )

            return self.read_csv(
                csv_path
            )

        csv_files = self.find_csv_files(
            self.zheen_dir
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in ZHEEN directory: "
                f"{self.zheen_dir}"
            )

        if len(csv_files) > 1:

            raise RuntimeError(
                "Multiple CSV files were found in the "
                "ZHEEN directory. Specify the filename "
                "explicitly."
            )

        return self.read_csv(
            csv_files[0]
        )

    # ========================================================
    # UCI HEART FAILURE
    # ========================================================

    def load_uci_heart_failure(
        self,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """
        Load the UCI Heart Failure Clinical Records dataset.

        Args:
            filename:
                Optional CSV filename.

                If omitted, CSV files are searched recursively.

        Returns:
            UCI Heart Failure DataFrame.
        """

        if filename is not None:

            csv_path = (
                self.uci_dir / filename
            )

            return self.read_csv(
                csv_path
            )

        csv_files = self.find_csv_files(
            self.uci_dir
        )

        if not csv_files:
            raise FileNotFoundError(
                "No CSV files found in UCI Heart Failure "
                f"directory: {self.uci_dir}"
            )

        if len(csv_files) > 1:

            raise RuntimeError(
                "Multiple CSV files were found in the "
                "UCI Heart Failure directory. "
                "Specify the filename explicitly."
            )

        return self.read_csv(
            csv_files[0]
        )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    @staticmethod
    def describe_dataframe(
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Return basic information about a tabular dataset.
        """

        return {
            "num_rows": len(dataframe),
            "num_columns": len(dataframe.columns),
            "columns": list(
                dataframe.columns
            ),
            "dtypes": {
                column: str(dtype)
                for column, dtype
                in dataframe.dtypes.items()
            },
            "missing_values": {
                column: int(
                    dataframe[column].isna().sum()
                )
                for column in dataframe.columns
            },
        }

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    def summary(self) -> dict:
        """
        Return information about available biomarker files.
        """

        zheen_files = self.find_csv_files(
            self.zheen_dir
        ) if self.zheen_dir.exists() else []

        uci_files = self.find_csv_files(
            self.uci_dir
        ) if self.uci_dir.exists() else []

        return {
            "zheen_directory": str(
                self.zheen_dir
            ),
            "zheen_csv_files": [
                str(path)
                for path in zheen_files
            ],
            "uci_directory": str(
                self.uci_dir
            ),
            "uci_csv_files": [
                str(path)
                for path in uci_files
            ],
        }