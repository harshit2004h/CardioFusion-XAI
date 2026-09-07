"""
EchoNet-Dynamic echocardiography data reader.

This module handles raw video and metadata ingestion from
the EchoNet-Dynamic dataset.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import ECHONET_DIR


# ============================================================
# ECHONET READER
# ============================================================

class EchoNetReader:
    """
    Reader for the EchoNet-Dynamic dataset.
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the EchoNet reader.

        Args:
            dataset_dir:
                Path to the EchoNet-Dynamic directory.
                Defaults to data/raw/echonet/.
        """

        if dataset_dir is None:
            dataset_dir = ECHONET_DIR

        self.dataset_dir = Path(dataset_dir)

        self.file_list_path = (
            self.dataset_dir / "FileList.csv"
        )

        self.volume_tracings_path = (
            self.dataset_dir / "VolumeTracings.csv"
        )

        self.video_dir = (
            self.dataset_dir / "Videos"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_dataset(self) -> None:
        """
        Validate the required EchoNet-Dynamic files.
        """

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"EchoNet directory does not exist: "
                f"{self.dataset_dir}"
            )

        if not self.file_list_path.exists():
            raise FileNotFoundError(
                f"FileList.csv not found: "
                f"{self.file_list_path}"
            )

        if not self.volume_tracings_path.exists():
            raise FileNotFoundError(
                f"VolumeTracings.csv not found: "
                f"{self.volume_tracings_path}"
            )

        if not self.video_dir.exists():
            raise FileNotFoundError(
                f"EchoNet Videos directory not found: "
                f"{self.video_dir}"
            )

    # ========================================================
    # FILE LIST
    # ========================================================

    def load_file_list(
        self,
    ) -> pd.DataFrame:
        """
        Load EchoNet FileList.csv.

        Returns:
            DataFrame containing video metadata.
        """

        if not self.file_list_path.exists():
            raise FileNotFoundError(
                f"FileList.csv not found: "
                f"{self.file_list_path}"
            )

        return pd.read_csv(
            self.file_list_path
        )

    # ========================================================
    # VOLUME TRACINGS
    # ========================================================

    def load_volume_tracings(
        self,
    ) -> pd.DataFrame:
        """
        Load VolumeTracings.csv.

        Returns:
            DataFrame containing ventricular volume/tracing
            annotations.
        """

        if not self.volume_tracings_path.exists():
            raise FileNotFoundError(
                f"VolumeTracings.csv not found: "
                f"{self.volume_tracings_path}"
            )

        return pd.read_csv(
            self.volume_tracings_path
        )

    # ========================================================
    # VIDEO PATH
    # ========================================================

    def get_video_path(
        self,
        video_name: str,
    ) -> Path:
        """
        Resolve an EchoNet video filename.

        Args:
            video_name:
                Video filename, for example:
                '0X1A2B3C.avi'

        Returns:
            Path to the video.

        Raises:
            FileNotFoundError:
                If the video does not exist.
        """

        video_path = (
            self.video_dir / video_name
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"EchoNet video not found: "
                f"{video_path}"
            )

        return video_path

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    def get_video_metadata(
        self,
        video_name: str,
    ) -> dict[str, Any]:
        """
        Return basic information about an EchoNet video.

        This function does not decode the complete video.
        """

        video_path = self.get_video_path(
            video_name
        )

        metadata: dict[str, Any] = {
            "filename": video_path.name,
            "path": str(video_path),
            "size_mb": (
                video_path.stat().st_size
                / (1024 * 1024)
            ),
        }

        return metadata

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    def summary(self) -> dict[str, Any]:
        """
        Return a basic EchoNet-Dynamic dataset summary.
        """

        file_list = self.load_file_list()

        volume_tracings = (
            self.load_volume_tracings()
        )

        return {
            "dataset": "EchoNet-Dynamic",
            "directory": str(self.dataset_dir),
            "num_file_list_rows": len(file_list),
            "num_volume_tracing_rows": len(
                volume_tracings
            ),
            "video_directory": str(
                self.video_dir
            ),
        }