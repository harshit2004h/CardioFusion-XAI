from pathlib import Path

import numpy as np

from src.ingestion.cloudinary import ALLOWED_FORMATS, _format_from_bytes
from src.ingestion.ecg_waveform import read_ecg_file


def test_supported_upload_formats_match_product_contract():
    assert ALLOWED_FORMATS["biomarkers"] == {"pdf", "jpg", "jpeg", "png"}
    assert ALLOWED_FORMATS["ecg"] == {"dicom", "dat", "csv", "scp", "edf"}
    assert ALLOWED_FORMATS["echo"] == {"mp4", "avi"}
    assert _format_from_bytes(b"", "text/csv", ".csv") == "csv"


def test_csv_ecg_is_normalized_to_bilstm_shape(tmp_path: Path):
    path = tmp_path / "ecg.csv"
    np.savetxt(path, np.zeros((1200, 12), dtype=np.float32), delimiter=",")

    waveform, metadata = read_ecg_file(path, "csv")

    assert waveform.shape == (12, 1000)
    assert metadata["sampling_rate"] == 100


def test_scp_numeric_ecg_is_supported(tmp_path: Path):
    path = tmp_path / "ecg.scp"
    np.savetxt(path, np.ones((12, 1000), dtype=np.float32), delimiter=",")

    waveform, metadata = read_ecg_file(path, "scp")

    assert waveform.shape == (12, 1000)
    assert metadata["format"] == "scp"
