from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.sequence import Sequence  # noqa: E402
from pydicom.uid import (  # noqa: E402
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from src.ingestion.cloudinary import _format_from_bytes  # noqa: E402
from src.ingestion.dicom import read_document, read_ecg_waveform, read_echo_frames  # noqa: E402


def _dataset(path: Path, sop_class=SecondaryCaptureImageStorage) -> FileDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    return dataset


def test_dicom_magic_bytes_are_detected():
    prefix = b"\0" * 128 + b"DICM" + b"payload"
    assert _format_from_bytes(prefix, "", ".dcm") == "dicom"


def test_dicom_document_text_is_extracted(tmp_path):
    path = tmp_path / "blood.dcm"
    dataset = _dataset(path)
    dataset.Modality = "SR"
    dataset.StudyDescription = "Troponin I: 0.82 ng/mL"
    dataset.save_as(path)

    result = read_document(path)
    assert "Troponin I" in result.text
    assert result.metadata["Modality"] == "SR"


def test_dicom_ecg_waveform_is_resampled_to_model_shape(tmp_path):
    path = tmp_path / "ecg.dcm"
    dataset = _dataset(path)
    dataset.Modality = "ECG"
    group = FileDataset(None, {}, file_meta=dataset.file_meta, preamble=b"")
    group.WaveformOriginality = "ORIGINAL"
    group.NumberOfWaveformChannels = 12
    group.NumberOfWaveformSamples = 1000
    group.SamplingFrequency = 100.0
    group.MultiplexGroupTimeOffset = 0.0
    group.TriggerTimeOffset = 0.0
    group.WaveformBitsAllocated = 16
    group.WaveformSampleInterpretation = "SS"
    group.ChannelDefinitionSequence = Sequence([])
    values = np.zeros((1000, 12), dtype="<i2")
    group.WaveformData = values.tobytes()
    dataset.WaveformSequence = Sequence([group])
    dataset.save_as(path)

    waveform, metadata = read_ecg_waveform(path)
    assert waveform.shape == (12, 1000)
    assert metadata["sampling_rate"] == 100


def test_dicom_echo_frames_are_embedded_as_model_input(tmp_path):
    path = tmp_path / "echo.dcm"
    dataset = _dataset(path)
    dataset.Modality = "US"
    dataset.Rows = 32
    dataset.Columns = 32
    dataset.NumberOfFrames = 32
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.zeros((32, 32, 32), dtype=np.uint8).tobytes()
    dataset.save_as(path)

    frames, metadata = read_echo_frames(path)
    assert frames.shape == (32, 112, 112, 3)
    assert metadata["frames_used"] == 32
