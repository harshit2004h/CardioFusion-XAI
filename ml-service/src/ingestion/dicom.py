from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np


class DicomProcessingError(ValueError):
    """The DICOM object is valid but not usable for the requested modality."""


@dataclass
class DicomContent:
    text: str
    metadata: dict[str, str | int | float]


def read_dataset(path: Path):
    try:
        import pydicom

        return pydicom.dcmread(path, force=False)
    except Exception as exc:
        raise DicomProcessingError(f"Could not read DICOM file: {exc}") from exc


def read_document(path: Path) -> DicomContent:
    dataset = read_dataset(path)
    text_parts: list[str] = []
    metadata: dict[str, str | int | float] = {}
    for element in dataset.iterall():
        if element.VR in {"OB", "OW", "OF", "SQ", "UN"}:
            continue
        value = element.value
        if isinstance(value, (str, int, float)):
            text_parts.append(f"{element.keyword or element.name}: {value}")
        if element.keyword in {"StudyDescription", "SeriesDescription", "Modality", "Manufacturer"}:
            metadata[element.keyword] = str(value)

    if getattr(dataset, "Modality", "") == "ECG" and hasattr(dataset, "WaveformSequence"):
        metadata["waveform_available"] = 1
    if hasattr(dataset, "EncapsulatedDocument"):
        try:
            from pypdf import PdfReader

            pdf = PdfReader(BytesIO(bytes(dataset.EncapsulatedDocument)))
            text_parts.append("\n".join(page.extract_text() or "" for page in pdf.pages))
        except Exception:
            metadata["encapsulated_document"] = "present_but_unreadable"
    return DicomContent(text="\n".join(text_parts), metadata=metadata)


def read_ecg_waveform(path: Path, target_samples: int = 1000, target_rate: int = 100) -> tuple[np.ndarray, dict[str, int | float]]:
    dataset = read_dataset(path)
    sequence = getattr(dataset, "WaveformSequence", None)
    if not sequence:
        raise DicomProcessingError("DICOM ECG contains no WaveformSequence.")
    try:
        from pydicom.waveforms.numpy_handler import multiplex_array

        waveform = np.asarray(multiplex_array(dataset, 0, as_raw=False), dtype=np.float32)
    except Exception as exc:
        raise DicomProcessingError(f"Could not decode DICOM ECG waveform: {exc}") from exc
    if waveform.ndim != 2:
        raise DicomProcessingError("DICOM ECG waveform has an invalid shape.")
    group = sequence[0]
    channels = int(getattr(group, "NumberOfWaveformChannels", waveform.shape[1]))
    sample_rate = float(getattr(group, "SamplingFrequency", target_rate))
    if channels < 12 or waveform.shape[1] < 12:
        raise DicomProcessingError(f"DICOM ECG provides {min(channels, waveform.shape[1])} leads; 12 are required.")
    waveform = waveform[:, :12]
    duration = waveform.shape[0] / sample_rate
    if duration < 10.0:
        raise DicomProcessingError("DICOM ECG recording is shorter than the required 10 seconds.")
    source_time = np.linspace(0.0, duration, waveform.shape[0], endpoint=False)
    target_time = np.arange(target_samples, dtype=np.float32) / target_rate
    waveform = np.stack([np.interp(target_time, source_time, waveform[:, lead]) for lead in range(12)], axis=0)
    return waveform.astype(np.float32), {
        "leads": 12,
        "duration_seconds": 10.0,
        "sampling_rate": target_rate,
        "source_sampling_rate": sample_rate,
        "quality_score": 1.0,
    }


def read_echo_frames(path: Path, frames: int = 32, size: int = 112) -> tuple[np.ndarray, dict[str, int | float | str]]:
    dataset = read_dataset(path)
    try:
        pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise DicomProcessingError(f"Could not decode DICOM Echo pixels: {exc}") from exc
    if pixels.ndim == 2:
        pixels = pixels[None, ...]
    if pixels.ndim == 4:
        pixels = pixels[..., :3].mean(axis=-1)
    if pixels.ndim != 3 or pixels.shape[0] < 1:
        raise DicomProcessingError("DICOM Echo does not contain a readable frame sequence.")
    import cv2

    indices = np.linspace(0, pixels.shape[0] - 1, frames, dtype=np.int64)
    sampled = []
    for index in indices:
        frame = pixels[index].astype(np.float32)
        frame -= frame.min()
        peak = frame.max()
        if peak > 0:
            frame /= peak
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        sampled.append(np.repeat(frame[..., None], 3, axis=2))
    return np.asarray(sampled, dtype=np.float32), {
        "format": "DICOM",
        "frames_used": frames,
        "source_frames": int(pixels.shape[0]),
    }