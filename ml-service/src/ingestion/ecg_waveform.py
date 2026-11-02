from __future__ import annotations

from pathlib import Path

import numpy as np

from .dicom import read_ecg_waveform


class ECGWaveformError(ValueError):
    pass


def _normalise_shape(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    signal = np.squeeze(signal)
    if signal.ndim != 2:
        raise ECGWaveformError(f"ECG data must be two-dimensional, got {signal.shape}.")
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    raise ECGWaveformError(f"ECG data must contain 12 leads, got {signal.shape}.")


def _resample(
    signal: np.ndarray,
    source_rate: float,
    target_rate: int = 100,
    target_samples: int = 1000,
) -> np.ndarray:
    if source_rate <= 0 or signal.shape[1] < 2:
        raise ECGWaveformError("ECG sampling rate and signal length must be valid.")
    source_time = np.arange(signal.shape[1], dtype=np.float32) / float(source_rate)
    target_time = np.arange(target_samples, dtype=np.float32) / target_rate
    if source_time[-1] < target_time[-1]:
        raise ECGWaveformError("ECG recording is shorter than the required 10 seconds.")
    return np.stack(
        [np.interp(target_time, source_time, lead) for lead in signal],
        axis=0,
    ).astype(np.float32)


def _read_numeric(path: Path) -> np.ndarray:
    try:
        values = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ECGWaveformError(f"Could not parse ECG text data: {exc}") from exc
    if values.size == 0 or np.isnan(values).all():
        raise ECGWaveformError("ECG text file contains no numeric samples.")
    return _normalise_shape(values)


def _read_wfdb(path: Path) -> tuple[np.ndarray, float]:
    try:
        import wfdb

        record = wfdb.rdrecord(str(path.with_suffix("")))
    except Exception as exc:
        raise ECGWaveformError("WFDB .dat input requires a matching .hea header file.") from exc
    return _normalise_shape(record.p_signal), float(record.fs)


def _read_edf(path: Path) -> tuple[np.ndarray, float]:
    try:
        import pyedflib

        reader = pyedflib.EdfReader(str(path))
    except ImportError as exc:
        raise ECGWaveformError("EDF support requires the pyedflib dependency.") from exc
    except Exception as exc:
        raise ECGWaveformError(f"Could not open EDF file: {exc}") from exc
    try:
        if reader.signals_in_file < 12:
            raise ECGWaveformError("EDF input must contain at least 12 channels.")
        signals = np.asarray(
            [reader.readSignal(index) for index in range(12)],
            dtype=np.float32,
        )
        rates = [reader.getSampleFrequency(index) for index in range(12)]
    finally:
        reader.close()
    if len(set(float(rate) for rate in rates)) != 1:
        raise ECGWaveformError("The first 12 EDF channels must use the same sampling rate.")
    return signals, float(rates[0])


def read_ecg_file(
    path: Path,
    file_format: str,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    if file_format == "dicom":
        return read_ecg_waveform(path)
    if file_format in {"csv", "scp"}:
        signal = _read_numeric(path)
        source_rate = 100.0
    elif file_format == "dat":
        signal, source_rate = _read_wfdb(path)
    elif file_format == "edf":
        signal, source_rate = _read_edf(path)
    else:
        raise ECGWaveformError(f"Unsupported ECG format: {file_format}")
    waveform = _resample(signal, source_rate)
    return waveform, {
        "format": file_format,
        "leads": 12,
        "duration_seconds": 10.0,
        "sampling_rate": 100,
        "source_sampling_rate": float(source_rate),
    }
