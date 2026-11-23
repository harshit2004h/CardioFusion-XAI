import numpy as np

from src.calibration.calibrator import PlattCalibrator
from src.inference.model_manager import ModelManager


def test_platt_calibrator_fit_predict_save_load(tmp_path):
    raw = np.linspace(0.05, 0.95, 20)
    labels = np.array([0] * 10 + [1] * 10)
    calibrator = PlattCalibrator().fit(raw, labels)
    calibrated = calibrator.predict_proba(raw)

    assert calibrated.shape == raw.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))

    path = tmp_path / "calibrator.pkl"
    calibrator.save(path)
    loaded = PlattCalibrator.load(path)
    np.testing.assert_allclose(calibrated, loaded.predict_proba(raw))


def test_selected_calibrator_requires_matching_model_hash(tmp_path):
    manager = ModelManager.__new__(ModelManager)
    manager.project_root = tmp_path
    checkpoint = tmp_path / "model.joblib"
    checkpoint.write_bytes(b"selected model bytes")
    artifact_path = tmp_path / "checkpoints" / "calibration" / "biomarkers_selected" / "zheen" / "acute_mi.pkl"
    artifact_path.parent.mkdir(parents=True)
    calibrator = PlattCalibrator().fit(np.linspace(0.05, 0.95, 20), np.array([0] * 10 + [1] * 10))
    calibrator.save(artifact_path)
    entry = {
        "status": "platt",
        "artifact": "biomarkers_selected/zheen/acute_mi.pkl",
        "base_model_sha256": manager._sha256(checkpoint),
    }

    assert manager._load_platt_if_matching(entry, checkpoint) is not None
    checkpoint.write_bytes(b"different model bytes")
    assert manager._load_platt_if_matching(entry, checkpoint) is None
