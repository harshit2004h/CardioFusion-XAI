import numpy as np

from src.calibration.calibrator import PlattCalibrator


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
