from pathlib import Path

import cv2
import numpy as np

from src.ecg_processing.image_waveform import digitize_ecg_image


def test_ecg_image_digitizer_returns_model_shape(tmp_path: Path):
    height, width = 900, 1200
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for row in range(3):
        for col in range(4):
            top = row * height // 3
            left = col * width // 4
            x = np.arange(left + 20, (col + 1) * width // 4 - 20)
            y = top + height // 6 + 35 * np.sin(np.linspace(0, 8 * np.pi, x.size))
            points = np.column_stack((x, y.astype(int))).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, (0, 0, 0), 3)
    path = tmp_path / "ecg.png"
    assert cv2.imwrite(str(path), image)

    waveform, metadata = digitize_ecg_image(path)

    assert waveform.shape == (12, 1000)
    assert metadata["leads"] == 12
    assert metadata["quality_score"] >= 0.35