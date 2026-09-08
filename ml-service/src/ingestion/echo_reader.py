import pandas as pd
import cv2
import numpy as np
from pathlib import Path

class EchoReader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir) / 'echonet'

    def read_metadata(self) -> pd.DataFrame:
        """Loads EchoNet-Dynamic volume and EF labels."""
        return pd.read_csv(self.data_dir / 'FileList.csv')

    def load_video_clip(self, filename: str, num_frames: int = 32, stride: int = 4) -> np.ndarray:
        """Samples frames from an AVI video, converts to 3-channel grayscale for R2+1D."""
        video_path = str(self.data_dir / 'Videos' / f"{filename}.avi")
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0

        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                frames.append(rgb_frame)
            count += 1

        cap.release()
        return np.array(frames) # Target shape: (T, H, W, C)