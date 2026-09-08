import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2

class EchoNetDataset(Dataset):
    """
    Loads EchoNet-Dynamic videos and continuous EF targets.
    """
    def __init__(self, metadata_df: pd.DataFrame, video_dir: str, transform=None):
        self.df = metadata_df.reset_index(drop=True)
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['FileName']
        ef_target = torch.tensor([row['EF']], dtype=torch.float32)
        
        # Load raw video tensor (simplified cv2 load; adapt if using torchvision.io)
        video_path = os.path.join(self.video_dir, f"{filename}.avi")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # BGR to RGB, then HWC to CHW
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        cap.release()
        
        # Stack to (C, T, H, W)
        if len(frames) > 0:
            video_tensor = torch.stack(frames, dim=1) 
        else:
            video_tensor = torch.zeros((3, 32, 112, 112))

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {
            'video': video_tensor,
            'ef_target': ef_target
        }