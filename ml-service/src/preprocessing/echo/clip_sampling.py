import torch
import numpy as np

class TemporalSubsample:
    """Samples a fixed number of frames from a video tensor."""
    def __init__(self, num_frames: int = 32, temporal_stride: int = 4):
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # video shape: (C, T, H, W)
        total_frames = video.shape[1]
        required_frames = self.num_frames * self.temporal_stride
        
        if total_frames < required_frames:
            # Pad by repeating the last frame
            padding = required_frames - total_frames
            last_frame = video[:, -1:, :, :]
            video = torch.cat([video, last_frame.repeat(1, padding, 1, 1)], dim=1)
            
        # Random start index for training, 0 for validation
        start_idx = np.random.randint(0, max(1, video.shape[1] - required_frames + 1))
        indices = torch.arange(start_idx, start_idx + required_frames, self.temporal_stride)
        
        return video[:, indices, :, :]