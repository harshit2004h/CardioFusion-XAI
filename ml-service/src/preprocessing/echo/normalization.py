import torch

class NormalizeVideo:
    """Normalizes video tensors using Kinetics-400 statistics."""
    def __init__(self, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)):
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # Expects video in [0, 1] range, shape (C, T, H, W)
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        return (video - self.mean) / self.std