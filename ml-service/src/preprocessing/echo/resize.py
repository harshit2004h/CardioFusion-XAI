import torch
import torchvision.transforms.functional as F

class ResizeVideo:
    """Resizes all frames in a video tensor to the specified spatial size."""
    def __init__(self, size: tuple = (112, 112)):
        self.size = size # R2+1D standard input size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # video shape: (C, T, H, W) -> transform requires (C, H, W) per frame
        # Reshape to (C*T, H, W)
        c, t, h, w = video.shape
        video = video.view(c * t, h, w)
        video = F.resize(video, self.size, antialias=True)
        # Reshape back to (C, T, H, W)
        return video.view(c, t, self.size[0], self.size[1])