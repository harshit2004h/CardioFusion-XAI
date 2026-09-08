import torchvision.transforms as T
from .clip_sampling import TemporalSubsample
from .resize import ResizeVideo
from .normalization import NormalizeVideo

def get_echo_train_transforms():
    return T.Compose([
        TemporalSubsample(num_frames=32, temporal_stride=4),
        ResizeVideo((128, 128)), # slightly larger for random crop
        # Insert RandomCropVideo here if using pytorchvideo, or apply frame-wise
        ResizeVideo((112, 112)),
        NormalizeVideo()
    ])

def get_echo_val_transforms():
    return T.Compose([
        TemporalSubsample(num_frames=32, temporal_stride=4), # Could fix start_idx=0
        ResizeVideo((112, 112)),
        NormalizeVideo()
    ])