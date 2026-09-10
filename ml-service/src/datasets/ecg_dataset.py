import torch
from torch.utils.data import Dataset
import numpy as np

class PTBXLDataset(Dataset):
    """
    Loads 100Hz PTB-XL numpy tensors and their multi-label targets.
    """
    def __init__(
        self,
        signals: np.ndarray,
        labels_df,
        superclass_cols: list,
        rhythm_cols: list,
        record_paths=None,
    ):
        self.signals = signals
        self.record_paths = (
            list(record_paths)
            if record_paths is not None
            else [str(index) for index in range(len(signals))]
        )

        if len(self.record_paths) != len(self.signals):
            raise ValueError(
                "record_paths must have the same length as signals."
            )

        # Ensure labels are binary matrices
        self.superclass_targets = labels_df[superclass_cols].values.astype(np.float32)
        self.rhythm_targets = labels_df[rhythm_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # signal shape (1000, 12) -> transpose to (12, 1000) for 1D CNN
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).transpose(0, 1)
        
        return {
            'signal': signal,
            'diagnostic': torch.tensor(self.superclass_targets[idx]),
            'rhythm': torch.tensor(self.rhythm_targets[idx]),
            'record_path': self.record_paths[idx],
        }