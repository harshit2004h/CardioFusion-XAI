import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MultiSourceBiomarkerDataset(Dataset):
    """
    Handles Zheen, UCI-HF, MI-Complications, and Framingham jointly.
    Implements masked-loss logic by tracking which targets are valid per row.
    """
    def __init__(self, dataframes_dict: dict, canonical_targets: list):
        self.canonical_targets = canonical_targets
        self.samples = []
        
        for source_name, df in dataframes_dict.items():
            features = df.drop(columns=[col for col in df.columns if col in canonical_targets]).values
            
            for idx, row in df.iterrows():
                # Build target vector and mask
                target_vec = np.zeros(len(canonical_targets), dtype=np.float32)
                mask_vec = np.zeros(len(canonical_targets), dtype=np.float32)
                
                for t_idx, t_name in enumerate(canonical_targets):
                    if t_name in df.columns and not pd.isna(row[t_name]):
                        target_vec[t_idx] = float(row[t_name])
                        mask_vec[t_idx] = 1.0 # Valid target for this source
                        
                self.samples.append({
                    'source': source_name,
                    'features': torch.tensor(features[idx], dtype=torch.float32),
                    'targets': torch.tensor(target_vec),
                    'mask': torch.tensor(mask_vec)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]