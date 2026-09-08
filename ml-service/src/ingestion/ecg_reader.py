import pandas as pd
import numpy as np
import wfdb
import ast
from pathlib import Path

class ECGReader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir) / 'ptbxl'

    def read_metadata(self) -> pd.DataFrame:
        """Loads PTB-XL metadata and parses SCP dictionaries."""
        df = pd.read_csv(self.data_dir / 'ptbxl_database.csv', index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        return df

    def read_scp_statements(self) -> pd.DataFrame:
        """Loads SCP mapping statements."""
        return pd.read_csv(self.data_dir / 'scp_statements.csv', index_col=0)

    def load_raw_signals(self, df: pd.DataFrame, sampling_rate: int = 100) -> np.ndarray:
        """Loads WFDB records into a numpy tensor."""
        filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        data = [wfdb.rdsamp(str(self.data_dir / f)) for f in df[filename_col]]
        signals = np.array([signal for signal, meta in data])
        return signals