import pandas as pd
from pathlib import Path


class BiomarkerReader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def read_zheen(self) -> pd.DataFrame:
        """Loads Zheen heart attack dataset."""
        return pd.read_csv(self.data_dir / 'zheen' / 'heart_attack.csv')

    def read_uci_hf(self) -> pd.DataFrame:
        """Loads UCI Heart Failure dataset and drops leakage columns."""
        source_dir = self.data_dir / "uci_heart_failure"
        preferred_names = (
            "Heart_failure_clinical_records_dataset.csv",
            "heart_failure_clinical_records_dataset.csv",
        )
        path = next((source_dir / name for name in preferred_names if (source_dir / name).is_file()), None)
        if path is None:
            matches = sorted(source_dir.glob("*.csv")) if source_dir.is_dir() else []
            if len(matches) != 1:
                raise FileNotFoundError(
                    "Expected one UCI Heart Failure CSV under "
                    f"{source_dir}; found {[item.name for item in matches]}."
                )
            path = matches[0]
        df = pd.read_csv(path)
        if 'time' in df.columns:
            # Drop follow-up time to prevent target leakage for DEATH_EVENT
            df = df.drop(columns=['time'])
        return df

    def read_mi_complications(self) -> pd.DataFrame:
        """Loads MI Complications dataset."""
        return pd.read_csv(self.data_dir / 'mi_complications' / 'mi_complications.csv')

    def read_framingham(self) -> pd.DataFrame:
        """Loads Framingham 10-year CHD dataset."""
        return pd.read_csv(self.data_dir / 'framingham' / 'framingham.csv')
