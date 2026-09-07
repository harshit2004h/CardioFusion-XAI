"""
Centralized filesystem paths for CardioFusion-XAI.
"""

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# File location:
#
# ml-service/
# └── src/
#     └── utils/
#         └── paths.py
#
# parents[0] -> utils
# parents[1] -> src
# parents[2] -> ml-service
#

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# MAIN DIRECTORIES
# ============================================================

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

TEST_DIR = PROJECT_ROOT / "tests"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


# ============================================================
# DATASET DIRECTORIES
# ============================================================

PTBXL_DIR = RAW_DATA_DIR / "ptbxl"

ECHONET_DIR = RAW_DATA_DIR / "echonet"

BIOMARKERS_DIR = RAW_DATA_DIR / "biomarkers"

ZHEEN_DIR = BIOMARKERS_DIR / "zheen"

UCI_HEART_FAILURE_DIR = (
    BIOMARKERS_DIR / "uci_heart_failure"
)

IMAGECAS_DIR = RAW_DATA_DIR / "imagecas"


# ============================================================
# CHECKPOINT DIRECTORIES
# ============================================================

ECG_CHECKPOINT_DIR = (
    CHECKPOINT_DIR / "ecg"
)

ECHO_CHECKPOINT_DIR = (
    CHECKPOINT_DIR / "echo"
)

BIOMARKER_CHECKPOINT_DIR = (
    CHECKPOINT_DIR / "biomarkers"
)

CCTA_CHECKPOINT_DIR = (
    CHECKPOINT_DIR / "ccta"
)

FUSION_CHECKPOINT_DIR = (
    CHECKPOINT_DIR / "fusion"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

LOG_DIR = OUTPUT_DIR / "logs"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

METRICS_DIR = OUTPUT_DIR / "metrics"

FIGURE_DIR = OUTPUT_DIR / "figures"

EMBEDDING_DIR = OUTPUT_DIR / "embeddings"

REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_project_directories() -> None:
    """
    Create directories required by the ML service.

    Existing directories are not modified.
    """

    directories = [
        # Data
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,

        # Checkpoints
        CHECKPOINT_DIR,
        ECG_CHECKPOINT_DIR,
        ECHO_CHECKPOINT_DIR,
        BIOMARKER_CHECKPOINT_DIR,
        CCTA_CHECKPOINT_DIR,
        FUSION_CHECKPOINT_DIR,

        # Outputs
        OUTPUT_DIR,
        LOG_DIR,
        PREDICTION_DIR,
        METRICS_DIR,
        FIGURE_DIR,
        EMBEDDING_DIR,
        REPORT_DIR,

        # Experiments
        EXPERIMENT_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# DATASET PATH LOOKUP
# ============================================================

def get_dataset_path(
    dataset_name: str,
) -> Path:
    """
    Return the raw directory for a supported dataset.

    Supported datasets:

        ptbxl
        echonet
        zheen
        uci_heart_failure
        imagecas

    Args:
        dataset_name:
            Dataset identifier.

    Returns:
        Path to the dataset directory.
    """

    datasets = {
        "ptbxl": PTBXL_DIR,
        "echonet": ECHONET_DIR,
        "zheen": ZHEEN_DIR,
        "uci_heart_failure": UCI_HEART_FAILURE_DIR,
        "imagecas": IMAGECAS_DIR,
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in datasets:

        valid_datasets = ", ".join(
            datasets.keys()
        )

        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Expected one of: {valid_datasets}"
        )

    return datasets[dataset_name]