"""
Configuration utilities for CardioFusion-XAI.

This module loads YAML configuration files and provides helpers
for accessing and merging configuration dictionaries.
"""

from pathlib import Path
from typing import Any

import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

# config.py is located at:
#
# ml-service/
# └── src/
#     └── utils/
#         └── config.py
#
# parents[0] -> utils
# parents[1] -> src
# parents[2] -> ml-service
#
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"


# ============================================================
# YAML LOADING
# ============================================================

def load_yaml(filename: str) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        filename: Name of the YAML file inside config/.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError:
            If the requested configuration file does not exist.

        ValueError:
            If the YAML file does not contain a dictionary.
    """

    config_path = CONFIG_DIR / filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping: "
            f"{config_path}"
        )

    return config


# ============================================================
# INDIVIDUAL CONFIGURATION LOADERS
# ============================================================

def load_base_config() -> dict[str, Any]:
    """Load the shared base configuration."""
    return load_yaml("base.yaml")


def load_ecg_config() -> dict[str, Any]:
    """Load the ECG configuration."""
    return load_yaml("ecg.yaml")


def load_echo_config() -> dict[str, Any]:
    """Load the echocardiography configuration."""
    return load_yaml("echo.yaml")


def load_biomarker_config() -> dict[str, Any]:
    """Load the biomarker configuration."""
    return load_yaml("biomarkers.yaml")


def load_ccta_config() -> dict[str, Any]:
    """Load the CCTA configuration."""
    return load_yaml("ccta.yaml")


# ============================================================
# NESTED CONFIGURATION ACCESS
# ============================================================

def get_config_value(
    config: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a configuration value using dot notation.

    Example:

        get_config_value(
            config,
            "training.batch_size"
        )

    This accesses:

        training:
          batch_size: 32

    Args:
        config: Configuration dictionary.
        key: Dot-separated configuration key.
        default: Value returned when the key is not found.

    Returns:
        Requested configuration value.
    """

    value: Any = config

    for part in key.split("."):
        if not isinstance(value, dict):
            return default

        if part not in value:
            return default

        value = value[part]

    return value


# ============================================================
# CONFIGURATION MERGING
# ============================================================

def merge_configs(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Values from the override configuration take precedence.

    Nested dictionaries are merged instead of completely
    replacing the parent dictionary.

    Args:
        base: Base configuration.
        override: Configuration containing overriding values.

    Returns:
        Merged configuration dictionary.
    """

    merged = dict(base)

    for key, value in override.items():

        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(
                merged[key],
                value,
            )

        else:
            merged[key] = value

    return merged


# ============================================================
# MODALITY CONFIGURATION
# ============================================================

def load_modality_config(
    modality: str,
) -> dict[str, Any]:
    """
    Load base.yaml together with a modality-specific YAML file.

    Supported modalities:

        ecg
        echo
        biomarkers
        ccta

    Example:

        config = load_modality_config("ecg")

    Returns:
        Complete merged configuration.
    """

    modality_files = {
        "ecg": "ecg.yaml",
        "echo": "echo.yaml",
        "biomarkers": "biomarkers.yaml",
        "ccta": "ccta.yaml",
    }

    modality = modality.lower()

    if modality not in modality_files:
        valid_modalities = ", ".join(
            modality_files.keys()
        )

        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Expected one of: {valid_modalities}"
        )

    base_config = load_base_config()

    modality_config = load_yaml(
        modality_files[modality]
    )

    return merge_configs(
        base_config,
        modality_config,
    )