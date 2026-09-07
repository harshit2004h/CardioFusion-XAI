"""
Dataset and experiment metadata utilities.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


# ============================================================
# DATASET METADATA
# ============================================================

@dataclass
class DatasetMetadata:
    """
    Metadata describing a dataset.
    """

    name: str

    modality: str

    version: str | None = None

    source: str | None = None

    description: str | None = None

    num_samples: int | None = None

    num_features: int | None = None

    target: str | None = None

    created_at: str | None = None

    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Set defaults after object creation.
        """

        if self.created_at is None:

            self.created_at = datetime.now(
                timezone.utc
            ).isoformat()

        if self.extra is None:

            self.extra = {}

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to a dictionary.
        """

        return asdict(self)


# ============================================================
# EXPERIMENT METADATA
# ============================================================

@dataclass
class ExperimentMetadata:
    """
    Metadata describing a model training experiment.
    """

    experiment_name: str

    modality: str

    model_name: str

    seed: int

    python_version: str | None = None

    pytorch_version: str | None = None

    device: str | None = None

    dataset_name: str | None = None

    started_at: str | None = None

    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Set defaults after object creation.
        """

        if self.started_at is None:

            self.started_at = datetime.now(
                timezone.utc
            ).isoformat()

        if self.extra is None:

            self.extra = {}

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to a dictionary.
        """

        return asdict(self)


# ============================================================
# DATASET METADATA FACTORY
# ============================================================

def create_dataset_metadata(
    name: str,
    modality: str,
    version: str | None = None,
    source: str | None = None,
    description: str | None = None,
    num_samples: int | None = None,
    num_features: int | None = None,
    target: str | None = None,
    **extra: Any,
) -> DatasetMetadata:
    """
    Convenience function for creating DatasetMetadata.
    """

    return DatasetMetadata(
        name=name,
        modality=modality,
        version=version,
        source=source,
        description=description,
        num_samples=num_samples,
        num_features=num_features,
        target=target,
        extra=extra,
    )


# ============================================================
# EXPERIMENT METADATA FACTORY
# ============================================================

def create_experiment_metadata(
    experiment_name: str,
    modality: str,
    model_name: str,
    seed: int = 42,
    python_version: str | None = None,
    pytorch_version: str | None = None,
    device: str | None = None,
    dataset_name: str | None = None,
    **extra: Any,
) -> ExperimentMetadata:
    """
    Convenience function for creating ExperimentMetadata.
    """

    return ExperimentMetadata(
        experiment_name=experiment_name,
        modality=modality,
        model_name=model_name,
        seed=seed,
        python_version=python_version,
        pytorch_version=pytorch_version,
        device=device,
        dataset_name=dataset_name,
        extra=extra,
    )