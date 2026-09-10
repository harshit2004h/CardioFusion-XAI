from pathlib import Path

import yaml


def load_disease_registry(
    path: str | Path,
) -> dict:
    """
    Load the central disease registry YAML.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Disease registry not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = yaml.safe_load(file)

    if not isinstance(
        registry,
        dict,
    ):
        raise ValueError(
            "Disease registry must contain a YAML mapping."
        )

    if "diseases" not in registry:
        raise ValueError(
            "Disease registry must contain a 'diseases' section."
        )

    return registry