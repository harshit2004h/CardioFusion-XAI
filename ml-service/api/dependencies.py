from pathlib import Path

from src.fusion.registry_loader import load_disease_registry
from src.fusion.staged_fusion import StagedFusionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "config" / "disease_registry.yaml"


def get_fusion_engine() -> StagedFusionEngine:
    """Build an engine from the checked-in registry for each API request."""
    return StagedFusionEngine(load_disease_registry(REGISTRY_PATH))
