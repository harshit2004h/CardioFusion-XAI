from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.fusion.registry_loader import load_disease_registry
from src.fusion.staged_fusion import StagedFusionEngine
from src.inference.model_manager import ModelManager
from src.inference.service import InferenceService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "config" / "disease_registry.yaml"
load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def get_fusion_engine() -> StagedFusionEngine:
    """Return the registry-backed fusion engine."""
    return StagedFusionEngine(load_disease_registry(REGISTRY_PATH))


@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    return ModelManager(PROJECT_ROOT)


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService(get_fusion_engine(), get_model_manager())
