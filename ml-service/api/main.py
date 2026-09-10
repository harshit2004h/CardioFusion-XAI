from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from api.dependencies import (
    PROJECT_ROOT,
    get_fusion_engine,
    get_inference_service,
    get_model_manager,
)
from api.schemas import InferenceRequest, InferenceResponse
from src.fusion.registry_loader import load_disease_registry
from src.fusion.staged_fusion import StagedFusionEngine
from src.inference.service import InferenceService


class PredictionRequest(BaseModel):
    predictions: dict[str, dict[str, Any]] = Field(default_factory=dict)


app = FastAPI(
    title="CardioFusion-XAI ML Service",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/registry")
def registry() -> dict:
    return load_disease_registry(PROJECT_ROOT / "config" / "disease_registry.yaml")


@app.get("/v1/health")
def versioned_health() -> dict[str, object]:
    manager = get_model_manager()
    return {
        "status": "ok",
        "service": "cardiofusion-ml",
        "device": manager.device,
        "models_loaded": manager.models_loaded,
        "models": manager.status,
    }


@app.post("/v1/inference", response_model=InferenceResponse)
def inference(
    request: InferenceRequest,
    service: InferenceService = Depends(get_inference_service),  # noqa: B008
) -> dict:
    return service.run(request)


@app.post("/predict")
def predict(
    request: PredictionRequest,
    engine: StagedFusionEngine = Depends(get_fusion_engine),  # noqa: B008
) -> dict[str, Any]:
    """Evaluate supplied modality predictions without manufacturing missing evidence."""
    return {
        "results": engine.evaluate_all(request.predictions),
        "available_modalities": sorted(request.predictions),
    }


def main() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("CARDIOFUSION_HOST", "0.0.0.0"),
        port=int(os.getenv("CARDIOFUSION_PORT", "3002")),
        reload=False,
    )


if __name__ == "__main__":
    main()
