from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from api.dependencies import get_fusion_engine
from src.fusion.staged_fusion import StagedFusionEngine


class PredictionRequest(BaseModel):
    predictions: dict[str, dict[str, Any]] = Field(default_factory=dict)


app = FastAPI(
    title="CardioFusion-XAI ML Service",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(
    request: PredictionRequest,
    engine: StagedFusionEngine = Depends(get_fusion_engine),
) -> dict[str, Any]:
    """Evaluate supplied modality predictions without manufacturing missing evidence."""
    return {
        "results": engine.evaluate_all(request.predictions),
        "available_modalities": sorted(request.predictions),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
