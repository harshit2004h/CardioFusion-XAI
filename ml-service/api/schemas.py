from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class FileReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    biomarkers: FileReference | None = None
    ecg: FileReference | None = None
    echo: FileReference | None = None


class ModalityStatus(BaseModel):
    available: bool
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    request_id: str
    service_version: str
    status: str
    modalities: dict[str, ModalityStatus]
    extraction: dict[str, Any] = Field(default_factory=dict)
    predictions: dict[str, Any] = Field(default_factory=dict)
    disease_results: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    gemini_recommendations: dict[str, Any] = Field(default_factory=dict)
    explainability: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, str]] = Field(default_factory=list)