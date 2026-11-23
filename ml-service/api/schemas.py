from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class FileReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    header_url: HttpUrl | None = None
    filename: str | None = None
    header_filename: str | None = None


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    biomarkers: FileReference | None = None
    ecg: FileReference | None = None
    echo: FileReference | None = None
    explain: bool = False

    @model_validator(mode="after")
    def require_at_least_one_modality(self) -> "InferenceRequest":
        if not any((self.biomarkers, self.ecg, self.echo)):
            raise ValueError("Provide at least one of biomarkers, ecg, or echo.")
        return self


class ModalityStatus(BaseModel):
    available: bool
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    request_id: str
    service_version: str
    status: str
    available_modalities: list[str] = Field(default_factory=list)
    result_scope: str = "none"
    modalities: dict[str, ModalityStatus]
    extraction: dict[str, Any] = Field(default_factory=dict)
    predictions: dict[str, Any] = Field(default_factory=dict)
    disease_results: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    gemini_recommendations: dict[str, Any] = Field(default_factory=dict)
    explainability: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, str]] = Field(default_factory=list)
