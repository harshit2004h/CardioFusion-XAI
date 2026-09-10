from pydantic import BaseModel, Field


class ExtractedTest(BaseModel):
    canonical_candidate: str | None = None
    source_label: str
    value: float | None = None
    unit: str | None = None
    reference_range: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentExtraction(BaseModel):
    raw_text_available: bool
    tests: list[ExtractedTest] = Field(default_factory=list)
    ignored_extra_tests: list[str] = Field(default_factory=list)
    missing_required_features: list[str] = Field(default_factory=list)
    extraction_quality: float = Field(default=0.0, ge=0.0, le=1.0)