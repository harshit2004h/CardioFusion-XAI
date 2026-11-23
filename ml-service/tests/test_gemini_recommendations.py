from src.recommendations.gemini import GeminiRecommendationService


class FakeResponse:
    text = '{"summary":"Review the result with a clinician.","recommendations":["Discuss the finding with a clinician.","Bring the report to your next visit."]}'


class FakeModels:
    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(model)
        if model in self.failing:
            raise RuntimeError("503 UNAVAILABLE")
        assert "Acute Myocardial Infarction" in contents
        assert "raw_text" not in contents
        assert "Cardiogenic Shock" not in contents
        return FakeResponse()


class FakeClient:
    def __init__(self, failing=()):
        self.models = FakeModels(failing)


MI_RESULT = {
    "disease": "acute_mi",
    "display_name": "Acute Myocardial Infarction",
    "risk": 0.82,
    "confidence": "MODERATE",
    "status": "primary_only",
    "primary_modality": "biomarkers",
    "raw_text": "patient private report text",
}
UNSUPPORTED = {"disease": "cardiogenic_shock", "display_name": "Cardiogenic Shock", "risk": None, "status": "unsupported"}
MISSING = {"disease": "hf_mortality", "display_name": "HF Mortality", "risk": None, "status": "primary_missing"}


def test_gemini_context_whitelists_fields_and_drops_results_without_output():
    context = GeminiRecommendationService._safe_context([MI_RESULT, UNSUPPORTED, MISSING])
    assert context == [
        {
            "disease": "acute_mi",
            "display_name": "Acute Myocardial Infarction",
            "risk": 0.82,
            "status": "primary_only",
            "detected_from": [],
        }
    ]


def test_gemini_context_keeps_uncalibrated_scores_and_measurements():
    context = GeminiRecommendationService._safe_context([
        {"disease": "atrial_fibrillation", "display_name": "AF", "risk": None, "model_score": 0.7, "status": "uncalibrated_primary"},
        {"disease": "reduced_lvef", "display_name": "LVEF", "risk": None, "status": "primary_only", "measurement": {"name": "EF", "value": 42.0, "unit": "%"}},
    ])
    assert [item["disease"] for item in context] == ["atrial_fibrillation", "reduced_lvef"]
    assert "risk" not in context[0]


def test_gemini_fallback_is_safe_without_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = GeminiRecommendationService()
    result = service.generate([{**MI_RESULT, "risk": 0.76}])
    assert result["source"] == "local_fallback"
    assert result["recommendations"]
    assert "diagnosis" in result["summary"]


def test_gemini_success_returns_structured_recommendations(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    service = GeminiRecommendationService()
    service._client = FakeClient()
    result = service.generate([MI_RESULT, UNSUPPORTED])
    assert result == {
        "source": "gemini",
        "status": "generated",
        "model": "gemini-test",
        "summary": "Review the result with a clinician.",
        "recommendations": [
            "Discuss the finding with a clinician.",
            "Bring the report to your next visit.",
        ],
    }


def test_gemini_falls_back_to_next_model_when_primary_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "busy-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup-model")
    service = GeminiRecommendationService()
    service._client = FakeClient(failing={"busy-model"})
    result = service.generate([MI_RESULT])
    assert service._client.models.calls == ["busy-model", "backup-model"]
    assert result["source"] == "gemini"
    assert result["model"] == "backup-model"
