from src.recommendations.gemini import GeminiRecommendationService


class FakeResponse:
    text = '{"summary":"Review the result with a clinician.","recommendations":["Discuss the finding with a clinician.","Bring the report to your next visit."]}'


class FakeModels:
    def generate_content(self, *, model, contents):
        assert model == "gemini-test"
        assert "Acute Myocardial Infarction" in contents
        assert "raw_text" not in contents
        return FakeResponse()


class FakeClient:
    models = FakeModels()


def test_gemini_context_whitelists_prediction_fields():
    context = GeminiRecommendationService._safe_context(
        [
            {
                "disease": "acute_mi",
                "display_name": "Acute Myocardial Infarction",
                "risk": 0.82,
                "confidence": "MODERATE",
                "status": "primary_only",
                "primary_modality": "biomarkers",
                "raw_text": "patient private report text",
            }
        ]
    )
    assert context == [
        {
            "disease": "acute_mi",
            "display_name": "Acute Myocardial Infarction",
            "risk": 0.82,
            "confidence": "MODERATE",
            "status": "primary_only",
            "primary_modality": "biomarkers",
            "detected_from": [],
        }
    ]


def test_gemini_fallback_is_safe_without_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = GeminiRecommendationService()
    result = service.generate(
        [{
            "disease": "atrial_fibrillation",
            "display_name": "Atrial Fibrillation",
            "risk": 0.76,
            "confidence": "LOW",
            "status": "primary_only",
        }]
    )
    assert result["source"] == "local_fallback"
    assert result["recommendations"]
    assert "diagnosis" in result["summary"]


def test_gemini_success_returns_structured_recommendations(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    service = GeminiRecommendationService()
    service._client = FakeClient()
    result = service.generate(
        [{
            "disease": "acute_mi",
            "display_name": "Acute Myocardial Infarction",
            "risk": 0.82,
            "confidence": "MODERATE",
            "status": "primary_only",
        }]
    )
    assert result == {
        "source": "gemini",
        "status": "generated",
        "summary": "Review the result with a clinician.",
        "recommendations": [
            "Discuss the finding with a clinician.",
            "Bring the report to your next visit.",
        ],
    }