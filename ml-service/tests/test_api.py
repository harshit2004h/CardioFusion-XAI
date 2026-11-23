from fastapi.testclient import TestClient

from api import main as api_main
from api.dependencies import get_inference_service
from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_preserves_primary_risk():
    response = client.post(
        "/predict",
        json={
            "predictions": {
                "biomarkers": {"acute_mi": 0.89, "_metadata": {"targets": {"acute_mi": {"calibrated": True}}}},
                "ecg": {"MI": 0.82, "_metadata": {"targets": {"MI": {"calibrated": True}}}},
            }
        },
    )
    assert response.status_code == 200
    acute_mi = response.json()["results"]["acute_mi"]
    assert acute_mi["risk"] == 0.89
    assert acute_mi["status"] == "primary_only"
    assert acute_mi["supporting_evidence"]["ecg"]["probability"] == 0.82


def test_predict_endpoint_does_not_label_unverified_score_as_risk():
    response = client.post("/predict", json={"predictions": {"biomarkers": {"acute_mi": 0.89}}})
    assert response.status_code == 200
    acute_mi = response.json()["results"]["acute_mi"]
    assert acute_mi["risk"] is None
    assert acute_mi["model_score"] == 0.89


def test_predict_endpoint_does_not_infer_missing_primary():
    response = client.post(
        "/predict",
        json={"predictions": {"biomarkers": {"acute_mi": 0.89}}},
    )
    assert response.status_code == 200
    atrial_fibrillation = response.json()["results"]["atrial_fibrillation"]
    assert atrial_fibrillation["risk"] is None
    assert atrial_fibrillation["status"] == "primary_missing"


def test_versioned_registry_endpoint():
    response = client.get("/v1/registry")
    assert response.status_code == 200
    assert "diseases" in response.json()


def test_versioned_health_reports_runtime_status(monkeypatch):
    from types import SimpleNamespace

    manager = SimpleNamespace(device="cpu", models_loaded=True, status={"ecg": "loaded"})
    service = SimpleNamespace(gemini=SimpleNamespace(status="not_configured"))
    monkeypatch.setattr(api_main, "get_model_manager", lambda: manager)
    monkeypatch.setattr(api_main, "get_inference_service", lambda: service)

    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["service"] == "cardiofusion-ml"
    assert response.json()["models_loaded"] is True
    assert response.json()["models"] == {"ecg": "loaded"}


def test_versioned_inference_validates_and_returns_service_schema():
    class FakeInferenceService:
        def run(self, request):
            assert request.biomarkers is not None
            assert str(request.biomarkers.url).startswith("https://")
            return {
                "request_id": "cf_test",
                "service_version": "1.0.0",
                "status": "completed",
                "available_modalities": ["biomarkers"],
                "result_scope": "biomarkers-only",
                "modalities": {"biomarkers": {"available": True, "status": "processed", "details": {}}},
                "extraction": {},
                "predictions": {},
                "disease_results": [],
                "recommendations": [],
                "gemini_recommendations": {},
                "explainability": {},
                "warnings": [],
            }

    app.dependency_overrides[get_inference_service] = FakeInferenceService
    try:
        response = client.post(
            "/v1/inference",
            json={"biomarkers": {"url": "https://res.cloudinary.com/example/report.pdf"}},
        )
    finally:
        app.dependency_overrides.pop(get_inference_service, None)

    assert response.status_code == 200
    assert response.json()["request_id"] == "cf_test"
    assert response.json()["available_modalities"] == ["biomarkers"]


def test_versioned_inference_rejects_an_empty_request():
    response = client.post("/v1/inference", json={})
    assert response.status_code == 422
