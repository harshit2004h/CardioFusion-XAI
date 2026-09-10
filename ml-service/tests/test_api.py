from fastapi.testclient import TestClient

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
                "biomarkers": {"acute_mi": 0.89},
                "ecg": {"MI": 0.82},
            }
        },
    )
    assert response.status_code == 200
    acute_mi = response.json()["results"]["acute_mi"]
    assert acute_mi["risk"] == 0.89
    assert acute_mi["status"] == "agreement"


def test_predict_endpoint_does_not_infer_missing_primary():
    response = client.post(
        "/predict",
        json={"predictions": {"biomarkers": {"acute_mi": 0.89}}},
    )
    assert response.status_code == 200
    atrial_fibrillation = response.json()["results"]["atrial_fibrillation"]
    assert atrial_fibrillation["risk"] is None
    assert atrial_fibrillation["status"] == "primary_missing"
