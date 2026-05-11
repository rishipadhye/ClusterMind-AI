from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PAYLOAD = {
    "model_type": "gpt",
    "framework": "pytorch",
    "gpu_type": "a100",
    "num_gpus": 4,
    "batch_size": 64,
    "dataset_size": 2400000,
    "cpu_cores": 32,
    "ram_gb": 128,
    "estimated_flops": 5.2e14,
    "distributed_training": True,
    "mixed_precision": True,
}


def test_predict_runtime_endpoint() -> None:
    response = client.post("/api/v1/predict/runtime", json=PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "value" in data and data["value"] > 0


def test_predict_failure_endpoint() -> None:
    response = client.post("/api/v1/predict/failure", json=PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["probability"] <= 1


def test_recommendation_endpoint() -> None:
    response = client.post("/api/v1/recommend/resources", json=PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "optimization_suggestions" in data

