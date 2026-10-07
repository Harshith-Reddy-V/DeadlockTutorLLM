import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "DeadlockTutorLLM" in data["message"]
    assert data["docs_url"] == "/docs"


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "DeadlockTutorLLM"
    assert data["llm_provider"] == "mock"


def test_chat_endpoint_theory_flow(client):
    payload = {
        "query": "Explain the Coffman conditions for deadlock."
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "theory"
    assert "Coffman Conditions" in data["answer"]
    assert "model_used" in data


def test_banker_safety_endpoint(client):
    payload = {
        "processes": ["P0", "P1"],
        "resources": ["A", "B"],
        "allocation": [[0, 1], [1, 0]],
        "max_matrix": [[1, 2], [2, 1]],
        "available": [1, 1]
    }
    response = client.post("/api/solver/banker-safety", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_safe"] is True
    assert data["need_matrix"] == [[1, 1], [1, 1]]
