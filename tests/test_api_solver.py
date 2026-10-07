"""Tests for Deadlock Solver API Endpoints.

Verifies:
- POST /api/solver/safety
- POST /api/solver/request
- POST /api/solver/detect/single
- POST /api/solver/detect/multi
- Bad request / validation error responses (HTTP 400)
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_solver_safety_endpoint(client):
    payload = {
        "processes": ["P0", "P1"],
        "resources": ["A", "B"],
        "allocation": [[0, 1], [1, 0]],
        "max_matrix": [[1, 2], [2, 1]],
        "available": [1, 1],
    }
    resp = client.post("/api/solver/safety", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_safe"] is True
    assert data["safe_sequence"] is not None
    assert data["need_matrix"] == [[1, 1], [1, 1]]


def test_api_solver_request_endpoint(client):
    payload = {
        "state": {
            "processes": ["P0", "P1"],
            "resources": ["A", "B"],
            "allocation": [[0, 1], [1, 0]],
            "max_matrix": [[1, 2], [2, 1]],
            "available": [1, 1],
        },
        "process_id": "P0",
        "request": [1, 0],
    }
    resp = client.post("/api/solver/request", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["granted"] is True
    assert data["status"] == "GRANTED"


def test_api_solver_detect_single_endpoint(client):
    # Cycle: P0 -> P1 -> P0
    payload = {
        "nodes": ["P0", "P1"],
        "edges": [["P0", "P1"], ["P1", "P0"]],
    }
    resp = client.post("/api/solver/detect/single", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_cycle"] is True
    assert data["is_deadlocked"] is True
    assert set(data["deadlocked_processes"]) == {"P0", "P1"}


def test_api_solver_detect_multi_endpoint(client):
    payload = {
        "processes": ["P0", "P1"],
        "resources": ["A"],
        "allocation": [[1], [1]],
        "request": [[1], [1]],
        "available": [0],
    }
    resp = client.post("/api/solver/detect/multi", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_deadlocked"] is True
    assert set(data["deadlocked_processes"]) == {"P0", "P1"}


def test_api_solver_invalid_input_400(client):
    # Allocation exceeds Max
    payload = {
        "processes": ["P0"],
        "resources": ["A"],
        "allocation": [[5]],
        "max_matrix": [[2]],
        "available": [1],
    }
    resp = client.post("/api/solver/safety", json=payload)
    assert resp.status_code == 422 or resp.status_code == 400
