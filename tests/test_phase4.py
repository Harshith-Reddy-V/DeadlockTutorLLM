"""Tests for Phase 4 - LLM Integration and Orchestration Pipeline."""

import pytest
from fastapi.testclient import TestClient

from backend.llm.provider import MockLLMProvider, LLMRequest
from backend.router.query_router import QueryRouter, QueryCategory
from backend.composer.response_composer import ResponseComposer
from backend.orchestrator import DeadlockTutorOrchestrator, ChatRequest
from backend.rag.retriever import BaseRetriever
from backend.solver.models import (
    BankerStateInput,
    SingleInstanceDetectionInput
)
from main import app


# ---------------------------------------------------------------------------
# 1. LLM Provider Tests
# ---------------------------------------------------------------------------

def test_mock_llm_provider_theory():
    provider = MockLLMProvider("test-mock")
    req = LLMRequest(prompt="What are the four Coffman conditions?")
    resp = provider.generate(req)
    
    assert resp.provider == "mock"
    assert resp.model_name == "test-mock"
    assert "Coffman Conditions" in resp.content
    assert "Mutual Exclusion" in resp.content


def test_mock_llm_provider_fallback():
    provider = MockLLMProvider()
    req = LLMRequest(prompt="Hello, what can you do?")
    resp = provider.generate(req)
    assert "Key OS deadlock topics" in resp.content


# ---------------------------------------------------------------------------
# 2. Query Router Tests
# ---------------------------------------------------------------------------

def test_query_router_comprehensive():
    router = QueryRouter()
    
    # Theory
    assert router.route("What is a deadlock?").category == QueryCategory.THEORY
    assert router.route("Explain Coffman conditions").category == QueryCategory.THEORY
    
    # Numerical
    assert router.route("Apply Banker's algorithm").category == QueryCategory.NUMERICAL
    assert router.route("Given the allocation matrix").category == QueryCategory.NUMERICAL
    assert router.route("Determine if the state is safe").category == QueryCategory.NUMERICAL
    assert router.route("Calculate the Need matrix").category == QueryCategory.NUMERICAL
    
    # Graph
    assert router.route("Draw the wait-for graph").category == QueryCategory.GRAPH
    assert router.route("Detect a cycle in the graph").category == QueryCategory.GRAPH
    
    # Lab
    assert router.route("Write a C program with pthreads").category == QueryCategory.LAB
    assert router.route("Dining philosophers deadlock simulation").category == QueryCategory.LAB


# ---------------------------------------------------------------------------
# 3. Response Composer Tests
# ---------------------------------------------------------------------------

def test_response_composer_theory_ungrounded():
    composer = ResponseComposer()
    router = QueryRouter()
    routing = router.route("What is deadlock?")
    
    # Ungrounded (no retrieval)
    resp = composer.compose(
        routing=routing,
        llm_text="Deadlock is...",
        retrieval=None
    )
    assert resp.category == QueryCategory.THEORY
    assert len(resp.citations) == 0
    assert any("not contain sufficient course-specific material" in note for note in resp.teaching_notes)


def test_response_composer_numerical_notes():
    composer = ResponseComposer()
    router = QueryRouter()
    routing = router.route("Apply Banker's algorithm")
    
    resp = composer.compose(
        routing=routing,
        llm_text="Here is the explanation...",
        solver_result=None
    )
    assert resp.category == QueryCategory.NUMERICAL
    # Numerical ALWAYS includes the unsafe != deadlock note
    assert any("UNSAFE state \u2260 DEADLOCK" in note for note in resp.teaching_notes)


# ---------------------------------------------------------------------------
# 4. Orchestrator Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_retriever():
    return BaseRetriever(embedding_model=None, vector_store=None)

@pytest.fixture
def mock_llm():
    return MockLLMProvider()

@pytest.fixture
def orchestrator(mock_llm, empty_retriever):
    return DeadlockTutorOrchestrator(llm_provider=mock_llm, retriever=empty_retriever)


def test_orchestrator_theory_flow(orchestrator):
    req = ChatRequest(query="What are Coffman conditions?")
    resp = orchestrator.handle(req)
    
    assert resp.category == QueryCategory.THEORY.value
    assert "Coffman Conditions" in resp.answer
    assert resp.grounded is False
    assert "Knowledge base contains no indexed documents" in resp.groundedness_message


def test_orchestrator_numerical_bankers_flow(orchestrator):
    state = BankerStateInput(
        processes=["P1"],
        resources=["A"],
        allocation=[[0]],
        max_matrix=[[1]],
        available=[1]
    )
    req = ChatRequest(query="Apply Banker's algorithm to this state", banker_state=state)
    resp = orchestrator.handle(req)
    
    assert resp.category == QueryCategory.NUMERICAL.value
    assert resp.grounded is True  # Grounded by solver
    assert resp.solver_result is not None
    assert resp.solver_result["is_safe"] is True
    # Verify trace is passed to composer
    assert len(resp.worked_steps) > 0


def test_orchestrator_graph_flow(orchestrator):
    graph = SingleInstanceDetectionInput(
        nodes=["P1", "P2"],
        edges=[["P1", "P2"], ["P2", "P1"]]
    )
    req = ChatRequest(query="Detect cycle in this graph", single_instance_graph=graph)
    resp = orchestrator.handle(req)
    
    assert resp.category == QueryCategory.GRAPH.value
    assert resp.grounded is True
    assert resp.solver_result is not None
    assert resp.solver_result["is_deadlocked"] is True
    # The deterministic solver returns the path closing the cycle: P1 -> P2 -> P1
    assert ["P1", "P2", "P1"] in resp.solver_result["cycles"] or ["P2", "P1", "P2"] in resp.solver_result["cycles"]


# ---------------------------------------------------------------------------
# 5. Chat API Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def test_chat_api_theory_mock(client):
    payload = {"query": "Explain deadlock"}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["category"] == "theory"
    assert data["grounded"] is False
    assert "theory" in data["category"]
    assert "mock" in data["llm_provider"]


def test_chat_api_numerical_mock(client):
    payload = {
        "query": "Apply Banker's algorithm",
        "banker_state": {
            "processes": ["P1"],
            "resources": ["A"],
            "allocation": [[0]],
            "max_matrix": [[1]],
            "available": [1]
        }
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["category"] == "numerical"
    assert data["grounded"] is True
    assert data["solver_result"]["is_safe"] is True
    assert "mock" in data["llm_provider"]


def test_chat_api_invalid_request(client):
    # Missing required query field
    payload = {"message": "wrong field"}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 422
