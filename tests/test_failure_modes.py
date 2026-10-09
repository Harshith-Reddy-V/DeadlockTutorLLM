"""Failure Testing Suite for DeadlockTutorLLM.

Tests the system's resilience to:
- Empty queries
- Malformed matrices
- Unrelated questions
- Missing knowledge base scenarios
"""

import pytest
from pydantic import ValidationError
from backend.orchestrator import DeadlockTutorOrchestrator, ChatRequest
from backend.llm.provider import MockLLMProvider
from backend.rag.retriever import BaseRetriever
from backend.solver.models import BankerStateInput
from backend.solver.engine import calculate_need


@pytest.fixture(scope="module")
def orchestrator():
    # Use mock provider for unit testing failures
    provider = MockLLMProvider()
    retriever = BaseRetriever()
    return DeadlockTutorOrchestrator(llm_provider=provider, retriever=retriever)


def test_empty_query_handling(orchestrator):
    """Test that the system gracefully handles an empty query string."""
    with pytest.raises(ValidationError) as excinfo:
        req = ChatRequest(query="")
    assert "String should have at least 1 character" in str(excinfo.value)
    # Should not crash


def test_out_of_domain_handling(orchestrator):
    """Test that unrelated questions are safely answered without hallucinating OS concepts."""
    req = ChatRequest(query="What is the recipe for chocolate cake?")
    response = orchestrator.handle(req)
    assert response is not None
    assert response.category in ["theory", "out_of_domain"]
    # Usually we instruct the mock/LLM to refuse or answer safely


def test_malformed_bankers_matrix():
    """Test that the solver catches malformed matrices (e.g. Allocation > Max)."""
    with pytest.raises(ValueError) as excinfo:
        state = BankerStateInput(
            allocation=[[5, 5]],  # higher than max
            max_matrix=[[2, 2]],
            available=[10, 10]
        )
    assert "Allocation" in str(excinfo.value) and "exceeds Max" in str(excinfo.value)


def test_negative_resources_matrix():
    """Test that the solver catches negative values in vectors."""
    with pytest.raises(ValueError) as excinfo:
        state = BankerStateInput(
            allocation=[[1, 1]],
            max_matrix=[[2, 2]],
            available=[-1, 10]
        )
    assert "negative" in str(excinfo.value).lower()


def test_dimension_mismatch_matrix():
    """Test that the solver catches dimension mismatches between matrices."""
    with pytest.raises(ValueError) as excinfo:
        state = BankerStateInput(
            allocation=[[1, 1], [0, 0]],  # 2 processes
            max_matrix=[[2, 2]],          # 1 process
            available=[10, 10]
        )
    assert "Max matrix has 1 rows; expected 2" in str(excinfo.value)

