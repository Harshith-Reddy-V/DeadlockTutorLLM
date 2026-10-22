"""Regression tests for grounding accuracy and Coffman condition verification.

Ensures:
1. When retrieved excerpts only mention/list condition names without defining them,
   the answer definition is NOT falsely marked as source-grounded.
2. The four necessary Coffman conditions are verified as prerequisites that permit
   deadlock, and are never falsely claimed to prevent or avoid deadlocks.
3. Factual inversions (e.g. 'mutual exclusion prevents deadlock') are caught by validation,
   marked ungrounded, and flagged with corrective teaching notes.
4. When retrieved excerpts DO provide substantive definitions, responses are marked grounded.
"""

import pytest
from backend.rag.ingestion import DocumentChunk
from backend.rag.retriever import BaseRetriever
from backend.rag.vector_store import NumpyVectorStore
from backend.rag.embeddings import DeterministicEmbedding
from backend.llm.provider import MockLLMProvider, BaseLLMProvider, LLMRequest, LLMResponse
from backend.orchestrator import (
    DeadlockTutorOrchestrator,
    ChatRequest,
    _check_condition_evidence,
    _detect_coffman_inversion,
    _sanitize_llm_response,
)
from backend.router.query_router import QueryRouter
from backend.composer.response_composer import ResponseComposer


@pytest.fixture
def notes_retriever(tmp_path):
    """Retriever loaded with the brief deadlock_notes.txt chunk (listing only)."""
    emb = DeterministicEmbedding(dim=64)
    store = NumpyVectorStore(dimension=64)
    retriever = BaseRetriever(
        top_k=3,
        relevance_threshold=0.20,
        embedding_model=emb,
        vector_store=store,
        index_path=tmp_path / "index.npy",
    )
    chunk = DocumentChunk(
        chunk_id="deadlock_notes_p1_c0",
        content=(
            "Deadlock Notes A deadlock is a situation in which a group of processes is permanently blocked "
            "because each process is waiting for a resource held by another process. "
            "The four necessary conditions for deadlock are mutual exclusion, hold and wait, no preemption, and circular wait. "
            "Banker's algorithm is a deadlock-avoidance algorithm. It checks whether granting a resource request leaves the system in a safe state. "
            "A safe state is a state in which there is at least one sequence of process execution that allows every process to finish. "
            "A wait-for graph represents processes as nodes. An edge from P1 to P2 indicates that P1 is waiting for a resource held by P2. "
            "A cycle in a single-instance wait-for graph indicates a deadlock."
        ),
        source="deadlock_notes.txt",
        page=1,
        doc_type="txt",
        topic="Deadlock Notes",
        citation="[Source: deadlock_notes.txt, page 1]",
    )
    retriever.index_chunks([chunk])
    return retriever


@pytest.fixture
def rich_retriever(tmp_path):
    """Retriever loaded with a textbook chunk that actually defines mutual exclusion."""
    emb = DeterministicEmbedding(dim=64)
    store = NumpyVectorStore(dimension=64)
    retriever = BaseRetriever(
        top_k=3,
        relevance_threshold=0.20,
        embedding_model=emb,
        vector_store=store,
        index_path=tmp_path / "rich_index.npy",
    )
    chunk = DocumentChunk(
        chunk_id="textbook_ch7_c1",
        content=(
            "Coffman Condition 1: Mutual Exclusion. At least one resource must be held in a non-shareable mode; "
            "that is, only one process at a time can use the resource. If another process requests that resource, "
            "the requesting process must be delayed until the resource has been released."
        ),
        source="Silberschatz_Deadlocks.pdf",
        page=312,
        doc_type="pdf",
        topic="Coffman Conditions",
        citation="[Source: Silberschatz_Deadlocks.pdf, page 312]",
    )
    retriever.index_chunks([chunk])
    return retriever


class InvertedHallucinatingLLM(BaseLLMProvider):
    """Stub LLM returning the exact mutual exclusion misconception for testing validation."""
    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=(
                "The mutual exclusion condition is essential for preventing deadlock because it ensures "
                "that only one process can access a resource at a time, avoiding a deadlock state."
            ),
            model_name="hallucinating-model",
            provider="test",
        )


def test_mutual_exclusion_unsupported_definition_not_grounded(notes_retriever):
    """When KB only lists 'mutual exclusion' without defining it, answer must NOT be marked grounded."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=MockLLMProvider(),
        retriever=notes_retriever,
    )
    req = ChatRequest(query="What is mutual exclusion in deadlock?")
    resp = orchestrator.handle(req)

    # 1. Grounded must be False because the chunk has no definition of mutual exclusion
    assert resp.grounded is False
    assert "do not contain its definition" in resp.groundedness_message
    assert any("Course grounding note" in note for note in resp.teaching_notes)

    # 2. Answer must correctly state the necessary condition and not claim it prevents deadlock
    assert "Mutual Exclusion" in resp.answer
    assert not _detect_coffman_inversion(resp.answer)[0]


def test_all_four_conditions_unsupported_definitions_not_grounded(notes_retriever):
    """Verify that all four conditions without definitions in notes are marked ungrounded."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=MockLLMProvider(),
        retriever=notes_retriever,
    )
    conditions = [
        ("hold and wait", "What is hold and wait in deadlock?"),
        ("no preemption", "Explain no preemption in deadlock."),
        ("circular wait", "Explain circular wait in deadlock."),
    ]
    for cond_name, q in conditions:
        resp = orchestrator.handle(ChatRequest(query=q))
        assert resp.grounded is False, f"Expected grounded=False for {cond_name}"
        assert f"'{cond_name}'" in resp.groundedness_message
        assert "do not contain its definition" in resp.groundedness_message


def test_four_conditions_listing_is_grounded(notes_retriever):
    """When query asks to list the four conditions, deadlock_notes.txt DOES answer it, so grounded=True."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=MockLLMProvider(),
        retriever=notes_retriever,
    )
    req = ChatRequest(query="What are the four necessary Coffman conditions for deadlock?")
    resp = orchestrator.handle(req)

    # Listing the 4 conditions is genuinely supported by the chunk in deadlock_notes.txt
    assert resp.grounded is True
    assert len(resp.sources) >= 1
    assert any("deadlock_notes.txt" in s.source for s in resp.sources)


def test_rich_chunk_supported_definition_is_grounded(rich_retriever):
    """When KB contains an actual definition of mutual exclusion, answer SHOULD be marked grounded."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=MockLLMProvider(),
        retriever=rich_retriever,
    )
    req = ChatRequest(query="What is mutual exclusion in deadlock?")
    resp = orchestrator.handle(req)

    # Rich chunk contains "non-shareable mode" and defines it
    assert resp.grounded is True
    assert any("Silberschatz_Deadlocks.pdf" in s.source for s in resp.sources)


def test_factual_inversion_detected_and_ungrounded(notes_retriever):
    """If an LLM hallucinates that mutual exclusion prevents deadlock, validation must catch it."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=InvertedHallucinatingLLM(),
        retriever=notes_retriever,
    )
    req = ChatRequest(query="Explain mutual exclusion.")
    resp = orchestrator.handle(req)

    # Factual error must force grounded=False and add correction note
    assert resp.grounded is False
    assert any("Factual correction" in note or "Factual Misconception" in note for note in resp.teaching_notes)
    has_inv, msg = _detect_coffman_inversion(resp.answer)
    assert has_inv is True


def test_does_mutual_exclusion_prevent_deadlock_accuracy(notes_retriever):
    """Verifies that asking whether mutual exclusion prevents deadlock yields a clear negative answer."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=MockLLMProvider(),
        retriever=notes_retriever,
    )
    req = ChatRequest(query="Does mutual exclusion prevent deadlock?")
    resp = orchestrator.handle(req)

    assert "NOT prevent deadlock" in resp.answer or "does not prevent deadlock" in resp.answer.lower()
    assert not _detect_coffman_inversion(resp.answer)[0]


def test_inversion_distinguishes_elimination_from_enforcement():
    """Verify that eliminating/breaking a condition to prevent deadlock is valid, while claiming the condition itself prevents deadlock is caught."""
    valid_prevention_statements = [
        "Eliminating mutual exclusion can prevent deadlock.",
        "By eliminating mutual exclusion, the system prevents deadlock.",
        "Breaking hold and wait prevents deadlock.",
        "Negating circular wait avoids deadlock.",
        "Removing no preemption avoids deadlock.",
        "Spooling eliminates mutual exclusion to prevent deadlock.",
        "Mutual exclusion does not prevent deadlock.",
        "It is not true that mutual exclusion prevents deadlock.",
        "Deadlock is not prevented by mutual exclusion.",
        "Mutual exclusion is a necessary condition for deadlock to occur.",
    ]
    for text in valid_prevention_statements:
        has_inv, msg = _detect_coffman_inversion(text)
        assert not has_inv, f"False positive inversion detected on valid OS statement: '{text}'"

    invalid_inversion_statements = [
        "Mutual exclusion prevents deadlock.",
        "The mutual exclusion condition is essential for preventing deadlock.",
        "Enforcing mutual exclusion will avoid deadlock.",
        "Hold and wait prevents deadlock by keeping resources busy.",
        "No preemption prevents deadlock in an operating system.",
        "Circular wait prevents deadlock.",
        "Deadlock is prevented by mutual exclusion.",
        "Mutual exclusion ensures no circular wait.",
    ]
    for text in invalid_inversion_statements:
        has_inv, msg = _detect_coffman_inversion(text)
        assert has_inv, f"Expected inversion detected on false claim: '{text}'"
        assert msg is not None


def test_correction_visibility_in_main_answer(notes_retriever):
    """When an inversion occurs, the correction banner must be prominently visible in resp.answer."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=InvertedHallucinatingLLM(),
        retriever=notes_retriever,
    )
    req = ChatRequest(query="Explain mutual exclusion.")
    resp = orchestrator.handle(req)

    assert resp.grounded is False
    # Prominent callout alert in the main answer
    assert "> ⚠️ **Factual Correction & Conceptual Alert**:" in resp.answer
    assert "Mutual exclusion is a necessary condition FOR deadlock to occur" in resp.answer
    assert "---" in resp.answer


def test_prompt_leakage_sanitization():
    """Ensures prompt scaffolding headings ('Step 7', 'Step 8', 'Core OS Ground Truth') are removed without stripping legitimate educational steps."""
    leaked_output = (
        "Certainly! Let's break down mutual exclusion in a step-by-step manner.\n\n"
        "### Step 1: Understanding Mutual Exclusion\n"
        "Mutual exclusion means resources cannot be shared simultaneously.\n\n"
        "### Step 2: Banker's Algorithm Trace\n"
        "Step 1: Calculate Need matrix.\n"
        "Step 2: Check available vector.\n\n"
        "### Step 7: Mutual Exclusion and Deadlock Prevention in RAG Questions\n"
        "In the context of RAG questions, it is important to clarify that mutual exclusion does not prevent deadlock.\n\n"
        "### Step 8: Core OS Ground Truth\n"
        "The four Coffman conditions are necessary conditions for deadlock to occur.\n\n"
        "### Conclusion\n"
        "Mutual exclusion permits deadlock to occur."
    )
    sanitized = _sanitize_llm_response(leaked_output)

    # Scaffolding headers must be removed
    assert "Step 7" not in sanitized
    assert "Step 8" not in sanitized
    assert "Core OS Ground Truth" not in sanitized
    assert "in RAG Questions" not in sanitized
    assert "In the context of RAG questions" not in sanitized

    # Legitimate educational steps must be preserved
    assert "Step 1: Understanding Mutual Exclusion" in sanitized
    assert "Step 2: Banker's Algorithm Trace" in sanitized
    assert "Step 1: Calculate Need matrix" in sanitized
    assert "Step 2: Check available vector" in sanitized


def test_grounding_scope_missing_topic_ungrounded(notes_retriever):
    """When a query asks about a topic absent from the notes (such as 'deadlock prevention'), it must be marked ungrounded despite high keyword similarity."""
    orchestrator = DeadlockTutorOrchestrator(
        llm_provider=MockLLMProvider(),
        retriever=notes_retriever,
    )
    req = ChatRequest(query="What is deadlock prevention?")
    resp = orchestrator.handle(req)

    assert resp.grounded is False
    assert "prevention" in resp.groundedness_message.lower()
    assert "standard os theory" in resp.groundedness_message.lower()

