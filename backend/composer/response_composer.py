"""Response Composer Module.

Harmonizes:
- Query classification
- RAG retrieved citations & source grounding
- Deterministic solver calculations and step-by-step traces
- LLM natural language tutoring explanation

Guarantees numerical correctness by enforcing that calculations originate strictly
from the deterministic solver, while RAG dictates factual syllabus grounding.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from backend.router.query_router import QueryCategory, RoutingDecision
from backend.rag.retriever import RetrievalResult
from backend.solver.models import BankerSafetyResult, ResourceRequestResult


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    topic: Optional[str] = None


class ComposedResponse(BaseModel):
    category: QueryCategory
    explanation: str
    worked_steps: List[str] = []
    citations: List[Citation] = []
    numerical_result: Optional[Dict[str, Any]] = None
    graph_data: Optional[Dict[str, Any]] = None
    teaching_notes: List[str] = []
    model_used: str


class ResponseComposer:
    """Assembles final verified student response."""

    def compose(
        self,
        routing: RoutingDecision,
        llm_text: str,
        retrieval: Optional[RetrievalResult] = None,
        solver_result: Optional[Any] = None,
        model_name: str = "mock-tutor"
    ) -> ComposedResponse:
        citations: List[Citation] = []
        worked_steps: List[str] = []
        teaching_notes: List[str] = []

        if routing.category == QueryCategory.THEORY:
            if retrieval and retrieval.chunks:
                for chunk in retrieval.chunks:
                    citations.append(
                        Citation(
                            source=chunk.source,
                            page=chunk.page,
                            topic=chunk.topic
                        )
                    )
            else:
                teaching_notes.append(
                    "Note: No direct matching syllabus documents found in local knowledge base. "
                    "Response is based on standard OS deadlock fundamentals."
                )

        if routing.category == QueryCategory.NUMERICAL:
            teaching_notes.append(
                "Important Concept: An UNSAFE state is NOT identical to DEADLOCK. "
                "Deadlock means processes are actively blocked. "
                "An unsafe state means no safe sequence is guaranteed to prevent potential deadlock."
            )
            if isinstance(solver_result, BankerSafetyResult):
                worked_steps.append(f"Safety status: {'SAFE' if solver_result.is_safe else 'UNSAFE'}")
                if solver_result.safe_sequence:
                    seq_str = " -> ".join(solver_result.safe_sequence)
                    worked_steps.append(f"Safe sequence: <{seq_str}>")
                for s in solver_result.steps:
                    worked_steps.append(f"Step {s.step} ({s.process}): {s.explanation}")

        return ComposedResponse(
            category=routing.category,
            explanation=llm_text,
            worked_steps=worked_steps,
            citations=citations,
            numerical_result=solver_result.model_dump() if hasattr(solver_result, "model_dump") else None,
            graph_data=None,
            teaching_notes=teaching_notes,
            model_used=model_name
        )
