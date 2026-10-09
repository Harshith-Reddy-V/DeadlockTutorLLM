"""Response Composer Module.

Harmonizes:
  - Query classification (category)
  - RAG-retrieved citations & source grounding
  - Deterministic solver calculations and step-by-step traces
  - LLM natural language tutoring explanation

Guarantees numerical correctness by enforcing that calculations originate
strictly from the deterministic solver, while RAG dictates factual syllabus
grounding.

Teaching style guide (applied in compose()):
  Numerical  : Given → Need → Initial State → Step-by-Step → Result → Explanation
  Theory     : Definition → Core Concept → Example → Exam Points → Source
  Graph      : Graph Description → Cycle Trace → Interpretation → Exam Notes
  Lab        : Problem → Code Example → Prevention → Lab Reference
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel

from backend.router.query_router import QueryCategory, RoutingDecision
from backend.rag.retriever import RetrievalResult
from backend.solver.models import (
    BankerSafetyResult,
    ResourceRequestResult,
    SingleInstanceDetectionResult,
    MultiInstanceDetectionResult,
)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    doc_type: Optional[str] = None
    topic: Optional[str] = None
    citation_string: Optional[str] = None


class ComposedResponse(BaseModel):
    """Final assembled student response."""
    category: QueryCategory
    explanation: str
    worked_steps: List[str] = []
    citations: List[Citation] = []
    numerical_result: Optional[Any] = None
    graph_data: Optional[Any] = None
    graph_diagram: Optional[str] = None
    teaching_notes: List[str] = []
    model_used: str


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

class ResponseComposer:
    """Assembles the final verified student response from all pipeline outputs."""

    def compose(
        self,
        routing: RoutingDecision,
        llm_text: str,
        retrieval: Optional[RetrievalResult] = None,
        solver_result: Optional[Any] = None,
        model_name: str = "mock-tutor",
    ) -> ComposedResponse:
        """Combine RAG context, solver output, and LLM text into a structured response.

        Args:
            routing:       The routing decision (category + reason).
            llm_text:      Generated explanation from the LLM.
            retrieval:     Optional RAG retrieval result with chunks and citations.
            solver_result: Optional output from the deterministic solver.
            model_name:    Name of the model that generated llm_text.

        Returns:
            ComposedResponse ready for API serialization.
        """
        category = routing.category
        citations: List[Citation] = []
        worked_steps: List[str] = []
        teaching_notes: List[str] = []
        numerical_result = None
        graph_data = None
        graph_diagram = None

        # ------------------------------------------------------------------
        # 1. Ingest citations from RAG retrieval
        # ------------------------------------------------------------------
        if retrieval and retrieval.chunks:
            seen_cites: set[str] = set()
            for chunk in retrieval.chunks:
                key = chunk.citation
                if key not in seen_cites:
                    seen_cites.add(key)
                    citations.append(Citation(
                        source=chunk.source,
                        page=chunk.page,
                        doc_type=chunk.doc_type,
                        topic=chunk.topic,
                        citation_string=chunk.citation,
                    ))

        # ------------------------------------------------------------------
        # 2. Process solver output by category
        # ------------------------------------------------------------------

        if category == QueryCategory.NUMERICAL:
            # Core misconception note — always included for numerical questions
            teaching_notes.append(
                "📌 Key distinction: An UNSAFE state ≠ DEADLOCK. "
                "An unsafe state means no safe sequence can be guaranteed if all processes request "
                "their maximum resources. Deadlock means processes are CURRENTLY blocked."
            )

            if isinstance(solver_result, BankerSafetyResult):
                numerical_result = solver_result.model_dump()
                state_label = "✅ SAFE" if solver_result.is_safe else "⚠️ UNSAFE"
                worked_steps.append(f"**State: {state_label}**")

                if solver_result.safe_sequence:
                    seq_str = " → ".join(solver_result.safe_sequence)
                    worked_steps.append(f"**Safe sequence:** ⟨{seq_str}⟩")

                worked_steps.append(f"**Initial Available:** {solver_result.initial_available}")
                worked_steps.append(f"**Need matrix:** {solver_result.need_matrix}")

                for step in solver_result.steps:
                    worked_steps.append(
                        f"**Step {step.step} ({step.process}):** {step.explanation}"
                    )

                if solver_result.unfinishable_processes:
                    worked_steps.append(
                        f"**Unfinishable processes:** {solver_result.unfinishable_processes}"
                    )

            elif isinstance(solver_result, ResourceRequestResult):
                numerical_result = solver_result.model_dump()
                status_icon = "✅" if solver_result.granted else "❌"
                worked_steps.append(f"**Request status: {status_icon} {solver_result.status}**")
                worked_steps.append(f"**Reason:** {solver_result.reason}")
                worked_steps.append(f"**Request ≤ Need:** {solver_result.request_valid}")
                worked_steps.append(f"**Request ≤ Available:** {solver_result.available_sufficient}")
                for line in solver_result.trace:
                    worked_steps.append(line)

            elif isinstance(solver_result, MultiInstanceDetectionResult):
                numerical_result = solver_result.model_dump()
                state_icon = "🔴" if solver_result.is_deadlocked else "✅"
                worked_steps.append(
                    f"**Detection result: {state_icon} {'DEADLOCK' if solver_result.is_deadlocked else 'NO DEADLOCK'}**"
                )
                if solver_result.deadlocked_processes:
                    worked_steps.append(f"**Deadlocked processes:** {solver_result.deadlocked_processes}")
                for step in solver_result.steps:
                    worked_steps.append(f"**Step {step.step} ({step.process}):** {step.explanation}")

        elif category == QueryCategory.GRAPH:
            if isinstance(solver_result, SingleInstanceDetectionResult):
                graph_data = solver_result.model_dump()
                state_icon = "🔴" if solver_result.is_deadlocked else "✅"
                worked_steps.append(
                    f"**Cycle detection: {state_icon} {'DEADLOCK' if solver_result.is_deadlocked else 'NO DEADLOCK'}**"
                )
                if solver_result.cycles:
                    for i, cycle in enumerate(solver_result.cycles, 1):
                        cycle_str = " → ".join(cycle) + " → " + cycle[0]
                        worked_steps.append(f"**Cycle {i}:** {cycle_str}")
                if solver_result.deadlocked_processes:
                    worked_steps.append(f"**Deadlocked processes:** {solver_result.deadlocked_processes}")
                for line in solver_result.trace:
                    worked_steps.append(line)
                teaching_notes.append(
                    "📌 In single-instance resource systems: deadlock ↔ cycle exists in the Wait-For Graph."
                )

                mermaid_lines = ["graph TD"]
                for edge in solver_result.input_edges:
                    u, v = edge[0], edge[1]
                    in_cycle = False
                    for cycle in solver_result.cycles:
                        for i in range(len(cycle)):
                            if u == cycle[i] and v == cycle[(i+1)%len(cycle)]:
                                in_cycle = True
                                break
                        if in_cycle: break
                    
                    if in_cycle:
                        mermaid_lines.append(f"  {u} -->|waits for| {v}")
                        mermaid_lines.append(f"  style {u} fill:#f99,stroke:#333,stroke-width:2px")
                        mermaid_lines.append(f"  style {v} fill:#f99,stroke:#333,stroke-width:2px")
                    else:
                        mermaid_lines.append(f"  {u} -->|waits for| {v}")
                
                for node in solver_result.input_nodes:
                    if not any(node in edge for edge in solver_result.input_edges):
                        mermaid_lines.append(f"  {node}")

                graph_diagram = "\n".join(mermaid_lines)

            elif isinstance(solver_result, MultiInstanceDetectionResult):
                graph_data = solver_result.model_dump()
                state_icon = "🔴" if solver_result.is_deadlocked else "✅"
                worked_steps.append(
                    f"**Multi-instance detection: {state_icon} {'DEADLOCK' if solver_result.is_deadlocked else 'NO DEADLOCK'}**"
                )
                if solver_result.deadlocked_processes:
                    worked_steps.append(f"**Deadlocked processes:** {solver_result.deadlocked_processes}")
                teaching_notes.append(
                    "📌 In multi-instance resource systems, a cycle in the RAG is necessary but "
                    "NOT sufficient for deadlock. Use matrix-based detection."
                )

        elif category == QueryCategory.THEORY:
            if not retrieval or not retrieval.has_sufficient_context:
                teaching_notes.append(
                    "⚠️ The available knowledge base does not contain sufficient course-specific material "
                    "for this query. The response below is based on standard OS textbook content and may not "
                    "reflect your specific course syllabus."
                )
            else:
                teaching_notes.append(
                    "✅ This response is grounded in your course materials. See citations below."
                )

        elif category == QueryCategory.LAB:
            if not retrieval or not retrieval.has_sufficient_context:
                teaching_notes.append(
                    "⚠️ No specific lab material was found in the knowledge base. "
                    "The code examples below use standard POSIX pthreads. "
                    "Always verify specific lab API requirements with your course instructor."
                )
            else:
                teaching_notes.append(
                    "✅ Lab context retrieved from course materials. See citations for source references."
                )

        return ComposedResponse(
            category=category,
            explanation=llm_text,
            worked_steps=worked_steps,
            citations=citations,
            numerical_result=numerical_result,
            graph_data=graph_data,
            graph_diagram=graph_diagram,
            teaching_notes=teaching_notes,
            model_used=model_name,
        )
