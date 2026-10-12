"""Orchestration Layer — End-to-End DeadlockTutorLLM Pipeline.

This module implements the central request orchestrator that:
  1. Receives a student query
  2. Classifies it (QueryRouter)
  3. Dispatches to the appropriate backend (RAG retriever / Deterministic solver)
  4. Assembles a grounded prompt for the LLM
  5. Calls the LLM provider
  6. Composes and returns the final structured response

Design contracts:
  - THEORY / LAB  : Factual content ONLY from RAG. If KB is insufficient, the response
                    explicitly states that it cannot be grounded in the course material.
  - NUMERICAL     : Numbers ALWAYS from the deterministic solver. The LLM explains the
                    solver's trace — it must NEVER independently recalculate.
  - GRAPH         : Graph topology from the deterministic cycle detector. LLM explains
                    what the graph means and why cycles imply deadlock.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.router.query_router import QueryCategory, QueryRouter, RoutingDecision
from backend.rag.retriever import BaseRetriever, RetrievalResult
from backend.solver.engine import (
    safety_check,
    request_check,
    detect_single_instance,
    detect_multi_instance,
)
from backend.solver.models import (
    BankerStateInput,
    BankerSafetyResult,
    ResourceRequestInput,
    ResourceRequestResult,
    SingleInstanceDetectionInput,
    SingleInstanceDetectionResult,
    MultiInstanceDetectionInput,
    MultiInstanceDetectionResult,
)
from backend.llm.provider import BaseLLMProvider, LLMRequest, LLMResponse
from backend.composer.response_composer import ResponseComposer, ComposedResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chat request/response schemas (used by the API layer)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Student chat request — supports plain text and optional structured solver input."""

    query: str = Field(..., min_length=1, description="Student's question or problem statement")

    # Optional structured inputs for numerical/graph queries
    banker_state: Optional[BankerStateInput] = Field(
        default=None,
        description="Structured Banker's algorithm state (for numerical safety queries)"
    )
    resource_request: Optional[ResourceRequestInput] = Field(
        default=None,
        description="Structured resource-request input (for Banker's request queries)"
    )
    single_instance_graph: Optional[SingleInstanceDetectionInput] = Field(
        default=None,
        description="Wait-for graph for single-instance cycle detection"
    )
    multi_instance_detection: Optional[MultiInstanceDetectionInput] = Field(
        default=None,
        description="Matrix input for multi-instance deadlock detection"
    )


class SourceCitation(BaseModel):
    """A single source citation from the RAG knowledge base."""
    source: str
    page: Optional[int] = None
    doc_type: Optional[str] = None
    citation_string: str


class ChatResponse(BaseModel):
    """Structured response returned to the student."""
    query: str
    category: str
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    solver_result: Optional[Dict[str, Any]] = None
    graph_diagram: Optional[str] = None
    grounded: bool
    groundedness_message: str
    teaching_notes: List[str] = Field(default_factory=list)
    worked_steps: List[str] = Field(default_factory=list)
    model_used: str
    llm_provider: str


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------

def _build_theory_prompt(query: str, retrieval: RetrievalResult) -> str:
    """Assemble a grounded theory prompt using RAG context."""
    if retrieval.has_sufficient_context and retrieval.chunks:
        context_text = "\n\n---\n\n".join(
            f"[{chunk.citation}]\n{chunk.content}"
            for chunk in retrieval.chunks[:settings.top_k_retrieval]
        )
        return (
            f"The student asks: {query}\n\n"
            f"Use ONLY the following excerpts from the course materials to answer. "
            f"Cite the sources. Do not add facts not present in the excerpts.\n\n"
            f"=== COURSE MATERIAL EXCERPTS ===\n{context_text}\n"
            f"=== END OF EXCERPTS ===\n\n"
            f"Now explain the concept clearly in a student-friendly, step-by-step manner."
        )
    else:
        return (
            f"The student asks: {query}\n\n"
            f"WARNING: The knowledge base does NOT contain sufficient course material "
            f"to ground this answer. Provide a general OS deadlock explanation but explicitly "
            f"state that it is not grounded in the student's specific course materials.\n\n"
            f"Explain the topic at a general university level."
        )


def _build_numerical_prompt(query: str, solver_result: Any) -> str:
    """Assemble a numerical prompt that includes the solver's verified trace."""
    if solver_result is None:
        return (
            f"The student asks a numerical deadlock question: {query}\n\n"
            f"No structured matrix data was provided. Ask the student to provide "
            f"the Allocation matrix, Max matrix, and Available vector so the deterministic "
            f"solver can compute the exact result."
        )

    # Serialize the trace for the LLM
    trace_lines = ""
    if hasattr(solver_result, "trace_log") and solver_result.trace_log:
        trace_lines = "\n".join(solver_result.trace_log)
    elif hasattr(solver_result, "trace") and solver_result.trace:
        trace_lines = "\n".join(solver_result.trace)

    result_summary = ""
    if isinstance(solver_result, BankerSafetyResult):
        state_word = "SAFE" if solver_result.is_safe else "UNSAFE"
        seq = " → ".join(solver_result.safe_sequence) if solver_result.safe_sequence else "None"
        result_summary = (
            f"State: {state_word}\n"
            f"Safe sequence: <{seq}>\n"
            f"Need matrix: {solver_result.need_matrix}\n"
        )
    elif isinstance(solver_result, ResourceRequestResult):
        result_summary = (
            f"Request {solver_result.status}\n"
            f"Reason: {solver_result.reason}\n"
        )
    elif isinstance(solver_result, SingleInstanceDetectionResult):
        state_word = "DEADLOCKED" if solver_result.is_deadlocked else "NO DEADLOCK"
        result_summary = (
            f"State: {state_word}\n"
            f"Cycles: {solver_result.cycles}\n"
            f"Deadlocked processes: {solver_result.deadlocked_processes}\n"
        )
    elif isinstance(solver_result, MultiInstanceDetectionResult):
        state_word = "DEADLOCKED" if solver_result.is_deadlocked else "NO DEADLOCK"
        result_summary = (
            f"State: {state_word}\n"
            f"Deadlocked processes: {solver_result.deadlocked_processes}\n"
        )

    return (
        f"The student asks: {query}\n\n"
        f"IMPORTANT: The following results were computed by a DETERMINISTIC SOLVER. "
        f"Do NOT recalculate or contradict these numbers. Explain each step to the student.\n\n"
        f"=== VERIFIED SOLVER RESULT ===\n{result_summary}"
        f"=== STEP-BY-STEP TRACE ===\n{trace_lines}\n"
        f"=== END OF SOLVER OUTPUT ===\n\n"
        f"Now explain these results step-by-step in a student-friendly way. "
        f"Point out common misconceptions where relevant (e.g. unsafe ≠ deadlocked)."
    )


def _build_graph_prompt(query: str, solver_result: Any) -> str:
    """Assemble a graph analysis prompt."""
    if solver_result is None:
        return (
            f"The student asks about a wait-for or resource-allocation graph: {query}\n\n"
            f"No graph data was provided. Ask the student to provide the nodes and edges "
            f"so the deterministic cycle detector can analyse it."
        )

    trace_lines = ""
    if hasattr(solver_result, "trace") and solver_result.trace:
        trace_lines = "\n".join(solver_result.trace)

    result_summary = ""
    if isinstance(solver_result, SingleInstanceDetectionResult):
        state_word = "DEADLOCKED (cycle detected)" if solver_result.is_deadlocked else "NO DEADLOCK (acyclic)"
        result_summary = (
            f"State: {state_word}\n"
            f"Cycles found: {solver_result.cycles}\n"
            f"Deadlocked processes: {solver_result.deadlocked_processes}\n"
        )

    return (
        f"The student asks: {query}\n\n"
        f"The deterministic graph analyser produced these results:\n\n"
        f"=== GRAPH ANALYSIS RESULT ===\n{result_summary}"
        f"=== DFS TRACE ===\n{trace_lines}\n"
        f"=== END ===\n\n"
        f"Explain WHY a cycle implies deadlock in single-instance resource systems. "
        f"Walk through the cycle step-by-step. Mention wait-for vs resource-allocation graph differences."
    )


def _build_lab_prompt(query: str, retrieval: RetrievalResult) -> str:
    """Assemble a lab/code-oriented prompt, grounded in RAG if available."""
    if retrieval.has_sufficient_context and retrieval.chunks:
        context_text = "\n\n---\n\n".join(
            f"[{chunk.citation}]\n{chunk.content}"
            for chunk in retrieval.chunks[:settings.top_k_retrieval]
        )
        return (
            f"The student has a lab/implementation question: {query}\n\n"
            f"Relevant course material:\n{context_text}\n\n"
            f"Provide a clear code-oriented explanation. Use C/pthreads examples where appropriate. "
            f"Stick to the course context provided."
        )
    else:
        return (
            f"The student has a lab/implementation question: {query}\n\n"
            f"No specific course lab material was found in the knowledge base. "
            f"Provide a general explanation using standard POSIX pthreads. "
            f"Explicitly note that specific lab API requirements should be verified against course materials."
        )


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class DeadlockTutorOrchestrator:
    """Central orchestration engine for the DeadlockTutorLLM pipeline.

    Wires together: QueryRouter → RAG/Solver → LLM → ResponseComposer
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        retriever: BaseRetriever,
        router: Optional[QueryRouter] = None,
        composer: Optional[ResponseComposer] = None,
    ):
        self.llm = llm_provider
        self.retriever = retriever
        self.router = router or QueryRouter()
        self.composer = composer or ResponseComposer()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle(self, request: ChatRequest) -> ChatResponse:
        """Process a student chat request end-to-end.

        Args:
            request: ChatRequest with query text and optional structured solver inputs.

        Returns:
            ChatResponse with answer, citations, solver result, and groundedness flag.
        """
        routing = self.router.route(request.query)
        category = routing.category

        retrieval: Optional[RetrievalResult] = None
        solver_result: Any = None

        try:
            if category == QueryCategory.THEORY:
                retrieval, solver_result, prompt = self._handle_theory(request)
            elif category == QueryCategory.NUMERICAL:
                retrieval, solver_result, prompt = self._handle_numerical(request)
            elif category == QueryCategory.GRAPH:
                retrieval, solver_result, prompt = self._handle_graph(request)
            else:  # LAB
                retrieval, solver_result, prompt = self._handle_lab(request)

            # LLM call
            llm_resp = self._call_llm(prompt, request, category, retrieval, solver_result)

        except Exception as exc:
            logger.exception("Orchestrator pipeline error for query '%s': %s", request.query, exc)
            llm_resp = LLMResponse(
                content=(
                    f"⚠️ An error occurred while processing your question: {exc}\n\n"
                    "Please check your input and try again."
                ),
                model_name=settings.primary_model,
                provider="error",
            )

        # Compose final response
        composed = self.composer.compose(
            routing=routing,
            llm_text=llm_resp.content,
            retrieval=retrieval,
            solver_result=solver_result,
            model_name=llm_resp.model_name,
        )

        # Build citations list
        sources = self._build_citations(retrieval)

        # Determine groundedness
        grounded, groundedness_message = self._assess_groundedness(category, retrieval, solver_result)

        return ChatResponse(
            query=request.query,
            category=category.value,
            answer=llm_resp.content,
            sources=sources,
            solver_result=solver_result.model_dump() if hasattr(solver_result, "model_dump") else None,
            graph_diagram=composed.graph_diagram,
            grounded=grounded,
            groundedness_message=groundedness_message,
            teaching_notes=composed.teaching_notes,
            worked_steps=composed.worked_steps,
            model_used=llm_resp.model_name,
            llm_provider=llm_resp.provider,
        )

    # ------------------------------------------------------------------
    # Category handlers
    # ------------------------------------------------------------------

    def _handle_theory(self, request: ChatRequest):
        retrieval = self.retriever.retrieve(request.query)
        prompt = _build_theory_prompt(request.query, retrieval)
        return retrieval, None, prompt

    def _handle_numerical(self, request: ChatRequest):
        solver_result = None

        # Priority: resource_request > banker_state > multi_instance > single (if explicitly graph)
        if request.resource_request is not None:
            try:
                solver_result = request_check(request.resource_request)
            except Exception as exc:
                logger.warning("Solver request_check failed: %s", exc)

        elif request.banker_state is not None:
            try:
                solver_result = safety_check(request.banker_state)
            except Exception as exc:
                logger.warning("Solver safety_check failed: %s", exc)

        elif request.multi_instance_detection is not None:
            try:
                solver_result = detect_multi_instance(request.multi_instance_detection)
            except Exception as exc:
                logger.warning("Solver detect_multi_instance failed: %s", exc)

        prompt = _build_numerical_prompt(request.query, solver_result)
        return None, solver_result, prompt

    def _handle_graph(self, request: ChatRequest):
        solver_result = None

        if request.single_instance_graph is not None:
            try:
                solver_result = detect_single_instance(request.single_instance_graph)
            except Exception as exc:
                logger.warning("Solver detect_single_instance failed: %s", exc)
        elif request.multi_instance_detection is not None:
            try:
                solver_result = detect_multi_instance(request.multi_instance_detection)
            except Exception as exc:
                logger.warning("Solver detect_multi_instance (graph) failed: %s", exc)

        # Also attempt RAG for explanatory context
        retrieval = self.retriever.retrieve(request.query)
        prompt = _build_graph_prompt(request.query, solver_result)
        return retrieval, solver_result, prompt

    def _handle_lab(self, request: ChatRequest):
        retrieval = self.retriever.retrieve(request.query)
        prompt = _build_lab_prompt(request.query, retrieval)
        return retrieval, None, prompt

    # ------------------------------------------------------------------
    # LLM call with graceful fallback
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, request: ChatRequest, category: QueryCategory, retrieval: Optional[RetrievalResult], solver_result: Any) -> LLMResponse:
        system_prompt = (
            "You are DeadlockTutorLLM, a specialized tutor for Operating Systems deadlocks. "
            "Always teach step-by-step. "
            f"Current Query Category: {category.value.upper()}.\n\n"
            "STRICT RULES:\n"
            "1. Do not invent numerical results.\n"
            "2. Use deterministic solver output as authoritative.\n"
            "3. For RAG questions: Answer using the retrieved course material and cite sources.\n"
            "4. For unsupported questions: Clearly state that sufficient course material was not found."
        )

        context_text = None
        if retrieval and retrieval.chunks:
            context_text = "\n\n---\n\n".join(
                f"[{chunk.citation}]\n{chunk.content}"
                for chunk in retrieval.chunks[:settings.top_k_retrieval]
            )

        llm_request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context_text,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        try:
            return self.llm.generate(llm_request)
        except Exception as exc:
            logger.error("LLM generate() failed: %s", exc)
            # Graceful degradation — return a stub response rather than crashing
            return LLMResponse(
                content=(
                    "⚠️ **LLM Unavailable**\n\n"
                    "The language model is currently unavailable. "
                    "Deterministic solver results and source citations are still shown below. "
                    f"(Error: {exc})"
                ),
                model_name=settings.primary_model,
                provider="error_fallback",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_citations(self, retrieval: Optional[RetrievalResult]) -> List[SourceCitation]:
        if retrieval is None or not retrieval.chunks:
            return []
        seen = set()
        citations = []
        for chunk in retrieval.chunks:
            key = chunk.citation
            if key not in seen:
                seen.add(key)
                citations.append(SourceCitation(
                    source=chunk.source,
                    page=chunk.page,
                    doc_type=chunk.doc_type,
                    citation_string=chunk.citation,
                ))
        return citations

    def _assess_groundedness(
        self,
        category: QueryCategory,
        retrieval: Optional[RetrievalResult],
        solver_result: Any,
    ) -> tuple[bool, str]:
        if category == QueryCategory.NUMERICAL:
            if solver_result is not None:
                return True, "Grounded in deterministic solver output — numerical results are verified."
            return False, "No structured solver input provided. Answer is explanatory only."

        if category == QueryCategory.GRAPH:
            if solver_result is not None:
                return True, "Grounded in deterministic graph cycle detector output."
            if retrieval and retrieval.has_sufficient_context:
                return True, "Grounded in course material (no graph data provided for cycle detection)."
            return False, "No graph data or sufficient course material available to ground this response."

        # THEORY / LAB — grounded if RAG found relevant context
        if retrieval and retrieval.has_sufficient_context:
            return True, retrieval.message
        msg = (
            retrieval.message
            if retrieval
            else "Knowledge base is empty. No course material available for grounding."
        )
        return False, msg
