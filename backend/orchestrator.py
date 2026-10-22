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
import re
from typing import Any, Dict, List, Optional, Tuple

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
# Grounding & Domain Validation Helpers
# ---------------------------------------------------------------------------

COFFMAN_CONDITIONS_DEFINING_TERMS: Dict[str, List[str]] = {
    "mutual exclusion": [
        "non-shareable", "nonshareable", "shareable", "one process at a time",
        "exclusive mode", "held exclusively", "held by only one", "cannot be shared",
        "single process at a time"
    ],
    "hold and wait": [
        "holds at least one", "holding at least one", "holding a resource and waiting",
        "waiting to acquire additional", "requesting additional", "requesting more resources",
        "already allocated", "holds resources while requesting", "holding resources while waiting"
    ],
    "no preemption": [
        "forcibly", "released voluntarily", "taken away", "cannot be forcibly",
        "cannot be preempted", "can only be released voluntarily", "after the process has completed",
        "only be released voluntarily", "involuntarily"
    ],
    "circular wait": [
        "circular chain", "circular list", "closed chain", "closed loop",
        "chain of processes", "circular dependency", "cycle of waiting",
        "needed by the next", "held by the next", "p0 is waiting", "p0 →"
    ],
}


def _check_condition_evidence(query: str, chunks: List[Any]) -> Tuple[Optional[str], bool]:
    """Check if the query targets a specific Coffman condition and whether
    the retrieved chunks contain substantive defining evidence for it,
    as opposed to merely listing its name.

    Returns:
        (condition_name, has_substantive_definition)
    """
    q_lower = query.lower()
    combined_content = " ".join(getattr(c, "content", "").lower() for c in chunks)

    for cond_name, defining_terms in COFFMAN_CONDITIONS_DEFINING_TERMS.items():
        if cond_name in q_lower:
            mentions_cond = cond_name in combined_content
            has_definition = any(term in combined_content for term in defining_terms)
            return cond_name, (mentions_cond and has_definition)

    return None, True


COFFMAN_CONDITIONS_LIST: List[str] = [
    "mutual exclusion",
    "hold and wait",
    "no preemption",
    "circular wait",
]

PREVENTION_VERBS_REGEX = r"\b(prevents?|prevented|preventing|prevention|avoids?|avoided|avoiding|breaks?|broken|stops?|stopped)\b"
ELIMINATION_PREFIXES_REGEX = r"\b(eliminat\w*|negat\w*|break\w*|broken|remov\w*|without|disrupt\w*|destroy\w*)\s+(?:of\s+)?(?:the\s+)?"
NEGATION_WORDS_REGEX = r"\b(not|never|cannot|cant|can't|doesnt|doesn't|does\s+not|did\s+not|is\s+not|are\s+not|false|incorrect|wrong|untrue|fails?\s+to)\b"

LEAKED_SCAFFOLDING_PATTERNS = [
    r"\bcore\s+os\s+ground\s+truth\b",
    r"\brag\s+questions?\b",
    r"\binstructions?\s+for\s+your\s+answer\b",
    r"\bimportant\s+evidence\s+note\b",
    r"\bprompt\s+instructions?\b",
]


def _detect_coffman_inversion(text: str) -> Tuple[bool, Optional[str]]:
    """Detect common factual inversions where a Coffman condition is claimed
    to prevent or avoid deadlock, or ensure lack of circular wait.

    Distinguishes valid prevention statements (eliminating or breaking a condition)
    and negated statements from genuine inversions (claiming the condition itself prevents deadlock).

    Returns:
        (has_inversion, explanation_message)
    """
    if not text:
        return False, None

    clauses = re.split(r"[.\n;!?]+", text)
    for clause in clauses:
        clause_str = clause.strip()
        if not clause_str:
            continue

        c_lower = clause_str.lower()
        if "deadlock" not in c_lower and "circular wait" not in c_lower:
            continue

        for cond in COFFMAN_CONDITIONS_LIST:
            if cond not in c_lower:
                continue

            has_prev_word = bool(re.search(PREVENTION_VERBS_REGEX, c_lower))
            has_no_circ = bool(re.search(r"\b(ensur|guarante)\w*\s+no\s+(such\s+)?circular\s+wait\b", c_lower))
            if not has_prev_word and not has_no_circ:
                continue

            # 1. Valid prevention: condition is being ELIMINATED/NEGATED/BROKEN to prevent deadlock
            elim_pattern = rf"{ELIMINATION_PREFIXES_REGEX}{cond}\b|\b{cond}\s+(?:is\s+)?(eliminated|negated|broken|removed|prevented|avoided)"
            if re.search(elim_pattern, c_lower):
                continue

            # 2. Correct negation: statement is negated before or after the condition
            if re.search(NEGATION_WORDS_REGEX, c_lower):
                continue

            # 3. Affirmative claims that the condition itself causes or ensures prevention
            claims_inversion = False
            if has_no_circ:
                claims_inversion = True
            elif re.search(rf"\b{cond}\b[^,;]*{PREVENTION_VERBS_REGEX}[^,;]*\bdeadlock\b", c_lower):
                claims_inversion = True
            elif re.search(rf"\b(enforc|ensur|requir|maintain)\w*\s+{cond}\b[^,;]*{PREVENTION_VERBS_REGEX}", c_lower):
                claims_inversion = True
            elif re.search(rf"\bdeadlock\b[^,;]*\b(is|are|can\s+be)\s+(prevented|avoided|stopped)\s+by\s+{cond}\b", c_lower):
                claims_inversion = True
            elif re.search(rf"\b{cond}\b[^,;]*(essential|needed|used|helpful|vital)\s+for\s+{PREVENTION_VERBS_REGEX}[^,;]*\bdeadlock\b", c_lower):
                claims_inversion = True

            if claims_inversion:
                cond_title = cond[0].upper() + cond[1:]
                msg = (
                    f"{cond_title} is a necessary condition FOR deadlock to occur, not a prevention technique. "
                    f"Eliminating or negating {cond} prevents deadlock; enforcing or satisfying it permits deadlock."
                )
                return True, msg

    return False, None


def _sanitize_llm_response(text: str) -> str:
    """Remove internal prompt scaffolding, meta-headings, and leaked prompt labels
    while preserving legitimate educational steps (e.g. 'Step 1: Calculate Need Matrix').
    """
    if not text:
        return text

    cleaned_lines = []
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()

        # Check if line is a header or step label containing leaked scaffolding phrases
        is_heading_or_label = bool(re.match(r"^(?:#{1,6}\s+|(?:\*{1,2}|_{1,2})|step\s+\d+[:.\s-])", stripped, re.IGNORECASE))
        has_leaked_phrase = any(re.search(pat, stripped, re.IGNORECASE) for pat in LEAKED_SCAFFOLDING_PATTERNS)

        if has_leaked_phrase and is_heading_or_label:
            # Drop the leaked scaffolding header completely
            continue

        # If a non-heading line starts directly with a leaked phrase, drop it
        if any(re.match(rf"^(?:[#*_\-\s]*){pat}", stripped, re.IGNORECASE) for pat in LEAKED_SCAFFOLDING_PATTERNS):
            continue

        # Clean up inline meta-commentary like "In the context of RAG questions, "
        cleaned_line = re.sub(r"(?i)\bIn the context of RAG questions,?\s*", "", line)
        cleaned_line = re.sub(r"(?i)\bAccording to (?:the\s+)?prompt instructions,?\s*", "", cleaned_line)
        cleaned_lines.append(cleaned_line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------

def _build_theory_prompt(query: str, retrieval: RetrievalResult) -> str:
    """Assemble a grounded theory prompt using RAG context and domain rules."""
    if retrieval.has_sufficient_context and retrieval.chunks:
        context_text = "\n\n---\n\n".join(
            f"[{chunk.citation}]\n{chunk.content}"
            for chunk in retrieval.chunks[:settings.top_k_retrieval]
        )

        cond_name, has_def = _check_condition_evidence(query, retrieval.chunks)

        if cond_name and not has_def:
            evidence_guidance = (
                f"Course context note: The syllabus excerpt mentions '{cond_name}' as one of the four necessary conditions "
                f"for deadlock, but does not provide its detailed definition.\n\n"
                f"Pedagogical guidance:\n"
                f"- Note gently that while the course notes list '{cond_name}' as a necessary condition, "
                f"the detailed explanation is drawn from standard Operating Systems theory.\n"
                f"- Provide the standard OS definition:\n"
                f"  * Mutual Exclusion: At least one resource is held in a non-shareable mode (only one process at a time).\n"
                f"  * Hold and Wait: A process holding resources requests additional resources held by other processes.\n"
                f"  * No Preemption: Resources cannot be taken by force; they are released only voluntarily.\n"
                f"  * Circular Wait: A closed chain of processes exists where each process waits for a resource held by the next.\n"
                f"- Operating Systems principle: These conditions permit deadlock to occur. Deadlock prevention requires "
                f"negating or eliminating at least one condition (e.g. sharing resources to negate mutual exclusion), not enforcing it.\n"
                f"- Explain the concept naturally and clearly for an OS student without repeating internal guidelines or prompt labels."
            )
        else:
            evidence_guidance = (
                f"Please base your explanation on the course material excerpts above and cite sources accurately.\n"
                f"Remember that the four Coffman conditions allow deadlock to occur; deadlock prevention requires eliminating at least one condition."
            )

        return (
            f"The student asks: {query}\n\n"
            f"=== COURSE MATERIAL EXCERPTS ===\n{context_text}\n"
            f"=== END OF EXCERPTS ===\n\n"
            f"{evidence_guidance}\n\n"
            f"Now explain the concept clearly in a student-friendly manner."
        )
    else:
        return (
            f"The student asks: {query}\n\n"
            f"Note: The knowledge base does not contain sufficient course-specific material to ground this answer. "
            f"Provide a clear, student-friendly explanation based on standard Operating Systems theory.\n\n"
            f"Core Operating Systems principle:\n"
            f"- Coffman conditions are prerequisites that allow deadlock to occur, not prevention techniques.\n"
            f"- Deadlock prevention requires negating or eliminating at least one condition.\n\n"
            f"Explain the topic at a clear university level."
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

        # Build citations list
        sources = self._build_citations(retrieval)

        # Determine groundedness & factual validation
        grounded, groundedness_message, factual_notes, has_inversion, inv_msg = self._assess_groundedness(
            category=category,
            retrieval=retrieval,
            solver_result=solver_result,
            query=request.query,
            llm_text=llm_resp.content,
        )

        final_answer = llm_resp.content
        if has_inversion and inv_msg:
            # Prepend prominent correction banner directly to the student answer
            correction_banner = (
                f"> ⚠️ **Factual Correction & Conceptual Alert**:\n"
                f"> {inv_msg}\n"
                f">\n"
                f"> *The explanation below contains a common misconception regarding deadlock conditions versus deadlock prevention. Please note the verified Operating Systems distinction above.*"
            )
            final_answer = f"{correction_banner}\n\n---\n\n{final_answer}"

        # Compose final response with verified grounding and notes
        composed = self.composer.compose(
            routing=routing,
            llm_text=final_answer,
            retrieval=retrieval,
            solver_result=solver_result,
            model_name=llm_resp.model_name,
            grounded=grounded,
            groundedness_message=groundedness_message,
            extra_teaching_notes=factual_notes,
        )

        return ChatResponse(
            query=request.query,
            category=category.value,
            answer=final_answer,
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
            "You are DeadlockTutorLLM, a specialized university tutor for Operating Systems deadlocks. "
            "Explain concepts clearly in student-friendly language.\n\n"
            "Key tutoring guidelines:\n"
            "1. Deterministic accuracy: Use verified solver outputs for numerical and graph results; do not invent numbers.\n"
            "2. Grounded explanations: When course excerpts are provided, cite sources accurately and do not invent claims unsupported by the excerpts.\n"
            "3. Conditions versus Prevention:\n"
            "   - Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait are necessary conditions that allow deadlock to occur.\n"
            "   - Deadlock prevention is achieved by eliminating or negating at least one condition (e.g. making resources shareable), never by enforcing them.\n"
            "4. Natural communication: Speak directly to the student as an expert tutor. Never include internal prompt labels, system instructions, or meta-commentary."
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
            raw_resp = self.llm.generate(llm_request)
            sanitized_content = _sanitize_llm_response(raw_resp.content)
            return LLMResponse(
                content=sanitized_content,
                model_name=raw_resp.model_name,
                provider=raw_resp.provider,
                metadata=raw_resp.metadata,
            )
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
        query: str = "",
        llm_text: str = "",
    ) -> tuple[bool, str, List[str], bool, Optional[str]]:
        factual_notes: List[str] = []

        if category == QueryCategory.NUMERICAL:
            if solver_result is not None:
                return True, "Grounded in deterministic solver output — numerical results are verified.", [], False, None
            return False, "No structured solver input provided. Answer is explanatory only.", [], False, None

        if category == QueryCategory.GRAPH:
            if solver_result is not None:
                return True, "Grounded in deterministic graph cycle detector output.", [], False, None
            if retrieval and retrieval.has_sufficient_context:
                return True, "Grounded in course material (no graph data provided for cycle detection).", [], False, None
            return False, "No graph data or sufficient course material available to ground this response.", [], False, None

        # THEORY / LAB
        # First: Check for factual inversion in llm_text regardless of retrieval status
        has_inversion, inv_msg = _detect_coffman_inversion(llm_text)
        if has_inversion and inv_msg:
            factual_notes.append(f"⚠️ Factual correction: {inv_msg}")

        if retrieval and retrieval.has_sufficient_context:
            chunks = retrieval.chunks or []
            combined_chunks = " ".join(getattr(c, "content", "").lower() for c in chunks)
            q_lower = query.lower()

            # Check 1: Does query target a condition whose substantive definition is missing in the chunks?
            cond_name, has_def = _check_condition_evidence(query, chunks)
            if cond_name is not None and not has_def:
                groundedness_msg = (
                    f"Course notes mention '{cond_name}' as a necessary condition for deadlock, "
                    f"but do not contain its definition. Explanation is drawn from standard OS theory."
                )
                return False, groundedness_msg, factual_notes, has_inversion, inv_msg

            # Check 2: Domain topic check — If query explicitly asks about concepts not in the syllabus excerpts
            # (e.g. prevention, recovery, livelock, starvation), mark ungrounded in syllabus
            domain_topics = [
                ("prevention", ["prevention", "preventing", "prevent"]),
                ("recovery", ["recovery", "recover"]),
                ("livelock", ["livelock"]),
                ("starvation", ["starvation", "starve"]),
            ]
            for topic_name, keywords in domain_topics:
                if any(k in q_lower for k in keywords) and not any(k in combined_chunks for k in keywords):
                    groundedness_msg = (
                        f"Course excerpts mention related deadlock concepts, but do not contain material on '{topic_name}'. "
                        f"Explanation is drawn from standard OS theory."
                    )
                    return False, groundedness_msg, factual_notes, has_inversion, inv_msg

            if has_inversion:
                return False, f"Ungrounded claim: {inv_msg}", factual_notes, has_inversion, inv_msg

            return True, retrieval.message, factual_notes, has_inversion, inv_msg

        msg = (
            retrieval.message
            if retrieval
            else "Knowledge base is empty. No course material available for grounding."
        )
        return False, msg, factual_notes, has_inversion, inv_msg
