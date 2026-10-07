"""FastAPI Application Routes — Phase 4 Complete Pipeline."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.router.query_router import QueryRouter, QueryCategory
from backend.rag.retriever import BaseRetriever, RetrievalResult
from backend.rag.ingestion import IngestionReport
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
from backend.llm.provider import get_llm_provider, LLMRequest, BaseLLMProvider
from backend.composer.response_composer import ResponseComposer, ComposedResponse
from backend.orchestrator import (
    DeadlockTutorOrchestrator,
    ChatRequest,
    ChatResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level singletons — instantiated once at startup
# ---------------------------------------------------------------------------
_query_router = QueryRouter()
_retriever = BaseRetriever()
_composer = ResponseComposer()
_llm: BaseLLMProvider = get_llm_provider()

_orchestrator = DeadlockTutorOrchestrator(
    llm_provider=_llm,
    retriever=_retriever,
    router=_query_router,
    composer=_composer,
)


# ---------------------------------------------------------------------------
# RAG Schemas
# ---------------------------------------------------------------------------

class RagSearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(default=settings.top_k_retrieval, ge=1, le=20)


class RagStatusResponse(BaseModel):
    indexed_chunks: int
    vector_db_type: str
    embedding_dimension: int
    index_path: str
    raw_documents_dir: str


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class HealthCheckResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    llm_provider: str
    primary_model: str
    rag_indexed_chunks: int


@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    """Returns application health and configuration metadata."""
    return HealthCheckResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        primary_model=settings.primary_model,
        rag_indexed_chunks=_retriever.count(),
    )


# ---------------------------------------------------------------------------
# Chat Endpoint — Main orchestration pipeline
# ---------------------------------------------------------------------------

@router.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Main DeadlockTutorLLM chat endpoint.

    Accepts a student query (and optional structured solver inputs), routes it,
    runs the appropriate backend (RAG / solver), calls the LLM, and returns a
    composed teaching response with citations and solver trace.

    Request body:
        query                  : Student's question (required)
        banker_state           : BankerStateInput for numerical safety queries (optional)
        resource_request       : ResourceRequestInput for request algorithm (optional)
        single_instance_graph  : Wait-for graph for cycle detection (optional)
        multi_instance_detection: Matrix inputs for multi-instance detection (optional)

    Returns:
        ChatResponse with answer, category, sources, solver_result, grounded flag.
    """
    try:
        return _orchestrator.handle(request)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ---------------------------------------------------------------------------
# RAG Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/rag/ingest")
def run_rag_ingest():
    """Triggers ingestion of documents in kb/raw/, chunks them, and builds vector index."""
    try:
        count = _retriever.index_raw_documents()
        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {count} document chunks into vector database.",
            "indexed_chunks": count,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/api/rag/search", response_model=RetrievalResult)
def run_rag_search(req: RagSearchRequest):
    """Retrieves top-k relevant course chunks and source citations for a query."""
    try:
        return _retriever.retrieve(query=req.query, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/api/rag/status", response_model=RagStatusResponse)
def get_rag_status():
    """Returns status and statistics of the local RAG knowledge base."""
    return RagStatusResponse(
        indexed_chunks=_retriever.count(),
        vector_db_type=settings.vector_db_type,
        embedding_dimension=_retriever.embedding_model.dimension,
        index_path=str(settings.vector_db_path),
        raw_documents_dir=str(settings.kb_raw_dir),
    )


# ---------------------------------------------------------------------------
# Deterministic Solver Endpoints (unchanged from Phase 2)
# ---------------------------------------------------------------------------

@router.post("/api/solver/safety", response_model=BankerSafetyResult)
@router.post("/api/solver/banker-safety", response_model=BankerSafetyResult)
def run_banker_safety(state: BankerStateInput):
    """Direct execution of Banker's safety algorithm."""
    try:
        return safety_check(state)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))


@router.post("/api/solver/request", response_model=ResourceRequestResult)
@router.post("/api/solver/resource-request", response_model=ResourceRequestResult)
def run_resource_request(req_input: ResourceRequestInput):
    """Direct execution of Banker's Resource-Request algorithm."""
    try:
        return request_check(req_input)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))


@router.post("/api/solver/detect/single", response_model=SingleInstanceDetectionResult)
@router.post("/api/solver/detect-single", response_model=SingleInstanceDetectionResult)
def run_detect_single(graph_input: SingleInstanceDetectionInput):
    """Direct cycle detection in single-instance wait-for graph."""
    try:
        return detect_single_instance(graph_input)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))


@router.post("/api/solver/detect/multi", response_model=MultiInstanceDetectionResult)
@router.post("/api/solver/detect-multi", response_model=MultiInstanceDetectionResult)
def run_detect_multi(det_input: MultiInstanceDetectionInput):
    """Direct multiple-instance matrix deadlock detection."""
    try:
        return detect_multi_instance(det_input)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
