"""FastAPI Application Routes."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.router.query_router import QueryRouter, QueryCategory
from backend.rag.retriever import BaseRetriever, RetrievalResult, ScoredChunk
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
from backend.llm.provider import get_llm_provider, LLMRequest
from backend.composer.response_composer import ResponseComposer, ComposedResponse

router = APIRouter()
query_router = QueryRouter()
retriever = BaseRetriever()
composer = ResponseComposer()
llm = get_llm_provider()


class HealthCheckResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    llm_provider: str
    primary_model: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="Student query or problem statement")
    banker_state: BankerStateInput = Field(default=None, description="Optional structured state for numerical questions")


# =====================================================================
# RAG Schemas
# =====================================================================

class RagSearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(default=settings.top_k_retrieval, ge=1, le=20)


class RagStatusResponse(BaseModel):
    indexed_chunks: int
    vector_db_type: str
    embedding_dimension: int
    index_path: str
    raw_documents_dir: str


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
    )


@router.post("/api/chat", response_model=ComposedResponse)
def chat_endpoint(request: ChatRequest):
    """Main orchestration endpoint for DeadlockTutorLLM."""
    try:
        routing = query_router.route(request.message)

        retrieval_res = None
        solver_res = None

        if routing.category == QueryCategory.THEORY:
            retrieval_res = retriever.retrieve(request.message)
            prompt = f"Student Question: {request.message}\nContext: {retrieval_res.chunks}"
        elif routing.category == QueryCategory.NUMERICAL:
            if request.banker_state:
                solver_res = safety_check(request.banker_state)
            prompt = f"Numerical Problem: {request.message}\nSolver Trace: {solver_res}"
        else:
            prompt = f"Problem: {request.message}"

        llm_response = llm.generate(LLMRequest(prompt=prompt))

        final_response = composer.compose(
            routing=routing,
            llm_text=llm_response.content,
            retrieval=retrieval_res,
            solver_result=solver_res,
            model_name=llm_response.model_name,
        )
        return final_response

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# =====================================================================
# RAG Endpoints
# Grounded document retrieval & syllabus citation indexing
# =====================================================================

@router.post("/api/rag/ingest")
def run_rag_ingest():
    """Triggers ingestion of documents in kb/raw/, chunks them, and builds vector index."""
    try:
        count = retriever.index_raw_documents()
        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {count} document chunks into vector database.",
            "indexed_chunks": count
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/api/rag/search", response_model=RetrievalResult)
def run_rag_search(req: RagSearchRequest):
    """Retrieves top-k relevant course chunks and source citations for a query."""
    try:
        return retriever.retrieve(query=req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/api/rag/status", response_model=RagStatusResponse)
def get_rag_status():
    """Returns status and statistics of the local RAG knowledge base."""
    return RagStatusResponse(
        indexed_chunks=retriever.count(),
        vector_db_type=settings.vector_db_type,
        embedding_dimension=retriever.embedding_model.dimension,
        index_path=str(settings.vector_db_path),
        raw_documents_dir=str(settings.kb_raw_dir),
    )


# =====================================================================
# Deterministic Solver Endpoints
# Pure symbolic computation - NO LLM, NO RAG, NO external services
# =====================================================================

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
