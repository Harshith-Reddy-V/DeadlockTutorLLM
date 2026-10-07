"""FastAPI Application Routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.router.query_router import QueryRouter, QueryCategory
from backend.rag.retriever import BaseRetriever
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/solver/banker-safety", response_model=BankerSafetyResult)
def run_banker_safety(state: BankerStateInput):
    """Direct execution of Banker's safety algorithm."""
    return safety_check(state)


@router.post("/api/solver/resource-request", response_model=ResourceRequestResult)
def run_resource_request(req_input: ResourceRequestInput):
    """Direct execution of Resource-Request algorithm."""
    return request_check(req_input)


@router.post("/api/solver/detect-single", response_model=SingleInstanceDetectionResult)
def run_detect_single(graph_input: SingleInstanceDetectionInput):
    """Direct cycle detection in single-instance wait-for graph."""
    return detect_single_instance(graph_input)


@router.post("/api/solver/detect-multi", response_model=MultiInstanceDetectionResult)
def run_detect_multi(det_input: MultiInstanceDetectionInput):
    """Direct multiple-instance matrix deadlock detection."""
    return detect_multi_instance(det_input)
