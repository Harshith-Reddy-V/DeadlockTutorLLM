from typing import List, Optional
from pydantic import BaseModel, Field


class BankerStateInput(BaseModel):
    """Input structure for Banker's Safety Algorithm."""
    processes: List[str] = Field(..., description="List of process identifiers, e.g. ['P0', 'P1', 'P2', 'P3', 'P4']")
    resources: List[str] = Field(..., description="List of resource identifiers, e.g. ['A', 'B', 'C']")
    allocation: List[List[int]] = Field(..., description="Allocation matrix of size [num_processes x num_resources]")
    max_matrix: List[List[int]] = Field(..., description="Maximum demand matrix of size [num_processes x num_resources]")
    available: List[int] = Field(..., description="Available resource vector of size [num_resources]")


class SafetyStep(BaseModel):
    """Step-by-step trace item for Banker's Safety Algorithm."""
    step: int
    process: str
    work_before: List[int]
    need: List[int]
    can_allocate: bool
    work_after: List[int]
    finish: List[bool]
    explanation: str


class BankerSafetyResult(BaseModel):
    """Structured output for Banker's safety check."""
    is_safe: bool
    safe_sequence: Optional[List[str]] = None
    need_matrix: List[List[int]]
    initial_available: List[int]
    steps: List[SafetyStep]
    message: str


class ResourceRequestInput(BaseModel):
    """Input structure for Banker's Resource-Request Algorithm."""
    state: BankerStateInput
    process_id: str
    request: List[int]


class ResourceRequestResult(BaseModel):
    """Structured output for Resource-Request check."""
    status: str  # "GRANTED" or "DENIED"
    reason: str
    request_valid: bool
    available_sufficient: bool
    safety_result: Optional[BankerSafetyResult] = None
    safe_sequence: Optional[List[str]] = None
    intermediate_state: Optional[BankerStateInput] = None


class SingleInstanceDetectionInput(BaseModel):
    """Input for single-instance wait-for graph cycle detection."""
    nodes: List[str] = Field(..., description="Processes involved in the wait-for graph")
    edges: List[List[str]] = Field(..., description="Directed edges [Pi, Pj] meaning Pi is waiting for Pj")


class SingleInstanceDetectionResult(BaseModel):
    """Structured result for single-instance cycle detection."""
    has_cycle: bool
    cycles: List[List[str]] = Field(default_factory=list)
    deadlocked_processes: List[str] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)
    message: str


class MultiInstanceDetectionInput(BaseModel):
    """Input for multiple-instance matrix deadlock detection."""
    processes: List[str]
    resources: List[str]
    allocation: List[List[int]]
    request: List[List[int]]
    available: List[int]


class MultiInstanceDetectionResult(BaseModel):
    """Structured result for multiple-instance deadlock detection."""
    is_deadlocked: bool
    deadlocked_processes: List[str]
    work_trace: List[List[int]]
    finish_vector: List[bool]
    message: str
