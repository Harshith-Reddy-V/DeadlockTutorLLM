"""Deterministic Deadlock Solver - Data Models.

Defines Pydantic schemas for:
- Banker's Safety Algorithm (state, steps, traces, results)
- Resource Request Algorithm (inputs, pretend allocation, results)
- Single-instance Wait-For Graph cycle detection
- Multi-instance matrix-based deadlock detection
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class BankerStateInput(BaseModel):
    """Input structure for Banker's Safety Algorithm."""
    processes: List[str] = Field(
        default_factory=list,
        description="List of process identifiers, e.g. ['P0', 'P1', 'P2', 'P3', 'P4']"
    )
    resources: List[str] = Field(
        default_factory=list,
        description="List of resource identifiers, e.g. ['A', 'B', 'C']"
    )
    allocation: List[List[int]] = Field(
        ...,
        description="Allocation matrix of size [num_processes x num_resources]"
    )
    max_matrix: List[List[int]] = Field(
        ...,
        description="Maximum demand matrix of size [num_processes x num_resources]"
    )
    available: List[int] = Field(
        ...,
        description="Available resource vector of size [num_resources]"
    )

    @model_validator(mode="after")
    def validate_dimensions_and_defaults(self):
        # 1. Non-empty check
        if not self.allocation or not isinstance(self.allocation, list):
            raise ValueError("Allocation matrix must be a non-empty 2D list.")
        if not self.max_matrix or not isinstance(self.max_matrix, list):
            raise ValueError("Max matrix must be a non-empty 2D list.")
        if not self.available or not isinstance(self.available, list):
            raise ValueError("Available vector must be a non-empty list.")

        num_processes = len(self.allocation)
        if num_processes == 0:
            raise ValueError("Allocation matrix must have at least one process.")

        num_resources = len(self.allocation[0])
        if num_resources == 0:
            raise ValueError("Allocation matrix must have at least one resource.")

        # 2. Check allocation row dimensions and non-negativity
        for i, row in enumerate(self.allocation):
            if len(row) != num_resources:
                raise ValueError(
                    f"Allocation matrix row {i} has {len(row)} resources; expected {num_resources}."
                )
            for j, val in enumerate(row):
                if val < 0:
                    raise ValueError(f"Allocation value at ({i}, {j}) is negative: {val}")

        # 3. Check max_matrix dimensions and non-negativity
        if len(self.max_matrix) != num_processes:
            raise ValueError(
                f"Max matrix has {len(self.max_matrix)} rows; expected {num_processes} to match Allocation."
            )
        for i, row in enumerate(self.max_matrix):
            if len(row) != num_resources:
                raise ValueError(
                    f"Max matrix row {i} has {len(row)} resources; expected {num_resources}."
                )
            for j, val in enumerate(row):
                if val < 0:
                    raise ValueError(f"Max matrix value at ({i}, {j}) is negative: {val}")

        # 4. Check available vector dimensions and non-negativity
        if len(self.available) != num_resources:
            raise ValueError(
                f"Available vector has length {len(self.available)}; expected {num_resources} resources."
            )
        for j, val in enumerate(self.available):
            if val < 0:
                raise ValueError(f"Available resource {j} is negative: {val}")

        # 5. Check Max >= Allocation condition
        for i in range(num_processes):
            for j in range(num_resources):
                if self.allocation[i][j] > self.max_matrix[i][j]:
                    raise ValueError(
                        f"Process {i} Allocation ({self.allocation[i][j]}) exceeds Max ({self.max_matrix[i][j]}) for resource {j}."
                    )

        # 6. Populate default process & resource labels if empty
        if not self.processes:
            self.processes = [f"P{i}" for i in range(num_processes)]
        elif len(self.processes) != num_processes:
            raise ValueError(
                f"Length of processes ({len(self.processes)}) does not match allocation rows ({num_processes})."
            )

        if not self.resources:
            self.resources = [f"R{j}" for j in range(num_resources)]
        elif len(self.resources) != num_resources:
            raise ValueError(
                f"Length of resources ({len(self.resources)}) does not match allocation columns ({num_resources})."
            )

        return self


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
    is_safe: bool = Field(..., description="Whether system is in a safe state")
    safe_sequence: Optional[List[str]] = Field(
        default=None,
        description="Safe execution sequence of process IDs if safe"
    )
    need_matrix: List[List[int]] = Field(
        ...,
        description="Calculated Need matrix: Need[i] = Max[i] - Allocation[i]"
    )
    initial_available: List[int] = Field(..., description="Initial available resource vector")
    work_evolution: List[List[int]] = Field(
        default_factory=list,
        description="Snapshots of Work vector after each step"
    )
    finish_status: Dict[str, bool] = Field(
        default_factory=dict,
        description="Final Finish status for each process"
    )
    unfinishable_processes: List[str] = Field(
        default_factory=list,
        description="Processes that could not finish if state is unsafe"
    )
    steps: List[SafetyStep] = Field(
        default_factory=list,
        description="Structured step-by-step trace"
    )
    trace_log: List[str] = Field(
        default_factory=list,
        description="Human/LLM readable textual trace"
    )
    message: str


class ResourceRequestInput(BaseModel):
    """Input structure for Banker's Resource-Request Algorithm."""
    state: BankerStateInput
    process_id: str = Field(..., description="Process identifier (e.g. 'P1') or index (e.g. '1')")
    request: List[int] = Field(..., description="Requested resource vector of size [num_resources]")

    @model_validator(mode="after")
    def validate_request_vector(self):
        num_resources = len(self.state.available)
        if len(self.request) != num_resources:
            raise ValueError(
                f"Request vector length ({len(self.request)}) does not match number of resources ({num_resources})."
            )
        for j, val in enumerate(self.request):
            if val < 0:
                raise ValueError(f"Request value for resource {j} cannot be negative: {val}")
        return self


class ResourceRequestResult(BaseModel):
    """Structured output for Resource-Request check."""
    status: str = Field(..., description="'GRANTED' or 'DENIED'")
    granted: bool = Field(..., description="True if request is granted, False otherwise")
    reason: str
    request_valid: bool = Field(..., description="True if Request <= Need")
    available_sufficient: bool = Field(..., description="True if Request <= Available")
    original_state: BankerStateInput
    requested_vector: List[int]
    resulting_state: Optional[BankerStateInput] = None
    safety_result: Optional[BankerSafetyResult] = None
    safe_sequence: Optional[List[str]] = None
    trace: List[str] = Field(default_factory=list)


class SingleInstanceDetectionInput(BaseModel):
    """Input for single-instance wait-for graph cycle detection."""
    nodes: List[str] = Field(..., description="Processes involved in the wait-for graph")
    edges: List[List[str]] = Field(
        ...,
        description="Directed edges [Pi, Pj] meaning process Pi is waiting for process Pj"
    )

    @model_validator(mode="after")
    def validate_graph(self):
        if not self.nodes:
            raise ValueError("Wait-for graph must contain at least one node.")
        node_set = set(self.nodes)
        if len(node_set) != len(self.nodes):
            raise ValueError("Node identifiers in wait-for graph must be unique.")
        for idx, edge in enumerate(self.edges):
            if len(edge) != 2:
                raise ValueError(f"Edge {idx} must be a pair [source, target], got: {edge}")
            u, v = edge[0], edge[1]
            if u not in node_set or v not in node_set:
                raise ValueError(f"Edge {idx} contains node not in declared nodes list: [{u}, {v}]")
        return self


class SingleInstanceDetectionResult(BaseModel):
    """Structured result for single-instance cycle detection."""
    has_cycle: bool
    is_deadlocked: bool
    cycles: List[List[str]] = Field(default_factory=list)
    deadlocked_processes: List[str] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)
    message: str
    input_nodes: List[str] = Field(default_factory=list)
    input_edges: List[List[str]] = Field(default_factory=list)


class MultiInstanceDetectionInput(BaseModel):
    """Input for multiple-instance matrix deadlock detection."""
    processes: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)
    allocation: List[List[int]] = Field(..., description="Allocation matrix [num_processes x num_resources]")
    request: List[List[int]] = Field(..., description="Current request matrix [num_processes x num_resources]")
    available: List[int] = Field(..., description="Available vector [num_resources]")

    @model_validator(mode="after")
    def validate_multi_instance(self):
        if not self.allocation or not isinstance(self.allocation, list):
            raise ValueError("Allocation matrix must be a non-empty 2D list.")
        if not self.request or not isinstance(self.request, list):
            raise ValueError("Request matrix must be a non-empty 2D list.")
        if not self.available or not isinstance(self.available, list):
            raise ValueError("Available vector must be a non-empty list.")

        num_processes = len(self.allocation)
        num_resources = len(self.allocation[0])
        if num_processes == 0 or num_resources == 0:
            raise ValueError("Allocation matrix must have at least one process and one resource.")

        # Validate allocation
        for i, row in enumerate(self.allocation):
            if len(row) != num_resources:
                raise ValueError(f"Allocation row {i} length {len(row)} does not match {num_resources}.")
            for j, val in enumerate(row):
                if val < 0:
                    raise ValueError(f"Allocation value at ({i}, {j}) is negative: {val}")

        # Validate request
        if len(self.request) != num_processes:
            raise ValueError(f"Request matrix rows ({len(self.request)}) does not match processes ({num_processes}).")
        for i, row in enumerate(self.request):
            if len(row) != num_resources:
                raise ValueError(f"Request row {i} length {len(row)} does not match {num_resources}.")
            for j, val in enumerate(row):
                if val < 0:
                    raise ValueError(f"Request value at ({i}, {j}) is negative: {val}")

        # Validate available
        if len(self.available) != num_resources:
            raise ValueError(f"Available vector length ({len(self.available)}) does not match {num_resources}.")
        for j, val in enumerate(self.available):
            if val < 0:
                raise ValueError(f"Available value at {j} is negative: {val}")

        if not self.processes:
            self.processes = [f"P{i}" for i in range(num_processes)]
        elif len(self.processes) != num_processes:
            raise ValueError(f"Processes count ({len(self.processes)}) != rows ({num_processes}).")

        if not self.resources:
            self.resources = [f"R{j}" for j in range(num_resources)]
        elif len(self.resources) != num_resources:
            raise ValueError(f"Resources count ({len(self.resources)}) != columns ({num_resources}).")

        return self


class DetectionStep(BaseModel):
    """Step in matrix-based multi-instance deadlock detection."""
    step: int
    process: str
    work_before: List[int]
    request: List[int]
    allocation_released: List[int]
    work_after: List[int]
    explanation: str


class MultiInstanceDetectionResult(BaseModel):
    """Structured result for multiple-instance deadlock detection."""
    is_deadlocked: bool
    deadlocked_processes: List[str]
    work_trace: List[List[int]]
    finish_vector: List[bool]
    steps: List[DetectionStep] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)
    message: str
