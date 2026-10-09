"""Deterministic Deadlock Solver Engine.

Implements pure deterministic algorithms:
1. Banker's Safety Algorithm (`safety_check`)
2. Banker's Resource-Request Algorithm (`request_check`)
3. Single-Instance Wait-For Graph Cycle Detection (`detect_single_instance`)
4. Multiple-Instance Matrix-based Deadlock Detection (`detect_multi_instance`)

No LLM, RAG, or external service calls. Guarantees 100% numerical correctness.
"""

from typing import List, Optional, Dict, Tuple, Set
from backend.solver.models import (
    BankerStateInput,
    SafetyStep,
    BankerSafetyResult,
    ResourceRequestInput,
    ResourceRequestResult,
    SingleInstanceDetectionInput,
    SingleInstanceDetectionResult,
    MultiInstanceDetectionInput,
    DetectionStep,
    MultiInstanceDetectionResult,
)


def calculate_need(max_matrix: List[List[int]], allocation: List[List[int]]) -> List[List[int]]:
    """Calculates Need matrix: Need[i][j] = Max[i][j] - Allocation[i][j].

    Validates that:
    - Matrices are non-empty
    - Dimensions match
    - Values are non-negative
    - Max >= Allocation for all entries
    """
    if not max_matrix or not isinstance(max_matrix, list):
        raise ValueError("Max matrix must be a non-empty 2D list.")
    if not allocation or not isinstance(allocation, list):
        raise ValueError("Allocation matrix must be a non-empty 2D list.")

    num_p = len(max_matrix)
    if num_p != len(allocation):
        raise ValueError(
            f"Row count mismatch: Max matrix has {num_p} rows, Allocation has {len(allocation)} rows."
        )

    num_r = len(max_matrix[0]) if num_p > 0 else 0
    if num_r == 0:
        raise ValueError("Max matrix rows must have at least one resource column.")

    need = []
    for i in range(num_p):
        if len(max_matrix[i]) != num_r:
            raise ValueError(f"Max matrix row {i} has length {len(max_matrix[i])}; expected {num_r}.")
        if len(allocation[i]) != num_r:
            raise ValueError(f"Allocation row {i} has length {len(allocation[i])}; expected {num_r}.")

        row = []
        for j in range(num_r):
            m_val = max_matrix[i][j]
            a_val = allocation[i][j]
            if m_val < 0:
                raise ValueError(f"Max matrix value at ({i}, {j}) is negative: {m_val}")
            if a_val < 0:
                raise ValueError(f"Allocation value at ({i}, {j}) is negative: {a_val}")
            diff = m_val - a_val
            if diff < 0:
                raise ValueError(
                    f"Invalid state: Allocation ({a_val}) exceeds Max ({m_val}) for process {i}, resource {j}."
                )
            row.append(diff)
        need.append(row)
    return need


def safety_check(
    state: Optional[BankerStateInput] = None,
    *,
    allocation: Optional[List[List[int]]] = None,
    max_matrix: Optional[List[List[int]]] = None,
    available: Optional[List[int]] = None,
    processes: Optional[List[str]] = None,
    resources: Optional[List[str]] = None,
) -> BankerSafetyResult:
    """Executes the standard Banker's safety algorithm and returns structured results.

    Accepts either a validated BankerStateInput object or raw matrix arguments.
    """
    if state is None:
        state = BankerStateInput(
            allocation=allocation,
            max_matrix=max_matrix,
            available=available,
            processes=processes or [],
            resources=resources or []
        )

    need = calculate_need(state.max_matrix, state.allocation)
    n = len(state.processes)
    m = len(state.resources)

    work = list(state.available)
    finish = [False] * n
    safe_sequence: List[str] = []
    steps: List[SafetyStep] = []
    trace_log: List[str] = [
        f"Initialization: Work = Available = {work}. Finish vector = {finish}."
    ]
    work_evolution: List[List[int]] = [list(work)]

    # Deterministic search for processes that can finish
    start_idx = 0
    while len(safe_sequence) < n:
        found = False
        for offset in range(n):
            i = (start_idx + offset) % n
            if not finish[i]:
                can_allocate = all(need[i][j] <= work[j] for j in range(m))
                if can_allocate:
                    work_before = list(work)
                    for j in range(m):
                        work[j] += state.allocation[i][j]
                    finish[i] = True
                    safe_sequence.append(state.processes[i])
                    work_after = list(work)
                    work_evolution.append(list(work))
                    found = True

                    explanation = (
                        f"Process {state.processes[i]}: Need {need[i]} <= Work {work_before} is TRUE. "
                        f"{state.processes[i]} completes and releases Allocation {state.allocation[i]}. "
                        f"Updated Work = {work_after}."
                    )
                    trace_log.append(explanation)

                    step_item = SafetyStep(
                        step=len(safe_sequence),
                        process=state.processes[i],
                        work_before=work_before,
                        need=need[i],
                        can_allocate=True,
                        work_after=work_after,
                        finish=list(finish),
                        explanation=explanation,
                    )
                    steps.append(step_item)

                    # Continue search starting from next process
                    start_idx = (i + 1) % n
                    break

        if not found:
            # No process could finish in this complete cycle -> UNSAFE
            break

    is_safe = (len(safe_sequence) == n)
    finish_status = {state.processes[i]: finish[i] for i in range(n)}

    if is_safe:
        seq_formatted = " -> ".join(safe_sequence)
        message = f"System is in a SAFE state. Safe sequence: <{seq_formatted}>."
        unfinishable_processes: List[str] = []
        trace_log.append(f"Conclusion: Safe sequence exists: <{seq_formatted}>.")
    else:
        unfinishable_processes = [state.processes[i] for i in range(n) if not finish[i]]
        safe_sequence_res = None
        message = (
            f"System is in an UNSAFE state. No safe sequence exists. "
            f"Unfinishable process(es): {unfinishable_processes}."
        )
        trace_log.append(
            f"Conclusion: State is UNSAFE. Processes {unfinishable_processes} cannot satisfy Need <= Work with available resources."
        )
        safe_sequence = safe_sequence_res

    return BankerSafetyResult(
        is_safe=is_safe,
        safe_sequence=safe_sequence,
        need_matrix=need,
        initial_available=list(state.available),
        work_evolution=work_evolution,
        finish_status=finish_status,
        unfinishable_processes=unfinishable_processes,
        steps=steps,
        trace_log=trace_log,
        message=message,
    )


def request_check(req_input: ResourceRequestInput) -> ResourceRequestResult:
    """Executes Banker's Resource-Request algorithm.

    1. Verify Request <= Need[p]
    2. Verify Request <= Available
    3. Pretend allocation and run safety check
    4. Grant only if resulting state is SAFE; otherwise rollback and reject
    """
    state = req_input.state
    req = req_input.request
    p_id = req_input.process_id
    m = len(state.resources)

    # Resolve process index
    if p_id in state.processes:
        p_idx = state.processes.index(p_id)
        p_name = p_id
    elif p_id.isdigit() and 0 <= int(p_id) < len(state.processes):
        p_idx = int(p_id)
        p_name = state.processes[p_idx]
    else:
        raise ValueError(
            f"Process identifier '{p_id}' not found in declared processes: {state.processes}."
        )

    need = calculate_need(state.max_matrix, state.allocation)
    trace = []

    # Step 1: Check Request <= Need
    for j in range(m):
        if req[j] > need[p_idx][j]:
            res_label = state.resources[j]
            reason = (
                f"Process {p_name} exceeded its maximum claim: "
                f"Request {req[j]} > Need {need[p_idx][j]} for resource '{res_label}'."
            )
            trace.append(f"Step 1: Check Request <= Need[{p_name}]: FAILED.")
            trace.append(f"Violation: {reason}")
            trace.append("Decision: Request is DENIED immediately.")
            return ResourceRequestResult(
                status="DENIED",
                granted=False,
                reason=reason,
                request_valid=False,
                available_sufficient=False,
                original_state=state,
                requested_vector=req,
                resulting_state=None,
                safety_result=None,
                safe_sequence=None,
                trace=trace,
            )

    trace.append(
        f"Step 1: Check Request <= Need[{p_name}]: PASSED ({req} <= {need[p_idx]})."
    )

    # Step 2: Check Request <= Available
    for j in range(m):
        if req[j] > state.available[j]:
            res_label = state.resources[j]
            reason = (
                f"Insufficient resources available: "
                f"Request {req[j]} > Available {state.available[j]} for resource '{res_label}'. "
                f"Process {p_name} must wait."
            )
            trace.append(f"Step 2: Check Request <= Available: FAILED.")
            trace.append(f"Violation: {reason}")
            trace.append(f"Decision: Request is DENIED. Process {p_name} must wait.")
            return ResourceRequestResult(
                status="DENIED",
                granted=False,
                reason=reason,
                request_valid=True,
                available_sufficient=False,
                original_state=state,
                requested_vector=req,
                resulting_state=None,
                safety_result=None,
                safe_sequence=None,
                trace=trace,
            )

    trace.append(
        f"Step 2: Check Request <= Available: PASSED ({req} <= {state.available})."
    )

    # Step 3: Pretend allocation
    new_available = [state.available[j] - req[j] for j in range(m)]
    new_allocation = [row[:] for row in state.allocation]
    new_allocation[p_idx] = [new_allocation[p_idx][j] + req[j] for j in range(m)]
    new_max = [row[:] for row in state.max_matrix]

    pretend_state = BankerStateInput(
        processes=state.processes,
        resources=state.resources,
        allocation=new_allocation,
        max_matrix=new_max,
        available=new_available,
    )

    trace.append(f"Step 3: Pretend allocation executed.")
    trace.append(f"        Updated Available = {new_available}.")
    trace.append(f"        Updated Allocation[{p_name}] = {new_allocation[p_idx]}.")

    # Step 4: Run safety algorithm on pretend state
    safety_res = safety_check(pretend_state)

    if safety_res.is_safe:
        seq_str = " -> ".join(safety_res.safe_sequence or [])
        reason = (
            f"Request is GRANTED. Resulting state is SAFE with safe sequence: <{seq_str}>."
        )
        trace.append(f"Step 4: Safety Check on resulting state: PASSED (SAFE).")
        trace.append(f"Decision: Request is GRANTED. Safe sequence: <{seq_str}>.")
        return ResourceRequestResult(
            status="GRANTED",
            granted=True,
            reason=reason,
            request_valid=True,
            available_sufficient=True,
            original_state=state,
            requested_vector=req,
            resulting_state=pretend_state,
            safety_result=safety_res,
            safe_sequence=safety_res.safe_sequence,
            trace=trace,
        )
    else:
        reason = (
            f"Request is DENIED. Granting request would transition system into an UNSAFE state. "
            f"Unfinishable process(es): {safety_res.unfinishable_processes}. Allocation rolled back."
        )
        trace.append(f"Step 4: Safety Check on resulting state: FAILED (UNSAFE).")
        trace.append(f"Step 5: Rollback pretend allocation to original state.")
        trace.append(f"Decision: Request is DENIED to prevent potential deadlock. Process {p_name} must wait.")
        return ResourceRequestResult(
            status="DENIED",
            granted=False,
            reason=reason,
            request_valid=True,
            available_sufficient=True,
            original_state=state,
            requested_vector=req,
            resulting_state=None,
            safety_result=safety_res,
            safe_sequence=None,
            trace=trace,
        )


def detect_single_instance(graph_input: SingleInstanceDetectionInput) -> SingleInstanceDetectionResult:
    """Detects cycles in single-instance wait-for graph using deterministic DFS.

    A directed edge [Pi, Pj] represents Pi is waiting for resource held by Pj.
    In single-instance systems, cycle <=> deadlock.
    """
    nodes = list(graph_input.nodes)
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for edge in graph_input.edges:
        u, v = edge[0], edge[1]
        adj[u].append(v)

    # 0 = UNVISITED, 1 = VISITING (in current recursion path), 2 = FULLY_VISITED
    color: Dict[str, int] = {n: 0 for n in nodes}
    parent_path: List[str] = []
    detected_cycles: List[List[str]] = []
    deadlocked_nodes_set: Set[str] = set()
    trace: List[str] = [
        f"Wait-for graph initialized with {len(nodes)} processes and {len(graph_input.edges)} waiting dependencies."
    ]

    def dfs(u: str):
        color[u] = 1
        parent_path.append(u)
        trace.append(f"Visiting process '{u}'. Current traversal stack: {' -> '.join(parent_path)}")

        for v in adj[u]:
            trace.append(f"Examining edge '{u}' -> '{v}'...")
            if color[v] == 1:
                # Back-edge detected! Reconstruct cycle
                cycle_start_idx = parent_path.index(v)
                cycle = parent_path[cycle_start_idx:] + [v]
                cycle_str = " -> ".join(cycle)
                trace.append(f"*** Back-edge detected! Cycle found: {cycle_str} ***")
                detected_cycles.append(cycle)
                for node_in_cycle in parent_path[cycle_start_idx:]:
                    deadlocked_nodes_set.add(node_in_cycle)
            elif color[v] == 0:
                dfs(v)
            else:
                trace.append(f"Process '{v}' already fully explored (no back-edge from this branch).")

        parent_path.pop()
        color[u] = 2
        trace.append(f"Finished exploration for process '{u}'.")

    # Explore all nodes deterministically
    for node in nodes:
        if color[node] == 0:
            trace.append(f"Initiating DFS branch from unvisited node '{node}'.")
            dfs(node)

    has_cycle = len(detected_cycles) > 0
    deadlocked_processes = sorted(list(deadlocked_nodes_set))

    if has_cycle:
        cycles_summary = "; ".join([" -> ".join(c) for c in detected_cycles])
        msg = (
            f"DEADLOCK DETECTED! Wait-for graph contains cycle(s): [{cycles_summary}]. "
            f"Deadlocked processes: {deadlocked_processes}."
        )
    else:
        msg = "No deadlock detected. The wait-for graph is acyclic; all processes can complete."

    trace.append(f"Result: {msg}")

    return SingleInstanceDetectionResult(
        has_cycle=has_cycle,
        is_deadlocked=has_cycle,
        cycles=detected_cycles,
        deadlocked_processes=deadlocked_processes,
        trace=trace,
        message=msg,
        input_nodes=list(graph_input.nodes),
        input_edges=[list(edge) for edge in graph_input.edges],
    )


def detect_multi_instance(det_input: MultiInstanceDetectionInput) -> MultiInstanceDetectionResult:
    """Detects deadlocks in multiple-instance systems using matrix reduction algorithm.

    Algorithm:
    1. Work = Available
    2. Finish[i] = True if Allocation[i] == 0 else False
    3. Find process i such that Finish[i] == False and Request[i] <= Work
       If found: Work = Work + Allocation[i], Finish[i] = True, repeat
    4. If any Finish[i] == False, system is deadlocked and those processes are deadlocked.
    """
    n = len(det_input.processes)
    m = len(det_input.resources)

    work = list(det_input.available)
    finish = [False] * n

    # Step 1: Initialize Finish
    for i in range(n):
        if all(det_input.allocation[i][j] == 0 for j in range(m)):
            finish[i] = True
        else:
            finish[i] = False

    work_trace: List[List[int]] = [list(work)]
    steps: List[DetectionStep] = []
    trace: List[str] = [
        f"Initialization: Work = Available = {work}.",
        f"Initial Finish vector: {[finish[i] for i in range(n)]} "
        f"(processes holding 0 resources initialized to True)."
    ]

    # Step 2 & 3: Iteratively satisfy requests
    step_num = 1
    while True:
        allocated = False
        for i in range(n):
            if not finish[i]:
                can_satisfy = all(det_input.request[i][j] <= work[j] for j in range(m))
                if can_satisfy:
                    work_before = list(work)
                    for j in range(m):
                        work[j] += det_input.allocation[i][j]
                    finish[i] = True
                    allocated = True
                    work_after = list(work)
                    work_trace.append(list(work))

                    explanation = (
                        f"Process {det_input.processes[i]}: Request {det_input.request[i]} <= Work {work_before} is TRUE. "
                        f"{det_input.processes[i]} completes and releases Allocation {det_input.allocation[i]}. "
                        f"Updated Work = {work_after}."
                    )
                    trace.append(explanation)
                    steps.append(
                        DetectionStep(
                            step=step_num,
                            process=det_input.processes[i],
                            work_before=work_before,
                            request=det_input.request[i],
                            allocation_released=det_input.allocation[i],
                            work_after=work_after,
                            explanation=explanation,
                        )
                    )
                    step_num += 1
                    break

        if not allocated:
            break

    # Step 4: Evaluate remaining Finish status
    deadlocked_indices = [i for i in range(n) if not finish[i]]
    deadlocked_processes = [det_input.processes[i] for i in deadlocked_indices]
    is_deadlocked = len(deadlocked_processes) > 0

    if is_deadlocked:
        msg = (
            f"DEADLOCK DETECTED! The following {len(deadlocked_processes)} process(es) are deadlocked: "
            f"{deadlocked_processes}. Available resources cannot satisfy their outstanding requests."
        )
    else:
        msg = "No deadlock detected. All processes can successfully acquire requested resources and terminate."

    trace.append(f"Result: {msg}")

    return MultiInstanceDetectionResult(
        is_deadlocked=is_deadlocked,
        deadlocked_processes=deadlocked_processes,
        work_trace=work_trace,
        finish_vector=finish,
        steps=steps,
        trace=trace,
        message=msg,
    )
