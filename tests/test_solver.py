"""Comprehensive Test Suite for Deterministic Deadlock Solver.

Validates:
A. SAFE Banker's Algorithm example (<P1, P3, P4, P0, P2>)
B. UNSAFE Banker's Algorithm example (unfinishable processes identified)
C. Valid resource request (granted when safe)
D. Request greater than Available (denied)
E. Request greater than Need (denied)
F. Request causing unsafe state (denied)
G. Single-instance deadlock (cycle detected)
H. Single-instance no deadlock (acyclic)
I. Multi-instance deadlock (deadlocked processes detected)
J. Multi-instance no deadlock (all complete)
K. Input validation errors (negative values, dimension mismatch, etc.)
"""

import pytest
from backend.solver.models import (
    BankerStateInput,
    ResourceRequestInput,
    SingleInstanceDetectionInput,
    MultiInstanceDetectionInput,
)
from backend.solver.engine import (
    calculate_need,
    safety_check,
    request_check,
    detect_single_instance,
    detect_multi_instance,
)


@pytest.fixture
def silberschatz_banker_state():
    """Classic 5-process, 3-resource example from Operating System Concepts (Silberschatz)."""
    return BankerStateInput(
        processes=["P0", "P1", "P2", "P3", "P4"],
        resources=["A", "B", "C"],
        allocation=[
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 2],
            [2, 1, 1],
            [0, 0, 2],
        ],
        max_matrix=[
            [7, 5, 3],
            [3, 2, 2],
            [9, 0, 2],
            [2, 2, 2],
            [4, 3, 3],
        ],
        available=[3, 3, 2],
    )


# =====================================================================
# Test A: SAFE Banker's Algorithm Example
# =====================================================================
def test_a_banker_safety_safe_state(silberschatz_banker_state):
    result = safety_check(silberschatz_banker_state)

    # 1. State must be safe
    assert result.is_safe is True

    # 2. Need matrix calculation
    expected_need = [
        [7, 4, 3],
        [1, 2, 2],
        [6, 0, 0],
        [0, 1, 1],
        [4, 3, 1],
    ]
    assert result.need_matrix == expected_need

    # 3. Exact safe sequence as required: <P1, P3, P4, P0, P2>
    assert result.safe_sequence == ["P1", "P3", "P4", "P0", "P2"]

    # 4. Work vector evolution check
    assert len(result.work_evolution) == 6  # Initial + 5 steps
    assert result.work_evolution[0] == [3, 3, 2]  # Initial Available
    assert result.work_evolution[-1] == [10, 5, 7]  # Total resources

    # 5. Finish vector check
    assert all(result.finish_status.values())
    assert result.unfinishable_processes == []

    # 6. Detailed step traces present
    assert len(result.steps) == 5
    assert "SAFE" in result.message


# =====================================================================
# Test B: UNSAFE Banker's Algorithm Example
# =====================================================================
def test_b_banker_safety_unsafe_state():
    # Modified state where Available is insufficient for any process to finish
    unsafe_state = BankerStateInput(
        processes=["P0", "P1", "P2"],
        resources=["A", "B"],
        allocation=[
            [2, 1],
            [1, 2],
            [3, 0],
        ],
        max_matrix=[
            [4, 3],  # Need: [2, 2]
            [3, 4],  # Need: [2, 2]
            [5, 1],  # Need: [2, 1]
        ],
        available=[1, 0],  # Cannot satisfy any need
    )

    result = safety_check(unsafe_state)

    assert result.is_safe is False
    assert result.safe_sequence is None
    assert set(result.unfinishable_processes) == {"P0", "P1", "P2"}
    assert not all(result.finish_status.values())
    assert "UNSAFE" in result.message


# =====================================================================
# Test C: Valid Resource Request
# =====================================================================
def test_c_valid_resource_request(silberschatz_banker_state):
    # P1 requests [1, 0, 2]
    # Need[P1] is [1, 2, 2], Available is [3, 3, 2]
    req_input = ResourceRequestInput(
        state=silberschatz_banker_state,
        process_id="P1",
        request=[1, 0, 2],
    )

    result = request_check(req_input)

    assert result.granted is True
    assert result.status == "GRANTED"
    assert result.request_valid is True
    assert result.available_sufficient is True
    assert result.resulting_state is not None
    assert result.resulting_state.available == [2, 3, 0]
    assert result.resulting_state.allocation[1] == [3, 0, 2]
    assert result.safety_result.is_safe is True
    assert result.safe_sequence == ["P1", "P3", "P4", "P0", "P2"]


# =====================================================================
# Test D: Request greater than Available
# =====================================================================
def test_d_request_greater_than_available(silberschatz_banker_state):
    # P4 requests [4, 3, 1], but Available is [3, 3, 2] (4 > 3 for resource A)
    req_input = ResourceRequestInput(
        state=silberschatz_banker_state,
        process_id="P4",
        request=[4, 3, 1],
    )

    result = request_check(req_input)

    assert result.granted is False
    assert result.status == "DENIED"
    assert result.request_valid is True
    assert result.available_sufficient is False
    assert "Insufficient resources" in result.reason
    assert result.resulting_state is None


# =====================================================================
# Test E: Request greater than Need
# =====================================================================
def test_e_request_greater_than_need(silberschatz_banker_state):
    # Need[P1] is [1, 2, 2]. Request [2, 0, 0] exceeds Need for A (2 > 1)
    req_input = ResourceRequestInput(
        state=silberschatz_banker_state,
        process_id="P1",
        request=[2, 0, 0],
    )

    result = request_check(req_input)

    assert result.granted is False
    assert result.status == "DENIED"
    assert result.request_valid is False
    assert "exceeded its maximum claim" in result.reason
    assert result.resulting_state is None


# =====================================================================
# Test F: Request that causes Unsafe State
# =====================================================================
def test_f_request_causing_unsafe_state(silberschatz_banker_state):
    # First, grant P1's request [1, 0, 2] resulting in Available = [2, 3, 0]
    p1_req = ResourceRequestInput(
        state=silberschatz_banker_state,
        process_id="P1",
        request=[1, 0, 2],
    )
    p1_res = request_check(p1_req)
    state_after_p1 = p1_res.resulting_state

    # Now P0 requests [0, 2, 0]:
    # Need[P0] is [7, 4, 3] (2 <= 4 -> True)
    # Available is [2, 3, 0] (2 <= 3 -> True)
    # But granting would reduce Available to [2, 1, 0], leaving no process able to finish!
    p0_req = ResourceRequestInput(
        state=state_after_p1,
        process_id="P0",
        request=[0, 2, 0],
    )
    p0_res = request_check(p0_req)

    assert p0_res.granted is False
    assert p0_res.status == "DENIED"
    assert p0_res.request_valid is True
    assert p0_res.available_sufficient is True
    assert "UNSAFE" in p0_res.reason
    assert p0_res.safety_result is not None
    assert p0_res.safety_result.is_safe is False
    assert p0_res.resulting_state is None  # Must rollback allocation


# =====================================================================
# Test G: Single-Instance Deadlock (Wait-For Graph with Cycle)
# =====================================================================
def test_g_single_instance_deadlock_cycle():
    # Cycle: P0 -> P1 -> P2 -> P0
    graph_input = SingleInstanceDetectionInput(
        nodes=["P0", "P1", "P2", "P3"],
        edges=[
            ["P0", "P1"],
            ["P1", "P2"],
            ["P2", "P0"],
            ["P3", "P0"],
        ],
    )

    result = detect_single_instance(graph_input)

    assert result.has_cycle is True
    assert result.is_deadlocked is True
    assert len(result.cycles) >= 1
    assert set(result.deadlocked_processes) == {"P0", "P1", "P2"}
    assert "DEADLOCK DETECTED" in result.message


# =====================================================================
# Test H: Single-Instance No Deadlock (Acyclic Wait-For Graph)
# =====================================================================
def test_h_single_instance_no_deadlock_acyclic():
    # DAG: P0 -> P1 -> P2, P3 -> P2
    graph_input = SingleInstanceDetectionInput(
        nodes=["P0", "P1", "P2", "P3"],
        edges=[
            ["P0", "P1"],
            ["P1", "P2"],
            ["P3", "P2"],
        ],
    )

    result = detect_single_instance(graph_input)

    assert result.has_cycle is False
    assert result.is_deadlocked is False
    assert result.cycles == []
    assert result.deadlocked_processes == []
    assert "acyclic" in result.message.lower()


# =====================================================================
# Test I: Multi-Instance Deadlock
# =====================================================================
def test_i_multi_instance_deadlock():
    # Silberschatz 7.6.2 Deadlock Case: P2 requests [0, 0, 1]
    det_input = MultiInstanceDetectionInput(
        processes=["P0", "P1", "P2", "P3", "P4"],
        resources=["A", "B", "C"],
        allocation=[
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 3],
            [2, 1, 1],
            [0, 0, 2],
        ],
        request=[
            [0, 0, 0],
            [2, 0, 2],
            [0, 0, 1],  # P2 needs C, which is not available
            [1, 0, 0],
            [0, 0, 2],
        ],
        available=[0, 0, 0],
    )

    result = detect_multi_instance(det_input)

    assert result.is_deadlocked is True
    # P0 finishes (releases [0, 1, 0]), but no other process can satisfy its request
    assert set(result.deadlocked_processes) == {"P1", "P2", "P3", "P4"}
    assert result.finish_vector[0] is True
    assert all(result.finish_vector[i] is False for i in [1, 2, 3, 4])
    assert "DEADLOCK DETECTED" in result.message


# =====================================================================
# Test J: Multi-Instance No Deadlock
# =====================================================================
def test_j_multi_instance_no_deadlock():
    # Silberschatz 7.6.2 No Deadlock Case: P2 requests [0, 0, 0]
    det_input = MultiInstanceDetectionInput(
        processes=["P0", "P1", "P2", "P3", "P4"],
        resources=["A", "B", "C"],
        allocation=[
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 3],
            [2, 1, 1],
            [0, 0, 2],
        ],
        request=[
            [0, 0, 0],
            [2, 0, 2],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 2],
        ],
        available=[0, 0, 0],
    )

    result = detect_multi_instance(det_input)

    assert result.is_deadlocked is False
    assert result.deadlocked_processes == []
    assert all(result.finish_vector)
    assert len(result.steps) == 5
    assert "No deadlock detected" in result.message


# =====================================================================
# Test K: Input Validation Errors
# =====================================================================
def test_k_negative_available_value():
    with pytest.raises(ValueError, match="negative"):
        BankerStateInput(
            allocation=[[1, 0]],
            max_matrix=[[2, 1]],
            available=[-1, 0],
        )


def test_k_negative_allocation_value():
    with pytest.raises(ValueError, match="negative"):
        BankerStateInput(
            allocation=[[-2, 1]],
            max_matrix=[[3, 2]],
            available=[1, 1],
        )


def test_k_allocation_exceeds_max():
    with pytest.raises(ValueError, match="exceeds Max"):
        BankerStateInput(
            allocation=[[3, 1]],
            max_matrix=[[2, 1]],
            available=[1, 1],
        )


def test_k_dimension_mismatch_allocation_and_max():
    with pytest.raises(ValueError, match="rows"):
        BankerStateInput(
            allocation=[[1, 0], [0, 1]],
            max_matrix=[[2, 1]],
            available=[1, 1],
        )


def test_k_dimension_mismatch_available_length():
    with pytest.raises(ValueError, match="resources"):
        BankerStateInput(
            allocation=[[1, 0]],
            max_matrix=[[2, 1]],
            available=[1, 1, 1],  # 3 elements instead of 2
        )


def test_k_invalid_process_id_in_request(silberschatz_banker_state):
    req_input = ResourceRequestInput(
        state=silberschatz_banker_state,
        process_id="P99",  # Not in declared processes
        request=[1, 0, 0],
    )
    with pytest.raises(ValueError, match="not found"):
        request_check(req_input)


def test_k_invalid_request_vector_length(silberschatz_banker_state):
    with pytest.raises(ValueError, match="Request vector length"):
        ResourceRequestInput(
            state=silberschatz_banker_state,
            process_id="P1",
            request=[1, 0],  # 2 elements instead of 3
        )


def test_k_negative_request_vector_value(silberschatz_banker_state):
    with pytest.raises(ValueError, match="negative"):
        ResourceRequestInput(
            state=silberschatz_banker_state,
            process_id="P1",
            request=[-1, 0, 1],
        )


def test_k_invalid_wait_for_graph_unknown_node():
    with pytest.raises(ValueError, match="not in declared nodes"):
        SingleInstanceDetectionInput(
            nodes=["P0", "P1"],
            edges=[["P0", "P99"]],
        )
