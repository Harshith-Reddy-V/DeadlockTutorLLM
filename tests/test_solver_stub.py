import pytest
from backend.solver.engine import calculate_need
from backend.solver.models import BankerStateInput


def test_calculate_need_matrix():
    # Silberschatz classic sample dimensions
    max_matrix = [
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3]
    ]
    alloc_matrix = [
        [0, 1, 0],
        [2, 0, 0],
        [3, 0, 2],
        [2, 1, 1],
        [0, 0, 2]
    ]

    expected_need = [
        [7, 4, 3],
        [1, 2, 2],
        [6, 0, 0],
        [0, 1, 1],
        [4, 3, 1]
    ]

    need = calculate_need(max_matrix, alloc_matrix)
    assert need == expected_need


def test_calculate_need_invalid_allocation():
    # Allocation exceeding Max should raise ValueError
    max_matrix = [[1, 2]]
    alloc_matrix = [[2, 1]]
    with pytest.raises(ValueError):
        calculate_need(max_matrix, alloc_matrix)
