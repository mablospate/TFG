import pytest

pytest.importorskip("projectq")

from python.projectq.grover import search


def test_grover_search() -> None:
    found, dist = search(3, 5, num_shots=100)
    assert found == 5, f"Expected target 5, got {found}"
    assert dist.get("101", 0) > 80, (
        f"Target probability too low: {dist.get('101', 0)}/100"
    )

    found, dist = search(4, 11, num_shots=100)
    assert found == 11, f"Expected target 11, got {found}"
    assert dist[format(11, f"0{4}b")] > 100 * 0.5


def test_grover_search_target_zero() -> None:
    found, _ = search(3, 0, num_shots=100)
    assert found == 0, f"Expected target 0, got {found}"


def test_grover_search_explicit_iterations() -> None:
    found, dist = search(3, 5, num_iterations=2, num_shots=100)
    assert found == 5, f"Expected target 5, got {found}"
    assert dist.get("101", 0) > 80, (
        f"Target probability too low: {dist.get('101', 0)}/100"
    )
