import pytest

pytest.importorskip("cudaq")

from python.cudaq.grover import search


def test_grover_search() -> None:
    found, dist = search(3, 5, num_shots=100)
    assert found == 5, f"Expected target 5, got {found}"
    assert dist.get("101", 0) > 80, f"Target probability too low: {dist.get('101', 0)}/100"

    found, _ = search(4, 11, num_shots=100)
    assert found == 11, f"Expected target 11, got {found}"


def test_grover_search_target_zero() -> None:
    found, _ = search(3, 0, num_shots=100)
    assert found == 0, f"Expected target 0, got {found}"


def test_grover_search_explicit_iterations() -> None:
    found, dist = search(3, 5, num_iterations=2, num_shots=100)
    assert found == 5, f"Expected target 5, got {found}"
    assert dist.get("101", 0) > 80, (
        f"Target probability too low: {dist.get('101', 0)}/100"
    )


def test_grover_search_n1() -> None:
    # The CUDA-Q Grover kernel has an explicit n==1 branch (single Z instead
    # of a multi-controlled Z). On a 1-qubit space Grover does not amplify
    # to certainty (the oracle phase is global), but the kernel must still
    # build, run, and produce a valid 1-qubit distribution without raising.
    found, dist = search(1, 1, num_iterations=1, num_shots=50)
    assert found in (0, 1), f"Expected a 1-qubit outcome, got {found}"
    # Distribution must only contain valid 1-qubit bitstrings and sum to num_shots.
    assert set(dist.keys()).issubset({"0", "1"})
    assert sum(dist.values()) == 50

    # Also verify with target=0 to exercise the X-flip path of the n=1 oracle.
    found0, dist0 = search(1, 0, num_iterations=1, num_shots=50)
    assert found0 in (0, 1)
    assert set(dist0.keys()).issubset({"0", "1"})
    assert sum(dist0.values()) == 50
