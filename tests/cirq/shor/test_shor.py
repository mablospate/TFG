import pytest

cirq = pytest.importorskip("cirq")

from python.cirq.shor.shor import find_factor, find_order


def test_find_order() -> None:
    simulator = cirq.Simulator(seed=5)

    got_order, _ = find_order(2, 15, simulator, num_shots=10)
    assert got_order == 4, f"Got {got_order}, want 4"

    got_order, _ = find_order(7, 15, simulator, num_shots=10)
    assert got_order == 4, f"Got {got_order}, want 4"


def test_find_factor() -> None:
    simulator = cirq.Simulator(seed=5)

    got_factor = find_factor(
        15, simulator, num_tries=1, num_shots_per_trial=1, seed=13
    )
    assert got_factor == 3, f"Got {got_factor}, want 3"
