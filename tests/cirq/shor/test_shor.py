import pytest

cirq = pytest.importorskip("cirq")

from python.cirq.shor.shor import find_factor, find_order, order_finding_circuit


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


def test_find_factor_even() -> None:
    # Classical pre-check: even N -> returns 2 without invoking the simulator.
    assert find_factor(14, None) == 2
    assert find_factor(100, None) == 2


def test_find_factor_prime_power() -> None:
    # Classical pre-check: prime power N = d^k -> returns d without quantum.
    assert find_factor(9, None) == 3
    assert find_factor(25, None) == 5
    assert find_factor(27, None) == 3


def test_find_factor_multiple_tries() -> None:
    simulator = cirq.Simulator(seed=5)

    got_factor = find_factor(
        15, simulator, num_tries=3, num_shots_per_trial=5, seed=42
    )
    assert got_factor in (3, 5), f"Expected a non-trivial factor of 15, got {got_factor}"


def test_order_finding_circuit_gcd_not_one() -> None:
    # gcd(6, 15) = 3, so the circuit isn't built and an empty Circuit is returned.
    qc = order_finding_circuit(6, 15)
    assert isinstance(qc, cirq.Circuit)
    assert len(qc) == 0  # Empty sentinel circuit, no operations.


def test_find_order_gcd_not_one() -> None:
    # find_order should short-circuit when gcd(A, N) > 1, without touching the simulator.
    result = find_order(6, 15, simulator=None)
    assert result == (0, {})
