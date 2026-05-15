import pytest

pytest.importorskip("projectq")

from python.projectq.shor.shor import find_factor, find_order


def test_find_order() -> None:
    got_order, _ = find_order(2, 15, num_shots=10)
    assert got_order == 4, f"Got {got_order}, want 4"

    got_order, _ = find_order(7, 15, num_shots=10)
    assert got_order == 4, f"Got {got_order}, want 4"


def test_find_factor() -> None:
    N = 15
    got_factor = find_factor(N, num_tries=1, num_shots_per_trial=1, seed=13)
    assert got_factor == 3, f"Got {got_factor}, want 3"
    assert N % got_factor == 0 and 1 < got_factor < N


def test_find_factor_even() -> None:
    assert find_factor(14) == 2
    assert find_factor(100) == 2


def test_find_factor_prime_power() -> None:
    assert find_factor(9) == 3
    assert find_factor(25) == 5
    assert find_factor(27) == 3


def test_find_factor_multiple_tries() -> None:
    got_factor = find_factor(15, num_tries=3, num_shots_per_trial=5, seed=42)
    assert got_factor in (3, 5), (
        f"Expected a non-trivial factor of 15, got {got_factor}"
    )
    assert 15 % got_factor == 0 and 1 < got_factor < 15


def test_find_order_gcd_not_one() -> None:
    # gcd(6, 15) = 3, so the quantum circuit is not built and find_order
    # returns the (0, {}) sentinel without raising.
    r, dist = find_order(6, 15, num_shots=10)
    assert r == 0
    assert dist == {}


def test_order_finding_circuit_gcd_not_one() -> None:
    # ProjectQ has no standalone order_finding_circuit function; the gcd guard
    # lives inside find_order. Verify that find_order returns the sentinel (0, {})
    # whenever gcd(A, N) > 1, covering the same early-exit path.
    r, dist = find_order(6, 15, num_shots=10)
    assert r == 0
    assert dist == {}
    r2, dist2 = find_order(10, 15, num_shots=10)
    assert r2 == 0
    assert dist2 == {}
