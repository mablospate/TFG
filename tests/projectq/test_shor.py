from python.projectq.shor import find_factor, find_order


def test_find_order_2_mod_15() -> None:
    """Order of 2 in Z_15 is 4 (2^4 = 16 = 15*1 + 1)."""
    r, dist = find_order(2, 15, num_shots=10)
    assert r == 4, f"Expected order 4, got {r}"


def test_find_order_7_mod_15() -> None:
    """Order of 7 in Z_15 is 4 (7^4 = 2401 = 15*160 + 1)."""
    r, dist = find_order(7, 15, num_shots=10)
    assert r == 4, f"Expected order 4, got {r}"


def test_find_factor_15() -> None:
    """Shor's algorithm finds a non-trivial factor of 15."""
    factor = find_factor(15, seed=13)
    assert factor in [3, 5], f"Expected 3 or 5, got {factor}"


def test_find_factor_even() -> None:
    """Even numbers are handled classically (factor = 2)."""
    factor = find_factor(6)
    assert factor == 2, f"Expected 2, got {factor}"


def test_find_factor_35() -> None:
    """Shor's algorithm finds a non-trivial factor of 35."""
    factor = find_factor(35, seed=7, num_shots_per_trial=20, num_tries=5)
    assert factor in [5, 7], f"Expected 5 or 7, got {factor}"
