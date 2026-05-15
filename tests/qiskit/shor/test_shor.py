from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from python.qiskit.shor.shor import find_factor, find_order, order_finding_circuit


def test_find_order() -> None:
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)  # Seed the sampler for deterministic test

    N = 15
    A = 2
    # 2^4 = 16 = 15*1 + 1
    want_order = 4
    # Case: Default circuit
    got_order, _ = find_order(A, N, aer_sampler, pm, num_shots=10)
    assert got_order == want_order, f"Got {got_order}, want {want_order}"
    # Case: One control circuit
    got_order, _ = find_order(
        A, N, aer_sampler, pm, num_shots=10, one_control_circuit=True
    )
    assert got_order == want_order, f"Got {got_order}, want {want_order}"

    N = 15
    A = 7
    # 7^4 = 2401 = 15*160 + 1
    want_order = 4
    # Case: Default circuit
    got_order, _ = find_order(A, N, aer_sampler, pm, num_shots=10)
    assert got_order == want_order, f"Got {got_order}, want {want_order}"
    # Case: One control circuit
    got_order, _ = find_order(
        A, N, aer_sampler, pm, num_shots=10, one_control_circuit=True
    )
    assert got_order == want_order, f"Got {got_order}, want {want_order}"


def test_find_factor() -> None:
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)  # Seed the sampler for deterministic test

    N = 15
    want_factor = 3
    got_factor = find_factor(
        N, aer_sampler, pm, num_tries=1, num_shots_per_trial=1, seed=13
    )

    assert got_factor == want_factor, f"Got {got_factor}, want {want_factor}"
    assert N % got_factor == 0 and 1 < got_factor < N


def test_find_factor_even() -> None:
    # Classical pre-check: N even -> returns 2 without invoking the quantum circuit.
    assert find_factor(14, None, None) == 2
    assert find_factor(100, None, None) == 2


def test_find_factor_prime_power() -> None:
    # Classical pre-check: N = d^k -> returns d without invoking the quantum circuit.
    assert find_factor(9, None, None) == 3
    assert find_factor(25, None, None) == 5
    assert find_factor(27, None, None) == 3


def test_find_factor_multiple_tries() -> None:
    # With num_tries=3 the algorithm gets multiple chances to land on a base
    # whose order leads to a non-trivial factor.
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)

    got_factor = find_factor(
        15, aer_sampler, pm, num_tries=3, num_shots_per_trial=5, seed=42
    )
    assert got_factor in (3, 5), (
        f"Expected a non-trivial factor of 15, got {got_factor}"
    )
    assert 15 % got_factor == 0 and 1 < got_factor < 15


def test_order_finding_circuit_gcd_not_one() -> None:
    # gcd(6, 15) = 3, so the quantum circuit is not built and a sentinel (0) is returned.
    assert order_finding_circuit(6, 15) == 0
    assert order_finding_circuit(10, 15) == 0


def test_find_order_gcd_not_one() -> None:
    # find_order should short-circuit when gcd(A, N) > 1, without touching the sampler/pm.
    result = find_order(6, 15, sampler=None, pass_manager=None)
    assert result == (0, {})
