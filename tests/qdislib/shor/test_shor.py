from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from python.qdislib.shor.shor import find_factor, find_order


def test_find_order() -> None:
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)

    got_order, _ = find_order(2, 15, aer_sampler, pm, num_shots=10)
    assert got_order == 4, f"Got {got_order}, want 4"

    got_order, _ = find_order(7, 15, aer_sampler, pm, num_shots=10)
    assert got_order == 4, f"Got {got_order}, want 4"


def test_find_factor() -> None:
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)

    got_factor = find_factor(
        15, aer_sampler, pm, num_tries=1, num_shots_per_trial=1, seed=13
    )
    assert got_factor == 3, f"Got {got_factor}, want 3"


def test_find_factor_even() -> None:
    # Classical pre-check: even N -> returns 2 without invoking the sampler.
    assert find_factor(14) == 2
    assert find_factor(100) == 2


def test_find_factor_prime_power() -> None:
    # Classical pre-check: prime power N = d^k -> returns d without quantum.
    assert find_factor(9) == 3
    assert find_factor(25) == 5
    assert find_factor(27) == 3


def test_find_factor_multiple_tries() -> None:
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)

    got_factor = find_factor(
        15, aer_sampler, pm, num_tries=3, num_shots_per_trial=5, seed=42
    )
    assert got_factor in (3, 5), f"Expected a non-trivial factor of 15, got {got_factor}"


def test_find_order_gcd_not_one() -> None:
    # gcd(6, 15) = 3, so the quantum circuit is not built. The qdislib
    # find_order function checks the sentinel from order_finding_circuit
    # and returns the (0, {}) tuple without raising.
    r, dist = find_order(6, 15, num_shots=10)
    assert r == 0
    assert dist == {}
