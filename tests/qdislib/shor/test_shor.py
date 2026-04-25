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
