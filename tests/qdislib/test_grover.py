from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from python.qdislib.grover import search


def make_backend():
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    aer_sampler = AerSampler(seed=5)
    return aer_sampler, pm


def test_grover_search() -> None:
    aer_sampler, pm = make_backend()

    found, dist = search(3, 5, aer_sampler, pm, num_shots=100)
    assert found == 5, f"Expected target 5, got {found}"
    assert dist.get("101", 0) > 80, f"Target probability too low: {dist.get('101', 0)}/100"

    found, _ = search(4, 11, aer_sampler, pm, num_shots=100)
    assert found == 11, f"Expected target 11, got {found}"


def test_grover_search_target_zero() -> None:
    aer_sampler, pm = make_backend()

    found, _ = search(3, 0, aer_sampler, pm, num_shots=100)
    assert found == 0, f"Expected target 0, got {found}"
