from python.projectq.grover import build_diffuser, build_oracle, grover_circuit, search

from projectq.ops import All, Measure


def test_grover_search_3_qubits() -> None:
    """Grover search in a 3-qubit space finds the correct target."""
    found, dist = search(3, 5, num_shots=100)
    assert found == 5, f"Expected target 5, got {found}"
    # The target should appear with high probability (> 80%)
    target_count = dist.get("101", 0)
    assert target_count > 80, f"Target probability too low: {target_count}/100"


def test_grover_search_4_qubits() -> None:
    """Grover search in a 4-qubit space finds the correct target."""
    found, dist = search(4, 11, num_shots=100)
    assert found == 11, f"Expected target 11, got {found}"


def test_grover_search_target_zero() -> None:
    """Grover search for target 0 works correctly."""
    found, dist = search(3, 0, num_shots=100)
    assert found == 0, f"Expected target 0, got {found}"


def test_grover_circuit_returns_eng_and_qureg() -> None:
    """grover_circuit returns a (engine, qubit register) tuple that can be measured."""
    eng, qureg = grover_circuit(3, 5)
    assert len(qureg) == 3
    All(Measure) | qureg
    eng.flush()
    bits = [int(q) for q in qureg]
    val = sum(b * 2**i for i, b in enumerate(bits))
    # With optimal iterations, we should get the target most of the time
    assert val == 5, f"Expected 5 from grover_circuit, got {val}"


def test_build_oracle_and_diffuser() -> None:
    """build_oracle and build_diffuser apply gates in-place without error."""
    from projectq import MainEngine
    from projectq.ops import H
    from projectq.backends import Simulator

    eng = MainEngine(backend=Simulator(), engine_list=[])
    qureg = eng.allocate_qureg(3)
    All(H) | qureg
    build_oracle(3, 5, eng, qureg)
    build_diffuser(3, eng, qureg)
    All(Measure) | qureg
    eng.flush()
    # Should not raise any errors
