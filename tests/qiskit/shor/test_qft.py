import pytest
from qiskit.circuit import QuantumCircuit, QuantumRegister

from python.qiskit.shor.qft import QFTFullGate


def test_qft():
    q_reg = QuantumRegister(4)
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4), q_reg)
    expected_gate_counts = {"h": 4, "cp": 6, "swap": 2}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    # No swap
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, do_swaps=False), q_reg)
    expected_gate_counts = {"h": 4, "cp": 6}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    # approximate QFT
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, approximation_degree=1), q_reg)
    expected_gate_counts = {"h": 4, "cp": 5, "swap": 2}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, approximation_degree=2), q_reg)
    expected_gate_counts = {"h": 4, "cp": 3, "swap": 2}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts

    # Insert barriers
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(4, insert_barriers=True), q_reg)
    expected_gate_counts = {"h": 4, "cp": 6, "swap": 2, "barrier": 4}
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == expected_gate_counts


def test_qft_approximation_degree_invalid():
    # The constructor enforces 0 <= approximation_degree < num_qubits.
    with pytest.raises(AssertionError):
        QFTFullGate(4, approximation_degree=4)
    with pytest.raises(AssertionError):
        QFTFullGate(4, approximation_degree=5)
    with pytest.raises(AssertionError):
        QFTFullGate(2, approximation_degree=2)


def test_qft_n1_degenerate():
    # The 1-qubit QFT degenerates to a single Hadamard (no controlled-phase, no swap).
    q_reg = QuantumRegister(1)
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(1), q_reg)
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == {"h": 1}

    # Same when swaps are disabled.
    qc = QuantumCircuit(q_reg)
    qc.append(QFTFullGate(1, do_swaps=False), q_reg)
    actual_gate_counts = qc.decompose().count_ops()
    assert actual_gate_counts == {"h": 1}
