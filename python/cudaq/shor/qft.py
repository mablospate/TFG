import math

import cudaq


def apply_qft(kernel: cudaq.Kernel, qubits: list, n: int) -> None:
    """
    Apply the Quantum Fourier Transform on the first n qubits.

    Args:
        kernel: CUDA-Q kernel being built.
        qubits: List of qubit references.
        n: Number of qubits to apply QFT on.
    """
    for i in range(n):
        kernel.h(qubits[i])
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            kernel.cr1(angle, qubits[j], qubits[i])

    for i in range(n // 2):
        kernel.swap(qubits[i], qubits[n - 1 - i])


def apply_inverse_qft(kernel: cudaq.Kernel, qubits: list, n: int) -> None:
    """
    Apply the inverse Quantum Fourier Transform on the first n qubits.

    Args:
        kernel: CUDA-Q kernel being built.
        qubits: List of qubit references.
        n: Number of qubits to apply inverse QFT on.
    """
    for i in range(n // 2):
        kernel.swap(qubits[i], qubits[n - 1 - i])

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            angle = -math.pi / (2 ** (j - i))
            kernel.cr1(angle, qubits[j], qubits[i])
        kernel.h(qubits[i])
