import math

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import ZGate


def build_oracle(n: int, target: int) -> QuantumCircuit:
    """
    Build a phase oracle that flips the phase of the target state |target>.
    Uses a multi-controlled Z gate decomposed following Barenco et al. (1995).

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
    Returns:
        QuantumCircuit: Oracle circuit on n qubits.
    """
    assert 0 <= target < 2**n, f"Target {target} out of range for {n} qubits."

    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # Flip qubits where target has a 0 bit, so |target> maps to |11...1>
    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    # Multi-controlled Z: flips phase of |11...1>
    qc.append(ZGate().control(n - 1), qr[:])

    # Undo the X flips
    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    """
    Build the Grover diffusion operator (inversion about the mean): 2|s><s| - I,
    where |s> is the uniform superposition state. Based on Grover (1996).

    Args:
        n: Number of qubits.
    Returns:
        QuantumCircuit: Diffuser circuit on n qubits.
    """
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # H on all qubits
    for i in range(n):
        qc.h(qr[i])

    # Phase flip on |00...0>: X → MCZ → X
    for i in range(n):
        qc.x(qr[i])

    qc.append(ZGate().control(n - 1), qr[:])

    for i in range(n):
        qc.x(qr[i])

    # H on all qubits
    for i in range(n):
        qc.h(qr[i])

    return qc


def grover_circuit(
    n: int, target: int, num_iterations: int | None = None
) -> QuantumCircuit:
    """
    Build the full Grover search circuit for n qubits searching for state |target>.
    Uses floor(pi/4 * sqrt(2^n)) iterations by default, as per Grover (1996).

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
    Returns:
        QuantumCircuit: Grover search circuit.
    """
    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="result")
    qc = QuantumCircuit(qr, cr)

    # Prepare uniform superposition
    for i in range(n):
        qc.h(qr[i])

    # Grover iterations: oracle + diffuser
    oracle = build_oracle(n, target)
    diffuser = build_diffuser(n)
    for _ in range(num_iterations):
        qc.compose(oracle, qubits=qr, inplace=True)
        qc.compose(diffuser, qubits=qr, inplace=True)

    # Measure
    qc.measure(qr, cr)

    return qc


def search(
    n: int,
    target: int,
    sampler,
    pass_manager,
    num_iterations: int | None = None,
    num_shots: int = 1024,
) -> tuple[int, dict[str, int]]:
    """
    Execute Grover's search algorithm on a simulator.

    Args:
        n: Number of qubits.
        target: Integer representation of the target state to search for.
        sampler: Sampler primitive.
        pass_manager: Pass manager for transpilation.
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
        num_shots: Number of circuit sampling runs. Default value: 1024.
    Returns:
        tuple[int, dict[str, int]]: The first element is the most frequent measurement
                                    outcome as an integer. The second element is the
                                    distribution of measurement outcomes.
    """
    iters = (
        num_iterations
        if num_iterations is not None
        else math.floor(math.pi / 4 * math.sqrt(2**n))
    )

    qc = grover_circuit(n, target, num_iterations=iters)
    qc_isa = pass_manager.run(qc)

    print(f"Start Grover search for |{target}> in {n}-qubit space ({iters} iterations)")
    dist = sampler.run([qc_isa], shots=num_shots).result()[0].data.result.get_counts()

    found = int(max(dist, key=dist.get), 2)
    if found == target:
        print(
            f"Found target state |{target}> with probability {dist[max(dist, key=dist.get)] / num_shots:.2%}"
        )
    else:
        print(f"Most frequent state was |{found}>, expected |{target}>")

    return found, dist
