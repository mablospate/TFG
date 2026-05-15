import math

import cirq


def build_oracle(n: int, target: int) -> cirq.Circuit:
    """
    Build a phase oracle that flips the phase of the target state |target>.
    Uses a multi-controlled Z gate decomposed following Barenco et al. (1995).

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
    Returns:
        cirq.Circuit: Oracle circuit on n qubits.
    """
    assert 0 <= target < 2**n, f"Target {target} out of range for {n} qubits."

    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # Flip qubits where target has a 0 bit, so |target> maps to |11...1>.
    # Cirq uses big-endian ordering: LineQubit(0) is the MSB when measured.
    # Bit i of target maps to qubit (n-1-i) so the measurement reads target.
    for i in range(n):
        if not (target >> i) & 1:
            circuit.append(cirq.X(qubits[n - 1 - i]))

    # Multi-controlled Z: flips phase of |11...1>
    mcz = cirq.Z.controlled(num_controls=n - 1)
    circuit.append(mcz.on(*qubits))

    # Undo the X flips
    for i in range(n):
        if not (target >> i) & 1:
            circuit.append(cirq.X(qubits[n - 1 - i]))

    return circuit


def build_diffuser(n: int) -> cirq.Circuit:
    """
    Build the Grover diffusion operator (inversion about the mean): 2|s><s| - I,
    where |s> is the uniform superposition state. Based on Grover (1996).

    Args:
        n: Number of qubits.
    Returns:
        cirq.Circuit: Diffuser circuit on n qubits.
    """
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # H on all qubits
    circuit.append(cirq.H.on_each(*qubits))

    # Phase flip on |00...0>: X -> MCZ -> X
    circuit.append(cirq.X.on_each(*qubits))

    mcz = cirq.Z.controlled(num_controls=n - 1)
    circuit.append(mcz.on(*qubits))

    circuit.append(cirq.X.on_each(*qubits))

    # H on all qubits
    circuit.append(cirq.H.on_each(*qubits))

    return circuit


def grover_circuit(
    n: int, target: int, num_iterations: int | None = None
) -> cirq.Circuit:
    """
    Build the full Grover search circuit for n qubits searching for state |target>.
    Uses floor(pi/4 * sqrt(2^n)) iterations by default, as per Grover (1996).

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
    Returns:
        cirq.Circuit: Grover search circuit.
    """
    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # Prepare uniform superposition
    circuit.append(cirq.H.on_each(*qubits))

    # Grover iterations: oracle + diffuser
    oracle = build_oracle(n, target)
    diffuser = build_diffuser(n)
    for _ in range(num_iterations):
        circuit += oracle
        circuit += diffuser

    # Measure
    circuit.append(cirq.measure(*qubits, key="result"))

    return circuit


def search(
    n: int,
    target: int,
    simulator,
    pass_manager=None,
    num_iterations: int | None = None,
    num_shots: int = 1024,
) -> tuple[int, dict[str, int]]:
    """
    Execute Grover's search algorithm on a simulator.

    Args:
        n: Number of qubits.
        target: Integer representation of the target state to search for.
        simulator: cirq.Simulator instance.
        pass_manager: Optional pass manager (Cirq has no mandatory transpiler for
                      simulation, so this can be None).
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
        num_shots: Number of circuit sampling runs. Default value: 1024.
    Returns:
        tuple[int, dict[str, int]]: The first element is the most frequent measurement
                                    outcome as an integer. The second element is the
                                    distribution of measurement outcomes as
                                    {bitstring: count}.
    """
    iters = (
        num_iterations
        if num_iterations is not None
        else math.floor(math.pi / 4 * math.sqrt(2**n))
    )

    qc = grover_circuit(n, target, num_iterations=iters)

    # Apply pass manager if provided (optional in Cirq)
    if pass_manager is not None:
        qc = pass_manager(qc)

    print(f"Start Grover search for |{target}> in {n}-qubit space ({iters} iterations)")
    result = simulator.run(qc, repetitions=num_shots)
    histogram = result.histogram(key="result")

    # Convert histogram {int: count} to {bitstring: count} for consistency with Qiskit
    dist = {}
    for value, count in histogram.items():
        bitstring = format(value, f"0{n}b")
        dist[bitstring] = count

    found = max(histogram, key=histogram.get)
    if found == target:
        print(
            f"Found target state |{target}> with probability "
            f"{histogram[found] / num_shots:.2%}"
        )
    else:
        print(f"Most frequent state was |{found}>, expected |{target}>")

    return found, dist
