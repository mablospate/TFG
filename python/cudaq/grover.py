import math

import cudaq


def build_oracle(n: int, target: int) -> cudaq.Kernel:
    """
    Build a phase oracle that flips the phase of the target state |target>.
    Uses a multi-controlled Z gate decomposed following Barenco et al. (1995).

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
    Returns:
        cudaq.Kernel: Oracle kernel on n qubits.
    """
    assert 0 <= target < 2**n, f"Target {target} out of range for {n} qubits."

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n)

    # Flip qubits where target has a 0 bit, so |target> maps to |11...1>
    for i in range(n):
        if not (target >> i) & 1:
            kernel.x(qubits[i])

    # Multi-controlled Z: flips phase of |11...1>
    if n == 1:
        kernel.z(qubits[0])
    else:
        controls = [qubits[i] for i in range(n - 1)]
        kernel.cz(controls, qubits[n - 1])

    # Undo the X flips
    for i in range(n):
        if not (target >> i) & 1:
            kernel.x(qubits[i])

    return kernel


def build_diffuser(n: int) -> cudaq.Kernel:
    """
    Build the Grover diffusion operator (inversion about the mean): 2|s><s| - I,
    where |s> is the uniform superposition state. Based on Grover (1996).

    Args:
        n: Number of qubits.
    Returns:
        cudaq.Kernel: Diffuser kernel on n qubits.
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n)

    # H on all qubits
    for i in range(n):
        kernel.h(qubits[i])

    # Phase flip on |00...0>: X -> MCZ -> X
    for i in range(n):
        kernel.x(qubits[i])

    if n == 1:
        kernel.z(qubits[0])
    else:
        controls = [qubits[i] for i in range(n - 1)]
        kernel.cz(controls, qubits[n - 1])

    for i in range(n):
        kernel.x(qubits[i])

    # H on all qubits
    for i in range(n):
        kernel.h(qubits[i])

    return kernel


def grover_circuit(
    n: int, target: int, num_iterations: int | None = None
) -> cudaq.Kernel:
    """
    Build the full Grover search circuit for n qubits searching for state |target>.
    Uses floor(pi/4 * sqrt(2^n)) iterations by default, as per Grover (1996).

    Since CUDA-Q kernels cannot easily compose sub-kernels dynamically,
    the full circuit (superposition + oracle + diffuser + measurement) is
    built inline in a single kernel.

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
    Returns:
        cudaq.Kernel: Grover search kernel.
    """
    assert 0 <= target < 2**n, f"Target {target} out of range for {n} qubits."

    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n)

    # Prepare uniform superposition
    for i in range(n):
        kernel.h(qubits[i])

    # Grover iterations: oracle + diffuser
    for _ in range(num_iterations):
        # --- Oracle: flip phase of |target> ---
        # X gates where target has a 0 bit
        for i in range(n):
            if not (target >> i) & 1:
                kernel.x(qubits[i])

        # Multi-controlled Z
        if n == 1:
            kernel.z(qubits[0])
        else:
            controls = [qubits[i] for i in range(n - 1)]
            kernel.cz(controls, qubits[n - 1])

        # Undo X flips
        for i in range(n):
            if not (target >> i) & 1:
                kernel.x(qubits[i])

        # --- Diffuser: inversion about the mean ---
        for i in range(n):
            kernel.h(qubits[i])

        for i in range(n):
            kernel.x(qubits[i])

        if n == 1:
            kernel.z(qubits[0])
        else:
            controls = [qubits[i] for i in range(n - 1)]
            kernel.cz(controls, qubits[n - 1])

        for i in range(n):
            kernel.x(qubits[i])

        for i in range(n):
            kernel.h(qubits[i])

    # Measure all qubits
    kernel.mz(qubits)

    return kernel


def search(
    n: int,
    target: int,
    simulator=None,
    pass_manager=None,
    num_iterations: int | None = None,
    num_shots: int = 1024,
) -> tuple[int, dict[str, int]]:
    """
    Execute Grover's search algorithm on a CUDA-Q simulator.

    Args:
        n: Number of qubits.
        target: Integer representation of the target state to search for.
        simulator: CUDA-Q target name (e.g. "qpp-cpu", "nvidia"). If None, uses default.
        pass_manager: Unused in CUDA-Q (compilation is handled by MLIR). Kept for interface
                      compatibility.
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
        num_shots: Number of circuit sampling runs. Default value: 1024.
    Returns:
        tuple[int, dict[str, int]]: The first element is the most frequent measurement
                                    outcome as an integer. The second element is the
                                    distribution of measurement outcomes.
    """
    if simulator is not None:
        cudaq.set_target(simulator)

    iters = num_iterations if num_iterations is not None else math.floor(
        math.pi / 4 * math.sqrt(2**n)
    )

    kernel = grover_circuit(n, target, num_iterations=iters)

    print(f"Start Grover search for |{target}> in {n}-qubit space ({iters} iterations)")
    result = cudaq.sample(kernel, shots_count=num_shots)

    # Convert SampleResult to dict[str, int].
    # CUDA-Q bitstrings have qubit 0 as the leftmost character (MSB).
    # To match the Qiskit convention where qubit 0 is the LSB, reverse the bitstrings.
    dist = {bitstring[::-1]: count for bitstring, count in result.items()}

    # Find the most frequent outcome.
    most_frequent = max(dist, key=dist.get)
    found = int(most_frequent, 2)
    if found == target:
        print(
            f"Found target state |{target}> with probability "
            f"{dist[most_frequent] / num_shots:.2%}"
        )
    else:
        print(f"Most frequent state was |{found}>, expected |{target}>")

    return found, dist
