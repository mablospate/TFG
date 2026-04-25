import math
import random
from fractions import Fraction

import cirq

from python.cirq.shor.modular_exp import ModularExp


def order_finding_circuit(
    A: int, N: int, precision: int | None = None
) -> cirq.Circuit:
    """
    Build circuit to find the order of A in Z_N using quantum phase estimation
    with modular exponentiation.

    Args:
        A: Base integer (must be coprime to N).
        N: Modulus.
        precision: Number of qubits for the exponent (phase estimation) register.
                   If None, uses 2 * ceil(log2(N)).
    Returns:
        cirq.Circuit: Order finding circuit.
    """
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return cirq.Circuit()

    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    # Create qubit registers
    exponent_qubits = cirq.LineQubit.range(m)
    target_qubits = cirq.LineQubit.range(m, m + n)

    circuit = cirq.Circuit()

    # Prepare exponent register in superposition
    circuit.append(cirq.H.on_each(*exponent_qubits))

    # Prepare target register in |1> state
    circuit.append(cirq.X(target_qubits[0]))

    # Apply modular exponentiation gate
    mod_exp = ModularExp(
        target_size=n,
        exponent_size=m,
        base=A,
        modulus=N,
    )
    circuit.append(mod_exp.on(*target_qubits, *exponent_qubits))

    # Apply inverse QFT on the exponent register
    circuit.append(cirq.qft(*exponent_qubits, inverse=True))

    # Measure the exponent register
    circuit.append(cirq.measure(*exponent_qubits, key="result"))

    return circuit


def _get_order_from_dist(
    dist: dict[int, int], A: int, N: int, precision: int
) -> int:
    """
    Classical post-processing: extract the order r from measurement outcomes
    using continued fractions.

    Args:
        dist: Histogram {measurement_int: count}.
        A: Base integer.
        N: Modulus.
        precision: Number of precision qubits used.
    Returns:
        int: The order r if found, 0 otherwise.
    """
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    for i in range(min(10, len(sorted_outputs))):
        x = sorted_outputs[i]
        if x == 0:
            continue
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            print(
                f"Found value {r} for the order of {A} in Z_{N}. "
                f"If running on noisy quantum hardware, {r} might be a "
                f"multiple of the order instead."
            )
            return r
    print(f"Failed to find order of {A} in Z_{N}")
    return 0


def find_order(
    A: int,
    N: int,
    simulator,
    pass_manager=None,
    precision: int | None = None,
    num_shots: int = 10,
) -> tuple[int, dict[str, int]]:
    """
    Carry out the quantum order-finding algorithm: find the integer r such that
    A^r = 1 mod N.

    Args:
        A: Base integer.
        N: Modulus.
        simulator: cirq.Simulator instance.
        pass_manager: Optional pass manager (can be None for Cirq simulation).
        precision: Number of qubits for phase estimation.
        num_shots: Number of circuit sampling runs. Default value: 10.
    Returns:
        tuple[int, dict[str, int]]: Order (or 0) and measurement distribution.
    """
    m = precision if precision is not None else 2 * math.ceil(math.log2(N))
    qc = order_finding_circuit(A, N, precision=m)

    if not qc:
        return 0, {}

    if pass_manager is not None:
        qc = pass_manager(qc)

    print(f"Start search for the order of {A} in Z_{N}")
    result = simulator.run(qc, repetitions=num_shots)
    histogram = result.histogram(key="result")

    r = _get_order_from_dist(histogram, A, N, precision=m)

    dist = {}
    for value, count in histogram.items():
        bitstring = format(value, f"0{m}b")
        dist[bitstring] = count

    return r, dist


def find_factor(
    N: int,
    simulator,
    pass_manager=None,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    seed: int | None = None,
) -> int:
    """
    Carry out Shor's algorithm for finding a non-trivial factor of N.

    Args:
        N: Integer to factor.
        simulator: cirq.Simulator instance.
        pass_manager: Optional pass manager (can be None for Cirq simulation).
        num_tries: Number of random base trials.
        num_shots_per_trial: Number of order finding circuit runs per trial.
        seed: Random seed for reproducibility.
    Returns:
        int: A non-trivial factor of N, or 1 if no factor was found.
    """
    if N % 2 == 0:
        print("Even number")
        return 2

    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            print(f"{N} is {d} to the power {k}")
            return d

    i = 0
    factor_found = False
    d = 1
    if seed is not None:
        random.seed(seed)

    while (not factor_found) and i < num_tries:
        a = random.randint(2, N - 1)
        d = math.gcd(a, N)
        if d > 1:
            factor_found = True
            print(f"Lucky guess of {a}, found factor {d}")
            return d

        r, _ = find_order(
            a, N, simulator, pass_manager, num_shots=num_shots_per_trial,
        )
        if r == 0:
            i += 1
            continue
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if d > 1 and d < N:
                factor_found = True
        i += 1

    if factor_found:
        print(f"Factor found: {d}")
        return d

    print("No factor found")
    return 1
