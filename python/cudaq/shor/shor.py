import math
import random
from fractions import Fraction

import cudaq

from python.cudaq.shor.permutation import build_mod_exp_permutation, controlled_swap_permutation
from python.cudaq.shor.qft import apply_inverse_qft


def order_finding_circuit(
    A: int, N: int, precision: int | None = None
) -> cudaq.Kernel:
    """
    Build circuit to find the order of A in Z_N using phase estimation.
    Uses a swap-based permutation network for modular exponentiation.

    Args:
        A: Base integer (must be coprime to N).
        N: Modulus.
        precision: Number of qubits for phase estimation. If None, uses 2*ceil(log2(N)).
    Returns:
        cudaq.Kernel: Order finding circuit.
    """
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return None

    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    total_qubits = m + n

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(total_qubits)

    ctrl_qubits = [qubits[i] for i in range(m)]
    tgt_qubits = [qubits[m + i] for i in range(n)]

    # Prepare control register in superposition
    for i in range(m):
        kernel.h(ctrl_qubits[i])

    # Prepare target register in |1> state
    kernel.x(tgt_qubits[0])

    # Controlled modular exponentiation
    for i in range(m):
        perm = build_mod_exp_permutation(A, N, 2**i)
        if perm:
            controlled_swap_permutation(kernel, ctrl_qubits[i], tgt_qubits, perm)

    # Apply inverse QFT on control register
    apply_inverse_qft(kernel, ctrl_qubits, m)

    # Measure control register
    for i in range(m):
        kernel.mz(ctrl_qubits[i])

    return kernel


def _get_order_from_dist(dist: dict, A: int, N: int, precision: int) -> int:
    """
    Extract the order r from measurement distribution using continued fractions.

    Args:
        dist: Measurement distribution {bitstring: count}.
        A: Base integer.
        N: Modulus.
        precision: Number of precision qubits used.
    Returns:
        int: The order r if found, 0 otherwise.
    """
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    for i in range(min(10, len(sorted_outputs))):
        bitstring = sorted_outputs[i]
        if all(c == '0' for c in bitstring):
            continue
        x = int(bitstring, 2)
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            print(
                f"Found value {r} for the order of {A} in Z_{N}. If running on "
                f"noisy quantum hardware, {r} might be a multiple of the order instead."
            )
            return r
    print(f"Failed to find order of {A} in Z_{N}")
    return 0


def find_order(
    A: int,
    N: int,
    simulator=None,
    pass_manager=None,
    precision: int | None = None,
    num_shots: int = 10,
) -> tuple[int, dict[str, int]]:
    """
    Find the order of the integer A in Z_N on a CUDA-Q simulator.

    Args:
        A: Base integer.
        N: Modulus.
        simulator: CUDA-Q target name (e.g. "qpp-cpu", "nvidia"). If None, uses default.
        pass_manager: Unused in CUDA-Q. Kept for interface compatibility.
        precision: Number of qubits for phase estimation.
        num_shots: Number of circuit sampling runs. Default value: 10.
    Returns:
        tuple[int, dict[str, int]]: Order (or 0) and measurement distribution.
    """
    if simulator is not None:
        cudaq.set_target(simulator)

    m = precision if precision is not None else 2 * math.ceil(math.log2(N))

    kernel = order_finding_circuit(A, N, precision=m)
    if kernel is None:
        return 0, {}

    print(f"Start search for the order of {A} in Z_{N}")
    result = cudaq.sample(kernel, shots_count=num_shots)

    n = math.ceil(math.log2(N))
    dist = {}
    for bitstring, count in result.items():
        ctrl_bits = bitstring[:m] if len(bitstring) > m else bitstring
        key = ctrl_bits[::-1]
        dist[key] = dist.get(key, 0) + count

    r = _get_order_from_dist(dist, A, N, precision=m)
    return r, dist


def find_factor(
    N: int,
    simulator=None,
    pass_manager=None,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    seed: int | None = None,
) -> int:
    """
    Find a factor of N using Shor's algorithm.

    Args:
        N: Integer to factor.
        simulator: CUDA-Q target name. If None, uses default.
        pass_manager: Unused in CUDA-Q. Kept for interface compatibility.
        num_tries: Number of trials.
        num_shots_per_trial: Number of order finding circuit runs per trial.
        seed: Random seed.
    Returns:
        int: Found factor or 1 if no success.
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
            a, N, simulator=simulator, pass_manager=pass_manager,
            num_shots=num_shots_per_trial,
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
