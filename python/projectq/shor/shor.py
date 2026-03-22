import math
import random
from fractions import Fraction

from projectq import MainEngine
from projectq.ops import H, X, R, All, Measure
from projectq.meta import Control
from projectq.backends import Simulator
from projectq.libs.math import MultiplyByConstantModN


def _run_order_finding_once(
    A: int, N: int, precision: int
) -> str:
    """
    Run a single shot of the order-finding circuit using semi-classical QFT.

    Uses ProjectQ's built-in MultiplyByConstantModN for controlled modular
    exponentiation. The simulator emulates this classically (shortcut), which
    is acceptable for correctness testing but should be documented for
    benchmarking fairness.

    The semi-classical QFT measures control qubits one at a time, applying
    classically-conditioned phase corrections, which reduces the circuit to
    a single control qubit reused m times.

    Args:
        A: Base for modular exponentiation (must be coprime to N).
        N: Modulus.
        precision: Number of bits of precision (m) for phase estimation.
    Returns:
        str: Bitstring of length `precision` representing the measured phase.
    """
    n = math.ceil(math.log2(N))
    m = precision

    eng = MainEngine(backend=Simulator(), engine_list=[])
    ctrl = eng.allocate_qubit()
    target = eng.allocate_qureg(n)

    # Initialize target register to |1>
    X | target[0]

    # Measured bits (collected from MSB to LSB of the phase)
    measured_bits = []

    for i in range(m):
        # Prepare control qubit in |+>
        H | ctrl

        # Controlled modular multiplication: |1> -> |A^(2^(m-1-i)) mod N>
        power = pow(A, 2 ** (m - 1 - i), N)
        with Control(eng, ctrl):
            MultiplyByConstantModN(power, N) | target

        # Semi-classical QFT: apply classically-conditioned phase corrections
        # based on previously measured bits
        for j in range(i):
            if measured_bits[j]:
                R(-math.pi / 2 ** (i - j)) | ctrl

        H | ctrl

        # Measure the control qubit
        Measure | ctrl
        eng.flush()
        bit = int(ctrl)
        measured_bits.append(bit)

        # Reset control qubit for reuse
        if bit:
            X | ctrl

    # Measure target register so ProjectQ can deallocate qubits cleanly
    All(Measure) | target
    eng.flush()

    # The semi-classical QFT outputs bits in reversed order: the first
    # measured bit is the LSB of the QFT result. Reverse to get MSB-first
    # bitstring that matches the standard int(bitstring, 2) convention.
    bitstring = "".join(str(b) for b in reversed(measured_bits))
    return bitstring


def _get_order_from_dist(dist: dict, A: int, N: int, precision: int) -> int:
    """
    Extract the order r from a distribution of phase estimation measurements
    using continued fractions.

    Args:
        dist: Distribution of measurement outcomes {bitstring: count}.
        A: Base for modular exponentiation.
        N: Modulus.
        precision: Number of bits of precision used.
    Returns:
        int: The order r if found, otherwise 0.
    """
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    # Look for the order in the 10 most frequent outputs
    for i in range(min(10, len(sorted_outputs))):
        if sorted_outputs[i] == "0" * precision:
            continue
        x = int(sorted_outputs[i], 2)
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            print(
                f"Found value {r} for the order of {A} in Z_{N}. If running "
                f"on noisy quantum hardware, {r} might be a multiple of the "
                f"order instead."
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
    Find the order of the integer A in Z_N, i.e. the integer r such that
    A^r = 1 mod N, using quantum phase estimation on a simulator.

    Assumes that N is odd, N is not a power of a prime integer, and A and N
    are coprime.

    Uses ProjectQ's built-in MultiplyByConstantModN with a semi-classical QFT
    (single control qubit, sequential measurements with classical feedback).

    Args:
        A: Integer whose order in Z_N is to be found.
        N: Modulus.
        simulator: Unused (ProjectQ creates its own Simulator backend).
                   Kept for interface compatibility.
        pass_manager: Unused (ProjectQ has no external pass manager).
                      Kept for interface compatibility.
        precision: Number of qubits for phase estimation. If None, uses 2*ceil(log2(N)).
        num_shots: Number of circuit sampling runs. Default value: 10.
    Returns:
        tuple[int, dict[str, int]]: The first element is the order (if found)
                                    or zero (if not). The second element is the
                                    distribution of the measurement outcomes.
    """
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return 0, {}

    m = precision if precision is not None else 2 * math.ceil(math.log2(N))

    print(f"Start search for the order of {A} in Z_{N}")

    dist: dict[str, int] = {}
    for _ in range(num_shots):
        bitstring = _run_order_finding_once(A, N, m)
        dist[bitstring] = dist.get(bitstring, 0) + 1

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
    Find a non-trivial factor of N using Shor's algorithm.

    Args:
        N: Integer to factor (must be composite, odd, and not a prime power).
        simulator: Unused (ProjectQ creates its own Simulator backend).
                   Kept for interface compatibility.
        pass_manager: Unused (ProjectQ has no external pass manager).
                      Kept for interface compatibility.
        num_tries: Number of trials with random bases.
        num_shots_per_trial: Number of order-finding circuit runs per trial.
        seed: Random seed for reproducibility.
    Returns:
        int: A non-trivial factor of N, or 1 if no factor was found.
    """
    # Check if N is even
    if N % 2 == 0:
        print("Even number")
        return 2

    # Check if N is a non-trivial power
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

        # Run order finding circuit
        r, _ = find_order(
            a,
            N,
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
