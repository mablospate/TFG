import math

from projectq import MainEngine
from projectq.ops import H, X, Z, All, Measure
from projectq.meta import Control
from projectq.backends import Simulator


def build_oracle(n: int, target: int, eng, qureg) -> None:
    """
    Apply a phase oracle that flips the phase of the target state |target>.
    Uses a multi-controlled Z gate, following Barenco et al. (1995).

    Gates are applied in-place to the given qubit register.

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
        eng: ProjectQ MainEngine.
        qureg: Qubit register of size n.
    """
    assert 0 <= target < 2**n, f"Target {target} out of range for {n} qubits."

    # Flip qubits where target has a 0 bit, so |target> maps to |11...1>
    for i in range(n):
        if not (target >> i) & 1:
            X | qureg[i]

    # Multi-controlled Z: flips phase of |11...1>
    with Control(eng, qureg[:-1]):
        Z | qureg[-1]

    # Undo the X flips
    for i in range(n):
        if not (target >> i) & 1:
            X | qureg[i]


def build_diffuser(n: int, eng, qureg) -> None:
    """
    Apply the Grover diffusion operator (inversion about the mean): 2|s><s| - I,
    where |s> is the uniform superposition state. Based on Grover (1996).

    Gates are applied in-place to the given qubit register.

    Args:
        n: Number of qubits.
        eng: ProjectQ MainEngine.
        qureg: Qubit register of size n.
    """
    # H on all qubits
    All(H) | qureg

    # Phase flip on |00...0>: X -> MCZ -> X
    All(X) | qureg

    with Control(eng, qureg[:-1]):
        Z | qureg[-1]

    All(X) | qureg

    # H on all qubits
    All(H) | qureg


def grover_circuit(
    n: int, target: int, num_iterations: int | None = None
) -> tuple:
    """
    Build and apply the full Grover search circuit for n qubits searching for
    state |target>. Uses floor(pi/4 * sqrt(2^n)) iterations by default, as per
    Grover (1996).

    In ProjectQ, gates are applied imperatively. This function creates a fresh
    engine and qubit register, applies all gates (superposition, oracle+diffuser
    iterations), but does NOT measure. Measurement is left to the caller.

    Args:
        n: Number of qubits.
        target: Integer representation of the target state (0 <= target < 2^n).
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
    Returns:
        tuple: (eng, qureg) — the engine and qubit register after applying
               all Grover operations (before measurement).
    """
    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    eng = MainEngine(backend=Simulator(), engine_list=[])
    qureg = eng.allocate_qureg(n)

    # Prepare uniform superposition
    All(H) | qureg

    # Grover iterations: oracle + diffuser
    for _ in range(num_iterations):
        build_oracle(n, target, eng, qureg)
        build_diffuser(n, eng, qureg)

    return eng, qureg


def search(
    n: int,
    target: int,
    simulator=None,
    pass_manager=None,
    num_iterations: int | None = None,
    num_shots: int = 1024,
) -> tuple[int, dict[str, int]]:
    """
    Execute Grover's search algorithm on a simulator.

    ProjectQ uses an engine-based imperative model. Each shot requires a full
    engine lifecycle (allocate, apply gates, measure, flush, deallocate).

    Args:
        n: Number of qubits.
        target: Integer representation of the target state to search for.
        simulator: Unused (ProjectQ creates its own Simulator backend).
                   Kept for interface compatibility.
        pass_manager: Unused (ProjectQ has no external pass manager).
                      Kept for interface compatibility.
        num_iterations: Number of Grover iterations. If None, uses the optimal value.
        num_shots: Number of independent simulation runs. Default value: 1024.
    Returns:
        tuple[int, dict[str, int]]: The first element is the most frequent measurement
                                    outcome as an integer. The second element is the
                                    distribution of measurement outcomes.
    """
    iters = num_iterations if num_iterations is not None else math.floor(
        math.pi / 4 * math.sqrt(2**n)
    )

    print(f"Start Grover search for |{target}> in {n}-qubit space ({iters} iterations)")

    dist: dict[str, int] = {}

    for _ in range(num_shots):
        # Build fresh engine and circuit for each shot
        eng = MainEngine(backend=Simulator(), engine_list=[])
        qureg = eng.allocate_qureg(n)

        # Prepare uniform superposition
        All(H) | qureg

        # Grover iterations
        for _iter in range(iters):
            build_oracle(n, target, eng, qureg)
            build_diffuser(n, eng, qureg)

        # Measure
        All(Measure) | qureg
        eng.flush()

        # Read result as bitstring (MSB first, matching Qiskit convention)
        bits = [int(qureg[i]) for i in range(n)]
        bitstring = "".join(str(bits[i]) for i in reversed(range(n)))

        dist[bitstring] = dist.get(bitstring, 0) + 1

    found = int(max(dist, key=dist.get), 2)
    if found == target:
        print(
            f"Found target state |{target}> with probability "
            f"{dist[max(dist, key=dist.get)] / num_shots:.2%}"
        )
    else:
        print(f"Most frequent state was |{found}>, expected |{target}>")

    return found, dist
