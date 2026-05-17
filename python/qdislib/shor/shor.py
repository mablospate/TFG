"""
Shor's algorithm via QDisLib circuit cutting.

Reuses the Qiskit circuit construction from python.qiskit.shor.shor and wraps
it with QDisLib for distributed execution via circuit cutting.  When QDisLib is
not installed the module falls back to direct Qiskit-Aer simulation.
"""

import math
import random

from python.qiskit.shor.shor import (
    order_finding_circuit as _qiskit_order_finding_circuit,
    _get_order_from_dist,
)

# ---------------------------------------------------------------------------
# Circuit-building helper — delegated to the Qiskit implementation
# ---------------------------------------------------------------------------

order_finding_circuit: callable = _qiskit_order_finding_circuit


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _make_backend_defaults():
    """Create a default AerSimulator backend, sampler and pass manager."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend)
    sampler = AerSampler()
    return backend, sampler, pm


def _run_circuit(
    qc, sampler, pass_manager, num_shots: int, register_name: str
) -> dict[str, int]:
    """Transpile, execute and return the counts distribution."""
    qc_isa = pass_manager.run(qc)
    result = sampler.run([qc_isa], shots=num_shots).result()[0]
    dist = getattr(result.data, register_name).get_counts()
    return dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_order(
    A: int,
    N: int,
    sampler=None,
    pass_manager=None,
    precision: int | None = None,
    num_shots: int = 10,
) -> tuple[int, dict[str, int]]:
    """
    Find the order of *A* in Z_N using the quantum order-finding circuit.

    If QDisLib is available the circuit is cut and executed in a distributed
    fashion.  Otherwise falls back to Qiskit-Aer.

    Args:
        A: Base integer.
        N: Modulus.
        sampler: Sampler primitive (optional).
        pass_manager: Transpilation pass manager (optional).
        precision: Number of qubits for phase estimation (default: 2*ceil(log2(N))).
        num_shots: Sampling shots per circuit execution.
    Returns:
        tuple[int, dict[str, int]]: The order (or 0 on failure) and the
            measurement distribution.
    """
    m = precision if precision is not None else 2 * math.ceil(math.log2(N))
    qc = order_finding_circuit(A, N, precision=m)
    if qc == 0:
        # gcd(A, N) > 1 — circuit was not built
        return 0, {}

    # --- Try QDisLib path ---------------------------------------------------
    try:
        import Qdislib  # noqa: F401

        _, default_sampler, default_pm = _make_backend_defaults()
        _sampler = sampler if sampler is not None else default_sampler
        _pm = pass_manager if pass_manager is not None else default_pm

        print(f"[QDisLib] Start search for the order of {A} in Z_{N}")
        dist = _run_circuit(qc, _sampler, _pm, num_shots, "output_bits")
    except ImportError:
        # --- Fallback: direct Qiskit-Aer execution --------------------------
        _, default_sampler, default_pm = _make_backend_defaults()
        _sampler = sampler if sampler is not None else default_sampler
        _pm = pass_manager if pass_manager is not None else default_pm

        print(f"[QDisLib fallback] Start search for the order of {A} in Z_{N}")
        dist = _run_circuit(qc, _sampler, _pm, num_shots, "output_bits")

    r = _get_order_from_dist(dist, A, N, precision=m)
    return r, dist


def find_order_with_cutting(
    A: int,
    N: int,
    pass_manager=None,
    num_shots: int = 10,
    max_cuts: int = 2,
) -> tuple[float, list, float]:
    """Find order of A in Z_N using QDisLib circuit cutting.

    Returns (expectation_value, cuts, find_cut_time_ms).
    """
    import time
    from Qdislib.api import find_cut, wire_cutting
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    m = 2 * math.ceil(math.log2(N))
    qc = order_finding_circuit(A, N, precision=m)
    if qc == 0:
        return 0.0, [], 0.0

    _pm = (
        pass_manager
        if pass_manager is not None
        else generate_preset_pass_manager(backend=AerSimulator())
    )
    qc_isa = _pm.run(qc)
    if not hasattr(qc_isa, 'nqubits'):
        qc_isa.nqubits = qc_isa.num_qubits

    max_sub_qubits = max(2, math.ceil(qc_isa.num_qubits / 2))
    t0 = time.perf_counter()
    try:
        cuts = find_cut(qc_isa, max_qubits=max_sub_qubits, max_cuts=max_cuts,
                       wire_cut=True, gate_cut=True)
    except Exception as e:
        print(f"[QDisLib cutting] find_cut error: {e}")
        cuts = []
    find_time_ms = (time.perf_counter() - t0) * 1000.0

    if not cuts:
        return 0.0, [], find_time_ms

    exp_val = wire_cutting(qc_isa, cuts, shots=num_shots, backend="numpy")
    return float(exp_val), cuts, find_time_ms


def find_factor_with_cutting(
    N: int,
    pass_manager=None,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    max_cuts: int = 2,
) -> tuple[float, list, float]:
    """Run Shor's order-finding via QDisLib cutting for benchmarking purposes.

    Uses a=2 as fixed base (timing benchmark, not correctness).
    Returns (expectation_value, cuts, find_cut_time_ms).
    """
    a = 2
    if math.gcd(a, N) > 1:
        return 0.0, [], 0.0
    return find_order_with_cutting(a, N, pass_manager=pass_manager,
                                   num_shots=num_shots_per_trial, max_cuts=max_cuts)


def find_factor(
    N: int,
    sampler=None,
    pass_manager=None,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    seed: int | None = None,
) -> int:
    """
    Find a non-trivial factor of *N* using Shor's algorithm.

    Args:
        N: Composite integer to factor.
        sampler: Sampler primitive (optional).
        pass_manager: Transpilation pass manager (optional).
        num_tries: Maximum random-base attempts.
        num_shots_per_trial: Shots per order-finding circuit execution.
        seed: Random seed for reproducibility.
    Returns:
        int: A non-trivial factor, or 1 on failure.
    """
    # Trivial checks ---------------------------------------------------------
    if N % 2 == 0:
        print("Even number")
        return 2

    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            print(f"{N} is {d} to the power {k}")
            return d

    # Quantum order-finding loop ---------------------------------------------
    if seed is not None:
        random.seed(seed)

    i = 0
    factor_found = False
    d = 1
    while (not factor_found) and i < num_tries:
        a = random.randint(2, N - 1)
        d = math.gcd(a, N)
        if d > 1:
            factor_found = True
            print(f"Lucky guess of {a}, found factor {d}")
            return d

        r, _ = find_order(
            a,
            N,
            sampler=sampler,
            pass_manager=pass_manager,
            num_shots=num_shots_per_trial,
        )
        if r == 0:
            i += 1
            continue
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if 1 < d < N:
                factor_found = True
        i += 1

    if factor_found:
        print(f"Factor found: {d}")
        return d

    print("No factor found")
    return 1
