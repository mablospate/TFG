"""QDisLib benchmark worker subprocess."""
from __future__ import annotations

import sys
import time
import traceback
import warnings

# QDisLib uses \( in docstrings which triggers SyntaxWarning in Python 3.12+
warnings.filterwarnings("ignore", "invalid escape sequence", SyntaxWarning)

from python.benchmark_core import BenchmarkConfig
from python.hardware import detect_hardware
from python.workers._base import (
    read_config,
    run_grover_worker,
    run_shor_worker,
    write_error,
    write_result,
)

# Circuit cutting is not executed. QDisLib's find_cut algorithm hangs
# indefinitely for Grover oracle circuits (densely entangled, graph
# partitioning does not converge) and Shor order-finding circuits
# (modular exponentiation sub-circuit is too connected). The
# implementations remain in python/qdislib/grover.py and shor/shor.py.


def _setup_grover(config: BenchmarkConfig):
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from python.qdislib.grover import search
    from python.qiskit.grover import grover_circuit as qiskit_grover_circuit

    t0 = time.perf_counter()
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    sampler = SamplerV2()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def search_call(n, target, num_shots):
        return search(n, target, sampler=sampler, pass_manager=pm, num_shots=num_shots)

    def build_call(n, target):
        return qiskit_grover_circuit(n, target)

    return startup_ms, search_call, build_call


def _setup_shor(config: BenchmarkConfig):
    from python.qdislib.shor.shor import find_factor as _ff
    from python.qiskit.shor.shor import order_finding_circuit as _qiskit_order_finding_circuit
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.perf_counter()
    _build_pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(N, num_tries=3, num_shots_per_trial=config.num_shots)

    def shor_build_call(N):
        qc = _qiskit_order_finding_circuit(2, N)
        if qc == 0:
            return None
        return _build_pm.run(qc)

    return startup_ms, factor_call, shor_build_call


def _parse_config():
    """Parse stdin JSON config. Returns (cfg, config, algo, n, contributor) or None on error."""
    try:
        cfg = read_config()
    except Exception as e:
        write_error(f"failed to read config: {e}")
        return None
    try:
        config = BenchmarkConfig(
            n_repetitions=cfg["n_repetitions"],
            num_shots=cfg["num_shots"],
        )
        return cfg, config, cfg["algo"], cfg["n"], cfg.get("contributor", "")
    except Exception as e:
        write_error(f"invalid config: {e}")
        return None


def main() -> None:
    parsed = _parse_config()
    if parsed is None:
        return
    cfg, config, algo, n, contributor = parsed

    try:
        hw = detect_hardware()
    except Exception as e:
        write_error(f"hardware detection failed: {e}")
        return

    try:
        import Qdislib  # noqa: F401
    except Exception as e:
        write_error(f"qdislib not available: {e}")
        return

    try:
        if algo == "grover":
            startup_ms, search_call, build_call = _setup_grover(config)
            result = run_grover_worker(
                "qdislib", n, config, hw, contributor,
                startup_ms, search_call, build_call,
            )
        elif algo == "shor":
            startup_ms, factor_call, shor_build_call = _setup_shor(config)
            result = run_shor_worker(
                "qdislib", n, config, hw, contributor,
                startup_ms, factor_call,
                shor_build_call=shor_build_call,
            )
        else:
            write_error(f"unknown algo: {algo}")
            return
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        write_error(f"qdislib {algo} n={n} failed: {e}")
        return

    write_result(result)


if __name__ == "__main__":
    main()
