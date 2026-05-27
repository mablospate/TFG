"""Qiskit benchmark worker subprocess.

Reads a JSON config from stdin, runs Grover or Shor at the requested n, and
emits a single enriched JSON result dict on stdout.
"""
from __future__ import annotations

import sys
import time
import traceback

from python.benchmark_core import BenchmarkConfig
from python.hardware import detect_hardware
from python.workers._base import (
    read_config,
    run_grover_worker,
    run_shor_worker,
    write_error,
    write_result,
)


def _setup_grover(config: BenchmarkConfig):
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from python.qiskit.grover import search, grover_circuit

    t0 = time.perf_counter()
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    sampler = SamplerV2()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def search_call(n, target, num_shots):
        return search(n, target, sampler, pm, num_shots=num_shots)

    def build_call(n, target):
        return grover_circuit(n, target)

    return startup_ms, search_call, build_call


def _setup_shor(config: BenchmarkConfig):
    from python.qiskit.shor.shor import find_factor as _ff
    from python.qiskit.shor.shor import order_finding_circuit
    from qiskit_aer.primitives import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.perf_counter()
    sampler = SamplerV2()
    pm = generate_preset_pass_manager(optimization_level=1, backend=sampler._backend)
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(N, sampler, pm, num_tries=3, num_shots_per_trial=config.num_shots)

    def shor_build_call(N):
        qc = order_finding_circuit(2, N)
        if qc == 0:
            return None
        return pm.run(qc)

    return startup_ms, factor_call, shor_build_call


def main() -> None:
    try:
        cfg = read_config()
    except Exception as e:
        write_error(f"failed to read config: {e}")
        return

    try:
        hw = detect_hardware()
        config = BenchmarkConfig(
            n_repetitions=cfg["n_repetitions"],
            num_shots=cfg["num_shots"],
        )
        algo = cfg["algo"]
        n = cfg["n"]
        contributor = cfg.get("contributor", "")
    except Exception as e:
        write_error(f"invalid config: {e}")
        return

    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
    except ImportError as e:
        write_error(f"qiskit not available: {e}")
        return

    try:
        if algo == "grover":
            startup_ms, search_call, build_call = _setup_grover(config)
            result = run_grover_worker(
                "qiskit", n, config, hw, contributor,
                startup_ms, search_call, build_call,
            )
        elif algo == "shor":
            startup_ms, factor_call, shor_build_call = _setup_shor(config)
            result = run_shor_worker(
                "qiskit", n, config, hw, contributor,
                startup_ms, factor_call,
                shor_build_call=shor_build_call,
            )
        else:
            write_error(f"unknown algo: {algo}")
            return
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        write_error(f"qiskit {algo} n={n} failed: {e}")
        return

    write_result(result)


if __name__ == "__main__":
    main()
