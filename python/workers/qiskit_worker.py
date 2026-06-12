"""Qiskit benchmark worker subprocess.

Reads a JSON config from stdin, runs Grover or Shor at the requested n, and
emits a single enriched JSON result dict on stdout.
"""
from __future__ import annotations

import time

from python.benchmark_core import BenchmarkConfig
from python.workers._base import worker_main


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
    worker_main(
        "qiskit",
        _setup_grover,
        _setup_shor,
        import_check=lambda: (__import__("qiskit"), __import__("qiskit_aer")),
    )


if __name__ == "__main__":
    main()
