"""Cirq benchmark worker subprocess."""
from __future__ import annotations

import time

from python.benchmark_core import BenchmarkConfig
from python.workers._base import worker_main


def _setup_grover(config: BenchmarkConfig):
    import cirq
    from python.cirq.grover import search, grover_circuit

    t0 = time.perf_counter()
    simulator = cirq.Simulator()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def search_call(n, target, num_shots):
        return search(n, target, simulator, num_shots=num_shots)

    def build_call(n, target):
        return grover_circuit(n, target)

    return startup_ms, search_call, build_call


def _setup_shor(config: BenchmarkConfig):
    import cirq
    from python.cirq.shor.shor import find_factor as _ff
    from python.cirq.shor.shor import order_finding_circuit

    t0 = time.perf_counter()
    sim = cirq.Simulator()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(N, sim, num_tries=3, num_shots_per_trial=config.num_shots)

    def shor_build_call(N):
        return order_finding_circuit(2, N)

    return startup_ms, factor_call, shor_build_call


def main() -> None:
    worker_main(
        "cirq",
        _setup_grover,
        _setup_shor,
        import_check=lambda: __import__("cirq"),
    )


if __name__ == "__main__":
    main()
