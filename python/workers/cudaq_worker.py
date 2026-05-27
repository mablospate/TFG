"""CUDA-Q benchmark worker subprocess."""
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


def _setup_grover(config: BenchmarkConfig, cudaq_target: str):
    import cudaq
    from python.cudaq.grover import search, grover_circuit

    t0 = time.perf_counter()
    cudaq.set_target(cudaq_target)
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def search_call(n, target, num_shots):
        return search(n, target, simulator=None, num_shots=num_shots)

    def build_call(n, target):
        return grover_circuit(n, target)

    return startup_ms, search_call, build_call


def _setup_shor(config: BenchmarkConfig, cudaq_target: str):
    from python.cudaq.shor.shor import find_factor as _ff
    from python.cudaq.shor.shor import order_finding_circuit

    t0 = time.perf_counter()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(
            N,
            simulator=cudaq_target,
            num_tries=3,
            num_shots_per_trial=config.num_shots,
        )

    def shor_build_call(N):
        return order_finding_circuit(2, N)

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
        cudaq_target = cfg.get("cudaq_target", "qpp-cpu")
    except Exception as e:
        write_error(f"invalid config: {e}")
        return

    try:
        import cudaq  # noqa: F401
    except ImportError as e:
        write_error(f"cudaq not available: {e}")
        return

    try:
        if algo == "grover":
            startup_ms, search_call, build_call = _setup_grover(config, cudaq_target)
            result = run_grover_worker(
                "cudaq", n, config, hw, contributor,
                startup_ms, search_call, build_call,
            )
        elif algo == "shor":
            startup_ms, factor_call, shor_build_call = _setup_shor(config, cudaq_target)
            result = run_shor_worker(
                "cudaq", n, config, hw, contributor,
                startup_ms, factor_call,
                shor_build_call=shor_build_call,
            )
        else:
            write_error(f"unknown algo: {algo}")
            return
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        write_error(f"cudaq {algo} n={n} failed: {e}")
        return

    write_result(result)


if __name__ == "__main__":
    main()
