"""QDisLib benchmark worker subprocess."""
from __future__ import annotations

import sys
import time
import traceback
import warnings

# QDisLib uses \( in docstrings which triggers SyntaxWarning in Python 3.12+
warnings.filterwarnings("ignore", "invalid escape sequence", SyntaxWarning)

import numpy as np

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
    from python.qdislib.grover import search, search_with_cutting
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

    def cutting_call(n, target, num_shots):
        return search_with_cutting(n, target, pass_manager=pm, num_shots=num_shots)

    return startup_ms, search_call, build_call, cutting_call


def _setup_shor(config: BenchmarkConfig):
    from python.qdislib.shor.shor import find_factor as _ff
    from python.qdislib.shor.shor import find_factor_with_cutting as _ffc
    from python.qiskit.shor.shor import order_finding_circuit as _qiskit_order_finding_circuit
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.perf_counter()
    _build_pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(N, num_tries=3, num_shots_per_trial=config.num_shots)

    def cutting_factor_call(N):
        return _ffc(N, num_shots_per_trial=config.num_shots)

    def shor_build_call(N):
        qc = _qiskit_order_finding_circuit(2, N)
        if qc == 0:
            return None
        return _build_pm.run(qc)

    return startup_ms, factor_call, cutting_factor_call, shor_build_call


def _run_cutting_loop(
    call_fn,
    n_reps: int,
    result: dict,
) -> None:
    """Run call_fn n_reps times, collect wall times and last outputs, write 3 cutting keys.

    call_fn must return (exp_val, _cuts, find_ms).
    """
    cutting_times: list[float] = []
    last_exp = 0.0
    last_find_ms = 0.0
    for _ in range(n_reps):
        t0 = time.perf_counter()
        exp_val, _cuts, find_ms = call_fn()
        cutting_times.append((time.perf_counter() - t0) * 1000.0)
        last_exp = exp_val
        last_find_ms = find_ms
    result["cutting_wall_time_ms"] = round(float(np.median(cutting_times)), 3)
    result["cutting_find_time_ms"] = round(last_find_ms, 3)
    result["cutting_expectation_value"] = round(last_exp, 6)


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
            startup_ms, search_call, build_call, cutting_call = _setup_grover(config)
            result = run_grover_worker(
                "qdislib", n, config, hw, contributor,
                startup_ms, search_call, build_call,
            )
        elif algo == "shor":
            startup_ms, factor_call, cutting_factor_call, shor_build_call = _setup_shor(config)
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

    if algo == "grover":
        try:
            _run_cutting_loop(
                lambda: cutting_call(n, n, config.num_shots),
                config.n_repetitions,
                result,
            )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(f"[QDisLib cutting] grover n={n} failed: {e}", file=sys.stderr)
    elif algo == "shor":
        try:
            _run_cutting_loop(
                lambda: cutting_factor_call(n),
                config.n_repetitions,
                result,
            )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(f"[QDisLib cutting] shor n={n} failed: {e}", file=sys.stderr)

    write_result(result)


if __name__ == "__main__":
    main()
