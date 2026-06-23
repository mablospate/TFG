"""QDisLib benchmark worker subprocess."""
from __future__ import annotations

import multiprocessing
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


_FIND_CUT_TIMEOUT_S = 30  # must match grover.py / shor/shor.py


def _run_cutting_loop(
    call_fn,
    n_reps: int,
    result: dict,
) -> None:
    """Run call_fn n_reps times and store per-rep cutting data.

    call_fn must return (exp_val, _cuts, find_ms, exec_ms).
    wire_cutting timeout is handled inside call_fn; exec_ms=0 when it times out.
    """
    cutting_times: list[float] = []
    find_times: list[float] = []
    exec_times: list[float] = []
    exp_vals: list[float] = []
    for i in range(n_reps):
        print(f"  [cutting] rep {i + 1}/{n_reps}...", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        exp_val, _cuts, find_ms, exec_ms = call_fn()
        cutting_times.append((time.perf_counter() - t0) * 1000.0)
        find_times.append(find_ms)
        exec_times.append(exec_ms)
        exp_vals.append(exp_val)
        print(f"  rep {i + 1}/{n_reps}  {cutting_times[-1]:.1f}ms  [cutting]", file=sys.stderr, flush=True)
        # If find_cut timed out on this rep it will also hang on all remaining
        # reps for the same n — skip them to avoid wasting minutes.
        if i == 0 and find_ms >= (_FIND_CUT_TIMEOUT_S - 1) * 1000:
            print(f"  [cutting] find_cut timed out, skipping remaining {n_reps - 1} reps",
                  file=sys.stderr, flush=True)
            break

    if not cutting_times:
        return
    result["raw_cutting_times_ms"] = [round(t, 3) for t in cutting_times]
    result["raw_cutting_find_times_ms"] = [round(t, 3) for t in find_times]
    result["raw_cutting_exec_times_ms"] = [round(t, 3) for t in exec_times]
    result["raw_cutting_exp_values"] = [round(v, 6) for v in exp_vals]
    result["cutting_find_time_ms"] = round(float(np.median(find_times)), 3)
    exec_nonzero = [t for t in exec_times if t > 0]
    result["cutting_exec_time_ms"] = round(float(np.median(exec_nonzero)), 3) if exec_nonzero else 0.0
    result["cutting_expectation_value"] = round(float(np.mean(exp_vals)), 6)


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

    # Use 'spawn' so wire_cutting's internal multiprocessing doesn't inherit
    # locks from this process (which is already a subprocess of run.py).
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

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
