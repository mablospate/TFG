from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import psutil

from python.benchmark_core import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_run,
    compute_jsd,
    measure_build_time,
)
from python.hardware import HardwareInfo, detect_hardware


def read_config() -> dict:
    return json.loads(sys.stdin.read())


def write_result(result: dict) -> None:
    print(json.dumps(result), flush=True)


def write_error(message: str) -> None:
    print(json.dumps({"status": "error", "error": message}), flush=True)
    sys.exit(1)


def _n_qubits_shor(N: int) -> int:
    return math.ceil(math.log2(N)) * 2


def _get_framework_version(framework: str) -> str:
    """Return the installed package version, or 'unknown' if not found."""
    try:
        return importlib.metadata.version(framework)
    except Exception:
        return "unknown"


def _build_base_envelope(
    hw: HardwareInfo,
    config: BenchmarkConfig,
    contributor: str,
    framework_version: str,
) -> dict:
    """Return the 14 hw/config keys shared by all result dicts."""
    return {
        "status": "ok",
        "contributor_name": contributor,
        "hostname": hw.hostname,
        "os": hw.os,
        "os_version": hw.os_version,
        "cpu_model": hw.cpu_model,
        "cpu_cores_physical": hw.cpu_cores_physical,
        "cpu_cores_logical": hw.cpu_cores_logical,
        "cpu_gflops": hw.cpu_gflops,
        "ram_total_gb": hw.ram_total_gb,
        "gpu_model": hw.gpu_model,
        "gpu_vram_gb": hw.gpu_vram_gb,
        "runtime_version": hw.python_version,
        "num_shots": config.num_shots,
        "n_repetitions": config.n_repetitions,
        "framework_version": framework_version,
    }


def run_grover_worker(
    framework: str,
    n: int,
    config: BenchmarkConfig,
    hw: HardwareInfo,
    contributor: str,
    startup_ms: float,
    search_call,
    build_call,
) -> dict:
    """Run Grover at qubit count `n` and return enriched result dict.

    Mirrors the logic of run.benchmark_grover_at_n.
    """
    target = n
    build_ms = measure_build_time(lambda: build_call(n, target))

    result = benchmark_run(
        lambda: search_call(n, target, config.num_shots),
        config,
        framework=framework,
        algorithm="grover",
        n_qubits=n,
    )

    if result.raw_times_ms:
        mean_ms = float(np.mean(result.raw_times_ms))
        std_ms = (
            float(np.std(result.raw_times_ms, ddof=1))
            if len(result.raw_times_ms) > 1
            else 0.0
        )
    else:
        mean_ms = std_ms = 0.0

    result.startup_time_ms = startup_ms
    result.build_time_ms = build_ms
    result.simulation_time_ms = max(
        0.0, result.wall_time_median_ms - build_ms
    )

    try:
        _found, dist = search_call(n, target, config.num_shots)
        total = sum(dist.values())
        empirical = {k: v / total for k, v in dist.items()} if total > 0 else {}
        theoretical = {format(target, f"0{n}b"): 1.0}
        result.jsd = compute_jsd(empirical, theoretical)
    except Exception as e:
        print(f"  [WARN] JSD failed for {framework} n={n}: {e}", file=sys.stderr)
        result.jsd = 0.0

    framework_version = _get_framework_version(framework)

    return {
        **_build_base_envelope(hw, config, contributor, framework_version),
        **dataclasses.asdict(result),
        "wall_time_mean_ms": mean_ms,
        "wall_time_std_ms": std_ms,
    }


def run_shor_worker(
    framework: str,
    N: int,
    config: BenchmarkConfig,
    hw: HardwareInfo,
    contributor: str,
    startup_ms: float,
    factor_call,
    shor_build_call=None,
) -> dict:
    """Run Shor for N and return enriched result dict.

    Mirrors the logic of run.benchmark_shor_at_n.
    """
    n_qubits = _n_qubits_shor(N)

    build_time_ms = (
        measure_build_time(lambda: shor_build_call(N))
        if shor_build_call is not None
        else 0.0
    )

    _proc = psutil.Process()
    _proc.cpu_percent()  # discard first call
    _max_cpu_pct = hw.cpu_cores_logical * 100.0

    times_ms: list[float] = []
    factors: list[int] = []
    peak_rss_mb: float = 0.0
    cpu_samples: list[float] = []
    for _ in range(config.n_repetitions):
        t0 = time.perf_counter()
        f = factor_call(N)
        times_ms.append((time.perf_counter() - t0) * 1000)
        factors.append(f)
        peak_rss_mb = max(peak_rss_mb, _proc.memory_info().rss / 1024 / 1024)
        cpu_samples.append(min(_proc.cpu_percent(), _max_cpu_pct))

    cpu_mean = float(np.mean(cpu_samples)) if cpu_samples else 0.0

    if not times_ms:
        raise RuntimeError("No se completó ninguna repetición")
    arr = np.array(times_ms)
    median_ms = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_ms = float(q75 - q25)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0
    success_rate = sum(1 for f in factors if f not in (1, N)) / len(factors)
    factor_found = max(set(factors), key=factors.count)

    framework_version = _get_framework_version(framework)

    result = BenchmarkResult(
        wall_time_median_ms=median_ms,
        wall_time_iqr_ms=iqr_ms,
        peak_memory_rss_mb=peak_rss_mb,
        cv=cv,
        startup_time_ms=startup_ms,
        build_time_ms=build_time_ms,
        simulation_time_ms=max(0.0, median_ms - startup_ms),
        cpu_percent_mean=cpu_mean,
        jsd=0.0,
        framework=framework,
        algorithm="shor",
        n_qubits=n_qubits,
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version,
        platform_info=platform.platform(),
        raw_times_ms=times_ms,
    )

    return {
        **_build_base_envelope(hw, config, contributor, framework_version),
        "n_to_factor": N,
        "factor_found": factor_found,
        "success_rate": success_rate,
        **dataclasses.asdict(result),
        "wall_time_mean_ms": mean_ms,
        "wall_time_std_ms": std_ms,
    }


def worker_main(
    framework_name: str,
    setup_grover_fn,
    setup_shor_fn,
    *,
    import_check=None,
    extra_cfg: dict | None = None,
) -> None:
    """Generic worker entry point — handles config parsing, routing, and error reporting.

    Args:
        framework_name: Name of the framework (e.g. "qiskit", "cirq").
        setup_grover_fn: Callable(config, **extra) -> (startup_ms, search_call, build_call).
        setup_shor_fn: Callable(config, **extra) -> (startup_ms, factor_call, shor_build_call).
        import_check: Optional callable that imports the framework and raises ImportError if unavailable.
        extra_cfg: Dict of {key: default_value} for extra cfg fields to extract (e.g. {"cudaq_target": "qpp-cpu"}).
    """
    import traceback

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
        extra = {k: cfg.get(k, v) for k, v in (extra_cfg or {}).items()}
    except Exception as e:
        write_error(f"invalid config: {e}")
        return

    if import_check is not None:
        try:
            import_check()
        except ImportError as e:
            write_error(f"{framework_name} not available: {e}")
            return

    try:
        if algo == "grover":
            startup_ms, search_call, build_call = setup_grover_fn(config, **extra)
            result = run_grover_worker(
                framework_name, n, config, hw, contributor,
                startup_ms, search_call, build_call,
            )
        elif algo == "shor":
            startup_ms, factor_call, shor_build_call = setup_shor_fn(config, **extra)
            result = run_shor_worker(
                framework_name, n, config, hw, contributor,
                startup_ms, factor_call,
                shor_build_call=shor_build_call,
            )
        else:
            write_error(f"unknown algo: {algo}")
            return
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        write_error(f"{framework_name} {algo} n={n} failed: {e}")
        return

    write_result(result)
