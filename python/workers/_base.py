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
        0.0, result.wall_time_median_ms - startup_ms - build_ms
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

    try:
        framework_version = importlib.metadata.version(framework)
    except Exception:
        framework_version = "unknown"

    return {
        "status": "ok",
        "contributor_name": contributor,
        "hostname": hw.hostname,
        "os": hw.os,
        "os_version": hw.os_version,
        "cpu_model": hw.cpu_model,
        "cpu_cores_physical": hw.cpu_cores_physical,
        "cpu_cores_logical": hw.cpu_cores_logical,
        "cpu_freq_mhz": hw.cpu_freq_mhz,
        "ram_total_gb": hw.ram_total_gb,
        "gpu_model": hw.gpu_model,
        "gpu_vram_gb": hw.gpu_vram_gb,
        "runtime_version": hw.python_version,
        "num_shots": config.num_shots,
        "n_repetitions": config.n_repetitions,
        "framework_version": framework_version,
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
) -> dict:
    """Run Shor for N and return enriched result dict.

    Mirrors the logic of run.benchmark_shor_at_n.
    """
    n_qubits = _n_qubits_shor(N)

    times_ms: list[float] = []
    factors: list[int] = []
    for _ in range(config.n_repetitions):
        t0 = time.perf_counter()
        f = factor_call(N)
        times_ms.append((time.perf_counter() - t0) * 1000)
        factors.append(f)

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

    try:
        framework_version = importlib.metadata.version(framework)
    except Exception:
        framework_version = "unknown"

    result = BenchmarkResult(
        wall_time_median_ms=median_ms,
        wall_time_iqr_ms=iqr_ms,
        peak_memory_rss_mb=0.0,
        cv=cv,
        startup_time_ms=startup_ms,
        build_time_ms=0.0,
        simulation_time_ms=max(0.0, median_ms - startup_ms),
        cpu_percent_mean=0.0,
        jsd=0.0,
        scaling_alpha=0.0,
        scaling_beta=0.0,
        scaling_data={},
        framework=framework,
        algorithm="shor",
        n_qubits=n_qubits,
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version,
        platform_info=platform.platform(),
        raw_times_ms=times_ms,
    )

    return {
        "status": "ok",
        "contributor_name": contributor,
        "hostname": hw.hostname,
        "os": hw.os,
        "os_version": hw.os_version,
        "cpu_model": hw.cpu_model,
        "cpu_cores_physical": hw.cpu_cores_physical,
        "cpu_cores_logical": hw.cpu_cores_logical,
        "cpu_freq_mhz": hw.cpu_freq_mhz,
        "ram_total_gb": hw.ram_total_gb,
        "gpu_model": hw.gpu_model,
        "gpu_vram_gb": hw.gpu_vram_gb,
        "runtime_version": hw.python_version,
        "num_shots": config.num_shots,
        "n_repetitions": config.n_repetitions,
        "framework_version": framework_version,
        "n_to_factor": N,
        "factor_found": factor_found,
        "success_rate": success_rate,
        **dataclasses.asdict(result),
        "wall_time_mean_ms": mean_ms,
        "wall_time_std_ms": std_ms,
    }
