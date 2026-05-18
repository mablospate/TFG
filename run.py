"""TFG quantum benchmarking entry point.

Run with: uv run python run.py
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import psutil

from python.benchmark_core import (
    BenchmarkConfig,
    compute_jsd,
    fit_scaling_curve,
)
from python.hardware import HardwareInfo, detect_hardware


DOCKER_IMAGE: str = os.getenv("DOCKER_IMAGE", "dev")


BANNER = r"""
==========================================================
  TFG  -  Quantum Benchmarking Suite
==========================================================
"""


# ---------------------------------------------------------------------------
# Framework availability detection
# ---------------------------------------------------------------------------



FRAMEWORKS: list[str] = ["qiskit", "cirq", "cudaq", "qdislib"]


# ---------------------------------------------------------------------------
# Rust framework binaries
# ---------------------------------------------------------------------------
#
# Each entry maps a framework name (as it appears in the output table) to the
# absolute path of its Grover binary in target/release/. The binaries are
# stand-alone CLI tools that accept --n / --target / --shots and emit a single
# JSON object on stdout containing at least:
#   {"framework", "algorithm", "n", "target", "shots", "found",
#    "time_ms", "distribution"}
#
# Stderr is ignored (some crates emit warnings there). The Rust-reported
# `time_ms` is used as the simulation time so we don't include Python
# subprocess.run() overhead.

_RUST_BIN_DIR = pathlib.Path(__file__).parent / "target" / "release"


def _rust_bin(name: str) -> pathlib.Path:
    found = shutil.which(name)
    return pathlib.Path(found) if found else _RUST_BIN_DIR / name


RUST_FRAMEWORKS: dict[str, pathlib.Path] = {
    "q1tsim":   _rust_bin("q1tsim-grover"),
    "quantr":   _rust_bin("quantr-grover"),
    "quantrs2": _rust_bin("quantrs2-grover"),
    "qcgpu":    _rust_bin("qcgpu-grover"),
}


RUST_FRAMEWORKS_SHOR: dict[str, pathlib.Path] = {
    "q1tsim":   _rust_bin("q1tsim-shor"),
    "quantr":   _rust_bin("quantr-shor"),
    "quantrs2": _rust_bin("quantrs2-shor"),
    "qcgpu":    _rust_bin("qcgpu-shor"),
}


def _n_qubits_shor(N: int) -> int:
    return math.ceil(math.log2(N)) * 2



# Frameworks that cannot run meaningfully on a given (os, arch) combination.
# Key: (hw.os, arch)  →  set of framework names to hard-exclude.
_PLATFORM_EXCLUSIONS: dict[tuple[str, str], set[str]] = {
    ("macos", "x86_64"): {"cudaq"},  # no wheels for Intel Mac
    ("windows", "x86_64"): {
        "cudaq",
        "qip",
    },  # no native support / no wheel / no CI
}

# Frameworks that run but produce results not directly comparable across platforms.
# Each entry: framework → (condition, warning message)
_PLATFORM_WARNINGS: dict[str, tuple[bool, str]] = {}  # populated at runtime


# ---------------------------------------------------------------------------
# Platform presets (used with --platform <id>)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PlatformConfig:
    frameworks: list[str]
    cudaq_target: str = "qpp-cpu"
    quantrs2_gpu: bool = False
    warnings: list[tuple[str, str, list[str]]] = dataclasses.field(default_factory=list)



def _cudaq_cpu_warning(os_name: str, arch: str) -> tuple[str, str, list[str]]:
    return (
        "cudaq",
        f"ejecutando en modo CPU-only (target qpp-cpu) en {os_name} {arch}",
        [
            "No se detectó GPU NVIDIA — se usa Q++ C++ con OpenMP en lugar de cuStateVec.",
            "cuStateVec (GPU) puede ser 10-100× más rápido en circuitos grandes.",
            "GPU disponible en: Linux x86_64 + NVIDIA (CUDA 12/13), Linux aarch64 + NVIDIA.",
            "Los tiempos registrados aquí no son comparables con resultados Linux+NVIDIA.",
            "La fidelidad (JSD) sí es comparable entre targets.",
        ],
    )




# --- Rust framework warnings -------------------------------------------------

_Q1TSIM_WARNING: tuple[str, str, list[str]] = (
    "q1tsim",
    "crate abandonado desde 2019, deps antiguas (ndarray 0.12, rand 0.4)",
    [
        "Última publicación en crates.io: 2019; sin commits relevantes desde entonces.",
        "Depende de ndarray 0.12 / rand 0.4 — varias generaciones por debajo del ecosistema actual.",
        "Sin SIMD ni paralelismo: simulación statevector single-threaded.",
        "Upstream declara crate-type=[dylib]; vendoreado en rust/q1tsim/vendor/ con rlib para",
        "evitar dependencia en libstd.so y libq1tsim.so en el contenedor de runtime.",
    ],
)

_QUANTR_WARNING: tuple[str, str, list[str]] = (
    "quantr",
    "límite práctico ~16 qubits, CI solo en ubuntu",
    [
        "Statevector denso (Vec<Complex<f64>>) sin tile/blocking; el coste de memoria explota >16 qubits.",
        "CI oficial corre únicamente en ubuntu-latest; macOS/Windows no se verifican upstream.",
        "API muy joven (0.6.x) — cambios menores entre minor versions.",
    ],
)

_QCGPU_WARNING: tuple[str, str, list[str]] = (
    "qcgpu",
    "requiere OpenCL; deprecated en macOS desde 10.14; abandonado desde 2018",
    [
        "OpenCL marcado deprecated en macOS desde 10.14 (Mojave); Apple sugiere Metal.",
        "El crate qcgpu 0.1 (2018) no recibe mantenimiento desde entonces.",
        "ocl-core 0.9 contiene código rechazado por rustc moderno; usamos un fork vendored",
        "  (rust/qcgpu/vendor/ocl-core) solo para compilar. Runtime sigue requiriendo OpenCL ICD.",
        "En sistemas sin GPU OpenCL apta, el binario reporta el error en su JSON y no aborta el run.",
    ],
)

_QUANTRS2_WARNING: tuple[str, str, list[str]] = (
    "quantrs2",
    "API inestable (RC 0.1.3); QFT/MCX implementados manualmente",
    [
        "Versión actual es un release-candidate (0.1.3); la API puede cambiar antes de 0.2.",
        "QFT y MCX no vienen como puertas nativas — los benchmarks los expanden a mano.",
        "Pure-Rust (sin FFI C) → builds fiables, pero sin SIMD/GPU activado el rendimiento",
        "  queda por debajo de simuladores con backends optimizados.",
    ],
)


# Standard warning bundles for the Rust frameworks. Reused across platforms.
_RUST_WARNINGS_ALL = [
    _Q1TSIM_WARNING,
    _QUANTR_WARNING,
    _QUANTRS2_WARNING,
    _QCGPU_WARNING,
]
_RUST_WARNINGS_NO_QCGPU = [
    _Q1TSIM_WARNING,
    _QUANTR_WARNING,
    _QUANTRS2_WARNING,
]


PLATFORM_CONFIGS: dict[str, PlatformConfig] = {
    "macos-arm64": PlatformConfig(
        # qip omitted: pendiente de añadir un binario rustqip dedicado al benchmark suite.
        frameworks=[
            "qiskit",
            "cirq",
            "cudaq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=True,
        warnings=[
            _cudaq_cpu_warning("macos", "arm64"),
            *_RUST_WARNINGS_NO_QCGPU,
        ],
    ),
    "macos-x86_64": PlatformConfig(
        frameworks=[
            "qiskit",
            "cirq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=True,
        warnings=[
            *_RUST_WARNINGS_NO_QCGPU,
        ],
    ),
    "linux-x86_64-nvidia": PlatformConfig(
        frameworks=[
            "qiskit",
            "cirq",
            "cudaq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
            "qcgpu",
        ],
        cudaq_target="nvidia",
        quantrs2_gpu=True,
        warnings=[
            *_RUST_WARNINGS_ALL,
        ],
    ),
    "linux-x86_64-cpu": PlatformConfig(
        frameworks=[
            "qiskit",
            "cirq",
            "cudaq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=False,
        warnings=[
            _cudaq_cpu_warning("linux", "x86_64"),
            *_RUST_WARNINGS_ALL,
        ],
    ),
    "linux-aarch64-nvidia": PlatformConfig(
        frameworks=[
            "qiskit",
            "cirq",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="nvidia",
        quantrs2_gpu=True,
        warnings=[
            *_RUST_WARNINGS_NO_QCGPU,
        ],
    ),
    "linux-aarch64-cpu": PlatformConfig(
        frameworks=[
            "qiskit",
            "cirq",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=False,
        warnings=[
            *_RUST_WARNINGS_NO_QCGPU,
        ],
    ),
    "windows-x86_64-gpu": PlatformConfig(
        frameworks=[
            "qiskit",
            "cirq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
            "qcgpu",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=True,
        warnings=[
            *_RUST_WARNINGS_ALL,
        ],
    ),
    "windows-x86_64-cpu": PlatformConfig(
        frameworks=[
            "qiskit",
            "cirq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=False,
        warnings=[
            *_RUST_WARNINGS_ALL,
        ],
    ),
    "windows-arm64-gpu": PlatformConfig(
        frameworks=[
            "cirq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=True,
        warnings=[
            *_RUST_WARNINGS_NO_QCGPU,
        ],
    ),
    "windows-arm64-cpu": PlatformConfig(
        frameworks=[
            "cirq",
            "qdislib",
            "q1tsim",
            "quantr",
            "quantrs2",
        ],
        cudaq_target="qpp-cpu",
        quantrs2_gpu=False,
        warnings=[
            *_RUST_WARNINGS_NO_QCGPU,
        ],
    ),
}



def print_config_warnings(config: PlatformConfig) -> None:
    if not config.warnings:
        return
    print("Advertencias de compatibilidad:")
    for fw, headline, details in config.warnings:
        print(f"  [WARN] {fw}: {headline}")
        for line in details:
            print(f"         {line}")


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------


def ask_contributor_name() -> str:
    while True:
        name = input("Nombre del contribuyente: ").strip()
        if name:
            return name
        print("  El nombre no puede estar vacío.")


# ---------------------------------------------------------------------------
# Hardware summary
# ---------------------------------------------------------------------------


def print_hardware_summary(hw: HardwareInfo) -> None:
    print()
    print("Hardware detectado:")
    print(f"  Hostname    : {hw.hostname}")
    print(f"  OS          : {hw.os} ({hw.os_version})")
    print(f"  CPU         : {hw.cpu_model}")
    print(
        f"  Cores       : {hw.cpu_cores_physical} físicos / "
        f"{hw.cpu_cores_logical} lógicos @ {hw.cpu_freq_mhz:.0f} MHz"
    )
    print(f"  RAM         : {hw.ram_total_gb:.1f} GB")
    if hw.gpu_model:
        print(f"  GPU         : {hw.gpu_model} ({hw.gpu_vram_gb:.1f} GB VRAM)")
    else:
        print("  GPU         : (no NVIDIA GPU detected)")
    print(f"  Python      : {hw.python_version}")
    print()


# ---------------------------------------------------------------------------
# Per-framework Grover benchmark
# ---------------------------------------------------------------------------


def _error_result(
    framework: str,
    algo: str,
    n: int,
    hw: HardwareInfo,
    contributor_name: str,
    error_msg: str,
) -> dict:
    """Return a minimal error result dict compatible with the result schema."""
    return {
        "status": "error",
        "framework": framework,
        "algorithm": algo,
        "n_qubits": n,
        "error": error_msg,
        "contributor_name": contributor_name,
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
        "wall_time_median_ms": 0.0,
        "peak_memory_rss_mb": 0.0,
        "cv": 0.0,
        "jsd": 0.0,
        "startup_time_ms": 0.0,
        "build_time_ms": 0.0,
        "simulation_time_ms": 0.0,
        "raw_times_ms": [],
        "num_shots": 0,
        "n_repetitions": 0,
        "framework_version": "",
        "runtime_version": "",
        "scaling_data": {},
        "scaling_alpha": 0.0,
        "scaling_beta": 0.0,
        "subprocess_wall_time_ms": 0.0,
    }


def _run_python_worker(
    framework: str,
    algo: str,
    n: int,
    config: BenchmarkConfig,
    contributor_name: str,
    hw: HardwareInfo,
    cudaq_target: str = "qpp-cpu",
    timeout_s: float = 600.0,
) -> dict:
    """Launch a framework worker subprocess, return its result dict.

    The worker module reads its JSON config from stdin and writes a single JSON
    result line to stdout. Stderr is inherited so the worker's progress prints
    flow through to the terminal. Any failure (timeout, non-zero exit, parse
    error, error status from the worker) is converted into an `_error_result`
    so the orchestrator loop can continue with the next framework.
    """
    worker_module = f"python.workers.{framework}_worker"
    payload = json.dumps({
        "algo": algo,
        "n": n,
        "n_repetitions": config.n_repetitions,
        "num_shots": config.num_shots,
        "contributor": contributor_name,
        "cudaq_target": cudaq_target,
    })
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", worker_module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    proc.stdin.write(payload)
    proc.stdin.close()

    cpu_samples: list[float] = []
    _cpu_stop = threading.Event()
    _cpu_thread: threading.Thread | None = None
    try:
        _psutil_proc = psutil.Process(proc.pid)
        _psutil_proc.cpu_percent()
        def _sample_cpu() -> None:
            while not _cpu_stop.is_set():
                try:
                    cpu_samples.append(_psutil_proc.cpu_percent())
                except psutil.NoSuchProcess:
                    break
                _cpu_stop.wait(0.1)
        _cpu_thread = threading.Thread(target=_sample_cpu, daemon=True)
        _cpu_thread.start()
    except psutil.NoSuchProcess:
        pass

    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            lines.append(line)
            try:
                json.loads(line)
            except json.JSONDecodeError:
                print(line)

    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return _error_result(framework, algo, n, hw, contributor_name, "timeout")

    subprocess_wall_ms = (time.perf_counter() - t_start) * 1000.0

    if not lines:
        return _error_result(framework, algo, n, hw, contributor_name,
                             f"no output (exit {proc.returncode})")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return _error_result(framework, algo, n, hw, contributor_name,
                             f"no JSON result (exit {proc.returncode})")
    if result.get("status") == "error":
        err = result.get("error", "unknown")
        print(f"  [worker error] {framework}: {err}", file=sys.stderr)
        return _error_result(framework, algo, n, hw, contributor_name, err)
    if proc.returncode != 0:
        if proc.returncode < 0:
            try:
                import signal as _signal
                sig_name = _signal.Signals(-proc.returncode).name
                exit_msg = f"killed by {sig_name} (exit {proc.returncode})"
            except (ValueError, AttributeError):
                exit_msg = f"exit {proc.returncode}"
        else:
            exit_msg = f"exit {proc.returncode}"
        return _error_result(framework, algo, n, hw, contributor_name, exit_msg)
    _cpu_stop.set()
    if _cpu_thread is not None:
        _cpu_thread.join(timeout=1.0)
    result["subprocess_wall_time_ms"] = round(subprocess_wall_ms, 3)
    result["cpu_percent_mean"] = round(float(np.mean(cpu_samples)), 2) if cpu_samples else 0.0
    return result



def _run_rust_binary(
    binary: pathlib.Path,
    n: int,
    target: int,
    num_shots: int,
    timeout_s: float = 300.0,
) -> dict:
    """Invoke a Rust Grover binary and return its parsed JSON output.

    Raises subprocess.TimeoutExpired, FileNotFoundError or ValueError on
    failure (caller is expected to catch and convert to a SKIP/ERROR row).
    """
    _t_start = time.perf_counter()
    proc = subprocess.Popen(
        [str(binary), "--n", str(n), "--target", str(target), "--shots", str(num_shots)],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    cpu_samples: list[float] = []
    _stop = threading.Event()
    _cpu_thread: threading.Thread | None = None
    try:
        _psutil_proc = psutil.Process(proc.pid)
        _psutil_proc.cpu_percent()
        def _sample() -> None:
            while not _stop.is_set():
                try:
                    cpu_samples.append(_psutil_proc.cpu_percent())
                except psutil.NoSuchProcess:
                    break
                _stop.wait(0.05)
        _cpu_thread = threading.Thread(target=_sample, daemon=True)
        _cpu_thread.start()
    except psutil.NoSuchProcess:
        pass
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            lines.append(line)
            # Print progress lines; the last line is JSON and will look like garbage — skip it
            try:
                json.loads(line)
            except json.JSONDecodeError:
                print(line)
    _stop.set()
    if _cpu_thread is not None:
        _cpu_thread.join(timeout=1.0)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} exited with code {proc.returncode}: (see stderr above)")
    if not lines:
        raise ValueError(f"{binary.name} produced no stdout")
    payload = json.loads(lines[-1])
    payload["cpu_percent_mean"] = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    payload["subprocess_wall_time_ms"] = (time.perf_counter() - _t_start) * 1000.0
    return payload


def benchmark_rust_grover(
    framework_name: str,
    binary: pathlib.Path,
    config: BenchmarkConfig,
    hw: HardwareInfo,
    contributor_name: str,
) -> dict:
    """Run the full Grover sweep against a Rust binary.

    Mirrors the structure of `benchmark_grover()` so the resulting dict can be
    appended to the same results list and rendered by `print_summary_table`.
    Uses the framework-reported `time_ms` (not subprocess wall time) for all
    timing fields.
    """
    n_main = 5
    target_main = 5
    times_ms: list[float] = []
    subprocess_wall_times_ms: list[float] = []
    cpu_percents: list[float] = []
    last_payload: dict | None = None

    # Warmup + repetitions (matches benchmark_run semantics).
    for _ in range(max(0, config.warmup_runs)):
        _run_rust_binary(binary, n_main, target_main, config.num_shots)
    for _ in range(config.n_repetitions):
        payload = _run_rust_binary(binary, n_main, target_main, config.num_shots)
        times_ms.append(float(payload.get("time_ms", 0.0)))
        subprocess_wall_times_ms.append(float(payload.get("subprocess_wall_time_ms", 0.0)))
        cpu_percents.append(float(payload.get("cpu_percent_mean", 0.0)))
        last_payload = payload

    arr = np.array(times_ms) if times_ms else np.array([0.0])
    median_ms = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_ms = float(q75 - q25)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0

    # JSD vs. theoretical δ on |target⟩.
    jsd = 0.0
    if last_payload is not None and "distribution" in last_payload:
        dist = last_payload["distribution"] or {}
        total = sum(dist.values())
        empirical = {k: v / total for k, v in dist.items()} if total > 0 else {}
        theoretical = {format(target_main, f"0{n_main}b"): 1.0}
        try:
            jsd = compute_jsd(empirical, theoretical)
        except Exception as e:
            print(f"  [WARN] Could not compute JSD for {framework_name}: {e}")

    # Scaling sweep — 3 reps per n to mirror benchmark_grover.
    scaling_data: dict[int, float] = {}
    for n in config.n_values:
        try:
            sub_times: list[float] = []
            for _ in range(3):
                payload = _run_rust_binary(binary, n, n, config.num_shots)
                sub_times.append(float(payload.get("time_ms", 0.0)))
            scaling_data[n] = float(np.median(sub_times)) if sub_times else 0.0
        except Exception as e:
            print(f"  [WARN] scaling n={n} failed for {framework_name}: {e}")

    if len(scaling_data) >= 2:
        try:
            alpha, beta = fit_scaling_curve(scaling_data)
        except Exception as e:
            print(f"  [WARN] curve fit failed for {framework_name}: {e}")
            alpha, beta = 0.0, 0.0
    else:
        alpha, beta = 0.0, 0.0

    enriched = {
        "contributor_name": contributor_name,
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
        "runtime_version": "rust (cargo --release)",
        "num_shots": config.num_shots,
        "n_repetitions": config.n_repetitions,
        "framework_version": last_payload.get("framework_version", "rust-binary") if last_payload else "rust-binary",
        "wall_time_median_ms": median_ms,
        "wall_time_iqr_ms": iqr_ms,
        "peak_memory_rss_mb": float(last_payload.get("mem_mb", 0.0)) if last_payload else 0.0,
        "cv": cv,
        "startup_time_ms": 0.0,
        "build_time_ms": 0.0,
        "simulation_time_ms": median_ms,
        "cpu_percent_mean": float(np.mean(cpu_percents)) if cpu_percents else 0.0,
        "jsd": jsd,
        "scaling_alpha": alpha,
        "scaling_beta": beta,
        "scaling_data": scaling_data,
        "framework": framework_name,
        "algorithm": "grover",
        "n_qubits": n_main,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": None,
        "platform_info": platform.platform(),
        "raw_times_ms": times_ms,
        "wall_time_mean_ms": mean_ms,
        "wall_time_std_ms": std_ms,
        "subprocess_wall_time_ms": float(np.median(subprocess_wall_times_ms)) if subprocess_wall_times_ms else 0.0,
    }
    return enriched





def benchmark_rust_grover_at_n(
    framework_name: str,
    binary: pathlib.Path,
    n: int,
    config: BenchmarkConfig,
    hw: HardwareInfo,
    contributor_name: str,
) -> dict:
    """Run config.n_repetitions of Grover at qubit count n using a Rust binary."""
    target = n
    times_ms: list[float] = []
    subprocess_wall_times_ms: list[float] = []
    last_payload: dict | None = None

    for _ in range(max(0, config.warmup_runs)):
        _run_rust_binary(binary, n, target, config.num_shots)
    for _ in range(config.n_repetitions):
        payload = _run_rust_binary(binary, n, target, config.num_shots)
        times_ms.append(float(payload.get("time_ms", 0.0)))
        subprocess_wall_times_ms.append(float(payload.get("subprocess_wall_time_ms", 0.0)))
        last_payload = payload

    arr = np.array(times_ms) if times_ms else np.array([0.0])
    median_ms = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_ms = float(q75 - q25)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0

    jsd = 0.0
    if last_payload and "distribution" in last_payload:
        dist = last_payload["distribution"] or {}
        total = sum(dist.values())
        empirical = {k: v / total for k, v in dist.items()} if total > 0 else {}
        theoretical = {format(target, f"0{n}b"): 1.0}
        try:
            jsd = compute_jsd(empirical, theoretical)
        except Exception as e:
            print(f"  [WARN] JSD failed for {framework_name} n={n}: {e}")

    return {
        "contributor_name": contributor_name,
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
        "runtime_version": "rust (cargo --release)",
        "num_shots": config.num_shots,
        "n_repetitions": config.n_repetitions,
        "framework_version": last_payload.get("framework_version", "rust-binary") if last_payload else "rust-binary",
        "wall_time_median_ms": median_ms,
        "wall_time_iqr_ms": iqr_ms,
        "peak_memory_rss_mb": float(last_payload.get("mem_mb", 0.0)) if last_payload else 0.0,
        "cv": cv,
        "startup_time_ms": 0.0,
        "build_time_ms": 0.0,
        "simulation_time_ms": median_ms,
        "cpu_percent_mean": 0.0,
        "jsd": jsd,
        "scaling_alpha": 0.0,
        "scaling_beta": 0.0,
        "scaling_data": {},
        "framework": framework_name,
        "algorithm": "grover",
        "n_qubits": n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": None,
        "platform_info": platform.platform(),
        "raw_times_ms": times_ms,
        "wall_time_mean_ms": mean_ms,
        "wall_time_std_ms": std_ms,
        "subprocess_wall_time_ms": float(np.median(subprocess_wall_times_ms)) if subprocess_wall_times_ms else 0.0,
    }


# ---------------------------------------------------------------------------
# Shor benchmarking
# ---------------------------------------------------------------------------


def _run_rust_shor_binary(
    binary: pathlib.Path, N: int, shots: int = 10, tries: int = 3, timeout_s: float = 300.0
) -> dict:
    """Invoke a Rust Shor binary and return its parsed JSON output.

    Raises subprocess.TimeoutExpired, FileNotFoundError or ValueError on
    failure (caller is expected to catch and convert to a SKIP/ERROR row).
    """
    proc = subprocess.Popen(
        [str(binary), "--N", str(N), "--shots", str(shots), "--tries", str(tries)],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    cpu_samples: list[float] = []
    _stop = threading.Event()
    _cpu_thread: threading.Thread | None = None
    try:
        _psutil_proc = psutil.Process(proc.pid)
        _psutil_proc.cpu_percent()
        def _sample() -> None:
            while not _stop.is_set():
                try:
                    cpu_samples.append(_psutil_proc.cpu_percent())
                except psutil.NoSuchProcess:
                    break
                _stop.wait(0.05)
        _cpu_thread = threading.Thread(target=_sample, daemon=True)
        _cpu_thread.start()
    except psutil.NoSuchProcess:
        pass
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            lines.append(line)
            # Print progress lines; the last line is JSON and will look like garbage — skip it
            try:
                json.loads(line)
            except json.JSONDecodeError:
                print(line)
    _stop.set()
    if _cpu_thread is not None:
        _cpu_thread.join(timeout=1.0)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} exited with code {proc.returncode}: (see stderr above)")
    if not lines:
        raise ValueError(f"{binary.name} produced no stdout")
    payload = json.loads(lines[-1])
    payload["cpu_percent_mean"] = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    return payload






def benchmark_rust_shor_at_n(
    framework_name: str,
    binary: pathlib.Path,
    N: int,
    config: BenchmarkConfig,
    hw: HardwareInfo,
    contributor_name: str,
) -> dict:
    n_qubits = _n_qubits_shor(N)
    times_ms: list[float] = []
    subprocess_wall_times_ms: list[float] = []
    factors: list[int] = []
    last_payload: dict | None = None

    cpu_percents_shor: list[float] = []
    for _ in range(config.n_repetitions):
        _t_sub = time.perf_counter()
        payload = _run_rust_shor_binary(binary, N, shots=config.num_shots, tries=3)
        subprocess_wall_times_ms.append((time.perf_counter() - _t_sub) * 1000.0)
        times_ms.append(float(payload.get("time_ms", 0.0)))
        factors.append(int(payload.get("factor", 1)))
        cpu_percents_shor.append(float(payload.get("cpu_percent_mean", 0.0)))
        last_payload = payload

    if not times_ms:
        raise RuntimeError("No se completó ninguna repetición")
    arr = np.array(times_ms) if times_ms else np.array([0.0])
    median_ms = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_ms = float(q75 - q25)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0

    factor_found = max(set(factors), key=factors.count) if factors else 1
    success_rate = 1.0 if factor_found not in (1, N) else 0.0

    return {
        "contributor_name": contributor_name,
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
        "runtime_version": "rust (cargo --release)",
        "num_shots": config.num_shots,
        "n_repetitions": config.n_repetitions,
        "framework_version": last_payload.get("framework_version", "rust-binary") if last_payload else "rust-binary",
        "n_to_factor": N,
        "factor_found": factor_found,
        "success_rate": success_rate,
        "wall_time_median_ms": median_ms,
        "wall_time_iqr_ms": iqr_ms,
        "peak_memory_rss_mb": float(last_payload.get("mem_mb", 0.0)) if last_payload else 0.0,
        "cv": cv,
        "startup_time_ms": 0.0,
        "build_time_ms": 0.0,
        "simulation_time_ms": median_ms,
        "cpu_percent_mean": float(np.mean(cpu_percents_shor)) if cpu_percents_shor else 0.0,
        "jsd": 0.0,
        "scaling_alpha": 0.0,
        "scaling_beta": 0.0,
        "scaling_data": {},
        "framework": framework_name,
        "algorithm": "shor",
        "n_qubits": n_qubits,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": None,
        "platform_info": platform.platform(),
        "raw_times_ms": times_ms,
        "wall_time_mean_ms": mean_ms,
        "wall_time_std_ms": std_ms,
        "subprocess_wall_time_ms": float(np.median(subprocess_wall_times_ms)) if subprocess_wall_times_ms else 0.0,
    }


def print_shor_summary_table(results: list[dict], statuses: dict[str, str]) -> None:
    print()
    print(
        "╔══════════════╦═══════════╦══════╦══════════╦═══════════╦══════════════════╦══════════╗"
    )
    print(
        "║ Framework    ║ Status    ║ N    ║ Factor   ║ Success%  ║ Median time (ms) ║ CV       ║"
    )
    print(
        "╠══════════════╬═══════════╬══════╬══════════╬═══════════╬══════════════════╬══════════╣"
    )
    ordered_names: list[str] = list(FRAMEWORKS) + [
        n for n in RUST_FRAMEWORKS_SHOR if n not in FRAMEWORKS
    ]
    by_pair: dict[tuple[str, int], dict] = {
        (r["framework"], r.get("n_to_factor", 0)): r for r in results
    }
    n_values_seen = sorted({r.get("n_to_factor", 0) for r in results})
    for fw_name in ordered_names:
        status = statuses.get(fw_name, "SKIP")
        printed_any = False
        for N_val in n_values_seen:
            key = (fw_name, N_val)
            if key in by_pair:
                r = by_pair[key]
                N = f"{r.get('n_to_factor', '—')}"
                fac = f"{r.get('factor_found', '—')}"
                succ = f"{r.get('success_rate', 0.0) * 100:.0f}%"
                median = f"{r['wall_time_median_ms']:.1f}"
                cv = f"{r['cv']:.3f}"
                print(
                    f"║ {fw_name:<12} ║ {status:<9} ║ {N:<4} ║ {fac:<8} ║ {succ:<9} ║ {median:<16} ║ {cv:<8} ║"
                )
                printed_any = True
        if not printed_any:
            print(
                f"║ {fw_name:<12} ║ {status:<9} ║ {'—':<4} ║ {'—':<8} ║ {'—':<9} ║ {'—':<16} ║ {'—':<8} ║"
            )
    print(
        "╚══════════════╩═══════════╩══════╩══════════╩═══════════╩══════════════════╩══════════╝"
    )


# ---------------------------------------------------------------------------
# Saving and reporting
# ---------------------------------------------------------------------------


def _build_output_doc(
    contributor_name: str,
    hw: HardwareInfo,
    config: BenchmarkConfig,
    results: list[dict],
    platform_id: str = "",
    emulated: bool = False,
    no_gpu: bool = False,
) -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_id": platform_id,
        "gpu_enabled": platform_id.endswith("-nvidia") and not no_gpu,
        "emulated": emulated,
        "no_gpu": no_gpu,
        "benchmark_image": os.getenv("DOCKER_IMAGE", "dev"),
        "contributor_name": contributor_name,
        "hardware": dataclasses.asdict(hw),
        "config": {
            "n_repetitions": config.n_repetitions,
            "n_values": list(config.n_values),
            "n_values_shor": list(config.n_values_shor),
            "num_shots": config.num_shots,
        },
        "results": results,
    }


def to_db_rows(doc: dict) -> list[dict]:
    """Convert output doc to flat rows for DB upload (drops raw_times_ms, hostname, etc.)."""
    run_meta = {
        "contributor": doc.get("contributor_name"),
        "platform_id": doc.get("platform_id"),
        "gpu_enabled": doc.get("gpu_enabled", False),
        "benchmark_image": doc.get("benchmark_image", "dev"),
        "cpu_model": doc.get("hardware", {}).get("cpu_model"),
        "cpu_physical_cores": doc.get("hardware", {}).get("cpu_cores_physical"),
        "cpu_logical_cores": doc.get("hardware", {}).get("cpu_cores_logical"),
        "cpu_freq_mhz": doc.get("hardware", {}).get("cpu_freq_mhz"),
        "ram_total_gb": doc.get("hardware", {}).get("ram_total_gb"),
        "gpu_model": doc.get("hardware", {}).get("gpu_model"),
        "gpu_vram_gb": doc.get("hardware", {}).get("gpu_vram_gb"),
    }
    rows = []
    for r in doc.get("results", []):
        row = {**run_meta}
        for field in [
            "framework", "framework_version", "algorithm", "n_qubits",
            "wall_time_median_ms", "wall_time_iqr_ms", "build_time_ms",
            "simulation_time_ms", "startup_time_ms", "peak_memory_rss_mb",
            "cpu_percent_mean", "jsd", "cv", "scaling_alpha", "scaling_beta",
            "scaling_data", "timestamp",
        ]:
            row[field] = r.get(field)
        rows.append(row)
    return rows


def _expand_result_to_rows(result: dict, run_meta: dict) -> list[dict]:
    """Expand a result dict into one Supabase row per repetition.

    Error results → single row (repetition_index=0, wall_time_ms=None).
    Skip results  → no rows.
    """
    status = result.get("status", "ok")
    if status == "skip":
        return []
    base = {
        **run_meta,
        "algorithm":                    result.get("algorithm"),
        "framework":                    result.get("framework"),
        "framework_version":            result.get("framework_version"),
        "n_qubits":                     result.get("n_qubits"),
        "num_shots":                    result.get("num_shots"),
        "n_repetitions":                result.get("n_repetitions"),
        "runtime_version":              result.get("runtime_version"),
        "python_version":               result.get("python_version"),
        "timestamp":                    result.get("timestamp"),
        "wall_time_median_ms":          result.get("wall_time_median_ms"),
        "wall_time_iqr_ms":             result.get("wall_time_iqr_ms"),
        "wall_time_mean_ms":            result.get("wall_time_mean_ms"),
        "wall_time_std_ms":             result.get("wall_time_std_ms"),
        "build_time_ms":                result.get("build_time_ms"),
        "simulation_time_ms":           result.get("simulation_time_ms"),
        "startup_time_ms":              result.get("startup_time_ms"),
        "subprocess_wall_time_ms":      result.get("subprocess_wall_time_ms"),
        "peak_memory_rss_mb":           result.get("peak_memory_rss_mb"),
        "cpu_percent_mean":             result.get("cpu_percent_mean"),
        "jsd":                          result.get("jsd"),
        "cv":                           result.get("cv"),
        "status":                       status,
        "error":                        result.get("error"),
        "scaling_alpha":                None,
        "scaling_beta":                 None,
        "scaling_data":                 None,
        "n_to_factor":                  result.get("n_to_factor"),
        "factor_found":                 result.get("factor_found"),
        "success_rate":                 result.get("success_rate"),
        "cutting_wall_time_ms":         result.get("cutting_wall_time_ms"),
        "cutting_find_time_ms":         result.get("cutting_find_time_ms"),
        "cutting_expectation_value":    result.get("cutting_expectation_value"),
    }
    if status == "error":
        return [{**base, "repetition_index": 0, "wall_time_ms": None}]
    raw_times = result.get("raw_times_ms") or []
    if not raw_times:
        return [{**base, "repetition_index": 0, "wall_time_ms": result.get("wall_time_median_ms")}]
    return [{**base, "repetition_index": i, "wall_time_ms": t} for i, t in enumerate(raw_times)]


def _supabase_insert(rows: list[dict], url: str, key: str) -> bool:
    if not rows:
        return True
    endpoint = f"{url}/rest/v1/benchmark_runs"
    payload = json.dumps(rows, default=str).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "return=minimal",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"  → Supabase: {len(rows)} filas insertadas ({resp.status})")
            return True
    except Exception as exc:
        print(f"  ⚠ Supabase insert falló: {exc}")
        return False


def _supabase_patch_scaling(
    run_id: str, algorithm: str, framework: str,
    alpha: float, beta: float, scaling_data: dict,
    url: str, key: str,
) -> None:
    endpoint = (
        f"{url}/rest/v1/benchmark_runs"
        f"?run_id=eq.{urllib.parse.quote(run_id)}"
        f"&algorithm=eq.{urllib.parse.quote(algorithm)}"
        f"&framework=eq.{urllib.parse.quote(framework)}"
    )
    payload = json.dumps(
        {"scaling_alpha": alpha, "scaling_beta": beta, "scaling_data": scaling_data},
        default=str,
    ).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "return=minimal",
        },
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception as exc:
        print(f"  ⚠ Supabase patch scaling falló ({framework}): {exc}")


def _save_json(path: str, doc: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)


def print_summary_table(results: list[dict], statuses: dict[str, str]) -> None:
    print()
    print(
        "╔══════════════╦═══════════╦══════════════════╦═══════════╦══════════╦═══════════╗"
    )
    print(
        "║ Framework    ║ Status    ║ Median time (ms) ║ Mem (MB)  ║ CV       ║ JSD       ║"
    )
    print(
        "╠══════════════╬═══════════╬══════════════════╬═══════════╬══════════╬═══════════╣"
    )
    by_name = {r["framework"]: r for r in results}
    # Python frameworks first, then Rust binaries — keeps the table readable.
    ordered_names: list[str] = list(FRAMEWORKS) + [
        n for n in RUST_FRAMEWORKS if n not in FRAMEWORKS
    ]
    for fw_name in ordered_names:
        status = statuses.get(fw_name, "SKIP")
        if fw_name in by_name and status == "OK":
            r = by_name[fw_name]
            median = f"{r['wall_time_median_ms']:.1f}"
            mem = f"{r['peak_memory_rss_mb']:.1f}"
            cv = f"{r['cv']:.3f}"
            jsd = f"{r['jsd']:.3f}"
        else:
            median = "—"
            mem = "—"
            cv = "—"
            jsd = "—"
        print(
            f"║ {fw_name:<12} ║ {status:<9} ║ {median:<16} ║ {mem:<9} ║ {cv:<8} ║ {jsd:<9} ║"
        )
    print(
        "╚══════════════╩═══════════╩══════════════════╩═══════════╩══════════╩═══════════╝"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Quantum framework benchmark runner")
    p.add_argument(
        "--platform",
        required=True,
        help="Platform ID (e.g. linux-x86_64-cpu). Set automatically by Docker entrypoint.",
    )
    p.add_argument(
        "--contributor",
        default=None,
        help="Contributor name. Skips interactive prompt.",
    )
    p.add_argument(
        "--emulated",
        action="store_true",
        default=False,
        help="Mark results as emulated (arm64 host running amd64 image)",
    )
    p.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="CPU-only pass (GPU disabled at docker level)",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Dev mode: 1 rep, smallest n only, fast exit",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(BANNER)

    if args.contributor is not None:
        contributor_name = args.contributor
    else:
        contributor_name = ask_contributor_name()

    print()
    print("AVISO: El benchmark completo puede tardar HORAS si se incluyen todos los")
    print("algoritmos y frameworks. Para Grover solo, estima entre 5 y 30 minutos.")
    print("Se agradece dar el máximo tiempo posible para obtener datos más completos.")
    print()

    config = BenchmarkConfig(n_repetitions=10, n_values=[3, 5, 7, 9, 11], num_shots=1024)

    if args.dev:
        config = BenchmarkConfig(
            n_repetitions=1,
            n_values=[config.n_values[0]],
            n_values_shor=[config.n_values_shor[0]],
            num_shots=10,
        )
        print("[DEV] Modo desarrollo: 1 repetición, n mínimo, 10 shots")

    hw = detect_hardware()
    print_hardware_summary(hw)

    if args.platform not in PLATFORM_CONFIGS:
        print(f"[ERROR] Unknown platform: {args.platform!r}")
        print(f"  Valid platforms: {', '.join(PLATFORM_CONFIGS)}")
        return
    platform_cfg = PLATFORM_CONFIGS[args.platform]
    all_enabled = list(platform_cfg.frameworks)
    cudaq_target = platform_cfg.cudaq_target

    # When --no-gpu is set, qcgpu requires OpenCL GPU and cannot run.
    if args.no_gpu:
        _gpu_only = {"qcgpu"}
        skipped = [fw for fw in all_enabled if fw in _gpu_only]
        all_enabled = [fw for fw in all_enabled if fw not in _gpu_only]
        if skipped:
            print(f"[no-gpu] Omitiendo frameworks sin fallback CPU: {', '.join(skipped)}")

    # When --emulated is set (amd64 image under QEMU on ARM), cudaq uses AVX/SSE
    # instructions that QEMU does not emulate and will SIGILL.
    if args.emulated:
        _qemu_broken = {"cudaq"}
        skipped_e = [fw for fw in all_enabled if fw in _qemu_broken]
        all_enabled = [fw for fw in all_enabled if fw not in _qemu_broken]
        if skipped_e:
            print(f"[emulated] Omitiendo frameworks incompatibles con emulacion QEMU: {', '.join(skipped_e)}")

    print(f"Plataforma seleccionada: {args.platform}")
    print(f"  Frameworks habilitados: {', '.join(all_enabled)}")
    print()
    print_config_warnings(platform_cfg)
    # Split into Python vs Rust based on registries.
    python_enabled = [n for n in all_enabled if n in FRAMEWORKS]
    rust_enabled = [n for n in all_enabled if n in RUST_FRAMEWORKS]

    # Filter Rust frameworks to only those whose binary actually exists (also
    # in --platform mode, in case the user didn't build them).
    rust_enabled = [
        n
        for n in rust_enabled
        if RUST_FRAMEWORKS[n].exists() and os.access(RUST_FRAMEWORKS[n], os.X_OK)
    ]

    enabled = list(python_enabled) + list(rust_enabled)
    if not enabled:
        print("\nNingún framework cuántico disponible. Abortando.")
        return

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    _supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    _supabase_key = os.getenv("SUPABASE_KEY", "")
    USE_SUPABASE = not args.dev and bool(_supabase_url and _supabase_key)
    _supabase_run_meta: dict = {
        "run_id": run_id,
        "contributor": contributor_name,
        "platform_id": args.platform,
        "benchmark_image": DOCKER_IMAGE,
        "gpu_enabled": not args.no_gpu and bool(hw.gpu_model),
        "emulated": args.emulated,
        "cpu_model": hw.cpu_model,
        "cpu_physical_cores": hw.cpu_cores_physical,
        "cpu_logical_cores": hw.cpu_cores_logical,
        "cpu_freq_mhz": hw.cpu_freq_mhz,
        "ram_total_gb": hw.ram_total_gb,
        "gpu_model": hw.gpu_model,
        "gpu_vram_gb": hw.gpu_vram_gb,
        "os": hw.os,
        "os_version": hw.os_version,
    }
    final_path = os.path.join("results", f"grover_{timestamp}.json")
    partial_path = os.path.join("results", f"grover_{timestamp}_partial.json")
    shor_timestamp = timestamp
    shor_final_path = os.path.join("results", f"shor_{shor_timestamp}.json")
    shor_partial_path = os.path.join("results", f"shor_{shor_timestamp}_partial.json")

    # ---- Grover state ----
    results: list[dict] = []
    statuses: dict[str, str] = {name: "SKIP" for name in FRAMEWORKS}
    for n in RUST_FRAMEWORKS:
        statuses.setdefault(n, "SKIP")
    scaling_by_fw: dict[str, dict[int, float]] = {}

    # ---- Shor state ----
    shor_results: list[dict] = []
    shor_statuses: dict[str, str] = {}
    shor_python_enabled = [n for n in python_enabled if n in FRAMEWORKS]
    shor_rust_enabled = [
        n for n in rust_enabled
        if n in RUST_FRAMEWORKS_SHOR
        and RUST_FRAMEWORKS_SHOR[n].exists()
        and os.access(RUST_FRAMEWORKS_SHOR[n], os.X_OK)
    ]
    shor_scaling_by_fw: dict[str, dict[int, float]] = {}

    grover_total = len(config.n_values) * (len(python_enabled) + len(rust_enabled))
    shor_total = len(config.n_values_shor) * (len(shor_python_enabled) + len(shor_rust_enabled))
    idx = 0
    shor_idx = 0

    print(
        f"Total Grover: {grover_total} operaciones "
        f"({len(python_enabled) + len(rust_enabled)} frameworks × {len(config.n_values)} valores de n)"
    )
    print(
        f"Total Shor:   {shor_total} operaciones "
        f"({len(shor_python_enabled) + len(shor_rust_enabled)} frameworks × {len(config.n_values_shor)} valores de N)"
    )

    # Intercalate Grover and Shor by qubit size: for each i, run Grover at n_i
    # then Shor at N_i. If lists differ in length, the shorter one stops contributing.
    n_grover_list = list(config.n_values)
    n_shor_list = list(config.n_values_shor)
    max_steps = max(len(n_grover_list), len(n_shor_list))

    for i in range(max_steps):
        n = n_grover_list[i] if i < len(n_grover_list) else None
        N_shor = n_shor_list[i] if i < len(n_shor_list) else None

        # --- Grover at n qubits ---
        if n is not None:
            print()
            print(f"{'─'*58}")
            print(
                f"  Grover — {n} qubits  ({2**n} estados)  "
                f"[{i+1}/{len(n_grover_list)}]"
            )
            print(f"{'─'*58}")
            n_series_results: list[dict] = []

            for fw_name in python_enabled:
                idx += 1
                print()
                print(f"[{idx}/{grover_total}] {fw_name} (python)  n={n} ...")
                try:
                    result = _run_python_worker(
                        fw_name, "grover", n, config, contributor_name, hw,
                        cudaq_target=cudaq_target,
                    )
                    results.append(result)
                    n_series_results.append(result)
                    if result.get("status") == "error":
                        statuses[fw_name] = "ERROR"
                        print(f"[ERROR] {fw_name} grover n={n}: {result.get('error', 'unknown')}")
                    else:
                        statuses[fw_name] = "OK"
                        scaling_by_fw.setdefault(fw_name, {})[n] = result["wall_time_median_ms"]
                except Exception as e:
                    statuses[fw_name] = "ERROR"
                    print(f"[ERROR] {fw_name} grover n={n}: {e}")

            for fw_name in rust_enabled:
                idx += 1
                binary = RUST_FRAMEWORKS[fw_name]
                print()
                print(f"[{idx}/{grover_total}] {fw_name} (rust: {binary.name})  n={n} ...")
                try:
                    result = benchmark_rust_grover_at_n(
                        fw_name, binary, n, config, hw, contributor_name,
                    )
                    results.append(result)
                    n_series_results.append(result)
                    statuses[fw_name] = "OK"
                    scaling_by_fw.setdefault(fw_name, {})[n] = result["wall_time_median_ms"]
                except OSError as e:
                    if e.errno in (8, 2):  # ENOEXEC, ENOENT — binary incompatible with this arch
                        statuses[fw_name] = "SKIP"
                        print(f"  [SKIP] {fw_name}: binario incompatible con esta arquitectura (errno {e.errno})")
                    else:
                        statuses[fw_name] = "ERROR"
                        print(f"[ERROR] {fw_name} grover n={n}: {e}")
                except FileNotFoundError as e:
                    statuses[fw_name] = "SKIP"
                    print(f"  [SKIP] {fw_name}: binario no encontrado ({e})")
                except subprocess.TimeoutExpired as e:
                    statuses[fw_name] = "ERROR"
                    print(f"[ERROR] {fw_name} grover n={n}: timed out after {e.timeout}s")
                except (json.JSONDecodeError, RuntimeError, ValueError) as e:
                    statuses[fw_name] = "ERROR"
                    print(f"[ERROR] {fw_name} grover n={n}: {e}")
                except Exception as e:
                    statuses[fw_name] = "ERROR"
                    print(f"[ERROR] {fw_name} grover n={n}: {e}")

            if USE_SUPABASE:
                _rows = []
                for r in n_series_results:
                    _rows.extend(_expand_result_to_rows(r, _supabase_run_meta))
                _supabase_insert(_rows, _supabase_url, _supabase_key)
            else:
                checkpoint_path = os.path.join("results", f"grover_{timestamp}_n{n}.json")
                _save_json(checkpoint_path, {
                    "schema_version": "1.0",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "emulated": args.emulated,
                    "n_qubits": n,
                    "n_repetitions": config.n_repetitions,
                    "num_shots": config.num_shots,
                    "results": n_series_results,
                })
                print(f"\n→ Checkpoint grover n={n}: {checkpoint_path}")
                partial_doc = _build_output_doc(
                    contributor_name, hw, config, results,
                    platform_id=args.platform, emulated=args.emulated, no_gpu=args.no_gpu,
                )
                _save_json(partial_path, partial_doc)

        # --- Shor at N_shor ---
        if N_shor is not None:
            n_qubits_val = _n_qubits_shor(N_shor)
            print(f"\n{'─' * 58}")
            print(f"  Shor — N={N_shor} ({n_qubits_val} qubits)  "
                  f"[{i + 1}/{len(n_shor_list)}]")
            print(f"{'─' * 58}")
            shor_n_series: list[dict] = []

            for fw in shor_python_enabled:
                shor_idx += 1
                print(f"\n[{shor_idx}/{shor_total}] {fw} (python)  N={N_shor} ...")
                try:
                    r = _run_python_worker(
                        fw, "shor", N_shor, config, contributor_name, hw,
                        cudaq_target=cudaq_target,
                    )
                    shor_results.append(r)
                    shor_n_series.append(r)
                    if r.get("status") == "error":
                        shor_statuses[fw] = "ERROR"
                        print(f"[ERROR] {fw} shor N={N_shor}: {r.get('error', 'unknown')}")
                    else:
                        shor_statuses[fw] = "OK"
                        shor_scaling_by_fw.setdefault(fw, {})[n_qubits_val] = r["wall_time_median_ms"]
                except Exception as e:
                    shor_statuses.setdefault(fw, "ERROR")
                    print(f"[ERROR] {fw} shor N={N_shor}: {e}")

            for fw in shor_rust_enabled:
                shor_idx += 1
                binary = RUST_FRAMEWORKS_SHOR[fw]
                print(f"\n[{shor_idx}/{shor_total}] {fw} (rust)  N={N_shor} ...")
                try:
                    r = benchmark_rust_shor_at_n(fw, binary, N_shor, config, hw, contributor_name)
                    shor_results.append(r)
                    shor_n_series.append(r)
                    shor_statuses[fw] = "OK"
                    shor_scaling_by_fw.setdefault(fw, {})[n_qubits_val] = r["wall_time_median_ms"]
                except OSError as e:
                    if e.errno in (8, 2):  # ENOEXEC, ENOENT — binary incompatible with this arch
                        shor_statuses[fw] = "SKIP"
                        print(f"  [SKIP] {fw}: binario incompatible con esta arquitectura (errno {e.errno})")
                    else:
                        shor_statuses.setdefault(fw, "ERROR")
                        print(f"[ERROR] {fw} shor N={N_shor}: {e}")
                except FileNotFoundError as e:
                    shor_statuses[fw] = "SKIP"
                    print(f"  [SKIP] {fw}: binario no encontrado ({e})")
                except Exception as e:
                    shor_statuses.setdefault(fw, "ERROR")
                    print(f"[ERROR] {fw} shor N={N_shor}: {e}")

            if USE_SUPABASE:
                _rows = []
                for r in shor_n_series:
                    _rows.extend(_expand_result_to_rows(r, _supabase_run_meta))
                _supabase_insert(_rows, _supabase_url, _supabase_key)
            else:
                checkpoint_path = os.path.join("results", f"shor_{shor_timestamp}_N{N_shor}.json")
                _save_json(checkpoint_path, {
                    "schema_version": "1.0",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "emulated": args.emulated,
                    "n_to_factor": N_shor,
                    "n_qubits": n_qubits_val,
                    "n_repetitions": config.n_repetitions,
                    "results": shor_n_series,
                })
                print(f"\n→ Checkpoint shor N={N_shor}: {checkpoint_path}")
                if shor_results:
                    shor_partial_doc = _build_output_doc(
                        contributor_name, hw, config, shor_results,
                        platform_id=args.platform, emulated=args.emulated, no_gpu=args.no_gpu,
                    )
                    _save_json(shor_partial_path, shor_partial_doc)

    # ---- Backfill Grover scaling curves ----
    for result in results:
        fw = result["framework"]
        sd = scaling_by_fw.get(fw, {})
        result["scaling_data"] = {int(k): v for k, v in sd.items()}
        if len(sd) >= 2:
            try:
                alpha, beta = fit_scaling_curve(sd)
            except Exception:
                alpha, beta = 0.0, 0.0
        else:
            alpha, beta = 0.0, 0.0
        result["scaling_alpha"] = alpha
        result["scaling_beta"] = beta

    final_doc = _build_output_doc(
        contributor_name, hw, config, results,
        platform_id=args.platform, emulated=args.emulated, no_gpu=args.no_gpu,
    )
    if not USE_SUPABASE:
        _save_json(final_path, final_doc)

    if USE_SUPABASE:
        _fw_seen: set[str] = set()
        for _r in results:
            _fw = _r.get("framework")
            if _fw and _fw not in _fw_seen:
                _fw_seen.add(_fw)
                _supabase_patch_scaling(
                    run_id, "grover", _fw,
                    _r.get("scaling_alpha", 0.0),
                    _r.get("scaling_beta", 0.0),
                    _r.get("scaling_data", {}),
                    _supabase_url, _supabase_key,
                )

    if not USE_SUPABASE and os.path.exists(partial_path):
        try:
            os.remove(partial_path)
        except OSError:
            pass

    print_summary_table(results, statuses)
    print()
    if not USE_SUPABASE:
        print(f"Resultados Grover guardados en: {final_path}")

    # ---- Backfill Shor scaling curves and finalize ----
    print("\n" + "=" * 58)
    print("  Shor — Factorización Cuántica")
    print("=" * 58)

    for r in shor_results:
        fw = r["framework"]
        sd = shor_scaling_by_fw.get(fw, {})
        r["scaling_data"] = {int(k): v for k, v in sd.items()}
        if len(sd) >= 2:
            try:
                alpha, beta = fit_scaling_curve(sd)
            except Exception:
                alpha, beta = 0.0, 0.0
        else:
            alpha, beta = 0.0, 0.0
        r["scaling_alpha"] = alpha
        r["scaling_beta"] = beta

    if shor_results:
        if not USE_SUPABASE:
            _save_json(shor_final_path, _build_output_doc(
                contributor_name, hw, config, shor_results,
                platform_id=args.platform, emulated=args.emulated, no_gpu=args.no_gpu,
            ))
        if USE_SUPABASE:
            _fw_seen_shor: set[str] = set()
            for _r in shor_results:
                _fw = _r.get("framework")
                if _fw and _fw not in _fw_seen_shor:
                    _fw_seen_shor.add(_fw)
                    _supabase_patch_scaling(
                        run_id, "shor", _fw,
                        _r.get("scaling_alpha", 0.0),
                        _r.get("scaling_beta", 0.0),
                        _r.get("scaling_data", {}),
                        _supabase_url, _supabase_key,
                    )
        print_shor_summary_table(shor_results, shor_statuses)
        if not USE_SUPABASE and os.path.exists(shor_partial_path):
            os.remove(shor_partial_path)
        if not USE_SUPABASE:
            print(f"\nResultados Shor guardados en: {shor_final_path}")


if __name__ == "__main__":
    import signal as _signal

    def _sigterm(_sig: int, _frame: object) -> None:
        print("\n→ Benchmark detenido.", flush=True)
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _sigterm)

    try:
        main()
    except KeyboardInterrupt:
        print("\n→ Benchmark interrumpido por el usuario.", flush=True)
        sys.exit(130)
