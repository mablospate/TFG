"""Hardware detection module for benchmark runs."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class HardwareInfo:
    hostname: str
    os: str
    os_version: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_gflops: float
    ram_total_gb: float
    gpu_model: str | None
    gpu_vram_gb: float | None
    python_version: str


def _normalize_os(system: str) -> str:
    s = system.lower()
    if s == "darwin":
        return "macos"
    if s.startswith("win"):
        return "windows"
    return s


def _detect_cpu_model() -> str:
    system = platform.system().lower()
    try:
        if system == "linux":
            with open("/proc/cpuinfo", "r") as fh:
                content = fh.read()
            # x86: "model name" field
            for line in content.splitlines():
                if line.startswith("model name"):
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
            # ARM: "Model name" via lscpu (most reliable across distros)
            try:
                out = subprocess.run(
                    ["lscpu"], capture_output=True, text=True, timeout=5
                )
                if out.returncode == 0:
                    for line in out.stdout.splitlines():
                        if line.startswith("Model name"):
                            val = line.split(":", 1)[1].strip()
                            if val:
                                return val
                        if line.startswith("Vendor ID"):
                            pass  # keep looking for Model name
            except Exception:
                pass
            # ARM fallback: "Hardware" or "Model" fields in /proc/cpuinfo
            for prefix in ("Model\t", "Model ", "Hardware"):
                for line in content.splitlines():
                    if line.startswith(prefix):
                        val = line.split(":", 1)[1].strip()
                        if val:
                            return val
            # Last resort: architecture string
            arch = platform.machine()
            return f"unknown ({arch})"
        elif system == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif system.startswith("win"):
            return platform.processor()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _measure_cpu_gflops() -> float:
    """64×64 float64 matmul microbenchmark, K=200 iterations.

    Score = (2 × 64³ × K) / elapsed_ns × 1000
    (1000× conventional GFLOPS; consistent scale across all platforms.)
    Returns 0.0 on any failure.
    """
    try:
        import numpy as np
        import time

        n, K = 64, 200
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        _ = A @ B  # warmup: initialise BLAS thread pool
        t0 = time.perf_counter_ns()
        for _ in range(K):
            _C = A @ B
        elapsed_ns = time.perf_counter_ns() - t0
        if elapsed_ns <= 0:
            return 0.0
        return (2 * n**3 * K) / elapsed_ns * 1000
    except Exception:
        return 0.0


def _detect_gpu() -> tuple[str | None, float | None]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None, None
        first_line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 2:
            return None, None
        name = parts[0]
        vram_mb = float(parts[1])
        return name, vram_mb / 1024.0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None, None


def _env_or(key: str, default):
    val = os.environ.get(key)
    return val if val and val.strip() else default


def detect_hardware() -> HardwareInfo:
    """Detect hardware characteristics of the current machine."""
    gpu_model, gpu_vram_gb = _detect_gpu()

    return HardwareInfo(
        hostname=_env_or("BENCH_HOSTNAME", platform.node()),
        os=_normalize_os(_env_or("BENCH_OS", platform.system())),
        os_version=_env_or("BENCH_OS_VERSION", platform.version()),
        cpu_model=_env_or("BENCH_CPU_MODEL", _detect_cpu_model()),
        cpu_cores_physical=int(_env_or("BENCH_CPU_CORES_PHYSICAL",
                                        str(psutil.cpu_count(logical=False) or 1))),
        cpu_cores_logical=int(_env_or("BENCH_CPU_CORES_LOGICAL",
                                       str(psutil.cpu_count(logical=True) or 1))),
        cpu_gflops=float(os.environ["BENCH_CPU_GFLOPS"]) if float(os.environ.get("BENCH_CPU_GFLOPS") or 0) > 0 else _measure_cpu_gflops(),
        ram_total_gb=float(_env_or("BENCH_RAM_GB",
                                    str(round(psutil.virtual_memory().total / (1024**3), 1)))),
        gpu_model=gpu_model,
        gpu_vram_gb=gpu_vram_gb,
        python_version=platform.python_version(),
    )
