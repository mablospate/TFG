"""Hardware detection module for benchmark runs."""

from __future__ import annotations

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
    cpu_freq_mhz: float
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
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif system.startswith("win"):
            return platform.processor()
    except Exception:
        pass
    return platform.processor() or "unknown"


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


def detect_hardware() -> HardwareInfo:
    """Detect hardware characteristics of the current machine."""
    freq = psutil.cpu_freq()
    gpu_model, gpu_vram_gb = _detect_gpu()

    return HardwareInfo(
        hostname=platform.node(),
        os=_normalize_os(platform.system()),
        os_version=platform.version(),
        cpu_model=_detect_cpu_model(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        cpu_freq_mhz=float(freq.max) if freq else 0.0,
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        gpu_model=gpu_model,
        gpu_vram_gb=gpu_vram_gb,
        python_version=platform.python_version(),
    )
