"""
benchmark_core.py — Sistema de benchmarking agnóstico de framework.

Dependencias externas:
    pip install psutil scipy numpy

No depende de ningún framework cuántico.
"""

from __future__ import annotations

import json
import platform
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Any

import numpy as np
import psutil
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Parámetros globales del experimento."""

    n_repetitions: int = 10  # Repeticiones para estadísticas
    warmup_runs: int = 1  # Ejecuciones de calentamiento (no se miden)
    n_values: list[int] = field(
        default_factory=lambda: [3, 5, 7, 9, 11]
    )  # Tamaños de n para escalabilidad
    n_values_shor: list[int] = field(
        default_factory=lambda: [15, 21, 35, 77, 143]
    )
    num_shots: int = 1024  # Shots para distribución empírica
    cpu_sample_interval: float = 0.05  # Intervalo de muestreo de CPU (s)


@dataclass
class BenchmarkResult:
    """Resultado completo de un experimento de benchmarking."""

    # --- Métricas obligatorias ---
    wall_time_median_ms: float = 0.0  # Mediana del tiempo total (ms)
    wall_time_iqr_ms: float = 0.0  # IQR del tiempo total (ms)
    peak_memory_rss_mb: float = 0.0  # Memoria RSS pico (MB)
    cv: float = 0.0  # Coeficiente de variación σ/μ

    # --- Métricas complementarias ---
    startup_time_ms: float = 0.0  # Tiempo de startup del framework (ms)
    build_time_ms: float = 0.0  # Tiempo de construcción del circuito (ms)
    simulation_time_ms: float = 0.0  # Tiempo de simulación pura (ms)
    cpu_percent_mean: float = 0.0  # CPU medio durante simulación (%)
    jsd: float = 0.0  # Jensen-Shannon divergence vs. teórico
    energy_j: float = 0.0  # Energía consumida (J); 0.0 = no medido

    # --- Escalabilidad ---
    scaling_alpha: float = 0.0  # Coeficiente α en α·2^(β·n)
    scaling_beta: float = 0.0  # Exponente β en α·2^(β·n)
    scaling_data: dict[int, float] = field(default_factory=dict)

    # --- Metadatos ---
    framework: str = ""
    algorithm: str = ""
    n_qubits: int = 0
    timestamp: str = ""
    python_version: str = ""
    platform_info: str = ""
    raw_times_ms: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Medición de uso de CPU en background
# ---------------------------------------------------------------------------


class _CpuSampler:
    """Muestrea el uso de CPU del proceso actual en un hilo secundario."""

    def __init__(self, interval: float = 0.05):
        self._interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._proc.cpu_percent()  # Primer llamado siempre devuelve 0; descartar
        self._thread.start()

    def stop(self) -> float:
        """Detiene el muestreo y devuelve el porcentaje medio."""
        self._stop.set()
        self._thread.join()
        return float(np.mean(self._samples)) if self._samples else 0.0

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._samples.append(self._proc.cpu_percent())


# ---------------------------------------------------------------------------
# Función principal de benchmarking
# ---------------------------------------------------------------------------


def benchmark_run(
    fn: Callable[[], Any],
    config: BenchmarkConfig | None = None,
    framework: str = "",
    algorithm: str = "",
    n_qubits: int = 0,
) -> BenchmarkResult:
    """
    Ejecuta `fn` repetidamente y recopila todas las métricas instrumentales.

    `fn` debe ser una callable sin argumentos que devuelva cualquier valor
    (el sistema ignora el valor de retorno; solo mide tiempo, memoria y CPU).
    Para pasar argumentos usa un lambda: ``lambda: mi_funcion(arg1, arg2)``.

    Retorna un BenchmarkResult con mediana, IQR y CV del wall time,
    memoria RSS pico y CPU medio.
    """
    if config is None:
        config = BenchmarkConfig()

    # --- Calentamiento ---
    for _ in range(config.warmup_runs):
        fn()

    times_ms: list[float] = []
    peak_rss_mb: float = 0.0

    # --- Repeticiones de medición ---
    for i in range(config.n_repetitions):
        # Memoria: tracemalloc captura asignaciones de Python;
        # psutil captura la RSS del proceso completo (incluye extensiones C).
        proc = psutil.Process()
        tracemalloc.start()

        cpu_sampler = _CpuSampler(config.cpu_sample_interval)
        cpu_sampler.start()

        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()

        cpu_mean = cpu_sampler.stop()

        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_mb = proc.memory_info().rss / 1024 / 1024

        elapsed_ms = (t1 - t0) * 1000.0
        times_ms.append(elapsed_ms)
        peak_rss_mb = max(peak_rss_mb, rss_mb)

    # --- Estadísticas ---
    arr = np.array(times_ms)
    median_ms = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_ms = float(q75 - q25)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0

    result = BenchmarkResult(
        wall_time_median_ms=median_ms,
        wall_time_iqr_ms=iqr_ms,
        peak_memory_rss_mb=peak_rss_mb,
        cv=cv,
        cpu_percent_mean=cpu_mean,
        framework=framework,
        algorithm=algorithm,
        n_qubits=n_qubits,
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version,
        platform_info=platform.platform(),
        raw_times_ms=times_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Medición de startup del framework
# ---------------------------------------------------------------------------


def measure_startup_time(import_fn: Callable[[], None]) -> float:
    """
    Mide el tiempo de importar e inicializar un framework en frío.

    `import_fn` debe realizar los imports y cualquier inicialización del
    simulador (p. ej. crear el backend, compilar JIT) pero no construir
    ningún circuito.

    Retorna el tiempo en milisegundos.
    """
    t0 = time.perf_counter()
    import_fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# Medición del tiempo de construcción del circuito
# ---------------------------------------------------------------------------


def measure_build_time(build_fn: Callable[[], Any], *args: Any) -> float:
    """
    Cronometra únicamente la construcción del circuito (sin ejecución).

    `build_fn` debe definir puertas y registros pero no llamar a
    `execute()`, `run()`, `sample()` ni equivalentes.

    Retorna el tiempo en milisegundos.
    """
    t0 = time.perf_counter()
    build_fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


# ---------------------------------------------------------------------------
# Escalabilidad con n
# ---------------------------------------------------------------------------


def measure_scaling(
    run_fn: Callable[..., Any],
    n_values: list[int] | None = None,
    config: BenchmarkConfig | None = None,
    **kwargs: Any,
) -> dict[int, float]:
    """
    Ejecuta `run_fn` para distintos valores de `n` y devuelve los tiempos medianos.

    `run_fn` debe aceptar `n` como primer argumento posicional.
    Los `kwargs` adicionales se pasan directamente a `run_fn`.

    Retorna ``{n: wall_time_median_ms}``.
    """
    if config is None:
        config = BenchmarkConfig()
    if n_values is None:
        n_values = config.n_values

    scaling: dict[int, float] = {}
    for n in n_values:
        result = benchmark_run(lambda n=n: run_fn(n, **kwargs), config=config)
        scaling[n] = result.wall_time_median_ms
    return scaling


def fit_scaling_curve(scaling_data: dict[int, float]) -> tuple[float, float]:
    """
    Ajusta la curva ``t(n) = α · 2^(β·n)`` a los datos de escalabilidad.

    Retorna ``(alpha, beta)``. Un β cercano a 1 indica escalado exponencial
    perfecto en base 2 (esperado para simulación de statevector).
    """
    ns = np.array(sorted(scaling_data.keys()), dtype=float)
    ts = np.array([scaling_data[int(n)] for n in ns])

    def model(n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return alpha * np.power(2.0, beta * n)

    try:
        popt, _ = curve_fit(model, ns, ts, p0=[1.0, 1.0], maxfev=10000)
        return float(popt[0]), float(popt[1])
    except RuntimeError:
        # Si no converge, estimación lineal en log-space
        log_ts = np.log2(ts + 1e-12)
        beta, log_alpha = np.polyfit(ns, log_ts, 1)
        return float(2.0**log_alpha), float(beta)


# ---------------------------------------------------------------------------
# Precisión numérica: Jensen-Shannon divergence
# ---------------------------------------------------------------------------


def compute_jsd(
    empirical_dist: dict[str, float],
    theoretical_dist: dict[str, float],
) -> float:
    """
    Calcula la divergencia Jensen-Shannon entre distribución empírica y teórica.

    Ambos diccionarios mapean estado (string de bits) → probabilidad.
    Las probabilidades se normalizan automáticamente; los estados ausentes
    reciben probabilidad 0.

    Retorna un valor en ``[0, 1]`` (0 = distribuciones idénticas).
    """
    all_states = sorted(set(empirical_dist) | set(theoretical_dist))
    p = np.array([empirical_dist.get(s, 0.0) for s in all_states], dtype=float)
    q = np.array([theoretical_dist.get(s, 0.0) for s in all_states], dtype=float)

    # Normalizar para garantizar distribuciones de probabilidad válidas
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum > 0:
        p /= p_sum
    if q_sum > 0:
        q /= q_sum

    return float(jensenshannon(p, q))


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------


def save_results(results: list[BenchmarkResult], path: str) -> None:
    """
    Guarda una lista de BenchmarkResult en un fichero JSON.

    El JSON incluye una cabecera de metadatos del sistema y la lista de
    resultados con todos los campos de la dataclass.
    """
    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
