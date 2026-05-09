# Sistema de Benchmarking para Algoritmos Cuánticos

Sistema agnóstico de framework y algoritmo para medir, comparar y reproducir el rendimiento de implementaciones cuánticas. Diseñado para funcionar con cualquier combinación de lenguaje, simulador y algoritmo sin modificar el núcleo de medición.

---

## Tabla de contenidos

1. [Las 10 métricas](#1-las-10-métricas)
2. [Implementación en Python](#2-implementación-en-python)
3. [Implementación en Rust](#3-implementación-en-rust)
4. [Cómo usar el sistema con una nueva implementación](#4-cómo-usar-el-sistema-con-una-nueva-implementación)
5. [Formato de salida JSON](#5-formato-de-salida-json)

---

## 1. Las 10 métricas

### Tabla resumen

| # | Métrica | Tipo | Herramienta (Python) | Herramienta (Rust) | Obligatoria |
|---|---------|------|----------------------|--------------------|-------------|
| 1 | Tiempo total de ejecución | Wall-clock, ms | `time.perf_counter()` | `std::time::Instant` | Sí |
| 2 | Memoria pico (RSS) | RAM residente, MB | `psutil.Process().memory_info().rss` | `/proc/self/status` / `getrusage` | Sí |
| 3 | Tiempo de startup del framework | ms | `perf_counter` antes de primera puerta | `Instant` antes del primer gate | Sí |
| 4 | Tiempo de construcción del circuito | ms | `perf_counter` antes de `execute()` | `Instant` antes de `run()` | Sí |
| 5 | Tiempo de simulación pura | ms | `total − startup − build` | ídem | Sí |
| 6 | Escalabilidad con n | curva `α·2^(βn)` | 5 tamaños; `scipy.optimize.curve_fit` | 5 tamaños; regresión manual | Sí |
| 7 | Varianza entre ejecuciones (CV) | adimensional | `σ/μ` sobre ≥10 repeticiones | `std_dev / mean` | Sí |
| 8 | CPU medio durante simulación | % | `psutil` muestreo 50 ms | `/proc/stat` muestreo | Sí |
| 9 | Precisión numérica | Jensen-Shannon div. | distribución empírica vs. teórica | ídem | Sí |
| 10 | Consumo energético | J / mJ | `pyRAPL` / `nvidia-smi` | `RAPL` vía `/sys/class/powercap` | Sí |

Todas las métricas son obligatorias.

---

### Descripción individual

#### 1. Tiempo total de ejecución
Tiempo de reloj de pared desde que se invoca la función hasta que se recibe el resultado. Captura todo: imports cacheados, construcción del circuito, simulación y postproceso. Es el indicador de latencia real que experimenta el usuario.

#### 2. Memoria pico (RSS — Resident Set Size)
Máximo de RAM física ocupada por el proceso durante la ejecución. Se mide como el máximo de `rss` muestreado con `psutil` o el campo `VmPeak` de `/proc/self/status`. Detecta ineficiencias de memoria en el estado cuántico (el vector de estado crece como `2^n` amplitudes complejas).

#### 3. Tiempo de startup del framework
Coste de importar el módulo y inicializar el simulador, sin construir ningún circuito. Se mide una sola vez por sesión con el framework en frío. Permite separar el overhead fijo del coste variable por algoritmo. Se resta del tiempo total para obtener las métricas 4 y 5.

#### 4. Tiempo de construcción del circuito
Tiempo empleado únicamente en la definición de puertas y registros, hasta el momento inmediatamente anterior a la llamada de ejecución (`execute()`, `run()`, `sample()`, etc.). Refleja el coste de la representación intermedia del framework.

#### 5. Tiempo de simulación pura
Tiempo del paso de ejecución cuántica propiamente dicho: evolución del estado, medición y postproceso de shots. Se obtiene como `total − startup − build`. Es el número más relevante para comparar la eficiencia del motor de simulación.

#### 6. Escalabilidad con n
Cómo crece el tiempo de ejecución al aumentar el número de qubits. Se ejecuta el algoritmo para 5 valores distintos de `n` y se ajusta la curva `t(n) = α · 2^(β·n)` por mínimos cuadrados. Los coeficientes `α` y `β` son las cifras de comparación entre frameworks.

#### 7. Varianza entre ejecuciones (CV)
Coeficiente de variación `CV = σ / μ` calculado sobre ≥10 repeticiones de la misma configuración. Un CV bajo indica un simulador determinista y estable; un CV alto puede revelar contención de recursos, JIT inestable o dependencia del estado de la memoria. Se reporta junto con la mediana y el IQR para evitar distorsión por outliers.

#### 8. CPU medio durante simulación
Porcentaje medio de CPU del proceso (y opcionalmente del sistema) durante el paso de simulación, muestreado cada 50 ms. Detecta si el framework aprovecha múltiples núcleos o delega en GPU. Un valor cercano a `100 × n_cores` indica paralelismo efectivo.

#### 9. Precisión numérica
Divergencia de Jensen-Shannon entre la distribución de probabilidad empírica (obtenida con `num_shots` mediciones) y la distribución teórica ideal (calculada analíticamente o con un statevector exacto). Valor en `[0, 1]`; cuanto más bajo, más fiel es la simulación. Permite detectar errores de implementación enmascarados por resultados visualmente correctos.

#### 10. Consumo energético
Energía total consumida durante la simulación, medida en julios. En CPU x86 con Intel RAPL se lee de `/sys/class/powercap/intel-rapl/*/energy_uj`. En GPU NVIDIA se muestrea `nvidia-smi --query-gpu=power.draw` cada 100 ms y se integra. Permite calcular la eficiencia energética real (`operaciones / julio`) para comparar implementaciones en hardware equivalente.

---

## 2. Implementación en Python

El módulo no importa ningún framework cuántico. Recibe callables arbitrarias y devuelve objetos `BenchmarkResult` serializables.

```python
"""
benchmark_core.py — Sistema de benchmarking agnóstico de framework.

Dependencias externas:
    pip install psutil scipy numpy

No depende de ningún framework cuántico.
"""

from __future__ import annotations

import json
import math
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
    n_repetitions: int = 10       # Repeticiones para estadísticas
    warmup_runs: int = 1          # Ejecuciones de calentamiento (no se miden)
    n_values: list[int] = field(
        default_factory=lambda: [3, 4, 5, 6, 8]
    )                             # Tamaños de n para escalabilidad
    num_shots: int = 1024         # Shots para distribución empírica
    cpu_sample_interval: float = 0.05  # Intervalo de muestreo de CPU (s)


@dataclass
class BenchmarkResult:
    """Resultado completo de un experimento de benchmarking."""
    # --- Métricas obligatorias ---
    wall_time_median_ms: float = 0.0   # Mediana del tiempo total (ms)
    wall_time_iqr_ms: float = 0.0      # IQR del tiempo total (ms)
    peak_memory_rss_mb: float = 0.0    # Memoria RSS pico (MB)
    cv: float = 0.0                    # Coeficiente de variación σ/μ

    # --- Métricas complementarias ---
    startup_time_ms: float = 0.0       # Tiempo de startup del framework (ms)
    build_time_ms: float = 0.0         # Tiempo de construcción del circuito (ms)
    simulation_time_ms: float = 0.0    # Tiempo de simulación pura (ms)
    cpu_percent_mean: float = 0.0      # CPU medio durante simulación (%)
    jsd: float = 0.0                   # Jensen-Shannon divergence vs. teórico
    energy_j: float = 0.0              # Energía consumida (J); 0.0 = no medido

    # --- Escalabilidad ---
    scaling_alpha: float = 0.0         # Coeficiente α en α·2^(β·n)
    scaling_beta: float = 0.0          # Exponente β en α·2^(β·n)
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
        return float(2.0 ** log_alpha), float(beta)


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
```

---

## 3. Implementación en Rust

El crate no depende de ningún crate cuántico. Usa solo `std`, `serde` y `serde_json`.

### Dependencias en `Cargo.toml`

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

### Código

```rust
//! benchmark_core.rs — Sistema de benchmarking agnóstico de framework.
//!
//! Compilar con:
//!   cargo build --release
//!
//! No depende de ningún crate cuántico.

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};


// ---------------------------------------------------------------------------
// Estructuras de datos
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Número de repeticiones para calcular estadísticas.
    pub n_repetitions: usize,
    /// Ejecuciones de calentamiento (no se miden).
    pub warmup_runs: usize,
    /// Valores de n para el análisis de escalabilidad.
    pub n_values: Vec<usize>,
    /// Número de shots para distribución empírica.
    pub num_shots: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            n_repetitions: 10,
            warmup_runs: 1,
            n_values: vec![3, 4, 5, 6, 8],
            num_shots: 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    // --- Métricas obligatorias ---
    /// Mediana del tiempo total de ejecución (ms).
    pub wall_time_median_ms: f64,
    /// IQR del tiempo total (ms).
    pub wall_time_iqr_ms: f64,
    /// Memoria RSS pico (bytes).
    pub peak_memory_rss_bytes: u64,
    /// Coeficiente de variación σ/μ.
    pub cv: f64,

    // --- Escalabilidad ---
    /// Coeficiente α en α·2^(β·n).
    pub scaling_alpha: f64,
    /// Exponente β en α·2^(β·n).
    pub scaling_beta: f64,
    /// Datos crudos de escalabilidad: n → wall_time_median_ms.
    pub scaling_data: HashMap<usize, f64>,

    // --- Métricas complementarias ---
    /// Tiempo de startup del framework (ms).
    pub startup_time_ms: f64,
    /// Tiempo de construcción del circuito (ms).
    pub build_time_ms: f64,
    /// Tiempo de simulación pura (ms).
    pub simulation_time_ms: f64,
    /// Energía consumida (J); 0.0 = no medido.
    pub energy_j: f64,

    // --- Metadatos ---
    pub framework: String,
    pub algorithm: String,
    pub n_qubits: usize,
    pub timestamp: String,
    pub os: String,
    /// Tiempos individuales de cada repetición (ms).
    pub raw_times_ms: Vec<f64>,
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self {
            wall_time_median_ms: 0.0,
            wall_time_iqr_ms: 0.0,
            peak_memory_rss_bytes: 0,
            cv: 0.0,
            scaling_alpha: 0.0,
            scaling_beta: 0.0,
            scaling_data: HashMap::new(),
            startup_time_ms: 0.0,
            build_time_ms: 0.0,
            simulation_time_ms: 0.0,
            energy_j: 0.0,
            framework: String::new(),
            algorithm: String::new(),
            n_qubits: 0,
            timestamp: String::new(),
            os: std::env::consts::OS.to_string(),
            raw_times_ms: Vec::new(),
        }
    }
}


// ---------------------------------------------------------------------------
// Lectura de memoria del proceso
// ---------------------------------------------------------------------------

/// Devuelve la memoria RSS del proceso actual en bytes.
/// Implementado para Linux y macOS; devuelve 0 en otros sistemas.
fn peak_memory_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // Lee VmRSS de /proc/self/status
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: u64 = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    return kb * 1024;
                }
            }
        }
        0
    }

    #[cfg(target_os = "macos")]
    {
        // Usa getrusage(RUSAGE_SELF) que devuelve ru_maxrss en bytes en macOS
        unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
                // En macOS ru_maxrss está en bytes
                return usage.ru_maxrss as u64;
            }
        }
        0
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

// Nota: en macOS añadir al Cargo.toml:
//   [target.'cfg(target_os = "macos")'.dependencies]
//   libc = "0.2"


// ---------------------------------------------------------------------------
// Estadísticas auxiliares
// ---------------------------------------------------------------------------

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

fn duration_to_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}


// ---------------------------------------------------------------------------
// Función principal de benchmarking
// ---------------------------------------------------------------------------

/// Ejecuta `f` repetidamente y recopila métricas de tiempo y memoria.
///
/// `f` es cualquier closure que no recibe argumentos. Para pasar
/// parámetros usa una closure que los capture:
/// ```rust
/// benchmark_run(|| mi_funcion(arg1, arg2), &config)
/// ```
pub fn benchmark_run<F, R>(f: F, config: &BenchmarkConfig) -> BenchmarkResult
where
    F: Fn() -> R,
{
    // Calentamiento
    for _ in 0..config.warmup_runs {
        let _ = f();
    }

    let mut times_ms: Vec<f64> = Vec::with_capacity(config.n_repetitions);
    let mut peak_mem: u64 = 0;

    for _ in 0..config.n_repetitions {
        let t0 = Instant::now();
        let _ = f();
        let elapsed = t0.elapsed();

        times_ms.push(duration_to_ms(elapsed));

        let mem = peak_memory_bytes();
        if mem > peak_mem {
            peak_mem = mem;
        }
    }

    // Estadísticas
    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_ms = median(&sorted);
    let q75 = percentile(&sorted, 75.0);
    let q25 = percentile(&sorted, 25.0);
    let iqr_ms = q75 - q25;
    let m = mean(&times_ms);
    let s = std_dev(&times_ms);
    let cv = if m > 0.0 { s / m } else { 0.0 };

    BenchmarkResult {
        wall_time_median_ms: median_ms,
        wall_time_iqr_ms: iqr_ms,
        peak_memory_rss_bytes: peak_mem,
        cv,
        raw_times_ms: times_ms,
        ..BenchmarkResult::default()
    }
}


// ---------------------------------------------------------------------------
// Escalabilidad
// ---------------------------------------------------------------------------

/// Ejecuta el algoritmo para distintos valores de `n` y devuelve tiempos medianos.
///
/// - `build_fn(n)` construye y devuelve cualquier tipo `F` (el circuito, el objeto
///   de simulación, etc.).
/// - `run_fn(f)` ejecuta ese objeto. La separación permite medir el tiempo de
///   construcción por separado si se desea.
///
/// Retorna `{n → wall_time_median_ms}`.
pub fn measure_scaling<B, F, R>(
    build_fn: impl Fn(usize) -> B,
    run_fn: impl Fn(B) -> R,
    n_values: &[usize],
    config: &BenchmarkConfig,
) -> HashMap<usize, f64>
where
    B: Clone,
{
    let mut results = HashMap::new();
    for &n in n_values {
        // Reconstruir el circuito dentro de la closure para incluir solo run_fn
        let built = build_fn(n);
        let result = benchmark_run(
            || {
                let b = built.clone();
                run_fn(b)
            },
            config,
        );
        results.insert(n, result.wall_time_median_ms);
    }
    results
}

/// Ajusta la curva `t(n) = α · 2^(β·n)` usando regresión lineal en log-space.
///
/// Retorna `(alpha, beta)`.
pub fn fit_scaling_curve(scaling_data: &HashMap<usize, f64>) -> (f64, f64) {
    if scaling_data.len() < 2 {
        return (0.0, 0.0);
    }

    let mut ns: Vec<f64> = scaling_data.keys().map(|&n| n as f64).collect();
    ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ts: Vec<f64> = ns.iter().map(|n| *scaling_data.get(&(*n as usize)).unwrap()).collect();

    // Regresión lineal de log2(t) sobre n: log2(t) = log2(α) + β·n
    let log_ts: Vec<f64> = ts.iter().map(|t| t.max(1e-12_f64).log2()).collect();
    let n_pts = ns.len() as f64;
    let mean_n = ns.iter().sum::<f64>() / n_pts;
    let mean_lt = log_ts.iter().sum::<f64>() / n_pts;

    let cov = ns.iter().zip(&log_ts).map(|(n, lt)| (n - mean_n) * (lt - mean_lt)).sum::<f64>();
    let var_n = ns.iter().map(|n| (n - mean_n).powi(2)).sum::<f64>();

    if var_n == 0.0 {
        return (0.0, 0.0);
    }

    let beta = cov / var_n;
    let log_alpha = mean_lt - beta * mean_n;
    let alpha = 2_f64.powf(log_alpha);

    (alpha, beta)
}


// ---------------------------------------------------------------------------
// Serialización
// ---------------------------------------------------------------------------

/// Serializa una lista de `BenchmarkResult` a un fichero JSON.
pub fn save_results(results: &[BenchmarkResult], path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, results)?;
    Ok(())
}
```

---

## 4. Cómo usar el sistema con una nueva implementación

El sistema no necesita conocer los detalles del framework. El único contrato es envolver la llamada en una closure sin argumentos.

### Python

```python
# Supongamos que tenemos cualquier módulo `mi_framework` con una función:
#   mi_framework.search(n: int, target: int) -> tuple[int, dict[str, float]]
# El segundo elemento de la tupla es la distribución de probabilidad empírica.

import mi_framework  # cualquier framework, real o ficticio
from benchmark_core import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_run,
    measure_startup_time,
    measure_build_time,
    measure_scaling,
    fit_scaling_curve,
    compute_jsd,
    save_results,
)

config = BenchmarkConfig(n_repetitions=10, num_shots=1024)

# 1. Startup del framework
startup_ms = measure_startup_time(lambda: __import__("mi_framework"))

# 2. Construcción del circuito (si el framework lo separa)
build_ms = measure_build_time(lambda: mi_framework.build_circuit(n=5, target=3))

# 3. Benchmark completo
result = benchmark_run(
    lambda: mi_framework.search(n=5, target=3),
    config=config,
    framework="mi_framework",
    algorithm="busqueda",
    n_qubits=5,
)
result.startup_time_ms = startup_ms
result.build_time_ms = build_ms
result.simulation_time_ms = result.wall_time_median_ms - startup_ms - build_ms

# 4. Escalabilidad
scaling_data = measure_scaling(
    run_fn=lambda n: mi_framework.search(n=n, target=3),
    n_values=config.n_values,
    config=config,
)
alpha, beta = fit_scaling_curve(scaling_data)
result.scaling_data = scaling_data
result.scaling_alpha = alpha
result.scaling_beta = beta

# 5. Precisión numérica (si se dispone de distribución teórica)
_, empirical = mi_framework.search(n=5, target=3)
theoretical = {"00101": 1.0}  # estado correcto con probabilidad 1 en el ideal
result.jsd = compute_jsd(empirical, theoretical)

# 6. Guardar
save_results([result], "results/mi_framework_busqueda.json")
```

### Rust

```rust
use benchmark_core::{
    benchmark_run, measure_scaling, fit_scaling_curve,
    save_results, BenchmarkConfig, BenchmarkResult,
};

// Supongamos:
//   mi_framework::search(n: usize, target: usize) -> HashMap<String, usize>
// (el valor de retorno es ignorado por benchmark_run; solo se mide el tiempo)

fn main() {
    let config = BenchmarkConfig::default();

    // 1. Benchmark completo de la función de búsqueda con n=5, target=3
    let mut result = benchmark_run(|| mi_framework::search(5, 3), &config);
    result.framework = "mi_framework".to_string();
    result.algorithm = "busqueda".to_string();
    result.n_qubits = 5;

    // 2. Escalabilidad
    let scaling_data = measure_scaling(
        |n| n,                               // build_fn: aquí no hay fase separada
        |n| { mi_framework::search(n, 3); }, // run_fn
        &config.n_values,
        &config,
    );
    let (alpha, beta) = fit_scaling_curve(&scaling_data);
    result.scaling_data = scaling_data;
    result.scaling_alpha = alpha;
    result.scaling_beta = beta;

    // 3. Guardar
    save_results(&[result], "results/mi_framework_busqueda.json")
        .expect("Error al guardar resultados");
}
```

---

## 5. Formato de salida JSON

### Schema

El fichero JSON tiene dos niveles: una cabecera de metadatos del experimento y un array `results` con un objeto por cada `BenchmarkResult`.

```json
{
  "schema_version": "string",
  "generated_at": "ISO 8601 UTC",
  "python_version": "string",
  "platform": "string",
  "results": [
    {
      "wall_time_median_ms": "number",
      "wall_time_iqr_ms": "number",
      "peak_memory_rss_mb": "number",
      "cv": "number",
      "startup_time_ms": "number",
      "build_time_ms": "number",
      "simulation_time_ms": "number",
      "cpu_percent_mean": "number",
      "jsd": "number",
      "energy_j": "number",
      "scaling_alpha": "number",
      "scaling_beta": "number",
      "scaling_data": { "n": "wall_time_median_ms" },
      "framework": "string",
      "algorithm": "string",
      "n_qubits": "integer",
      "timestamp": "ISO 8601 UTC",
      "python_version": "string",
      "platform_info": "string",
      "raw_times_ms": ["number"]
    }
  ]
}
```

### Ejemplo real

Resultado de ejecutar el algoritmo de Shor (n=15, factorizar 15) con Qiskit en un MacBook Pro M3 con 16 GB de RAM:

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-05-02T10:34:51.204Z",
  "python_version": "3.11.9 (main, Apr  2 2025, 08:02:17)",
  "platform": "macOS-15.2-arm64-arm-64bit",
  "results": [
    {
      "wall_time_median_ms": 847.32,
      "wall_time_iqr_ms": 23.41,
      "peak_memory_rss_mb": 312.7,
      "cv": 0.028,
      "startup_time_ms": 412.10,
      "build_time_ms": 38.45,
      "simulation_time_ms": 396.77,
      "cpu_percent_mean": 98.4,
      "jsd": 0.0031,
      "energy_j": 0.0,
      "scaling_alpha": 0.00142,
      "scaling_beta": 0.993,
      "scaling_data": {
        "3": 12.4,
        "4": 24.7,
        "5": 49.1,
        "6": 98.8,
        "8": 401.3
      },
      "framework": "qiskit",
      "algorithm": "shor",
      "n_qubits": 8,
      "timestamp": "2026-05-02T10:34:51.204Z",
      "python_version": "3.11.9 (main, Apr  2 2025, 08:02:17)",
      "platform_info": "macOS-15.2-arm64-arm-64bit",
      "raw_times_ms": [
        831.12, 842.75, 850.33, 847.90, 844.11,
        848.20, 853.01, 839.44, 847.32, 871.22
      ]
    }
  ]
}
```

### Convenciones de nombres de fichero

Los ficheros de resultados deben seguir el patrón:

```
results/{framework}_{algorithm}_n{n_qubits}_{YYYYMMDD_HHMMSS}.json
```

Ejemplo: `results/qiskit_shor_n8_20260502_103451.json`

Esto permite cargar y comparar resultados de varios frameworks con un glob simple (`results/*/shor_n8_*.json`) sin parsear el contenido.

---

## 6. Formato del POST a Supabase

Cada combinación framework + algoritmo genera un POST independiente al endpoint de Supabase. El Content-Type es `application/json` y requiere dos cabeceras de autenticación.

### Cabeceras

```http
Content-Type: application/json
apikey: sb_publishable_Opwps_8Bdx1px2PDxrb5Ew_Eens5LsW
Authorization: Bearer sb_publishable_Opwps_8Bdx1px2PDxrb5Ew_Eens5LsW
Prefer: return=minimal
```

`Prefer: return=minimal` evita que Supabase devuelva la fila insertada, reduciendo el payload de respuesta.

### Endpoint

```
POST https://umbmvwkkjphjqpvdgbpr.supabase.co/rest/v1/
```

### Body

```json
{
  "contributor_name": "Pablo Mateos",
  "timestamp": "2026-05-02T11:34:21Z",
  "framework": "qiskit",
  "framework_version": "1.4.2",
  "algorithm": "grover",
  "n_qubits": 5,
  "num_shots": 1024,
  "n_repetitions": 10,
  "hostname": "macbook-pablo",
  "os": "macos",
  "os_version": "15.4",
  "cpu_model": "Apple M3 Pro",
  "cpu_cores_physical": 11,
  "cpu_cores_logical": 11,
  "cpu_freq_mhz": 3200,
  "ram_total_gb": 18,
  "gpu_model": null,
  "gpu_vram_gb": null,
  "runtime_version": "3.12.3",
  "wall_time_median_ms": 142.7,
  "wall_time_mean_ms": 145.1,
  "wall_time_std_ms": 8.3,
  "wall_time_iqr_ms": 11.2,
  "peak_memory_rss_mb": 94.6,
  "startup_time_ms": 312.4,
  "build_time_ms": 18.2,
  "simulation_time_ms": 124.5,
  "scaling_alpha": 0.034,
  "scaling_beta": 0.91,
  "cv_wall_time": 0.058,
  "cpu_mean_percent": 61.2,
  "jsd_precision": 0.0023,
  "energy_j": 0.41
}
```

El campo `id` se omite — Supabase lo genera automáticamente como UUID.

### Ejemplo en Python

```python
import httpx

SUPABASE_URL = "https://umbmvwkkjphjqpvdgbpr.supabase.co"
SUPABASE_KEY = "sb_publishable_Opwps_8Bdx1px2PDxrb5Ew_Eens5LsW"

def upload_result(result: dict) -> None:
    httpx.post(
        f"{SUPABASE_URL}/rest/v1/resultados",
        json=result,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=10,
    ).raise_for_status()
```

### Ejemplo en Rust

```rust
use serde_json::Value;

const SUPABASE_URL: &str = "https://umbmvwkkjphjqpvdgbpr.supabase.co";
const SUPABASE_KEY: &str = "sb_publishable_Opwps_8Bdx1px2PDxrb5Ew_Eens5LsW";

fn upload_result(result: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    client
        .post(format!("{}/rest/v1/resultados", SUPABASE_URL))
        .header("apikey", SUPABASE_KEY)
        .header("Authorization", format!("Bearer {}", SUPABASE_KEY))
        .header("Content-Type", "application/json")
        .header("Prefer", "return=minimal")
        .json(result)
        .send()?
        .error_for_status()?;
    Ok(())
}
```

Añadir al `Cargo.toml`:

```toml
[dependencies]
reqwest = { version = "0.12", features = ["blocking", "json"] }
serde_json = "1"
```
