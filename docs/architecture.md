# Arquitectura del Sistema de Benchmarking Cuántico

**Proyecto**: TFG — Benchmarking de Algoritmos Cuánticos
**Versión del documento**: 3.0
**Fecha**: 2026-05-17

---

## Tabla de contenidos

1. [Resumen](#1-resumen)
2. [Flujo de usuario](#2-flujo-de-usuario)
3. [Matriz de plataformas](#3-matriz-de-plataformas)
4. [Estructura de la imagen Docker](#4-estructura-de-la-imagen-docker)
5. [Stack tecnológico](#5-stack-tecnológico)
6. [Estructura de directorios](#6-estructura-de-directorios)
7. [Protocolo de medición](#7-protocolo-de-medición)
8. [Formato de salida](#8-formato-de-salida)
9. [Exclusiones por plataforma](#9-exclusiones-por-plataforma)
10. [Decisiones de diseño](#10-decisiones-de-diseño)

---

## 1. Resumen

El sistema es una herramienta de benchmarking que mide y compara el rendimiento de distintos frameworks de simulación cuántica ejecutando dos algoritmos canónicos: la búsqueda de Grover y la factorización de Shor. La comparación abarca frameworks Python (Qiskit, Cirq, CUDA-Q, QDisLib) y frameworks Rust (q1tsim, quantr, quantrs2, qcgpu).

El **único punto de entrada** es Docker. El usuario ejecuta un único script (`./bench` en Linux/macOS o `.\bench.ps1` en Windows) y el sistema se encarga de todo lo demás: detección de hardware, selección de plataforma, ejecución de benchmarks y almacenamiento de resultados. No se requiere instalar Python, Rust, ni ninguna dependencia en el sistema anfitrión. La imagen se publica como `mablospate/tfg-bench:latest`.

El objetivo del benchmark es comparar frameworks entre sí bajo condiciones idénticas, no medir el rendimiento absoluto del hardware. Esto hace que el overhead de virtualización de Docker Desktop (~5-15%) sea irrelevante: afecta a todos los frameworks por igual y los rankings relativos permanecen válidos.

---

## 2. Flujo de usuario

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SISTEMA ANFITRIÓN (host)                         │
│                                                                     │
│  Linux/macOS:   ./bench  [--time-budget N] [--dev]                  │
│  Windows:       .\bench.ps1  [-TimeBudget N] [-Dev]                 │
│                                                                     │
│  Detección de hardware:                                             │
│    · uname -m  → arquitectura (x86_64 / arm64 / aarch64)            │
│    · nvidia-smi → GPU NVIDIA presente en host                       │
│    · sysctl / lscpu → modelo de CPU                                 │
│    · Prompt al usuario: presupuesto de tiempo (--time-budget)       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
            ┌──────────────────┴──────────────────┐
            │                                     │
        AMD64                                  ARM64
            │                                     │
   ┌────────┴────────┐                  ┌────────┴────────┐
   │                 │                  │                 │
 NVIDIA           Sin GPU              ARM nativa        AMD64 emulada (QEMU)
   │                 │                  │                 │
   ▼                 ▼                  ▼                 ▼
Pasada 1: GPU   Pasada única     Pasada 1: arm64    Pasada 2: amd64 vía QEMU
(--gpus all)    (CPU)            (nativa)           (--platform linux/amd64)
   │                                                      │
   ▼                                                      ▼
Pasada 2: CPU                                       qcgpu se ejecuta en QEMU
(--no-gpu)                                          cudaq se OMITE (SIGILL en
qcgpu se OMITE                                      AVX bajo QEMU)
                               │
                               ▼
        docker run [--gpus all] \
          -v $(pwd)/results:/app/results \
          mablospate/tfg-bench:latest
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CONTENEDOR Docker                              │
│                                                                     │
│  docker/entrypoint.sh                                               │
│    ├─ Detecta GPU dentro del contenedor (nvidia-smi)                │
│    ├─ Detecta arquitectura efectiva (uname -m)                      │
│    └─ Selecciona platform_id                                        │
│           │                                                         │
│           ▼                                                         │
│  uv run python run.py --platform <platform_id>                      │
│    ├─ Carga PLATFORM_CONFIGS[platform_id]                           │
│    ├─ Sweep interleaved:                                            │
│    │    Grover n=3, Shor N=15, Grover n=5, Shor N=21, ...           │
│    ├─ Por cada framework activo, lanza subprocess:                  │
│    │    ├─ Python: `python -m python.workers.<fw>_worker`           │
│    │    │   · Config JSON por stdin                                 │
│    │    │   · Resultado JSON = última línea de stdout               │
│    │    │   · Líneas no-JSON → progreso al terminal                 │
│    │    └─ Rust: binario en /app/bin/<crate>-<algo>                 │
│    │        · Args CLI: --n / --N / --shots / --target              │
│    │        · Reporta time_ms propio (sin overhead subprocess)      │
│    ├─ Checkpoint por tamaño: results/{algo}_{ts}_{n|N}{v}.json      │
│    └─ Tras el sweep: scaling fit t(n) = α · 2^(β·n)                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ volumen montado
                               ▼
                   ./results/<timestamp>.json  (host)

  En cualquier momento durante la ejecución:
    · Tecla 'q'  → detiene el benchmark de forma limpia
    · --time-budget agotado → el contenedor se detiene; los checkpoints
      ya escritos en results/ persisten
```

### Descripción paso a paso

1. El usuario invoca `./bench` (Linux/macOS) o `.\bench.ps1` (Windows). El script detecta la arquitectura del host (`x86_64`, `arm64`, `aarch64`) y la presencia de GPU NVIDIA mediante `nvidia-smi`.
2. **Antes** de lanzar el contenedor, el script pregunta al usuario por el presupuesto de tiempo (`--time-budget` en minutos). Si se agota durante la ejecución, el contenedor se detiene y los checkpoints ya escritos persisten en `./results/`.
3. **AMD64 con NVIDIA**: se ejecutan dos pasadas. La primera con `--gpus all` (frameworks usan GPU); la segunda con `--no-gpu` (CPU-only). En la segunda pasada, `qcgpu` se omite porque depende de OpenCL/GPU.
4. **AMD64 sin GPU**: una única pasada CPU.
5. **ARM64**: la primera pasada es nativa arm64. Si el entorno tiene `binfmt_misc` con QEMU configurado, se ejecuta una segunda pasada en `--platform linux/amd64` (modo emulado). Bajo `--emulated`, `cudaq` se omite (genera `SIGILL` al ejecutar instrucciones AVX en QEMU).
6. En todas las pasadas, `docker run` monta `./results` sobre `/app/results`.
7. Dentro del contenedor, `docker/entrypoint.sh` re-detecta GPU y arquitectura efectiva, deriva el `platform_id` correspondiente y lanza `uv run python run.py --platform <platform_id>`.
8. `run.py` consulta la tabla `PLATFORM_CONFIGS`, intercala Grover y Shor por tamaño, escribe un checkpoint JSON tras cada tamaño y, al terminar el sweep, calcula la curva de escalado.
9. Durante toda la ejecución, la tecla `q` (vigilada tanto por `bench` como por `bench.ps1`) detiene el benchmark de forma limpia.

---

## 3. Matriz de plataformas

Hay 10 identificadores de plataforma estáticos en `PLATFORM_CONFIGS` (`run.py`). Qiskit y Cirq están en TODOS; los frameworks Rust `q1tsim`, `quantr` y `quantrs2` también. ProjectQ ha sido eliminado del benchmark.

| Platform ID | cudaq | qdislib | qcgpu | Aceleración | Caveats |
|---|---|---|---|---|---|
| `macos-arm64` | Sí | Sí | No | Ninguna (Hypervisor.framework no expone GPU) | Imagen arm64 nativa |
| `macos-x86_64` | No | Sí | No | Ninguna | cudaq sin wheels Intel Mac |
| `linux-x86_64-nvidia` | Sí | Sí | Sí | CUDA (cudaq + quantrs2-cuQuantum) + OpenCL (qcgpu) | Pasada GPU |
| `linux-x86_64-cpu` | Sí (CPU target) | Sí | No | Ninguna | Pasada CPU del par AMD64+NVIDIA o host sin GPU |
| `linux-aarch64-nvidia` | No | No | No | CUDA limitada (quantrs2) | cudaq sin wheels arm64; pymetis sin wheel arm64 |
| `linux-aarch64-cpu` | No | No | No | Ninguna | Mismas restricciones aarch64 |
| `windows-x86_64-gpu` | No | Sí | Sí | OpenCL (qcgpu) | cudaq no tiene soporte nativo Windows |
| `windows-x86_64-cpu` | No | Sí | No | Ninguna | cudaq excluido en Windows |
| `windows-arm64-gpu` | No | Sí | No | Ninguna | qcgpu requiere OpenCL x86 |
| `windows-arm64-cpu` | No | Sí | No | Ninguna | — |

### Windows + NVIDIA: GPU Paravirtualization (GPU-PV)

Docker Desktop en Windows usa WSL2 como hipervisor. WSL2 implementa GPU Paravirtualization (GPU-PV): el driver NVIDIA del host Windows expone el dispositivo `/dev/dxg` al kernel Linux de WSL2. Los contenedores Linux acceden a CUDA a través de `/usr/lib/wsl/lib/libcuda.so.1`, que actúa como proxy hacia el driver Windows.

El script `bench.ps1` detecta si `nvidia-container-toolkit` está instalado en el entorno WSL2 y lo instala automáticamente si es necesario.

**Requisitos**: Windows 10 21H2+ o Windows 11, driver NVIDIA 471.41+, Docker Desktop con integración WSL2 activa.

### macOS: solo CPU

El framework de virtualización de Apple (`Hypervisor.framework`) no expone el bus PCIe ni la GPU Metal a las máquinas virtuales. Por tanto, los contenedores Linux que ejecuta Docker Desktop en macOS no tienen acceso a la GPU, independientemente de que el Mac tenga Apple Silicon o Intel. En Apple Silicon, la imagen arm64 se ejecuta nativamente sin Rosetta.

---

## 4. Estructura de la imagen Docker

La imagen se construye mediante un `Dockerfile` multi-stage con **cinco etapas**. La separación atiende a dos requisitos: (a) compilar `qcgpu` nativamente para amd64 incluso cuando el host es ARM, y (b) cross-compilar el resto de crates Rust a la arquitectura objetivo.

```
Dockerfile
│
├── Stage 0: qcgpu-amd64
│     FROM --platform=linux/amd64 rust:slim-bookworm
│     · Compila qcgpu NATIVAMENTE para amd64 (vía QEMU si el host es ARM)
│     · qcgpu depende de headers OpenCL que no están disponibles en
│       cross-compilación ARM→AMD64; por eso se aísla en su propia etapa.
│     · Produce: /qcgpu-bins/qcgpu-grover, /qcgpu-bins/qcgpu-shor
│
├── Stage 1: rust-builder
│     FROM --platform=$BUILDPLATFORM rust:slim-bookworm
│     · Compila q1tsim, quantr, quantrs2 para $TARGETARCH (cross-compilación)
│     · Si $TARGETARCH == amd64 (y el host es ARM), copia /qcgpu-bins/
│       desde stage 0 — los binarios qcgpu ya están listos.
│     · Produce: /binaries/{crate}-{grover,shor}
│
├── Stage 2: base-amd64
│     FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
│     · Base CUDA para imágenes amd64 (permite GPU en runtime)
│
├── Stage 3: base-arm64
│     FROM python:3.12-slim-bookworm
│     · Base mínima arm64; sin CUDA porque no hay wheels arm64 de cuda-quantum
│
└── Stage 4: python-deps + runtime
      FROM base-${TARGETARCH}
      · Instala Python deps con uv:
          - amd64: uv sync --extra x86only --extra gpu
          - arm64: uv sync (sin extras)
      · Copia binarios Rust desde stage 1 a /app/bin/  (en PATH)
      · Copia código Python (run.py, python/, docker/)
      · ENTRYPOINT: docker/entrypoint.sh
```

La imagen se publica en `mablospate/tfg-bench:latest`. El entrypoint selecciona en tiempo de ejecución qué plataforma activar basándose en la presencia de GPU y la arquitectura, sin necesidad de imágenes separadas por configuración.

---

## 5. Stack tecnológico

### Frameworks Python

| Framework | Versión | Restricción de plataforma |
|---|---|---|
| Qiskit | 2.0 | Ninguna (todas las plataformas) |
| Cirq | >=1.6.1 | Ninguna |
| CUDA-Q (cudaq) | >=0.14.0 | Solo Linux x86_64 (sin wheels Intel Mac, sin soporte Windows nativo, sin wheels arm64) |
| QDisLib (qdislib) | >=1.0.0 | Todas EXCEPTO Linux aarch64 (dependencia `pymetis` sin wheel arm64) |

ProjectQ ha sido eliminado del benchmark. Ver `docs/framework_exclusions.md` para detalles.

### Frameworks Rust

| Crate | Versión | Restricción de plataforma |
|---|---|---|
| q1tsim | 0.5.0 | Ninguna |
| quantr | 0.6.0 | Ninguna |
| quantrs2 | 0.1.3 | Aceleración CUDA solo en Linux x86_64 |
| qcgpu | 0.1.0 | Solo `linux-x86_64-nvidia` y `windows-x86_64-gpu` (requiere OpenCL GPU) |

### Algoritmos implementados

- **Búsqueda de Grover**: barrido `n = [3, 5, 7, 9, 11]` qubits.
- **Factorización de Shor**: barrido `N = [15, 21, 35, 77, 143]` (números compuestos).

### Herramientas de soporte

| Herramienta | Uso |
|---|---|
| `uv` | Gestión de dependencias Python y ejecución dentro del contenedor |
| Docker multi-stage | Construcción reproducible, imagen mínima de producción |
| QEMU + binfmt_misc | Cross-arch y emulación amd64 sobre hosts ARM |
| `nvidia-smi` | Detección de GPU en host y en contenedor |
| `nvidia-container-toolkit` | Acceso GPU en contenedores (auto-instalado en WSL2 si falta) |
| `tracemalloc` + `psutil` | Medición de pico de memoria y CPU |
| `scipy.optimize.curve_fit` | Ajuste de la curva de escalado post-sweep |

---

## 6. Estructura de directorios

```
TFG/
  run.py                    — orquestador (PLATFORM_CONFIGS, sweep interleaved)
  bench                     — punto de entrada Linux/macOS (bash)
  bench.ps1                 — punto de entrada Windows (PowerShell)
  Dockerfile                — multi-stage (5 etapas)
  docker/
    entrypoint.sh           — detección de GPU/arch, selección de platform_id
  docker-compose.yml        — alias de conveniencia para desarrollo local
  pyproject.toml            — dependencias Python (extras: x86only, gpu)
  uv.lock                   — lockfile reproducible
  python/                   — código Python
    benchmark_core.py       — motor de medición compartido
    hardware.py             — detección de hardware (CPU/GPU/RAM)
    workers/                — un worker por framework (subprocess)
      qiskit_worker.py
      cirq_worker.py
      cudaq_worker.py
      qdislib_worker.py
    qiskit/                 — implementación Qiskit
      grover.py
      shor/shor.py
    cirq/                   — implementación Cirq
    cudaq/                  — implementación CUDA-Q
    qdislib/                — implementación QDisLib (directa + cutting)
  rust/                     — implementaciones Rust por crate
    {crate}/
      src/
        grover.rs
        shor/mod.rs
        bin/
          grover.rs
          shor.rs
      Cargo.toml
  tests/                    — tests unitarios e integración
  results/                  — salida de benchmarks (en .gitignore)
  docs/
    architecture.md         — este documento
    framework_exclusions.md
    framework_analysis.md
```

---

## 7. Protocolo de medición

### Motor compartido: `python/benchmark_core.py`

Todas las mediciones pasan por una API uniforme:

- `BenchmarkConfig(n_repetitions, warmup_runs, n_values, n_values_shor, num_shots, cpu_sample_interval)` — parámetros del run.
- `benchmark_run(fn, config, ...)` — ejecuta `warmup_runs` repeticiones de calentamiento (sin medir), luego `n_repetitions` repeticiones medidas con `tracemalloc` para memoria, `psutil` RSS para pico residente y un sampler en segundo plano para `cpu_percent_mean`.
- `BenchmarkResult` — dataclass con: `wall_time_median_ms`, `wall_time_iqr_ms`, `wall_time_mean_ms`, `wall_time_std_ms`, `peak_memory_rss_mb`, `cv`, `startup_time_ms`, `build_time_ms`, `simulation_time_ms`, `cpu_percent_mean`, `jsd`, `scaling_alpha`, `scaling_beta`, `scaling_data`, `raw_times_ms`.
- `compute_jsd` — Jensen-Shannon divergence entre la distribución empírica de medidas y la distribución teórica `{|target⟩: 1.0}`.
- `measure_build_time` — tiempo de construcción del circuito sin ejecución.

### Protocolo subprocess Python

Cada framework Python se ejecuta como **subprocess independiente**: `python -m python.workers.<fw>_worker`.

- **Entrada**: el orquestador envía por `stdin` un JSON con `{"algo": "grover"|"shor", "n": int, "n_repetitions": int, "num_shots": int, "contributor": str, "cudaq_target": str}`.
- **Salida**: cualquier línea no-JSON emitida por el worker pasa por `stdout` heredado y se muestra como progreso al terminal del usuario; la **última línea** de `stdout` es el JSON con el `BenchmarkResult` serializado.
- **Aislamiento**: timeout de 600 s. Si el subprocess crashea (SIGSEGV, SIGILL, OOM), el orquestador captura el código de salida y registra `status: error`; el proceso padre nunca queda contaminado.

### Protocolo binario Rust

Los binarios Rust están precompilados en `/app/bin/` (en PATH).

- **CLI Grover**: `<binario>-grover --n <n> --target <t> --shots <s>`
- **CLI Shor**: `<binario>-shor --N <N> --shots <s> --tries 3`
- El binario **reporta su propio `time_ms`**, medido internamente — no incluye el overhead de creación del subprocess Python.
- En error, emite un JSON `{"error": "..."}` y sale con código **0** (para no interrumpir el sweep).

### Sweep interleaved con checkpoints

Para que una interrupción no pierda todos los datos, el orquestador intercala los algoritmos por tamaño:

```
Grover n=3  →  Shor N=15
Grover n=5  →  Shor N=21
Grover n=7  →  Shor N=35
Grover n=9  →  Shor N=77
Grover n=11 →  Shor N=143
```

Tras cada tamaño se escribe un checkpoint:

- `results/grover_{timestamp}_n{n}.json`
- `results/shor_{timestamp}_N{N}.json`

### Scaling fit post-sweep

Una vez completado (o interrumpido) el sweep, se ajusta una curva exponencial:

```
t(n) = α · 2^(β · n)
```

mediante `scipy.optimize.curve_fit` sobre los `wall_time_median_ms` por tamaño. Los coeficientes `scaling_alpha` y `scaling_beta`, junto con los puntos `scaling_data`, se incluyen en el resultado final.

---

## 8. Formato de salida

Cada sweep produce checkpoints y un documento final en `./results/`.

### Documento outer (top-level)

| Campo | Tipo | Descripción |
|---|---|---|
| `platform_id` | string | Identificador de plataforma |
| `gpu_enabled` | boolean | Si la pasada usó aceleración GPU |
| `benchmark_image` | string | Tag de imagen Docker (`mablospate/tfg-bench:latest`) |
| `hardware` | object | Ver subsección siguiente |
| `config` | object | `n_repetitions`, `num_shots`, `framework_version` |
| `results` | array | Resultados por framework × algoritmo × tamaño |

### Campos hardware

`hostname`, `os`, `os_version`, `cpu_model`, `cpu_cores_physical`, `cpu_cores_logical`, `cpu_gflops`, `ram_total_gb`, `gpu_model`, `gpu_vram_gb`.

### Campos de resultado por framework

Comunes: `framework`, `algorithm` (`grover`|`shor`), `n` (Grover) o `n_to_factor` (Shor), `wall_time_median_ms`, `wall_time_iqr_ms`, `wall_time_mean_ms`, `wall_time_std_ms`, `peak_memory_rss_mb`, `cv`, `startup_time_ms`, `build_time_ms`, `simulation_time_ms`, `cpu_percent_mean`, `jsd`, `scaling_alpha`, `scaling_beta`, `scaling_data`, `raw_times_ms`, `status` (`ok`|`error`|`skip`), `error` (si `status=error`).

Solo Python: `subprocess_wall_time_ms` (tiempo total subprocess, incluye startup).

Solo Shor: `factor_found`, `success_rate`.

Solo QDisLib (cuando la ruta de cutting se ejecuta): `cutting_wall_time_ms`, `cutting_find_time_ms`, `cutting_expectation_value`.

### Ejemplo de nombre de archivo

```
results/grover_2026-05-17T14-32-01_n7.json
results/shor_2026-05-17T14-32-01_N35.json
results/run_2026-05-17T14-32-01_linux-x86_64-nvidia.json
```

### Envío opcional a base de datos

Si la variable de entorno `DB_ENDPOINT` está definida, `run.py` realiza un `POST` al endpoint con el payload JSON completo al finalizar la ejecución. El almacenamiento local en `./results/` es siempre el comportamiento base.

---

## 9. Exclusiones por plataforma

Verificado a fecha 2026-05-17. Ver `docs/framework_exclusions.md` para el análisis detallado.

| Framework | Excluido en | Motivo |
|---|---|---|
| cudaq | Linux aarch64 (nvidia/cpu), macOS x86_64, Windows (todas) | Sin wheels arm64; sin wheels Intel Mac; sin soporte Windows nativo. Además se omite bajo `--emulated` (SIGILL con AVX en QEMU) |
| qdislib | Linux aarch64 (nvidia/cpu) | `pymetis` (dep de `find_cut`) sin wheel arm64; compilación desde source requiere GCC en el contenedor |
| qcgpu | Todas las plataformas CPU, macOS (todas), Linux aarch64, Windows arm64 | Requiere OpenCL GPU; macOS deprecó OpenCL en 10.14 |
| ProjectQ | **Todas (eliminado del benchmark)** | Abandonado; sin Python 3.12 (distutils); sin SIMD ARM; sin wheels Windows |

---

## 10. Decisiones de diseño

### a. Docker como único punto de entrada

La decisión de usar Docker como única vía de ejecución elimina la necesidad de gestionar entornos de ejecución nativos en el sistema del usuario. Un único comando (`./bench` o `.\bench.ps1`) produce resultados reproducibles independientemente del sistema operativo, la distribución de Linux o las dependencias preinstaladas del usuario.

Las alternativas consideradas (launchers nativos, scripts de instalación, entornos Conda) implican mayor complejidad de mantenimiento, mayor superficie de fallos por diferencias entre sistemas, y mayor riesgo de contaminación del entorno del usuario.

### b. Comparación de frameworks, no de hardware

El objetivo del proyecto es establecer rankings relativos entre frameworks. El overhead de virtualización de Docker Desktop en macOS y Windows (~5-15% respecto a ejecución nativa) afecta a todos los frameworks en igual medida dentro de una misma pasada, por lo que los rankings relativos son válidos incluso bajo virtualización.

Esta decisión también justifica que Metal (macOS) y DirectX 12 (Windows) no se expongan a los contenedores: no son necesarios para comparar frameworks entre sí.

### c. Doble pasada CPU + GPU (y ARM nativa + AMD64 emulada)

Cuando el sistema detecta GPU NVIDIA en host AMD64, se ejecutan dos pasadas: primero con `--gpus all`, después con `--no-gpu`. `qcgpu` se omite en la pasada sin GPU.

Sobre hosts ARM64 con QEMU/binfmt_misc disponible, se ejecuta también una doble pasada: la primera nativa arm64, la segunda emulada amd64 (`--platform linux/amd64`). En la pasada emulada, `cudaq` se omite porque genera `SIGILL` al ejecutar instrucciones AVX bajo QEMU.

En AMD64 sin GPU se realiza una única pasada CPU.

### d. PLATFORM_CONFIGS estático en run.py

La lista de frameworks activos por plataforma está declarada como tabla estática en `run.py`, no se descubre dinámicamente. Esto garantiza reproducibilidad: dos ejecuciones con el mismo `platform_id` y la misma imagen Docker producen exactamente el mismo conjunto de frameworks. Esquema actual:

```python
PLATFORM_CONFIGS = {
    "macos-arm64":           [qiskit, cirq, cudaq_cpu, qdislib, rust_base],
    "macos-x86_64":          [qiskit, cirq, qdislib, rust_base],
    "linux-x86_64-nvidia":   [qiskit, cirq, cudaq_gpu, qdislib, rust_base, qcgpu],
    "linux-x86_64-cpu":      [qiskit, cirq, cudaq_cpu, qdislib, rust_base],
    "linux-aarch64-nvidia":  [qiskit, cirq, rust_base],
    "linux-aarch64-cpu":     [qiskit, cirq, rust_base],
    "windows-x86_64-gpu":    [qiskit, cirq, qdislib, rust_base, qcgpu],
    "windows-x86_64-cpu":    [qiskit, cirq, qdislib, rust_base],
    "windows-arm64-gpu":     [qiskit, cirq, qdislib, rust_base],
    "windows-arm64-cpu":     [qiskit, cirq, qdislib, rust_base],
}
```

### e. Aislamiento por subprocess

Cada framework Python se ejecuta en un **proceso hijo independiente** (`python -m python.workers.<fw>_worker`). Si un framework crashea (SIGSEGV en una extensión C, SIGILL por AVX en QEMU, OOM Killer del kernel) el orquestador captura el código de salida y continúa con el siguiente framework. Sin aislamiento, un crash de un solo framework abortaría todo el sweep.

### f. Protocolo stdin/stdout JSON

El contrato worker↔orquestador minimiza acoplamiento:

- Config al worker vía **stdin** como JSON: `{"algo", "n", "n_repetitions", "num_shots", "contributor", "cudaq_target"}`.
- Resultado: la **última línea** de `stdout` debe ser JSON válido con el `BenchmarkResult`.
- Cualquier otra línea no-JSON se reenvía al terminal del usuario como progreso (logs, barras, mensajes informativos del propio framework).

Este protocolo evita ficheros temporales, sockets o protocolos binarios; cualquier worker que pueda imprimir JSON al final es compatible.

### g. Timing Rust auto-reportado

Los binarios Rust **miden su propio `time_ms`** internamente. El valor reportado excluye el overhead de creación del subprocess Python (que puede ser >100 ms en cold start). Esto permite comparar directamente `time_ms` de Rust contra `simulation_time_ms` de Python, ya que ambos representan "tiempo real de simulación cuántica" sin overhead de proceso.

### h. QDisLib como benchmark dual

QDisLib se ejecuta sobre la **misma imagen Docker** con dos rutas de medición:

- **Ruta directa**: `search()` / `find_factor()` construyen el circuito y lo ejecutan con `AerSampler`. Mide wall time, JSD, memoria como cualquier otro framework.
- **Ruta de cutting**: `search_with_cutting()` / `find_order_with_cutting()` transpilan, llaman a `find_cut()` (API de Qdislib) y, si el resultado no es vacío, invocan `wire_cutting(backend="numpy")`. **PyCOMPSs no está instalado** en la imagen, por lo que la ejecución de subcircuitos es **local y en serie**. Se añaden campos `cutting_wall_time_ms`, `cutting_find_time_ms`, `cutting_expectation_value`.

En circuitos pequeños (n=3) o con puertas de 3+ qubits (Shor, Toffoli), `find_cut()` puede devolver `[]` — en ese caso no hay cutting y `exp_val=0.0`.

PyCOMPSs sería necesario para una distribución HPC real; el benchmark mide el cost-benefit local del cutting sin distribución.

### i. Sweep interleaved con checkpoints

Intercalar Grover y Shor por tamaño permite que, si la ejecución se interrumpe (por `q`, por `--time-budget` agotado o por crash), los checkpoints ya escritos contengan datos útiles para **ambos algoritmos en los tamaños pequeños**, en lugar de tener Grover completo y Shor vacío (o viceversa). El scaling fit `t(n) = α·2^(β·n)` se calcula al final con todos los `n` disponibles.

### j. qcgpu en builds cross-compilados

El crate `qcgpu` requiere headers OpenCL al compilarse. En cross-compilación ARM→AMD64, esos headers no están disponibles para la arquitectura objetivo. La solución es la etapa `qcgpu-amd64` (`FROM --platform=linux/amd64 rust:slim-bookworm`): se compila **nativamente** para amd64; si el host es ARM, Docker ejecuta esta etapa **vía QEMU** durante el build. Los binarios resultantes (`/qcgpu-bins/qcgpu-grover`, `/qcgpu-bins/qcgpu-shor`) se copian luego al rust-builder, garantizando que `qcgpu` aparezca en imágenes amd64 construidas desde hosts ARM, en lugar de excluirse por imposibilidad de cross-compilación.

## 11. Persistencia de datos

### Dev mode (`--dev`)
Los resultados se escriben como JSON en `results/` (volumen montado desde el host). Sin conexión de red. Los ficheros siguen el patrón `grover_{timestamp}.json` / `shor_{timestamp}.json` con checkpoints por n/N.

### Normal mode
Los resultados se envían a Supabase (tabla `benchmark_runs`) via PostgREST REST API. Los ficheros JSON locales **no se escriben**.

**Timing de envío:**
- **Incremental** — tras completar cada valor de n (Grover) o N (Shor) para todos los frameworks, se insertan las filas de esa serie (una fila por repetición individual).
- **Scaling backfill** — al finalizar cada algoritmo se hace `PATCH` a todas las filas del run para poblar `scaling_alpha`, `scaling_beta`, `scaling_data`.

**Filas por resultado:**
- `status='ok'`: una fila por repetición (expandida desde `raw_times_ms`). 8 frameworks × 5 n_values × 10 reps ≈ 400 filas por algoritmo por run.
- `status='error'`: una fila con `wall_time_ms=NULL`.
- `status='skip'`: no se envían.

**Credenciales:** `SUPABASE_URL` y `SUPABASE_KEY` leídas del fichero `.env` junto a `bench` / `bench.ps1`, inyectadas al contenedor como variables de entorno. El `.env` está en `.gitignore` y nunca se sube al repositorio.

**Schema completo:** `docs/supabase_schema.sql`.
