# Arquitectura del Sistema de Benchmarking Cuántico

**Proyecto**: TFG — Benchmarking de Algoritmos Cuánticos  
**Versión del documento**: 2.0  
**Fecha**: 2026-05-15

---

## Tabla de contenidos

1. [Resumen](#1-resumen)
2. [Flujo de usuario](#2-flujo-de-usuario)
3. [Matriz de plataformas](#3-matriz-de-plataformas)
4. [Estructura de la imagen Docker](#4-estructura-de-la-imagen-docker)
5. [Stack tecnológico](#5-stack-tecnológico)
6. [Estructura de directorios](#6-estructura-de-directorios)
7. [Formato de salida](#7-formato-de-salida)
8. [Exclusiones por arquitectura](#8-exclusiones-por-arquitectura)
9. [Decisiones de diseño](#9-decisiones-de-diseño)

---

## 1. Resumen

El sistema es una herramienta de benchmarking que mide y compara el rendimiento de distintos frameworks de simulación cuántica ejecutando dos algoritmos canónicos: la búsqueda de Grover y la factorización de Shor. La comparación abarca tanto frameworks Python (Qiskit, Cirq, ProjectQ, CUDA-Q, Qdislib) como frameworks Rust (q1tsim, quantr, quantrs2, qcgpu).

El **único punto de entrada** es Docker. El usuario ejecuta un único script (`./run-benchmark.sh` o `.\run-benchmark.ps1`) y el sistema se encarga de todo lo demás: detección de hardware, selección de plataforma, ejecución de benchmarks y almacenamiento de resultados. No se requiere instalar Python, Rust, ni ninguna dependencia en el sistema anfitrión.

El objetivo del benchmark es comparar frameworks entre sí bajo condiciones idénticas, no medir el rendimiento absoluto del hardware. Esto hace que el overhead de virtualización de Docker Desktop (~5-15%) sea irrelevante: afecta a todos los frameworks por igual y los rankings relativos permanecen válidos.

---

## 2. Flujo de usuario

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SISTEMA ANFITRIÓN (host)                         │
│                                                                     │
│  Linux/macOS:   ./run-benchmark.sh                                  │
│  Windows:       .\run-benchmark.ps1                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                   ¿nvidia-smi disponible en host?
                               │
               ┌───────────────┴───────────────┐
               │ No                            │ Sí
               ▼                               ▼
        Pasada única CPU              Dos pasadas: CPU luego GPU
        (~1x duración)                (~2x duración, aviso al usuario)
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
        docker run [--gpus all] \
          -v $(pwd)/results:/app/results \
          ghcr.io/mablospate/tfg-bench
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CONTENEDOR Docker                              │
│                                                                     │
│  docker/entrypoint.sh                                               │
│    ├─ Detecta GPU dentro del contenedor (nvidia-smi)                │
│    └─ Selecciona platform_id                                        │
│           │                                                         │
│           ▼                                                         │
│  uv run python run.py --platform <platform_id>                      │
│    ├─ Carga PLATFORM_CONFIGS[platform_id]                           │
│    ├─ Ejecuta frameworks activos para esa plataforma                │
│    │    ├─ Python: subprocess por framework                         │
│    │    └─ Rust: binario precompilado como subprocess               │
│    └─ Escribe JSON de resultados en /app/results/                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ volumen montado
                               ▼
                   ./results/<timestamp>.json  (host)
```

### Descripción paso a paso

1. El usuario ejecuta el script de entrada correspondiente a su OS.
2. El script comprueba si `nvidia-smi` responde en el host.
3. Si hay GPU: ejecuta primero la pasada CPU (`--platform linux-x86_64-cpu` o equivalente arm64) y luego la pasada GPU (`--platform linux-x86_64-nvidia`). Informa al usuario que el proceso tardará aproximadamente el doble.
4. Si no hay GPU: ejecuta una única pasada CPU.
5. Cada pasada lanza `docker run` con el volumen `./results` montado en `/app/results`. La pasada GPU añade `--gpus all`.
6. El entrypoint del contenedor (`docker/entrypoint.sh`) vuelve a ejecutar `nvidia-smi` dentro del contenedor para confirmar el acceso a la GPU y deriva el `platform_id` correcto.
7. `run.py` recibe el `platform_id` como argumento `--platform`, consulta la tabla `PLATFORM_CONFIGS` y ejecuta los frameworks activos para esa plataforma.
8. Los resultados se escriben en `/app/results/` dentro del contenedor, que está montado sobre `./results/` en el host, donde persisten al terminar el contenedor.

---

## 3. Matriz de plataformas

| Host | GPU | Platform ID | Frameworks activos | Aceleración |
|---|---|---|---|---|
| Linux x86_64 | Ninguna | `linux-x86_64-cpu` | qiskit, cirq, projectq, cudaq (CPU), qdislib + Rust | Ninguna |
| Linux x86_64 | NVIDIA | `linux-x86_64-nvidia` | mismo + qcgpu | CUDA (cudaq, quantrs2-cuQuantum, qcgpu-OpenCL) |
| Linux aarch64 | Ninguna | `linux-aarch64-cpu` | qiskit, cirq, qdislib + Rust | Ninguna |
| Linux aarch64 | NVIDIA | `linux-aarch64-nvidia` | mismo (sin qcgpu) | CUDA (cudaq no disponible en arm64) |
| Windows (cualquier) | NVIDIA + WSL2 | `linux-x86_64-nvidia` | igual que Linux NVIDIA | CUDA via GPU-PV |
| Windows (cualquier) | Sin NVIDIA | `linux-x86_64-cpu` | igual que Linux CPU | Ninguna |
| macOS (cualquier) | Metal | `linux-{arch}-cpu` | qiskit, cirq, qdislib + Rust | Ninguna |

### Windows + NVIDIA: GPU Paravirtualization (GPU-PV)

Docker Desktop en Windows usa WSL2 como hipervisor. WSL2 implementa GPU Paravirtualization (GPU-PV): el driver NVIDIA del host Windows expone el dispositivo `/dev/dxg` al kernel Linux de WSL2. Los contenedores Linux acceden a CUDA a través de `/usr/lib/wsl/lib/libcuda.so.1`, que actúa como proxy hacia el driver Windows.

El script `run-benchmark.ps1` detecta si `nvidia-container-toolkit` está instalado en el entorno WSL2 y lo instala automáticamente si es necesario.

**Requisitos**: Windows 10 21H2+ o Windows 11, driver NVIDIA 471.41+, Docker Desktop con integración WSL2 activa.

### macOS: solo CPU

El framework de virtualización de Apple (`Hypervisor.framework`) no expone el bus PCIe ni la GPU Metal a las máquinas virtuales. Por tanto, los contenedores Linux que ejecuta Docker Desktop en macOS no tienen acceso a la GPU, independientemente de que el Mac tenga Apple Silicon o Intel.

Esto es aceptable para los objetivos del proyecto: el benchmark compara frameworks entre sí, no el rendimiento absoluto del hardware. En Apple Silicon, la imagen arm64 se ejecuta nativamente sin necesidad de emulación Rosetta.

---

## 4. Estructura de la imagen Docker

La imagen se construye mediante un `Dockerfile` multi-stage para minimizar el tamaño final y separar las dependencias de compilación de las de ejecución.

```
Dockerfile
│
├── Stage 1: rust-builder
│     Base: rust:latest
│     Compila todos los crates Rust (grover, shor para cada framework)
│     Produce binarios en /usr/local/bin/
│
├── Stage 2: cpu   (imagen de producción CPU)
│     Base: python:3.12-slim
│     Copia binarios Rust desde rust-builder
│     Instala dependencias Python con uv
│       · En amd64: uv sync --extra x86only  (incluye projectq, cudaq-cpu)
│       · En arm64: sin extras de arquitectura
│     Entrypoint: docker/entrypoint.sh
│
└── Stage 3: cuda  (imagen de producción GPU, extiende cpu)
      Base: nvidia/cuda:... (imagen oficial NVIDIA)
      Misma config que cpu
      Añade: uv sync --extra gpu  (cuda-quantum-cu13 y dependencias CUDA)
```

La imagen se publica en `ghcr.io/mablospate/tfg-bench`. El entrypoint selecciona en tiempo de ejecución qué plataforma activar basándose en la presencia de GPU, sin necesidad de imágenes separadas para CPU y GPU (la imagen `cuda` contiene ambos conjuntos de dependencias).

---

## 5. Stack tecnológico

### Frameworks Python

| Framework | Versión | Restricción de plataforma |
|---|---|---|
| Qiskit | 2.0 | Ninguna |
| Cirq | >=1.6.1 | Ninguna |
| ProjectQ | 0.8.0 | Solo x86_64 (sin wheels arm64) |
| CUDA-Q (cudaq) | >=0.14.0 | Solo x86_64 (sin wheels arm64) |
| Qdislib | >=1.0.0 | Ninguna (pure Python) |

### Frameworks Rust

| Crate | Versión | Restricción de plataforma |
|---|---|---|
| q1tsim | 0.5.0 | Ninguna |
| quantr | 0.6.0 | Ninguna |
| quantrs2 | 0.1.3 | Aceleración CUDA solo en x86_64 |
| qcgpu | 0.1.0 | Solo Linux x86_64 con NVIDIA (OpenCL/CUDA) |

### Algoritmos implementados

- **Búsqueda de Grover**: búsqueda no estructurada en un espacio de N elementos. Implementado en todos los frameworks.
- **Factorización de Shor**: algoritmo de factorización cuántica. Implementado en todos los frameworks. La implementación incluye la estimación de fase cuántica (QPE) y el módulo de permutación modular.

### Herramientas de soporte

| Herramienta | Uso |
|---|---|
| `uv` | Gestión de dependencias Python y ejecución dentro del contenedor |
| Docker multi-stage | Construcción reproducible, imagen mínima de producción |
| `nvidia-smi` | Detección de GPU en host y en contenedor |
| `nvidia-container-toolkit` | Acceso GPU en contenedores (auto-instalado en WSL2 si falta) |

---

## 6. Estructura de directorios

```
TFG/
  run.py                    — orquestador del benchmark (requiere --platform)
  run-benchmark.sh          — punto de entrada Linux/macOS
  run-benchmark.ps1         — punto de entrada Windows (configura GPU automáticamente)
  Dockerfile                — construcción multi-stage (rust-builder / cpu / cuda)
  docker/
    entrypoint.sh           — detección de GPU en contenedor, selección de platform_id
  docker-compose.yml        — alias de conveniencia para desarrollo local
  pyproject.toml            — declaración de dependencias Python (uv)
  uv.lock                   — lockfile reproducible de dependencias
  python/                   — implementaciones Python por framework
    {framework}/
      grover.py
      shor/
        shor.py
  rust/                     — implementaciones Rust por crate
    {crate}/
      src/
        grover.rs           — módulo de librería
        shor/mod.rs         — módulo de librería
        bin/
          grover.rs         — wrapper ejecutable
          shor.rs           — wrapper ejecutable
      Cargo.toml
  tests/                    — tests unitarios e integración
    {framework}/
      test_grover.py
      shor/test_shor.py
  results/                  — salida de benchmarks (en .gitignore)
  docs/
    framework_exclusions.md — justificación de exclusiones por arquitectura
    architecture.md         — este documento
```

---

## 7. Formato de salida

Cada ejecución produce un archivo JSON en `./results/`. El nombre del archivo incluye el timestamp de la ejecución.

### Campos del JSON

| Campo | Tipo | Descripción |
|---|---|---|
| `platform_id` | string | Identificador de plataforma (`linux-x86_64-cpu`, etc.) |
| `gpu_enabled` | boolean | Si la pasada usó aceleración GPU |
| `benchmark_image` | string | Tag de la imagen Docker usada |
| `hardware` | object | Información del hardware: CPU, RAM, modelo de GPU si aplica |
| `results` | array | Array de resultados por framework |
| `results[].framework` | string | Nombre del framework |
| `results[].algorithm` | string | `grover` o `shor` |
| `results[].wall_time_ms` | number | Tiempo de ejecución en milisegundos |
| `results[].status` | string | `ok`, `skip`, o `error` |

### Envío opcional a base de datos

Si la variable de entorno `DB_ENDPOINT` está definida, `run.py` realiza un `POST` al endpoint con el payload JSON completo al finalizar la ejecución. Esta variable no se define por defecto; el almacenamiento local en `./results/` es siempre el comportamiento base.

---

## 8. Exclusiones por arquitectura

Verificado en PyPI a fecha 2026-05-15. Ver `docs/framework_exclusions.md` para el análisis detallado.

### arm64 (linux-aarch64-*)

| Framework | Motivo de exclusión |
|---|---|
| cudaq | Solo distribución como source (`.tar.gz`); sin wheels para arm64 en PyPI |
| projectq | Solo wheels `x86_64`; compilación desde source requiere extensiones C no portadas |

Todos los demás frameworks disponen de wheels arm64 o son pure Python y se incluyen sin restricciones.

### qcgpu (linux-x86_64-nvidia únicamente)

qcgpu usa OpenCL como backend de aceleración, que en Linux requiere drivers NVIDIA y hardware compatible. Se excluye de todas las plataformas CPU y de arm64 donde la presencia de drivers OpenCL NVIDIA no está garantizada.

---

## 9. Decisiones de diseño

### Docker como único punto de entrada

La decisión de usar Docker como única vía de ejecución elimina la necesidad de gestionar entornos de ejecución nativos en el sistema del usuario. Un único comando (`./run-benchmark.sh`) produce resultados reproducibles independientemente del sistema operativo, la distribución de Linux o las dependencias preinstaladas del usuario.

Las alternativas consideradas (launchers nativos, scripts de instalación, entornos Conda) implican mayor complejidad de mantenimiento, mayor superficie de fallos por diferencias entre sistemas, y mayor riesgo de contaminación del entorno del usuario.

### Comparación de frameworks, no de hardware

El objetivo del proyecto es establecer rankings relativos entre frameworks de simulación cuántica. El overhead de virtualización de Docker Desktop en macOS y Windows (~5-15% respecto a ejecución nativa) afecta a todos los frameworks en igual medida dentro de una misma pasada, por lo que los rankings relativos son válidos incluso bajo virtualización.

Esta decisión también justifica que Metal (macOS) y DirectX 12 (Windows) no se expongan a los contenedores: no son necesarios para comparar frameworks entre sí, y su ausencia simplifica enormemente la arquitectura.

### Doble pasada CPU + GPU

Cuando el sistema detecta una GPU NVIDIA en el host, ejecuta automáticamente dos pasadas independientes: una en modo CPU y otra en modo GPU. Esto permite al usuario obtener en una sola invocación del script tanto el rendimiento base como el rendimiento acelerado por GPU, maximizando la información recogida sin requerir ejecuciones manuales adicionales.

### PLATFORM_CONFIGS estático en run.py

La lista de frameworks activos por plataforma está declarada como una tabla estática (`PLATFORM_CONFIGS`) en `run.py`, no se descubre dinámicamente en tiempo de ejecución. Esto garantiza reproducibilidad: dos ejecuciones con el mismo `platform_id` y la misma imagen Docker producen exactamente los mismos frameworks, sin dependencias de detección de estado del sistema.

```
PLATFORM_CONFIGS = {
    "linux-x86_64-cpu":    [qiskit, cirq, projectq, cudaq_cpu, qdislib, rust_all],
    "linux-x86_64-nvidia": [qiskit, cirq, projectq, cudaq_gpu, qdislib, rust_all, qcgpu],
    "linux-aarch64-cpu":   [qiskit, cirq, qdislib, rust_all],
    "linux-aarch64-nvidia":[qiskit, cirq, qdislib, rust_all],
}
```

### Sin runners nativos

No existe un camino de ejecución fuera de Docker. Esta restricción simplifica el árbol de código (no hay lógica de detección de entorno nativo), elimina una clase entera de bugs relacionados con versiones de Python o Rust del sistema, y hace que el entorno de CI/CD sea idéntico al entorno del usuario final.
