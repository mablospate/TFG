# Compatibilidad de Plataformas por Framework

> Exploración generada el 2026-05-12. Fuentes: documentación oficial, GitHub, PyPI/crates.io, arXiv (2024-2026).

## Matriz resumen

Leyenda: ✅ OK · ⚠️ DEGRADADO · ❌ NO SOPORTADO · 🔬 Sin CI oficial

| Framework | macOS arm64 | macOS x86_64 | Linux x86_64 | Linux aarch64 | Windows x86_64 |
|---|---|---|---|---|---|
| **qiskit-aer** | ✅ | ✅ | ✅ (mejor) | ✅ | ✅ |
| **cirq** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **projectq** | ⚠️ ~790× más lento | ⚠️ Sin AVX | ⚠️ Sin batching | ⚠️ Sin SIMD | ❌ Sin wheel |
| **cudaq** | ⚠️ CPU-only | ❌ | ✅ GPU+CPU | ✅ GPU+CPU | ❌ (WSL2) |
| **qdislib** | ⚠️ Stub=Qiskit | ⚠️ Stub=Qiskit | ⚠️ Stub=Qiskit | ⚠️ Stub=Qiskit | ⚠️ Stub=Qiskit |
| **quantrs2** | ✅ (Alpha) | ✅ (Alpha) | ✅ (Alpha) | ✅ (cross) | ✅ (Alpha) |
| **roqoqo** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **qip/RustQIP** | 🔬 | 🔬 | 🔬 | 🔬 | ❌ Sin CI |
| **quantr** | 🔬 ≤16 qubits | 🔬 ≤16 qubits | ⚠️ ≤16 qubits | 🔬 ≤16 qubits | 🔬 ≤16 qubits |
| **q1tsim** | 🔬 | 🔬 | 🔬 | 🔬 | 🔬 |
| **qcgpu** | ❌ Sin OpenCL | ❌ Sin OpenCL | ✅ (OpenCL) | ❌ OpenCL ARM | ✅ (OpenCL) |

---

## Python Frameworks

### qiskit-aer (v0.17.2, sep 2025)

**Estado general:** El simulador más portable del proyecto. Wheels precompiladas para todas las plataformas principales.

| Plataforma | Soporte | Notas |
|---|---|---|
| macOS arm64 | ✅ | Wheels nativas desde v0.11.0. macOS 11.0+. |
| macOS x86_64 | ✅ | macOS 10.9+. |
| Linux x86_64 | ✅ | Mejor opción: GPU via `qiskit-aer-gpu` (CUDA). |
| Linux aarch64 | ✅ | manylinux, glibc 2.17+. |
| Windows x86_64 | ✅ | Wheel estándar. |

**Simulación:** Statevector C++ con `Transformer` SIMD/AVX2 en x86_64. En arm64 se compila con las optimizaciones disponibles del toolchain (no documentado si incluye NEON). OpenMP disponible en todas las plataformas.

**GPU:** `qiskit-aer-gpu` (cuStateVec) solo en **Linux x86_64**. macOS y Windows sin GPU.

**Cuándo no usar:** No hay plataforma donde deba evitarse. En benchmarks comparativos es la referencia base.

**Fuentes:** [PyPI qiskit-aer 0.17.2](https://pypi.org/project/qiskit-aer/) · [Docs oficiales v0.17.1](https://qiskit.github.io/qiskit-aer/getting_started.html)

---

### cirq (v1.6.1, ago 2024)

**Estado general:** Framework puramente Python + NumPy. Totalmente agnóstico a la arquitectura.

| Plataforma | Soporte | Notas |
|---|---|---|
| macOS arm64 | ✅ | Usa Apple Accelerate vía NumPy. |
| macOS x86_64 | ✅ | |
| Linux x86_64 | ✅ | |
| Linux aarch64 | ✅ | |
| Windows x86_64 | ✅ | |

**Simulación:** `cirq.Simulator()` evoluciona el statevector con operaciones NumPy (einsum/matmul en C). Shots múltiples en una sola llamada — muestrea la distribución final sin re-simular. No hay código SIMD propio; la aceleración viene del BLAS del sistema.

**Rendimiento vs arquitectura:** Sin brecha documentada entre x86_64 y arm64. Al ser Python puro en el bucle externo, el rendimiento es consistente entre plataformas a igual frecuencia de CPU.

**Regresiones recientes:** Python mínimo subió a 3.11 en v1.6.0 (jul 2024).

**Cuándo no usar:** No hay plataforma donde deba evitarse.

**Fuentes:** [Cirq Install Docs (2025-05-16)](https://quantumai.google/cirq/start/install) · [GitHub Releases](https://github.com/quantumlib/cirq/releases)

---

### projectq (v0.8.0, oct 2022 — abandonware)

**⚠️ Advertencia crítica de rendimiento:** ProjectQ ejecuta cada shot en un bucle Python que recrea un `MainEngine` completo, incluyendo asignación de qubits, aplicación de todas las puertas y medición. En 1024 shots esto supone 1024 ciclos de vida completos del motor.

**Resultado empírico** (n=5, target=5, 1024 shots, macOS arm64, Python 3.11):
- ProjectQ: **10.994 ms** mediana
- Qiskit-Aer: **13,87 ms** mediana
- **Ratio: ~793×** más lento

Este ratio **no es una anomalía de ARM64** — es estructural en cualquier plataforma. En x86_64 sería algo mejor por AVX2, pero el cuello de botella es el overhead Python por shot.

| Plataforma | Soporte | Notas |
|---|---|---|
| macOS arm64 | ⚠️ | Sin SIMD ARM (sin Neon/SVE). C++ compilado sin vectorización ARM. |
| macOS x86_64 | ⚠️ | AVX disponible pero bucle Python domina igualmente. |
| Linux x86_64 | ⚠️ | Mismo problema. Marginalmente más rápido que arm64 por AVX2. |
| Linux aarch64 | ⚠️ | Sin SIMD ARM documentado. |
| Windows x86_64 | ❌ | Sin wheel precompilada. Build de C++ compleja, no documentada para Windows. |

**Estado del proyecto:** Último release: v0.8.0 (18 oct 2022). Sin commits desde entonces. Issue comunitario de optimización (#476) cerrado sin resolución (2024). No hay roadmap de SIMD ARM.

**Cuándo no usar:** No hay plataforma donde deba ignorarse del todo (es parte del benchmark), pero los resultados **no son comparables** con los demás frameworks en latencia. El JSD sí es comparable. Documentar siempre el contexto al presentar los datos.

**Fuentes:** [GitHub Releases](https://github.com/ProjectQ-Framework/ProjectQ/releases) · [Issue #444 Apple Silicon fix](https://github.com/ProjectQ-Framework/ProjectQ/issues/444)

---

### cudaq / CUDA-Q (v0.14.0, mar 2026)

**Estado general:** El framework más potente con GPU, pero con la distribución de plataformas más restrictiva.

| Plataforma | Soporte | Target disponible | Notas |
|---|---|---|---|
| macOS arm64 | ⚠️ | `qpp-cpu` | CPU-only. Soportado desde v0.14 (mar 2026). ~53 ms medido. |
| macOS x86_64 | ❌ | — | Sin wheels. Sin soporte oficial. |
| Linux x86_64 | ✅ | `nvidia` (GPU) + `qpp-cpu` | Mejor rendimiento. CUDA 12 o 13. |
| Linux aarch64 | ✅ | `nvidia` (GPU) + `qpp-cpu` | Full support desde v0.14. |
| Windows x86_64 | ❌ | — | Sin soporte nativo. Usar WSL2 → Linux path. |

**Simulación:** MLIR/LLVM IR compilado (JIT). `qpp-cpu` usa Q++ C++ con OpenMP. `nvidia` usa cuStateVec (GPU). Shots en batch sin retorno a Python entre iteraciones.

**En este proyecto (macOS arm64):** Se detecta `gpu_model=None` en hardware.py → `run.py` usa `cudaq.set_target("qpp-cpu")` automáticamente.

**Cuándo no usar:** macOS x86_64 (Intel Mac) — no hay soporte. Windows nativo — usar WSL2. En macOS arm64, resultados válidos pero sin aceleración GPU; los tiempos no son comparables con una máquina Linux+NVIDIA.

**Fuentes:** [CUDA-Q Local Installation Docs](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html) · [CUDA-Q 0.14 Release Blog (2026-03-16)](https://nvidia.github.io/cuda-quantum/blogs/blog/2026/03/16/cudaq-0.14/)

---

### qdislib / QDisLib (v1.0.0, 2025)

**⚠️ Advertencia de implementación:** El circuit cutting de QDisLib **no está activo** en la implementación actual del proyecto. `python/qdislib/grover.py` comprueba `import Qdislib` pero en ambas ramas (con y sin QDisLib) ejecuta directamente `AerSampler.run()`. Los resultados son **estadísticamente idénticos a qiskit** (~14 ms).

| Plataforma | Soporte (stub) | Soporte (real, HPC) |
|---|---|---|
| macOS arm64 | ⚠️ Funciona como Qiskit | ❌ No documentado por BSC |
| macOS x86_64 | ⚠️ Funciona como Qiskit | ❌ No documentado |
| Linux x86_64 | ⚠️ Funciona como Qiskit | ✅ Target primario (MareNostrum 5) |
| Linux aarch64 | ⚠️ Funciona como Qiskit | ❌ No documentado |
| Windows x86_64 | ⚠️ Funciona como Qiskit | ❌ No documentado |

**QDisLib real requiere:** PyCOMPSs ≥ 3.3 (instalación separada con dependencias nativas del sistema). Diseñado para clusters HPC Linux x86_64.

**Referencia científica:** Tejedor et al., "Distributed Quantum Circuit Cutting for Hybrid Quantum-Classical High-Performance Computing", arXiv:2505.01184 (may 2025).

**Cuándo no usar como métrica independiente:** En la implementación actual, QDisLib y Qiskit producen resultados equivalentes. No tiene sentido comparar sus tiempos como si fueran independientes. Documentar esto en la tesis.

**Fuentes:** [GitHub bsc-wdc/qdislib](https://github.com/bsc-wdc/qdislib) · [Docs v1.0.0](https://qdislib.readthedocs.io/en/latest/) · [arXiv:2505.01184](https://arxiv.org/abs/2505.01184)

---

## Rust Frameworks (pendientes de implementar)

### quantrs2 (v0.1.3 Alpha, mar 2026)

**Estado:** Activo, pero Alpha con ~1.095 descargas totales. Elegido como implementación primaria Rust en este proyecto.

| Plataforma | Soporte | CI |
|---|---|---|
| macOS arm64 | ✅ | CI explícito (`aarch64-apple-darwin`) |
| macOS x86_64 | ✅ | CI explícito (`x86_64-apple-darwin`) |
| Linux x86_64 | ✅ | CI (manylinux) |
| Linux aarch64 | ✅ | CI (cross-compilation) |
| Windows x64 | ✅ | CI |

**Backends de simulación disponibles:** statevector (denso), MPS (matrix product states), estabilizador/Clifford, tensor network, decision diagrams, path integral, Monte Carlo cuántico.

**SIMD/paralelismo:** Via `scirs2-core` con features `simd` y `parallel`. GPU opcional via `wgpu` o `cuQuantum`.

**Política:** Pure Rust (sin dependencias C/FFI) → builds multiplataforma fiables sin problemas de toolchain.

**Riesgo:** Estado Alpha. Si la API cambia entre RC y estable, el código necesitará adaptación. Alternativa: roqoqo.

**Fuentes:** [crates.io/quantrs2](https://crates.io/crates/quantrs2) · [GitHub cool-japan/quantrs](https://github.com/cool-japan/quantrs)

---

### roqoqo (v1.21.0, ago 2025)

**Estado:** Estable. 233.737 descargas. Financiado por UE (PlanQK, QSolid, PhoQuant). Opción de contingencia si quantrs2 presenta problemas.

| Plataforma | Soporte | CI |
|---|---|---|
| macOS arm64 | ✅ | CI explícito (desde v1.19.0) |
| macOS x86_64 | ✅ | CI |
| Linux x86_64 | ✅ | CI |
| Linux aarch64 | ✅ | CI (cross) |
| Windows x64 | ✅ | CI |

**Importante:** roqoqo es una **representación de circuitos (IR)**, no un simulador. Necesita un backend:
- `qoqo_quest` → QuEST C library (dependencia C, puede complicar cross-compilation)
- `qoqo_qiskit` → delega a Qiskit
- `qoqo_for_braket` → Amazon Braket

**Cuándo preferirlo a quantrs2:** Si se necesita estabilidad de API garantizada o integración con QPUs reales via Braket.

**Fuentes:** [crates.io/roqoqo](https://crates.io/crates/roqoqo) · [GitHub HQSquantumsimulations/qoqo](https://github.com/HQSquantumsimulations/qoqo)

---

### qip / RustQIP (v1.5.0, dic 2025)

**Estado:** Mantenimiento de baja actividad. Sin CI. Último commit dic 2025. 26+ issues abiertas sin triage.

| Plataforma | Soporte | CI |
|---|---|---|
| macOS arm64 | 🔬 Probable | Sin CI |
| macOS x86_64 | 🔬 Probable | Sin CI |
| Linux x86_64 | 🔬 Probable | Sin CI |
| Linux aarch64 | 🔬 Probable | Sin CI |
| Windows x86_64 | ❌ | Sin CI, sin build verificado |
| Windows arm64 | ❌ | Sin CI, sin build verificado |

**Simulación:** Statevector disperso, grafo de circuito con garantías del borrow checker. Paralelismo via rayon (feature por defecto). Sin SIMD, sin GPU.

**Cuándo no usar:** Windows (cualquier arquitectura) — sin evidencia de build funcional. En Linux/macOS se incluye como best-effort con anotación explícita.

**Fuentes:** [crates.io/qip](https://crates.io/crates/qip) · [GitHub Renmusxd/RustQIP](https://github.com/Renmusxd/RustQIP)

---

### quantr (v0.6.0, jul 2024)

**Estado:** Bajo mantenimiento. Último release julio 2024. CI únicamente en `ubuntu-latest`; sin CI en macOS ni Windows.

| Plataforma | Soporte | CI |
|---|---|---|
| macOS arm64 | 🔬 Probable | Sin CI upstream |
| macOS x86_64 | 🔬 Probable | Sin CI upstream |
| Linux x86_64 | ⚠️ | CI en ubuntu-latest |
| Linux aarch64 | 🔬 Probable | Sin CI upstream |
| Windows x86_64 | 🔬 Probable | Sin CI upstream |

**Simulación:** Statevector denso. La memoria crece como 2^n amplitudes complejas. Límite práctico: **~16 qubits**. No hay optimizaciones para state vectors grandes.

**Restricción en este proyecto:** Excluido de benchmarks con n > 16 qubits. Solo válido para circuitos pequeños.

**Build:** Pure Rust. Sin dependencias C.

**Fuentes:** [crates.io/quantr](https://crates.io/crates/quantr)

---

### q1tsim (v0.5.0, nov 2019)

**Estado:** Abandonado. Último release noviembre 2019 (6+ años sin actividad). Sin CI.

| Plataforma | Soporte | CI |
|---|---|---|
| Todas | 🔬 Desconocido | Sin CI |

**Problema de dependencias:** Requiere `ndarray 0.12` y `rand 0.4`. En un workspace Cargo moderno (ndarray 0.15+, rand 0.8+), Cargo compilará ambas versiones en paralelo (son semver-incompatibles), lo cual aumenta tiempos de compilación. Pueden producirse conflictos de resolución que requieran `[patch.crates-io]`.

**Simulación:** Statevector. Diseño para circuitos de 1 qubit (nombre: "one-qubit-at-a-time simulator"). Sin GPU, sin SIMD.

**Incluido para:** completitud histórica. Si la compilación falla en el workspace, se excluye con nota explícita.

**Build:** Pure Rust, pero árbol de dependencias obsoleto.

**Fuentes:** [crates.io/q1tsim](https://crates.io/crates/q1tsim)

---

### qcgpu (v0.1.0, abr 2018)

**Estado:** Abandonado. Único release: abril 2018 (8+ años sin actividad).

| Plataforma | Soporte | Notas |
|---|---|---|
| macOS arm64 | ❌ | OpenCL eliminado de macOS desde 10.14 Mojave |
| macOS x86_64 | ❌ | OpenCL eliminado de macOS desde 10.14 Mojave |
| Linux x86_64 | ✅ | OpenCL disponible vía drivers NVIDIA/AMD/Intel |
| Linux aarch64 | ❌ | OpenCL en ARM es dependiente del driver; no universal |
| Windows x86_64 | ✅ | OpenCL disponible vía drivers GPU |
| Windows arm64 | ❌ | Disponibilidad de OpenCL incierta |

**Dependencia obligatoria:** OpenCL (no opcional). Sin fallback CPU. La simulación completa ocurre en GPU via kernels OpenCL.

**Por qué está incluido:** Valor histórico. qcgpu (2018) fue uno de los primeros simuladores cuánticos Rust con aceleración GPU. Punto de comparación para evaluar la evolución del ecosistema.

**Build:** No es pure Rust — tiene bindings C via el crate `ocl`. Requiere OpenCL headers y runtime en tiempo de compilación.

**Fuentes:** [crates.io/qcgpu](https://crates.io/crates/qcgpu)

---

---

## Aceleración por Hardware por Plataforma

### macOS arm64

| Mecanismo | Frameworks beneficiados | Notas |
|---|---|---|
| **Metal (wgpu)** | quantrs2 (`features=["gpu"]`) | GPU compute via wgpu → Metal backend. Activación automática en macOS. |
| **Apple Accelerate (AMX)** | Cirq, Qiskit-Aer CPU, cudaq qpp-cpu | NumPy 2.x enlaza con Accelerate BLAS en macOS arm64. Las operaciones matriciales usan AMX (Apple Matrix Extensions) de forma transparente. |
| **OpenMP** | Qiskit-Aer, cudaq qpp-cpu | Paralelismo multi-core en el simulador C++. |

**Nota Accelerate:** El beneficio de AMX es automático si se usa la distribución Python estándar para macOS arm64 (NumPy 2.x). No requiere configuración explícita. Cirq y Qiskit (CPU) se benefician del mismo BLAS.

---

### Linux x86_64 + NVIDIA GPU

| Mecanismo | Frameworks beneficiados | Notas |
|---|---|---|
| **CUDA / cuStateVec** | cudaq (`nvidia` target), qiskit-aer-gpu | Simulación statevector en GPU NVIDIA. cudaq requiere CUDA 12 o 13. |
| **Vulkan compute (wgpu)** | quantrs2 (`features=["gpu"]`) | Alternativa a CUDA via wgpu Vulkan backend. Disponible junto con cuQuantum. |
| **cuQuantum** | quantrs2 (`features=["cuquantum"]`) | Integración directa con la librería cuQuantum de NVIDIA. Requiere CUDA toolkit + cuQuantum instalados. |
| **OpenMP** | Qiskit-Aer, cudaq qpp-cpu | Multi-core CPU. |

---

### Linux AMD GPU (ROCm)

| Mecanismo | Frameworks beneficiados | Notas |
|---|---|---|
| **Vulkan compute (wgpu)** | quantrs2 (`features=["gpu"]`) | wgpu soporta Vulkan en AMD; funciona con drivers Mesa/AMDVLK. |
| **ROCm + qiskit-aer** | — | **No incluido.** Requiere compilar qiskit-aer desde fuente con ROCm BLAS. Sin wheel oficial en PyPI. |

**Nota:** quantrs2 es el único framework con aceleración GPU AMD lista para usar en la configuración estándar del proyecto. El resto de frameworks usa CPU paths en hardware AMD.

---

### Windows x86_64

| Mecanismo | Frameworks beneficiados | Notas |
|---|---|---|
| **DirectX 12 (wgpu)** | quantrs2 (`features=["gpu"]`) | wgpu usa DX12 en Windows. Activación automática con `features=["gpu"]`. |
| **CPU (OpenMP)** | Qiskit-Aer, cirq | Todos los demás frameworks usan CPU en Windows. |

---

### Runners CPU-only (sin GPU disponible)

Todos los frameworks usan únicamente rutas CPU. Notas específicas:

- **cudaq:** usa el target `qpp-cpu` (Q++ C++ + OpenMP + LLVM JIT). Detectado automáticamente en hardware.py cuando `gpu_model=None`.
- **quantrs2:** features de GPU deshabilitadas explícitamente (`features=["gpu"]` omitido) para garantizar reproducibilidad.
- **qiskit-aer:** statevector C++ con AVX2 (x86_64) o las optimizaciones disponibles del toolchain (arm64).
- **cirq:** NumPy puro; rendimiento determinado por el BLAS del sistema.

---

## Resumen de decisiones para el TFG

| Situación | Recomendación |
|---|---|
| Benchmarks en macOS arm64 (desarrollo) | Todos los frameworks OK excepto cudaq sin GPU. ProjectQ ~800× más lento — documentar explícitamente. |
| Benchmarks en Linux x86_64 + NVIDIA (CI/servidor) | Todos los frameworks en su mejor rendimiento. cudaq con target `nvidia`. |
| Benchmarks en Windows | Solo qiskit, cirq, qdislib. ProjectQ sin wheel. cudaq solo via WSL2. |
| Comparación cuantitativa de tiempos | Excluir ProjectQ de comparativas de latencia (bucle Python no representativo). Incluirlo solo para JSD/precisión. |
| Comparación cuantitativa qdislib vs qiskit | Documentar que son estadísticamente equivalentes hasta que se implemente el circuit cutting real. |
| Elección de framework Rust | quantrs2 primero (más backends, SIMD, GPU). roqoqo como fallback estable. qip desaconsejado. |
