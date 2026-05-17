# Framework Exclusions and Limitations

> Document last updated 2026-05-17. Records which frameworks are excluded from specific platforms and why.

## 1. Introduction — Criteria for Exclusion

Frameworks or platform combinations are excluded from benchmarks when one or more of the following conditions apply:

1. **Abandonment**: No release or commit activity for multiple years, combined with known incompatibilities with current toolchains.
2. **Broken dependencies**: The package cannot be installed or built on the target platform due to missing wheels, removed standard library modules, or missing system libraries.
3. **Platform restrictions by the vendor**: The framework vendor explicitly does not support the platform (no wheels published, no CI, no documentation).
4. **Non-comparable results**: The framework produces results on the platform that are structurally incomparable to other frameworks — not due to algorithmic differences but due to infrastructure overhead (e.g., per-shot Python loop).

Frameworks that are included with caveats are marked as **degraded** rather than excluded; their data is reported with explicit annotations.

---

## 2. Python Frameworks

### projectq (v0.8.0) — Eliminado del benchmark (histórico)

> **Estado actual: ProjectQ ha sido eliminado de todos los `PLATFORM_CONFIGS` y no participa en ninguna pasada del benchmark.** Esta sección se conserva como referencia histórica.

**Last PyPI release:** v0.8.0, October 2022. No commits since then. Effectively abandoned.

**Python version constraint:** projectq requires Python 3.11. Python 3.12 removed `distutils` from the standard library. The projectq C++ extension build fails on Python 3.12+ with:

```
AttributeError: 'Compiler' object has no attribute 'dry_run'
```

The C++ simulator (`_cppsim`) cannot be built without `distutils`. The benchmark image is based on Python 3.12, so projectq cannot be installed in the runtime environment.

**ARM performance degradation (macOS arm64 / Linux aarch64):** The `_cppsim` extension compiles without SIMD ARM optimizations. There is no NEON or SVE vectorization in the projectq codebase. On arm64 hosts, performance was approximately 800× slower than native SIMD simulators. The dominant cost was the per-shot Python overhead (projectq creates a full `MainEngine` cycle per shot); the absence of ARM SIMD vectorization made the gap wider than on x86_64 with AVX2.

**Windows (x86_64 and arm64):** No precompiled wheel on PyPI. Building the C++ extension on Windows requires a manually configured MSVC toolchain with steps not documented by the projectq project.

**Razón de la eliminación:** la combinación de abandono, incompatibilidad con Python 3.12, ausencia de SIMD ARM y falta de wheels Windows hace que ProjectQ no pueda ejecutarse en la imagen Docker actual (Python 3.12, base multi-arquitectura). Mantenerlo solo en x86_64 con Python 3.11 desviaba la base de la imagen del resto del benchmark sin aportar valor comparativo.

**When this might change:**
- If the projectq project merges an ARM SIMD (NEON/SVE) pull request.
- If projectq adds Python 3.12 compatibility via a `setuptools` `distutils` shim.
- If a third party publishes Windows wheels.

---

### cudaq / CUDA-Q (v0.14.0) — Excluded on macOS x86_64, Linux aarch64, and Windows

**Linux aarch64 — excluded:**

No `manylinux_aarch64` wheels are published for `cuda-quantum-cu13` (the real installable package). cudaq is only available on Linux x86_64.

**macOS x86_64 (Intel Mac) — excluded completely:**

As of May 2026, only `macosx_arm64` wheels are published. No `macosx_x86_64` wheels exist for any version from v0.14 onward. Intel Mac support was not announced and is not expected given Apple's transition to Apple Silicon.

**macOS arm64 (Apple Silicon) — included:**

Wheels are published for `macosx_arm64`. cudaq is active in the `macos-arm64` platform config.

**Windows — excluded completely:**

There is no native Windows support for cudaq. The official workaround is WSL2 (Linux subsystem), which is not applicable to a native Windows benchmark runner. No `win_amd64` or `win_arm64` wheels are published on PyPI.

**ARM host running emulated AMD64 (QEMU) — excluded at runtime:**

Even though cudaq is configured for `linux-x86_64-*` platforms, when an ARM host runs the AMD64 pass via QEMU (`--platform linux/amd64`), cudaq is skipped automatically. cudaq's AVX-optimized code paths trigger `SIGILL` under QEMU emulation.

**When this might change:**
- macOS x86_64: only if NVIDIA publishes Intel Mac wheels. Very unlikely.
- Windows: only if NVIDIA adds native Windows support.
- Linux aarch64: only if NVIDIA publishes arm64 wheels.

---

### qdislib / QDisLib — Excluded on Linux aarch64

**Linux aarch64 (nvidia and cpu) — excluded:**

QDisLib's `find_cut` function depends on `pymetis`, a Python binding for the METIS graph partitioning library. `pymetis` does **not** publish a `manylinux_aarch64` wheel on PyPI. Installing it would require building from source inside the container, which in turn requires GCC and METIS development headers — these are not part of the slim arm64 base image, and adding them would significantly increase image size for a single dependency.

In `pyproject.toml`, qdislib is declared under the `x86only` extra; `uv sync --extra x86only` is only invoked in amd64 image builds. On Windows arm64, qdislib **is** included because the platform extra rules differ (Windows installs use precompiled paths that avoid pymetis source builds for the basic API; the cutting path then degrades gracefully).

**All other platforms — included:**

QDisLib is in `macos-arm64`, `macos-x86_64`, `linux-x86_64-nvidia`, `linux-x86_64-cpu`, `windows-x86_64-gpu`, `windows-x86_64-cpu`, `windows-arm64-gpu`, and `windows-arm64-cpu`.

**When this might change:**
- If `pymetis` publishes a `manylinux_aarch64` wheel on PyPI.
- If the arm64 base image is switched to one that includes GCC and METIS dev headers (would increase image size).

---

### qiskit-aer on Windows arm64 — Excluded

`qiskit-aer` does not publish a wheel for `win_arm64` on PyPI. Installation on Windows arm64 would require building from source, which is not part of the standard benchmark setup.

**When this might change:** If Qiskit adds Windows arm64 to its CI matrix and publishes the corresponding wheel.

---

### qiskit-aer-gpu with ROCm (AMD GPU) — Not distributed

`qiskit-aer-gpu` on PyPI uses CUDA (cuStateVec). There is no official PyPI wheel built against ROCm. AMD GPU acceleration with qiskit-aer requires building from source with ROCm BLAS libraries. This is not included in any benchmark runner configuration.

**What would be required:** A user would need to clone the qiskit-aer repository and build with ROCm support enabled manually.

---

## 3. Rust Frameworks

### qcgpu (v0.1.0) — Linux x86_64 (NVIDIA) and Windows x86_64 (GPU) only

**Last release:** April 2018. This is the only release. The project has been abandoned for 8+ years.

**OpenCL requirement:** qcgpu requires OpenCL as a mandatory dependency (not optional). OpenCL is the only compute backend.

**Cross-compilation note (Dockerfile `qcgpu-amd64` stage):** qcgpu depends on OpenCL headers that are not available when cross-compiling from ARM to AMD64. To avoid excluding qcgpu from cross-compiled amd64 images, the Dockerfile introduces a dedicated stage:

```
Stage 0: qcgpu-amd64
  FROM --platform=linux/amd64 rust:slim-bookworm
  · Compiles qcgpu natively for amd64
  · Runs under QEMU when the host is ARM
  · Produces /qcgpu-bins/qcgpu-grover and /qcgpu-bins/qcgpu-shor
```

The `rust-builder` stage then copies these prebuilt binaries when `$TARGETARCH == amd64`. **Result: qcgpu is now present in cross-compiled amd64 images, which was not the case before this stage existed.**

**Active platforms:** `linux-x86_64-nvidia` and `windows-x86_64-gpu`. In the CPU pass of the AMD64+NVIDIA double-pass run, qcgpu is **skipped** because the OpenCL/GPU backend cannot run without a GPU.

**macOS — excluded:** Apple deprecated OpenCL in macOS 10.14 Mojave (2018) and has since removed it in favor of Metal. qcgpu cannot function on current macOS versions without a compatibility layer such as MoltenVK's OpenCL support, which is not a standard system component and is not part of the benchmark environment.

**Linux aarch64 — excluded:** OpenCL support on ARM Linux is driver-dependent and not universally available. Standard ARM Linux environments (including CI runners) do not guarantee an OpenCL runtime.

**Windows arm64 — excluded:** OpenCL availability on Windows arm64 is uncertain. No evidence of a working build has been found.

**Included on x86_64 (Linux NVIDIA and Windows GPU) for:** historical reference value. qcgpu represents an early approach (2018) to GPU-accelerated quantum simulation using OpenCL.

**When this might change:** Only if the project is forked and ported to Vulkan or Metal, which would require a substantial rewrite of the compute backend.

---

### q1tsim (v0.5.0) — Included with warnings

**Last release:** November 2019. No activity since.

**Dependency conflicts:** q1tsim depends on `ndarray 0.12` and `rand 0.4`. These are significantly older than what modern Cargo workspaces typically resolve to (`ndarray 0.15+`, `rand 0.8+`). This may cause dependency resolution conflicts or require explicit version pinning in `Cargo.toml`.

**No CI:** There is no CI infrastructure. Compilation and test status on current Rust toolchains (1.80+) is unknown.

**Included for:** completeness. q1tsim represents an early Rust quantum simulation attempt. If compilation fails in the workspace, it will be excluded from results with an explicit note.

**When this might change:** If someone forks q1tsim and updates the dependency tree to modern `ndarray` and `rand` versions.

---

### quantr (v0.6.0) — Included with warnings; excluded from large-N benchmarks

**Last release:** July 2024. Low but not zero maintenance activity.

**CI:** Only `ubuntu-latest` in the CI matrix. No macOS or Windows CI. Cross-platform behavior is untested upstream.

**Qubit limit:** quantr's state vector grows as 2^n. The library is designed for small circuits and the practical qubit limit is approximately 16 qubits. The benchmark sweep (`n = [3, 5, 7, 9, 11]`) stays comfortably within this limit.

**Included for:** small-circuit benchmarks where its simplicity and API clarity are relevant comparison points.

---

### quantrs2 (v0.1.3-alpha) — Included on all platforms; GPU disabled on CPU-only runners

quantrs2 is in active development with CI on all platforms. No platform-level exclusions.

**GPU features disabled on CPU-only runners:** quantrs2 supports GPU acceleration via wgpu (`features=["gpu"]`), which auto-selects Metal on macOS, Vulkan on Linux, and DirectX 12 on Windows. On CPU-only benchmark runners where GPU availability is not guaranteed, these features are explicitly disabled to ensure reproducibility. cuQuantum integration (`features=["cuquantum"]`) is available on Linux with NVIDIA hardware but is not enabled in the standard benchmark configuration.

---

## 4. Specific Unsupported Configurations

| Configuration | Status | Reason |
|---|---|---|
| cudaq + AMD GPU (ROCm) | Not supported | cudaq is NVIDIA/CUDA-only; no ROCm backend exists |
| cudaq + Intel GPU (oneAPI) | Not supported | cudaq is NVIDIA/CUDA-only; no oneAPI backend exists |
| cudaq on Linux aarch64 | Excluded | No `manylinux_aarch64` wheels published |
| cudaq on macOS x86_64 | Excluded | No wheels published for Intel Mac |
| cudaq on Windows (native) | Excluded | No native Windows support; WSL2 workaround not applicable |
| cudaq under QEMU emulation | Skipped at runtime | SIGILL on AVX instructions |
| qdislib on Linux aarch64 | Excluded | `pymetis` (required by `find_cut`) has no arm64 wheel; build-from-source needs GCC + METIS headers not present in slim arm64 base image |
| quantrs2 GPU on CPU-only runners | Disabled intentionally | GPU not guaranteed; disabled for reproducibility |
| qcgpu on macOS | Excluded | OpenCL removed from macOS; MoltenVK OpenCL layer not standard |
| qcgpu on Linux aarch64 | Excluded | OpenCL runtime not guaranteed |
| qcgpu on Windows arm64 | Excluded | OpenCL availability uncertain |
| qcgpu in the CPU pass of AMD64+NVIDIA double run | Skipped at runtime | Requires GPU; cannot run with `--no-gpu` |
| projectq (any platform) | Removed from benchmark | Abandoned; Python 3.12 incompatibility; no ARM SIMD; no Windows wheels |
| qiskit-aer on Windows arm64 | Excluded | No wheel published for `win_arm64` |

---

## 5. When Exclusions Might Change

| Exclusion | Condition for change |
|---|---|
| projectq (re-inclusion) | Python 3.12 compatibility shim + ARM SIMD support + Windows wheels |
| cudaq on Linux aarch64 | NVIDIA publishes `manylinux_aarch64` wheels |
| cudaq on macOS x86_64 | NVIDIA releases Intel Mac wheels (assessed as very unlikely) |
| cudaq on Windows | NVIDIA adds native Windows support (no public roadmap) |
| qdislib on Linux aarch64 | `pymetis` publishes arm64 wheel, OR base image switched to one with GCC + METIS headers |
| qcgpu on macOS / arm64 | Project is forked and ported to Vulkan or Metal |
| q1tsim dependency conflicts | Fork updates ndarray and rand to modern versions |
| qiskit-aer-gpu ROCm | qiskit-aer publishes official ROCm wheels to PyPI |
| qiskit-aer on Windows arm64 | Qiskit adds Windows arm64 to CI and publishes wheel |
