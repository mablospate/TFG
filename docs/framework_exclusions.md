# Framework Exclusions and Limitations

> Document generated 2026-05-13. Records which frameworks are excluded from specific platforms and why.

## 1. Introduction — Criteria for Exclusion

Frameworks or platform combinations are excluded from benchmarks when one or more of the following conditions apply:

1. **Abandonment**: No release or commit activity for multiple years, combined with known incompatibilities with current toolchains.
2. **Broken dependencies**: The package cannot be installed or built on the target platform due to missing wheels, removed standard library modules, or missing system libraries.
3. **Platform restrictions by the vendor**: The framework vendor explicitly does not support the platform (no wheels published, no CI, no documentation).
4. **Non-comparable results**: The framework produces results on the platform that are structurally incomparable to other frameworks — not due to algorithmic differences but due to infrastructure overhead (e.g., per-shot Python loop).

Frameworks that are included with caveats are marked as **degraded** rather than excluded; their data is reported with explicit annotations.

---

## 2. Python Frameworks

### projectq (v0.8.0) — Included with warnings on all platforms except Windows

**Last PyPI release:** v0.8.0, October 2022. No commits since then. Effectively abandoned.

**Python version constraint:** projectq requires Python 3.11. Python 3.12 removed `distutils` from the standard library. The projectq C++ extension build fails on Python 3.12+ with:

```
AttributeError: 'Compiler' object has no attribute 'dry_run'
```

The C++ simulator (`_cppsim`) cannot be built without `distutils`. Python 3.11 must be used (where `distutils` is still present in the stdlib).

**ARM performance degradation (macOS arm64 / Linux aarch64):** The `_cppsim` extension compiles without SIMD ARM optimizations. There is no NEON or SVE vectorization in the projectq codebase. On arm64 hosts, performance is approximately 800× slower than native SIMD simulators. This is not solely an ARM issue — the dominant cost is the per-shot Python overhead (projectq creates a full `MainEngine` cycle per shot) — but the absence of ARM SIMD vectorization makes the gap wider than on x86_64 with AVX2.

**Included because:** projectq is functionally correct and produces valid probability distributions. It is useful as a historical baseline and for verifying JSD metrics against other simulators. Latency results are explicitly annotated as non-comparable.

**Windows (x86_64 and arm64):** Excluded completely. There is no precompiled Windows wheel on PyPI. Building the C++ extension on Windows requires a manually configured MSVC toolchain with steps not documented by the projectq project.

**When this might change:**
- If the projectq project merges an ARM SIMD (NEON/SVE) pull request.
- If projectq adds Python 3.12 compatibility via a `setuptools` `distutils` shim.
- If a third party publishes Windows wheels.

---

### cudaq / CUDA-Q (v0.14.0) — Excluded on macOS x86_64 and Windows

**macOS x86_64 (Intel Mac) — excluded completely:**

The real installable package is `cuda-quantum-cu13` (not the stub `cudaq` on PyPI). As of May 2026, only `macosx_arm64` wheels are published. No `macosx_x86_64` wheels exist for any version from v0.14 onward. Intel Mac support was not announced and is not expected given Apple's transition to Apple Silicon.

**Windows — excluded completely:**

There is no native Windows support for cudaq. The official workaround is WSL2 (Linux subsystem), which is not applicable to a native Windows benchmark runner. No `win_amd64` or `win_arm64` wheels are published on PyPI.

**When this might change:**
- macOS x86_64: only if NVIDIA publishes Intel Mac wheels. This is considered very unlikely given the Apple Silicon transition and the direction of Apple's platform roadmap.
- Windows: only if NVIDIA adds native Windows support. No public roadmap indication of this as of May 2026.

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

### qcgpu (v0.1.0) — Linux x86_64 and Windows x86_64 only

**Last release:** April 2018. This is the only release. The project has been abandoned for 8+ years.

**OpenCL requirement:** qcgpu requires OpenCL as a mandatory dependency (not optional). OpenCL is the only compute backend.

**macOS — excluded:** Apple deprecated OpenCL in macOS 10.14 Mojave (2018) and has since removed it in favor of Metal. qcgpu cannot function on current macOS versions without a compatibility layer such as MoltenVK's OpenCL support, which is not a standard system component and is not part of the benchmark environment.

**Linux aarch64 — excluded:** OpenCL support on ARM Linux is driver-dependent and not universally available. Standard ARM Linux environments (including CI runners) do not guarantee an OpenCL runtime.

**Windows arm64 — excluded:** OpenCL availability on Windows arm64 is uncertain. No evidence of a working build has been found.

**Included on x86_64 (Linux and Windows) for:** historical reference value. qcgpu represents an early approach (2018) to GPU-accelerated quantum simulation using OpenCL. Its inclusion provides a comparison point for how the field has evolved.

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

**Qubit limit:** quantr's state vector grows as 2^n. The library is designed for small circuits and the practical qubit limit is approximately 16 qubits. It is excluded from any benchmark configuration involving more than 16 qubits.

**Included for:** small-circuit benchmarks where its simplicity and API clarity are relevant comparison points.

---

### qip / RustQIP (v1.5.0) — Excluded on Windows

**Activity:** Last commit December 2025. 26+ open issues without triage. No CI on any platform.

**Windows (x86_64 and arm64) — excluded:** There is no CI evidence of a working Windows build. With 26 untriaged issues and no CI, correctness on Windows cannot be verified. Excluded from all Windows benchmark configurations.

**Linux and macOS — included as best-effort:** qip is included on Linux and macOS platforms as a best-effort inclusion. Results carry a note that no CI guarantees are in place.

---

### roqoqo (v1.21.0) — Included on all platforms

roqoqo is actively maintained by HQS Quantum Simulations with EU funding (PlanQK, QSolid, PhoQuant). CI covers all platforms. No exclusions.

**Note on the `qoqo_quest` backend:** The `qoqo_quest` simulation backend depends on the QuEST C library (a C/Fortran external dependency). This adds build complexity, particularly for cross-compilation. The project documents this dependency and its CI handles it, but local builds on non-standard environments may require additional setup.

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
| quantrs2 GPU on CPU-only runners | Disabled intentionally | GPU not guaranteed; disabled for reproducibility |
| qcgpu on macOS | Excluded | OpenCL removed from macOS; MoltenVK OpenCL layer not standard |
| projectq on Python 3.12+ | Broken | `distutils` removed; C++ extension fails to build |
| cudaq on macOS x86_64 | Excluded | No wheels published for Intel Mac |
| cudaq on Windows (native) | Excluded | No native Windows support; WSL2 workaround not applicable |
| projectq on Windows | Excluded | No precompiled wheel; MSVC build undocumented |
| qiskit-aer on Windows arm64 | Excluded | No wheel published for `win_arm64` |
| qip on Windows | Excluded | No CI; build unverified |

---

## 5. When Exclusions Might Change

| Exclusion | Condition for change |
|---|---|
| projectq on Python 3.12+ | projectq adds Python 3.12 compatibility via setuptools distutils shim |
| projectq ARM performance | ARM SIMD (NEON/SVE) PR merged into projectq |
| projectq on Windows | projectq publishes precompiled Windows wheels |
| cudaq on macOS x86_64 | NVIDIA releases Intel Mac wheels (assessed as very unlikely) |
| cudaq on Windows | NVIDIA adds native Windows support (no public roadmap) |
| qcgpu on macOS / arm64 | Project is forked and ported to Vulkan or Metal |
| q1tsim dependency conflicts | Fork updates ndarray and rand to modern versions |
| qip on Windows | CI is added and a working Windows build is verified |
| qiskit-aer-gpu ROCm | qiskit-aer publishes official ROCm wheels to PyPI |
| qiskit-aer on Windows arm64 | Qiskit adds Windows arm64 to CI and publishes wheel |
