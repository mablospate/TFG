# Rust Quantum Frameworks — Research Notes

> Exploration document generated 2026-05-13. Not user-facing. Sources: crates.io, GitHub, official documentation.

---

## 1. Framework Overview Table

| Crate | Version | Last Release | CI Platforms | GPU Support | Qubit Limit | Pure Rust |
|---|---|---|---|---|---|---|
| **quantrs2** | 0.1.3-alpha | March 2026 | Linux x86_64, Linux aarch64, macOS arm64, macOS x86_64, Windows x64 | wgpu (Metal/Vulkan/DX12), cuQuantum (Linux NVIDIA) | No fixed limit (multiple backends) | Yes |
| **roqoqo** | 1.21.0 | August 2025 | Linux x86_64, Linux aarch64, macOS arm64, macOS x86_64, Windows x64 | None (IR layer; backend-dependent) | No fixed limit | Yes (qoqo_quest has C dep) |
| **qip / RustQIP** | 1.5.0 | December 2025 | None | None | No fixed limit (sparse statevector) | Yes |
| **quantr** | 0.6.0 | July 2024 | ubuntu-latest only | None | ~16 qubits practical | Yes |
| **q1tsim** | 0.5.0 | November 2019 | None | None | No fixed limit | Yes |
| **qcgpu** | 0.1.0 | April 2018 | None documented | OpenCL (mandatory) | No fixed limit | No (OpenCL C bindings) |

---

## 2. Per-Framework Notes

### quantrs2 (v0.1.3-alpha)

**Source:** https://crates.io/crates/quantrs2 · https://github.com/cool-japan/quantrs

**Status:** Active development. ~1,095 total downloads as of research date. Alpha quality — API may change before stable release.

**Simulation backends:**
- Statevector (dense)
- Matrix Product States (MPS)
- Stabilizer / Clifford
- Tensor network
- Decision diagrams
- Path integral
- Quantum Monte Carlo

**SIMD and parallelism:** Implemented via the `scirs2-core` dependency with `simd` and `parallel` feature flags. These enable vectorized state vector operations and multi-threaded gate application via rayon.

**GPU acceleration:**
- `features=["gpu"]` activates wgpu-based GPU acceleration. wgpu auto-selects the appropriate graphics API: Metal on macOS, Vulkan on Linux, DirectX 12 on Windows. Single feature flag covers all platforms.
- `features=["cuquantum"]` activates NVIDIA cuQuantum integration. Requires CUDA toolkit and cuQuantum libraries. Only applicable on Linux with an NVIDIA GPU.

**Build:** Pure Rust. No C/FFI dependencies in the default configuration. The `cuquantum` feature introduces an FFI layer to the cuQuantum C library.

**Risk:** Alpha state. If the API changes between current alpha and a stable release, the benchmark implementations will require updating.

---

### roqoqo (v1.21.0)

**Source:** https://crates.io/crates/roqoqo · https://github.com/HQSquantumsimulations/qoqo

**Status:** Stable. 233,737 total downloads. Actively maintained by HQS Quantum Simulations with EU funding (PlanQK, QSolid, PhoQuant projects).

**Architecture note:** roqoqo is a quantum circuit intermediate representation (IR), not a simulator. A backend crate is required for actual simulation or hardware execution:
- `qoqo_quest`: wraps the QuEST C library (see compilation notes below)
- `qoqo_qiskit`: delegates to Qiskit via Python interop
- `qoqo_for_braket`: targets Amazon Braket QPUs and simulators

**GPU:** No GPU support at the roqoqo IR layer. The `qoqo_quest` backend runs on CPU. GPU would require a custom backend.

**Build:** The roqoqo crate itself is pure Rust. The `qoqo_quest` backend has a C dependency (QuEST library, written in C/Fortran). This is handled by a build script that compiles QuEST from source. Cross-compilation may require manual sysroot setup.

---

### qip / RustQIP (v1.5.0)

**Source:** https://crates.io/crates/qip · https://github.com/Renmusxd/RustQIP

**Status:** Low activity. Last commit December 2025. 26+ open issues without triage. No CI on any platform.

**Simulation approach:** Sparse statevector representation using a circuit graph. The borrow checker enforces correct qubit usage at compile time. Parallelism via rayon (enabled by default feature).

**Notable:** The type-safe circuit construction API is a design interest — qubit ownership is tracked via Rust's ownership system, making it impossible to apply a gate to a qubit that has already been measured or moved.

**GPU:** None.

**Build:** Pure Rust.

---

### quantr (v0.6.0)

**Source:** https://crates.io/crates/quantr · https://github.com/quantr-project/quantr (verify URL)

**Status:** Low maintenance. Last release July 2024. CI only on `ubuntu-latest`. No macOS or Windows CI.

**Simulation approach:** Dense statevector. State grows as 2^n complex amplitudes. Designed and tested for small circuits (up to ~16 qubits in practice). Beyond 16 qubits, memory becomes prohibitive and no optimizations exist for large state vectors.

**GPU:** None.

**Build:** Pure Rust.

**Usage in this project:** Restricted to small-circuit benchmarks (n ≤ 16). Excluded from large-N configurations.

---

### q1tsim (v0.5.0)

**Source:** https://crates.io/crates/q1tsim

**Status:** Abandoned. Last release November 2019. No GitHub activity since.

**Dependency versions (old):**
- `ndarray 0.12` (current: 0.15+)
- `rand 0.4` (current: 0.8+)

These version constraints are likely to cause Cargo dependency resolution failures in a modern workspace unless explicitly pinned. The `Cargo.lock` would need careful management to include q1tsim alongside modern crates.

**GPU:** None.

**Build:** Pure Rust, but old dependency tree is a practical barrier.

---

### qcgpu (v0.1.0)

**Source:** https://crates.io/crates/qcgpu · https://github.com/lisnidispha/qcgpu-rust (verify URL)

**Status:** Abandoned. Only release: April 2018. 8+ years without activity.

**OpenCL dependency:** OpenCL is a mandatory, non-optional dependency. The entire simulation backend is implemented in OpenCL kernel code. There is no CPU fallback.

**GPU acceleration approach (2018 context):** qcgpu used OpenCL for GPU-accelerated statevector simulation. At the time, this was one of the few Rust quantum simulators with GPU support. The design predates modern GPU compute frameworks (wgpu, Vulkan compute) in the Rust ecosystem.

**Platform restrictions:**
- macOS: OpenCL deprecated in macOS 10.14 (2018); removed on current versions. qcgpu cannot run on current macOS.
- Linux aarch64: OpenCL support is driver-dependent; not universally available on ARM Linux.
- Windows arm64: OpenCL availability uncertain.
- Linux x86_64 and Windows x86_64: OpenCL available via NVIDIA/AMD/Intel GPU drivers.

**Build:** Has OpenCL C bindings via `ocl` crate. Not pure Rust. Requires an OpenCL runtime at link time and at runtime.

---

## 3. Hardware Acceleration Research

### quantrs2 + wgpu (Cross-Platform GPU)

quantrs2's GPU acceleration uses the `wgpu` crate, which provides a portable WebGPU API across multiple graphics backends:

| Platform | wgpu backend | Status |
|---|---|---|
| macOS arm64 | Metal | Native Apple GPU acceleration |
| macOS x86_64 | Metal | Intel integrated/discrete GPU |
| Linux x86_64 (NVIDIA/AMD) | Vulkan | Full GPU compute support |
| Linux aarch64 | Vulkan (driver-dependent) | Varies by hardware |
| Windows x86_64 | DirectX 12 / Vulkan | Both backends available |

A single `features=["gpu"]` Cargo feature activates all of the above. The wgpu backend selection is automatic at runtime based on available drivers.

**In this project:** GPU features are disabled on CPU-only runners for reproducibility. On hardware with GPU availability (NVIDIA Linux server, Apple Silicon Mac), enabling `features=["gpu"]` is straightforward.

### quantrs2 + cuQuantum (NVIDIA Linux)

For NVIDIA GPU acceleration beyond wgpu (higher performance, tighter CUDA integration), quantrs2 provides `features=["cuquantum"]`. This requires:
- CUDA toolkit (12+)
- cuQuantum library installed on the system
- Linux x86_64 or Linux aarch64 with NVIDIA GPU

This feature enables the cuQuantum-based statevector simulator, which uses NVIDIA's optimized quantum circuit simulation library.

### AMX / Accelerate on macOS arm64 (NumPy-based frameworks)

Frameworks that use NumPy internally (Cirq, Qiskit CPU path) benefit indirectly from Apple's Accelerate framework on macOS arm64. NumPy 2.x links against Accelerate's BLAS/LAPACK implementation by default on macOS arm64. The Accelerate framework uses AMX (Apple Matrix Extensions) hardware when available on M-series chips.

This means:
- `cirq.Simulator()` matrix operations are executed via Accelerate BLAS (AMX-accelerated matmul/einsum)
- Qiskit-Aer's CPU statevector also benefits from Accelerate for the linear algebra portions
- The benefit is transparent — no user-facing configuration is required; it comes from the NumPy build included in the macOS arm64 Python distribution

No equivalent automatic acceleration exists on Linux arm64 (OpenBLAS or BLIS is used instead, without AMX).

### qcgpu OpenCL (Historical Reference)

qcgpu used OpenCL kernels for GPU statevector simulation. The approach stores the full 2^n complex amplitude vector in GPU memory and applies gate operations as GPU kernel launches. This was a reasonable approach in 2018 but is now obsolete compared to:
- Vulkan compute (supported by wgpu in modern Rust)
- CUDA/cuQuantum (NVIDIA-specific, higher performance)
- Metal compute shaders (Apple-specific)

qcgpu is included on x86_64 platforms as a historical data point, not as a competitive option.

---

## 4. Compilation Notes

### Pure Rust crates (no C dependencies)

These crates build with a standard Rust toolchain (`cargo build`) without requiring any system libraries:

- quantrs2 (default features)
- roqoqo (IR layer only)
- qip / RustQIP
- quantr
- q1tsim

Cross-compilation is straightforward for pure Rust crates: `cargo build --target <triple>` works without sysroot setup.

### Crates with C dependencies

| Crate | C dependency | When required | Cross-compilation impact |
|---|---|---|---|
| quantrs2 | cuQuantum C library | `features=["cuquantum"]` only | Not applicable (NVIDIA Linux only) |
| roqoqo / qoqo_quest | QuEST (C/Fortran) | `qoqo_quest` backend | Requires C cross-toolchain |
| qcgpu | OpenCL runtime | Always (mandatory) | Requires OpenCL headers + runtime |

### Notes on workspace dependency conflicts (q1tsim)

q1tsim 0.5.0 specifies `ndarray = "0.12"` and `rand = "0.4"`. In a Cargo workspace that also includes crates depending on `ndarray 0.15` and `rand 0.8`, these will coexist as separate semver-incompatible versions (Cargo allows this). The workspace will compile both versions of each dependency. This increases compile times and binary size but should not cause linker errors. If it does cause resolution failures, `[patch.crates-io]` or workspace `[dependencies]` overrides may be needed.
