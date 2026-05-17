# Libraries to benchmark
## Python
- [Qiskit] — IBM's open-source SDK for quantum computing
  - Shor implementation sourced from https://github.com/benjamin-assel/qiskit-shor (based on [Beauregard, 2002])
- [CUDA-Q] — NVIDIA's quantum-classical programming model
- [Cirq] — Google's quantum computing framework
- [QDisLib][QDisLib docs] — Distributed quantum computing library
- [ProjectQ] — Open-source quantum computing framework (ETH Zurich)
## Rust
- [QCGPU] — GPU-accelerated quantum simulator ([Kelly, 2018])
- [quantrs] — Quantum computing library in Rust
- [quantr] — Quantum circuit simulator in Rust
- [q1tsim] — Quantum simulator in Rust

# References

## Algorithms

- **Shor, P. W.** (1995). *Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer.* SIAM Review, 41(2), 303–332. arXiv: [quant-ph/9508027](https://arxiv.org/abs/quant-ph/9508027)
- **Grover, L. K.** (1996). *A fast quantum mechanical algorithm for database search.* Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC), 212–219. arXiv: [quant-ph/9605043](https://arxiv.org/abs/quant-ph/9605043)

## Circuit design and gate decomposition

- **Beauregard, S.** (2002). *Circuit for Shor's algorithm using 2n+3 qubits.* Quantum Information and Computation, 3(2), 175–185. arXiv: [quant-ph/0205095](https://arxiv.org/abs/quant-ph/0205095)
- **Draper, T. G.** (2000). *Addition on a quantum computer.* arXiv preprint: [quant-ph/0008033](https://arxiv.org/abs/quant-ph/0008033)
- **Vedral, V., Barenco, A. & Ekert, A.** (1996). *Quantum networks for elementary arithmetic operations.* Physical Review A, 54(1), 147–153. arXiv: [quant-ph/9511018](https://arxiv.org/abs/quant-ph/9511018)
- **Barenco, A. et al.** (1995). *Elementary gates for quantum computation.* Physical Review A, 52(5), 3457–3467. arXiv: [quant-ph/9503016](https://arxiv.org/abs/quant-ph/9503016)

## Benchmarking

- **Lubinski, T. et al.** (2023). *Application-oriented performance benchmarks for quantum computing.* IEEE Transactions on Quantum Engineering, 4, 1–32. arXiv: [2110.03137](https://arxiv.org/abs/2110.03137)
- **Li, A. et al.** (2020). *QASMBench: A low-level QASM benchmark suite for NISQ evaluation and simulation.* arXiv: [2005.13018](https://arxiv.org/abs/2005.13018)

## Frameworks and simulators

- **Qiskit.** IBM. *Open-source SDK for quantum computing.* [github.com/Qiskit](https://github.com/Qiskit) | [docs](https://docs.quantum.ibm.com)
- **CUDA-Q.** NVIDIA. *Quantum-classical programming model.* [github.com/NVIDIA/cuda-quantum](https://github.com/NVIDIA/cuda-quantum) | [docs](https://nvidia.github.io/cuda-quantum)
- **Cirq.** Google Quantum AI. *Python framework for NISQ circuits.* [github.com/quantumlib/Cirq](https://github.com/quantumlib/Cirq) | [docs](https://quantumai.google/cirq)
- **ProjectQ.** ETH Zurich. *Open-source quantum computing framework.* [github.com/ProjectQ-Framework/ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ) | [docs](https://projectq.readthedocs.io)
- **QDisLib.** BSC. *Distributed quantum circuit cutting.* [github.com/bsc-wdc/qdislib](https://github.com/bsc-wdc/qdislib) | arXiv: [2505.01184](https://arxiv.org/abs/2505.01184)
- **QCGPU.** Kelly, A. (2018). *Simulating quantum computers using OpenCL.* [github.com/QCGPU/qcgpu-rust](https://github.com/QCGPU/qcgpu-rust) | arXiv: [1805.00988](https://arxiv.org/abs/1805.00988)
- **quantrs.** *Quantum computing library in Rust.* [github.com/Entropy-Foundation/quantrs](https://github.com/Entropy-Foundation/quantrs) | [docs.rs](https://docs.rs/quantrs)
- **quantr.** *Quantum circuit simulator in Rust.* [github.com/a-barlow/quantr](https://github.com/a-barlow/quantr) | [docs.rs](https://docs.rs/quantr) | [book](https://a-barlow.github.io/quantr-book)
- **q1tsim.** *Quantum simulator in Rust.* [github.com/Q1tBV/q1tsim](https://github.com/Q1tBV/q1tsim) | [docs.rs](https://docs.rs/q1tsim)

## Framework documentation sources

### Cirq
- [Gates and operations](https://quantumai.google/cirq/build/gates)
- [Gate Zoo](https://quantumai.google/cirq/gatezoo)
- [Cirq basics](https://quantumai.google/cirq/start/basics)
- [Simulation](https://quantumai.google/cirq/simulate/simulation)
- [Noisy simulation](https://quantumai.google/cirq/simulate/noisy_simulation)
- [Circuit transformers](https://quantumai.google/cirq/transform/transformers)
- [Custom gates](https://quantumai.google/cirq/build/custom_gates)
- [Shor's algorithm example](https://quantumai.google/cirq/experiments/shor)
- [Textbook algorithms (Grover)](https://quantumai.google/cirq/experiments/textbook_algorithms)
- [Grover example (GitHub)](https://github.com/quantumlib/Cirq/blob/main/examples/grover.py)
- [qsim Cirq interface](https://quantumai.google/qsim/cirq_interface)
- [qsim GitHub](https://github.com/quantumlib/qsim)
- [cirq.ControlledGate reference](https://quantumai.google/reference/python/cirq/ControlledGate)
- [Representing noise](https://quantumai.google/cirq/noise/representing_noise)
- [Cirq on NVIDIA cuQuantum](https://docs.nvidia.com/cuda/cuquantum/latest/appliance/cirq.html)

### CUDA-Q
- [Quantum operations API](https://nvidia.github.io/cuda-quantum/latest/api/default_ops.html)
- [Quantum intrinsic operations spec](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/operations.html)
- [Quantum kernels spec](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/kernels.html)
- [Simulation backends](https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html)
- [Multi-GPU workflows](https://nvidia.github.io/cuda-quantum/latest/using/examples/multi_gpu_workflows.html)
- [Shor's algorithm example](https://nvidia.github.io/cuda-quantum/latest/applications/python/shors.html)
- [Example programs (Grover, QPE)](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/examples.html)
- [QFT implementation](https://nvidia.github.io/cuda-quantum/latest/applications/python/quantum_fourier_transform.html)
- [Building kernels](https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html)
- [Common quantum programming patterns](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/patterns.html)
- [Dynamic kernels](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/dynamic_kernels.html)
- [Algorithmic primitives](https://nvidia.github.io/cuda-quantum/latest/specification/cudaq/algorithmic_primitives.html)
- [Noisy simulators](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/noisy.html)

### ProjectQ
- [ProjectQ ops module](https://projectq.readthedocs.io/en/latest/_doc_gen/projectq.ops.html)
- [ProjectQ backends](https://projectq.readthedocs.io/en/latest/_doc_gen/projectq.backends.html)
- [ProjectQ compiler engines](https://projectq.readthedocs.io/en/latest/_doc_gen/projectq.cengines.html)
- [Decomposition rules](https://projectq.readthedocs.io/en/fix-docs/projectq.setups.decompositions.html)
- [Tutorials](https://projectq.readthedocs.io/en/latest/tutorials.html)
- [Examples](https://projectq.readthedocs.io/en/latest/examples.html)
- [Shor's algorithm source](https://github.com/ProjectQ-Framework/ProjectQ/blob/develop/examples/shor.py)
- [Simulator tutorial](https://notebook.community/ProjectQ-Framework/ProjectQ/examples/simulator_tutorial)
- [Meta module (Control, Compute, Dagger)](https://projectq.readthedocs.io/en/v0.3.6/projectq.meta.html)
- [ProjectQ paper](https://arxiv.org/abs/1612.08091) (arXiv: 1612.08091)

### QDisLib
- [GitHub repository](https://github.com/bsc-wdc/qdislib)
- [Paper: Distributed Quantum Circuit Cutting](https://arxiv.org/abs/2505.01184) (arXiv: 2505.01184)

### q1tsim
- [docs.rs/q1tsim](https://docs.rs/q1tsim/0.5.0/q1tsim/)
- [GitHub: Q1tBV/q1tsim](https://github.com/Q1tBV/q1tsim)

### qcgpu
- [docs.rs/qcgpu](https://docs.rs/qcgpu/0.1.0/qcgpu/)
- [GitHub: QCGPU/qcgpu-rust](https://github.com/QCGPU/qcgpu-rust)
- [Paper: Simulating Quantum Computers Using OpenCL](https://arxiv.org/abs/1805.00988) (arXiv: 1805.00988)

### quantr
- [docs.rs/quantr](https://docs.rs/quantr/0.6.0/quantr/)
- [GitHub: a-barlow/quantr](https://github.com/a-barlow/quantr)
- [quantr book](https://a-barlow.github.io/quantr-book)

### quantrs
- [GitHub: Entropy-Foundation/quantrs](https://github.com/Entropy-Foundation/quantrs) (NOTE: repo may not exist, verify)
- [docs.rs/quantrs](https://docs.rs/quantrs)

### Other references
- [Benchmarking Quantum Simulators (2024)](https://arxiv.org/html/2401.09076v2)
- [cuStateVec blog (NVIDIA)](https://developer.nvidia.com/blog/accelerating-quantum-circuit-simulation-with-nvidia-custatevec/)
- [Awesome Quantum Computing list](https://github.com/bramathon/awesome-quantum-computing)

# Roadmap
## Investigation 1
1. Create full list of libraries to benchmark
2. Learn how to implement first round of algorithms in all libraries to run in local
## First round of algorithms
- Shor using premade implementation if possible ([Shor, 1995]; circuit: [Beauregard, 2002])
- Grover using premade implementation if possible ([Grover, 1996])
## Benchmarks & result processing
- Parametrize algorithms
  - Take into account setup time, implementation time, correlation with input sizes
  - Parameters
    - Input Size
    - Maybe pick parameters from [QASMBench] and [Lubinski, 2023]
## Report results
- Write memory up to this point
# Extra roadmap (if time allows)
## Increase scope
- Add all possible parameters from [QASMBench] and [Lubinski, 2023] to the existing investigation
> # Loop
> ## Investigation
> - Pick one algorithm from the BenchMark Procedures doc
> - Learn how to implement on each library
> ## New algorithm round
> - Implement new algorithm
> ## Benchmark and process results
> - Just as before but taking into account the new parameters too
> ## Report results
> - Add to memory
# Extra Extra Roadmap
## Build
- Create a tool to which you can feed a quantum algorithm's parameters and it recommends you a library

---

<!-- Reference key definitions -->
[Shor, 1995]: https://arxiv.org/abs/quant-ph/9508027 "Polynomial-time algorithms for prime factorization"
[Grover, 1996]: https://arxiv.org/abs/quant-ph/9605043 "A fast quantum mechanical algorithm for database search"
[Beauregard, 2002]: https://arxiv.org/abs/quant-ph/0205095 "Circuit for Shor's algorithm using 2n+3 qubits"
[Draper, 2000]: https://arxiv.org/abs/quant-ph/0008033 "Addition on a quantum computer"
[Vedral, 1996]: https://arxiv.org/abs/quant-ph/9511018 "Quantum networks for elementary arithmetic operations"
[Barenco, 1995]: https://arxiv.org/abs/quant-ph/9503016 "Elementary gates for quantum computation"
[Lubinski, 2023]: https://arxiv.org/abs/2110.03137 "Application-oriented performance benchmarks for quantum computing"
[QASMBench]: https://arxiv.org/abs/2005.13018 "QASMBench: A low-level QASM benchmark suite"
[Kelly, 2018]: https://arxiv.org/abs/1805.00988 "Simulating quantum computers using OpenCL (QCGPU)"
[QDisLib docs]: docs/papers/QDisLib%20-%20Distributed%20Quantum%20Computing%20Library.pdf "QDisLib documentation"
[Qiskit]: https://github.com/Qiskit "Qiskit — IBM quantum SDK"
[CUDA-Q]: https://github.com/NVIDIA/cuda-quantum "CUDA-Q — NVIDIA quantum-classical programming"
[Cirq]: https://github.com/quantumlib/Cirq "Cirq — Google quantum framework"
[ProjectQ]: https://github.com/ProjectQ-Framework/ProjectQ "ProjectQ — ETH Zurich quantum framework"
[QCGPU]: https://github.com/QCGPU/qcgpu-rust "QCGPU — GPU-accelerated quantum simulator"
[quantrs]: https://github.com/Entropy-Foundation/quantrs "quantrs — Quantum computing in Rust"
[quantr]: https://docs.rs/quantr "quantr — Quantum circuit simulator in Rust"
[q1tsim]: https://docs.rs/q1tsim "q1tsim — Quantum simulator in Rust"

## How to run

### Quick start

```bash
./bench [--time-budget MINUTES] [--dev]
```

On Windows:

```powershell
.\bench.ps1 [-TimeBudget MINUTES] [-Dev]
```

**Options:**
- `--time-budget` / `-TimeBudget`: Set maximum runtime in minutes (default: 60). The benchmark stops cleanly when time expires.
- `--dev`: Dev mode — results saved as JSON in `results/` locally. No database connection required.

### Configuración Supabase (modo normal)

Crea un fichero `.env` junto al script `bench` / `bench.ps1`:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<secret-key>
```

Ejecuta el SQL de `docs/supabase_schema.sql` en el SQL Editor de tu proyecto Supabase para crear la tabla.

En modo `--dev` Supabase se ignora y los resultados se guardan como JSON en `results/`.