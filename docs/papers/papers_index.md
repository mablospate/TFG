# Papers Index — TFG Quantum Simulator Benchmark

## Section mapping table

Sections:
- **2.1.1** Bits y qubits
- **2.1.2** Modelos de computación cuántica (circuitos)
- **2.1.3** Problemas clásicamente intratables (Grover, Shor)
- **2.1.4** Simulación de circuitos cuánticos (statevector, métodos)
- **2.1.5** Hardware cuántico real y sus limitaciones
- **2.2.1** Métricas de evaluación de simuladores
- **2.2.2** Comparativas y suites de benchmarks existentes
- **2.2.3** Limitaciones de benchmarks previos
- **2.3.1** Rust como lenguaje de sistemas moderno
- **2.3.2** Adopción de Rust en HPC y computación científica
- **2.3.3** El patrón Rust+Python (PyO3)

| Filename | Short title | Year | 2.1.1 | 2.1.2 | 2.1.3 | 2.1.4 | 2.1.5 | 2.2.1 | 2.2.2 | 2.2.3 | 2.3.1 | 2.3.2 | 2.3.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Barenco 1995 - Elementary Gates for Quantum Computation.pdf | Elementary gates (Barenco) | 1995 | | ✓ | | | | | | | | | |
| Beauregard 2002 - Circuit for Shors Algorithm Using 2n+3 Qubits.pdf | Shor circuit 2n+3 qubits | 2002 | | ✓ | ✓ | | | | | | | | |
| Draper 2000 - Addition on a Quantum Computer.pdf | QFT-based addition | 2000 | | ✓ | ✓ | | | | | | | | |
| Grover 1996 - Fast Quantum Search Algorithm.pdf | Grover's search (updated) | 1996 | ✓ | ✓ | ✓ | | | | | | | | |
| Grover_early_days_1709.01236.pdf | Early days of Grover | 2017 | | ✓ | ✓ | | | | | | | | |
| Grover_original_1996_quant-ph_9605043.pdf | Grover's search (original) | 1996 | ✓ | ✓ | ✓ | | | | | | | | |
| HPC_simulators_quantum_volume_2412.20518.pdf | HPC simulators & QV | 2024 | | | | ✓ | ✓ | ✓ | ✓ | ✓ | | | |
| Kelly 2018 - Simulating Quantum Computers Using OpenCL (QCGPU).pdf | QCGPU (OpenCL sim) | 2018 | ✓ | | | ✓ | | ✓ | ✓ | ✓ | | | |
| Lubinski 2023 - Application-Oriented Performance Benchmarks.pdf | QED-C app-oriented benchmarks | 2023 | | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | | | |
| MQTbench_2204.13719.pdf | MQT Bench | 2022 | | ✓ | | | | ✓ | ✓ | ✓ | | | |
| QASMbench_2005.13018.pdf | QASMBench | 2020 | | ✓ | | | ✓ | ✓ | ✓ | ✓ | | | |
| QDisLib - Distributed Quantum Computing Library.pdf | Qdislib (circuit cutting) | 2025 | ✓ | ✓ | | ✓ | ✓ | | | | | | |
| Rust_safety_performance_2206.05503.pdf | Rust: Safety & Performance | 2022 | | | | | | | | | ✓ | ✓ | |
| Rust_vs_C_HPC_nbody_2107.11912.pdf | Rust vs C HPC N-Body | 2021 | | | | | | | | | ✓ | ✓ | |
| Shor 1995 - Polynomial-Time Algorithms for Prime Factorization.pdf | Shor SIAM 1995 (arXiv v2) | 1995 | ✓ | ✓ | ✓ | | | | | | | | |
| Shor_algorithm_quant-ph_0010034.pdf | Shor lecture (Lomonaco) | 2000 | | ✓ | ✓ | | | | | | | | |
| Shor_original_1997_quant-ph_9508027.pdf | Shor's factoring (original) | 1997 | ✓ | ✓ | ✓ | | | | | | | | |
| SupermarQ_2202.11045.pdf | SupermarQ | 2022 | | ✓ | | | ✓ | ✓ | ✓ | ✓ | | | |
| Vedral 1996 - Quantum Networks for Elementary Arithmetic.pdf | Quantum arithmetic networks | 1995 | ✓ | ✓ | ✓ | | | | | | | | |
| benchmark_performance_quantum_software_2409.08844.pdf | Benchpress (IBM) | 2025 | | | | | | ✓ | ✓ | ✓ | | | |
| benchmark_simulation_software_2401.09076.pdf | Sim SW benchmark (Jamadagni) | 2024 | | | | ✓ | | ✓ | ✓ | ✓ | | | |
| classical_simulation_herculean_2302.08880.pdf | Herculean classical sim | 2023 | ✓ | ✓ | | ✓ | | ✓ | | ✓ | | | |
| scalable_parallel_simulation_CPU_GPU_2509.04955.pdf | Q²Chemistry CPU/GPU | 2025 | | | | ✓ | | ✓ | ✓ | | | ✓ | |
| statevector_gpu_tensor_2401.06188.pdf | Statevector & tensor GPU | 2025 | | ✓ | | ✓ | | ✓ | ✓ | ✓ | | | |

---

## Per-paper entries

### Barenco 1995 - Elementary Gates for Quantum Computation.pdf
- **Full title**: Elementary gates for quantum computation
- **Authors**: Adriano Barenco, Charles H. Bennett, Richard Cleve, David P. DiVincenzo, Norman Margolus, Peter Shor, Tycho Sleator, John Smolin, Harald Weinfurter
- **Year**: 1995
- **Venue**: Physical Review A 52(5):3457–3467, 1995; arXiv:quant-ph/9503016
- **Contribution**: Proves that all one-qubit gates together with the two-qubit CNOT gate form a universal set for quantum computation, and derives tight upper and lower bounds on the number of elementary gates required to implement arbitrary n-bit unitary operations including the Deutsch-Toffoli family.

---

### Beauregard 2002 - Circuit for Shors Algorithm Using 2n+3 Qubits.pdf
- **Full title**: Circuit for Shor's algorithm using 2n+3 qubits
- **Authors**: Stéphane Beauregard
- **Year**: 2002
- **Venue**: Quantum Information and Computation 3(2):175–185, 2003; arXiv:quant-ph/0205095
- **Contribution**: Presents a space-optimised quantum circuit for Shor's factoring algorithm that requires only 2n+3 qubits and O(n³ log n) elementary gates, the most qubit-efficient complete factoring circuit known at the time of publication.

---

### Draper 2000 - Addition on a Quantum Computer.pdf
- **Full title**: Addition on a quantum computer
- **Authors**: Thomas G. Draper
- **Year**: 2000
- **Venue**: arXiv:quant-ph/0008033 (written 1998, revised 2000)
- **Contribution**: Introduces a carry-free quantum adder based on the quantum Fourier transform that eliminates the need for temporary carry qubits, enables adding a classical number into a quantum superposition without encoding it in a quantum register, and supports massive parallelisation.

---

### Grover 1996 - Fast Quantum Search Algorithm.pdf
- **Full title**: A fast quantum mechanical algorithm for database search
- **Authors**: Lov K. Grover
- **Year**: 1996
- **Venue**: Proceedings of the 28th Annual ACM STOC 1996, Philadelphia PA, pp. 212–219 (updated version)
- **Contribution**: Presents the canonical O(√N) quantum search algorithm with amplitude amplification, provides a matching Ω(√N) lower bound (citing BBBV96), and introduces the Walsh-Hadamard initialisation and selective-phase-rotation primitives that underpin the algorithm.

---

### Grover_early_days_1709.01236.pdf
- **Full title**: Early days following Grover's quantum search algorithm
- **Authors**: Fang Song
- **Year**: 2017
- **Venue**: arXiv:1709.01236 [quant-ph] (lecture notes companion, Portland State University)
- **Contribution**: Reviews and contextualises the early theoretical results around Grover's algorithm — multiple-marked-items variants, quantum counting, and the optimality proof — collecting scattered literature into a single coherent reference.

---

### Grover_original_1996_quant-ph_9605043.pdf
- **Full title**: A fast quantum mechanical algorithm for database search
- **Authors**: Lov K. Grover
- **Year**: 1996
- **Venue**: Proceedings of the 28th Annual ACM STOC 1996, Philadelphia PA, pp. 212–219
- **Contribution**: Introduces the original O(√N) quantum search algorithm, demonstrating the first quadratic speedup over classical search via amplitude amplification on an unstructured database.

---

### HPC_simulators_quantum_volume_2412.20518.pdf
- **Full title**: A comparison of HPC-based quantum computing simulators using Quantum Volume
- **Authors**: Lourens van Niekerk, Dhiraj Kumar, Aasish Kumar Sharma, Tino Meisel, Martin Leandro Paleico, Christian Boehme
- **Year**: 2024
- **Venue**: arXiv:2412.20518 [quant-ph]
- **Contribution**: Benchmarks multiple HPC quantum simulators (CPU and GPU) using Quantum Volume as the single metric, providing a systematic comparison of simulation time and scalability that is directly relevant to evaluating simulator performance in the NISQ era.

---

### Kelly 2018 - Simulating Quantum Computers Using OpenCL (QCGPU).pdf
- **Full title**: Simulating quantum computers using OpenCL
- **Authors**: Adam Kelly
- **Year**: 2018
- **Venue**: arXiv:1805.00988 [quant-ph], November 2018
- **Contribution**: Describes QCGPU, a portable GPU-accelerated statevector simulator built on OpenCL that achieves higher throughput than contemporary CPU-based simulators by parallelising gate application across all 2ⁿ amplitudes while remaining hardware-agnostic across NVIDIA, AMD, and Intel devices.

---

### Lubinski 2023 - Application-Oriented Performance Benchmarks.pdf
- **Full title**: Application-Oriented Performance Benchmarks for Quantum Computing
- **Authors**: Thomas Lubinski, Sonika Johri, Paul Varosy, Jeremiah Coleman, Luning Zhao, Jason Necaise, Charles H. Baldwin, Karl Mayer, Timothy Proctor (QED-C collaboration)
- **Year**: 2023
- **Venue**: arXiv:2110.03137 [quant-ph], dated January 2023; IEEE Transactions on Quantum Engineering
- **Contribution**: Introduces an open-source suite of application-oriented benchmarks (Grover, QFT, phase estimation, VQE, Shor order-finding, etc.) that measures result fidelity as a function of circuit width and depth using volumetric benchmarking, providing end-users with a practical quality-vs-time-to-solution metric for comparing quantum hardware and simulators.

---

### MQTbench_2204.13719.pdf
- **Full title**: MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing
- **Authors**: Nils Quetschlich, Lukas Burgholzer, Robert Wille
- **Year**: 2022 (accepted Quantum 2023)
- **Venue**: Quantum, accepted 2023-06-07; arXiv:2204.13719 [quant-ph]
- **Contribution**: Proposes a cross-level benchmark suite of over 70,000 circuits (2–130 qubits) on four abstraction levels, enabling reproducible and comparable evaluation of quantum software tools across the entire compilation stack.

---

### QASMbench_2005.13018.pdf
- **Full title**: QASMBench: A Low-Level Quantum Benchmark Suite for NISQ Evaluation and Simulation
- **Authors**: Ang Li, Samuel Stein, Sriram Krishnamoorthy, James Ang
- **Year**: 2020
- **Venue**: ACM Transactions on Quantum Computing, Vol. 37, No. 4, Article 111 (August 2020)
- **Contribution**: Provides a low-level OpenQASM benchmark suite with four circuit metrics (gate density, retention lifespan, measurement density, entanglement variance) to characterise NISQ device and simulator performance across diverse application domains.

---

### QDisLib - Distributed Quantum Computing Library.pdf
- **Full title**: Distributed Quantum Circuit Cutting for Hybrid Quantum-Classical High-Performance Computing
- **Authors**: Mar Tejedor, Berta Casas, Javier Conejero, Alba Cervera-Lierta, Rosa M. Badia
- **Year**: 2025
- **Venue**: arXiv:2505.01184 [cs.DC], May 2025; Barcelona Supercomputing Center
- **Contribution**: Introduces Qdislib, a distributed and flexible library for quantum circuit cutting (wire and gate cutting) that integrates with the PyCOMPSs task-based HPC programming model to execute large quantum circuits across heterogeneous CPU, GPU, and QPU resources in a fully parallelised manner.

---

### Rust_safety_performance_2206.05503.pdf
- **Full title**: Rust: The Programming Language for Safety and Performance
- **Authors**: William Bugden, Ayman Alahmar
- **Year**: 2022
- **Venue**: IGSCONG'22 — 2nd International Graduate Studies Congress, June 2022
- **Contribution**: Surveys recent benchmarking research comparing Rust against C, C++, Go, Java, and Python, concluding that Rust achieves C/C++-level performance while providing superior memory safety and security guarantees.

---

### Rust_vs_C_HPC_nbody_2107.11912.pdf
- **Full title**: Performance vs Programming Effort between Rust and C on Multicore Architectures: Case Study in N-Body
- **Authors**: Manuel Costanzo, Enzo Rucci, Marcelo Naiouf, Armando De Giusti
- **Year**: 2021
- **Venue**: 2021 Latin American Computing Conference (CLEI); arXiv:2107.11912 [cs.PL]
- **Contribution**: Demonstrates through an N-Body HPC case study that Rust delivers performance comparable to C on multicore architectures while substantially reducing programming effort, supporting Rust as a viable C alternative for HPC workloads.

---

### Shor 1995 - Polynomial-Time Algorithms for Prime Factorization.pdf
- **Full title**: Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer
- **Authors**: Peter W. Shor
- **Year**: 1995 (arXiv v2, January 1996; preliminary version FOCS 1994)
- **Venue**: arXiv:quant-ph/9508027v2; published in SIAM Journal on Computing 26(5):1484–1509, 1997
- **Contribution**: The foundational arXiv preprint establishing polynomial-time quantum algorithms for integer factorisation and discrete logarithms, challenging Church's thesis in the quantitative sense and introducing the quantum Fourier transform as the key subroutine.

---

### Shor_algorithm_quant-ph_0010034.pdf
- **Full title**: A Lecture on Shor's Quantum Factoring Algorithm, Version 1.1
- **Authors**: Samuel J. Lomonaco, Jr.
- **Year**: 2000
- **Venue**: AMS Short Course lecture notes; arXiv:quant-ph/0010034
- **Contribution**: Provides a self-contained, pedagogically structured exposition of Shor's quantum factoring algorithm — covering number-theoretic prerequisites, the quantum phase estimation subroutine, and a worked example — making the algorithm accessible for course and reference use.

---

### Shor_original_1997_quant-ph_9508027.pdf
- **Full title**: Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer
- **Authors**: Peter W. Shor
- **Year**: 1997 (preliminary version FOCS 1994)
- **Venue**: SIAM Journal on Computing 26(5):1484–1509, 1997; arXiv:quant-ph/9508027
- **Contribution**: Proves that integer factorisation and discrete logarithm can be solved in polynomial time on a quantum computer, establishing the foundational exponential speedup over the best known classical algorithms and motivating the field of quantum algorithm design.

---

### SupermarQ_2202.11045.pdf
- **Full title**: SupermarQ: A Scalable Quantum Benchmark Suite
- **Authors**: Teague Tomesh, Pranav Gokhale, Victory Omole, Gokul Subramanian Ravi, Kaitlin N. Smith, Joshua Viszlai, Xin-Chuan Wu, Nikos Hardavellas, Margaret R. Martonosi, Frederic T. Chong
- **Year**: 2022
- **Venue**: arXiv:2202.11045 [quant-ph]; IEEE HPCA 2022
- **Contribution**: Introduces a hardware-agnostic scalable quantum benchmark suite that uses application-level feature vectors (connectivity, parallelism, two-qubit gate ratio, etc.) to characterise circuit workloads and predict QPU performance across IBM, IonQ, and AQT platforms.

---

### Vedral 1996 - Quantum Networks for Elementary Arithmetic.pdf
- **Full title**: Quantum networks for elementary arithmetic operations
- **Authors**: Vlatko Vedral, Adriano Barenco, Artur Ekert
- **Year**: 1995 (submitted November 1995)
- **Venue**: Physical Review A 54(1):147–153, 1996; arXiv:quant-ph/9511018
- **Contribution**: Provides explicit space-optimised quantum network constructions for addition, modular addition, multiplication, and modular exponentiation — the core subroutine of Shor's algorithm — showing that the auxiliary memory required for reversible modular exponentiation grows only linearly with the size of the number to be factored.

---

### benchmark_performance_quantum_software_2409.08844.pdf
- **Full title**: Benchmarking the performance of quantum computing software
- **Authors**: Paul D. Nation, Abdullah Ash Saki, Sebastian Brandhofer, Luciano Bello, Shelly Garion, Matthew Treinish, Ali Javadi-Abhari
- **Year**: 2025 (submitted 2024)
- **Venue**: arXiv:2409.08844 [quant-ph] (IBM Quantum)
- **Contribution**: Presents Benchpress, an open-source suite of over 1,000 tests measuring circuit construction, manipulation, and transpilation performance across seven major quantum SDKs (Qiskit, Cirq, BQSKit, Braket, Tket, Staq, QTS) on circuits up to 930 qubits.

---

### benchmark_simulation_software_2401.09076.pdf
- **Full title**: Benchmarking Quantum Computer Simulation Software Packages: State Vector Simulators
- **Authors**: Amit Jamadagni, Andreas M. Läuchli, Cornelius Hempel
- **Year**: 2024
- **Venue**: arXiv:2401.09076 [quant-ph] (PSI / ETH Zürich)
- **Contribution**: Systematically benchmarks 24 statevector simulation packages on an HPC cluster across three canonical quantum tasks, revealing performance differences of over two orders of magnitude and identifying the crossover point (~25–30 qubits) where exponential scaling dominates.

---

### classical_simulation_herculean_2302.08880.pdf
- **Full title**: A Herculean task: Classical simulation of quantum computers
- **Authors**: Xiaosi Xu, Simon Benjamin, Jinzhao Sun, Xiao Yuan, Pan Zhang
- **Year**: 2023
- **Venue**: arXiv:2302.08880 [quant-ph]
- **Contribution**: Reviews the state-of-the-art classical simulation methods (statevector, density matrix, MPS, tensor network, stabiliser) with a detailed complexity comparison, discussing practical use cases and the fundamental limits that define the boundary between classically tractable and intractable quantum circuits.

---

### scalable_parallel_simulation_CPU_GPU_2509.04955.pdf
- **Full title**: Scalable Parallel Simulation of Quantum Circuits on CPU and GPU Systems
- **Authors**: Guolong Zhong, Yi Fan, Zhenyu Li
- **Year**: 2025
- **Venue**: arXiv:2509.04955 [quant-ph] (University of Science and Technology of China)
- **Contribution**: Presents Q²Chemistry's full-amplitude simulator with batch-buffered overlap processing, dependency-aware gate contraction, and staggered multi-gate parallelism, consistently outperforming open-source state-of-the-art simulators on HPC CPU and GPU platforms.

---

### statevector_gpu_tensor_2401.06188.pdf
- **Full title**: State of practice: evaluating GPU performance of state vector and tensor network methods
- **Authors**: Marzio Vallero, Paolo Rech, Flavio Vella
- **Year**: 2025 (submitted 2024)
- **Venue**: arXiv:2401.06188 [quant-ph] (University of Trento)
- **Contribution**: Evaluates GPU performance of statevector and tensor-network simulation on eight quantum circuit subroutines using CINECA's Leonardo supercomputer, correlating circuit topological features with simulation time and showing that circuit-aware strategy selection can improve performance by up to an order of magnitude.
