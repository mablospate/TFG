[QASMBench document](PDFs/QASMBench.pdf) ([Li, 2020])

[Quantum simulation frameworks list](https://github.com/bramathon/awesome-quantum-computing)

# Benchmarking variables / measures
Since I'm implementing the same algorithms and circuits in every simulator, inherent characteristics of the algorithms such as gate density or circuit depth won't be benchmarked. Instead I'll benchmark other metrics that may vary with each simulator ([Lubinski, 2023]):
- CPU usage over time
- GPU usage over time
- Time
- Memory use
- Lines of code

# Benchmarking targets
It turns out that I have a list of different components that I don't know yet how to structure into tests, so I will just dump it here until further guidance

## Frameworks

Full-stack software frameworks for constructing circuits and executing on quantum computing hardware.

* [Qiskit](https://github.com/Qiskit) - IBM's open-source SDK for working with quantum computers at the level of extended quantum circuits, operators, and primitives.
  * TEST THIS
* QDisLib
  * TEST THIS
  * Runs in local through Qiskit-Aer and CudaQ
* [Cirq](https://github.com/quantumlib/Cirq) - Google's python framework for creating, editing, and invoking Noisy Intermediate Scale Quantum (NISQ) circuits.
  * Runs in local![alt text](image.png)
* [Pyquil](https://github.com/rigetti/pyquil) - Rigetti's python library for constructing and executing programs using Quil.
* [Pennylane](https://github.com/PennyLaneAI/pennylane) - Xanadu's cross-platform Python library for quantum computing, quantum machine learning, and quantum chemistry.
* [qBraid](https://github.com/qBraid/qBraid) - A platform-agnostic quantum runtime framework
* [Ocean](https://github.com/dwavesystems/dwave-ocean-sdk) - SDK for D-Wave's quantum annealers.
* [Bloqade](https://github.com/QuEraComputing/Bloqade.jl) - Package for the quantum computation and quantum simulation based on Quera's neutral-atom architecture.
* [Qadence](https://github.com/pasqal-io/qadence) - Digital-analog quantum programming interface.
* [Braket](https://github.com/amazon-braket/amazon-braket-sdk-python) - A Python SDK for interacting with quantum devices on Amazon Braket.
* [Cuda-Q](https://github.com/NVIDIA/cuda-quantum) - C++ and Python support for the CUDA Quantum programming model for heterogeneous quantum-classical workflows.
  * TEST THIS
* [ProjectQ](https://github.com/ProjectQ-Framework/ProjectQ) - ProjectQ: An open source software framework for quantum computing.
  
## Simulators

Simulators for quantum circuits. Determine the output of a quantum circuit using a classical computer.

* [QVM](https://github.com/quil-lang/qvm) - A high-performance state-vector simulator for Quil programs. Written in common-lisp.
* [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) - Qiskit simulation package. Provides an interface to numerous backends including state-vector, tensor-network and matrix-product-state simulators. Supports noise and GPUs depending on the backend.
* [Qsim](https://github.com/quantumlib/qsim) - A high-performance state-vector simulator which supports Cirq. Supports noise and GPUs.
* [Pennylane Lightning](https://github.com/PennyLaneAI/pennylane-lightning) - High-performance state-vector simulator written in C++ for use with Pennylane. Offers some unique features around GPUs and differentiation.
* [Qutip-qip](https://github.com/qutip/qutip-qip) - Circuit simulation extension of the general-purpose Qutip package. Offers unique features around noise and pulse-level simulation.
* [CuQuantum](https://github.com/NVIDIA/cuQuantum) - Nvidia's cuQuantum sdk which provides GPU-based simulation of quantum circuits. State-vector and tensor-network are supported. Fairly low-level and can used as a backend by Qiskit Aer and others.
* [qHiPSTER](https://github.com/intel/intel-qs) - Intel's high-performance quantum simulator. State-vector simulator highly optimized for multiple CPUs.
* [QRack](https://github.com/unitaryfund/qrack) - Comprehensive, GPU accelerated framework from the unitary fund.
* [ITensor](https://github.com/ITensor/ITensors.jl) - State-of-the-art tensor network simulations written in Julia.
* [QuEST](https://github.com/QuEST-Kit/QuEST) - A multithreaded, distributed, GPU-accelerated simulator of quantum computers
* [Qulacs](https://github.com/qulacs/) - Qulacs is a Python/C++ library for fast simulation of large, noisy, or parametric quantum circuits.
* [Dynamiqs](https://github.com/dynamiqs/dynamiqs) - High-performance quantum systems simulation with JAX
* [Qibo](https://github.com/qiboteam/qibo) - Efficient simulation backends with GPU, multi-GPU and CPU with multi-threading support.
* [Quimb](https://github.com/jcmgray/quimb) - A python library for quantum information and many-body calculations including tensor networks.
* [Stim](https://github.com/quantumlib/Stim) -  A fast stabilizer circuit library. 

## Compilers

Compilers transform logical quantum circuits 

* [QuilC](https://github.com/quil-lang/quilc) - A full-stack compiler written in common lisp for the Quil language.qis
* [Qiskit.compiler](https://github.com/Qiskit/qiskit/tree/main/qiskit/compiler) - Qiskit's built-in full stack compiler with dozens of compilation passes.
* [Tket](https://github.com/CQCL/tket) - C++-based full-stack quantum compiler written by Quantinuum.
* [BQSKit](https://github.com/BQSKit/bqskit) - Berkeley Quantum Synthesis Toolkit is a full-stack quantum compiler that uses numerical and approximate methods to find high-performing compilations.
* [Cirq.transformers](https://github.com/quantumlib/Cirq/tree/main/cirq-core/cirq/transformers) - Cirq's built-in transformers offer a variety of compilation passes, but does not include routing.
* [PyZX](https://github.com/zxcalc/pyzx) - Python library for quantum circuit rewriting and optimisation using the ZX-calculus 

## Error Mitigation

Error mitigation can mitigate the impact of errors in quantum devices.

* [Mitiq](https://github.com/unitaryfund/mitiq) - An open source toolkit for implementing error mitigation techniques on most current intermediate-scale quantum computers from the Unitary Fund.
* [Qermit](https://github.com/CQCL/Qermit) - Python module for running error-mitigation protocols with the pytket quantum SDK
* [PyIBU](https://github.com/sidsrinivasan/PyIBU) - A scalable implementation of iterative Bayesian unfolding for measurement error mitigation.
* [AutomatedPERTools](https://github.com/benmcdonough20/AutomatedPERTools) - Autonomous error mitigation based on probabilistic error reduction.

## Optimal Control

Libraries for controling quantum systems and realizing gates.

* [Piccolo](https://github.com/kestrelquantum/Piccolo.jl) - Quantum optimal control using the Pade Integrator COllocation (PICO) method.
* [SCQubits](https://github.com/scqubits/scqubits) - An open-source Python library for simulating superconducting qubits.
* [C3](https://github.com/q-optimize/c3) - An integrated tool-set for Control, Calibration and Characterization.
* [Quandary](https://github.com/LLNL/quandary) - Quandary implements an optimization solver for open and closed optimal quantum control.
* [Qiskit Dynamics](https://github.com/Qiskit-Extensions/qiskit-dynamics) - Qiskit Dynamics is an open-source project for building, transforming, and solving time-dependent quantum systems in Qiskit.
* [Qibo](https://github.com/qiboteam/qibo) - Qibo is an open-source full stack API for quantum simulation and quantum hardware control.
  
# Benchmarking procedures
These are all the quantum algorithms and circuits that I'll be benchmarking, divided in scale, as per [QASMBench] ([Li, 2020]).

## Small-scale

### Full Adder
[Explanation](https://www.quantum-inspire.com/kbase/full-adder/)

Uses qubits as regular bits with the purpose to add numbers together. ([Draper, 2000] for QFT-based adder; [Vedral, 1996] for ripple-carry adder)

<img src="Images/Full Adder.png">

### ~~Basis Change~~ • Quantum chemistry benchmark
### ~~Basis Trotter~~ • Quantum chemistry benchmark
### Bell State
[Explanation](https://en.wikipedia.org/wiki/Bell_state)

Algorithm necessary to generate the Bell State, entangling the qubits involved.

<img src= "Images/Bell State.png">

### ~~Cat State~~ MAYBE • Just the Haddamard gate
[Explanation](https://en.wikipedia.org/wiki/Cat_state)
### Deutsch-Josza
[Explanation](https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm)

Algorithm designed to test if an oracle's output is constant or balanced.

<img src="Images/Deutsch-Josza.png">

### ~~Deep Neural Networks (DNN)~~ • Non-applicable, not an algorithm
### ~~Fredkin~~ MAYBE • Just the CSWAP gate
[Explanation](https://en.wikipedia.org/wiki/Fredkin_gate)
### ~~Quantum Error Correction (QEC)~~ MAYBE ADDED TO OTHERS • Not a circuit per se, more like extending other algorithms to introduce redundancy
[Explanation](https://en.wikipedia.org/wiki/Quantum_error_correction)
### Grover
[Explanation](https://en.wikipedia.org/wiki/Grover%27s_algorithm)

Oracle-based quantum search algorithm. ([Grover, 1996]; MCZ decomposition via [Barenco, 1995])

<img src="Images/Grover.png">

### Hidden subgroup problems (HS4)
### QFT & Inverse QFT ([Draper, 2000])
### iSWAP
### LinearSolver
### Learning parity with noise (LPN)
### Quantum phase-estimation (QPE / Pea) ([Shor, 1995])
### Quantum approximate optimization algorithm (QAOA)
### QuantumWalks
### Quantum error corrections (QEC) SM & EN
### Quantum random number generation (QRNG)
### Shor ([Shor, 1995]; circuit: [Beauregard, 2002]; arithmetic: [Draper, 2000], [Vedral, 1996])
### Toffoli ([Barenco, 1995])
### Teleportation
### Variational quantum (VQE)
### W state

## Medium-scale

### BB84
### Bernstein-Vazirani (BV)
### Ising
### Multiplier ([Vedral, 1996]; [Beauregard, 2002])
### Boolean Satisfiability Problem (SAT)
### Simon

## Large-scale

### Counterfeit coin (CC)
### Greenberger-Horne-Zeilinger (GHZ) state
### Squarer root
### Swap test

---

# Reference keys

- [Shor, 1995]: Shor, P. W. — *Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer.* SIAM Review. arXiv: [quant-ph/9508027](https://arxiv.org/abs/quant-ph/9508027)
- [Grover, 1996]: Grover, L. K. — *A fast quantum mechanical algorithm for database search.* STOC. arXiv: [quant-ph/9605043](https://arxiv.org/abs/quant-ph/9605043)
- [Beauregard, 2002]: Beauregard, S. — *Circuit for Shor's algorithm using 2n+3 qubits.* QIC. arXiv: [quant-ph/0205095](https://arxiv.org/abs/quant-ph/0205095)
- [Draper, 2000]: Draper, T. G. — *Addition on a quantum computer.* arXiv: [quant-ph/0008033](https://arxiv.org/abs/quant-ph/0008033)
- [Vedral, 1996]: Vedral, V., Barenco, A. & Ekert, A. — *Quantum networks for elementary arithmetic operations.* Phys. Rev. A. arXiv: [quant-ph/9511018](https://arxiv.org/abs/quant-ph/9511018)
- [Barenco, 1995]: Barenco, A. et al. — *Elementary gates for quantum computation.* Phys. Rev. A. arXiv: [quant-ph/9503016](https://arxiv.org/abs/quant-ph/9503016)
- [Lubinski, 2023]: Lubinski, T. et al. — *Application-oriented performance benchmarks for quantum computing.* IEEE TQE. arXiv: [2110.03137](https://arxiv.org/abs/2110.03137)
- [Li, 2020]: Li, A. et al. — *QASMBench: A low-level QASM benchmark suite for NISQ evaluation and simulation.* arXiv: [2005.13018](https://arxiv.org/abs/2005.13018)
- [QASMBench]: See [Li, 2020]