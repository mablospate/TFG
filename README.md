# Libraries to benchmark
## Python
- qiskit
  - Shor implementation sourced from https://github.com/benjamin-assel/qiskit-shor (based on [Beauregard, 2002])
- cuda-q
- CirQ
- QDisLib ([QDisLib docs])
- ProjectQ
## Rust
- QCGPU ([Kelly, 2018])
- Quantrs
- quantr
- q1tsim

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

## Frameworks

- **Kelly, A.** (2018). *Simulating quantum computers using OpenCL (QCGPU).* arXiv: [1805.00988](https://arxiv.org/abs/1805.00988)
- **QDisLib.** *Distributed quantum computing library.* (Internal documentation)

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