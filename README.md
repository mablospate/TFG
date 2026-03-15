# Libraries to benchmark
## Python
- qiskit
- cuda-q
- CirQ
- QDisLib
- ProjectQ
## Rust
- QCGPU
- Quantrs
- quantr
- q1tsim

# Roadmap
## Investigation 1
1. Create full list of libraries to benchmark
2. Learn how to implement first round of algorithms in all libraries to run in local
## First round of algorithms
- Shor using premade implementation if possible
- Grover using premade implementation if possible
## Benchmarks & result processing
- Parametrize algorithms
  - Take into account setup time, implementation time, correlation with input sizes
  - Parameters
    - Input Size
    - Maybe pick parameters from QASMBench doc
## Report results
- Write memory up to this point
# Extra roadmap (if time allows)
## Increase scope
- Add all possible parameters from QASMBench doc to the existing investigation
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