# Plan de implementacion de Grover y Shor en todos los frameworks

## Interfaz comun obligatoria

Todas las implementaciones deben exponer la misma interfaz publica que las de Qiskit,
para poder llamarlas de forma uniforme desde el benchmarking.

### Grover — interfaz de referencia (python/qiskit/grover.py)

```python
def build_oracle(n: int, target: int) -> Circuit
def build_diffuser(n: int) -> Circuit
def grover_circuit(n: int, target: int, num_iterations: int | None = None) -> Circuit
def search(
    n: int,
    target: int,
    sampler,            # primitiva de sampling del framework
    pass_manager,       # compilador/transpiler del framework (None si no aplica)
    num_iterations: int | None = None,
    num_shots: int = 1024,
) -> tuple[int, dict[str, int]]
```

### Shor — interfaz de referencia (python/qiskit/shor/shor.py)

```python
def order_finding_circuit(A: int, N: int, precision: int | None = None) -> Circuit
def find_order(
    A: int,
    N: int,
    sampler,
    pass_manager,
    precision: int | None = None,
    num_shots: int = 10,
) -> tuple[int, dict[str, int]]
def find_factor(
    N: int,
    sampler,
    pass_manager,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    seed: int | None = None,
) -> int
```

### Reglas
- **Todo debe correr en local** (sin cloud, sin hardware cuantico real)
- `sampler` y `pass_manager` son especificos de cada framework (ver seccion de ejecucion local por framework)
- En frameworks sin transpiler (Rust, ProjectQ con emulacion), `pass_manager` puede ser `None` o una funcion identidad
- Los return types deben ser equivalentes: `int` para el resultado, `dict[str, int]` para la distribucion de medidas
- En Rust, la interfaz equivalente seria funciones con las mismas firmas adaptadas a tipos Rust (`HashMap<String, usize>` para la distribucion)
- `Circuit` es el tipo de circuito nativo del framework (QuantumCircuit en Qiskit, cirq.Circuit en Cirq, etc.)

## Ejecucion local por framework

Todos los frameworks deben ejecutarse en local con simuladores. Sin acceso a cloud ni QPUs reales.

### Python

| Framework | Simulador local | Instalacion | sampler | pass_manager |
|---|---|---|---|---|
| **Qiskit** | Qiskit Aer (statevector) | `pip install qiskit-aer` | `SamplerV2` de `qiskit_aer.primitives` | `generate_preset_pass_manager(backend=AerSimulator())` |
| **Cirq** | `cirq.Simulator` (built-in) | `pip install cirq` (incluido). Para alto rendimiento: `pip install qsimcirq` | `cirq.Simulator().run(circuit, repetitions=n)` | `cirq.optimize_for_target_gateset(circuit, gateset=...)` o `None` |
| **CUDA-Q** | `qpp-cpu` (CPU, built-in) o `nvidia` (GPU si hay CUDA) | `pip install cuda-quantum` | `cudaq.sample(kernel, shots_count=n)` | No aplica (compilacion via MLIR integrada en el kernel) |
| **ProjectQ** | `Simulator` (C++ con OpenMP, built-in) | `pip install projectq` | `eng.flush()` + `int(qubit)` | Pipeline de engines configurable: `AutoReplacer`, `LocalOptimizer`, etc. |
| **QDisLib** | Qiskit Aer (como backend de subcircuitos) | `pip install qdislib` + Qiskit Aer | Misma interfaz que Qiskit (circuitos Qiskit como input) | Misma interfaz que Qiskit |

### Rust

| Framework | Simulador | Instalacion | Ejecucion |
|---|---|---|---|
| **q1tsim** | Statevector (built-in) + Stabilizer | `q1tsim = "0.5"` en Cargo.toml | `circuit.execute(shots)` + `circuit.histogram()` |
| **qcgpu** | Statevector GPU (OpenCL) | `qcgpu = "0.1"` en Cargo.toml. Requiere runtime OpenCL | `state.measure_many(shots)` |
| **quantr** | Statevector (built-in) | `quantr = "0.6"` en Cargo.toml | `circuit.simulate()` + `simulated.measure_all(shots)` |
| **quantrs** | Verificar disponibilidad | `quantrs = "0.1"` en Cargo.toml | Verificar API |

### Notas sobre ejecucion local
- **CUDA-Q sin GPU**: por defecto usa `qpp-cpu` (simulador C++ con OpenMP). Seleccionar con `cudaq.set_target("qpp-cpu")`
- **qcgpu sin GPU**: requiere OpenCL runtime. En macOS viene incluido. En Linux: `apt install ocl-icd-opencl-dev`
- **ProjectQ emulacion**: el simulador puede hacer shortcuts clasicos para math gates. Para benchmarking justo, forzar descomposicion con `InstructionFilter`
- **Cirq + qsim**: `pip install qsimcirq` para simulador C++ optimizado. Drop-in replacement: `qsimcirq.QSimSimulator()` en vez de `cirq.Simulator()`

## Estado actual

| Framework | Shor | Grover |
|---|---|---|
| Qiskit (Python) | ✅ Beauregard completo | ✅ MCZ + diffuser |
| Cirq (Python) | ❌ | ❌ |
| CUDA-Q (Python) | ❌ | ❌ |
| ProjectQ (Python) | ❌ | ❌ |
| QDisLib (Python) | ❌ | ❌ |
| q1tsim (Rust) | ❌ | ❌ |
| qcgpu (Rust) | ❌ | ❌ |
| quantr (Rust) | ❌ | ❌ |
| quantrs (Rust) | ❌ | ❌ |

---

## Algoritmo de Grover — detalles de implementacion

### Estructura del circuito
1. Preparar superposicion uniforme: H en todos los n qubits
2. Repetir floor(pi/4 * sqrt(2^n)) veces:
   - **Oraculo**: flip de fase del estado target |t> usando MCZ
   - **Difusor**: inversion sobre la media (2|s><s| - I)
3. Medir todos los qubits

### Oraculo (fase flip en |target>)
- Aplicar X en qubits donde target tiene bit 0
- Aplicar Multi-Controlled Z (MCZ) en todos los qubits
- Deshacer las X
- Referencia: [Barenco, 1995] para descomposicion del MCZ

### Difusor (inversion sobre la media)
- H en todos los qubits
- X en todos los qubits
- MCZ en todos los qubits
- X en todos los qubits
- H en todos los qubits
- Referencia: [Grover, 1996]

### Componente critico: Multi-Controlled Z (MCZ)
- Es la unica puerta no trivial del circuito
- Su descomposicion depende de las capacidades de cada framework
- Frameworks con MCZ nativo: delegar al transpiler
- Frameworks sin MCZ: descomponer manualmente en Toffoli + ancillas, o en CZ + single-qubit gates

---

## Algoritmo de Shor — detalles de implementacion

### Estructura del circuito (Beauregard)
1. Registro de control (m qubits) en superposicion: H en todos
2. Registro target (n qubits) inicializado en |1>
3. Exponenciacion modular controlada: |x>|y> -> |x>|A^x * y mod N>
4. QFT inversa en registro de control
5. Medir registro de control
6. Post-procesamiento clasico: fracciones continuas para extraer el orden r

### Jerarquia de aritmetica modular (Beauregard/Draper)
```
add_classical (suma clasica en espacio de Fourier)
  -> c_add_classical (controlada)
    -> add_classical_modulo (modulo N, con ancilla)
      -> c_add_classical_modulo (controlada)
        -> add_quantum_modulo (suma cuantica modular)
          -> c_add_quantum_modulo (controlada)
            -> multiply_modulo (multiplicacion modular in-place)
              -> c_multiply_modulo (controlada)
                -> exponentiate_modulo (A^x mod N)
```

### Componentes criticos
- **QFT**: necesaria para sumas en espacio de Fourier y para phase estimation
- **Controlled phase gates**: CRk (rotacion de fase 2pi/2^k) para QFT
- **Toffoli / multi-controlled X**: para aritmetica modular
- **SWAP controlado**: para multiplicacion modular in-place
- Referencias: [Beauregard, 2002], [Draper, 2000], [Vedral, 1996]

### Variante simplificada para frameworks limitados
Para frameworks sin controlled-phase gates o QFT:
- Usar sumador ripple-carry de [Vedral, 1996] (solo Toffoli + CNOT)
- Mas qubits (O(n) ancillas) pero solo necesita puertas basicas
- Adecuado para: frameworks Rust con gate sets limitados

### Variante single-control (2n+3 qubits)
- Un solo qubit de control reutilizado con medidas secuenciales
- Requiere classical feedforward (mid-circuit measurement + condicionales)
- Solo viable en frameworks con soporte de circuitos dinamicos

---

## Implementacion por framework

---

### 1. Cirq (Google)

**Puntos fuertes para benchmarking:**
- Simulador qsim (C++ optimizado) como drop-in replacement de cirq.Simulator
- GPU via cuStateVec/cuQuantum: hasta 30x speedup sobre CPU
- Modelo de momentos (Moment-based): control explicito sobre paralelismo de puertas
- Transformers de compilacion configurables (CZTargetGateset, SqrtIswapTargetGateset)
- Soporte de ruido con calibracion real de procesadores Google

**Grover:**
- `gate.controlled(num_controls=n)` para MCZ: `cirq.Z.controlled(num_controls=n-1)`
- `cirq.H.on_each(*qubits)` para aplicar H a todos los qubits de golpe
- Compose como operaciones en momentos para maximo paralelismo
- Ejemplo oficial en `examples/grover.py` como referencia

**Shor:**
- `cirq.ArithmeticGate` para definir operaciones aritmeticas reversibles (modular exponentiation)
- `cirq.qft(*qubits)` built-in para QFT, con `inverse=True`
- `cirq.PhaseGradientGate` para sumas en espacio de Fourier
- `.controlled_by(qubit)` para controlar cualquier operacion
- Ejemplo oficial en `docs/experiments/shor.ipynb`
- Ventaja: ArithmeticGate permite definir la semantica matematica y Cirq genera el circuito

**Ejecucion local:**
```python
import cirq
# Simulador basico (built-in, sin dependencias extra)
sim = cirq.Simulator()
result = sim.run(circuit, repetitions=1024)
counts = result.histogram(key='result')
# Alto rendimiento (requiere pip install qsimcirq)
import qsimcirq
sim = qsimcirq.QSimSimulator()
result = sim.run(circuit, repetitions=1024)
```

**Ventaja competitiva:** qsim + GPU para simulaciones grandes. Moment-based scheduling para minimizar profundidad.

**Ficheros:** `python/cirq/grover.py`, `python/cirq/shor.py`

---

### 2. CUDA-Q (NVIDIA)

**Puntos fuertes para benchmarking:**
- GPU nativa via cuStateVec: hasta 425x speedup sobre CPU
- Multi-GPU (nvidia-mgpu): pooling de memoria para >33 qubits
- Kernel compilation via MLIR/LLVM: optimizacion a nivel de compilador
- Gate fusion automatica configurable
- Multiples backends: statevector, tensor network, MPS

**Grover:**
- `z.ctrl([ctrl1, ctrl2, ..., ctrln], target)` para MCZ nativo
- `cudaq.compute_action(compute_fn, action_fn)` para el patron U-V-U† del difusor
  - compute_fn: H + X
  - action_fn: MCZ
  - CUDA-Q invierte automaticamente el compute_fn
- Iteration count: `round(0.25 * pi * sqrt(2^n))`

**Shor:**
- `cudaq.adjoint(kernel)` para invertir kernels enteros (QFT inversa)
- `cudaq.control(kernel, controls)` para controlar sub-circuitos completos
  - Critico: permite controlar la exponenciacion modular entera, no gate por gate
- `r1(angle, qubit)` para rotaciones de fase (QFT)
- `cudaq.register_operation("name", matrix)` para puertas custom
- Kernel composition: definir QFT, modular_mult, etc. como kernels separados

**Ejecucion local:**
```python
import cudaq
# CPU (sin GPU) — por defecto si no hay CUDA
cudaq.set_target("qpp-cpu")
# GPU (si hay NVIDIA + CUDA)
cudaq.set_target("nvidia")

@cudaq.kernel
def my_kernel(n: int):
    qubits = cudaq.qvector(n)
    h(qubits[0])
    mz(qubits)

result = cudaq.sample(my_kernel, n, shots_count=1024)
print(result)  # dict-like con counts
```

**Ventaja competitiva:** GPU masiva. compute_action para Grover. control(kernel) para Shor. Tensor network backend para circuitos muy anchos.

**Limitaciones:**
- @cudaq.kernel no permite llamadas Python arbitrarias dentro
- Tensor network backends no soportan mid-circuit measurement
- No hay aritmetica modular built-in, hay que implementarla

**Ficheros:** `python/cudaq/grover.py`, `python/cudaq/shor.py`

---

### 3. ProjectQ (ETH Zurich)

**Puntos fuertes para benchmarking:**
- Shor's INCLUIDO como ejemplo completo con aritmetica modular built-in
- `projectq.libs.math`: AddConstant, AddConstantModN, MultiplyByConstantModN
- Emulacion: el simulador puede ejecutar math gates sin descomponerlas (shortcut clasico)
- Automatic uncomputation con Compute/Uncompute
- Pipeline de compilacion modular con 16 reglas de descomposicion
- QFT y QPE como gates built-in
- QAA (Quantum Amplitude Amplification) como gate built-in (generalizacion de Grover)
- Simulador C++ con OpenMP, gate fusion y SIMD

**Grover:**
- `QAA(algorithm, oracle)` gate built-in para amplitude amplification
- `C(gate, n)` para multi-controlled gates: `C(Z, n-1)` para MCZ
- `with Control(eng, ctrl_qubits): Z | target` para controlar bloques enteros
- `All(H) | qureg` para aplicar H a todo el registro
- `with Compute(eng): ... Uncompute(eng)` para el difusor

**Shor:**
- `MultiplyByConstantModN(a, N) | x` directamente disponible
- `with Control(eng, ctrl): MultiplyByConstantModN(a, N) | x` para exponenciacion controlada
- Semi-classical QFT con medidas secuenciales ya implementada en el ejemplo
- El simulador puede emular la multiplicacion modular clasicamente (sin descomposicion cuantica)
- `ResourceCounter` backend para contar puertas sin ejecutar

**Ventaja competitiva:** Maxima productividad. Math library built-in. Emulacion clasica para Shor permite factorizar numeros grandes en laptop. QAA built-in para Grover.

**Ejecucion local:**
```python
from projectq import MainEngine
from projectq.ops import H, Measure, All
from projectq.backends import Simulator

eng = MainEngine(backend=Simulator(), engine_list=[])
qureg = eng.allocate_qureg(n)
All(H) | qureg
# ... aplicar puertas ...
All(Measure) | qureg
eng.flush()
results = [int(q) for q in qureg]
```

**Atencion para benchmarking:**
- La emulacion clasica del simulador hace trampa: no descompone las math gates
- Para benchmarking justo, usar `InstructionFilter` para forzar descomposicion completa
- O usar `ResourceCounter` para contar las puertas que SE USARIAN

**Ficheros:** `python/projectq/grover.py`, `python/projectq/shor.py`

---

### 4. QDisLib (BSC)

**Puntos fuertes para benchmarking:**
- Distribucion de circuitos en HPC via PyCOMPSs
- Circuit cutting: wire cutting (8^k subcircuitos) y gate cutting (6^k subcircuitos)
- Ejecucion hibrida CPU/GPU/QPU simultanea
- FindCut: particionamiento automatico de circuitos con algoritmos de grafos
- Demostrada escalabilidad: 54.4x speedup en 64 nodos para circuito de 96 qubits

**Grover:**
- Implementar el circuito en Qiskit (ya lo tenemos)
- Pasar a QDisLib para distribucion
- Para n grande: usar circuit cutting para dividir el oraculo MCZ
- La ventaja esta en la ejecucion distribuida, no en la construccion del circuito

**Shor:**
- Implementar el circuito en Qiskit (ya lo tenemos)
- Pasar a QDisLib para circuit cutting de la exponenciacion modular
- La exponenciacion modular es el candidato ideal para cutting: muchas puertas de 2 qubits

**Ventaja competitiva:** Escalabilidad HPC. La unica libreria que puede distribuir un circuito grande en multiples nodos/GPUs/QPUs.

**Ejecucion local:**
```python
# QDisLib usa circuitos Qiskit como input y Qiskit-Aer como simulador local
# Sin PyCOMPSs, ejecuta los subcircuitos secuencialmente en local
from qdislib import QDisLib
from qiskit_aer import AerSimulator

# Construir circuito en Qiskit (reutilizar implementacion existente)
qc = grover_circuit(n, target)  # del modulo qiskit

# Ejecutar via QDisLib con Aer como backend
backend = AerSimulator()
# ... API de QDisLib para cutting y ejecucion
```

**Limitaciones:**
- Overhead exponencial del cutting (6^k o 8^k)
- No aporta ventaja para circuitos pequenos
- Requiere PyCOMPSs para paralelismo real (sin PyCOMPSs, ejecucion secuencial)
- Actualmente solo soporta Qiskit y Qibo como input

**Ficheros:** `python/qdislib/grover.py`, `python/qdislib/shor.py`

---

### 5. q1tsim (Rust)

**Puntos fuertes para benchmarking:**
- Gate set mas completo de las librerias Rust: 35+ puertas incluyendo CCX, CCZ, CU1, CRZ
- Simulador stabilizer para circuitos Clifford (miles de qubits)
- Export a OpenQASM, c-QASM y LaTeX
- Medida en bases X, Y, Z (no solo computacional)

**Grover:**
- CCX (Toffoli) nativo para construir oraculo
- CCZ nativo para MCZ de 3 qubits
- Para MCZ de n>3 qubits: anidar `C<G>` o descomponer en Toffolis + ancillas
- H, X, Z todos disponibles para difusor

**Shor:**
- CU1, CRZ, CRX, CRY disponibles para QFT
- CCX para aritmetica modular
- Swap nativo
- Hay que implementar toda la jerarquia de Beauregard manualmente
- Alternativa: Vedral ripple-carry adder usando solo CCX y CX

**Ejecucion local:**
```rust
use q1tsim::circuit::Circuit;
let mut circuit = Circuit::new(n, n); // n qubits, n classical bits
circuit.h(0);
circuit.cx(0, 1);
circuit.measure_all(&[0, 1]);
circuit.execute(1024);       // 1024 shots
let hist = circuit.histogram_string().unwrap();
```

**Ventaja competitiva:** Gate set rico. Export a OpenQASM para verificacion cruzada con otros simuladores.

**Limitaciones:** Proyecto inactivo (2019). Sin GPU. Sin optimizacion de circuitos.

**Ficheros:** `rust/q1tsim/src/bin/grover.rs`, `rust/q1tsim/src/bin/shor.rs`

---

### 6. qcgpu (Rust)

**Puntos fuertes para benchmarking:**
- GPU via OpenCL: la unica libreria Rust con aceleracion GPU
- `pow_mod(x, n, input_width, output_width)` built-in para Shor
- Ejemplos de Grover y Shor incluidos en el repositorio

**Grover:**
- `toffoli(c1, c2, target)` nativo
- `apply_controlled_gate(control, target, gate)` para controlled-Z
- MCZ de n>3 qubits: descomponer en Toffolis
- H, X, Z disponibles

**Shor:**
- `pow_mod` built-in para exponenciacion modular
- `measure_first(n, iterations)` para medir solo qubits de control
- `add_scratch(n)` para ancillas dinamicas
- Rotaciones de fase via `r(angle)` + `apply_controlled_gate`

**Ejecucion local:**
```rust
use qcgpu::State;
let mut state = State::new(n, 0); // n qubits, device 0 (OpenCL)
state.h(0);
state.cx(0, 1);
let result = state.measure_many(1024); // 1024 shots
// Requiere OpenCL runtime instalado (macOS: incluido, Linux: apt install ocl-icd-opencl-dev)
```

**Ventaja competitiva:** GPU acceleration + pow_mod built-in para Shor. Unica libreria Rust que escala con hardware.

**Limitaciones:** Precision f32 (no f64). API imperativa sin abstraccion de circuito. Proyecto inactivo (2018). Solo single-control nativo.

**Ficheros:** `rust/qcgpu/src/bin/grover.rs`, `rust/qcgpu/src/bin/shor.rs`

---

### 7. quantr (Rust)

**Puntos fuertes para benchmarking:**
- Proyecto activo (el unico de los Rust)
- `CRk(k, control)` nativo: exactamente la puerta necesaria para QFT (rotacion 2pi/2^k)
- Custom gates via funciones Rust (flexible)
- Minimas dependencias (fastrand, num-complex)
- Diagramas de circuito en terminal

**Grover:**
- Toffoli(c1, c2) nativo
- CZ(control) nativo
- Para MCZ de n>3: custom gate con funcion de mapeo
- H, X, Z disponibles

**Shor:**
- `CRk(k, control)` es la pieza clave: permite QFT directa sin calcular angulos manualmente
- Toffoli para aritmetica
- Custom gates para operaciones modulares complejas

**Ejecucion local:**
```rust
use quantr::circuit::Circuit;
use quantr::gates::Gate;
let mut circuit = Circuit::new(n).unwrap();
circuit.add_gates(&[Gate::H, Gate::X]).unwrap();
circuit.add_gate(Gate::CNot(0), 1).unwrap();
let simulated = circuit.simulate();
let measurements = simulated.measure_all(1024);
```

**Ventaja competitiva:** CRk nativo (ideal para QFT). Unica libreria Rust activamente mantenida.

**Limitaciones:** Limite practico de ~16 qubits. Single-threaded. Sin GPU. Sin ruido. Inestable (los autores recomiendan validar resultados).

**Ficheros:** `rust/quantr/src/bin/grover.rs`, `rust/quantr/src/bin/shor.rs`

---

### 8. quantrs (Rust)

**Estado:** El repositorio github.com/Entropy-Foundation/quantrs NO EXISTE (404). Existen dos proyectos con nombres similares:

- `quantrs` en crates.io (v0.1.8): libreria de finanzas cuantitativas, NO computacion cuantica
- QuantRS2 (cool-japan/quantrs): proyecto muy temprano (v0.1.2, build falla en docs.rs)

**Recomendacion:** Verificar cual es el proyecto correcto antes de implementar. Si es cool-japan/quantrs, investigar si realmente funciona antes de invertir tiempo.

**Ficheros:** `rust/quantrs/src/bin/grover.rs`, `rust/quantrs/src/bin/shor.rs`

---

## Orden de implementacion recomendado

### Fase 1: Python (maximo impacto)
1. **ProjectQ** — tiene math library y QAA built-in, implementacion mas rapida
2. **Cirq** — buen gate set, ejemplos oficiales, qsim para benchmarking
3. **CUDA-Q** — GPU nativa, pero sintaxis de kernel requiere adaptacion
4. **QDisLib** — wrapper sobre circuitos Qiskit, requiere infraestructura HPC

### Fase 2: Rust (mas trabajo manual)
5. **q1tsim** — mejor gate set, mas straightforward
6. **qcgpu** — GPU + pow_mod, interesante para Shor
7. **quantr** — CRk nativo, pero limite de 16 qubits
8. **quantrs** — investigar viabilidad primero

---

## Metricas de benchmarking por framework

Para cada implementacion, medir ([Lubinski, 2023], [QASMBench]):
- **Tiempo de construccion del circuito** (circuit building time)
- **Tiempo de transpilacion/compilacion** (si aplica)
- **Tiempo de simulacion** (ejecucion)
- **Memoria maxima** (RAM/VRAM)
- **CPU/GPU usage** (% utilizacion)
- **Profundidad del circuito transpilado** (circuit depth)
- **Numero de puertas de 2 qubits** (2q gate count)
- **Lineas de codigo** (complejidad de implementacion)
- **Fidelidad** (probabilidad del resultado correcto)

Parametrizar por:
- **Grover**: n = 3, 4, 5, ..., max_qubits del framework
- **Shor**: N = 15, 21, 33, 35, ... (numeros compuestos crecientes)
