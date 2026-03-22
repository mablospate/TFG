# Analisis de frameworks Python: bondades y beneficios por algoritmo

## Resumen de lineas de codigo

| Framework | Grover | Shor (total) | Total |
|---|---|---|---|
| Qiskit | 151 | 754 (adder 473 + qft 38 + shor 243) | 905 |
| Cirq | 161 | 272 (modular_exp 73 + shor 199) | 433 |
| CUDA-Q | 206 | 368 (permutation 130 + qft 41 + shor 197) | 574 |
| ProjectQ | 171 | 234 | 405 |
| QDisLib | 115 | 176 | 291 |

---

## Qiskit (IBM)

### Bondades generales
- **Ecosistema mas completo**: transpiler, primitivas (Sampler, Estimator), Aer, hardware real
- **QFTGate built-in** con soporte de aproximacion configurable
- **Pass manager** con niveles de optimizacion (0-3) que adaptan el circuito al backend
- **SamplerV2** como interfaz unificada para simulacion y hardware real
- **Mayor comunidad y documentacion** de todos los frameworks

### Beneficio en Grover
- `ZGate().control(n-1)` delega la descomposicion del MCZ al transpiler, que la optimiza para la topologia del backend
- `compose(circuit, inplace=True)` permite reutilizar subcircuitos sin copiarlos
- El transpiler puede fusionar puertas adyacentes entre iteraciones de Grover

### Beneficio en Shor
- **Maximo control sobre el circuito**: implementacion completa de Beauregard con jerarquia de aritmetica modular (adder → modular adder → multiplier → exponentiation)
- **QFT aproximada** via `QFTFullGate(approximation_degree=d)`: reduce profundidad eliminando rotaciones de angulo pequeno
- **Variante de 2n+3 qubits** con un solo qubit de control y medidas secuenciales
- Es la implementacion mas fiel al paper de Beauregard y la que mas puertas cuanticas genera realmente

### Desventaja
- **Mas codigo** (905 lineas): la jerarquia de aritmetica es compleja
- Requiere `qiskit-aer` como dependencia adicional para simulacion

---

## Cirq (Google)

### Bondades generales
- **Modelo de momentos**: los circuitos son secuencias de `Moment` (capas de puertas paralelas), dando control explicito sobre scheduling
- **`cirq.H.on_each(*qubits)` y `cirq.X.on_each(*qubits)`**: aplica puertas a multiples qubits en una sola linea — codigo mas conciso
- **`gate.controlled(num_controls=n)`**: cualquier puerta se hace multi-controlada en una linea
- **qsim** como simulador C++ drop-in: ordenes de magnitud mas rapido que `cirq.Simulator` para >20 qubits
- **Sin transpiler obligatorio**: los circuitos se ejecutan directamente en el simulador
- **Histograma nativo**: `result.histogram(key=...)` devuelve int directamente, sin parsing de bitstrings

### Beneficio en Grover
- `cirq.Z.controlled(num_controls=n-1)` + `cirq.H.on_each()` → el oraculo y difusor se expresan en muy pocas lineas
- `circuit += oracle` para componer subcircuitos (operador `+` sobrecargado)
- Scheduling por momentos agrupa automaticamente puertas que pueden ejecutarse en paralelo

### Beneficio en Shor
- **`cirq.ArithmeticGate`**: permite definir la semantica matematica (modular exponentiation) y el simulador la evalua clasicamente
- `cirq.qft(*qubits, inverse=True)` built-in, sin necesidad de implementar QFT manualmente
- Resultado: **la implementacion de Shor mas corta** (272 lineas vs 754 de Qiskit), porque ArithmeticGate encapsula toda la logica de exponenciacion modular

### Desventaja
- **Big-endian qubit ordering** (LineQubit(0) = MSB): requiere invertir bits en el oraculo para que los resultados coincidan con la convencion estandar
- ArithmeticGate es un shortcut del simulador: no genera un circuito cuantico real de puertas

---

## CUDA-Q (NVIDIA)

### Bondades generales
- **GPU nativa via cuStateVec**: hasta 425x speedup sobre CPU
- **Compilacion MLIR/LLVM**: los kernels se compilan a codigo maquina optimizado, no se interpretan
- **Gate fusion automatica**: el runtime fusiona puertas consecutivas antes de aplicarlas al state vector
- **Multi-GPU** (`nvidia-mgpu`): pooling de memoria para simular >33 qubits
- **Sin transpiler necesario**: la compilacion es interna al kernel

### Beneficio en Grover
- `kernel.cz(controls, target)` para MCZ nativo con lista arbitraria de controles
- El circuito completo se construye inline en un solo kernel → la compilacion MLIR puede optimizar todo el flujo de puertas de una vez
- `cudaq.sample(kernel, shots_count=n)` ejecuta todos los shots en una sola llamada al runtime GPU

### Beneficio en Shor
- **`kernel.cr1(angle, control, target)`** para rotaciones controladas de la QFT
- **Permutation network** para exponenciacion modular: descompone ciclos de permutacion en transposiciones de un solo bit, cada una implementada como un multi-controlled X
- La compilacion MLIR puede optimizar las cadenas de puertas controladas del permutation network
- La GPU es critica aqui: el permutation network genera muchas puertas, pero la ejecucion en GPU las absorbe

### Desventaja
- **Modelo de kernel rigido**: los kernels no admiten llamadas Python arbitrarias, lo que obliga a inline todo (206 lineas solo para Grover, el mas largo)
- **No compone subcircuitos**: `build_oracle` y `build_diffuser` devuelven kernels independientes pero `grover_circuit` tiene que duplicar su logica inline
- **Bitstring ordering invertido**: qubit 0 es MSB en CUDA-Q, requiere reversion manual

---

## ProjectQ (ETH Zurich)

### Bondades generales
- **Sintaxis inspirada en Dirac**: `H | qubit` lee como H aplicado a |qubit⟩
- **`with Control(eng, ctrls):`**: cualquier bloque de codigo se hace controlado automaticamente
- **`All(H) | qureg`**: aplica una puerta a todo el registro en una linea
- **Math library built-in**: `MultiplyByConstantModN`, `AddConstantModN`
- **Emulacion clasica**: el simulador puede ejecutar math gates sin descomponerlas
- **Simulador C++ con OpenMP y SIMD**: alto rendimiento sin dependencias GPU

### Beneficio en Grover
- `with Control(eng, qureg[:-1]): Z | qureg[-1]` — MCZ en 2 lineas con context manager
- `All(H) | qureg` y `All(X) | qureg` — difusor expresado de forma muy natural
- El patron `Control` permite que el compilador interno decida la descomposicion optima del MCZ

### Beneficio en Shor
- **`MultiplyByConstantModN(a, N) | target`**: la exponenciacion modular es **una sola linea** de codigo cuantico
- **Semi-classical QFT** con un unico qubit de control reutilizado: minimo uso de qubits
- El simulador emula `MultiplyByConstantModN` clasicamente → permite factorizar numeros grandes en laptop sin explotar el state vector
- **Menor cantidad de codigo para Shor** entre los frameworks nativos (234 lineas)

### Desventaja
- **Modelo imperativo sin circuito reutilizable**: cada shot requiere crear un engine nuevo y reaplicar todas las puertas desde cero
- La emulacion clasica de math gates "hace trampa": no genera el circuito cuantico real
- Para benchmarking justo, habria que forzar descomposicion con `InstructionFilter`
- Proyecto menos activo que Qiskit o Cirq

---

## QDisLib (BSC)

### Bondades generales
- **Unica libreria de circuit cutting**: divide circuitos grandes en subcircuitos ejecutables en hardware pequeno
- **Distribucion HPC via PyCOMPSs**: paraleliza la ejecucion de subcircuitos en multiples nodos/GPUs
- **FindCut automatico**: usa algoritmos de grafos para encontrar cortes optimos
- **Agnositca al framework**: acepta circuitos Qiskit y Qibo

### Beneficio en Grover
- Para n grande, puede cortar el oraculo MCZ en subcircuitos que quepan en QPUs pequenas
- La distribucion permite ejecutar los 6^k o 8^k subcircuitos en paralelo

### Beneficio en Shor
- La exponenciacion modular de Beauregard tiene muchas puertas de 2 qubits → candidato ideal para gate cutting
- Puede distribuir las multiplicaciones modulares entre multiples nodos

### Desventaja
- **No aporta valor para simulacion local**: el cutting tiene overhead exponencial (6^k, 8^k) que empeora el rendimiento
- **Es un wrapper**: no implementa logica cuantica propia, delega a Qiskit
- Requiere PyCOMPSs para la ventaja real (distribucion)
- Menor cantidad de codigo (291 lineas) precisamente porque no implementa nada propio

---

## Tabla comparativa: features clave por framework

| Feature | Qiskit | Cirq | CUDA-Q | ProjectQ | QDisLib |
|---|---|---|---|---|---|
| MCZ nativo | `ZGate().control(n-1)` | `Z.controlled(n-1)` | `cz(controls, tgt)` | `Control(eng, ctrls)` | (Qiskit) |
| QFT built-in | `QFTGate` + aproximada | `cirq.qft()` | Manual | `QFT` gate | (Qiskit) |
| Math modular | Manual (Beauregard) | `ArithmeticGate` | Manual (permutaciones) | `MultiplyByConstantModN` | (Qiskit) |
| Transpiler | Pass manager (0-3) | Transformers | MLIR (interno) | Engine pipeline | (Qiskit) |
| GPU | Via Aer (cuQuantum) | Via qsim (cuStateVec) | Nativa (cuStateVec) | No | Via cuQuantum |
| Composicion | `compose(inplace)` | `circuit +=` | Inline (no compone) | Imperativo | (Qiskit) |
| Multi-shot | 1 llamada | 1 llamada | 1 llamada | N engines | 1 llamada |
| Circuit cutting | No | No | No | No | Si |

## Conclusiones por algoritmo

### Grover
- **Mas conciso**: Cirq y ProjectQ (sintaxis expresiva para MCZ y H-on-all)
- **Mas rapido en simulacion**: CUDA-Q (GPU + compilacion MLIR)
- **Mas escalable**: CUDA-Q con multi-GPU o QDisLib con circuit cutting
- **Mejor para hardware real**: Qiskit (transpiler optimiza para topologia)

### Shor
- **Menos codigo**: ProjectQ (MultiplyByConstantModN resuelve todo en 1 linea)
- **Mas fiel al paper**: Qiskit (Beauregard completo con Draper adder)
- **Mas corto sin trampa**: Cirq (ArithmeticGate encapsula la matematica pero genera circuito real)
- **Mas rapido en simulacion grande**: CUDA-Q (GPU absorbe el permutation network)
- **Mas escalable**: QDisLib (circuit cutting de la exponenciacion modular)
