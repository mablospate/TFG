# Analisis de frameworks Python: bondades y beneficios por algoritmo

## Resumen de lineas de codigo

| Framework | Grover | Shor (total) | Total |
|---|---|---|---|
| Qiskit | 151 | 754 (adder 473 + qft 38 + shor 243) | 905 |
| Cirq | 161 | 272 (modular_exp 73 + shor 199) | 433 |
| CUDA-Q | 206 | 368 (permutation 130 + qft 41 + shor 197) | 574 |
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

## QDisLib (BSC)

### Bondades generales
- **Unica libreria de circuit cutting del benchmark**: divide circuitos grandes en subcircuitos ejecutables en hardware pequeno
- **Distribucion HPC via PyCOMPSs**: la API esta disenada para paralelizar la ejecucion de subcircuitos en multiples nodos/GPUs (PyCOMPSs NO esta instalado en la imagen del benchmark; ver Desventaja)
- **FindCut automatico**: usa algoritmos de grafos para encontrar cortes optimos (depende de `pymetis`)
- **Agnostica al framework**: acepta circuitos Qiskit y Qibo
- **Doble ruta de medicion en el benchmark**:
  - Ruta **directa**: `search()` / `find_factor()` ejecutan el circuito con `AerSampler` igual que cualquier otro framework
  - Ruta de **cutting** (IMPLEMENTADA y ejecutandose en el benchmark): `search_with_cutting()` / `find_order_with_cutting()` llaman a `find_cut()` y, si devuelve cortes, a `wire_cutting(backend="numpy")`. El resultado aporta los campos extra `cutting_wall_time_ms`, `cutting_find_time_ms`, `cutting_expectation_value`

### Beneficio en Grover
- Para n grande, puede cortar el oraculo MCZ en subcircuitos que quepan en QPUs pequenas
- En presencia de PyCOMPSs, los 6^K u 8^K subcircuitos se distribuyen en paralelo entre nodos

### Beneficio en Shor
- La exponenciacion modular de Beauregard tiene muchas puertas de 2 qubits → candidato ideal para gate cutting
- Puede distribuir las multiplicaciones modulares entre multiples nodos cuando PyCOMPSs esta disponible

### Desventaja
- **Sin PyCOMPSs en la imagen Docker**, la ruta de cutting ejecuta `wire_cutting` **localmente y en serie**: el overhead exponencial del cutting (8^K subcircuitos) anula el beneficio comparado con la ejecucion directa
- **Circuitos pequenos (n ≤ 5)**: `find_cut()` devuelve `[]` (no encuentra cortes utiles), por lo que el cutting no se aplica y `cutting_expectation_value = 0.0`
- **Puertas de 3+ qubits (Toffoli, presentes en Shor)**: tambien hacen que `find_cut()` devuelva `[]`
- **Monkey-patch requerido**: QDisLib usa la API antigua de Qiskit (`.nqubits`); el benchmark inyecta una propiedad compatible al importar el modulo para mantener compatibilidad con Qiskit 2.0
- **Wrapper sobre Qiskit**: no implementa logica cuantica propia; la "ventaja real" requiere PyCOMPSs + cluster
- Menor cantidad de codigo (291 lineas) precisamente porque no implementa nada propio
- Excluido en Linux aarch64 porque `pymetis` (dependencia de `find_cut`) no tiene wheel arm64

---

## Tabla comparativa: features clave por framework

| Feature | Qiskit | Cirq | CUDA-Q | QDisLib |
|---|---|---|---|---|
| MCZ nativo | `ZGate().control(n-1)` | `Z.controlled(n-1)` | `cz(controls, tgt)` | (Qiskit) |
| QFT built-in | `QFTGate` + aproximada | `cirq.qft()` | Manual | (Qiskit) |
| Math modular | Manual (Beauregard) | `ArithmeticGate` | Manual (permutaciones) | (Qiskit) |
| Transpiler | Pass manager (0-3) | Transformers | MLIR (interno) | (Qiskit) |
| GPU | Via Aer (cuQuantum) | Via qsim (cuStateVec) | Nativa (cuStateVec) | Via cuQuantum (Qiskit) |
| Composicion | `compose(inplace)` | `circuit +=` | Inline (no compone) | (Qiskit) |
| Multi-shot | 1 llamada | 1 llamada | 1 llamada | 1 llamada |
| Circuit cutting | No | No | No | Sí (local, serie; PyCOMPSs para distribución HPC) |

## Conclusiones por algoritmo

### Grover
- **Mas conciso**: Cirq (sintaxis expresiva para MCZ y H-on-all)
- **Mas rapido en simulacion**: CUDA-Q (GPU + compilacion MLIR)
- **Mas escalable**: CUDA-Q con multi-GPU o QDisLib con circuit cutting (con PyCOMPSs)
- **Mejor para hardware real**: Qiskit (transpiler optimiza para topologia)

### Shor
- **Menos codigo (que genere circuito real)**: Cirq (ArithmeticGate encapsula la matematica pero genera circuito real)
- **Mas fiel al paper**: Qiskit (Beauregard completo con Draper adder)
- **Mas rapido en simulacion grande**: CUDA-Q (GPU absorbe el permutation network)
- **Mas escalable**: QDisLib (circuit cutting de la exponenciacion modular, con PyCOMPSs)
