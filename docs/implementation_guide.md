# Guia de implementacion: Grover y Shor en Cirq, CUDA-Q, ProjectQ y QDisLib

Este documento te guia paso a paso para implementar ambos algoritmos en los 4 frameworks
Python que faltan. Usa la implementacion de Qiskit como referencia de la interfaz y la logica.

> **Convencion**: en todo el documento, `n = ceil(log2(N))` para Shor y `n` = numero de qubits
> de busqueda para Grover. `m` = precision (qubits de phase estimation), por defecto `2*n`.

---

## Tabla de referencia rapida: API de cada framework

| Operacion | Qiskit | Cirq | CUDA-Q | ProjectQ |
|---|---|---|---|---|
| Crear circuito | `QuantumCircuit(n)` | `cirq.Circuit()` | `cudaq.make_kernel()` | `MainEngine(Simulator(), [])` |
| Allocar qubits | `QuantumRegister(n)` | `cirq.LineQubit.range(n)` | `kernel.qalloc(n)` | `eng.allocate_qureg(n)` |
| H en todos | `qc.h(qr[i])` en loop | `cirq.H.on_each(*qubits)` | `kernel.h(qubits[i])` en loop | `All(H) \| qureg` |
| X en qubit i | `qc.x(qr[i])` | `cirq.X(qubits[i])` | `kernel.x(qubits[i])` | `X \| qureg[i]` |
| MCZ | `ZGate().control(n-1)` | `cirq.Z.controlled(num_controls=n-1)` | `kernel.cz(controls, target)` | `with Control(eng, ctrls): Z \| tgt` |
| Medir todos | `qc.measure(qr, cr)` | `cirq.measure(*qbs, key="k")` | `kernel.mz(qubits)` | `All(Measure) \| qureg` |
| Ejecutar | `sampler.run([qc], shots=n)` | `sim.run(circ, repetitions=n)` | `cudaq.sample(kern, shots_count=n)` | `eng.flush()` + leer bits |
| Componer | `qc.compose(sub, inplace=True)` | `circuit += sub` | No compone (inline) | Imperativo (in-place) |
| QFT | `QFTGate(n)` | `cirq.qft(*qbs)` | Manual (H + CR1 + SWAP) | `QFT` gate built-in |
| QFT inversa | `QFTGate(n).inverse()` | `cirq.qft(*qbs, inverse=True)` | Manual (inverso del anterior) | `get_inverse(QFT)` |
| Rotacion fase | `qc.p(theta, qubit)` | `cirq.rz(theta)(qubit)` | `kernel.r1(theta, qubit)` | `R(theta) \| qubit` |
| Rot. fase controlada | `CPhaseGate(theta)` | `.controlled_by(ctrl)` | `kernel.cr1(theta, ctrl, tgt)` | `with Control(eng, ctrl): R(th) \| tgt` |
| SWAP | `qc.swap(a, b)` | `cirq.SWAP(a, b)` | `kernel.swap(a, b)` | `Swap \| (a, b)` |

---

# PARTE 1: GROVER

## Estructura comun de Grover (todos los frameworks)

El algoritmo de Grover tiene siempre la misma estructura:

```
1. Superposicion uniforme: H en los n qubits
2. Repetir floor(pi/4 * sqrt(2^n)) veces:
   a. ORACULO: flip de fase del estado |target>
   b. DIFUSOR: inversion sobre la media (2|s><s| - I)
3. Medir todos los qubits
```

### Oraculo (fase flip en |target>)

El oraculo marca el estado `|target>` invirtiendo su fase. La logica es:

```
Para cada bit i del target:
    Si bit i es 0: aplicar X al qubit i    (para mapear |target> a |11...1>)
Aplicar Multi-Controlled Z (MCZ)            (invierte fase de |11...1>)
Para cada bit i del target:
    Si bit i es 0: aplicar X al qubit i    (deshacer)
```

**Detalle del bit flipping**: `(target >> i) & 1` te da el bit i del target.
Si es 0, aplicas X. Esto convierte |target> en |11...1>, que es el unico estado
afectado por el MCZ.

### Difusor (inversion sobre la media)

```
H en todos los qubits
X en todos los qubits
MCZ en todos los qubits
X en todos los qubits
H en todos los qubits
```

Es la misma estructura que el oraculo pero sin el target-dependent bit flipping,
porque el difusor siempre invierte sobre |00...0> (que con X se mapea a |11...1>).

### Numero optimo de iteraciones

```python
import math
num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))
```

### Interfaz requerida

Todas las implementaciones deben exponer estas 4 funciones. Mira `python/qiskit/grover.py`
para la implementacion de referencia.

```
build_oracle(n, target) -> CircuitType
build_diffuser(n) -> CircuitType
grover_circuit(n, target, num_iterations=None) -> CircuitType
search(n, target, simulator/sampler, pass_manager, num_iterations=None, num_shots=1024) -> (int, dict)
```

---

## 1.1 Grover en Cirq

**Fichero**: `python/cirq/grover.py`

### Imports necesarios

```python
import math
import cirq
```

### Conceptos clave de Cirq

- **LineQubit**: `cirq.LineQubit.range(n)` crea n qubits numerados 0..n-1.
- **Big-endian**: LineQubit(0) es el MSB. Esto afecta al bit mapping del oraculo.
  Si quieres que el resultado de la medida coincida con `target` como entero, el
  bit i del target debe mapearse al qubit `n-1-i` (no al qubit `i`).
- **Momentos**: un circuito Cirq es una secuencia de `Moment`s. Puedes ignorar esto
  para la funcionalidad basica (Cirq los crea automaticamente con `circuit.append()`).
- **on_each**: `cirq.H.on_each(*qubits)` aplica H a todos los qubits en un solo `Moment`.

### Pasos para build_oracle

1. Crear qubits con `cirq.LineQubit.range(n)`.
2. Crear circuito vacio con `cirq.Circuit()`.
3. Para cada bit i del target que sea 0: `circuit.append(cirq.X(qubits[n-1-i]))`.
   **Ojo**: se usa `n-1-i` por la convencion big-endian de Cirq.
4. Crear MCZ: `mcz = cirq.Z.controlled(num_controls=n-1)`.
5. Aplicar MCZ: `circuit.append(mcz.on(*qubits))`.
6. Deshacer las X (mismo bucle que paso 3).
7. Devolver el circuito.

### Pasos para build_diffuser

1. Crear qubits y circuito.
2. H en todos: `circuit.append(cirq.H.on_each(*qubits))`.
3. X en todos: `circuit.append(cirq.X.on_each(*qubits))`.
4. MCZ (igual que en el oraculo).
5. X en todos.
6. H en todos.
7. Devolver el circuito.

### Pasos para grover_circuit

1. Calcular iteraciones optimas si no se proporcionan.
2. Crear qubits y circuito.
3. Superposicion: `circuit.append(cirq.H.on_each(*qubits))`.
4. En un bucle: `circuit += oracle` y `circuit += diffuser` (operador `+=` compone circuitos).
5. Medir: `circuit.append(cirq.measure(*qubits, key="result"))`.
6. Devolver el circuito.

### Pasos para search

1. Construir el circuito con `grover_circuit(...)`.
2. Si hay pass_manager, aplicarlo: `qc = pass_manager(qc)`.
3. Ejecutar: `result = simulator.run(qc, repetitions=num_shots)`.
4. Obtener histograma: `histogram = result.histogram(key="result")`.
   - `histogram` es un `dict[int, int]` — mapea el entero medido a su conteo.
5. Convertir a formato `dict[str, int]` (bitstring -> count):
   - `format(value, f"0{n}b")` convierte entero a bitstring de n bits.
6. El mas frecuente: `max(histogram, key=histogram.get)`.
7. Devolver `(found_int, dist_dict)`.

### Gotcha principal

El qubit ordering big-endian de Cirq. Si no mapeas los bits del target correctamente
(con `n-1-i`), el oraculo marcara un estado diferente al que esperas.
Puedes verificar imprimiendo el circuito: `print(circuit)` muestra un diagrama ASCII.

---

## 1.2 Grover en CUDA-Q

**Fichero**: `python/cudaq/grover.py`

### Imports necesarios

```python
import math
import cudaq
```

### Conceptos clave de CUDA-Q

- **Kernel model**: todo circuito se construye dentro de un kernel. `cudaq.make_kernel()`
  crea un kernel builder.
- **No composicion**: los kernels de CUDA-Q no se pueden componer facilmente como
  subcircuitos. La solucion mas simple es construir todo inline en un solo kernel.
  Puedes definir `build_oracle` y `build_diffuser` como funciones que devuelven un kernel
  separado (para mantener la interfaz), pero `grover_circuit` tendra que duplicar la logica
  inline.
- **qalloc**: `kernel.qalloc(n)` alloca n qubits. Se accede con `qubits[i]`.
- **MCZ**: `kernel.cz(controls_list, target)` donde `controls_list` es una lista de qubits.
  Para n=1: usar `kernel.z(qubits[0])` directamente.
- **Bitstring ordering**: CUDA-Q pone qubit 0 como MSB (leftmost). Para que coincida
  con la convencion de Qiskit (qubit 0 = LSB), hay que invertir cada bitstring.
- **No hay on_each**: hay que hacer un loop para aplicar H a todos los qubits.

### Pasos para build_oracle y build_diffuser

Igual que en Cirq, pero:
- Crear kernel con `cudaq.make_kernel()` y qubits con `kernel.qalloc(n)`.
- H con `kernel.h(qubits[i])` en un for loop.
- X con `kernel.x(qubits[i])`.
- MCZ con `kernel.cz([qubits[0], ..., qubits[n-2]], qubits[n-1])`.
  Construye la lista de controles: `controls = [qubits[i] for i in range(n-1)]`.
- **Caso n=1**: no hay controles, usar `kernel.z(qubits[0])`.

### Pasos para grover_circuit (todo inline)

Este es el paso mas diferente. Como CUDA-Q no compone kernels facilmente, la
funcion `grover_circuit` construye **todo** en un solo kernel:

1. Crear kernel y qubits.
2. H en todos (loop).
3. Para cada iteracion:
   - **Oracle inline**: X donde target tiene bit 0, MCZ, deshacer X.
   - **Diffuser inline**: H todos, X todos, MCZ, X todos, H todos.
4. Medir: `kernel.mz(qubits)`.

No necesitas bit reversal (`n-1-i`) para el target en CUDA-Q — los qubits se
indexan directamente (qubit i mapea a bit i del target).

### Pasos para search

1. Construir kernel con `grover_circuit(...)`.
2. Si hay simulador especifico: `cudaq.set_target(simulator)` (e.g., "qpp-cpu").
3. Ejecutar: `result = cudaq.sample(kernel, shots_count=num_shots)`.
4. `result` es iterable con `.items()` que da `(bitstring, count)`.
5. **Invertir bitstrings**: `{bs[::-1]: count for bs, count in result.items()}`.
6. Encontrar el mas frecuente, convertir a int con `int(bitstring, 2)`.
7. Devolver `(found_int, dist_dict)`.

### Gotcha principal

El modelo de kernel es rigido. No puedes llamar funciones Python arbitrarias
dentro de un kernel. Todo tiene que ser operaciones de puertas cuanticas.
La composicion de subcircuitos es limitada.

---

## 1.3 Grover en ProjectQ

**Fichero**: `python/projectq/grover.py`

### Imports necesarios

```python
import math
from projectq import MainEngine
from projectq.ops import H, X, Z, All, Measure
from projectq.meta import Control
from projectq.backends import Simulator
```

### Conceptos clave de ProjectQ

- **Modelo imperativo**: no hay "circuito" como objeto. Las puertas se aplican
  directamente al qubit register y se acumulan en el engine.
- **Sintaxis Dirac**: `H | qubit` aplica H al qubit. `All(H) | qureg` aplica H a todos.
- **Control context manager**: `with Control(eng, ctrl_qubits): Gate | target` hace
  cualquier bloque de puertas controlado.
- **Un engine por shot**: cada shot requiere crear un nuevo `MainEngine`, aplicar todas
  las puertas, medir y flush. No hay forma de "reusar" un circuito.
- **Medida**: `Measure | qubit` mide el qubit. Despues de `eng.flush()`, lees el
  resultado con `int(qubit)`.

### Pasos para build_oracle

La funcion recibe `(n, target, eng, qureg)` — no devuelve un circuito, sino que
aplica puertas in-place:

1. X donde target tiene bit 0: `X | qureg[i]`.
2. MCZ: `with Control(eng, qureg[:-1]): Z | qureg[-1]`.
   - `qureg[:-1]` son los controles (todos menos el ultimo).
   - `qureg[-1]` es el target del Z.
3. Deshacer X.

### Pasos para build_diffuser

Tambien recibe `(n, eng, qureg)` y aplica in-place:

1. `All(H) | qureg`
2. `All(X) | qureg`
3. MCZ con Control (igual que en oracle)
4. `All(X) | qureg`
5. `All(H) | qureg`

### Pasos para grover_circuit

A diferencia de los otros frameworks, aqui la funcion crea el engine y el register,
aplica todas las puertas, y devuelve `(eng, qureg)` **sin medir** (la medida se
deja para `search`):

1. Calcular iteraciones.
2. `eng = MainEngine(backend=Simulator(), engine_list=[])`.
3. `qureg = eng.allocate_qureg(n)`.
4. `All(H) | qureg` (superposicion).
5. Loop: llamar a `build_oracle` y `build_diffuser`.
6. Devolver `(eng, qureg)`.

### Pasos para search

Aqui esta la diferencia mayor: **cada shot requiere un engine nuevo**.

1. Loop de `num_shots`:
   a. Crear engine y qureg frescos.
   b. `All(H) | qureg` (superposicion).
   c. Loop de iteraciones: oracle + diffuser.
   d. `All(Measure) | qureg`.
   e. `eng.flush()`.
   f. Leer bits: `int(qureg[i])` para cada qubit.
   g. Construir bitstring. **Atencion al ordering**: para que coincida con Qiskit,
      el bit del qubit i debe ir en la posicion i (LSB). Es decir, el bitstring
      MSB-first seria `"".join(str(bits[i]) for i in reversed(range(n)))`.
   h. Acumular en el diccionario de distribucion.
2. Encontrar el mas frecuente.
3. Devolver `(found_int, dist_dict)`.

### Gotcha principal

El modelo de 1-engine-por-shot. Si intentas reusar un engine despues de medir y
flush, falla. Cada shot es un ciclo completo: crear engine -> aplicar puertas ->
medir -> flush -> leer resultado.

---

## 1.4 Grover en QDisLib

**Fichero**: `python/qdislib/grover.py`

### Conceptos clave de QDisLib

QDisLib es un **wrapper** sobre Qiskit. No implementa logica cuantica propia.
La idea es:

1. Construir el circuito usando las funciones de Qiskit que ya tienes.
2. Ejecutarlo a traves de QDisLib para aprovechar circuit cutting / distribucion.
3. Si QDisLib no esta instalado, hacer fallback a Qiskit-Aer directo.

### Implementacion

1. **Importar** las funciones de Qiskit:
   ```python
   from python.qiskit.grover import (
       build_oracle as _qiskit_build_oracle,
       build_diffuser as _qiskit_build_diffuser,
       grover_circuit as _qiskit_grover_circuit,
   )
   ```

2. **Re-exportar** las funciones de construccion de circuito:
   ```python
   build_oracle = _qiskit_build_oracle
   build_diffuser = _qiskit_build_diffuser
   grover_circuit = _qiskit_grover_circuit
   ```

3. **Implementar `search`** con el patron try/except:
   - `try: import qdislib` — si funciona, usar QDisLib path.
   - `except ImportError:` — fallback a Qiskit-Aer directo.
   - En ambos paths: crear sampler y pass_manager si no se proporcionan,
     transpilar con `pass_manager.run(qc)`, ejecutar con sampler.

4. **Defaults** cuando sampler/pass_manager son None:
   ```python
   from qiskit_aer import AerSimulator
   from qiskit_aer.primitives import SamplerV2 as AerSampler
   from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

   backend = AerSimulator()
   pm = generate_preset_pass_manager(backend=backend)
   sampler = AerSampler()
   ```

5. **Ejecutar y obtener counts**:
   ```python
   qc_isa = pm.run(qc)
   dist = sampler.run([qc_isa], shots=num_shots).result()[0].data.result.get_counts()
   ```
   El nombre del registro clasico es `"result"` (viene del `grover_circuit` de Qiskit).

### Nota

Lo que hace que QDisLib sea interesante para el TFG no es la implementacion del algoritmo
(que es identica a Qiskit), sino el overhead del circuit cutting en los benchmarks.
En un benchmark real, pasarias el circuito transpilado por la pipeline de cutting de QDisLib
antes de ejecutarlo. El placeholder en `search` marca donde iria esa logica.

---

# PARTE 2: SHOR

## Estructura comun de Shor (todos los frameworks)

El algoritmo de Shor tiene tres capas:

### Capa 1: `find_factor(N, ...)` — clasico

Esta funcion es **identica en todos los frameworks** porque es puramente clasica:

```
1. Si N es par: devolver 2
2. Si N es una potencia perfecta (d^k = N): devolver d
3. Repetir num_tries veces:
   a. Elegir a aleatorio en [2, N-1]
   b. Si gcd(a, N) > 1: "lucky guess", devolver gcd(a, N)
   c. Llamar a find_order(a, N, ...) para obtener r
   d. Si r != 0 y r es par:
      - Calcular x = a^(r/2) mod N - 1
      - Calcular d = gcd(x, N)
      - Si 1 < d < N: devolver d
4. Si no se encontro factor: devolver 1
```

Puedes copiar esta logica directamente de la implementacion de Qiskit,
cambiando solo la llamada a `find_order` para que use los parametros de cada framework.

### Capa 2: `find_order(A, N, ...)` — hibrido

1. Calcular `m = precision or 2 * ceil(log2(N))`.
2. Comprobar `gcd(A, N) == 1`.
3. Construir circuito con `order_finding_circuit(A, N, m)`.
4. Ejecutar el circuito (especifico de cada framework).
5. Post-procesar con `_get_order_from_dist(dist, A, N, m)`.

### Capa 3: `order_finding_circuit(A, N, precision)` — cuantico (lo mas diferente entre frameworks)

```
1. Registro de control (m qubits) en superposicion: H en todos
2. Registro target (n qubits): inicializar qubit 0 a |1>
3. Exponenciacion modular controlada (ESPECIFICO DE CADA FRAMEWORK)
4. QFT inversa en registro de control
5. Medir registro de control
```

### Post-procesamiento clasico: `_get_order_from_dist`

Tambien **identico en todos los frameworks**:

```
1. Ordenar los resultados de medida por frecuencia (descendente)
2. Para los 10 mas frecuentes:
   a. Ignorar el resultado 0
   b. Convertir bitstring a entero x
   c. Aplicar fracciones continuas: Fraction(x / 2^m).limit_denominator(N-1).denominator
   d. Si A^r mod N == 1: hemos encontrado el orden r, devolverlo
3. Si ningun candidato funciona: devolver 0
```

**Atencion**: la funcion `Fraction(x / 2**m)` puede perder precision para m grande.
Esto no afecta para N=15 o N=21, pero es algo a documentar en el TFG.

---

## 2.1 Shor en Cirq

**Ficheros**: `python/cirq/shor/shor.py` y `python/cirq/shor/modular_exp.py`

### Estrategia: ArithmeticGate

Cirq proporciona `cirq.ArithmeticGate`, una clase abstracta que te permite definir la
**semantica matematica** de una operacion y el simulador la evalua clasicamente. Esto es
un shortcut (no genera un circuito de puertas reales), sino que el simulador la evalua
clasicamente. Esto es un shortcut (no genera un circuito de puertas reales), pero es la forma idiomatica de
implementar Shor en Cirq.

### Fichero 1: `modular_exp.py` — La clase ModularExp

Debes crear una clase que herede de `cirq.ArithmeticGate`. Necesitas implementar
4 metodos:

#### `__init__(self, target_size, exponent_size, base, modulus)`
Guarda los parametros:
- `target_size`: n qubits del registro target
- `exponent_size`: m qubits del registro de exponente (control)
- `base`: A (la base de la exponenciacion)
- `modulus`: N

#### `registers(self) -> list`
Define los registros del gate. Debe devolver una lista donde cada elemento
describe un registro:
- Un registro de n qubits se describe como `[2] * n` (cada qubit tiene 2 estados).
- El gate tiene 2 registros: target (n qubits) y exponent (m qubits).
- Devuelve: `[[2] * self.target_size, [2] * self.exponent_size]`

#### `with_registers(self, *new_registers) -> ModularExp`
Cirq llama esto internamente cuando necesita redimensionar los registros.
Debe devolver una nueva instancia con los tamanios actualizados.
Inferir el tamanio de cada registro: si es un `int`, tamanio = 1; si es una lista,
tamanio = `len(lista)`.

#### `apply(self, target_value, exponent_value) -> tuple`
La logica matematica. Recibe los valores enteros de los registros y devuelve
los nuevos valores:
- Si `target_value < modulus`:
  devolver `((target_value * pow(base, exponent_value, modulus)) % modulus, exponent_value)`
- Si no: devolver los valores sin cambiar.

**Nota**: `pow(base, exponent_value, modulus)` hace exponenciacion modular eficiente en Python.

Tambien necesitas `__repr__`, `__eq__` y `__hash__` para que Cirq pueda comparar y
cachear gates. Compara los 4 atributos para `__eq__` y hashealos para `__hash__`.

### Fichero 2: `shor.py` — Las funciones de Shor

#### `order_finding_circuit(A, N, precision=None)`

1. Verificar `gcd(A, N) == 1`.
2. Calcular `n = ceil(log2(N))`, `m = precision or 2*n`.
3. Crear qubits:
   - `exponent_qubits = cirq.LineQubit.range(m)` — para phase estimation
   - `target_qubits = cirq.LineQubit.range(m, m+n)` — para el resultado de la exp. modular
4. Circuito:
   a. H en todos los exponent qubits: `cirq.H.on_each(*exponent_qubits)`.
   b. Inicializar target en |1>: `cirq.X(target_qubits[0])`.
   c. Aplicar el gate ModularExp:
      ```
      mod_exp = ModularExp(target_size=n, exponent_size=m, base=A, modulus=N)
      circuit.append(mod_exp.on(*target_qubits, *exponent_qubits))
      ```
      **Atencion al orden**: primero target, luego exponent (asi lo espera `registers()`).
   d. QFT inversa: `cirq.qft(*exponent_qubits, inverse=True)`.
   e. Medir: `cirq.measure(*exponent_qubits, key="result")`.

#### `_get_order_from_dist(dist, A, N, precision)`

Igual que la version general descrita arriba. La unica diferencia es que el histograma
de Cirq ya da enteros (no bitstrings), asi que no necesitas `int(bitstring, 2)`.

**Detalle**: `result.histogram(key="result")` devuelve `{int: count}`.
Para la funcion generica que espera ordenar por frecuencia, puedes iterar directamente.
Si `x == 0`, skip. Luego `Fraction(x / 2**precision).limit_denominator(N-1).denominator`.

#### `find_order(A, N, simulator, pass_manager=None, precision=None, num_shots=10)`

1. Calcular m.
2. Construir circuito con `order_finding_circuit(A, N, m)`.
3. Aplicar pass_manager si existe.
4. Ejecutar: `result = simulator.run(qc, repetitions=num_shots)`.
5. Obtener histograma: `histogram = result.histogram(key="result")`.
6. Llamar a `_get_order_from_dist(histogram, A, N, m)`.
7. Convertir histograma a formato `dict[str, int]` para la interfaz comun.

#### `find_factor(N, simulator, pass_manager=None, ...)`

Logica clasica identica a Qiskit. La unica diferencia: la llamada a `find_order`
pasa `simulator` y `pass_manager` en vez de `sampler` y `pass_manager`.

---

## 2.2 Shor en CUDA-Q

**Ficheros**: `python/cudaq/shor/shor.py`, `python/cudaq/shor/qft.py`, `python/cudaq/shor/permutation.py`

### Estrategia: Permutation Network

CUDA-Q no tiene ArithmeticGate ni math library. La exponenciacion modular se
implementa como un **permutation network** — una red de SWAPs controlados que
permutan los estados base segun la tabla de multiplicacion modular.

### Fichero 1: `qft.py` — QFT manual

CUDA-Q no tiene QFT built-in. Hay que implementarla manualmente.

#### `apply_qft(kernel, qubits, n)`

La QFT sobre n qubits:
```
Para i de 0 a n-1:
    H en qubits[i]
    Para j de i+1 a n-1:
        CR1(pi / 2^(j-i), control=qubits[j], target=qubits[i])
Para i de 0 a n//2 - 1:
    SWAP(qubits[i], qubits[n-1-i])
```

Usa `kernel.h(qubits[i])`, `kernel.cr1(angle, control, target)`, `kernel.swap(a, b)`.

#### `apply_inverse_qft(kernel, qubits, n)`

La inversa de la QFT — mismas operaciones pero en orden inverso y con angulos negados:
```
Para i de 0 a n//2 - 1:
    SWAP(qubits[i], qubits[n-1-i])
Para i de n-1 a 0 (descendente):
    Para j de n-1 a i+1 (descendente):
        CR1(-pi / 2^(j-i), control=qubits[j], target=qubits[i])
    H en qubits[i]
```

### Fichero 2: `permutation.py` — Exponenciacion modular via permutaciones

La idea central: la multiplicacion modular `|y> -> |A^power * y mod N>` es una
permutacion de los estados base {0, 1, ..., N-1}. Podemos descomponer esa
permutacion en transposiciones (swaps de pares de estados), y cada transposicion
se implementa como un multi-controlled X.

#### `build_mod_exp_permutation(A, N, power) -> dict[int, int]`

Construye la tabla de permutacion:
```
a_power = pow(A, power, N)
permutation = {}
Para cada y de 0 a N-1:
    target = (a_power * y) % N
    Si y != target:
        permutation[y] = target
Devolver permutation
```

#### `controlled_swap_permutation(kernel, ctrl, target_qubits, permutation)`

Descompone la permutacion en ciclos disjuntos y luego en transposiciones:
```
visited = set()
Para cada start en sorted(permutation.keys()):
    Si start ya visitado: continuar
    Construir el ciclo: seguir permutation[current] hasta volver a start
    Si el ciclo tiene longitud <= 1: continuar
    Para idx de 1 a len(cycle)-1:
        controlled_transposition(kernel, ctrl, target_qubits, cycle[0], cycle[idx])
```

#### `controlled_transposition(kernel, ctrl, target_qubits, a, b)`

Intercambia |a> y |b> controlado por ctrl. Si a y b difieren en un solo bit,
es una transposicion de un solo bit (mas eficiente). Si difieren en mas bits,
se descompone recursivamente:

```
diff_bits = a XOR b
diff_positions = [posiciones donde difieren]
Si solo 1 posicion:
    Llamar a _controlled_single_bit_transposition(...)
Sino:
    pivot = diff_positions[0]
    a_prime = a XOR (1 << pivot)   # estado intermedio que difiere de a en un bit
    controlled_transposition(a, a_prime)     # 1 bit de diferencia
    controlled_transposition(a_prime, b)     # un bit menos de diferencia
    controlled_transposition(a, a_prime)     # deshacer el primero
```

#### `_controlled_single_bit_transposition(kernel, ctrl, target_qubits, a, b)`

Caso base: a y b difieren en exactamente un bit (posicion `flip_bit`).
```
1. Identificar flip_bit: (a XOR b).bit_length() - 1
2. Otras posiciones: todos los bits que no son flip_bit
3. Para cada otra posicion pos:
   Si bit pos de a es 0: aplicar X al qubit pos (para matchear |a>)
4. Multi-controlled X: kernel.cx([ctrl] + [target_qubits[pos] for pos in others], target_qubits[flip_bit])
5. Deshacer las X
```

### Fichero 3: `shor.py` — Las funciones de Shor

#### `order_finding_circuit(A, N, precision=None)`

1. Verificar gcd, calcular n y m, total_qubits = m + n.
2. Crear kernel y allocar qubits.
3. Separar en registros: ctrl_qubits (primeros m) y tgt_qubits (siguientes n).
4. H en todos los ctrl_qubits.
5. X en tgt_qubits[0] (inicializar |1>).
6. **Exponenciacion modular controlada**:
   ```
   Para i de 0 a m-1:
       perm = build_mod_exp_permutation(A, N, 2^i)
       Si perm no vacio:
           controlled_swap_permutation(kernel, ctrl_qubits[i], tgt_qubits, perm)
   ```
   Esto aplica la permutacion correspondiente a A^(2^i) mod N, controlada por el qubit i.
7. QFT inversa en ctrl_qubits: `apply_inverse_qft(kernel, ctrl_qubits, m)`.
8. Medir ctrl_qubits: `kernel.mz(ctrl_qubits[i])` para cada i.

#### `find_order` y `find_factor`

Misma logica clasica que los otros frameworks. Diferencias en ejecucion:
- `cudaq.set_target(simulator)` si se proporciona.
- `result = cudaq.sample(kernel, shots_count=num_shots)`.
- **Bitstring extraction**: el resultado incluye todos los qubits. Necesitas extraer
  solo los primeros m bits (los de control) e invertirlos (MSB/LSB flip).
  ```
  Para cada (bitstring, count) en result:
      ctrl_bits = bitstring[:m]    # primeros m caracteres
      key = ctrl_bits[::-1]        # invertir
      dist[key] = dist.get(key, 0) + count
  ```

---

## 2.3 Shor en ProjectQ

**Fichero**: `python/projectq/shor/shor.py`

### Estrategia: MultiplyByConstantModN + Semi-classical QFT

ProjectQ tiene `MultiplyByConstantModN` built-in y soporta mid-circuit measurement,
lo que permite implementar la variante **semi-classical** de Shor con un unico qubit
de control reutilizado.

### Imports necesarios

```python
import math
import random
from fractions import Fraction
from projectq import MainEngine
from projectq.ops import H, X, R, All, Measure
from projectq.meta import Control
from projectq.backends import Simulator
from projectq.libs.math import MultiplyByConstantModN
```

### Funcion clave: `_run_order_finding_once(A, N, precision)`

Esta funcion ejecuta **un solo shot** del algoritmo. Como ProjectQ requiere un
engine por shot, esto se llama m veces desde `find_order`.

**Variante semi-classical**: en vez de m qubits de control, usa **1 solo qubit de control**
que se mide y reutiliza m veces. Las rotaciones de fase de la QFT inversa se aplican
como correcciones clasicas basadas en los bits ya medidos.

Pasos:
```
1. Crear engine, allocar 1 qubit de control + n qubits de target
2. Inicializar target en |1>: X | target[0]
3. measured_bits = []
4. Para i de 0 a m-1:
   a. H | ctrl                              (poner en superposicion)
   b. power = pow(A, 2^(m-1-i), N)          (precalcular clasicamente)
   c. with Control(eng, ctrl):
        MultiplyByConstantModN(power, N) | target
   d. Para j de 0 a i-1:                    (correcciones de fase = QFT semi-classical)
        Si measured_bits[j] es 1:
            R(-pi / 2^(i-j)) | ctrl
   e. H | ctrl
   f. Measure | ctrl
      eng.flush()
      bit = int(ctrl)
      measured_bits.append(bit)
   g. Si bit es 1: X | ctrl                 (reset para reusar)
5. Medir target (necesario para que ProjectQ pueda deallocar):
   All(Measure) | target
   eng.flush()
6. El bitstring resultado es measured_bits INVERTIDO:
   bitstring = "".join(str(b) for b in reversed(measured_bits))
   Devolver bitstring
```

**Por que invertir measured_bits**: el semi-classical QFT mide los bits en orden
LSB-first. Para que `int(bitstring, 2)` de el valor correcto, necesitamos el
bitstring en formato MSB-first, asi que invertimos.

### `find_order(A, N, simulator=None, pass_manager=None, precision=None, num_shots=10)`

1. Verificar gcd.
2. Calcular m.
3. Loop de `num_shots`:
   - `bitstring = _run_order_finding_once(A, N, m)`
   - `dist[bitstring] = dist.get(bitstring, 0) + 1`
4. `r = _get_order_from_dist(dist, A, N, m)`
5. Devolver `(r, dist)`.

**Nota**: `simulator` y `pass_manager` se ignoran (ProjectQ crea su propio engine).
Se mantienen en la firma para compatibilidad con la interfaz comun.

### `find_factor(N, ...)`

Logica clasica identica. La llamada a `find_order` no pasa simulator/pass_manager.

### Gotchas

1. **MultiplyByConstantModN es emulacion clasica**: el simulador de ProjectQ computa
   la multiplicacion modular clasicamente (shortcut), no genera un circuito cuantico
   real. Esto hace que sea rapido pero no es un benchmark justo de puertas cuanticas.
   Documenta esto en el TFG.

2. **R gate**: en ProjectQ, `R(theta)` es la rotacion de fase `diag(1, e^(i*theta))`.
   Para la correccion del semi-classical QFT, usa angulos negativos: `R(-pi / 2^k)`.

3. **Reset del qubit**: despues de medir, si el bit es 1, aplicas X para resetearlo a |0>.
   ProjectQ lo permite porque la medida colapsa el estado.

---

## 2.4 Shor en QDisLib

**Fichero**: `python/qdislib/shor/shor.py`

### Estrategia: Wrapper de Qiskit

Igual que para Grover, QDisLib reutiliza la implementacion de Qiskit:

1. Importar `order_finding_circuit` y `_get_order_from_dist` de `python.qiskit.shor.shor`.
2. Re-exportar `order_finding_circuit`.
3. Implementar `find_order` y `find_factor` con el patron try/except para QDisLib fallback.

### Helper: `_make_backend_defaults()`

Crea el backend, sampler y pass_manager por defecto (reutilizable en find_order y find_factor):
```python
backend = AerSimulator()
pm = generate_preset_pass_manager(backend=backend)
sampler = AerSampler()
return backend, sampler, pm
```

### Helper: `_run_circuit(qc, sampler, pass_manager, num_shots, register_name)`

Ejecuta un circuito y devuelve los counts:
```python
qc_isa = pass_manager.run(qc)
result = sampler.run([qc_isa], shots=num_shots).result()[0]
dist = getattr(result.data, register_name).get_counts()
return dist
```

**Atencion**: `register_name` debe ser el nombre del `ClassicalRegister` del circuito
de Qiskit. Para el circuito de Shor de Qiskit es `"output_bits"`.

### `find_order(A, N, sampler=None, pass_manager=None, ...)`

1. Construir circuito con `order_finding_circuit(A, N, m)`.
2. Comprobar que `qc != 0` (gcd check).
3. Try QDisLib / except fallback.
4. `_run_circuit(qc, sampler, pm, num_shots, "output_bits")`.
5. `_get_order_from_dist(dist, A, N, m)`.

### `find_factor(N, ...)`

Logica clasica identica, llamando a `find_order` con sampler y pass_manager.

---

# PARTE 3: FICHEROS __init__.py

Cada framework necesita un `__init__.py` en su directorio raiz y en el subdirectorio `shor/`.
Pueden estar vacios (solo un comentario o nada). Ejemplo:

- `python/cirq/__init__.py` — vacio
- `python/cirq/shor/__init__.py` — vacio
- `python/cudaq/__init__.py` — vacio
- `python/cudaq/shor/__init__.py` — vacio
- etc.

---

# PARTE 4: RESUMEN DE FICHEROS POR FRAMEWORK

| Framework | Ficheros a crear | Estrategia Shor |
|---|---|---|
| **Cirq** | `grover.py`, `shor/shor.py`, `shor/modular_exp.py` | ArithmeticGate (shortcut del simulador) |
| **CUDA-Q** | `grover.py`, `shor/shor.py`, `shor/qft.py`, `shor/permutation.py` | Permutation network + QFT manual |
| **ProjectQ** | `grover.py`, `shor/shor.py` | MultiplyByConstantModN + semi-classical QFT |
| **QDisLib** | `grover.py`, `shor/shor.py` | Wrapper de Qiskit (import + fallback) |

---

# PARTE 5: TESTING

Para verificar que tus implementaciones funcionan, ejecuta para cada framework:

### Grover
```python
# N=3 qubits, target=5
found, dist = search(3, 5, simulator, None, num_shots=1024)
assert found == 5  # debe encontrar el estado correcto
assert dist[format(5, '03b')] > 900  # alta probabilidad (~96% para 3 qubits)
```

### Shor
```python
# Factorizar 15 (= 3 * 5)
factor = find_factor(15, simulator, None, seed=42)
assert factor in (3, 5)  # debe encontrar un factor no trivial
```

### Valores de referencia
- Grover 3 qubits: ~96% de probabilidad del target, 2 iteraciones
- Grover 4 qubits: ~96% de probabilidad del target, 3 iteraciones
- Shor N=15: factores 3 y 5, ordenes posibles r=2 (para a=4) o r=4 (para a=2,7,8,11,13,14)

---

# PARTE 6: ORDEN DE IMPLEMENTACION RECOMENDADO

1. **Grover en Cirq** — el mas parecido a Qiskit, buena practica.
2. **Grover en ProjectQ** — la sintaxis Dirac es diferente pero expresiva.
3. **Grover en CUDA-Q** — el modelo de kernel inline requiere adaptacion.
4. **Grover en QDisLib** — trivial (wrapper de Qiskit).
5. **Shor en Cirq** — ArithmeticGate es elegante pero requiere entender la herencia.
6. **Shor en ProjectQ** — semi-classical QFT es el concepto mas interesante.
7. **Shor en CUDA-Q** — el permutation network es el mas complejo de implementar.
8. **Shor en QDisLib** — trivial (wrapper de Qiskit).
