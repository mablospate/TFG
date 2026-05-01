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

---

# PARTE 7: FRAMEWORKS DE RUST

Esta parte cubre la implementacion equivalente en cuatro frameworks de Rust:
**q1tsim**, **quantr**, **qcgpu** y **quantrs2**. La interfaz publica de cada
modulo refleja la de Python (`build_oracle`, `build_diffuser`, `grover_circuit`,
`search` para Grover; `order_finding_circuit`, `find_order`, `find_factor`,
`_get_order_from_dist` para Shor), pero adaptada a Rust:

- En q1tsim, quantr y quantrs2, los helpers de construccion reciben un
  `&mut Circuit` y aplican puertas in-place (no devuelven un sub-circuito porque
  no existe el concepto generico de "componer subcircuitos" en estos frameworks).
- En qcgpu, el modelo es puramente imperativo sobre `&mut State`.
- `search` y `find_order` devuelven `(u64, HashMap<String, usize>)`.
- `find_factor` devuelve `u64`.

Los ficheros viven en el workspace `rust/` (miembros: `q1tsim`, `qcgpu`,
`quantr`, `quantrs`). Cada miembro tiene `src/bin/grover.rs` y
`src/bin/shor.rs` como puntos de entrada; los modulos de Shor crecen a
`src/bin/shor/` (con `shor.rs`, `qft.rs`, etc.) cuando la complejidad lo pide.

> **Convencion**: como en Python, `n = ceil(log2(N))` para Shor y `n` = numero de
> qubits de busqueda para Grover. `m` = qubits de phase estimation
> (`precision`), por defecto `2 * n`.

---

## 7.0 Tabla de referencia rapida: API de cada framework Rust

| Operacion | q1tsim | quantr | qcgpu | quantrs2 |
|---|---|---|---|---|
| Cargo dep | `q1tsim = "0.5"` | `quantr = "0.6"` | `qcgpu = "0.1"` | `quantrs2 = "0.1"` |
| Crear circuito | `Circuit::new(n, n)` | `Circuit::new(n).unwrap()` | `State::new(n, seed)` | `Circuit::<N>::new()` |
| Allocar qubits | implicito en `new` | implicito en `new` | implicito en `new` | implicito (`<N>`) |
| H en todos | `for i in 0..n { c.h(i)?; }` | `qc.add_repeating_gate(Gate::H, &qs)?` | `for i in 0..n { s.h(i); }` | `for i in 0..N { c.h(i)?; }` |
| X en qubit i | `c.x(i)?` | `qc.add_gate(Gate::X, i)?` | `s.x(i)` | `c.x(i)?` |
| MCZ | manual (custom `Gate`) | manual (`Gate::Custom`) | manual (ancilla / decomposicion) | `c.mcx`+`H` o nativo |
| Medir todos | `c.measure_all(&[0..n-1])?` | `sim.measure_all(shots)` | `s.measure_many(shots)` | `result.measure_all(shots)?` |
| Ejecutar | `c.execute(shots)?` | `qc.simulate()` | implicito (mide directamente) | `sim.run(&c)?` |
| QFT | manual (H + CU1 + Swap) | manual (H + CRk + Swap) | manual (H + cU1 + swap) | `c.qft(start, len)?` (built-in) |
| QFT inversa | manual (orden inverso, angulos negativos) | manual (`CRk(-k, ...)`) | manual (orden inverso) | `c.inverse_qft(start, len)?` |
| Rotacion fase | `add_gate(CU1::new(t), &[q])` | `Gate::Rk(k)` (solo `2pi/2^k`) | `apply_gate(q, gates::u1(t))` | `c.phase(q, t)?` |
| Rot. fase controlada | `add_gate(CU1::new(t), &[c, q])` | `Gate::CRk(k, ctrl)` | `apply_controlled_gate(c, q, gates::u1(t))` | `c.controlled_phase(c, q, t)?` |
| SWAP | `add_gate(Swap::new(), &[a,b])` | `Gate::Swap(b)` aplicado a `a` | `s.swap(a, b)` | `c.swap(a, b)?` |
| Endianness | LSB-first (qubit 0 = LSB) | **MSB-first** (qubit 0 = MSB) | LSB-first | TBD (verificar) |
| Limite practico | ~25 qubits (RAM) | ~16 qubits (single-thread) | depende de GPU/OpenCL | ~25 qubits (RC) |

> **Nota sobre quantrs2**: el crate esta en RC (`0.1.0-rc.1`). Antes de escribir
> codigo, ejecuta `cargo build` contra el crate y comprueba que las APIs aqui
> documentadas existen. Si la build rompe o las APIs cambian, el plan de
> contingencia es sustituir por **RustQIP** (`qip = "1.5"`) o **roqoqo**
> (`roqoqo = "1.16"`); ambos exponen un modelo de `Circuit` parecido al de
> q1tsim.

---

# PARTE 7-A: GROVER EN RUST

La estructura algoritmica (oraculo + difusor + iteracion optima) es identica a
la version Python descrita en la PARTE 1. Lo que cambia entre frameworks de
Rust es:

1. Como se aplica un **MCZ** (Multi-Controlled Z) cuando el framework no tiene
   uno nativo.
2. El **endianness** del bitstring que sale de la medida.
3. Que tipo necesita la firma de `build_oracle`, `build_diffuser` y
   `grover_circuit` (hay que devolver `Result` por el manejo de errores).

---

## 7.1 Grover en q1tsim

**Fichero**: `rust/q1tsim/src/bin/grover.rs`

### Cargo.toml

```toml
[dependencies]
q1tsim = "0.5"
num-complex = "0.4"
```

### Imports necesarios

```rust
use std::collections::HashMap;
use std::error::Error;
use std::f64::consts::PI;
use q1tsim::circuit::Circuit;
use q1tsim::gates::{CCZ, CU1, Swap};
// Si necesitas un MCZ generico para n>=4, define un struct propio (ver 7.10).
```

### Conceptos clave de q1tsim

- **Modelo de circuito**: `Circuit::new(n, n)` crea un circuito con `n` qubits y
  `n` bits clasicos. Las puertas se aplican mediante metodos que devuelven
  `Result<(), Box<dyn Error>>` (las APIs basicas como `h`, `x`, `cx` tienen
  metodos directos; el resto se aplica con `add_gate`).
- **Ejecucion**: `circuit.execute(num_shots)?` corre la simulacion. Despues
  `circuit.histogram_string()?` devuelve un `HashMap<String, usize>` con los
  bitstrings y sus frecuencias.
- **LSB-first**: en `histogram_string()`, el bit clasico 0 es el caracter mas
  a la derecha. Si mides el qubit `i` al bit clasico `i`, el qubit 0 es el LSB
  igual que en Qiskit. **No hay que invertir.**
- **MCZ no es nativo**: para n=3 controles existe `CCZ`; para n>3 hay que
  decomponer (ver 7.10).
- **Composicion limitada**: q1tsim no compone subcircuitos como Qiskit. Por eso
  `build_oracle` y `build_diffuser` reciben `&mut Circuit` y aplican puertas
  in-place.

### Pasos para build_oracle

Firma:

```rust
pub fn build_oracle(
    circuit: &mut Circuit,
    n: usize,
    target: u64,
) -> Result<(), Box<dyn Error>> { ... }
```

1. Para cada `i` en `0..n`: si `(target >> i) & 1 == 0`, aplicar
   `circuit.x(i)?` para mapear `|target>` a `|11...1>`.
2. Aplicar MCZ sobre los `n` qubits:
   - Si `n == 1`: `circuit.z(0)?`.
   - Si `n == 2`: `circuit.add_gate(q1tsim::gates::CZ::new(), &[0, 1])?`.
   - Si `n == 3`: `circuit.add_gate(CCZ::new(), &[0, 1, 2])?`.
   - Si `n >= 4`: usar la estrategia de `Gate` custom (seccion 7.10).
3. Deshacer las X (mismo bucle que paso 1).

### Pasos para build_diffuser

```rust
pub fn build_diffuser(
    circuit: &mut Circuit,
    n: usize,
) -> Result<(), Box<dyn Error>> { ... }
```

1. `for i in 0..n { circuit.h(i)?; }`
2. `for i in 0..n { circuit.x(i)?; }`
3. Aplicar MCZ (mismo helper que en el oraculo).
4. `for i in 0..n { circuit.x(i)?; }`
5. `for i in 0..n { circuit.h(i)?; }`

### Pasos para grover_circuit

```rust
pub fn grover_circuit(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
) -> Result<Circuit, Box<dyn Error>> { ... }
```

1. Calcular las iteraciones optimas:
   ```rust
   let iters = num_iterations.unwrap_or_else(|| {
       (PI / 4.0 * (2f64.powi(n as i32)).sqrt()).floor() as usize
   });
   ```
2. Crear el circuito: `let mut circuit = Circuit::new(n, n)?;`.
3. Superposicion inicial: `for i in 0..n { circuit.h(i)?; }`.
4. Bucle de Grover:
   ```rust
   for _ in 0..iters {
       build_oracle(&mut circuit, n, target)?;
       build_diffuser(&mut circuit, n)?;
   }
   ```
5. Medir todos los qubits: `circuit.measure_all(&(0..n).collect::<Vec<_>>())?;`.
6. Devolver `Ok(circuit)`.

### Pasos para search

```rust
pub fn search(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
    num_shots: usize,
) -> Result<(u64, HashMap<String, usize>), Box<dyn Error>> { ... }
```

1. Construir el circuito: `let mut circuit = grover_circuit(n, target, num_iterations)?;`.
2. Ejecutar: `circuit.execute(num_shots)?;`.
3. Obtener histograma: `let dist = circuit.histogram_string()?;`.
4. Encontrar la entrada con mayor conteo:
   ```rust
   let (best_bs, _) = dist.iter().max_by_key(|(_, &c)| c).expect("histograma vacio");
   let found = u64::from_str_radix(best_bs, 2)?;
   ```
5. Devolver `Ok((found, dist))`.

### Gotcha principal

q1tsim no tiene MCZ nativo para n>=4. La opcion mas limpia para tamanos
moderados (n<=6) es declarar un `struct PhaseFlipAllOnes(usize)` que implemente
el trait `Gate` y devuelva la matriz de `2^n x 2^n` que solo cambia el signo
del estado `|11...1>`. Para n>=7, esa matriz pesa demasiado y conviene cambiar a
la decomposicion con ancillas (seccion 7.10), aunque eso cuesta `n-2` qubits y
bits clasicos extra. Los valores de prueba (Grover 3 y 4 qubits) viven en la
PARTE 5; con `n<=4` basta el camino de `CCZ` directo.

---

## 7.2 Grover en quantr

**Fichero**: `rust/quantr/src/bin/grover.rs`

### Cargo.toml

```toml
[dependencies]
quantr = "0.6"
num-complex = "0.4"
```

### Imports necesarios

```rust
use std::collections::HashMap;
use std::f64::consts::PI;
use quantr::{Circuit, Gate, Measurement::Observable};
use num_complex::Complex64;
```

### Conceptos clave de quantr

- **Circuito por columnas**: las puertas se anaden con `qc.add_gate(Gate::H, i)`
  o, para varias puertas en la misma columna, `qc.add_gates(&[...])`.
- **Repeticion**: `qc.add_repeating_gate(Gate::H, &[0,1,2])` aplica H a varios
  qubits en una sola columna.
- **MSB-first**: quantr es **big-endian**. En la `ProductState` que devuelve
  `measure_all`, el qubit 0 es el caracter mas a la izquierda. Para obtener el
  entero con la convencion qubit-0 = LSB (Qiskit), hay que **invertir el
  bitstring** antes de `u64::from_str_radix`.
- **MCZ**: solo `Toffoli(c1, c2)` es multi-control nativo (n=2 controles). Para
  n>=3 se usa `Gate::Custom(fn, mapping)`, una closure que recibe el slice de
  amplitudes y devuelve el slice modificado.
- **Limite practico**: ~16 qubits (single-thread, statevector denso).
- **Simulacion**: `qc.simulate()` devuelve un objeto sobre el que se llama
  `measure_all(num_shots)`, que retorna `Measurement::Observable(HashMap<ProductState, usize>)`.

### Pasos para build_oracle

Firma:

```rust
pub fn build_oracle(
    qc: &mut Circuit,
    n: usize,
    target: u64,
) -> Result<(), quantr::error::QuantrError> { ... }
```

1. Para cada `i` en `0..n`: si `(target >> i) & 1 == 0`, aplicar
   `qc.add_gate(Gate::X, i)?`. Como quantr es big-endian, conviene mapear el bit
   `i` del target al qubit `n-1-i` igual que en Cirq, **o** ser consistente
   tratando el qubit `i` como bit `i` del target y revertir despues en la
   medida. La eleccion mas simple: trata el qubit `i` como bit `i` y, en
   `search`, invierte el bitstring antes de pasarlo a entero.
2. Aplicar MCZ sobre todos los qubits:
   - `n == 1`: `qc.add_gate(Gate::Z, 0)?`.
   - `n == 2`: `qc.add_gate(Gate::CZ(0), 1)?` (control en 0, target en 1).
   - `n >= 3`: `Gate::Custom` que invierta la amplitud del estado `|11...1>`
     (ver 7.10).
3. Deshacer las X.

### Pasos para build_diffuser

```rust
pub fn build_diffuser(qc: &mut Circuit, n: usize) -> Result<(), _> { ... }
```

1. `qc.add_repeating_gate(Gate::H, &(0..n).collect::<Vec<_>>())?;`
2. `qc.add_repeating_gate(Gate::X, &(0..n).collect::<Vec<_>>())?;`
3. MCZ (mismo helper que en oraculo).
4. `qc.add_repeating_gate(Gate::X, ...)?;`
5. `qc.add_repeating_gate(Gate::H, ...)?;`

### Pasos para grover_circuit

```rust
pub fn grover_circuit(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
) -> Result<Circuit, _> { ... }
```

1. Calcular iteraciones (igual que en q1tsim).
2. `let mut qc = Circuit::new(n).unwrap();`.
3. Superposicion: `qc.add_repeating_gate(Gate::H, &(0..n).collect::<Vec<_>>())?;`.
4. Bucle: `build_oracle` + `build_diffuser` `iters` veces.
5. Devolver `Ok(qc)`.

> Nota: quantr no tiene operaciones de medida programaticas en el circuito;
> la medida la solicita el simulador (`measure_all`). No anadas medidas al
> `Circuit`.

### Pasos para search

```rust
pub fn search(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
    num_shots: usize,
) -> Result<(u64, HashMap<String, usize>), _> { ... }
```

1. Construir el circuito con `grover_circuit`.
2. `let sim = qc.simulate();`.
3. `let counts = match sim.measure_all(num_shots) { Observable(c) => c, _ => unreachable!() };`.
4. Convertir `HashMap<ProductState, usize>` en `HashMap<String, usize>`:
   ```rust
   let mut dist: HashMap<String, usize> = HashMap::new();
   for (state, count) in counts {
       let bs = state.to_string();      // big-endian
       let bs_lsb = bs.chars().rev().collect::<String>();  // pasar a LSB-first
       dist.insert(bs_lsb, count);
   }
   ```
5. Encontrar el maximo:
   ```rust
   let (best_bs, _) = dist.iter().max_by_key(|(_, &c)| c).unwrap();
   let found = u64::from_str_radix(best_bs, 2)?;
   ```
6. Devolver `Ok((found, dist))`.

### Gotcha principal

El **endianness big-endian** es la trampa numero uno. Si te olvidas de invertir
el bitstring antes de `from_str_radix`, los tests pasaran para targets simetricos
(como `target = 0b101`) pero fallaran para asimetricos (como `target = 0b001`),
porque el qubit 0 medido sera el bit mas significativo en la string. Verifica
siempre con un target asimetrico (e.g. `target = 1` con `n = 3`).

---

## 7.3 Grover en qcgpu

**Fichero**: `rust/qcgpu/src/bin/grover.rs`

### Cargo.toml

```toml
[dependencies]
qcgpu = "0.1"
```

> **Requisito de runtime**: qcgpu corre sobre OpenCL. En macOS esta integrado en
> el sistema. En Linux se necesita `apt install ocl-icd-opencl-dev` y un
> driver compatible. La CI debe provisionar esto antes de ejecutar los tests.

### Imports necesarios

```rust
use std::collections::HashMap;
use std::f64::consts::PI;
use qcgpu::State;
use qcgpu::gates;  // u1(angle) y demas
```

### Conceptos clave de qcgpu

- **Modelo imperativo**: no existe un objeto `Circuit`. Se trabaja sobre un
  `State` mutable que representa el statevector. Las puertas son metodos
  imperativos que mutan el state.
- **Multi-shot**: `state.measure_many(num_shots)` devuelve
  `HashMap<String, i32>` haciendo *resampling* sobre la distribucion final.
  Esto evita reconstruir el state por cada shot (que es lo que sucede con
  `state.measure()`, que ademas colapsa el state).
- **Precision f32**: los angulos son `f32`. Para `m` grande (Shor con `m>=20`)
  la perdida de precision se acumula. Para Grover con `n<=6` no es problema.
- **MCZ**: `state.toffoli(c1, c2, t)` es el control mas alto nativo. Para n>=3
  controles hay que decomponer con ancilla o usar la estrategia de la seccion
  7.10.
- **LSB-first**: la string de `measure_many` tiene el qubit 0 como caracter
  mas a la derecha (LSB). Coincide con Qiskit.

### Pasos para build_oracle

Como qcgpu no tiene `Circuit`, los helpers reciben `&mut State`:

```rust
pub fn build_oracle(state: &mut State, n: usize, target: u64) { ... }
```

1. Para cada `i` en `0..n`: si `(target >> i) & 1 == 0`, `state.x(i);`.
2. MCZ sobre los `n` qubits:
   - `n == 1`: `state.z(0);`.
   - `n == 2`: `state.cz(0, 1);`.
   - `n == 3`: implementar CCZ via `H` + `Toffoli` + `H`:
     ```rust
     state.h(2);
     state.toffoli(0, 1, 2);
     state.h(2);
     ```
   - `n >= 4`: ancilla-based MCZ (seccion 7.10).
3. Deshacer las X.

### Pasos para build_diffuser

```rust
pub fn build_diffuser(state: &mut State, n: usize) { ... }
```

1. `for i in 0..n { state.h(i); }`
2. `for i in 0..n { state.x(i); }`
3. MCZ.
4. `for i in 0..n { state.x(i); }`
5. `for i in 0..n { state.h(i); }`

### Pasos para grover_circuit

En qcgpu el equivalente a `grover_circuit` es una funcion que **devuelve un
state ya inicializado y con todas las puertas aplicadas**, listo para medir.

```rust
pub fn grover_circuit(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
) -> State { ... }
```

1. Calcular iteraciones.
2. `let mut state = State::new(n, 0);` (segundo argumento: seed RNG).
3. Superposicion: `for i in 0..n { state.h(i); }`.
4. Bucle: oracle + diffuser, `iters` veces.
5. Devolver `state`.

### Pasos para search

```rust
pub fn search(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
    num_shots: usize,
) -> (u64, HashMap<String, usize>) { ... }
```

1. Construir el state con `grover_circuit`.
2. `let raw = state.measure_many(num_shots);` (`HashMap<String, i32>`).
3. Convertir conteos a `usize` para casar con la interfaz comun:
   ```rust
   let dist: HashMap<String, usize> =
       raw.into_iter().map(|(k, v)| (k, v as usize)).collect();
   ```
4. `let (best_bs, _) = dist.iter().max_by_key(|(_, &c)| c).unwrap();`
5. `let found = u64::from_str_radix(best_bs, 2).unwrap();`
6. Devolver `(found, dist)`.

### Gotcha principal

`State::measure()` (singular) **colapsa** el state, asi que despues de llamarlo
no se pueden seguir aplicando puertas ni medir otra vez de forma significativa.
Para multi-shot SIEMPRE se usa `measure_many`, que resamplea la distribucion
sin colapsar. Si por alguna razon necesitas mediciones independientes con
re-ejecucion completa, hay que reconstruir el state desde cero en cada shot
(e.g. en bucles donde la seed cambia entre shots).

---

## 7.4 Grover en quantrs2

**Fichero**: `rust/quantrs/src/bin/grover.rs` (directorio `quantrs/`, crate `quantrs2`).

### Cargo.toml

```toml
[dependencies]
quantrs2 = "0.1"
quantrs2-circuit = "0.1"
quantrs2-sim = "0.1"
```

> **Aviso de estabilidad**: `quantrs2` esta en RC. Algunas APIs aqui
> documentadas (especialmente `qft`, `inverse_qft` y `mcx`) pueden cambiar
> entre RCs. Antes de empezar, ejecuta `cargo build` en el crate y, si falla,
> activa el plan de contingencia (RustQIP o roqoqo).

### Imports necesarios

```rust
use std::collections::HashMap;
use std::f64::consts::PI;
use quantrs2_circuit::builder::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
```

### Conceptos clave de quantrs2

- **Const generics**: `Circuit::<N>::new()` recibe el numero de qubits como
  parametro de tipo. Esto significa que **N debe ser un literal o constante en
  tiempo de compilacion**. Para soportar `n` dinamico hay tres opciones:
  1. Hacer todas las funciones genericas: `fn build_oracle<const N: usize>(target: u64) -> Circuit<N>`.
     Esto exige instanciar el binario para cada `N` que quieras probar.
  2. Usar `DynCircuit` si el crate la expone (verificar en RC actual).
  3. Para los tests reales, expone funciones publicas para los tamanos que
     usen los tests (e.g. `grover_circuit_3`, `grover_circuit_4`) y luego un
     dispatcher por `match n { 3 => grover_circuit_3(target), 4 => ... }`.
- **Method chaining**: las puertas devuelven `Result<&mut Circuit<N>>`, asi
  que se encadenan: `circuit.h(0)?.x(1)?.cnot(0, 1)?;`.
- **Built-ins**: el README documenta `qft`, `inverse_qft` y `mcx` (multi-control
  X) directamente sobre el circuito; `mcz` se obtiene como `H` + `mcx` + `H`.
- **Endianness**: documentado como TBD. Antes de validar tests, ejecuta un
  circuito conocido (e.g. preparar `|001>` y medir) para confirmar.

### Pasos para build_oracle

```rust
pub fn build_oracle<const N: usize>(
    circuit: &mut Circuit<N>,
    target: u64,
) -> Result<(), Box<dyn std::error::Error>> { ... }
```

1. Para cada `i` en `0..N`: si `(target >> i) & 1 == 0`, `circuit.x(i)?;`.
2. Aplicar MCZ sobre los `N` qubits:
   - `N == 1`: `circuit.z(0)?;`.
   - `N == 2`: `circuit.cz(0, 1)?;`.
   - `N >= 3`: el patron canonico es `H(target) + mcx(controls, target) + H(target)`:
     ```rust
     let target_q = N - 1;
     let controls: Vec<usize> = (0..N - 1).collect();
     circuit.h(target_q)?;
     circuit.mcx(&controls, target_q)?;
     circuit.h(target_q)?;
     ```
3. Deshacer las X.

> Si `mcx` no estuviera disponible, ver seccion 7.10 (decomposicion con
> ancillas o con Toffolis).

### Pasos para build_diffuser

```rust
pub fn build_diffuser<const N: usize>(circuit: &mut Circuit<N>) -> Result<(), _> { ... }
```

1. `for i in 0..N { circuit.h(i)?; }`
2. `for i in 0..N { circuit.x(i)?; }`
3. MCZ (mismo helper).
4. `for i in 0..N { circuit.x(i)?; }`
5. `for i in 0..N { circuit.h(i)?; }`

### Pasos para grover_circuit

```rust
pub fn grover_circuit<const N: usize>(
    target: u64,
    num_iterations: Option<usize>,
) -> Result<Circuit<N>, _> { ... }
```

1. Calcular iteraciones a partir de `N`.
2. `let mut circuit = Circuit::<N>::new();`.
3. Superposicion: `for i in 0..N { circuit.h(i)?; }`.
4. Bucle: `build_oracle::<N>` + `build_diffuser::<N>`.
5. Devolver `Ok(circuit)`.

### Pasos para search

```rust
pub fn search<const N: usize>(
    target: u64,
    num_iterations: Option<usize>,
    num_shots: usize,
) -> Result<(u64, HashMap<String, usize>), _> { ... }
```

1. Construir el circuito con `grover_circuit::<N>(...)`.
2. `let sim = StateVectorSimulator::new();`.
3. `let result = sim.run(&circuit)?;`.
4. `let dist = result.measure_all(num_shots)?;` (`HashMap<String, usize>`).
5. Si la verificacion previa de endianness mostro big-endian, invertir las keys
   antes de buscar el maximo.
6. `let (best_bs, _) = dist.iter().max_by_key(|(_, &c)| c).unwrap();`
7. `let found = u64::from_str_radix(best_bs, 2)?;`
8. Devolver `Ok((found, dist))`.

### Gotcha principal

Los **const generics** convierten `n` en parte del tipo. Esto rompe el patron
"el binario lee `n` por argv y llama a `grover_circuit(n, target)`", porque no
puedes elegir `N` en runtime. Soluciones practicas:

- Para ejecutar como CLI, expon funciones para los `N` exactos que necesiten
  los tests (3, 4 son los que estan en la PARTE 5) y dispatcha con un `match`
  sobre `n`.
- Para pruebas internas, todo es generico y `cargo test` resuelve `N` en
  tiempo de compilacion.

Si esto te bloquea, considera el `DynCircuit` (si existe en la version del
crate) o pasa al fallback (RustQIP).

---

# PARTE 7-B: SHOR EN RUST

La estructura clasica (`find_factor`) es **identica** en los cuatro frameworks
de Rust y replica la version Python descrita en la PARTE 2. Lo que cambia es
**como se construye el circuito de phase estimation** y, especialmente, **como
se implementa la exponenciacion modular controlada**.

Resumen de estrategias:

| Framework | Estrategia Shor | QFT | Mod. exp. controlada |
|---|---|---|---|
| q1tsim | Manual completo | Manual (H + CU1 + Swap) | Permutation network |
| quantr | Manual con `Gate::CRk` | Manual (H + CRk + Swap) | Permutation network via `Gate::Custom` |
| qcgpu | Manual completo | Manual (H + cU1 + swap) | Permutation network (clon de CUDA-Q) |
| quantrs2 | Built-in QFT + custom mod-exp | `circuit.qft(...)` | Permutation network o `mcx` ladders |

> **Sobre el post-procesamiento clasico**: `_get_order_from_dist` es **identica
> en los cuatro frameworks** y a la version Python (PARTE 2). Toma
> `HashMap<String, usize>`, ordena por frecuencia descendente, ignora `x == 0`,
> aplica fracciones continuas (`Fraction(x, 2^m).limit_denominator(N-1)`) y
> verifica con `pow(A, r, N) == 1`. Para Rust se usa la crate
> [`num-rational`](https://docs.rs/num-rational) o una implementacion propia
> de algoritmo de fracciones continuas. Pega esa funcion en un modulo
> compartido (e.g. `rust/common/src/post.rs`) si el workspace lo permite.

---

## 7.5 Shor en q1tsim

**Ficheros**: `rust/q1tsim/src/bin/shor/shor.rs`, `rust/q1tsim/src/bin/shor/qft.rs`,
`rust/q1tsim/src/bin/shor/permutation.rs`.

### Cargo.toml

```toml
[dependencies]
q1tsim = "0.5"
num-complex = "0.4"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

### Estrategia

q1tsim no tiene aritmetica modular built-in. Se replica la estrategia de
**permutation network** que ya se usa en CUDA-Q (Python, seccion 2.2): cada
`A^(2^i) mod N` es una permutacion sobre `{0, ..., N-1}` que se descompone en
ciclos disjuntos, cada ciclo en transposiciones, y cada transposicion en una
secuencia de multi-controlled X.

### Estructura del modulo Shor

```
rust/q1tsim/src/bin/shor/
  shor.rs         <- entrypoint con order_finding_circuit, find_order, find_factor
  qft.rs          <- apply_qft, apply_inverse_qft (manuales)
  permutation.rs  <- build_mod_exp_permutation, controlled_swap_permutation,
                     controlled_transposition (paralelo a python/cudaq/shor/permutation.py)
```

### Pasos para `qft.rs`

```rust
pub fn apply_qft(c: &mut Circuit, qubits: &[usize]) -> Result<(), _> {
    let n = qubits.len();
    for i in 0..n {
        c.h(qubits[i])?;
        for j in (i + 1)..n {
            let k = (j - i + 1) as i32;
            let angle = PI / 2f64.powi(k - 1);
            c.add_gate(CU1::new(angle), &[qubits[j], qubits[i]])?;
        }
    }
    for i in 0..n / 2 {
        c.add_gate(Swap::new(), &[qubits[i], qubits[n - 1 - i]])?;
    }
    Ok(())
}

pub fn apply_inverse_qft(c: &mut Circuit, qubits: &[usize]) -> Result<(), _> {
    let n = qubits.len();
    for i in 0..n / 2 {
        c.add_gate(Swap::new(), &[qubits[i], qubits[n - 1 - i]])?;
    }
    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            let k = (j - i + 1) as i32;
            let angle = -PI / 2f64.powi(k - 1);
            c.add_gate(CU1::new(angle), &[qubits[j], qubits[i]])?;
        }
        c.h(qubits[i])?;
    }
    Ok(())
}
```

### Pasos para `permutation.rs`

Replica la logica de `python/cudaq/shor/permutation.py` en Rust:

1. `build_mod_exp_permutation(A: u64, N: u64, power: u64) -> HashMap<u64, u64>`:
   ```rust
   let a_power = mod_pow(A, power, N);
   let mut perm = HashMap::new();
   for y in 0..N {
       let target = (a_power * y) % N;
       if y != target { perm.insert(y, target); }
   }
   ```

2. `controlled_swap_permutation(c, ctrl, target_qubits, perm)`:
   - Identificar ciclos disjuntos siguiendo `perm`.
   - Por cada ciclo `[c0, c1, c2, ...]`, descomponer en transposiciones
     `(c0, c1)`, `(c0, c2)`, ... y aplicar `controlled_transposition` para cada
     una.

3. `controlled_transposition(c, ctrl, target_qubits, a, b)`:
   - Si `a XOR b` tiene exactamente un bit a 1: caso base, ver paso 4.
   - Si tiene varios: tomar el primer bit como pivote, calcular
     `a' = a XOR (1 << pivot)`, y recursar:
     `transposition(a, a')` -> `transposition(a', b)` -> `transposition(a, a')`.

4. Caso base de transposicion de un solo bit:
   ```rust
   // diff_bit es la unica posicion donde a y b difieren.
   // Para que el multi-control X dispare en el estado |a> (o |b>), aplicamos
   // X temporalmente a los qubits cuyo bit en a sea 0 (excluyendo diff_bit).
   for pos in 0..target_qubits.len() {
       if pos == diff_bit { continue; }
       if (a >> pos) & 1 == 0 { c.x(target_qubits[pos])?; }
   }
   // Multi-controlled X con ctrl + todos los target_qubits[pos != diff_bit]
   // como controles, y target_qubits[diff_bit] como objetivo.
   apply_mcx(c, &controls, target_qubits[diff_bit])?;
   for pos in 0..target_qubits.len() {
       if pos == diff_bit { continue; }
       if (a >> pos) & 1 == 0 { c.x(target_qubits[pos])?; }
   }
   ```
   `apply_mcx` para 1-2 controles usa `cx`/`CCX` directamente; para mas
   controles, decomposicion con ancilla (seccion 7.10) o `Gate` custom.

### Pasos para `order_finding_circuit`

Firma:

```rust
pub fn order_finding_circuit(
    a: u64,
    n: u64,
    precision: Option<usize>,
) -> Result<Circuit, _> { ... }
```

1. Verificar `gcd(a, n) == 1` (sino `Err`).
2. Calcular `width = ceil(log2(n))` (numero de qubits del registro target),
   `m = precision.unwrap_or(2 * width)`, `total = m + width`.
3. Crear `let mut c = Circuit::new(total, m)?;` (los `m` bits clasicos solo
   miden los qubits de control).
4. H en todos los qubits de control: `for i in 0..m { c.h(i)?; }`.
5. X en `target_qubits[0]` (qubit `m`): `c.x(m)?;`.
6. Exponenciacion modular controlada:
   ```rust
   let target_qubits: Vec<usize> = (m..total).collect();
   for i in 0..m {
       let perm = build_mod_exp_permutation(a, n, 1u64 << i);
       if perm.is_empty() { continue; }
       controlled_swap_permutation(&mut c, i, &target_qubits, &perm)?;
   }
   ```
7. QFT inversa en los `m` qubits de control:
   `apply_inverse_qft(&mut c, &(0..m).collect::<Vec<_>>())?;`.
8. Medir solo los qubits de control: `for i in 0..m { c.measure(i, i)?; }`.
9. Devolver `Ok(c)`.

### Pasos para `find_order`

```rust
pub fn find_order(
    a: u64,
    n: u64,
    precision: Option<usize>,
    num_shots: usize,
) -> Result<(u64, HashMap<String, usize>), _> { ... }
```

1. Verificar gcd.
2. Calcular `m`.
3. Construir el circuito.
4. Ejecutar: `c.execute(num_shots)?;`.
5. Obtener distribucion: `let dist = c.histogram_string()?;`.
6. Llamar al post-procesado clasico: `let r = get_order_from_dist(&dist, a, n, m);`.
7. Devolver `(r, dist)`.

### Pasos para `find_factor`

Logica clasica identica a Python (seccion 2.0):

```rust
pub fn find_factor(n: u64, num_tries: u32, seed: Option<u64>) -> u64 { ... }
```

1. Si `n` par: devolver 2.
2. Si `n = d^k` para algun `d, k > 1`: devolver `d`.
3. Bucle `num_tries` veces:
   a. Elegir `a` aleatorio en `[2, n-1]`.
   b. Si `gcd(a, n) > 1`: lucky guess, devolver `gcd(a, n)`.
   c. Llamar a `find_order(a, n, ...)`.
   d. Si `r != 0` y `r % 2 == 0`: calcular `x = a^(r/2) mod n - 1`,
      `d = gcd(x, n)`. Si `1 < d < n`: devolver `d`.
4. Devolver `1` si no se encontro.

### Gotchas

1. **MCX para >=3 controles**: el caso base de `controlled_transposition`
   acaba pidiendo un MCX con `1 + (width - 1)` controles (el `ctrl` mas todos
   los qubits target salvo el `diff_bit`). Para `width >= 3` ya son 3 o mas
   controles. q1tsim no tiene esto nativo: usa decomposicion con ancilla
   (seccion 7.10).

2. **Endianness al post-procesar**: `histogram_string()` da bitstrings con bit
   clasico 0 a la derecha. En `_get_order_from_dist`, parseando con
   `u64::from_str_radix(bs, 2)` se obtiene el entero `x` correcto sin
   inversion. Verificalo con `a = 7`, `n = 15`, donde los picos esperados son
   `x = 0`, `64`, `128`, `192` (con `m = 8`).

3. **Tamano**: para `n = 15` (`width = 4`, `m = 8`) el total es 12 qubits. Es
   factible pero ya esta cerca del limite practico de q1tsim en single-thread.

---

## 7.6 Shor en quantr

**Ficheros**: `rust/quantr/src/bin/shor/shor.rs`, `rust/quantr/src/bin/shor/qft.rs`,
`rust/quantr/src/bin/shor/permutation.rs`.

### Cargo.toml

```toml
[dependencies]
quantr = "0.6"
num-complex = "0.4"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

### Estrategia

Igual que en q1tsim: **permutation network** + QFT manual. La diferencia es
que quantr tiene `Gate::CRk(k, ctrl)` que aplica `2*pi / 2^k` exactamente, lo
cual encaja perfectamente con la QFT (las rotaciones de la QFT son
`pi / 2^(k-1) = 2*pi / 2^k`). No hay rotaciones continuas en quantr, asi que
para Shor canonico (que solo usa potencias de dos) basta y sobra.

### Estructura del modulo Shor

```
rust/quantr/src/bin/shor/
  shor.rs
  qft.rs
  permutation.rs
```

### Pasos para `qft.rs`

```rust
pub fn apply_qft(qc: &mut Circuit, start: usize, n: usize) -> Result<(), _> {
    for i in 0..n {
        qc.add_gate(Gate::H, start + i)?;
        for j in (i + 1)..n {
            let k = (j - i + 1) as i32;
            // CRk(k, ctrl) aplicado a target: rotacion 2*pi / 2^k
            qc.add_gate(Gate::CRk(k, start + j), start + i)?;
        }
    }
    for i in 0..n / 2 {
        qc.add_gate(Gate::Swap(start + n - 1 - i), start + i)?;
    }
    Ok(())
}

pub fn apply_inverse_qft(qc: &mut Circuit, start: usize, n: usize) -> Result<(), _> {
    for i in 0..n / 2 {
        qc.add_gate(Gate::Swap(start + n - 1 - i), start + i)?;
    }
    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            let k = (j - i + 1) as i32;
            qc.add_gate(Gate::CRk(-k, start + j), start + i)?;  // angulo negativo
        }
        qc.add_gate(Gate::H, start + i)?;
    }
    Ok(())
}
```

> **Detalle clave**: en quantr, `CRk(-k, ctrl)` codifica un angulo
> `-2*pi / 2^k` (la libreria interpreta `k` con signo). Esto evita tener que
> implementar una rotacion continua, que quantr no soporta.

### Pasos para `permutation.rs`

La logica es la misma que en q1tsim, pero con dos diferencias:

1. **MCX**: quantr solo tiene `Toffoli(c1, c2)` (2 controles). Para mas
   controles, la opcion mas idiomatica es `Gate::Custom(fn, mapping)`. Una
   alternativa mas elegante (y mas eficiente para Shor) es **definir la
   permutacion completa como un solo `Gate::Custom`** sobre el registro target
   y los qubits de control implicados en ese factor:
   ```rust
   // Para el factor i (controlado por qubit i de control), construye una
   // closure que lee el bit del control en el statevector y, si es 1,
   // permuta segun a_power. Si es 0, deja el state intacto.
   let perm_gate = Gate::Custom(
       move |amps: &[Complex64]| -> Vec<Complex64> {
           // implementacion concreta de la permutacion
           ...
       },
       all_qubits_involved,  // lista [ctrl_i, target_qubits...]
   );
   qc.add_gate(perm_gate, 0)?;
   ```
   Esto es esencialmente el mismo "shortcut" que `cirq.ArithmeticGate` (seccion
   2.1 de Python): el simulador evalua la matriz directamente sin descomponer
   en puertas reales, lo cual es valido para benchmarking de orden cuantico
   pero NO es un benchmark realista de gates. Documentalo en el TFG.

2. **Endianness**: `Gate::Custom` opera sobre el statevector con la convencion
   big-endian de quantr (qubit 0 = MSB). Al construir la tabla de permutacion,
   tenlo en cuenta: el indice de un estado `|q_0 q_1 ... q_{N-1}>` en la lista
   de amplitudes es `(q_0 << (N-1)) | (q_1 << (N-2)) | ... | q_{N-1}`. Si en la
   referencia de Python (CUDA-Q) usas la convencion `(q_{N-1} << (N-1)) | ...`,
   tendras que invertir la asignacion.

### Pasos para `order_finding_circuit`

```rust
pub fn order_finding_circuit(a: u64, n: u64, precision: Option<usize>) -> Result<Circuit, _> { ... }
```

1. Verificar `gcd`.
2. Calcular `width`, `m`, `total`.
3. `let mut qc = Circuit::new(total).unwrap();`.
4. Superposicion de control: `qc.add_repeating_gate(Gate::H, &(0..m).collect::<Vec<_>>())?;`.
5. Inicializar target a `|1>`: `qc.add_gate(Gate::X, m)?;`.
6. Aplicar el `Gate::Custom` que codifica la exponenciacion modular controlada
   (o, alternativamente, el bucle de transposiciones via Toffoli + ancillas).
7. QFT inversa sobre los qubits 0..m.
8. Devolver `Ok(qc)`.

### Pasos para `find_order`

```rust
pub fn find_order(a: u64, n: u64, precision: Option<usize>, num_shots: usize) -> Result<(u64, HashMap<String, usize>), _> { ... }
```

1. Construir el circuito.
2. `let sim = qc.simulate(); let counts = sim.measure_all(num_shots);`.
3. Convertir `HashMap<ProductState, usize>` a `HashMap<String, usize>`,
   **invirtiendo** los bitstrings para pasar a LSB-first.
4. Extraer solo los `m` primeros bits (los de control). Si la inversion ya
   coloca el control en posiciones LSB, basta truncar.
5. `let r = get_order_from_dist(&dist, a, n, m);`.
6. Devolver `(r, dist)`.

### Pasos para `find_factor`

Identico a q1tsim (logica clasica). Solo cambia la llamada interna a
`find_order`.

### Gotchas

1. **Sin rotacion continua**: quantr **solo** tiene `Rk` y `CRk`. Si por algun
   motivo (e.g. variantes de Shor con phase kickback no diadico) necesitas
   `R(theta)` con theta no diadico, quantr no lo permite — tendrias que
   escribir un `Gate::Custom` con la matriz de la rotacion.

2. **Endianness al construir la permutacion**: como `Gate::Custom` recibe
   amplitudes indexadas big-endian, si reciclas la logica de Python tienes que
   reordenar los bits o reordenar la tabla de permutacion. La forma menos
   propensa a error es escribir un test `n = 15`, `a = 7`, `m = 8` y
   verificar que los picos del histograma estan en `0, 64, 128, 192` (despues
   de invertir).

3. **Limite practico**: ~16 qubits significa que para `n = 21` (`width = 5`,
   `m = 10`) el total es 15 qubits — al limite. Para `n = 33` ya no entra.

---

## 7.7 Shor en qcgpu

**Ficheros**: `rust/qcgpu/src/bin/shor/shor.rs`, `rust/qcgpu/src/bin/shor/qft.rs`,
`rust/qcgpu/src/bin/shor/permutation.rs`.

### Cargo.toml

```toml
[dependencies]
qcgpu = "0.1"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

### Estrategia

**Permutation network**, copia practicamente identica de la version Python
para CUDA-Q. La unica diferencia material es que las rotaciones de la QFT
manual son en `f32`, lo cual introduce algo de error para `m` grande.

### Estructura del modulo Shor

```
rust/qcgpu/src/bin/shor/
  shor.rs
  qft.rs
  permutation.rs
```

### Pasos para `qft.rs`

```rust
pub fn apply_qft(state: &mut State, qubits: &[usize]) {
    let n = qubits.len();
    for i in 0..n {
        state.h(qubits[i]);
        for j in (i + 1)..n {
            let angle = (PI / 2f64.powi((j - i) as i32)) as f32;
            state.apply_controlled_gate(qubits[j], qubits[i], gates::u1(angle));
        }
    }
    for i in 0..n / 2 {
        state.swap(qubits[i], qubits[n - 1 - i]);
    }
}

pub fn apply_inverse_qft(state: &mut State, qubits: &[usize]) {
    let n = qubits.len();
    for i in 0..n / 2 {
        state.swap(qubits[i], qubits[n - 1 - i]);
    }
    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            let angle = -(PI / 2f64.powi((j - i) as i32)) as f32;
            state.apply_controlled_gate(qubits[j], qubits[i], gates::u1(angle));
        }
        state.h(qubits[i]);
    }
}
```

### Pasos para `permutation.rs`

Misma estructura que la version de q1tsim, pero operando sobre `&mut State`:

- `controlled_swap_permutation` aplica controlled-X usando
  `state.cx(...)` y para multi-control usa una funcion auxiliar
  `apply_mcx(state, controls, target)` que usa Toffoli + ancilla cuando hay
  >2 controles.
- `controlled_transposition` y `_controlled_single_bit_transposition`
  exactamente como en CUDA-Q (Python).

### Pasos para `order_finding_circuit`

En qcgpu no hay objeto circuito. La funcion correspondiente devuelve un
**State ya preparado** (sin medir):

```rust
pub fn order_finding_state(a: u64, n: u64, precision: Option<usize>) -> State { ... }
```

> **Nota de naming**: como qcgpu no tiene "circuit", la funcion publica
> mantiene el nombre `order_finding_circuit` por consistencia con la interfaz
> comun y devuelve un `State`. Internamente puede aliasarla a
> `order_finding_state` si prefieres claridad.

1. Calcular `width`, `m`, `total`.
2. `let mut state = State::new(total, 0);`.
3. H en los `m` qubits de control: `for i in 0..m { state.h(i); }`.
4. X en `state.x(m)` (target a `|1>`).
5. Bucle de exp. modular: para cada `i` en `0..m`, construir la permutacion
   correspondiente a `a^(2^i) mod n` y aplicar
   `controlled_swap_permutation(&mut state, i, &target_qubits, &perm);`.
6. QFT inversa: `apply_inverse_qft(&mut state, &(0..m).collect::<Vec<_>>());`.
7. Devolver `state`.

### Pasos para `find_order`

```rust
pub fn find_order(a: u64, n: u64, precision: Option<usize>, num_shots: usize) -> (u64, HashMap<String, usize>) { ... }
```

1. Construir el state.
2. `let raw = state.measure_many(num_shots);`.
3. Para cada bitstring, **extraer solo los primeros `m` bits del lado LSB**
   (el qcgpu coloca qubit 0 a la derecha, y el control esta en los qubits 0..m
   por construccion):
   ```rust
   let mut dist: HashMap<String, usize> = HashMap::new();
   for (bs, count) in raw {
       let total = bs.len();
       // qubit 0 esta a la derecha; los qubits 0..m son los m caracteres
       // mas a la derecha
       let ctrl_bits = &bs[total - m..];
       *dist.entry(ctrl_bits.to_string()).or_insert(0) += count as usize;
   }
   ```
4. `let r = get_order_from_dist(&dist, a, n, m);`.
5. Devolver `(r, dist)`.

### Pasos para `find_factor`

Identica a la version de q1tsim.

### Gotchas

1. **Precision f32**: para `m = 8` (suficiente para `n = 15`) la perdida es
   tolerable. Para `m >= 16` (que harias para `n = 33` o mayor) las rotaciones
   acumulan ruido suficiente para que los picos de la QFT se desplacen. Si los
   tests fallan con `n` grande, prueba a **reducir `precision`** explicitamente
   o documentalo como limitacion del backend.

2. **OpenCL en CI**: la build local en macOS funciona out-of-the-box; en CI
   Linux hay que instalar `ocl-icd-opencl-dev` y un driver. Si la CI no lo
   tiene, los tests de qcgpu deben skippearse con un cfg-flag, no fallar.

3. **`measure_many` colapsa o no?**: a diferencia de `measure()`,
   `measure_many` muestrea repetidamente sin recolapsar el state global, asi
   que es seguro llamarlo despues de construir el state.

---

## 7.8 Shor en quantrs2

**Ficheros**: `rust/quantrs/src/bin/shor/shor.rs`, posiblemente
`rust/quantrs/src/bin/shor/permutation.rs` (si los built-ins no cubren todo).

### Cargo.toml

```toml
[dependencies]
quantrs2 = "0.1"
quantrs2-circuit = "0.1"
quantrs2-sim = "0.1"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

### Estrategia

quantrs2 dice tener `qft` e `inverse_qft` built-in. Estrategia:

1. Usar `circuit.qft(start, len)?` y `circuit.inverse_qft(start, len)?`
   directamente.
2. Para la exp. modular controlada, NO hay built-in: implementar el
   permutation network con `circuit.mcx(&controls, target)?` cuando esta
   disponible. Si `mcx` no esta, escribir un `Gate::Custom`-equivalente o
   decomponer con Toffoli + ancillas (seccion 7.10).

### Estructura del modulo Shor

```
rust/quantrs/src/bin/shor/
  shor.rs
  permutation.rs    (solo si mcx no es suficiente)
```

No hace falta `qft.rs` separada porque la QFT viene con el crate.

### Pasos para `order_finding_circuit`

> **Atencion al const generic**: como en Grover, `Circuit::<TOTAL>` requiere
> que `TOTAL = M + WIDTH` sea conocido en compile-time. Para `n = 15` y
> `precision = 2 * width = 8` esto da `TOTAL = 12`. Define una funcion
> generica con dos parametros const:
>
> ```rust
> pub fn order_finding_circuit<const M: usize, const WIDTH: usize>(
>     a: u64,
>     n: u64,
> ) -> Result<Circuit<{ M + WIDTH }>, _> { ... }
> ```
>
> Esto requiere `feature(generic_const_exprs)` en nightly, o evitarlo
> definiendo un solo parametro `TOTAL` y pasando `M` y `WIDTH` como argumentos
> de runtime usados solo para indexar.

1. Verificar `gcd(a, n) == 1`.
2. Crear `let mut c = Circuit::<TOTAL>::new();`.
3. Superposicion de control: `for i in 0..m { c.h(i)?; }`.
4. Inicializar target a `|1>`: `c.x(m)?;`.
5. Exponenciacion modular controlada (permutation network igual que en
   secciones 7.5-7.7, usando `c.mcx(&controls, target)?` para los multi-X).
6. QFT inversa: `c.inverse_qft(0, m)?;`.
7. Devolver `Ok(c)`.

### Pasos para `find_order`

```rust
pub fn find_order<const TOTAL: usize>(
    a: u64,
    n: u64,
    precision: usize,
    num_shots: usize,
) -> Result<(u64, HashMap<String, usize>), _> { ... }
```

1. Construir el circuito.
2. `let sim = StateVectorSimulator::new(); let result = sim.run(&c)?;`.
3. `let counts = result.measure_all(num_shots)?;` — `HashMap<String, usize>`.
4. Verificar endianness con un caso conocido y, si hace falta, invertir.
5. Truncar al prefijo/sufijo correspondiente para extraer solo los bits de
   control.
6. `let r = get_order_from_dist(&dist, a, n, precision);`.
7. Devolver `(r, dist)`.

### Pasos para `find_factor`

Identica a las otras versiones (logica clasica).

### Gotchas

1. **Const generics + Shor**: como `TOTAL` es compile-time, no puedes usar
   `find_factor` con `n` arbitrario. Define entradas para los `n` que estan en
   los tests (PARTE 5: principalmente `n = 15`) y compone un dispatch por
   `match`.

2. **API en RC**: si `qft`, `inverse_qft` o `mcx` cambian de nombre o firma
   entre RCs, la build se rompera silenciosamente. Pin la version exacta:
   `quantrs2 = "=0.1.0-rc.1"` para que `cargo update` no rompa el build.

3. **Plan B**: si quantrs2 no compila o falta `mcx`, sustituye por:
   - **RustQIP** (`qip = "1.5"`): API muy parecida a Qiskit con
     `register_circuit_builder`, soporta multi-control y QFT.
   - **roqoqo** (`roqoqo = "1.16"`): orientado a circuitos secuenciales,
     soporta `MultiControlledX` y QFT manual.
   En ambos casos, la estrategia algoritmica (permutation network + QFT) se
   reutiliza tal cual; solo cambian los nombres de las puertas.

---

## 7.9 Ficheros Cargo.toml

Bloque `[dependencies]` minimo para cada miembro del workspace:

### `rust/q1tsim/Cargo.toml`

```toml
[package]
name = "q1tsim_bench"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "grover"
path = "src/bin/grover.rs"

[[bin]]
name = "shor"
path = "src/bin/shor/shor.rs"

[dependencies]
q1tsim = "0.5"
num-complex = "0.4"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

### `rust/quantr/Cargo.toml`

```toml
[package]
name = "quantr_bench"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "grover"
path = "src/bin/grover.rs"

[[bin]]
name = "shor"
path = "src/bin/shor/shor.rs"

[dependencies]
quantr = "0.6"
num-complex = "0.4"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

### `rust/qcgpu/Cargo.toml`

```toml
[package]
name = "qcgpu_bench"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "grover"
path = "src/bin/grover.rs"

[[bin]]
name = "shor"
path = "src/bin/shor/shor.rs"

[dependencies]
qcgpu = "0.1"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

### `rust/quantrs/Cargo.toml`

```toml
[package]
name = "quantrs_bench"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "grover"
path = "src/bin/grover.rs"

[[bin]]
name = "shor"
path = "src/bin/shor/shor.rs"

[dependencies]
quantrs2 = "=0.1.0-rc.1"
quantrs2-circuit = "=0.1.0-rc.1"
quantrs2-sim = "=0.1.0-rc.1"
num-integer = "0.1"
num-rational = "0.4"
rand = "0.8"
```

> **Workspace root** (`rust/Cargo.toml`):
>
> ```toml
> [workspace]
> resolver = "2"
> members = ["q1tsim", "qcgpu", "quantr", "quantrs"]
> ```

---

## 7.10 Estrategias de MCZ sin gate nativo

Esta seccion centraliza las **tres estrategias** disponibles para implementar
un MCZ (Multi-Controlled Z) cuando el framework no lo provee nativamente. La
eleccion depende del tamano (`n`) y del framework.

### Estrategia A: usar `CCZ` para n=3, `CZ` para n=2, `Z` para n=1

Soportada en q1tsim (`CCZ::new()`, `CZ::new()`) y como composicion en qcgpu y
quantrs2 (`H + CCX + H` o `H + mcx + H`). Es la mas eficiente para `n <= 3` y
no requiere ancillas.

```
n = 1 -> Z(0)
n = 2 -> CZ(0, 1)
n = 3 -> CCZ(0, 1, 2)  o  H(2); Toffoli(0,1,2); H(2)
```

### Estrategia B: ancilla qubits + Toffoli ladder (recomendada para n grande)

Para `n` controles, con `n - 2` qubits ancilla inicializados a `|0>`:

```
Toffoli(c0, c1, a0)
Toffoli(c2, a0, a1)
Toffoli(c3, a1, a2)
...
Toffoli(c_{n-1}, a_{n-3}, a_{n-2})
H(target); CX(a_{n-2}, target); H(target)   <- equivalente a CZ
[deshacer la escalera de Toffolis en orden inverso]
```

Coste: `n - 2` ancillas, `2*(n - 2) + 1` Toffolis.

**Implementacion practica**:
- En q1tsim: `Circuit::new(n_main + n_anc, n_main)` (los ancilla no se miden,
  por eso solo `n_main` bits clasicos).
- En quantr: igual, `Circuit::new(n_main + n_anc).unwrap()`.
- En qcgpu: `State::new(n_main + n_anc, seed)`; los ancilla se quedan en `|0>`
  al final del MCZ (siempre que la escalera se deshaga correctamente).
- En quantrs2: si `mcx` esta disponible, esta estrategia es innecesaria.

> **Cuidado**: si la escalera no se deshace, los ancillas quedan entrelazados
> y la siguiente medida los volcara, "ensuciando" la distribucion. La escalera
> de Toffolis SIEMPRE debe revertirse despues del CZ central.

### Estrategia C: Gate / matriz custom

Implementar un struct que codifique directamente la matriz `2^n x 2^n` que
solo cambia el signo de la entrada `|11...1>`. El simulador la aplica de un
golpe sin descomponerla.

- En q1tsim: implementar el trait `Gate` con metodos `description()`,
  `nr_affected_bits()` y `matrix()` que devuelve un `ndarray` con la matriz
  diagonal `diag(1, 1, ..., 1, -1)`.
- En quantr: `Gate::Custom(closure, mapping)` donde la closure recibe el slice
  de amplitudes y devuelve uno nuevo con la amplitud de `|11...1>` negada.
- En qcgpu: NO disponible directamente (no hay punto de extension para gates
  custom desde la API publica). Para qcgpu, usa estrategia B.
- En quantrs2: si `mcx` existe, no necesitas esto; sino, deberias volver a A
  o B.

**Coste**: la matriz pesa `O(2^n) x O(2^n)`, asi que para `n >= 8` la
construccion ya es costosa. Util para `n <= 6`. La ventaja es que el simulador
no descompone, asi que es la opcion mas rapida en numero de operaciones para
`n` pequeno.

### Tabla resumen

| n controles | q1tsim | quantr | qcgpu | quantrs2 |
|---|---|---|---|---|
| 1 | `Z` directo | `Z` directo | `z()` | `z()?` |
| 2 | `CZ` directo | `CZ` directo | `cz()` | `cz()?` |
| 3 | `CCZ` directo | `H+CCX+H` (`Toffoli`) | `H+toffoli+H` | `H+ccx+H` o `mcx` |
| >=4 | C (Gate custom) o B (ancilla) | C (`Gate::Custom`) o B | B (ancilla) | `mcx` o B |

---

## 7.11 Resumen de ficheros por framework (Rust)

| Framework | Ficheros a crear | Estrategia Shor | Estado del crate |
|---|---|---|---|
| **q1tsim** | `src/bin/grover.rs`, `src/bin/shor/{shor,qft,permutation}.rs` | Permutation network + QFT manual (CU1) | Estable (`0.5`) |
| **quantr** | `src/bin/grover.rs`, `src/bin/shor/{shor,qft,permutation}.rs` | Permutation network + QFT manual (`CRk`) | Estable (`0.6`), limite ~16 qubits |
| **qcgpu** | `src/bin/grover.rs`, `src/bin/shor/{shor,qft,permutation}.rs` | Permutation network + QFT manual (f32) | Estable (`0.1`), requiere OpenCL |
| **quantrs2** | `src/bin/grover.rs`, `src/bin/shor/shor.rs` (+ permutation si falta `mcx`) | QFT built-in + permutation custom | RC (`0.1.0-rc.1`), pin exacto, plan B: RustQIP |

---

# PARTE 8: NOTAS TRANSVERSALES PARA RUST

1. **Tests**: los valores de referencia (Grover 3 y 4 qubits, Shor `N=15`)
   estan en la **PARTE 5** de este documento. Los tests Rust deben ejecutarse
   con `cargo test --release` (release es importante para Shor; en debug los
   ~12 qubits pueden tardar minutos).

2. **Post-procesamiento clasico compartido**: la funcion
   `_get_order_from_dist` es identica en los cuatro frameworks. Conviene
   extraerla a un crate del workspace (e.g. `rust/common/`) y reutilizarla:
   ```toml
   [dependencies]
   common = { path = "../common" }
   ```
   El crate `common` exporta `get_order_from_dist`, `mod_pow`, `gcd`,
   `is_perfect_power` y la logica de fracciones continuas (basada en
   `num-rational`).

3. **Manejo de errores**: cada framework devuelve su propio tipo de error.
   Para mantener firmas limpias, define un `Box<dyn std::error::Error>` o un
   enum `ShorError` en `common` que implemente `From` para los errores de
   q1tsim, quantr y quantrs2. qcgpu no devuelve errores (panic-on-failure),
   asi que las funciones que lo usan no necesitan `Result`.

4. **Reproducibilidad**: los tests de Shor pasan `seed: u64` a `find_factor`
   para fijar el RNG del bucle clasico. Internamente:
   ```rust
   use rand::{SeedableRng, rngs::StdRng};
   let mut rng = StdRng::seed_from_u64(seed);
   let a = rng.gen_range(2..n);
   ```
   Esto garantiza que `cargo test` da resultados deterministas.

5. **Endianness — checklist**: antes de declarar correcta una implementacion,
   ejecuta este test de sanidad:
   - Grover con `n = 3, target = 1` (asimetrico).
   - Verifica que el bitstring mas frecuente, parseado a entero, da `1`.
   - Si da `4` (= `0b100`), tienes inversion de bits pendiente.

6. **Orden recomendado de implementacion**:
   1. **Grover en q1tsim** — el mas parecido a Qiskit estructuralmente.
   2. **Grover en qcgpu** — el modelo imperativo es directo, valida el
      pipeline OpenCL.
   3. **Grover en quantr** — el endianness MSB-first es la primera trampa
      seria.
   4. **Grover en quantrs2** — los const generics requieren refactor de
      firmas; dejalo para cuando tengas la logica clara en otros frameworks.
   5. **Shor en q1tsim** — se reutiliza la estructura de Grover y se anade
      QFT manual + permutation network.
   6. **Shor en qcgpu** — practicamente clon del de CUDA-Q (Python).
   7. **Shor en quantr** — `Gate::Custom` para la permutacion es la pieza
      nueva.
   8. **Shor en quantrs2** — depende de la salud del crate; activar plan B
      si rompe.
