# Qiskit — Framework de Simulación Cuántica de IBM

## 1. Introducción a Qiskit

### 1.1 Historia y origen

Qiskit nació en 2017 como un proyecto de código abierto de IBM Research. Su nombre es
un acrónimo de _Quantum Information Science Kit_. La motivación original era doble: por un
lado, ofrecer acceso programático a los procesadores cuánticos reales de IBM Quantum
Experience; por otro, disponer de un simulador clásico de referencia con el que desarrollar
y depurar algoritmos antes de ejecutarlos en hardware real.

Desde 2017 hasta 2023 el proyecto creció de forma modular alrededor de cuatro paquetes
independientes que se instalaban juntos como `qiskit-terra` (el núcleo de circuitos),
`qiskit-aer` (el simulador), `qiskit-ignis` (caracterización de ruido) y `qiskit-ibmq-provider`
(acceso a la nube). Esta arquitectura resultó confusa para los usuarios nuevos, que debían
entender qué paquete hacía cada cosa.

**Qiskit 1.0 (febrero 2024)** unificó todo en un único paquete `qiskit`. El código de `terra`
pasó a ser el propio `qiskit`. Los proveedores de hardware y los simuladores quedaron como
paquetes opcionales (`qiskit-aer`, `qiskit-ibm-runtime`). La API se estabilizó por primera
vez con garantías de compatibilidad semántica.

**Qiskit 2.0 (marzo 2025)** consolidó la nueva arquitectura de _primitivas_ como interfaz
canónica, eliminó definitivamente `execute()` (la función de ejecución heredada de terra),
y completó la migración a `SamplerV2` / `EstimatorV2` como la única forma de lanzar
circuitos. También introdujo mejoras significativas en el `PassManager` y en la
representación interna de los circuitos (migración completa al backend en Rust).

### 1.2 Arquitectura interna

```
┌─────────────────────────────────────────────────────┐
│                     qiskit (núcleo)                  │
│  QuantumCircuit · QuantumRegister · ClassicalRegister│
│  Gates (H, X, CX, MCX, CP …)                        │
│  PassManager / transpiler                            │
└──────────────────────┬──────────────────────────────┘
                       │ circuito ISA
          ┌────────────▼────────────┐
          │       qiskit-aer        │
          │  AerSimulator           │
          │  SamplerV2 / EstimatorV2│
          └─────────────────────────┘
```

**qiskit (núcleo)**: contiene la descripción de circuitos, puertas, registros, operaciones
clásicas condicionales y el sistema de transpilación. Es el único paquete obligatorio.

**qiskit-aer**: simulador de alto rendimiento escrito en C++/CUDA. Implementa varios
métodos de simulación (vector de estado, matriz densidad, tensor network, estabilizadores…).
Se instala por separado: `pip install qiskit-aer`.

**PassManager**: el pipeline de transformación de circuitos. Convierte un circuito lógico
(con puertas arbitrarias) en un circuito _ISA_ (Instruction Set Architecture), es decir,
expresado únicamente con las puertas que el backend objetivo soporta, con el mapa de
conectividad respetado y con el nivel de optimización elegido.

### 1.3 Posición en el ecosistema cuántico

Qiskit es, con diferencia, el framework cuántico con mayor adopción:

- Más de 600 000 usuarios registrados en IBM Quantum (datos 2024).
- Ecosistema de extensiones: `qiskit-nature` (química), `qiskit-machine-learning`,
  `qiskit-optimization`.
- Integración directa con hardware real a través de `qiskit-ibm-runtime`.
- Base académica: la mayoría de los artículos que publican código cuántico usan Qiskit o
  incluyen una traducción a Qiskit.

Los competidores directos son Cirq (Google), PennyLane (Xanadu) y Braket (Amazon). Qiskit
destaca por la madurez del transpilador y por ser el único con acceso gratuito a hardware
real de superconductores.

---

## 2. Modelo de Circuitos

### 2.1 `QuantumCircuit(n_qubits, n_classical_bits)`

La clase central de Qiskit es `QuantumCircuit`. Representa un programa cuántico como una
secuencia ordenada de operaciones sobre qubits. Internamente almacena:

- Una lista de `QuantumRegister` (los qubits).
- Una lista de `ClassicalRegister` (los bits clásicos para medir).
- Una lista de instrucciones `CircuitInstruction`, cada una compuesta por una puerta y los
  qubits/bits sobre los que actúa.

La forma más sencilla de crear un circuito es:

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(3, 3)   # 3 qubits, 3 bits clásicos
```

Esto crea internamente un `QuantumRegister` anónimo de 3 qubits y un `ClassicalRegister`
anónimo de 3 bits.

### 2.2 `QuantumRegister` y `ClassicalRegister`

Cuando se necesita nombrar los registros (para depuración, visualización o composición),
se crean explícitamente:

```python
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit

qr = QuantumRegister(4, name="q")        # registro de 4 qubits llamado "q"
cr = ClassicalRegister(4, name="result") # registro de 4 bits clásicos
qc = QuantumCircuit(qr, cr)
```

Los registros no son solo etiquetas: son objetos iterables cuyos elementos son referencias
a qubits/bits individuales. Cuando se pasa `qr[i]` a una puerta, Qiskit resuelve el índice
global del qubit en el circuito.

En el código de este proyecto, `grover.py` sigue este patrón explícito en todas las
funciones:

```python
# grover.py, función build_oracle
qr = QuantumRegister(n)
qc = QuantumCircuit(qr)
```

Usar registros explícitos facilita la composición posterior con `.compose()`, ya que permite
especificar sobre qué qubits del circuito destino se insertan los qubits del subcircuito.

### 2.3 Añadir puertas

Qiskit ofrece métodos de conveniencia en `QuantumCircuit` para todas las puertas estándar:

| Método             | Puerta          | Descripción                                     |
|--------------------|-----------------|--------------------------------------------------|
| `.h(qubit)`        | Hadamard        | Crea superposición uniforme: `|0⟩ → (|0⟩+|1⟩)/√2` |
| `.x(qubit)`        | Pauli-X (NOT)   | Invierte el qubit: `|0⟩ → |1⟩`                  |
| `.cx(ctrl, tgt)`   | CNOT            | NOT controlado                                   |
| `.cz(ctrl, tgt)`   | CZ              | Z controlado                                     |
| `.cp(θ, ctrl, tgt)`| Phase controlada| Aplica `e^{iθ}` al estado `|11⟩`                |
| `.mcx(ctrls, tgt)` | Toffoli/MCX     | NOT multi-controlado                             |
| `.p(θ, qubit)`     | Phase           | Aplica `e^{iθ}` al estado `|1⟩`                 |

Para puertas más complejas o con parámetros que no tienen método directo, se usa `.append()`:

```python
from qiskit.circuit.library import ZGate

# ZGate controlada por n-1 qubits (multi-controlled Z)
qc.append(ZGate().control(n - 1), qr[:])
```

`.control(k)` es un método de cualquier `Gate` de Qiskit que genera la versión controlada
por `k` qubits de esa puerta. El resultado es una `ControlledGate` que se puede pasar a
`.append()`.

### 2.4 Composición: `.compose()` y `.append()`

`.compose(other, qubits=...)` inserta el circuito `other` dentro del circuito actual, mapeando
los qubits de `other` a los qubits especificados en `qubits`. Con `inplace=True` modifica el
circuito en sitio; sin él, devuelve una copia.

```python
# grover.py, función grover_circuit
oracle   = build_oracle(n, target)
diffuser = build_diffuser(n)

for _ in range(num_iterations):
    qc.compose(oracle,   qubits=qr, inplace=True)
    qc.compose(diffuser, qubits=qr, inplace=True)
```

Esto es semánticamente distinto de concatenar listas: Qiskit inserta las instrucciones del
subcircuito en la representación interna del circuito padre, respetando el orden temporal.
Las instrucciones del oráculo preceden a las del difusor en cada iteración.

`.append(instruction, qargs, cargs)` inserta una instrucción individual (una puerta con sus
argumentos). Se usa cuando la puerta se construye en tiempo de ejecución (por ejemplo,
`ZGate().control(n-1)`).

### 2.5 Mediciones

Qiskit ofrece dos formas de medir:

**`.measure_all()`**: añade un `ClassicalRegister` nuevo (si no existe) y mide todos los
qubits en ese registro. Conveniente para circuitos sencillos.

**`.measure(qubits, clbits)`**: medición explícita sobre registros específicos. Es la forma
usada en este proyecto, porque permite nombrar el registro clásico y referenciarlo luego
en `get_counts()`:

```python
# grover.py, función grover_circuit
cr = ClassicalRegister(n, name="result")
qc = QuantumCircuit(qr, cr)
# ... puertas ...
qc.measure(qr, cr)
```

El nombre `"result"` es importante: al acceder a los conteos en el worker, se usa
`pub_result.data.result.get_counts()`, donde `result` coincide exactamente con el nombre
del `ClassicalRegister`.

---

## 3. Ejecución y Simulación

### 3.1 `AerSimulator` y sus modos

`AerSimulator` es el simulador de referencia de Qiskit. Soporta varios métodos de
simulación que se seleccionan con el parámetro `method`:

| Método              | Descripción                              | Límite práctico  |
|---------------------|------------------------------------------|------------------|
| `statevector`       | Almacena el vector completo de 2^n amplitudes | ~30 qubits    |
| `density_matrix`    | Matriz densidad 2^n × 2^n (con ruido)    | ~20 qubits       |
| `stabilizer`        | Eficiente para circuitos Clifford        | miles de qubits  |
| `matrix_product_state` | Tensor network para estados poco entrelazados | ~100 qubits  |
| `automatic`         | Qiskit elige el mejor método             | depende          |

Por defecto (`AerSimulator()` sin argumentos) usa el modo `automatic`, que selecciona
`statevector` para la mayoría de los circuitos generales. Para Grover y Shor esto implica
almacenar un vector complejo de `2^n` componentes en memoria RAM.

```python
# qiskit_worker.py
from qiskit_aer import AerSimulator
backend = AerSimulator()
```

### 3.2 `SamplerV2`: la primitiva de muestreo

Las _primitivas_ son la interfaz canónica de Qiskit 2.0 para ejecutar circuitos. Abstraen
la diferencia entre simuladores y hardware real: el mismo código funciona con `AerSimulator`
o con un procesador de IBM Quantum.

`SamplerV2` ejecuta circuitos y devuelve distribuciones de medición (conteos de shots).
Su interfaz es:

```python
sampler = SamplerV2()

# Crear un PUB (Primitive Unified Bloc): tupla (circuito_isa, [params], shots)
job    = sampler.run([qc_isa], shots=1024)
result = job.result()          # bloqueante
pub_result = result[0]         # resultado del primer PUB
counts = pub_result.data.meas.get_counts()
```

Puntos clave:

1. **`SamplerV2` acepta listas de PUBs**: se pueden enviar varios circuitos en una sola
   llamada y obtener los resultados indexando `result[i]`.
2. **`.data.<nombre_registro>.get_counts()`**: accede al registro clásico por nombre. Si el
   registro se llama `"result"`, se accede como `.data.result.get_counts()`. Si se llama
   `"output_bits"`, como `.data.output_bits.get_counts()`.
3. **La entrada debe ser un circuito ISA**: ver sección 3.3.

En el worker de Grover:

```python
# grover.py, función search
dist = sampler.run([qc_isa], shots=num_shots).result()[0].data.result.get_counts()
```

En el worker de Shor:

```python
# shor.py, función find_order
dist = sampler.run([qc_isa], shots=num_shots).result()[0].data.output_bits.get_counts()
```

La diferencia en el nombre del registro (`result` vs `output_bits`) refleja los distintos
`ClassicalRegister` creados en cada algoritmo.

### 3.3 `PassManager` y transpilación

Un circuito lógico de Qiskit puede contener puertas arbitrarias (Toffoli, multi-controlled Z,
QFT…) que no existen en el conjunto de instrucciones de ningún backend real. El
`PassManager` transforma ese circuito en uno _ISA_ (Instruction Set Architecture): un
circuito equivalente expresado únicamente con las puertas nativas del backend y respetando
su mapa de conectividad.

`generate_preset_pass_manager` crea un `PassManager` predefinido con un nivel de
optimización elegido:

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
```

El pipeline de pases de `optimization_level=1` incluye, entre otros:

1. **Unroll**: expande puertas compuestas en primitivas.
2. **Layout**: asigna qubits lógicos a qubits físicos del backend.
3. **Routing**: inserta puertas SWAP para respetar la conectividad.
4. **Optimization**: peephole optimization (cancela puertas redundantes adyacentes).
5. **Translation**: convierte las puertas al conjunto nativo (típicamente `{CX, U3}` o
   `{ECR, RZ, SX, X}` para hardware IBM).

### 3.4 Circuitos ISA

Un circuito ISA es el resultado de aplicar el `PassManager`. Se obtiene con:

```python
qc_isa = pm.run(qc)
```

Es obligatorio pasar el circuito ISA a `SamplerV2.run()`. Si se pasa el circuito original,
`SamplerV2` lanzará un error indicando que el circuito no es compatible con el backend.
Esta distinción fuerza al programador a ser explícito sobre cuándo ocurre la transpilación,
lo cual es importante para medir los tiempos con precisión: la transpilación puede consumir
tanto tiempo como la simulación en circuitos grandes.

### 3.5 Shots, conteos y probabilidades

`shots` es el número de veces que se ejecuta (o se muestrea) el circuito. Cada ejecución
colapsa el estado cuántico y produce una cadena de bits clásica. El conjunto de resultados
forma una distribución de frecuencias llamada `counts`:

```python
counts = {"0101": 512, "0110": 256, "0000": 128, ...}
```

La probabilidad estimada del estado `|s⟩` es `counts[s] / shots`. Para obtener el estado
más probable:

```python
found = int(max(dist, key=dist.get), 2)
```

`max(dist, key=dist.get)` devuelve la clave (cadena binaria) con el mayor valor.
`int(..., 2)` la convierte a entero interpretando la base 2.

Con pocos shots la distribución es ruidosa. Grover usa 1024 shots por defecto; Shor usa
solo 10 por defecto, porque el circuito es mucho más costoso y la señal (el orden) se
puede extraer de pocas muestras con fracciones continuas.

---

## 4. Grover en Qiskit

### 4.1 Estructura general del algoritmo

El algoritmo de Grover busca un elemento marcado en un espacio no estructurado de `2^n`
elementos. El circuito consta de tres partes:

1. **Preparación**: poner todos los qubits en superposición uniforme con H⊗n.
2. **Iteraciones de Grover**: aplicar repetidamente (oráculo + difusor).
3. **Medición**: colapsar el estado y leer el resultado.

El número óptimo de iteraciones es `⌊(π/4) · √(2^n)⌋`. Con menos iteraciones la
amplitud del estado marcado no llega a su máximo; con más, la supera y empieza a decrecer.

```python
# grover.py, función grover_circuit
if num_iterations is None:
    num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))
```

### 4.2 La función `build_oracle`

El oráculo de fase marca el estado objetivo `|target⟩` aplicándole un cambio de fase de
`-1` (equivalente a multiplicar la amplitud por `-1`). La implementación se basa en la
estrategia estándar de _phase kickback_ con una puerta Z multi-controlada:

```python
# grover.py
def build_oracle(n: int, target: int) -> QuantumCircuit:
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # Paso 1: flippear los qubits donde target tiene un 0
    # Así |target⟩ se mapea a |11...1⟩
    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    # Paso 2: MCZ — aplica -1 al estado |11...1⟩
    qc.append(ZGate().control(n - 1), qr[:])

    # Paso 3: deshacer los X del paso 1
    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    return qc
```

**Paso 1 — mapeo a `|11...1⟩`**: el bit `i` del entero `target` se extrae con
`(target >> i) & 1`. Si ese bit es `0`, se aplica una puerta X en `qr[i]`, convirtiendo
`|0⟩` en `|1⟩` para ese qubit. Tras este bloque, el único estado de la base computacional
que tiene todos los qubits en `|1⟩` es el estado `|target⟩` original.

**Paso 2 — Z multi-controlada**: `ZGate().control(n-1)` crea una puerta Z controlada por
`n-1` qubits. Al aplicarse sobre `qr[:]` (todos los `n` qubits), el qubit objetivo de la Z
es `qr[n-1]` y los controles son `qr[0]` … `qr[n-2]`. La puerta Z sobre `|1⟩` devuelve
`-|1⟩`. Por tanto, el estado `|11...1⟩` recibe un cambio de fase de `-1`; todos los demás
estados permanecen inalterados.

**Paso 3 — deshacer**: las puertas X son autoconjugadas (X² = I), así que aplicarlas de
nuevo deshace el mapeo del paso 1. El resultado neto es que únicamente `|target⟩` ha
recibido un cambio de fase de `-1`.

### 4.3 La función `build_diffuser`

El difusor implementa la reflexión sobre el estado de superposición uniforme
`|s⟩ = H⊗n|0...0⟩`. Matemáticamente es el operador `2|s⟩⟨s| - I`. Su implementación
por conjugación es:

```
H⊗n · (2|0...0⟩⟨0...0| - I) · H⊗n
```

Y `2|0...0⟩⟨0...0| - I` se implementa igual que el oráculo pero para el estado `|0...0⟩`:

```python
# grover.py
def build_diffuser(n: int) -> QuantumCircuit:
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # H en todos los qubits: transforma |s⟩ en |0...0⟩ (en promedio)
    for i in range(n):
        qc.h(qr[i])

    # Phase flip en |0...0⟩: X → MCZ → X
    for i in range(n):
        qc.x(qr[i])

    qc.append(ZGate().control(n - 1), qr[:])

    for i in range(n):
        qc.x(qr[i])

    # H en todos los qubits
    for i in range(n):
        qc.h(qr[i])

    return qc
```

La secuencia `X⊗n → MCZ → X⊗n` en el bloque central hace lo mismo que el oráculo pero
para `target=0`: convierte `|0...0⟩` en `|1...1⟩`, aplica la fase, y lo devuelve a
`|0...0⟩`. Los H de antes y después conjugan la operación para que actúe sobre `|s⟩` en
lugar de sobre `|0...0⟩`.

Nótese que el difusor **no tiene ancilla qubit**: a diferencia de otras implementaciones de
Grover que usan un qubit auxiliar inicializado en `|−⟩`, aquí se usa exclusivamente la
técnica de reflexión conjugada con Hadamard, que no requiere qubit adicional.

### 4.4 La función `grover_circuit`

```python
# grover.py
def grover_circuit(n: int, target: int,
                   num_iterations: int | None = None) -> QuantumCircuit:

    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="result")
    qc = QuantumCircuit(qr, cr)

    # Superposición uniforme inicial
    for i in range(n):
        qc.h(qr[i])

    # Construir oracle y difusor una sola vez
    oracle   = build_oracle(n, target)
    diffuser = build_diffuser(n)

    # Componer las iteraciones
    for _ in range(num_iterations):
        qc.compose(oracle,   qubits=qr, inplace=True)
        qc.compose(diffuser, qubits=qr, inplace=True)

    # Medir
    qc.measure(qr, cr)

    return qc
```

Aspectos importantes:

- El oráculo y el difusor se construyen **una sola vez** fuera del bucle y se reutilizan con
  `.compose()`. Esto es eficiente porque la construcción del circuito (objetos Python) es
  separada de la simulación.
- El `ClassicalRegister` se nombra `"result"`, lo que determina el nombre del campo en
  `pub_result.data` al acceder a los conteos.
- El circuito resultante tiene exactamente `n` qubits y `n` bits clásicos.

### 4.5 La función `search`

```python
# grover.py
def search(n, target, sampler, pass_manager,
           num_iterations=None, num_shots=1024):

    iters = (num_iterations if num_iterations is not None
             else math.floor(math.pi / 4 * math.sqrt(2**n)))

    qc     = grover_circuit(n, target, num_iterations=iters)
    qc_isa = pass_manager.run(qc)      # transpilación

    dist = (sampler
            .run([qc_isa], shots=num_shots)
            .result()[0]
            .data.result
            .get_counts())

    found = int(max(dist, key=dist.get), 2)
    return found, dist
```

La función separa claramente tres fases: construcción del circuito (`grover_circuit`),
transpilación (`pass_manager.run`) y ejecución (`sampler.run`). Esta separación es
fundamental para la medición de tiempos en el worker (ver sección 6).

---

## 5. Shor en Qiskit

### 5.1 Arquitectura del algoritmo

El algoritmo de Shor factoriza un entero `N` buscando el _orden_ de un elemento `A` en el
grupo multiplicativo `Z_N`: el menor `r` tal que `A^r ≡ 1 (mod N)`. Si `r` es par y
`A^(r/2) ≢ -1 (mod N)`, entonces `gcd(A^(r/2) ± 1, N)` da un factor no trivial de `N`.

El circuito cuántico implementa la _estimación de fase cuántica_ (QPE) de la unitaria de
multiplicación modular `U_A: |y⟩ → |Ay mod N⟩`.

### 5.2 Registros del circuito de order-finding

```python
# shor.py, función order_finding_circuit
n = math.ceil(math.log2(N))   # bits necesarios para representar N
m = 2 * n                     # bits de precisión de fase (por defecto)

control_register  = QuantumRegister(m)      # registro de fase
target_register   = QuantumRegister(n)      # |y⟩ — estado de la multiplicación
ancilla_register  = QuantumRegister(n + 2)  # qubits auxiliares para el sumador
output_register   = ClassicalRegister(m, name="output_bits")
```

El circuito usa un total de `m + n + (n+2) = 4n + 2` qubits (con `m = 2n`). Esto crece
linealmente con `log2(N)`, pero el número de puertas crece como `O(n^3)`, lo que hace que
los circuitos para `N` moderadamente grande (p.ej. `N = 15`) ya tengan cientos de puertas.

### 5.3 La QFT con `QFTGate`

La Transformada de Fourier Cuántica (QFT) se usa en dos lugares: implícitamente en la
construcción de la exponenciación modular y explícitamente como la QFT inversa al final
del circuito de order-finding.

Qiskit proporciona `QFTGate` en `qiskit.circuit.library`, que encapsula la QFT completa
como una única puerta. Se usa directamente en `shor.py`:

```python
# shor.py
from qiskit.circuit.library import QFTGate

# QFT inversa sobre el registro de control
qc.compose(QFTGate(m).inverse(), qubits=control_register, inplace=True)
```

`.inverse()` genera la puerta adjunta (la QFT† = QFT⁻¹), que es la que se necesita en QPE
para extraer la fase.

El proyecto también define `QFTFullGate` en `qft.py`, una subclase de `QFTGate` que
expone los parámetros internos de `synth_qft_full`:

```python
# qft.py
from qiskit.circuit.library import QFTGate
from qiskit.synthesis import synth_qft_full

class QFTFullGate(QFTGate):
    do_swaps: bool = True
    approximation_degree: int = 0
    insert_barriers: bool = False

    def __init__(self, num_qubits, do_swaps=True,
                 approximation_degree=0, insert_barriers=False):
        super().__init__(num_qubits=num_qubits)
        self.do_swaps = do_swaps
        self.approximation_degree = approximation_degree
        self.insert_barriers = insert_barriers

    def _define(self):
        self.definition = synth_qft_full(
            num_qubits=self.num_qubits,
            do_swaps=self.do_swaps,
            approximation_degree=self.approximation_degree,
            insert_barriers=self.insert_barriers,
        )
```

El método `_define` es el mecanismo estándar de Qiskit para proporcionar la descomposición
de una puerta personalizada. Qiskit lo llama cuando necesita expandir la puerta en
instrucciones primitivas (durante la transpilación o la visualización). Al sobreescribir
`_define`, `QFTFullGate` delega en `synth_qft_full`, que genera la secuencia canónica de
la QFT: H en el qubit más significativo, luego puertas de fase controlada `CP(2π/2^k)` para
cada par de qubits, y finalmente los SWAPs que invierten el orden de los qubits.

El parámetro `approximation_degree` permite usar la _QFT aproximada_: cuando `d > 0`,
se omiten las rotaciones con ángulo menor que `π/2^(n-d)`, reduciendo el número de
puertas a costa de una pequeña pérdida de precisión. Esto es relevante para hardware real
con ruido.

### 5.4 El circuito de order-finding completo

```python
# shor.py, función order_finding_circuit
def order_finding_circuit(A, N, precision=None):
    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    control_register  = QuantumRegister(m)
    target_register   = QuantumRegister(n)
    ancilla_register  = QuantumRegister(n + 2)
    output_register   = ClassicalRegister(m, name="output_bits")

    qc = AdderCircuit(control_register, target_register,
                      ancilla_register, output_register)

    # Paso 1: superposición en el registro de control
    for i in range(m):
        qc.h(control_register[i])

    # Paso 2: preparar |1⟩ en el registro objetivo
    qc.x(target_register[0])

    # Paso 3: exponenciación modular controlada U^(2^k)
    qc.exponentiate_modulo(A=A, x_reg=control_register,
                           y_reg=target_register,
                           ancilla_reg=ancilla_register, N=N)

    # Paso 4: QFT inversa sobre el registro de control
    qc.compose(QFTGate(m).inverse(), qubits=control_register, inplace=True)

    # Paso 5: medir el registro de control
    qc.measure(control_register, output_register)

    return qc
```

`AdderCircuit` es una subclase de `QuantumCircuit` definida en el módulo `adder.py` del
proyecto. Añade los métodos `exponentiate_modulo` y `c_multiply_modulo` que implementan,
respectivamente, la exponenciación modular completa y una única multiplicación modular
controlada.

La exponenciación modular calcula `A^x mod N` donde `x` es el valor codificado en el
registro de control en superposición: aplica la unitaria `U_A^(2^k)` controlada por el
qubit `control_register[k]` para cada `k` de 0 a `m-1`.

### 5.5 Variante de un solo qubit de control

`order_finding_circuit_one_control` implementa la misma QPE pero con un único qubit de
control y mediciones intermedias, técnica conocida como _semi-classical QFT_:

```python
# shor.py
c_bit = control_register[0]
for i in range(m):
    qc.h(c_bit)
    qc.c_multiply_modulo(
        control_reg=c_bit,
        A=pow(A, 2 ** (m - i - 1), N),
        x_reg=target_register,
        ...
        N=N,
    )
    # Corrección de fase basada en mediciones anteriores
    for j in range(i):
        with qc.if_test((output_register[j], 1)):
            qc.p(-math.pi / 2 ** (i - j), c_bit)
    qc.h(c_bit)
    qc.measure(c_bit, output_register[i])
    with qc.if_test((output_register[i], 1)):
        qc.x(c_bit)          # reset del qubit de control
```

Este enfoque reduce el número de qubits de `4n+2` a `2n+3`, pero requiere operaciones
clásicas condicionales (`if_test`) dentro del circuito, lo que sólo es soportado
nativamente en hardware real y en el simulador `AerSimulator` con la opción correcta.

### 5.6 Post-procesado: fracciones continuas para extraer `r`

La medición del circuito de order-finding produce una cadena de `m` bits que representa
una aproximación a `k/r` (para algún entero `k` coprimo con `r`), escalada por `2^m`.
Para extraer `r` se aplica el algoritmo de fracciones continuas:

```python
# shor.py, función _get_order_from_dist
def _get_order_from_dist(dist, A, N, precision):
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)

    for i in range(min(10, len(sorted_outputs))):
        if sorted_outputs[i] == "0" * precision:
            continue

        x = int(sorted_outputs[i], 2)

        # Fracción continua: aproximar x/2^m por p/q con q <= N-1
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator

        # Verificar que r es realmente el orden
        if pow(A, r, N) == 1:
            return r

    return 0
```

`Fraction(x / 2**m).limit_denominator(N-1)` usa el algoritmo de fracciones continuas de
Python para encontrar la fracción más simple `p/q` con `q ≤ N-1` que se aproxima a
`x/2^m`. El denominador `q` es un candidato para el orden `r`. La verificación
`pow(A, r, N) == 1` confirma que es correcto.

Se busca entre los 10 resultados más frecuentes para manejar ruido y los casos en que la
fracción continua da un divisor de `r` en lugar de `r` directamente.

### 5.7 La función `find_factor`

`find_factor` implementa el algoritmo de Shor completo, incluyendo las comprobaciones
clásicas que permiten evitar el circuito cuántico en casos triviales:

```python
# shor.py
def find_factor(N, sampler, pass_manager, num_tries=3,
                num_shots_per_trial=10, one_control_circuit=False, seed=None):

    # Caso trivial 1: N es par
    if N % 2 == 0:
        return 2

    # Caso trivial 2: N es potencia perfecta (N = d^k)
    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            return d

    # Bucle principal: elegir A aleatorio y buscar el orden
    i = 0
    while (not factor_found) and i < num_tries:
        a = random.randint(2, N - 1)
        d = math.gcd(a, N)

        if d > 1:
            # Suerte: a y N comparten factor directamente
            return d

        r, _ = find_order(a, N, sampler, pass_manager,
                          num_shots=num_shots_per_trial,
                          one_control_circuit=one_control_circuit)

        if r == 0:
            continue

        # Si r es par, extraer factor
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if d > 1 and d < N:
                factor_found = True
        i += 1
```

Las dos comprobaciones clásicas al inicio son fundamentales: para entradas típicas de
benchmarks (`N = 15, 21, 33…`) la mayor parte de los casos pueden resolverse
inmediatamente sin ejecutar ningún circuito cuántico, lo que tiene gran impacto en el
tiempo de benchmark.

---

## 6. El Worker de Qiskit

### 6.1 Responsabilidad del worker

El fichero `qiskit_worker.py` es el punto de entrada del proceso hijo que el sistema de
benchmarking lanza en un subproceso independiente. Su responsabilidad es:

1. Leer la configuración del benchmark desde `stdin` (JSON).
2. Inicializar el simulador y el transpilador (medir `startup_time`).
3. Ejecutar el benchmark (medir `build_time` y `simulation_time` por separado).
4. Escribir el resultado en `stdout` (JSON enriquecido).

Esta arquitectura de subproceso permite aislar el estado Python de cada framework: si un
worker falla o consume demasiada memoria, no afecta al proceso principal de benchmarking.

### 6.2 Inicialización: `_setup_grover`

```python
# qiskit_worker.py
def _setup_grover(config: BenchmarkConfig):
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from python.qiskit.grover import search, grover_circuit

    t0 = time.perf_counter()
    backend = AerSimulator()
    pm      = generate_preset_pass_manager(backend=backend, optimization_level=1)
    sampler = SamplerV2()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def search_call(n, target, num_shots):
        return search(n, target, sampler, pm, num_shots=num_shots)

    def build_call(n, target):
        return grover_circuit(n, target)

    return startup_ms, search_call, build_call
```

El tiempo de startup (`startup_ms`) mide cuánto tarda Qiskit en inicializar:
`AerSimulator()` carga las bibliotecas C++ de Aer (`.so`/`.dylib`), que no están en
memoria al inicio. `generate_preset_pass_manager` construye la cadena de pases. Este
tiempo puede ser de varios segundos en la primera ejecución y es independiente del
algoritmo que se vaya a ejecutar.

Los closures `search_call` y `build_call` capturan el `sampler` y el `pm` inicializados,
y exponen la misma firma esperada por `run_grover_worker` en `_base.py`, que llama a
`build_call` para medir `build_time` y a `search_call` para medir `simulation_time`.

### 6.3 Inicialización: `_setup_shor`

```python
# qiskit_worker.py
def _setup_shor(config: BenchmarkConfig):
    from python.qiskit.shor.shor import find_factor as _ff
    from qiskit_aer.primitives import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.perf_counter()
    sampler = SamplerV2()
    pm      = generate_preset_pass_manager(
                  optimization_level=1,
                  backend=sampler._backend)
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(N, sampler, pm, num_tries=3,
                   num_shots_per_trial=config.num_shots)

    return startup_ms, factor_call
```

La diferencia con `_setup_grover` es que aquí no se necesita `AerSimulator()` explícito:
`SamplerV2()` por defecto usa `AerSimulator` internamente, y se accede a su backend via
`sampler._backend` para construir el `PassManager`.

### 6.4 Flujo completo del `main`

```python
# qiskit_worker.py
def main():
    cfg    = read_config()           # lee JSON de stdin
    hw     = detect_hardware()       # CPU, RAM, plataforma
    config = BenchmarkConfig(...)

    # Verificar que qiskit y qiskit-aer están instalados
    import qiskit
    import qiskit_aer

    if algo == "grover":
        startup_ms, search_call, build_call = _setup_grover(config)
        result = run_grover_worker("qiskit", n, config, hw, contributor,
                                   startup_ms, search_call, build_call)
    elif algo == "shor":
        startup_ms, factor_call = _setup_shor(config)
        result = run_shor_worker("qiskit", n, config, hw, contributor,
                                  startup_ms, factor_call)

    write_result(result)             # escribe JSON a stdout
```

`run_grover_worker` y `run_shor_worker` son funciones de `_base.py` que implementan la
lógica común de benchmarking (repeticiones, medición de tiempos, agregación estadística).
El worker de Qiskit sólo necesita proveer las funciones de setup y las closures de
ejecución.

### 6.5 Separación `build_time` vs `simulation_time`

El sistema de benchmarking mide por separado:

- **`build_time`**: tiempo de llamar a `grover_circuit(n, target)` o `order_finding_circuit(A, N)`,
  es decir, construir el objeto `QuantumCircuit` en Python. Esto incluye la creación de
  puertas y registros pero no la transpilación.
- **`simulation_time`**: tiempo total de `search_call` o `factor_call`, que incluye la
  transpilación (`pm.run`) y la ejecución en Aer (`sampler.run`).
- **`startup_time`**: tiempo de inicialización del backend, medido una sola vez.

Esta separación permite comparar frameworks cuánticos no sólo en velocidad total sino en
qué parte del pipeline es el cuello de botella.

---

## 7. Decisiones de Diseño y Particularidades

### 7.1 Por qué `optimization_level=1` y no `3`

`generate_preset_pass_manager` acepta niveles de 0 a 3:

| Nivel | Estrategia                                        | Tiempo de transpilación |
|-------|---------------------------------------------------|-------------------------|
| 0     | Sin optimización: solo layout y routing triviales  | muy rápido              |
| 1     | Optimización ligera: cancelación de puertas adyacentes | moderado            |
| 2     | Optimización media: heurísticas más agresivas     | lento                   |
| 3     | Optimización máxima: búsqueda exhaustiva de layout | muy lento              |

Para benchmarks de rendimiento, `optimization_level=3` puede tardar minutos en circuitos
de Shor con `N ≥ 15`, lo que distorsiona la medición de `simulation_time`. El nivel `1`
ofrece un equilibrio razonable: aplica las optimizaciones básicas (que son rápidas y tienen
gran impacto) sin hacer que la transpilación domine el tiempo medido.

Adicionalmente, para el simulador `AerSimulator` las optimizaciones de nivel 3 aportan
poco beneficio práctico: el simulador no tiene restricciones de conectividad real (acepta
cualquier puerta multi-qubit), por lo que el routing es trivial y las optimizaciones
adicionales no reducen el tiempo de simulación de forma significativa.

### 7.2 Por qué `SamplerV2` en lugar de `execute()`

`execute()` fue la función principal de ejecución en Qiskit Terra (hasta Qiskit 0.x). En
Qiskit 1.0 quedó _deprecated_ y en Qiskit 2.0 fue eliminada completamente.

Las razones del cambio:

1. **Abstracción de backend**: `SamplerV2` tiene la misma interfaz tanto en `AerSimulator`
   como en hardware IBM Quantum, lo que permite portar código entre simulación y hardware
   real sin cambios.
2. **Batching de circuitos**: `sampler.run([c1, c2, c3])` ejecuta múltiples circuitos en
   una sola llamada, lo que puede ser más eficiente que llamadas secuenciales.
3. **Modelo de resultados tipado**: `pub_result.data.<registro>.get_counts()` es más
   explícito que el diccionario plano devuelto por `execute().result().get_counts()`.

### 7.3 Limitaciones de `AerSimulator` para circuitos grandes

`AerSimulator` con `method=statevector` (el modo por defecto) almacena el vector de estado
completo: `2^n` números complejos de 128 bits (16 bytes cada uno). El consumo de memoria
crece exponencialmente:

| Qubits | Memoria RAM       |
|--------|-------------------|
| 20     | 16 MB             |
| 25     | 512 MB            |
| 30     | 16 GB             |
| 32     | 64 GB             |
| 35     | 512 GB            |

Para el circuito de Shor con `N = 15`, el circuito completo usa alrededor de `4·4+2 = 18`
qubits. Para `N = 21`, `4·5+2 = 22` qubits, lo que ya requiere ~64 MB. Para `N = 35`, los
qubits necesarios son `4·6+2 = 26`, requiriendo ~1 GB. En la práctica, el verdadero
limitante no es la memoria del vector de estado sino el número de puertas: los circuitos de
Shor tienen profundidad de puerta del orden de `O(n^3)`, y cada puerta de dos qubits
sobre el vector de estado requiere `O(2^n)` operaciones. Esto hace que para `N ≥ 35` los
tiempos de simulación pasen de segundos a minutos u horas.

Para Grover, el circuito es más ligero: `n` qubits con `O(√(2^n))` iteraciones, cada una
con `O(n)` puertas. El límite práctico en un laptop moderno es alrededor de `n = 20`
(simulación en minutos) o `n = 25` con mucha paciencia.

### 7.4 El patrón `AdderCircuit`

`AdderCircuit` es una subclase de `QuantumCircuit` definida en el módulo `adder.py` del
proyecto. Este patrón es común en implementaciones complejas de Qiskit: se hereda de
`QuantumCircuit` para añadir métodos de dominio específico (`exponentiate_modulo`,
`c_multiply_modulo`) que encapsulan bloques de circuito complejos como operaciones de alto
nivel.

Esto tiene la ventaja de que el código de `shor.py` lee como un pseudocódigo del
algoritmo, ocultando los detalles de la implementación aritmética cuántica (sumadores de
Draper basados en QFT, multiplicadores modulares, etc.) en métodos de la clase.

### 7.5 Operaciones clásicas condicionales con `if_test`

```python
# shor.py, variante de un control
with qc.if_test((output_register[j], 1)):
    qc.p(-math.pi / 2 ** (i - j), c_bit)
```

`qc.if_test((bit, value))` es el mecanismo de Qiskit para operaciones cuánticas
condicionadas a bits clásicos medidos previamente. Esto implementa el _feed-forward_
clásico-cuántico necesario en la semi-classical QFT.

`AerSimulator` soporta estas operaciones de forma nativa cuando se usa con el modo de
simulación correcto. En hardware IBM Quantum real también están soportadas en los
procesadores con control en tiempo real (arquitectura _Heron_ y posteriores).

---

## Resumen

Qiskit modela los algoritmos cuánticos como grafos dirigidos acíclicos de operaciones sobre
registros nombrados. La separación entre la descripción del circuito (Python puro, cero
ejecución) y la transpilación (PassManager) y la ejecución (SamplerV2 + AerSimulator)
permite medir cada fase por separado y escribir código portable entre simulación y hardware
real.

El algoritmo de Grover se expresa de forma natural con `.compose()`: se construyen el
oráculo y el difusor como subcircuitos independientes y se componen repetidamente.
El oráculo usa la técnica de reflexión de fase (X → MCZ → X) sin ancilla. El difusor
usa la conjugación Hadamard (H → X → MCZ → X → H).

El algoritmo de Shor es arquitectónicamente más complejo: usa QPE con la unitaria de
multiplicación modular, implementada en `AdderCircuit` mediante aritmética cuántica basada
en QFT. La QFT inversa al final del circuito se toma directamente de `qiskit.circuit.library`
como `QFTGate(m).inverse()`. El post-procesado clásico con fracciones continuas extrae
el orden `r` de la distribución de medición.

El worker `qiskit_worker.py` encapsula todo este flujo en un subproceso aislado que
comunica mediante JSON en stdin/stdout, mide los tiempos de startup, construcción y
simulación por separado, y es intercambiable con los workers de otros frameworks
(Cirq, PennyLane, Braket) gracias a la interfaz común de `_base.py`.
