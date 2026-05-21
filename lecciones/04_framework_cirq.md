# Cirq — Framework de Simulación Cuántica de Google

> Lección 04 — Serie: Frameworks de Computación Cuántica  
> Nivel: universitario (grado + posgrado)  
> Prerequisitos: álgebra lineal, puertas cuánticas básicas, algoritmos de Grover y Shor

---

## 1. Introducción a Cirq

### 1.1 Historia y origen

Cirq es un framework de computación cuántica de código abierto desarrollado por **Google Quantum AI**, publicado en 2018. Su creación coincide con el periodo en que Google intensificó su programa de computación cuántica superconductora, que culminaría en 2019 con la demostración del procesador **Sycamore** (53 qubits, ~200 segundos para una tarea que un supercomputador clásico estimó en ~10 000 años en la época, aunque el estimado fue revisado posteriormente).

El nombre es un acrónimo de *Circuit*. El repositorio principal está en `quantumlib/Cirq` en GitHub y la biblioteca puede instalarse con `pip install cirq`.

### 1.2 Filosofía NISQ-first

El principio de diseño central de Cirq es **NISQ-first** (*Noisy Intermediate-Scale Quantum*). Mientras que frameworks como Qiskit nacen con la aspiración de abstraer el hardware y facilitar la portabilidad, Cirq asume desde el principio que el usuario quiere controlar explícitamente:

- en qué qubits físicos se colocan las puertas,
- en qué orden temporal se ejecutan (momentos),
- cómo se transpila (o no se transpila) el circuito,
- cómo interactúa la descripción del circuito con el hardware real de Google.

Esta filosofía se traduce en que Cirq **no aplica ninguna optimización ni transpilación implícita**. El circuito que construyes es exactamente el circuito que se ejecuta. Eso es deliberado: en hardware NISQ, cada puerta adicional introduce ruido, y el programador debe ser consciente del coste real de cada operación.

### 1.3 Google Quantum AI y Sycamore

El procesador **Sycamore** es la plataforma de hardware de referencia para Cirq. Sycamore usa qubits superconductores de tipo *transmon* dispuestos en una red 2D. Las operaciones nativas del hardware no son Hadamard o CNOT, sino puertas como `√X`, `√Y` y la puerta de dos qubits `fsim(θ, φ)` (una combinación de intercambio parcial y fase condicional).

Cirq expone estas puertas directamente (`cirq.FSimGate`, `cirq.PhasedXZGate`, etc.) y permite describir circuitos en términos del conjunto de puertas nativas del dispositivo objetivo. Esta es otra diferencia fundamental con Qiskit: en Qiskit el programador trabaja con un conjunto de puertas lógicas abstractas y el transpilador las convierte; en Cirq el programador puede (y en producción, debe) trabajar directamente con el conjunto nativo.

### 1.4 Diferencia filosófica con Qiskit

| Dimensión | Qiskit | Cirq |
|---|---|---|
| Abstracción por defecto | Alta (puertas lógicas universales) | Baja (puertas cercanas al hardware) |
| Transpilación | Automática antes de ejecutar | Manual / ausente en simulación |
| Ordenamiento temporal | Implícito en el DAG | Explícito con `Moment` |
| Registro de qubits | `QuantumRegister` nombrado | `LineQubit` o `NamedQubit` |
| Lectura de resultados | `Counts` (dict string→int) | `Result` con `histogram()` |
| Hardware objetivo | IBM (backends variados) | Google Sycamore |
| Curva de aprendizaje | Más suave para principiantes | Más escarpada, más potente |

La filosofía de Cirq se resume en: *"muéstrame exactamente lo que ocurre"*. Esta transparencia es una ventaja para investigación y una fuente de verbosidad en código de alto nivel.

---

## 2. Modelo de Circuitos en Cirq

### 2.1 Qubits: LineQubit y NamedQubit

En Cirq los qubits son **objetos Python de primera clase**, no índices enteros de un registro. Hay varios tipos:

```python
import cirq

# LineQubit: qubits numerados sobre una línea. Útil para simulación.
q0, q1, q2 = cirq.LineQubit.range(3)   # crea q[0], q[1], q[2]

# NamedQubit: qubits identificados por nombre. Útil para legibilidad.
alice = cirq.NamedQubit("alice")
bob   = cirq.NamedQubit("bob")

# GridQubit: para hardware 2D como Sycamore.
g00 = cirq.GridQubit(0, 0)
g01 = cirq.GridQubit(0, 1)
```

`LineQubit.range(n)` devuelve una lista `[LineQubit(0), LineQubit(1), ..., LineQubit(n-1)]`. Es la forma más habitual en código de simulación porque el índice tiene una semántica directa (qubit 0, qubit 1, etc.).

**Convención de orden (big-endian):** En Cirq, `LineQubit(0)` es el qubit de **mayor peso** (MSB) cuando se mide. Esto es al revés de Qiskit, donde `qubit[0]` es el LSB. Esta diferencia afecta a cómo se mapea un entero binario a los qubits y es la fuente de bugs más frecuente al portar código entre frameworks.

### 2.2 Operaciones y puertas

Una **puerta** en Cirq es un objeto que describe una transformación unitaria (p.ej. `cirq.H`, `cirq.X`, `cirq.CNOT`). Una **operación** es una puerta aplicada a uno o varios qubits concretos:

```python
# Puerta (gate): objeto abstracto
gate = cirq.H

# Operación (operation): puerta aplicada a un qubit específico
op = cirq.H(q0)        # equivalente a cirq.H.on(q0)
op2 = cirq.CNOT(q0, q1)

# Aplicar una puerta a múltiples qubits a la vez
ops_list = cirq.H.on_each(q0, q1, q2)  # devuelve una lista de operaciones
```

Puertas más usadas:

| Puerta | Descripción | Sintaxis Cirq |
|---|---|---|
| Hadamard | Superposición | `cirq.H` |
| Pauli-X | NOT cuántico | `cirq.X` |
| Pauli-Y, Z | Rotaciones | `cirq.Y`, `cirq.Z` |
| CNOT | NOT controlado | `cirq.CNOT` o `cirq.CX` |
| Toffoli | CNOT con 2 controles | `cirq.CCX` o `cirq.CCNOT` |
| Swap | Intercambio | `cirq.SWAP` |
| Medición | Lectura clásica | `cirq.measure` |
| ZPowGate | Rotación Z parametrizada | `cirq.ZPowGate(exponent=k)` |
| CZPowGate | CZ parametrizada | `cirq.CZPowGate(exponent=k)` |

### 2.3 cirq.Moment: capas de paralelismo explícito

El concepto más importante y distintivo de Cirq frente a otros frameworks es el **Moment**. Un `Moment` representa un instante de tiempo en el circuito: todas las operaciones dentro de un mismo `Moment` se ejecutan en paralelo (son simultáneas). No puede haber dos operaciones en el mismo `Moment` que actúen sobre el mismo qubit.

```python
# Construir un Moment manualmente
momento = cirq.Moment([cirq.H(q0), cirq.H(q1), cirq.H(q2)])

# El circuito es una secuencia de Moments
circuit = cirq.Circuit([
    cirq.Moment([cirq.H(q0), cirq.H(q1)]),
    cirq.Moment([cirq.CNOT(q0, q1)]),
    cirq.Moment([cirq.measure(q0, q1, key="resultado")]),
])
```

Cuando se usa `circuit.append()`, Cirq aplica una estrategia de inserción automática llamada `InsertStrategy`. La estrategia por defecto es `EARLIEST`, que coloca cada operación en el primer Moment en que cabe (sin solapar qubits ocupados). Otras estrategias:

- `NEW`: cada operación va en un nuevo Moment.
- `NEW_THEN_INLINE`: el primero en un Moment nuevo, los siguientes en el más temprano posible.

```python
# Forma más habitual: append con estrategia EARLIEST (por defecto)
circuit = cirq.Circuit()
circuit.append(cirq.H.on_each(q0, q1, q2))   # un Moment con 3 H en paralelo
circuit.append(cirq.CNOT(q0, q1))             # un nuevo Moment
circuit.append(cirq.measure(q0, q1, key="m"))
```

### 2.4 Construcción de circuitos: tres estilos

**Estilo 1: operador `+=`**

```python
subcircuito = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])
circuit += subcircuito   # concatena los Moments
```

Este es el estilo que usa `grover.py`: el oráculo y el difusor son circuitos independientes que se concatenan con `+=` en cada iteración.

**Estilo 2: lista de operaciones en el constructor**

```python
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="m"),
)
```

**Estilo 3: `append` repetido**

```python
circuit = cirq.Circuit()
circuit.append(cirq.H(q0))
circuit.append(cirq.CNOT(q0, q1))
```

### 2.5 Puertas parametrizadas

Cirq tiene un sistema de puertas parametrizadas basado en exponentes de la clase `EigenGate`. Por ejemplo, la familia de puertas Z:

```python
# ZPowGate(exponent=t) implementa Z^t = diag(1, e^{iπt})
z_half = cirq.ZPowGate(exponent=0.5)   # raíz cuadrada de Z (S gate)
z_qtr  = cirq.ZPowGate(exponent=0.25)  # T gate

# CZPowGate: versión controlada
czpow = cirq.CZPowGate(exponent=1/4)
```

Esta familia es crucial para la **QFT** (Transformada de Fourier Cuántica), donde se necesitan rotaciones de fase de la forma `e^{2πi/2^k}`.

---

## 3. Ejecución y Simulación

### 3.1 El simulador statevector

Cirq proporciona `cirq.Simulator`, que es un simulador de **vector de estado** (*statevector simulator*). Mantiene en memoria el vector complejo de amplitudes de todos los `2^n` estados de la base computacional. Es exacto (sin ruido) salvo errores de punto flotante de 64 bits.

```python
import cirq

simulator = cirq.Simulator()
```

La creación del simulador es muy ligera (menos de 1 ms). No carga el circuito; la carga ocurre al ejecutar.

### 3.2 Modos de ejecución

Cirq tiene dos modos principales:

**`simulator.run(circuit, repetitions=N)`**: ejecuta el circuito `N` veces con mediciones y devuelve un objeto `cirq.Result`. Este es el modo de benchmarking (equivale a `backend.run()` en Qiskit).

**`simulator.simulate(circuit)`**: ejecuta una sola vez sin mediciones y devuelve el vector de estado final. Útil para depuración y verificación:

```python
result_statevec = simulator.simulate(circuit)
print(result_statevec.final_state_vector)  # array numpy de complejos
```

### 3.3 Lectura de resultados

```python
result = simulator.run(circuit, repetitions=1024)

# Acceder a mediciones brutas: array numpy shape (repetitions, n_qubits)
raw = result.measurements["resultado"]   # "resultado" es el key de cirq.measure

# Histograma: {valor_entero: conteo}
histogram = result.histogram(key="resultado")
# Ejemplo: {3: 512, 1: 256, 2: 200, 0: 56}

# El valor entero se construye big-endian: bit[0]*2^(n-1) + ... + bit[n-1]*2^0
```

La diferencia clave con Qiskit: Qiskit devuelve un dict `{"bitstring": count}` donde las claves son strings como `"101"`. Cirq devuelve un dict `{int: count}` por defecto en `histogram()`. Para convertir al formato de Qiskit:

```python
n = 3
dist = {}
for value, count in histogram.items():
    bitstring = format(value, f"0{n}b")
    dist[bitstring] = count
# dist = {"101": 512, "001": 256, ...}
```

Esta conversión aparece explícitamente en `grover.py` y `shor.py`.

### 3.4 Comparación de modelos de resultado

| Aspecto | Qiskit | Cirq |
|---|---|---|
| Tipo devuelto | `Counts` (dict-like) | `cirq.Result` |
| Claves del histograma | strings (`"101"`) | enteros (`5`) |
| Orden de bits | LSB primero | MSB primero |
| Acceso a shots brutos | `.get_counts()` | `.measurements[key]` |
| Statevector | `statevector_simulator` separado | mismo `Simulator` con `.simulate()` |

---

## 4. Grover en Cirq

### 4.1 Estructura general del algoritmo

El algoritmo de Grover en `n` qubits busca un elemento marcado `|target⟩` en un espacio de `2^n` estados. Los pasos son:

1. Preparar superposición uniforme: `H⊗n |0⟩^n`
2. Aplicar `k = ⌊(π/4)√(2^n)⌋` iteraciones de:
   a. Oráculo de fase (marca `|target⟩`)
   b. Difusor (inversión sobre la media)
3. Medir todos los qubits

### 4.2 El oráculo: construcción en Cirq

El oráculo de fase aplica un signo negativo al estado `|target⟩`:

```
U_f |x⟩ = -|x⟩  si x == target
U_f |x⟩ =  |x⟩  en caso contrario
```

La construcción estándar en circuitos es:

1. Aplicar `X` a los qubits donde el bit correspondiente del target es `0` (para convertir `|target⟩` en `|11...1⟩`).
2. Aplicar una puerta Z multi-controlada (MCZ) sobre todos los qubits (que actúa solo cuando todos son `|1⟩`).
3. Revertir las `X` del paso 1.

Código real de `grover.py`, función `build_oracle`:

```python
def build_oracle(n: int, target: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # Paso 1: X en qubits donde el bit del target es 0
    # Convención big-endian: bit i del target → qubit (n-1-i)
    for i in range(n):
        if not (target >> i) & 1:
            circuit.append(cirq.X(qubits[n - 1 - i]))

    # Paso 2: Z multi-controlada (n-1 controles, 1 objetivo)
    mcz = cirq.Z.controlled(num_controls=n - 1)
    circuit.append(mcz.on(*qubits))

    # Paso 3: Revertir X del paso 1
    for i in range(n):
        if not (target >> i) & 1:
            circuit.append(cirq.X(qubits[n - 1 - i]))

    return circuit
```

**Análisis línea a línea:**

- `cirq.LineQubit.range(n)` devuelve `[q[0], q[1], ..., q[n-1]]`.

- El bucle `for i in range(n)` itera sobre las posiciones de bit. El bit `i` del target corresponde al qubit `q[n-1-i]` por la convención big-endian. Si ese bit es 0, se aplica `X` para voltearlo a 1.

  Ejemplo con `n=3, target=5 (=101b)`:
  - bit 0 = 1 → qubit q[2]: no se aplica X
  - bit 1 = 0 → qubit q[1]: se aplica X
  - bit 2 = 1 → qubit q[0]: no se aplica X

  Tras las X, el estado `|101⟩` se ha convertido en `|111⟩`.

- `cirq.Z.controlled(num_controls=n-1)` construye una puerta Z con `n-1` qubits de control. Cuando se llama `.on(*qubits)`, los primeros `n-1` qubits son controles y el último es el objetivo. Esta operación aplica Z al último qubit solo cuando todos los controles son `|1⟩`, lo que produce el flip de fase sobre `|11...1⟩`.

- El segundo bucle revierte las X, restaurando el estado original con el signo negativo añadido.

**Diferencia con Qiskit:** En Qiskit se usa típicamente un qubit ancilla inicializado en `|−⟩ = (|0⟩ - |1⟩)/√2` para convertir el oráculo de inversión de bits en oráculo de fase (trick del ancilla). En Cirq se usa directamente la Z multi-controlada, que actúa en el espacio de fase sin ancilla explícito. Esto simplifica el circuito a costa de requerir puertas multi-controladas más complejas.

### 4.3 El difusor: inversión sobre la media

El difusor implementa el operador `2|s⟩⟨s| − I` donde `|s⟩` es la superposición uniforme. En circuito:

1. `H⊗n`: lleva la superposición uniforme al estado `|0⟩^n`.
2. `X⊗n`: invierte todos los bits.
3. MCZ: flip de fase sobre `|11...1⟩` (que en este contexto corresponde a `|00...0⟩` tras la H y X).
4. `X⊗n`: revierte el paso 2.
5. `H⊗n`: regresa al espacio de superposición.

Código real de `grover.py`, función `build_diffuser`:

```python
def build_diffuser(n: int) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # H en todos los qubits
    circuit.append(cirq.H.on_each(*qubits))

    # Flip de fase en |00...0>: X → MCZ → X
    circuit.append(cirq.X.on_each(*qubits))

    mcz = cirq.Z.controlled(num_controls=n - 1)
    circuit.append(mcz.on(*qubits))

    circuit.append(cirq.X.on_each(*qubits))

    # H en todos los qubits
    circuit.append(cirq.H.on_each(*qubits))

    return circuit
```

**Análisis:**

- `cirq.H.on_each(*qubits)` y `cirq.X.on_each(*qubits)` devuelven generadores de operaciones. Al pasarlos a `circuit.append()`, Cirq los coloca en el primer `Moment` disponible, maximizando el paralelismo. Las `n` puertas H del primer `circuit.append` quedan todas en el mismo `Moment`.

- La estructura H–X–MCZ–X–H es idéntica al difusor de Qiskit en lógica matemática. La diferencia sintáctica es que en Qiskit se usaría `qc.h(qubits)`, `qc.x(qubits)`, y la puerta multi-controlada se construiría con `qc.h(ancilla); qc.mcx(controls, target); qc.h(ancilla)`.

### 4.4 El circuito completo de Grover

```python
def grover_circuit(
    n: int, target: int, num_iterations: int | None = None
) -> cirq.Circuit:
    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()

    # Superposición uniforme inicial
    circuit.append(cirq.H.on_each(*qubits))

    # Iteraciones de Grover: oráculo + difusor
    oracle   = build_oracle(n, target)
    diffuser = build_diffuser(n)
    for _ in range(num_iterations):
        circuit += oracle
        circuit += diffuser

    # Medición
    circuit.append(cirq.measure(*qubits, key="result"))

    return circuit
```

**Puntos clave:**

- `num_iterations = ⌊(π/4)√(2^n)⌋` es el número óptimo de iteraciones según Grover (1996). Para `n=3` esto da 2 iteraciones; para `n=4`, 3 iteraciones.

- `circuit += oracle` usa el operador `+=` de `cirq.Circuit`, que concatena los `Moment`s del subcircuito al final. No hay fusión: los momentos del oráculo se añaden literalmente a continuación de los momentos existentes.

- `cirq.measure(*qubits, key="result")`: el argumento `key` es el nombre de la medición. Es importante nombrarlo para acceder a él después con `result.histogram(key="result")`. En Qiskit la medición no tiene key explícito por defecto.

### 4.5 La función search

```python
def search(
    n: int, target: int, simulator,
    pass_manager=None,
    num_iterations: int | None = None,
    num_shots: int = 1024,
) -> tuple[int, dict[str, int]]:
    iters = (
        num_iterations if num_iterations is not None
        else math.floor(math.pi / 4 * math.sqrt(2**n))
    )

    qc = grover_circuit(n, target, num_iterations=iters)

    if pass_manager is not None:
        qc = pass_manager(qc)

    result = simulator.run(qc, repetitions=num_shots)
    histogram = result.histogram(key="result")

    # Conversión a formato string (compatibilidad con Qiskit)
    dist = {}
    for value, count in histogram.items():
        bitstring = format(value, f"0{n}b")
        dist[bitstring] = count

    found = max(histogram, key=histogram.get)
    return found, dist
```

El parámetro `pass_manager` existe por simetría de interfaz con la implementación de Qiskit. En Cirq no hay transpilador automático, así que en la práctica siempre es `None`.

### 4.6 Comparación directa Grover: Cirq vs Qiskit

| Aspecto | Cirq | Qiskit |
|---|---|---|
| Oráculo | MCZ directo sin ancilla | Ancilla `\|−⟩` + MCX |
| Puerta multi-controlada | `cirq.Z.controlled(n-1)` | `qc.mcx(controls, target)` |
| Difusor | H–X–MCZ–X–H | Misma estructura lógica |
| Orden de bits | Big-endian: ajuste `n-1-i` | Little-endian: ajuste directo |
| Medición | `cirq.measure(*qubits, key="result")` | `qc.measure(qreg, creg)` |
| Resultados | `histogram()` → `{int: count}` | `get_counts()` → `{str: count}` |
| Transpilación | Ninguna antes de simular | Implícita (o explícita con PassManager) |

---

## 5. Shor en Cirq

El algoritmo de Shor tiene dos partes: una parte cuántica (encontrar el orden `r` de `A mod N`) y una parte clásica (usar `r` para calcular un factor). La parte cuántica usa **Estimación de Fase Cuántica (QPE)** con la operación de **exponenciación modular**.

### 5.1 La Transformada de Fourier Cuántica en Cirq

La QFT sobre `m` qubits implementa:

```
QFT |j⟩ = (1/√(2^m)) Σ_{k=0}^{2^m-1} e^{2πijk/2^m} |k⟩
```

Cirq proporciona la función `cirq.qft(*qubits, inverse=False)` que devuelve directamente las operaciones de QFT (no un circuito separado, sino una operación compuesta que el simulador entiende nativamente):

```python
# QFT directa sobre m qubits
qft_ops = cirq.qft(*exponent_qubits)

# QFT inversa (usada en QPE)
iqft_ops = cirq.qft(*exponent_qubits, inverse=True)
```

Internamente, `cirq.qft` descompone la transformada en puertas Hadamard y `CZPowGate` con exponentes fraccionarios. La implementación manual equivalente en `n` qubits sería:

```python
def manual_qft(qubits):
    ops = []
    n = len(qubits)
    for i in range(n):
        ops.append(cirq.H(qubits[i]))
        for k in range(2, n - i + 1):
            ops.append(
                cirq.CZPowGate(exponent=2 / 2**k).on(qubits[i], qubits[i + k - 1])
            )
    return ops
```

Esto construye la red de Hadamards y rotaciones de fase controladas `R_k = diag(1, 1, 1, e^{2πi/2^k})` de la QFT estándar.

### 5.2 La exponenciación modular: ModularExp

La exponenciación modular es el componente más complejo de Shor. La operación cuántica que necesitamos es:

```
U_f |y⟩|x⟩ = |y⟩|y · A^x mod N⟩
```

donde `|x⟩` es el registro exponente (en superposición) e `|y⟩` es el registro objetivo (inicializado en `|1⟩`).

Cirq ofrece la clase abstracta `cirq.ArithmeticGate` que permite definir puertas aritméticas especificando únicamente su acción clásica mediante el método `apply`. El simulador entonces calcula la acción cuántica automáticamente aplicando la función clásica a la base computacional.

Código completo de `modular_exp.py`:

```python
class ModularExp(cirq.ArithmeticGate):
    """
    Puerta de exponenciación modular:
    |x>|y> -> |x>|y * base^x mod modulus>
    """

    def __init__(
        self,
        target_size: int,
        exponent_size: int,
        base: int,
        modulus: int,
    ) -> None:
        self.target_size  = target_size
        self.exponent_size = exponent_size
        self.base    = base
        self.modulus = modulus

    def registers(self) -> list[int | list[int]]:
        # Define los registros: lista de qubits para cada argumento de apply()
        # [2]*n significa n qubits binarios (cada uno en base 2)
        return [[2] * self.target_size, [2] * self.exponent_size]

    def with_registers(self, *new_registers) -> "ModularExp":
        # Requerido por ArithmeticGate: construye una copia con nuevos registros
        target_size = (
            1 if isinstance(new_registers[0], int)
            else len(new_registers[0])
        )
        exponent_size = (
            1 if isinstance(new_registers[1], int)
            else len(new_registers[1])
        )
        return ModularExp(
            target_size=target_size,
            exponent_size=exponent_size,
            base=self.base,
            modulus=self.modulus,
        )

    def apply(self, target_value: int, exponent_value: int) -> tuple[int, int]:
        # La acción clásica: calcula |y * A^x mod N> para estado base |y>|x>
        if target_value < self.modulus:
            return (
                (target_value * pow(self.base, exponent_value, self.modulus))
                % self.modulus,
                exponent_value,
            )
        # Si el valor está fuera del rango del módulo, no se modifica
        return target_value, exponent_value
```

**Explicación de `ArithmeticGate`:**

- `registers()` define cuántos qubits tiene cada argumento de `apply`. `[[2]*n, [2]*m]` significa que `apply` recibirá dos enteros: el primero codificado en `n` qubits y el segundo en `m` qubits.

- `apply(target_value, exponent_value)` es la función clásica. El simulador de Cirq la evalúa para cada estado de la base computacional y construye la matriz unitaria completa. Así se evita implementar manualmente la aritmética cuántica (que involucraría cientos de puertas CNOT y Toffoli).

- El método `pow(self.base, exponent_value, self.modulus)` usa la exponenciación modular eficiente de Python (algoritmo de cuadrado y multiplicación), que es `O(log exponent)`.

- La condición `if target_value < self.modulus` es necesaria porque el registro de `n` qubits puede representar valores de 0 a `2^n − 1`, pero solo los valores de 0 a `N−1` son válidos en `Z_N`. Para valores fuera del rango, la puerta actúa como identidad.

**Eficiencia:** Esta aproximación es válida para simulación clásica, donde la computación cuántica se simula de forma exacta. En hardware real, la exponenciación modular requeriría una red de puertas cuánticas primitivas explícita, lo que involucra aritméticas cuánticas de suma, multiplicación y reducción modular.

### 5.3 El circuito de búsqueda de orden

```python
def order_finding_circuit(A: int, N: int, precision: int | None = None) -> cirq.Circuit:
    n = math.ceil(math.log2(N))      # qubits para representar N
    m = precision if precision is not None else 2 * n  # qubits de precisión

    # Dos registros separados sobre LineQubit continuo
    exponent_qubits = cirq.LineQubit.range(m)
    target_qubits   = cirq.LineQubit.range(m, m + n)

    circuit = cirq.Circuit()

    # 1. Superposición en el registro exponente
    circuit.append(cirq.H.on_each(*exponent_qubits))

    # 2. Inicializar registro objetivo en |1>
    circuit.append(cirq.X(target_qubits[0]))

    # 3. Exponenciación modular controlada
    mod_exp = ModularExp(
        target_size=n,
        exponent_size=m,
        base=A,
        modulus=N,
    )
    circuit.append(mod_exp.on(*target_qubits, *exponent_qubits))

    # 4. QFT inversa sobre el registro exponente
    circuit.append(cirq.qft(*exponent_qubits, inverse=True))

    # 5. Medir el registro exponente
    circuit.append(cirq.measure(*exponent_qubits, key="result"))

    return circuit
```

**Análisis línea a línea:**

- `n = ceil(log2(N))`: el número mínimo de qubits para representar valores hasta `N-1`. Para `N=15`, `n=4`.

- `m = 2*n`: la precisión estándar para QPE. Con `2n` qubits de fase se puede recuperar el orden con alta probabilidad cuando `r ≤ N`.

- `exponent_qubits = LineQubit.range(m)` y `target_qubits = LineQubit.range(m, m+n)`: los dos registros son segmentos contiguos de la línea de qubits. Esto es Cirq idiomático: no hay registros con nombres, solo rangos de `LineQubit`.

- `cirq.X(target_qubits[0])`: inicializa el registro objetivo en `|1⟩`. En la representación big-endian de Cirq, `target_qubits[0]` es el qubit de mayor peso. El estado inicial del registro objetivo es `|0...01⟩ = |1⟩` en valor entero.

- `mod_exp.on(*target_qubits, *exponent_qubits)`: aplica la puerta `ModularExp` con el registro objetivo primero y el registro exponente después, en correspondencia con la firma de `apply(target_value, exponent_value)`.

- `cirq.qft(*exponent_qubits, inverse=True)`: la QFT inversa extrae la fase acumulada por la exponenciación modular. Es el corazón del algoritmo QPE.

### 5.4 Postprocesado clásico: fracciones continuas

```python
def _get_order_from_dist(dist: dict[int, int], A: int, N: int, precision: int) -> int:
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    for i in range(min(10, len(sorted_outputs))):
        x = sorted_outputs[i]
        if x == 0:
            continue
        # El valor medido x ≈ k·2^m/r para algún entero k
        # La fracción x/2^m ≈ k/r, y el denominador de su forma reducida es r
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            return r
    return 0
```

El algoritmo de fracciones continuas convierte el valor medido `x` (un entero de `m` bits) en el orden `r`:

1. `x / 2^m` es una aproximación de `k/r` para algún entero `k`.
2. `Fraction(...).limit_denominator(N-1)` encuentra la fracción racional más cercana con denominador ≤ `N-1`.
3. Si el denominador `r` de esa fracción satisface `A^r ≡ 1 (mod N)`, hemos encontrado el orden.

Se prueban los 10 valores más frecuentes del histograma para aumentar la robustez ante ruido.

### 5.5 El algoritmo completo de Shor

```python
def find_factor(
    N: int, simulator, pass_manager=None,
    num_tries: int = 3, num_shots_per_trial: int = 10,
    seed: int | None = None,
) -> int:
    # Caso trivial: N par
    if N % 2 == 0:
        return 2

    # Verificar si N es una potencia perfecta
    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            return d

    # Bucle principal: elegir base A aleatoria y buscar el orden
    if seed is not None:
        random.seed(seed)

    for i in range(num_tries):
        a = random.randint(2, N - 1)
        d = math.gcd(a, N)
        if d > 1:
            return d          # Factor encontrado por suerte

        r, _ = find_order(a, N, simulator, pass_manager,
                          num_shots=num_shots_per_trial)
        if r == 0:
            continue
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if 1 < d < N:
                return d
    return 1
```

La lógica clásica es idéntica a cualquier implementación de Shor. Lo que varía entre frameworks es únicamente la llamada a `find_order`, que usa el simulador específico.

### 5.6 Diferencias en la construcción del circuito vs Qiskit

| Componente | Cirq | Qiskit |
|---|---|---|
| QFT | `cirq.qft(*qubits, inverse=True)` | `QFT(m).inverse()` de `qiskit.circuit.library` |
| Exp. modular | `cirq.ArithmeticGate` con `apply()` | `Unitary()` o implementación manual |
| Registros | Rangos de `LineQubit` | `QuantumRegister` con nombre |
| Inicialización `\|1⟩` | `cirq.X(target_qubits[0])` | `qc.x(target[0])` |
| Medición | `cirq.measure(*exponent_qubits, key="result")` | `qc.measure(exponent, classical)` |

La diferencia más significativa es `ArithmeticGate`: Cirq proporciona un mecanismo de alto nivel para definir puertas por su acción clásica, que el simulador resuelve automáticamente. Qiskit no tiene un equivalente directo en su API pública estándar y suele requerir construir la unitaria explícita o usar extensiones específicas.

---

## 6. El Worker de Cirq

### 6.1 Rol del worker en la arquitectura

El archivo `cirq_worker.py` es el punto de entrada del proceso hijo que el sistema de benchmarking lanza como subproceso independiente. Cada ejecución de benchmark crea un proceso nuevo, lo que garantiza que el tiempo de inicialización del framework se mide de forma limpia y que el estado no se contamina entre ejecuciones.

### 6.2 Inicialización del simulador y medición de startup

```python
def _setup_grover(config: BenchmarkConfig):
    import cirq
    from python.cirq.grover import search, grover_circuit

    t0 = time.perf_counter()
    simulator = cirq.Simulator()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def search_call(n, target, num_shots):
        return search(n, target, simulator, num_shots=num_shots)

    def build_call(n, target):
        return grover_circuit(n, target)

    return startup_ms, search_call, build_call
```

El tiempo `startup_ms` mide únicamente la creación de `cirq.Simulator()`. El `import cirq` no se incluye en la medición porque el import ocurre en el momento de la primera llamada a `_setup_grover` y se puede solapar con otros overheads del proceso. El simulador de Cirq es muy ligero: no inicializa GPU ni carga modelos de ruido, así que `startup_ms` suele ser < 5 ms.

### 6.3 La función `build_call`

`build_call(n, target)` construye el circuito sin ejecutarlo. Se usa para medir el tiempo de **compilación** del circuito (en Cirq: solo la construcción de la estructura de datos Python, sin transpilación). El tiempo de build en Cirq es típicamente 1–50 ms dependiendo del tamaño del circuito.

### 6.4 La función `search_call` y `factor_call`

```python
def search_call(n, target, num_shots):
    return search(n, target, simulator, num_shots=num_shots)
```

Esta closure captura el `simulator` ya inicializado. Cada llamada a `search_call` ejecuta el circuito completo `num_shots` veces. El simulator de Cirq ejecuta todas las repeticiones en una sola pasada vectorizada sobre el statevector.

Para Shor:

```python
def _setup_shor(config: BenchmarkConfig):
    import cirq
    from python.cirq.shor.shor import find_factor as _ff

    t0 = time.perf_counter()
    sim = cirq.Simulator()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(N, sim, num_tries=3, num_shots_per_trial=config.num_shots)

    return startup_ms, factor_call
```

`num_shots_per_trial=config.num_shots` controla cuántas veces se ejecuta el circuito de búsqueda de orden por cada base `A` probada. Con pocas repeticiones (p. ej. 10) se reduce el tiempo de simulación a costa de menos robustez en el postprocesado clásico.

### 6.5 Flujo completo del worker

```
main()
  │
  ├─ read_config()          # Lee parámetros del proceso padre (JSON via stdin o file)
  ├─ detect_hardware()      # Info CPU/RAM/GPU para los metadatos del resultado
  │
  ├─ [algo == "grover"]
  │    ├─ _setup_grover()   # Crea simulator, mide startup_ms
  │    └─ run_grover_worker(...)
  │         ├─ build_call(n, target)   # Mide tiempo de construcción del circuito
  │         └─ search_call(n, target, num_shots)  # Mide tiempo de ejecución
  │
  ├─ [algo == "shor"]
  │    ├─ _setup_shor()     # Crea simulator, mide startup_ms
  │    └─ run_shor_worker(...)
  │         └─ factor_call(N)  # Mide tiempo de factorización completa
  │
  └─ write_result(result)   # Serializa BenchmarkResult al proceso padre
```

La separación entre `_setup_*` y `run_*_worker` permite que `run_*_worker` (definido en `_base.py` y compartido por todos los frameworks) maneje la lógica genérica de benchmark (repeticiones, medición de tiempos, construcción del objeto resultado) sin conocer los detalles de Cirq.

---

## 7. Particularidades y Diferencias vs Qiskit

### 7.1 Sin transpilación automática

En Qiskit, al llamar `backend.run(circuit)`, el framework aplica automáticamente un conjunto de pasadas de optimización: descomposición en el conjunto de puertas del backend, layout de qubits físicos, routing para la conectividad del hardware, y optimizaciones de peephole. Todo esto ocurre de forma implícita.

En Cirq, **nada de esto ocurre automáticamente**. El circuito se ejecuta exactamente como está descrito. Para hardware real, el programador debe:

1. Descomponer manualmente las puertas en el conjunto nativo del dispositivo (`√X`, `fSim`, etc.).
2. Aplicar explícitamente los transformadores de Cirq (`cirq.optimize_for_target_gateset`, `cirq.align_left`, etc.).

Para simulación, esto es una ventaja: el simulador entiende directamente puertas de alto nivel como `cirq.H`, `cirq.CCX` o `cirq.Z.controlled(n)` sin descomposición.

### 7.2 Puertas multi-controladas: más verbosas en Cirq

En Qiskit, la puerta `MCX` (multi-controlled X) con `k` controles se obtiene simplemente como `qc.mcx(controls, target)`.

En Cirq:

```python
# Z con n-1 controles
mcz = cirq.Z.controlled(num_controls=n - 1)
circuit.append(mcz.on(*qubits))  # primeros n-1 son controles, el último es target
```

El método `.controlled(num_controls)` existe para cualquier puerta de Cirq, lo que es potente y consistente, pero requiere entender el modelo de la puerta base. Para Grover se usa `Z.controlled` porque la MCZ en el sistema de fase sin ancilla es equivalente y más natural.

Para Toffoli (2 controles):

```python
cirq.CCX(q0, q1, q2)    # equivalente a Toffoli de Qiskit
cirq.CCNOT(q0, q1, q2)  # alias
```

### 7.3 Comparación de la MCZ con el ancilla trick de Qiskit

El oráculo de Grover en Qiskit típicamente usa un qubit ancilla inicializado en `|−⟩` y una puerta MCX:

```python
# Qiskit (con ancilla)
ancilla = QuantumRegister(1, 'ancilla')
qc.x(ancilla)
qc.h(ancilla)
qc.mcx(control_qubits, ancilla)   # flip controlado del ancilla
# El ancilla en |−> acumula la fase global como fase relativa
```

La equivalencia matemática es:
```
MCX · |x⟩|−⟩ = (-1)^{f(x)} |x⟩|−⟩   (si f(x) = 1 para x = target)
```

El efecto neto es el mismo que la MCZ directa de Cirq, pero con un qubit extra. Cirq evita el ancilla usando la Z multi-controlada que actúa directamente en el espacio de fase. Esto reduce el número de qubits en 1, lo que es relevante en hardware NISQ real.

### 7.4 Ventajas de Cirq

1. **Control preciso sobre los Moments**: puedes ver y manipular exactamente en qué capa temporal está cada puerta. Crucial para optimización manual de profundidad de circuito.

2. **Depuración directa**: `print(circuit)` muestra la representación ASCII del circuito con los Moments visibles. `simulator.simulate(circuit)` da el statevector exacto en cualquier punto.

3. **Integración nativa con hardware Google**: el mismo código que simula en Cirq puede enviarse a un procesador Sycamore real via Google Cloud.

4. **`ArithmeticGate`**: permite definir puertas aritméticas por su lógica clásica, lo que simplifica enormemente algoritmos como Shor sin perder corrección matemática.

5. **Sin magia implícita**: el comportamiento del framework es siempre predecible y auditable.

### 7.5 Desventajas de Cirq

1. **Menos abstracciones de alto nivel**: no hay equivalente a las `qiskit.circuit.library` de Qiskit (que incluye QFT, oráculos, QAOA, VQE preempaquetados).

2. **Verbosidad**: construir circuitos complejos requiere más líneas de código que en Qiskit.

3. **Menor ecosistema**: Qiskit tiene más tutoriales, extensiones de terceros y comunidad activa.

4. **Sin transpilación automática**: una ventaja para expertos, una fuente de errores para principiantes que despliegan en hardware real.

5. **Convención big-endian**: la mayoría de recursos de computación cuántica usan little-endian (convención Qiskit). Trabajar en Cirq requiere vigilancia constante sobre el orden de bits.

---

## 8. Ejercicios y Extensiones

### Ejercicio 1: Construir y visualizar un circuito de Bell

```python
import cirq

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key="bell"),
])
print(circuit)

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
print(result.histogram(key="bell"))
# Esperado: {0: ~500, 3: ~500} (estados |00> e |11>)
```

### Ejercicio 2: Grover con 4 qubits

```python
from python.cirq.grover import grover_circuit
import cirq

n = 4
target = 11  # |1011>
circuit = grover_circuit(n, target)
print(f"Profundidad del circuito: {len(circuit)}")
print(circuit)

sim = cirq.Simulator()
result = sim.run(circuit, repetitions=2048)
hist = result.histogram(key="result")
print(f"Estado más frecuente: {max(hist, key=hist.get)}")  # debe ser 11
```

### Ejercicio 3: Inspeccionar el statevector tras el oráculo

```python
from python.cirq.grover import build_oracle
import cirq

n = 3
target = 5  # |101>
qubits = cirq.LineQubit.range(n)

# Circuito: superposición + oráculo
circuit = cirq.Circuit()
circuit.append(cirq.H.on_each(*qubits))
circuit += build_oracle(n, target)

sim = cirq.Simulator()
result = sim.simulate(circuit)
sv = result.final_state_vector
print("Amplitudes tras el oráculo:")
for i, amp in enumerate(sv):
    sign = "-" if amp.real < -0.01 else "+"
    print(f"  |{i:03b}>: {sign}{abs(amp.real):.4f}")
# El estado |101> = |5> debe tener amplitud negativa
```

### Ejercicio 4: Añadir ruido al simulador

```python
import cirq

# Simulador con canal de depolarización
noise = cirq.depolarize(p=0.01)   # 1% de error por puerta
noisy_sim = cirq.DensityMatrixSimulator(noise=noise)

# El mismo circuito de Grover pero con ruido
from python.cirq.grover import grover_circuit
circuit = grover_circuit(3, 5)

result = noisy_sim.run(circuit, repetitions=1024)
print(result.histogram(key="result"))
```

---

## 9. Resumen Conceptual

Cirq es un framework diseñado para ingenieros que necesitan control total sobre el circuito cuántico. Sus tres contribuciones conceptuales principales son:

1. **`Moment` como primitiva de primer orden**: el tiempo es explícito en el modelo de circuito. No hay un DAG implícito; el programador es responsable de la estructura temporal.

2. **Sin transpilación implícita**: lo que describes es lo que se ejecuta. Esto maximiza la previsibilidad y es esencial para hardware NISQ donde cada puerta cuesta.

3. **`ArithmeticGate`**: un mecanismo elegante para definir puertas cuánticas por su acción clásica, permitiendo simular algoritmos aritméticos complejos (como Shor) sin implementar la aritmética cuántica a nivel de puertas primitivas.

Para algoritmos de búsqueda (Grover) la diferencia con Qiskit es principalmente sintáctica y de convención de bits. Para algoritmos de factorización (Shor), la diferencia es más sustancial: `ArithmeticGate` es una abstracción específica de Cirq sin equivalente directo en Qiskit, que simplifica considerablemente la implementación de la exponenciación modular cuántica.

---

## Referencias

### Citas Obligatorias

1. Cirq Developers (2023). "Cirq." Zenodo. https://doi.org/10.5281/zenodo.4062499.

2. Arute, F., et al. (2019). "Quantum supremacy using a programmable superconducting processor." *Nature*, 574, 505–510. https://doi.org/10.1038/s41586-019-1666-5.

3. Nielsen, M.A. & Chuang, I.L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.

4. Harrigan, M.P., et al. (2021). "Quantum approximate optimization of non-planar graph problems on a planar superconducting processor." *Nature Physics*, 17, 332–336. https://doi.org/10.1038/s41567-021-01333-4.

5. Babbush, R., et al. (2021). "Focus beyond quadratic speedups for error-corrected quantum advantage." *PRX Quantum*, 2, 010103. https://doi.org/10.1103/PRXQuantum.2.010103.

6. Boixo, S., et al. (2018). "Characterizing quantum supremacy in near-term devices." *Nature Physics*, 14, 595–600. https://doi.org/10.1038/s41567-018-0124-x.

### Citas Adicionales: NISQ, Simulación Cuántica y Algoritmos en Cirq

7. Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond." *Quantum*, 2, 79. https://doi.org/10.22331/q-2018-08-06-79.

8. Barenco, A., et al. (1995). "Elementary gates for quantum computation." *Physical Review A*, 52(5), 3457. https://doi.org/10.1103/PhysRevA.52.3457.

9. Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search." In *Proceedings of the 28th Annual ACM Symposium on Theory of Computing* (STOC '96), pp. 212–219. https://doi.org/10.1145/237814.237866.

10. Shor, P. W. (1994). "Algorithms for quantum computation: discrete logarithms and factoring." In *Proceedings of the 35th Annual Symposium on Foundations of Computer Science* (FOCS '94), pp. 124–134. https://doi.org/10.1109/SFCS.1994.365700.

11. Google Quantum AI (2022). "Suppressing quantum errors by scaling a surface code." *Nature*, 614, 676–681. https://doi.org/10.1038/s41586-022-05434-3.

12. Bruzewicz, C.D., et al. (2019). "Trapped-ion quantum computing: progress and challenges." *Applied Physics Reviews*, 6, 021314. https://doi.org/10.1063/1.5088164.
