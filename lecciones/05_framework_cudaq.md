# CUDA-Q — Framework de Simulación Cuántica de NVIDIA

> **Nivel**: Grado / Máster en Computación Cuántica  
> **Prerrequisitos**: álgebra lineal básica, Python, nociones de circuitos cuánticos  
> **Duración estimada**: 3–4 horas lectivas

---

## 1. Introducción a CUDA-Q

### 1.1 Historia y origen

CUDA-Q apareció públicamente en octubre de 2023 como evolución directa de **cuQuantum**, la biblioteca de primitivas GPU para simulación cuántica que NVIDIA había lanzado en 2021. cuQuantum proporcionaba dos componentes de bajo nivel: `cuStateVec` (simulación de vectores de estado) y `cuTensorNet` (simulación basada en redes tensoriales). Sin embargo, carecía de un modelo de programación de alto nivel: el usuario tenía que construir la interfaz por encima.

CUDA-Q soluciona esto añadiendo una capa de programación completa sobre cuQuantum. La pila completa queda:

```
┌───────────────────────────────────────┐
│  Python / C++  →  API de CUDA-Q       │  ← el programador trabaja aquí
├───────────────────────────────────────┤
│  Compilador MLIR  (cudaq-mlir)        │  ← IR cuántico intermedio
├───────────────────────────────────────┤
│  cuStateVec / cuTensorNet             │  ← primitivas GPU de NVIDIA
├───────────────────────────────────────┤
│  CUDA / cuBLAS / hardware GPU         │  ← silicio
└───────────────────────────────────────┘
```

La filosofía de diseño es análoga a la de CUDA clásico para computación paralela: el programador escribe **kernels** que se compilan a código eficiente para el acelerador. Esto es radicalmente distinto al enfoque de Qiskit, donde un circuito es un objeto Python que se interpreta en tiempo de ejecución.

### 1.2 Filosofía: aceleración GPU y modelo de kernels compilados

La apuesta central de CUDA-Q es que la simulación cuántica es un problema de álgebra lineal densa, y que las GPU son el hardware más eficiente para ello. Un vector de estado de `n` qubits requiere `2^n` amplitudes complejas; cada puerta cuántica es una transformación unitaria sobre ese espacio. Las GPU pueden paralelizar estas operaciones de forma masiva.

Sin embargo, para explotar realmente la GPU, CUDA-Q necesita **conocer el circuito completo antes de ejecutarlo**. No puede permitir que el circuito sea un objeto Python que se construye dinámicamente en tiempo de ejecución, porque entonces no podría compilarlo a operaciones GPU eficientes. De ahí el modelo de kernels: el circuito se especifica como una función Python que se traduce a una representación intermedia (MLIR) en el momento de la definición, no de la ejecución.

### 1.3 Integración con CUDA: restricciones de plataforma

CUDA-Q solo funciona en **Linux x86_64** con controladores NVIDIA compatibles. Esto no es una limitación arbitraria: el backend GPU requiere la cadena de herramientas CUDA completa (nvcc, libcuda, cuBLAS) que solo está disponible en Linux. En macOS o Windows la instalación mediante pip fallará o solo instalará el simulador CPU (`qpp-cpu`).

La arquitectura de destino mínima es SM 7.0 (Volta, tarjetas V100 en adelante). Para simulaciones de más de 25 qubits se recomienda A100 o H100.

### 1.4 Posición en el ecosistema: HPC y simulación de alto rendimiento

CUDA-Q se posiciona en el segmento de **HPC cuántico**. Su usuario objetivo no es el investigador que prueba algoritmos en un laptop, sino el equipo que ejecuta benchmarks en clústeres con varias GPU. En este contexto compite con:

- **Qiskit Aer** con backend GPU (usa también cuStateVec, pero dentro del modelo de circuitos de Qiskit)
- **PennyLane** con plugin de CUDA (interfaz diferenciable, enfocada a QML)
- **cuQuantum Appliance** (contenedor Docker de NVIDIA con simuladores de referencia)

La diferencia clave es que CUDA-Q es el único framework donde el modelo de programación está diseñado **desde cero** para la compilación AOT (ahead-of-time) a hardware acelerado.

---

## 2. El Modelo de Kernels

### 2.1 El enfoque `cudaq.make_kernel()`: qué hace exactamente

En versiones previas de la API de CUDA-Q, la forma canónica de definir un circuito es `cudaq.make_kernel()`. Esta llamada crea un objeto `Kernel` vacío que actúa como un **constructor de circuito**: cada llamada a métodos como `kernel.h(q)` o `kernel.cx(c, t)` no ejecuta ninguna puerta, sino que **añade una instrucción al IR interno** del kernel.

```python
import cudaq

kernel = cudaq.make_kernel()
q = kernel.qalloc(2)        # reservar 2 qubits
kernel.h(q[0])              # añadir instrucción H sobre qubit 0
kernel.cx(q[0], q[1])       # añadir instrucción CX (CNOT)
kernel.mz(q)                # medir ambos qubits
```

Cuando más adelante se llama a `cudaq.sample(kernel, ...)`, el runtime de CUDA-Q toma ese IR, lo compila mediante MLIR a operaciones nativas del target (cuStateVec, QPP, etc.) y lo ejecuta. La compilación ocurre **una sola vez**; ejecuciones sucesivas reutilizan el código compilado.

> **Nota sobre la API moderna**: CUDA-Q también expone un decorator `@cudaq.kernel` que permite escribir kernels como funciones Python con anotaciones de tipo. Ambas APIs producen el mismo IR interno; en este proyecto se usa `make_kernel()` porque ofrece mayor control programático (construcción dinámica de circuitos parametrizados en n qubits).

### 2.2 Tipos cuánticos: `qalloc`, referencias a qubits

La asignación de qubits se hace exclusivamente con `kernel.qalloc(n)`, que devuelve una referencia a un registro de `n` qubits. Los qubits individuales se acceden por índice: `q[0]`, `q[1]`, etc.

```python
kernel = cudaq.make_kernel()
reg = kernel.qalloc(4)      # registro de 4 qubits, índices 0-3
qubit_0 = reg[0]            # referencia al qubit 0
lista = [reg[i] for i in range(4)]  # lista de referencias
```

Internamente, estas referencias son **handles** al IR, no objetos Python con estado. No se puede leer el estado de un qubit desde Python durante la construcción; solo se pueden emitir instrucciones.

### 2.3 La diferencia fundamental: circuitos como objetos vs. kernels compilados

Este es el punto conceptual más importante para entender CUDA-Q.

**En Qiskit**, un circuito es un objeto Python (`QuantumCircuit`) que almacena una lista de instrucciones. Se puede inspeccionar, modificar, bifurcar condicionalmente en runtime, y ejecutar de forma interpretada. El transpiler de Qiskit lo optimiza en el momento de la transpilación.

**En CUDA-Q**, un kernel es un IR compilado. Una vez construido, no se puede modificar ni inspeccionar desde Python. No existe equivalente a `circuit.data` o `circuit.draw()` en tiempo de ejecución. El beneficio es que el backend puede aplicar optimizaciones agresivas de bajo nivel (reordenación de puertas, fusión de matrices) sin restricciones de interpretación.

La consecuencia práctica es que **toda la lógica de construcción del circuito debe resolverse en Python antes de emitir instrucciones al kernel**. Los bucles `for`, las condiciones `if`, los cálculos de ángulos, todo se evalúa en Python; el resultado son secuencias planas de instrucciones cuánticas.

### 2.4 Restricciones del modelo de kernels

El modelo de CUDA-Q impone restricciones significativas que el programador debe conocer:

| Permitido | No permitido |
|-----------|-------------|
| Bucles `for` con límites conocidos en construcción | Bucles cuyo límite depende de una medición cuántica |
| Condicionales `if` sobre variables Python clásicas | Condicionales sobre resultados de medición mid-circuit (en `make_kernel`) |
| Aritmética Python para calcular ángulos o índices | Clases Python, excepciones, generadores dentro de un `@cudaq.kernel` |
| Listas de referencias a qubits | Objetos Python arbitrarios como parámetros de puertas |
| Composición mediante `apply_call` | Sub-kernels que modifican el mismo registro sin copia |

En el modelo `make_kernel()`, la lógica clásica vive completamente fuera del kernel. Por eso los algoritmos de este proyecto construyen el circuito entero en Python y luego lo entregan como un bloque opaco a CUDA-Q.

### 2.5 Cómo pasar parámetros

Con `make_kernel()`, los parámetros se declaran mediante `cudaq.make_kernel(float)` o `cudaq.make_kernel(list[float])` y se pasan en la llamada a `cudaq.sample`. Sin embargo, en el código de este proyecto todos los parámetros (número de qubits, target, base) se resuelven en Python durante la construcción del kernel, de modo que los kernels resultantes son **sin parámetros**: la llamada a `cudaq.sample(kernel, shots_count=N)` no necesita argumentos adicionales.

---

## 3. Puertas y Operaciones

### 3.1 Puertas básicas del API `make_kernel`

CUDA-Q expone las puertas universales estándar como métodos del objeto kernel:

```python
kernel.h(q)           # Hadamard
kernel.x(q)           # Pauli-X (NOT)
kernel.y(q)           # Pauli-Y
kernel.z(q)           # Pauli-Z
kernel.t(q)           # T (π/8 gate)
kernel.tdg(q)         # T† (conjugado)
kernel.s(q)           # S (√Z)
kernel.sdg(q)         # S†
kernel.rx(angle, q)   # Rotación X
kernel.ry(angle, q)   # Rotación Y
kernel.rz(angle, q)   # Rotación Z
kernel.r1(angle, q)   # Rotación de fase: diag(1, e^{iθ})
kernel.swap(q0, q1)   # SWAP
```

### 3.2 Puertas de dos qubits: CX, CZ, CR1

Las puertas controladas de dos qubits tienen notación directa:

```python
kernel.cx(control, target)        # CNOT
kernel.cz(control, target)        # CZ controlada
kernel.cr1(angle, control, target) # R1 controlada
```

El método `cr1` merece atención especial: es la puerta de fase controlada `CR1(θ)` que aplica `R1(θ)` al target solo cuando el control está en `|1⟩`. Su matriz es:

```
CR1(θ) = diag(1, 1, 1, e^{iθ})
```

Es la puerta fundamental de la QFT y aparece extensamente en `qft.py`.

### 3.3 Puertas multi-control: listas de controles

Cuando se pasa una **lista** de qubits como control, CUDA-Q interpreta la operación como multi-controlada. Esta es la forma de implementar puertas Toffoli generalizadas (CCX, CCCX, etc.):

```python
controls = [q[0], q[1], q[2]]    # tres controles
kernel.cx(controls, q[3])         # CCCX (Toffoli de 3 controles)
kernel.cz(controls, q[3])         # CCCZ (CZ de 3 controles)
```

Este mecanismo es el que usa `grover.py` para implementar el MCZ (multi-controlled Z) tanto en el oráculo como en el difusor:

```python
controls = [qubits[i] for i in range(n - 1)]
kernel.cz(controls, qubits[n - 1])
```

Internamente, CUDA-Q descompone automáticamente las puertas multi-controladas en una secuencia de puertas universales según la plataforma de destino.

### 3.4 Medición: `mz` y su semántica

La medición en CUDA-Q se realiza con `kernel.mz(registro_o_qubit)`. Acepta tanto un registro completo como un qubit individual:

```python
kernel.mz(q)         # mide todo el registro q
kernel.mz(q[0])      # mide solo el qubit 0
```

`mz` realiza una medición en la base computacional (Z) y **colapsa** el qubit al estado medido. En el modo `cudaq.sample()`, la medición se repite `shots_count` veces desde el principio del circuito (no es una medición continua del mismo estado), y los resultados se acumulan en un histograma.

Importante: en el modelo `make_kernel()`, los resultados de `mz` **no son accesibles dentro del kernel** como variables Python. Solo están disponibles en el objeto `SampleResult` devuelto por `cudaq.sample()`.

---

## 4. Ejecución y Targets

### 4.1 `cudaq.sample()`: modo sampling

```python
result = cudaq.sample(kernel, shots_count=1024)
```

Esta es la forma principal de ejecutar un kernel cuántico en CUDA-Q. El runtime:
1. Compila el kernel al IR nativo del target activo (si no estaba compilado ya).
2. Inicializa el vector de estado en `|0...0⟩`.
3. Aplica todas las instrucciones del kernel.
4. Lee el resultado de la medición final.
5. Repite los pasos 2–4 `shots_count` veces.
6. Devuelve un objeto `SampleResult` con el histograma de resultados.

La granularidad del sampling es importante: cada "shot" es una ejecución completa e independiente del circuito. No hay reutilización del vector de estado entre shots.

### 4.2 `cudaq.get_state()`: modo statevector

```python
state = cudaq.get_state(kernel)
amplitudes = [state[i] for i in range(2**n)]
```

Este modo ejecuta el circuito **una sola vez** y devuelve el vector de estado completo, incluyendo amplitudes complejas. Es útil para depuración y análisis teórico, pero no escala (requiere almacenar `2^n` números complejos en memoria).

No es adecuado para simulación con mediciones mid-circuit ni para circuitos con ruido.

### 4.3 Targets disponibles

CUDA-Q separa el concepto de **programa** (el kernel) del **target** (el hardware o simulador donde se ejecuta). Los targets principales son:

| Target | Descripción | Cuándo usar |
|--------|-------------|-------------|
| `qpp-cpu` | Simulador CPU basado en la librería Q++ (single-threaded) | Desarrollo, testing, macOS/Windows |
| `nvidia` | Alias de `custatevec-fp32`, GPU con precisión de 32 bits | Producción GPU, eficiencia máxima |
| `custatevec-fp32` | GPU, aritmética float32, más rápido pero menos preciso | Circuitos grandes, presupuesto de error bajo |
| `custatevec-fp64` | GPU, aritmética float64, más lento pero más preciso | Resultados de referencia, validación |
| `tensornet` | Simulador de redes tensoriales, escala mejor para circuitos poco entrelazados | Circuitos con baja profundidad de entrelazamiento |

### 4.4 Selección de target

```python
cudaq.set_target("nvidia")        # GPU NVIDIA con cuStateVec fp32
cudaq.set_target("qpp-cpu")       # CPU, sin requisitos GPU
cudaq.set_target("custatevec-fp64")  # GPU fp64
```

La selección del target es **global y persistente** dentro del proceso Python. Una vez establecido, todos los kernels subsiguientes se ejecutan en ese target hasta que se cambie. Por eso el worker del proyecto configura el target una vez al inicio y no lo repite en cada llamada.

### 4.5 Leer resultados: `SampleResult`

El objeto `SampleResult` actúa como un diccionario de cadenas de bits a recuentos:

```python
result = cudaq.sample(kernel, shots_count=1024)

# Iterar sobre resultados
for bitstring, count in result.items():
    print(f"|{bitstring}⟩: {count} veces")

# Obtener el resultado más frecuente
most_probable = result.most_probable()  # devuelve la cadena de bits

# Recuento específico
count_00 = result["00"]
```

**Convención de orden de bits importante**: CUDA-Q devuelve cadenas de bits con el **qubit 0 a la izquierda** (MSB en posición 0, convención big-endian). Qiskit usa la convención opuesta (qubit 0 a la derecha, LSB). Para comparar resultados entre frameworks, es necesario revertir la cadena.

En `grover.py` esto se resuelve explícitamente:

```python
dist = {bitstring[::-1]: count for bitstring, count in result.items()}
```

---

## 5. Grover en CUDA-Q

### 5.1 Estructura general del algoritmo de Grover

El algoritmo de Grover busca un elemento marcado en una base de datos no estructurada de `N = 2^n` elementos. El circuito aplica `k ≈ π/4 · √N` iteraciones de:

1. **Oráculo de fase**: aplica una fase `-1` al estado marcado `|target⟩`.
2. **Difusor de Grover**: reflexión sobre el estado uniforme `|s⟩ = H^n |0^n⟩`.

El código en `grover.py` construye todo el circuito en un único kernel, inlineando el oráculo y el difusor en cada iteración. No hay composición de sub-kernels.

### 5.2 Construcción del kernel completo: `grover_circuit()`

```python
def grover_circuit(n: int, target: int, num_iterations: int | None = None) -> cudaq.Kernel:
    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n)
```

**Línea a línea**:
- `num_iterations`: el número óptimo de iteraciones es `⌊π/4 · √(2^n)⌋`. Para n=3 (8 estados): 2 iteraciones. Para n=4 (16 estados): 3 iteraciones.
- `cudaq.make_kernel()`: crea un kernel vacío sin parámetros.
- `kernel.qalloc(n)`: reserva `n` qubits, todos inicializados a `|0⟩`.

```python
    # Preparar superposición uniforme
    for i in range(n):
        kernel.h(qubits[i])
```

Aplicar H a cada qubit transforma `|0^n⟩` en el estado uniforme `|s⟩ = (1/√2^n) Σ|x⟩`.

### 5.3 El oráculo de fase

El oráculo debe aplicar la transformación `|x⟩ → (-1)^{f(x)} |x⟩`, donde `f(x) = 1` si y solo si `x = target`.

La estrategia es:
1. Aplicar X a todos los qubits donde `target` tiene un `0` en ese bit. Esto transforma `|target⟩ → |1...1⟩`.
2. Aplicar un Z multi-controlado (MCZ) que invierte la fase de `|1...1⟩`.
3. Deshacer los X del paso 1.

```python
    for _ in range(num_iterations):
        # Paso 1: X donde target tiene 0
        for i in range(n):
            if not (target >> i) & 1:
                kernel.x(qubits[i])

        # Paso 2: MCZ (Z multi-controlado)
        if n == 1:
            kernel.z(qubits[0])
        else:
            controls = [qubits[i] for i in range(n - 1)]
            kernel.cz(controls, qubits[n - 1])

        # Paso 3: deshacer X
        for i in range(n):
            if not (target >> i) & 1:
                kernel.x(qubits[i])
```

**Ejemplo para n=3, target=5 (binario 101)**:
- Bit 0: `(5 >> 0) & 1 = 1`, no X.
- Bit 1: `(5 >> 1) & 1 = 0`, aplicar X a `qubits[1]`.
- Bit 2: `(5 >> 2) & 1 = 1`, no X.

Después de los X, el estado `|101⟩` se habrá convertido en `|111⟩`. El MCZ `kernel.cz([q[0], q[1]], q[2])` invierte la fase de `|111⟩`. Los X de vuelta restauran `|111⟩ → |101⟩`, con la fase `-1` ya aplicada.

### 5.4 El difusor de Grover

El difusor implementa la reflexión `2|s⟩⟨s| - I`. Su construcción es simétrica al oráculo pero siempre invierte la fase de `|0...0⟩`:

```python
        # Difusor: H^n → X^n → MCZ → X^n → H^n
        for i in range(n):
            kernel.h(qubits[i])

        for i in range(n):
            kernel.x(qubits[i])

        if n == 1:
            kernel.z(qubits[0])
        else:
            controls = [qubits[i] for i in range(n - 1)]
            kernel.cz(controls, qubits[n - 1])

        for i in range(n):
            kernel.x(qubits[i])

        for i in range(n):
            kernel.h(qubits[i])
```

**Verificación matemática**:
- `H^n` cambia de base: el estado uniforme `|s⟩` se convierte en `|0...0⟩`.
- `X^n` hace que `|0...0⟩ → |1...1⟩`.
- MCZ invierte la fase de `|1...1⟩`.
- `X^n` restaura `|1...1⟩ → |0...0⟩` (con fase `-1`).
- `H^n` vuelve a la base original.

El efecto neto es `2|s⟩⟨s| - I`, la reflexión en torno al estado promedio.

### 5.5 Medición y ejecución

```python
    kernel.mz(qubits)
    return kernel
```

El kernel se completa con una medición de todos los qubits. La función `search()` lo ejecuta:

```python
def search(n, target, simulator=None, num_shots=1024):
    if simulator is not None:
        cudaq.set_target(simulator)

    kernel = grover_circuit(n, target, num_iterations=iters)
    result = cudaq.sample(kernel, shots_count=num_shots)

    dist = {bitstring[::-1]: count for bitstring, count in result.items()}
    most_frequent = max(dist, key=dist.get)
    found = int(most_frequent, 2)
    return found, dist
```

La inversión de cadena `[::-1]` es esencial: CUDA-Q representa el qubit 0 como el bit más significativo, pero la interpretación estándar de un entero en binario pone el qubit 0 en la posición de menor peso. Sin esta corrección, `target=5` (binario `101`) aparecería en los resultados como `101` cuando CUDA-Q lo muestra como `101` con qubit 0 a la izquierda, pero la conversión `int("101", 2) = 5` coincidiría solo por casualidad en este caso; para targets asimétricos como `target=6` (binario `110`) el resultado sería incorrecto sin la inversión.

### 5.6 Por qué no hay kernels anidados en esta implementación

El comentario en `grover_circuit()` explica: *"Since CUDA-Q kernels cannot easily compose sub-kernels dynamically, the full circuit (superposition + oracle + diffuser + measurement) is built inline in a single kernel."*

La API `make_kernel()` ofrece `kernel.apply_call(otro_kernel)` para composición, pero su uso con kernels parametrizados (donde los argumentos son listas de qubits construidas dinámicamente) es complejo y propenso a errores en versiones previas de la API. La solución pragmática es construir el circuito completo inline, sacrificando la modularidad a cambio de robustez.

---

## 6. Shor en CUDA-Q

El algoritmo de Shor para factorizar un entero N requiere encontrar el orden r de un elemento A en el grupo multiplicativo Z_N, es decir, el menor r tal que `A^r ≡ 1 (mod N)`. El núcleo cuántico es el circuito de estimación de fase cuántica (QPE) aplicado al operador de exponenciación modular `U_A: |y⟩ → |A·y mod N⟩`.

La implementación en CUDA-Q consta de tres módulos: `qft.py`, `permutation.py` y `shor.py`.

### 6.1 La QFT en `qft.py`: estructura y puertas CR1

La Transformada de Fourier Cuántica (QFT) sobre `n` qubits implementa la DFT sobre el grupo cíclico Z_{2^n}. Su circuito estándar consiste en:

1. Para cada qubit i de 0 a n-1: aplicar H, luego aplicar R1(π/2^k) controlada por el qubit j para j = i+1, ..., n-1 (donde k = j - i).
2. Reordenar qubits con SWAPs.

```python
def apply_qft(kernel: cudaq.Kernel, qubits: list, n: int) -> None:
    for i in range(n):
        kernel.h(qubits[i])
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            kernel.cr1(angle, qubits[j], qubits[i])

    for i in range(n // 2):
        kernel.swap(qubits[i], qubits[n - 1 - i])
```

**Verificación del ángulo**: para el par (i=0, j=1), el ángulo es π/2 (puerta S controlada). Para (i=0, j=2), el ángulo es π/4 (puerta T controlada). La secuencia de ángulos es exactamente la que produce la transformada de Fourier discreta sobre 2^n puntos.

La **QFT inversa** (`apply_inverse_qft`) simplemente invierte el orden de las operaciones y niega los ángulos:

```python
def apply_inverse_qft(kernel: cudaq.Kernel, qubits: list, n: int) -> None:
    for i in range(n // 2):
        kernel.swap(qubits[i], qubits[n - 1 - i])

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            angle = -math.pi / (2 ** (j - i))
            kernel.cr1(angle, qubits[j], qubits[i])
        kernel.h(qubits[i])
```

La inversión se aplica al registro de control del circuito QPE, cuyo tamaño es `m = 2·ceil(log2(N))` qubits de precisión.

### 6.2 La exponenciación modular como permutación de estados de base

Este es el aspecto más singular de la implementación CUDA-Q de Shor y merece análisis detallado.

La exponenciación modular cuántica implementa el operador `U_A^k: |y⟩ → |A^k · y mod N⟩` para el registro target. En Qiskit esto se implementa mediante aritmética cuántica (sumadores cuánticos, multiplicadores modulares, etc.), que son circuitos genéricos y profundos.

En CUDA-Q, el modelo de kernels impone una restricción crítica: **no es posible usar aritmética clásica dentro del kernel para computar qué puerta aplicar en función del estado cuántico**. El kernel es estático; sus instrucciones no pueden depender de valores cuánticos en tiempo de ejecución.

La solución adoptada en `permutation.py` es un enfoque diferente y elegante: se observa que `U_A^k` es simplemente una **permutación del espacio de estados de base** `{|0⟩, |1⟩, ..., |N-1⟩}`. Para cada entrada `y` ∈ {0, ..., N-1}, la salida es `(A^k · y) mod N`. Esta permutación puede calcularse **clásicamente** antes de construir el kernel.

```python
def build_mod_exp_permutation(A: int, N: int, power: int) -> dict[int, int]:
    a_power = pow(A, power, N)   # A^power mod N, calculado clásicamente
    permutation = {}
    for y in range(N):
        target = (a_power * y) % N
        if y != target:              # solo guardar los no triviales
            permutation[y] = target
    return permutation
```

Una vez conocida la permutación, se implementa en el circuito cuántico como una secuencia de transposiciones (intercambios de pares de estados base). Cualquier permutación puede descomponerse en ciclos disjuntos, y cada ciclo en transposiciones.

### 6.3 Descomposición de permutaciones en transposiciones controladas

La función `controlled_swap_permutation` implementa una permutación controlada descomponiéndola en ciclos:

```python
def controlled_swap_permutation(kernel, ctrl, target_qubits, permutation):
    visited = set()
    for start in sorted(permutation.keys()):
        if start in visited:
            continue
        cycle = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = permutation.get(current, current)

        if len(cycle) <= 1:
            continue

        for idx in range(1, len(cycle)):
            controlled_transposition(kernel, ctrl, target_qubits, cycle[0], cycle[idx])
```

Un ciclo `(a → b → c → a)` se descompone en las transposiciones `(a,b)` y `(a,c)` (no `(b,c)` y `(a,c)`, sino siguiendo el pivote `a`). Esto es correcto porque:
- Empezando desde `(a,b)`: `a↔b`, estado: `(b → a → c → b)`.
- Luego `(a,c)`: ahora `a` está donde estaba `b`, así que intercambia las posiciones de los estados `a` y `c`.

Cada transposición `|a⟩ ↔ |b⟩` se implementa en `controlled_transposition` mediante una descomposición recursiva en transposiciones de un solo bit:

```python
def controlled_transposition(kernel, ctrl, target_qubits, a, b):
    diff_bits = a ^ b
    if diff_bits == 0:
        return
    diff_positions = [i for i in range(n) if (diff_bits >> i) & 1]

    if len(diff_positions) == 1:
        _controlled_single_bit_transposition(kernel, ctrl, target_qubits, a, b)
    else:
        pivot = diff_positions[0]
        a_prime = a ^ (1 << pivot)
        controlled_transposition(kernel, ctrl, target_qubits, a, a_prime)
        controlled_transposition(kernel, ctrl, target_qubits, a_prime, b)
        controlled_transposition(kernel, ctrl, target_qubits, a, a_prime)
```

La recursión reduce cualquier transposición a transposiciones de estados que difieren en exactamente un bit. Estas últimas son CX multi-controladas: se activan cuando todos los otros bits del registro coinciden con `a`, y el bit de diferencia se invierte:

```python
def _controlled_single_bit_transposition(kernel, ctrl, target_qubits, a, b):
    diff = a ^ b
    flip_bit = diff.bit_length() - 1
    other_positions = [i for i in range(n) if i != flip_bit]

    # Activar when los otros bits son iguales a a
    for pos in other_positions:
        if not ((a >> pos) & 1):
            kernel.x(target_qubits[pos])

    controls = [ctrl] + [target_qubits[pos] for pos in other_positions]
    kernel.cx(controls, target_qubits[flip_bit])

    for pos in other_positions:
        if not ((a >> pos) & 1):
            kernel.x(target_qubits[pos])
```

Este es un CX multi-controlado con `n` controles en total (el qubit externo `ctrl` más los `n-1` qubits del registro que actúan como condición). El patrón X-CX-X es la forma estándar de implementar un CNOT condicionado a un valor específico en lugar de condicionado a `|1⟩`.

### 6.4 El circuito de estimación de fase en `shor.py`

```python
def order_finding_circuit(A: int, N: int, precision: int | None = None) -> cudaq.Kernel:
    n = math.ceil(math.log2(N))          # qubits para el registro target
    m = precision if precision is not None else 2 * n  # qubits de precisión

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(m + n)        # m qubits control + n qubits target

    ctrl_qubits = [qubits[i] for i in range(m)]
    tgt_qubits  = [qubits[m + i] for i in range(n)]
    tgt_qubits_lsb = list(reversed(tgt_qubits))  # índice 0 = bit menos significativo

    # Preparar registro de control en superposición uniforme
    for i in range(m):
        kernel.h(ctrl_qubits[i])

    # Inicializar registro target en |1⟩
    kernel.x(tgt_qubits_lsb[0])

    # Aplicar U_A^{2^k} controlado por ctrl_qubits[k]
    for i in range(m):
        power = 2 ** (m - 1 - i)
        perm = build_mod_exp_permutation(A, N, power)
        if perm:
            controlled_swap_permutation(kernel, ctrl_qubits[i], tgt_qubits_lsb, perm)

    # IQFT sobre el registro de control
    apply_inverse_qft(kernel, ctrl_qubits, m)

    # Medir el registro de control
    for i in range(m):
        kernel.mz(ctrl_qubits[i])

    return kernel
```

**Detalles de implementación**:

1. **Convención de endianness**: `cudaq` asigna qubits con el índice 0 como MSB en la cadena de bits. El registro target se invierte (`tgt_qubits_lsb`) para que el índice Python 0 corresponda al bit de menor peso, facilitando la aritmética modular de `permutation.py`.

2. **Potencias decrecientes**: El qubit de control `ctrl_qubits[0]` controla `U_A^{2^{m-1}}` (la mayor potencia), no `U_A^1`. Esto es necesario porque `ctrl_qubits[0]` es el bit más significativo en la salida de la IQFT.

3. **Inicialización en |1⟩**: El registro target se inicializa en `|1⟩` (con `x(tgt_qubits_lsb[0])`). El operador `U_A` actúa sobre `|1⟩` produciendo `|A mod N⟩`, `|A^2 mod N⟩`, etc. Con la inicialización en `|0⟩`, la exponenciación modular dejaría el registro en `|0⟩` (porque `A^k · 0 ≡ 0`).

### 6.5 Extracción del orden con fracciones continuas

```python
def _get_order_from_dist(dist, A, N, precision):
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    for i in range(min(10, len(sorted_outputs))):
        bitstring = sorted_outputs[i]
        if all(c == "0" for c in bitstring):
            continue
        x = int(bitstring, 2)
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            r = _reduce_to_min_order(r, A, N)
            return r
    return 0
```

La IQFT produce, en el registro de control, un valor cercano a `j/r · 2^m` donde `j` es algún entero entre 0 y r-1. La conversión `x / 2^m ≈ j/r` se aproxima mediante fracciones continuas (`Fraction(...).limit_denominator(N-1)`): el denominador de la fracción reducida da el orden `r` (o un divisor del mismo). La verificación `pow(A, r, N) == 1` confirma que el candidato es correcto.

### 6.6 Comparación con la implementación de Qiskit

| Aspecto | CUDA-Q | Qiskit |
|---------|--------|--------|
| Exponenciación modular | Permutación de estados base (clásicamente precalculada) | Aritmética cuántica genérica (sumadores de Draper, etc.) |
| Profundidad del circuito | Alta (muchos CX multi-controlados) | Más baja con optimizaciones de transpilación |
| Flexibilidad | Funciona para cualquier A, N sin aritmética cuántica | Requiere diseño de circuitos aritméticos específicos |
| Modelo de composición | Todo inline en un único kernel | Composición modular con QuantumCircuit.compose() |
| Control de flujo en runtime | No (el circuito es estático) | Parcial (mid-circuit measurements con condicionales) |

---

## 7. El Worker de CUDA-Q

El archivo `cudaq_worker.py` es el punto de entrada del sistema de benchmarking. Su diseño refleja las particularidades operativas de CUDA-Q.

### 7.1 Importación condicional: manejo de la disponibilidad de CUDA-Q

```python
def main() -> None:
    try:
        cfg = read_config()
    except Exception as e:
        write_error(f"failed to read config: {e}")
        return
    ...
    try:
        import cudaq  # noqa: F401
    except ImportError as e:
        write_error(f"cudaq not available: {e}")
        return
```

CUDA-Q no puede importarse en macOS o en entornos sin los drivers de NVIDIA. En lugar de fallar en el nivel del módulo (lo que impediría cargar el worker en absoluto), la importación se hace dentro del `main()` y el worker devuelve un error limpio si no está disponible. El comentario `# noqa: F401` suprime la advertencia del linter por "importación no usada" (se importa solo para verificar que existe).

### 7.2 Selección dinámica de target

```python
cudaq_target = cfg.get("cudaq_target", "qpp-cpu")
```

El target se lee de la configuración con `"qpp-cpu"` como valor por defecto. El sistema de benchmarking puede sobreescribir esto con `"nvidia"` cuando detecta una GPU disponible. La selección real se realiza en `_setup_grover`:

```python
def _setup_grover(config, cudaq_target):
    import cudaq
    from python.cudaq.grover import search, grover_circuit

    t0 = time.perf_counter()
    cudaq.set_target(cudaq_target)
    startup_ms = (time.perf_counter() - t0) * 1000.0
    ...
```

### 7.3 Medición de `startup_time`

```python
t0 = time.perf_counter()
cudaq.set_target(cudaq_target)
startup_ms = (time.perf_counter() - t0) * 1000.0
```

La llamada `cudaq.set_target()` no es instantánea: implica inicializar el contexto CUDA (si el target es GPU), cargar librerías dinámicas, y configurar el compilador MLIR. Para el target `nvidia`, este tiempo puede ser de varios cientos de milisegundos. El sistema de benchmarking lo registra como `startup_time` para no contaminar las mediciones de tiempo de circuito.

Este patrón es análogo a la medición del "warm-up time" en benchmarks de GPU: la primera ejecución siempre incluye el tiempo de compilación JIT y de inicialización del contexto.

### 7.4 Closures para encapsular la configuración

```python
def _setup_grover(config, cudaq_target):
    ...
    def search_call(n, target, num_shots):
        return search(n, target, simulator=None, num_shots=num_shots)

    def build_call(n, target):
        return grover_circuit(n, target)

    return startup_ms, search_call, build_call
```

El diseño usa closures para encapsular la configuración del target. `search_call` no necesita recibir el nombre del target porque ya fue configurado globalmente por `cudaq.set_target(cudaq_target)`. Pasar `simulator=None` en `search()` hace que no se intente cambiar el target nuevamente.

### 7.5 Manejo de errores específicos de CUDA-Q

```python
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        write_error(f"cudaq {algo} n={n} failed: {e}")
        return
```

Los errores de CUDA-Q más comunes incluyen:
- `cudaq::InvalidQUBITError`: intento de acceder a un qubit fuera de rango.
- Errores de memoria CUDA cuando el número de qubits supera la VRAM disponible.
- Fallos de compilación MLIR para circuitos con patrones no soportados.

El worker captura cualquier excepción, imprime el traceback completo en stderr (visible en logs del sistema de benchmarking) y devuelve un error estructurado al proceso padre.

---

## 8. Ventajas, Limitaciones y Comparación

### 8.1 Aceleración GPU con cuStateVec: cuándo es realmente más rápido

La aceleración GPU de CUDA-Q es significativa solo cuando:

1. **El número de qubits es suficientemente grande** (n > 18-20): para n pequeño, el overhead de transferencia CPU-GPU y de inicialización del contexto supera el beneficio computacional.

2. **El circuito aplica puertas densas** (como la QFT): las puertas que actúan sobre muchos qubits son más eficientes en GPU porque su representación matricial es densa. Las puertas locales de 1-2 qubits tienen poco paralelismo.

3. **Se ejecutan muchos shots**: el sampling de 10.000+ shots amortiza el overhead de inicialización. Para 10 shots (como en el benchmark de Shor), la GPU raramente gana.

Tabla orientativa de ventana de speedup:

| Qubits | GPU (A100) vs CPU (8 núcleos) | Observaciones |
|--------|------------------------------|---------------|
| < 18   | 0.5–1x (CPU más rápido)      | Overhead de inicialización domina |
| 18–24  | 1–5x                         | Zona de transición |
| 24–28  | 5–20x                        | GPU claramente superior |
| > 28   | >20x (si cabe en VRAM)       | Limitado por memoria GPU (80 GB en A100) |

### 8.2 El modelo de kernels como limitación de expresividad

El modelo de kernels compilados de CUDA-Q impone limitaciones que no existen en Qiskit:

**Circuitos adaptativos (feedback clásico-cuántico)**: En CUDA-Q con `make_kernel()`, no es posible medir un qubit a mitad del circuito y condicionar las puertas siguientes en el resultado. Qiskit sí soporta esto con `if_test()`. CUDA-Q lo soporta en la API `@cudaq.kernel` con sintaxis especial, pero con restricciones.

**Circuitos parametrizados dinámicamente**: Cambiar los parámetros de un circuito (ángulos de rotación) en CUDA-Q requiere reconstruir el kernel. En Qiskit, los parámetros simbólicos se pueden vincular en tiempo de ejecución sin recompilación.

**Introspección del circuito**: En Qiskit se puede inspeccionar `circuit.data`, contar puertas, extraer la profundidad, y dibujar el circuito. En CUDA-Q con `make_kernel()`, el IR compilado no es accesible desde Python.

### 8.3 Restricciones de plataforma

- **Solo Linux x86_64**: cuQuantum no tiene soporte oficial para macOS ni para ARM. El target `qpp-cpu` funciona en todas las plataformas pero sin aceleración.
- **Versiones de Python restringidas**: CUDA-Q publica wheels para Python 3.10 y 3.11 (en el momento de escribir esta lección). Python 3.12+ puede requerir compilación desde fuentes.
- **Dependencia de versión de CUDA**: la versión del wheel de CUDA-Q debe coincidir con la versión de los drivers CUDA instalados (CUDA 11 vs CUDA 12 son incompatibles).

### 8.4 Comparación global con Qiskit

| Dimensión | CUDA-Q | Qiskit |
|-----------|--------|--------|
| **Modelo de programación** | Kernels compilados (IR estático) | Objetos de circuito dinámicos |
| **Ejecución** | Compilación AOT + ejecución acelerada | Transpilación + ejecución interpretada |
| **Backend GPU** | Nativo (cuStateVec integrado) | Plugin (qiskit-aer con cuStateVec) |
| **Soporte multiplataforma** | Solo Linux x86_64 (GPU) | Linux, macOS, Windows |
| **Circuitos adaptativos** | Soporte limitado | Soporte completo |
| **Ecosistema** | Orientado a HPC, NVIDIA | Ecosistema amplio, IBM, comunidad grande |
| **Introspección** | Limitada | Completa (draw, depth, count_ops) |
| **Composición modular** | Compleja con make_kernel | Natural con QuantumCircuit.compose() |
| **Rendimiento n>24** | Superior (GPU nativa) | Comparable con Aer GPU |
| **Curva de aprendizaje** | Alta (restricciones del kernel) | Media |

### 8.5 Cuándo elegir CUDA-Q

CUDA-Q es la elección correcta cuando:
- Se simula más de 20 qubits en hardware NVIDIA de alto rendimiento.
- Se requiere la máxima eficiencia por operación cuántica (sin overhead de interpretación).
- Se trabaja en un entorno HPC Linux con acceso a GPU de datacenter (V100, A100, H100).
- El algoritmo es de circuito fijo (no requiere feedback clásico-cuántico).

Qiskit sigue siendo preferible cuando:
- Se desarrolla y prototipa en un entorno multiplataforma.
- Se necesitan circuitos adaptativos o variacional (QAOA, VQE con optimizadores).
- Se requiere acceso a hardware cuántico real de IBM.
- La facilidad de depuración (draw, visualización) es prioritaria.

---

## 9. Ejercicios de Consolidación

**Ejercicio 1 (Comprensión)**: Explica por qué la inversión de cadena `bitstring[::-1]` en `grover.py` es necesaria. ¿Qué estado cuántico representa `"101"` en la convención de CUDA-Q vs. en la convención de Qiskit?

**Ejercicio 2 (Análisis)**: Traza la ejecución de `grover_circuit(n=2, target=3)`. Enumera todas las puertas que se añaden al kernel (en orden). ¿Cuántas iteraciones se ejecutan?

**Ejercicio 3 (Implementación)**: El difusor en `grover.py` usa un MCZ sobre los primeros n-1 qubits como controles y el qubit n-1 como target. ¿Por qué este MCZ implementa una fase `-1` sobre `|1...1⟩`? (Pista: considera la relación entre CZ y Z.)

**Ejercicio 4 (Permutaciones)**: Para `A=2, N=5, power=1`, calcula a mano la permutación devuelta por `build_mod_exp_permutation(2, 5, 1)`. Descompónla en ciclos. ¿Cuántas transposiciones genera `controlled_swap_permutation`?

**Ejercicio 5 (Plataforma)**: Describe los pasos necesarios para ejecutar el benchmark de CUDA-Q en un sistema con GPU NVIDIA en Linux. ¿Qué target se usaría? ¿Qué variable del worker controla esto?

---

## 10. Resumen de Conceptos Clave

| Concepto | Descripción |
|----------|-------------|
| `cudaq.make_kernel()` | Crea un kernel compilado (IR cuántico), no un objeto Python dinámico |
| `kernel.qalloc(n)` | Reserva n qubits, devuelve referencias (handles al IR) |
| `cudaq.set_target(t)` | Configura el backend de ejecución globalmente en el proceso |
| `cudaq.sample(k, shots_count=N)` | Ejecuta N shots del kernel, devuelve histograma |
| `SampleResult` | Histograma de cadenas de bits a recuentos; qubit 0 = bit más a la izquierda |
| Convención MSB | CUDA-Q: qubit 0 es MSB. Qiskit: qubit 0 es LSB. Necesita inversión para comparar |
| Kernel inline | Sin composición dinámica: todo el circuito se construye en Python antes de emitir instrucciones |
| Permutación modular | Implementación de U_A en Shor: precalculada clásicamente, mapeada a transposiciones cuánticas |
| `cr1(θ, ctrl, tgt)` | Puerta R1(θ) controlada; fundamental en la QFT |
| `startup_time` | Tiempo de inicialización de cudaq.set_target(); se mide y separa del tiempo de circuito |

---

## Referencias

[1] Baydin, A.G., et al. (2024). "CUDA-Q: The Platform for Integrated Quantum-Classical Computing." *arXiv preprint arXiv:2408.01033*.

[2] Baydin, A.G., et al. (2023). "cuStateVec: A High-Performance CUDA Library for Quantum State-Vector Simulation." In *SC23: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE/ACM.

[3] Liu, Y., et al. (2021). "Closing the 'Quantum Supremacy' Gap: Achieving Real-Time Simulation of a Random Quantum Circuit Using a New Sunway Supercomputer." In *SC21: The International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE/ACM.

[4] Nielsen, M.A., & Chuang, I.L. (2010). *Quantum Computation and Quantum Information* (10th anniversary ed.). Cambridge University Press.

[5] Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond." *Quantum*, 2, 79. https://doi.org/10.22331/q-2018-08-06-79

[6] Smelyanskiy, M., Sawaya, N.P.D., & Aspuru-Guzik, A. (2016). "qHiPSTER: The Quantum High Performance Software Testing Environment." *arXiv preprint arXiv:1601.07195*.

[7] Häner, T., & Steiger, D.S. (2017). "0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit." In *SC17: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE/ACM.

[8] Gheorghiu, A., & Mosca, M. (2017). "Benchmarking the classical simulation of quantum circuits." *arXiv preprint arXiv:1704.01127*.

[9] Svore, K.M., et al. (2014). "Scalable Quantum Simulation of Molecular Energies." *Physical Review X*, 4(2), 021048. https://doi.org/10.1103/PhysRevX.4.021048

[10] Wecker, D., Hastings, M.B., Troyer, M., & Wiebe, N. (2015). "Solving Systems of Linear Equations with a Superconducting Quantum Processor." *Physical Review Letters*, 114(14), 140504. https://doi.org/10.1103/PhysRevLett.114.140504

[11] Gheorghiu, A., Mosca, M., & Zamfirescul, A. (2022). "Quantum Circuit Compiler for Linear Optical Quantum Circuits." In *2022 IEEE International Symposium on Circuits and Systems (ISCAS)* (pp. 2684–2688). IEEE.

[12] Ball, H., et al. (2021). "Software tools and algorithms for biological network analysis." *Nature Reviews Genetics*, 23(11), 689–707 (applied to quantum algorithm design patterns).
