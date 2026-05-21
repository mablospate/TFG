# Lección 01 — El Algoritmo de Grover

> *"La búsqueda cuántica no es magia: es interferencia constructiva controlada con precisión geométrica."*

Esta lección desarrolla el algoritmo de búsqueda cuántica de Lov K. Grover (1996) en el contexto del proyecto de benchmarking multi-framework de este TFG. El objetivo no es que el lector copie el código, sino que entienda por qué cada compuerta está donde está, qué ocurre en cada paso del estado cuántico, y por qué la implementación tiene la forma exacta que tiene en cada uno de los frameworks (Qiskit, Cirq, CUDA-Q, QDisLib y quantr). La lectura está calibrada para un estudiante de grado avanzado o máster en computación cuántica.

---

## Índice

1. Motivación: el problema de la búsqueda no estructurada
2. Fundamentos previos: superposición, interferencia, Hadamard, *phase kickback*
3. El algoritmo paso a paso: estado inicial, oráculo, difusor, iteraciones, medición
4. Construcción genérica del circuito: oráculo, difusor, ancillas y secuencia completa
5. Recursos del circuito: qubits, profundidad y conteo de puertas
6. Parámetros de benchmarking en este proyecto

---

## 1. Motivación

### 1.1 El problema clásico

Sea un conjunto de N elementos sin ninguna estructura interna (no están ordenados, no hay índice, no hay función hash) y una función oráculo `f : {0, 1, ..., N-1} -> {0, 1}` tal que `f(x) = 1` si y sólo si `x = ω`, donde `ω` es el elemento buscado y desconocido. La pregunta es: ¿cuántas evaluaciones de `f` se necesitan para localizar `ω`?

En el modelo clásico, no se puede hacer nada mejor que probar uno a uno. En el peor caso se requieren `N - 1` evaluaciones y, en promedio, `N/2`. La complejidad asintótica es por tanto `O(N)`. Esta cota es óptima clásicamente: cualquier algoritmo determinista o aleatorio que use sólo el oráculo necesita un número de consultas lineal en el tamaño del espacio de búsqueda.

### 1.2 La promesa cuántica

Grover demostró en 1996 que, si el oráculo se implementa como un operador unitario que se puede consultar en superposición, basta con `O(√N)` evaluaciones para encontrar `ω` con alta probabilidad. La ganancia es cuadrática: no es la ganancia exponencial de Shor, pero es genérica (aplicable a cualquier problema de búsqueda) y demostrablemente óptima dentro del modelo de consultas cuánticas (Bennett, Bernstein, Brassard, Vazirani, 1997).

| Modelo | Consultas al oráculo |
|---|---|
| Clásico determinista | `O(N)` |
| Clásico aleatorio | `O(N)` (constante mejor, mismo orden) |
| Cuántico (Grover) | `O(√N)` |
| Cota inferior cuántica | `Ω(√N)` |

En el régimen `N = 2^n` (n qubits codificando todos los índices) la diferencia es enorme: para `n = 30`, lo clásico exige unas `10^9` consultas mientras que Grover converge en `~30 000`. Por eso esta familia de algoritmos aparece como subrutina en problemas de NP (SAT, vendedor viajante con restricciones), criptoanálisis simétrico (búsqueda de claves AES, donde la ganancia cuadrática obliga a duplicar el tamaño de clave), análisis de bases de datos cuánticas y muchos otros.

### 1.3 ¿Por qué nos interesa medirlo?

En este proyecto de benchmarking, Grover es el "caballo de batalla" universal: su circuito tiene una estructura simple y predecible, su escalado teórico es perfectamente conocido, y el coste de simulación clásica crece exponencialmente con `n` (porque el simulador debe representar `2^n` amplitudes complejas). Esto lo convierte en un test ideal para comparar:

- **Tiempo de pared** entre frameworks (Qiskit, Cirq, CUDA-Q, QDisLib, quantr).
- **Memoria RSS** consumida por el simulador, que crece como `O(2^n)` bytes.
- **Profundidad de circuito** efectiva después de transpilación.
- **Calidad estadística** de la distribución medida frente a la teórica (Jensen-Shannon divergence).

---

## 2. Fundamentos previos necesarios

### 2.1 Qubits y superposición

Un qubit es un sistema cuántico de dos niveles con espacio de estados `H = C^2`, con base computacional `{|0⟩, |1⟩}`. Un estado puro genérico es

```
|ψ⟩ = α |0⟩ + β |1⟩,    α, β ∈ C,    |α|² + |β|² = 1
```

donde `|α|²` y `|β|²` son las probabilidades de obtener `0` o `1` al medir en la base computacional. Lo crucial es que `α` y `β` son números complejos: su **fase** no se observa directamente, pero sí afecta a las interferencias cuando varias amplitudes se suman.

Para `n` qubits, el espacio de estados es el producto tensorial `(C^2)^⊗n = C^(2^n)`. Un estado genérico

```
|ψ⟩ = Σ_{x=0}^{2^n - 1} c_x |x⟩,    Σ |c_x|² = 1
```

vive en una superposición de `2^n` cadenas binarias. **Esta es la fuente del paralelismo cuántico**: una sola aplicación de un operador unitario `U` actúa simultáneamente sobre las `2^n` componentes. Pero —y esto es esencial— la medición destruye la superposición y devuelve un único valor `x` con probabilidad `|c_x|²`. El reto del diseño de algoritmos cuánticos es disponer las fases de modo que, al medir, las amplitudes destructivas cancelen los estados "malos" y las constructivas refuercen los "buenos".

### 2.2 Amplitud de probabilidad e interferencia cuántica

Si dos caminos cuánticos contribuyen al mismo estado final `|x⟩` con amplitudes `a_1` y `a_2`, la amplitud total es `a_1 + a_2` y la probabilidad es `|a_1 + a_2|²`, no `|a_1|² + |a_2|²`. Si `a_1 = -a_2` (interferencia destructiva), la probabilidad se anula. Si `a_1 = a_2` (constructiva), la probabilidad se cuadruplica respecto a una sola contribución.

Grover explota esta dualidad: el oráculo introduce un signo negativo selectivamente, y el difusor amplifica esa diferencia mediante "inversión respecto a la media". El resultado es que la amplitud del estado buscado crece monótonamente con cada iteración hasta acercarse a 1, mientras que las amplitudes de los demás estados se reducen.

### 2.3 La puerta de Hadamard

La puerta de Hadamard es la pieza más importante del catálogo cuántico básico:

```
H = (1/√2) [ 1   1 ]
            [ 1  -1 ]
```

Sus propiedades clave:

- `H |0⟩ = (|0⟩ + |1⟩) / √2 = |+⟩`
- `H |1⟩ = (|0⟩ - |1⟩) / √2 = |-⟩`
- `H² = I` (es involución)
- `H` es unitaria y hermítica

Aplicada a `n` qubits simultáneamente (`H^⊗n`) sobre el estado `|0⟩^⊗n` produce la **superposición uniforme**:

```
|s⟩ = H^⊗n |0⟩^⊗n = (1/√(2^n)) Σ_{x=0}^{2^n - 1} |x⟩
```

Todas las amplitudes valen `1/√(2^n)`. Geométricamente, este estado es el "centro" del espacio de estados base; la inversión respecto a `|s⟩` es la operación geométrica fundamental del difusor.

¿Por qué crean superposición uniforme? Porque la matriz `H^⊗n` aplicada a `|00...0⟩` distribuye la amplitud entrante por los `2^n` estados base con el mismo signo (todas positivas), debido a que el `0` en cada qubit individual se mapea a `(|0⟩ + |1⟩)/√2`. Si la entrada fuera `|11...1⟩`, las amplitudes serían `(±1)/√(2^n)` con un patrón de Walsh-Hadamard que cambia de signo según la paridad de bits compartidos.

### 2.4 El concepto de *phase kickback* (inversión de fase)

Sea `U` un operador unitario con un autoestado `|ω⟩` y autovalor `-1`, es decir `U |ω⟩ = -|ω⟩`. Si aplicamos `U` a una superposición

```
|ψ⟩ = Σ_x c_x |x⟩
```

la componente `|ω⟩` recibe un cambio de signo y las demás quedan igual:

```
U |ψ⟩ = -c_ω |ω⟩ + Σ_{x ≠ ω} c_x |x⟩
```

Esto es **exactamente** lo que hace el oráculo de Grover. Conceptualmente, esta inversión de signo es una "marca" invisible (no afecta a las probabilidades directamente) pero detectable por interferencia. El término *phase kickback* viene del hecho de que, en muchas implementaciones, esta fase se "patea hacia atrás" desde un qubit auxiliar a los qubits de control de una compuerta controlada. Por ejemplo: una compuerta CZ entre un control en `|+⟩` y un target en `|-⟩` deja al target intacto pero introduce un `-1` global cuando el control está en `|1⟩`. Esa fase queda "kickback" en el control.

En la práctica, la forma estándar de implementar el oráculo de Grover es:

1. Llevar el estado `|ω⟩` a `|11...1⟩` aplicando puertas X en los qubits donde `ω` tiene un bit 0.
2. Aplicar una compuerta **multi-controlled Z (MCZ)** que invierte la fase únicamente del estado `|11...1⟩`.
3. Deshacer las X iniciales para restaurar la base computacional.

El resultado neto es `U_ω = I - 2|ω⟩⟨ω|`. Notemos que `U_ω` es hermítica y unitaria, con autovalor `-1` en el subespacio generado por `|ω⟩` y `+1` en su ortogonal.

---

## 3. El algoritmo paso a paso

### 3.1 Vista global

El algoritmo de Grover consta de cuatro fases:

```
|0⟩^n  --[H^⊗n]-->  |s⟩  --[U_ω · U_s]^k-->  |ψ_k⟩  --[medición]-->  ω (con alta prob.)
```

donde

- `|s⟩` es la superposición uniforme,
- `U_ω` es el oráculo (invierte la fase del estado buscado),
- `U_s = 2|s⟩⟨s| - I` es el difusor (inversión respecto a la media),
- `k = ⌊π/4 · √N⌋` es el número óptimo de iteraciones con `N = 2^n`.

### 3.2 Preparación del estado inicial

Partimos de `|0⟩^⊗n` y aplicamos `H` a cada qubit. En código (extracto de Qiskit):

```python
# Prepare uniform superposition
for i in range(n):
    qc.h(qr[i])
```

El estado resultante es

```
|s⟩ = (1/√N) Σ_{x=0}^{N-1} |x⟩
```

donde `N = 2^n`. Geométricamente, este estado tiene **un solapamiento idéntico con todos los estados base**: `⟨x|s⟩ = 1/√N` para todo `x`. En particular, `⟨ω|s⟩ = 1/√N`, lo que significa que si midiéramos en este punto la probabilidad de obtener `ω` sería `1/N` — exactamente lo que ofrece la búsqueda aleatoria clásica.

### 3.3 El oráculo `U_ω`

#### 3.3.1 Definición matemática

```
U_ω |x⟩ = (-1)^{f(x)} |x⟩
```

donde `f(x) = 1` si `x = ω` y `0` en otro caso. Matricialmente,

```
U_ω = I - 2 |ω⟩⟨ω|
```

Esto es una reflexión a través del hiperplano ortogonal a `|ω⟩`. Aplicado a la superposición uniforme:

```
U_ω |s⟩ = (1/√N) [ Σ_{x ≠ ω} |x⟩  -  |ω⟩ ]
```

La probabilidad de medir `ω` **no ha cambiado** (sigue siendo `1/N`): la fase no es observable por sí sola. Lo crítico viene en el siguiente paso, cuando el difusor convierte esa diferencia de fase en una diferencia de amplitud.

#### 3.3.2 Oráculo de fase vs. oráculo de bit-flip

Hay dos formas estándar de codificar un oráculo:

- **Oráculo de fase**: `|x⟩ -> (-1)^{f(x)} |x⟩`. Es lo que Grover necesita directamente.
- **Oráculo de bit-flip (XOR)**: `|x⟩|y⟩ -> |x⟩|y ⊕ f(x)⟩` sobre un qubit auxiliar `|y⟩`.

Ambos son equivalentes: si inicializas el qubit auxiliar en `|-⟩ = (|0⟩ - |1⟩)/√2`, el bit-flip provoca *phase kickback* y el efecto global es exactamente el oráculo de fase, sin que el auxiliar se entrelace con el resto al final (queda factorizado en `|-⟩`).

En este proyecto **todos los frameworks implementan directamente el oráculo de fase** mediante una compuerta MCZ rodeada de puertas X, evitando el qubit auxiliar para el oráculo. La excepción es la decomposición interna de la MCZ en quantr, donde sí aparecen ancillas para descomponer la compuerta multi-controlada (lo veremos en §4.3).

#### 3.3.3 Construcción concreta del oráculo

El truco estándar consiste en transformar el problema "marcar fase de `|ω⟩`" en "marcar fase de `|11...1⟩`". Si el bit `i` de `ω` vale 0, se inserta una X en el qubit `i`: eso convierte `|ω⟩` en `|11...1⟩` y, recíprocamente, `|11...1⟩` en `|ω⟩`. Tras aplicar una MCZ —que sólo distingue `|11...1⟩`— se deshacen las X para restaurar la base original.

Extracto de Qiskit:

```python
def build_oracle(n: int, target: int) -> QuantumCircuit:
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # Flip qubits where target has a 0 bit, so |target> maps to |11...1>
    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    # Multi-controlled Z: flips phase of |11...1>
    qc.append(ZGate().control(n - 1), qr[:])

    # Undo the X flips
    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    return qc
```

Y la versión Cirq, idéntica salvo por el orden big-endian de qubits que obliga a invertir el índice:

```python
# Cirq uses big-endian ordering: LineQubit(0) is the MSB when measured.
# Bit i of target maps to qubit (n-1-i) so the measurement reads target.
for i in range(n):
    if not (target >> i) & 1:
        circuit.append(cirq.X(qubits[n - 1 - i]))

mcz = cirq.Z.controlled(num_controls=n - 1)
circuit.append(mcz.on(*qubits))
```

Esta sutileza de ordenamiento es relevante: Qiskit y CUDA-Q (tras invertir la cadena) son LSB-first, mientras que Cirq y quantr son MSB-first. La capa de benchmarking unifica todo a LSB-first en las distribuciones finales.

### 3.4 El operador difusor `U_s`

#### 3.4.1 Definición y descomposición

```
U_s = 2 |s⟩⟨s| - I
```

Geométricamente: `U_s` refleja cualquier vector a través de `|s⟩`. Si un vector está casi alineado con `|s⟩`, queda casi igual; si está casi ortogonal, sufre prácticamente una inversión de signo.

La descomposición práctica usa que `H^⊗n |0⟩^⊗n = |s⟩`, lo que implica `H^⊗n (2|0⟩⟨0| - I) H^⊗n = 2|s⟩⟨s| - I` (porque `H^⊗n` es autoinverso). Así:

```
U_s = H^⊗n · (2|0⟩⟨0| - I) · H^⊗n
```

Y `(2|0⟩⟨0| - I)` es una inversión de fase sobre `|0⟩^⊗n` que se implementa con la receta dual del oráculo: aplicar X a todos los qubits para llevar `|0...0⟩` a `|1...1⟩`, hacer una MCZ y deshacer las X. Sumando todo, el difusor es:

```
H^⊗n - X^⊗n - MCZ - X^⊗n - H^⊗n
```

Extracto de Qiskit:

```python
def build_diffuser(n: int) -> QuantumCircuit:
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # H on all qubits
    for i in range(n):
        qc.h(qr[i])

    # Phase flip on |00...0>: X → MCZ → X
    for i in range(n):
        qc.x(qr[i])

    qc.append(ZGate().control(n - 1), qr[:])

    for i in range(n):
        qc.x(qr[i])

    # H on all qubits
    for i in range(n):
        qc.h(qr[i])

    return qc
```

Obsérvese que el cuerpo central del difusor (entre las dos capas de H) es **idéntico estructuralmente al oráculo del estado `|0...0⟩`**. Es decir, el difusor es "el oráculo de cero" sándwich entre dos capas de Hadamard.

#### 3.4.2 Inversión respecto a la media (visión geométrica clásica)

Esta es la interpretación pedagógicamente más útil. Sea `|ψ⟩ = Σ c_x |x⟩` un estado arbitrario y `μ = (1/N) Σ c_x` el valor medio de las amplitudes. Aplicar `U_s` produce un nuevo estado con amplitudes

```
c_x' = 2μ - c_x
```

Cada amplitud se refleja en torno a la media. Si una amplitud está por encima de la media, baja; si está por debajo, sube. Si una amplitud es **negativa** mientras las demás son positivas (justo el caso tras aplicar el oráculo a `|s⟩`), la inversión convierte ese signo negativo en una amplitud muy positiva y reduce ligeramente las otras. Eso es exactamente la amplificación de Grover.

Numéricamente, tras un oráculo aplicado a `|s⟩`:

```
c_x = 1/√N para x ≠ ω,   c_ω = -1/√N
μ = (N-1) · (1/√N)/N - (1/√N)/N = (N-2)/(N √N) ≈ 1/√N para N grande
c_ω' = 2μ - c_ω ≈ 2/√N + 1/√N = 3/√N
```

Tras una iteración, la amplitud de `ω` ha pasado de `1/√N` a aproximadamente `3/√N`, **triplicada en una sola consulta**. Esto es la firma cuadrática del algoritmo.

### 3.5 Iteraciones: el famoso `⌊π/4 · √N⌋`

#### 3.5.1 El plano de Grover

Sea `|ω⟩` el estado marcado y `|s'⟩` la superposición uniforme de los estados **no marcados**:

```
|s'⟩ = (1/√(N-1)) Σ_{x ≠ ω} |x⟩
```

Estos dos vectores son ortonormales y generan un subespacio bidimensional invariante bajo `U_ω` y `U_s`. El estado inicial `|s⟩` vive en este plano:

```
|s⟩ = sin(θ) |ω⟩ + cos(θ) |s'⟩,    con sin(θ) = 1/√N
```

Para `N` grande, `θ ≈ 1/√N`.

#### 3.5.2 Las reflexiones

- `U_ω` es la reflexión respecto al eje `|s'⟩` (cambia el signo de la componente `|ω⟩`).
- `U_s` es la reflexión respecto al vector `|s⟩`.

La composición de dos reflexiones en un plano es una **rotación**: el ángulo total de rotación es `2θ` por cada iteración de Grover (`U_s U_ω`). Tras `k` iteraciones, el estado es

```
|ψ_k⟩ = sin((2k+1) θ) |ω⟩ + cos((2k+1) θ) |s'⟩
```

#### 3.5.3 El número óptimo

Queremos `sin((2k+1) θ) ≈ 1`, es decir `(2k+1) θ ≈ π/2`. Despejando,

```
k_opt = (π / (4 θ)) - 1/2 ≈ (π/4) √N
```

Como `k` ha de ser entero,

```
k = ⌊π/4 · √N⌋
```

En código (común a todos los frameworks):

```python
num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))
```

Y en Rust:

```rust
let iterations = num_iterations.unwrap_or_else(|| {
    let n_states = (1u64 << n) as f64;
    ((std::f64::consts::PI / 4.0) * n_states.sqrt()).floor() as usize
});
```

Para `n` pequeños la fórmula no es estrictamente óptima (el ángulo `θ` deja de ser pequeño y la aproximación se degrada), pero sigue dando probabilidad de acierto > 90% en todos los casos del benchmark.

#### 3.5.4 ¡Cuidado con sobre-iterar!

La rotación es periódica: si se aplican demasiadas iteraciones, el estado se pasa de `|ω⟩` y la probabilidad vuelve a bajar. Esto se llama "sobre-rotación" y es una propiedad muchas veces sorprendente: en Grover, **hacer más cuesta más y empeora el resultado**.

### 3.6 Medición final

Se mide en la base computacional. El resultado más frecuente, con probabilidad

```
P(ω) = sin²((2k+1) θ) ≈ 1 - O(1/N)
```

es exactamente `ω`. En código:

```python
qc.measure(qr, cr)
```

```python
result = simulator.run(qc, repetitions=num_shots)
histogram = result.histogram(key="result")
```

Y en quantr:

```rust
let sim = qc.simulate();
let counts = match sim.measure_all(args.shots) {
    Measurement::Observable(c) => c,
    Measurement::NonObservable(c) => c,
};
```

La función `search` de cada framework devuelve `(found, dist)` donde `found` es el modo de la distribución (el resultado más frecuente como entero) y `dist` es el diccionario `{bitstring -> count}` completo.

---

## 4. Construcción del circuito en código (modelo genérico)

Esta sección reconstruye el circuito desde primeros principios siguiendo las convenciones reales de los archivos del proyecto.

### 4.1 `build_oracle(n, target)`

**Entrada**: número de qubits `n` y entero `target` en `[0, 2^n)`.
**Salida**: subcircuito que aplica `U_target = I - 2|target⟩⟨target|`.

**Pasos**:

1. *Asegurar el rango*: `assert 0 <= target < 2**n`. Si `target` está fuera de rango es un error de programación, no algorítmico.
2. *Convertir `target` a binario y aplicar X* a los qubits donde el bit correspondiente vale 0. Tras este paso, el estado `|target⟩` en la base original corresponde al estado `|11...1⟩` en la nueva base.
3. *Aplicar MCZ* (Z multi-controlada con `n-1` controles y 1 target, todos sobre el mismo registro). Esto invierte la fase de `|11...1⟩` y deja todos los demás estados invariantes.
4. *Deshacer las X* del paso 2 para volver a la base original.

El detalle más sutil es el orden bit-a-bit. La línea

```python
if not (target >> i) & 1:
    qc.x(qr[i])
```

aplica X al qubit `i` si el bit `i` de `target` (LSB = bit 0) es cero. Esto fija la convención **LSB-first** en Qiskit, CUDA-Q y quantr. En Cirq, la línea equivalente es

```python
circuit.append(cirq.X(qubits[n - 1 - i]))
```

porque Cirq mide en orden big-endian (qubit 0 = MSB).

### 4.2 `build_diffuser(n)`

**Entrada**: número de qubits `n`.
**Salida**: subcircuito que aplica `U_s = 2|s⟩⟨s| - I`.

**Pasos**:

1. `H` sobre todos los qubits.
2. `X` sobre todos los qubits.
3. MCZ multi-controlada.
4. `X` sobre todos los qubits.
5. `H` sobre todos los qubits.

El paso central (X-MCZ-X) implementa la inversión de fase respecto a `|0...0⟩`, y las capas de Hadamard la conjugan para convertirla en inversión respecto a `|s⟩`. La simetría del operador es perfecta: aplicar el difusor dos veces equivale a la identidad (`U_s` es involución).

### 4.3 La compuerta MCZ y la necesidad de ancillas

La compuerta Z multi-controlada con `n-1` controles **no es nativa** en la mayoría de hardware ni en muchos backends de simulación. Hay tres estrategias para implementarla:

#### Estrategia A: MCZ nativa (Qiskit, Cirq, CUDA-Q)

Qiskit ofrece `ZGate().control(n-1)` y deja al transpilador la responsabilidad de descomponerla. Cirq usa `cirq.Z.controlled(num_controls=n-1)`. CUDA-Q usa `kernel.cz(controls, target)` con una lista de controles. En los tres casos, la descomposición efectiva al transpilar suele seguir la receta de Barenco et al. (1995): cadena de Toffoli con qubits auxiliares ó descomposición ABA con rotaciones controladas.

```python
# Qiskit
qc.append(ZGate().control(n - 1), qr[:])
```

```python
# Cirq
mcz = cirq.Z.controlled(num_controls=n - 1)
circuit.append(mcz.on(*qubits))
```

```python
# CUDA-Q
if n == 1:
    kernel.z(qubits[0])
else:
    controls = [qubits[i] for i in range(n - 1)]
    kernel.cz(controls, qubits[n - 1])
```

Notemos el caso particular `n == 1` en CUDA-Q: con un solo qubit no hay controles, así que la MCZ se reduce a una Z simple.

#### Estrategia B: ancillas + Toffoli ladder (quantr)

quantr es el único framework del proyecto que **no expone una MCZ nativa**: sólo dispone de `Gate::Toffoli(c1, c2)` (dos controles). Por tanto, para `n ≥ 3` controles se necesita una descomposición manual con qubits auxiliares ("ancillas") y una escalera de Toffoli. La idea es:

1. Calcular el AND lógico de los primeros dos controles en `ancillas[0]`.
2. Calcular el AND de `ancillas[0]` con el tercer control en `ancillas[1]`.
3. Repetir hasta `ancillas[k-3]` que contiene el AND de los primeros `k-1` controles.
4. Aplicar una Toffoli final con `(controls[k-1], ancillas[k-3])` como controles y el target como objetivo.
5. **Descomputar** la escalera en orden inverso para devolver todas las ancillas a `|0⟩`.

Este patrón se llama "ladder de Toffoli con uncomputación" y necesita exactamente `k - 2` ancillas para `k` controles. La uncomputación es esencial: si los auxiliares quedaran entrelazados con el registro principal, contaminarían el resultado al medir. Tras la descomputación, las ancillas se factorizan en `|0⟩` y se pueden ignorar (o reutilizar).

```rust
pub fn add_mcx(
    qc: &mut Circuit,
    controls: &[usize],
    target: usize,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let k = controls.len();
    match k {
        0 => qc.add_gate(Gate::X, target)?,
        1 => qc.add_gate(Gate::CNot(controls[0]), target)?,
        2 => qc.add_gate(Gate::Toffoli(controls[0], controls[1]), target)?,
        _ => {
            // Forward ladder: compute the AND of all controls into ancilla[k-3].
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
            for i in 2..(k - 1) {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i - 2]), ancillas[i - 1])?;
            }
            // Central gate flips the target if the cumulative AND is 1.
            qc.add_gate(Gate::Toffoli(controls[k - 1], ancillas[k - 3]), target)?;
            // Reverse ladder uncomputes the ancillas back to |0>.
            for i in (2..(k - 1)).rev() {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i - 2]), ancillas[i - 1])?;
            }
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
        }
    }
    Ok(())
}
```

Para convertir MCX en MCZ basta con envolver el target en `H`:

```rust
pub fn add_mcz(qc: &mut Circuit, n: usize, ancillas: &[usize]) -> Result<(), QuantrError> {
    match n {
        0 => {}
        1 => qc.add_gate(Gate::Z, 0)?,
        2 => qc.add_gate(Gate::CZ(0), 1)?,
        _ => {
            let target = n - 1;
            let controls: Vec<usize> = (0..target).collect();
            qc.add_gate(Gate::H, target)?;
            add_mcx(qc, &controls, target, ancillas)?;
            qc.add_gate(Gate::H, target)?;
        }
    }
    Ok(())
}
```

La identidad `MCZ = H · MCX · H` aplicada al target sólo funciona porque `H Z H = X` y `H X H = Z`. Lo único que cambia es la base del target; los controles no se tocan.

#### Estrategia C: delegar al transpilador (QDisLib)

QDisLib reutiliza el circuito de Qiskit y lo procesa con su pipeline de *circuit cutting*. La descomposición de la MCZ no se redefine: se hereda de Qiskit. Esto es coherente con el espíritu del framework, que opera a nivel de "cortar" un circuito grande en subcircuitos más pequeños ejecutables en hardware (o simuladores) con menos qubits.

```python
build_oracle: callable = _qiskit_build_oracle
build_diffuser: callable = _qiskit_build_diffuser
grover_circuit: callable = _qiskit_grover_circuit
```

### 4.4 Por qué hacen falta ancillas (la versión conceptual)

La razón es topológica más que algorítmica: una compuerta multi-controlada con `k > 2` controles tiene un soporte sobre `k+1` qubits, pero las puertas físicamente realizables suelen actuar sobre 1 o 2 qubits a la vez. La única manera de "fabricar" un AND lógico de varios bits sin perder reversibilidad es **almacenar resultados intermedios** en qubits adicionales (ancillas) y luego **borrarlos sin medir** (descomputación). Sin ancillas, no hay forma reversible y unitaria de implementar la MCZ con sólo Toffolis.

Esta es una manifestación de un teorema más general (Bennett, 1973): cualquier computación reversible puede simularse con un coste constante en ancillas si se permite descomputación. En Grover, ese coste es lineal en `n`: `n - 2` ancillas adicionales para `n ≥ 3` qubits de búsqueda.

### 4.5 La secuencia completa

```python
def grover_circuit(n, target, num_iterations=None):
    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="result")
    qc = QuantumCircuit(qr, cr)

    # 1. Prepare uniform superposition
    for i in range(n):
        qc.h(qr[i])

    # 2. Grover iterations: oracle + diffuser
    oracle = build_oracle(n, target)
    diffuser = build_diffuser(n)
    for _ in range(num_iterations):
        qc.compose(oracle, qubits=qr, inplace=True)
        qc.compose(diffuser, qubits=qr, inplace=True)

    # 3. Measure
    qc.measure(qr, cr)
    return qc
```

En CUDA-Q, dado que el modelo de "kernel" no permite componer subkernels dinámicamente con facilidad, el circuito completo se "desenrolla" en un único kernel:

```python
def grover_circuit(n, target, num_iterations=None):
    if num_iterations is None:
        num_iterations = math.floor(math.pi / 4 * math.sqrt(2**n))

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n)

    # Prepare uniform superposition
    for i in range(n):
        kernel.h(qubits[i])

    # Grover iterations: oracle + diffuser
    for _ in range(num_iterations):
        # --- Oracle: flip phase of |target> ---
        for i in range(n):
            if not (target >> i) & 1:
                kernel.x(qubits[i])
        if n == 1:
            kernel.z(qubits[0])
        else:
            controls = [qubits[i] for i in range(n - 1)]
            kernel.cz(controls, qubits[n - 1])
        for i in range(n):
            if not (target >> i) & 1:
                kernel.x(qubits[i])

        # --- Diffuser: inversion about the mean ---
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

    # Measure all qubits
    kernel.mz(qubits)
    return kernel
```

La equivalencia semántica con la versión Qiskit es total; lo único que cambia es la API.

Y en quantr, la versión más explícita con ancillas:

```rust
pub fn grover_circuit(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
) -> Result<Circuit, QuantrError> {
    let iterations = num_iterations.unwrap_or_else(|| {
        let n_states = (1u64 << n) as f64;
        ((std::f64::consts::PI / 4.0) * n_states.sqrt()).floor() as usize
    });

    let n_anc = if n >= 3 { n - 2 } else { 0 };
    let total = n + n_anc;
    let ancillas: Vec<usize> = (n..total).collect();

    let mut qc = Circuit::new(total)?;
    let search: Vec<usize> = (0..n).collect();
    qc.add_repeating_gate(Gate::H, &search)?;
    for _ in 0..iterations {
        build_oracle(&mut qc, n, target, &ancillas)?;
        build_diffuser(&mut qc, n, &ancillas)?;
    }
    Ok(qc)
}
```

Obsérvese que `total = n + n_anc` con `n_anc = n - 2` para `n ≥ 3`. El registro de búsqueda son los qubits `0..n` y las ancillas viven en `n..total`.

---

## 5. Recursos del circuito

### 5.1 Número de qubits

| Caso | Qubits de búsqueda | Ancillas | Total |
|---|---|---|---|
| Qiskit / Cirq / CUDA-Q / QDisLib | `n` | 0 (gestionadas por el transpilador) | `n` |
| quantr (`n` ≥ 3) | `n` | `n - 2` | `2n - 2` |

En los frameworks que ofrecen MCZ nativa, las ancillas pueden existir internamente en el código del backend, pero no se cuentan en la memoria del *frontend*. En quantr son explícitas.

### 5.2 Profundidad del circuito

Cada iteración de Grover contiene:

- 1 capa de X (parcial, sólo en bits 0 del target) — profundidad 1.
- 1 MCZ — profundidad `O(n)` con descomposición Toffoli ladder o similar.
- 1 capa de X (deshacer) — profundidad 1.
- 1 capa H — profundidad 1.
- 1 capa X — profundidad 1.
- 1 MCZ del difusor — profundidad `O(n)`.
- 1 capa X — profundidad 1.
- 1 capa H — profundidad 1.

Total por iteración: `O(n)`. Con `k = O(√N) = O(√(2^n))` iteraciones, la profundidad total es

```
D_total = O(n · √(2^n)) = O(n · 2^(n/2))
```

### 5.3 Conteo de puertas

Si llamamos `G` al coste de una MCZ (típicamente `Θ(n)` Toffolis con descomposición tradicional, o `Θ(n²)` sin ancillas) y `g_1` al coste de las puertas de un solo qubit, por iteración hay:

- `2 · (n_zeros)` puertas X (donde `n_zeros` es el número de bits 0 en `target`).
- `2 · n` puertas X en el difusor.
- `2 · n` puertas H en el difusor.
- `2 · G` puertas para las dos MCZ.

Esto da `O(n + G)` puertas por iteración, y por tanto

```
G_total = O((n + G) · √(2^n))
```

Para la descomposición Toffoli ladder de quantr, `G = O(n)` (cada MCZ son `2(n-1) - 3 = 2n - 5` Toffolis aproximadamente), así que el coste asintótico de puertas físicas es `O(n · 2^(n/2))`.

### 5.4 Coste de simulación clásica

El simulador cuántico clásico debe representar `2^n` amplitudes complejas (16 bytes cada una en doble precisión compleja), lo que implica:

```
memoria = 16 · 2^n  bytes ≈ 16 GB para n = 30
```

Y cada aplicación de puerta cuesta `O(2^n)` operaciones para puertas de un solo qubit y `O(2^n)` para CNOT. La MCZ multi-controlada también escala como `O(2^n)`. Combinado con la profundidad `O(n √(2^n))`, el coste total es

```
T_sim = O(n · 2^(3n/2))
```

Para `n = 11`: `n · 2^(3n/2) ≈ 11 · 90 510 ≈ 10^6` operaciones de simulación: trivial. Para `n = 30` ya estaríamos en `10^14`. Por eso el benchmark se queda en `n ≤ 11`.

---

## 6. Parámetros de benchmarking en este proyecto

Estos parámetros viven en `benchmark_core.py` y son comunes a todos los frameworks.

### 6.1 `n_values = [3, 5, 7, 9, 11]`

```python
n_values: list[int] = field(
    default_factory=lambda: [3, 5, 7, 9, 11]
)  # Tamaños de n para escalabilidad
```

La elección está motivada por:

- **`n = 3` (N = 8)**: caso mínimo no trivial. El óptimo `k = ⌊π/4 · √8⌋ = 2`. Permite verificar correctness con un único shot dominante.
- **`n = 5` (N = 32)**: `k = 4`. Ya supera el régimen "small". La memoria del simulador es despreciable.
- **`n = 7` (N = 128)**: `k = 8`. Régimen donde la descomposición de la MCZ empieza a dominar la profundidad.
- **`n = 9` (N = 512)**: `k = 17`. Suficiente para extraer una ley de potencia limpia en el ajuste de escalabilidad.
- **`n = 11` (N = 2048)**: `k = 35`. Tope superior antes de que la simulación clásica empiece a sufrir en máquinas modestas.

La progresión impar tiene una razón sutil: con `n` impar, el número óptimo de iteraciones suele coincidir más limpiamente con la aproximación `π/4 · √N` sin sobre-rotación. Para `n` par, a veces hay un desajuste de ~1 iteración que reduce la probabilidad de acierto al 90-95 % en lugar del 100 %.

Cinco puntos en escala geométrica (`+2` por punto) permiten ajustar dos parámetros (`α`, `β`) de la ley exponencial con tres grados de libertad sobrantes, lo cual da estabilidad estadística.

### 6.2 `num_shots = 1024`

```python
num_shots: int = 1024  # Shots para distribución empírica
```

Es el número de veces que se ejecuta el circuito para construir la distribución empírica. `1024 = 2^10` es la convención de facto en el ecosistema de IBM Quantum y se ha estandarizado en el resto. Estadísticamente, con un acierto teórico `P ≈ 1 - 1/N`, el número esperado de aciertos es `~1024 · (1 - 1/N)` y la desviación estándar binomial es `√(1024 · (1-1/N) · 1/N) ≈ √(1024/N)`, que se mantiene pequeña frente a la media. Esto basta para resolver con holgura los modos de la distribución y para estimar la Jensen-Shannon divergence frente a la distribución teórica con error < 1 %.

### 6.3 La función de escalado `t(n) = α · 2^(β·n)`

En `BenchmarkResult` aparecen dos campos:

```python
scaling_alpha: float = 0.0  # Coeficiente α en α·2^(β·n)
scaling_beta: float = 0.0  # Exponente β en α·2^(β·n)
```

La idea: el tiempo de ejecución empírico (en milisegundos) se ajusta a una ley de potencia exponencial en `n`:

```
t(n) = α · 2^(β · n)
```

donde:

- `α` (ms): "constante de fricción" del framework. Captura overheads de startup, compilación, transpilación y dispatch que no dependen de `n`.
- `β` (adimensional): exponente de escalabilidad. Su valor teórico ideal para una simulación tipo state-vector de Grover es **`β ≈ 1.5`**, correspondiente al `O(2^(3n/2))` derivado en §5.4. Valores más altos indican overheads cuadráticos o cúbicos en `n` que no se ven compensados por optimizaciones; valores más bajos indican que el framework está aprovechando paralelismo, caché o reducciones (matrix-product-states, *circuit cutting* en QDisLib, GPU en CUDA-Q).

El ajuste se hace con `scipy.optimize.curve_fit`:

```python
from scipy.optimize import curve_fit
```

sobre la tupla `(n_values, median_times_ms)`. El logaritmo de `t(n)` es lineal en `n`:

```
log₂(t) = log₂(α) + β · n
```

así que un ajuste por mínimos cuadrados en escala log-lineal recupera `α` y `β` con varianza pequeña.

### 6.4 Otras métricas relevantes

- `wall_time_median_ms` y `wall_time_iqr_ms`: mediana y rango intercuartílico del tiempo total, robustos frente a outliers (preferidos sobre la media por la presencia de GC pauses, jitter de SO, etc.).
- `peak_memory_rss_mb`: pico de RSS medido vía `psutil` (frameworks Python) o `/proc/self/status` (quantr).
- `cv`: coeficiente de variación `σ/μ` sobre `n_repetitions` ejecuciones (`30` por defecto). Permite detectar inestabilidad de medición.
- `jsd`: Jensen-Shannon divergence entre la distribución empírica y la teórica. Para Grover, la distribución teórica es `P(x) = sin²((2k+1)θ)` si `x = ω` y `P(x) = cos²((2k+1)θ)/(N-1)` en otro caso. Una JSD baja (<0.01) indica que el simulador es estadísticamente fiel.
- `simulation_time_ms`, `build_time_ms`, `startup_time_ms`: desglose temporal interno, útil para separar el coste de construir el circuito (que no escala exponencialmente) del coste de simularlo.

### 6.5 Interpretación de los resultados esperados

Para Grover en este benchmark:

- **β esperada**: alrededor de 1.5 para state-vector puro; alrededor de 1.0-1.3 para frameworks con optimizaciones (cuTensorNet, MPS); alrededor de 1.5-1.7 para implementaciones sin optimizaciones (quantr en CPU single-thread).
- **α esperada**: < 1 ms para frameworks ligeros (Cirq, quantr); 10-100 ms para frameworks pesados (Qiskit con transpilación completa); >100 ms para CUDA-Q con kernel JIT en primera invocación (la *warmup_run* lo absorbe).
- **JSD**: < 0.01 en todos los casos correctos.
- **Probabilidad de acierto del modo**: > 95 % para `n ≥ 5`; cercana a 1.0 para `n` impar y >100 shots.

---

## Apéndice A: Ecuaciones de cierre

### A.1 Estado tras `k` iteraciones

```
|ψ_k⟩ = sin((2k+1) θ) |ω⟩ + cos((2k+1) θ) |s'⟩,    sin θ = 1/√N
```

### A.2 Probabilidad de éxito

```
P_éxito(k) = sin²((2k+1) θ)
```

### A.3 Número óptimo de iteraciones

```
k_opt = ⌊(π / (4 θ)) - 1/2⌋ ≈ ⌊(π/4) √N⌋
```

### A.4 Probabilidad para `k_opt`

```
P_éxito(k_opt) ≥ 1 - 1/N
```

### A.5 Identidades operatoriales útiles

```
H X H = Z
H Z H = X
U_s = H^⊗n (2|0⟩⟨0| - I) H^⊗n
U_s U_ω = rotación de ángulo 2θ en el plano de Grover
```

---

## Apéndice B: Comparativa rápida entre frameworks

| Framework | Lenguaje | MCZ | Ancillas explícitas | Convención bits |
|---|---|---|---|---|
| Qiskit | Python | `ZGate().control(n-1)` | No (transpilador) | LSB-first |
| Cirq | Python | `cirq.Z.controlled(n-1)` | No (transpilador) | MSB-first (LineQubit(0) = MSB) |
| CUDA-Q | Python (kernel MLIR) | `kernel.cz(controls, target)` | No | MSB-first; se invierte a LSB en post |
| QDisLib | Python (sobre Qiskit) | Heredada de Qiskit | No | LSB-first |
| quantr | Rust | Descomposición H-MCX-H con Toffoli ladder | Sí, `n - 2` para `n ≥ 3` | MSB-first; se invierte a LSB en post |

Esta tabla resume las decisiones de implementación que afectan a la comparación directa: si bien el algoritmo es idéntico semánticamente, las constantes de profundidad y el número de puertas físicas pueden variar entre frameworks por estas elecciones, y eso impacta directamente en los exponentes `β` medidos.

---

## Apéndice C: Mapa de archivos del proyecto relevantes para Grover

- `python/qiskit/grover.py` — implementación de referencia, modular en `build_oracle`, `build_diffuser`, `grover_circuit`, `search`.
- `python/cirq/grover.py` — equivalente en Cirq con manejo explícito de la convención big-endian.
- `python/cudaq/grover.py` — versión inline en un único kernel MLIR, requerida por la API de CUDA-Q.
- `python/qdislib/grover.py` — wrapper sobre Qiskit con fallback a Aer cuando QDisLib no está disponible; incluye `search_with_cutting` para *circuit cutting* explícito.
- `rust/quantr/src/grover.rs` — implementación más explícita, con descomposición manual de MCZ a Toffolis y ancillas; incluye CLI y emisión JSON.
- `python/benchmark_core.py` — define `BenchmarkConfig`, `BenchmarkResult`, parámetros globales (`n_values`, `num_shots`) y el ajuste de escalado `α · 2^(β·n)`.

---

## Apéndice D: Lecturas recomendadas

1. Grover, L. K. (1996). *A fast quantum mechanical algorithm for database search*. Proceedings of the 28th Annual ACM Symposium on Theory of Computing.
2. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*, 10th anniversary edition, Cambridge University Press. Capítulo 6.
3. Bennett, C. H., Bernstein, E., Brassard, G., & Vazirani, U. (1997). *Strengths and weaknesses of quantum computing*. SIAM Journal on Computing 26(5).
4. Barenco, A., Bennett, C. H., Cleve, R., DiVincenzo, D. P., Margolus, N., Shor, P., Sleator, T., Smolin, J. A., & Weinfurter, H. (1995). *Elementary gates for quantum computation*. Physical Review A 52(5).
5. Boyer, M., Brassard, G., Høyer, P., & Tapp, A. (1998). *Tight bounds on quantum searching*. Fortschritte der Physik 46(4-5).

---

*Fin de la Lección 01.*
