# Algoritmo de Shor — Factorización Cuántica

> Lección universitaria autocontenida. Su objetivo es que, tras leerla,
> el lector pueda **reproducir la implementación** del algoritmo de Shor
> en Qiskit (y entender la variante en Cirq) sin necesidad de consultar
> ningún otro material. Está pensada para alumnos de máster o doctorado
> con conocimientos básicos de mecánica cuántica, álgebra lineal y
> teoría de números.

---

## 1. Motivación

### 1.1. El problema de la factorización

Dado un entero compuesto `N`, el problema de la factorización entera
consiste en encontrar un divisor no trivial `d` tal que
`1 < d < N` y `d | N`. A pesar de ser un problema con un enunciado
extraordinariamente simple, no se conoce ningún algoritmo **clásico**
que lo resuelva en tiempo polinómico respecto a `log N` (el número de
bits necesarios para escribir `N`).

El mejor algoritmo clásico conocido para enteros generales es el
**General Number Field Sieve** (GNFS), con una complejidad
sub-exponencial:

```
L_N[1/3, c] = exp( (c + o(1)) · (log N)^(1/3) · (log log N)^(2/3) )
```

con `c ≈ 1.923`. Esto significa que duplicar el tamaño en bits de `N`
no duplica el coste, pero crece mucho más despacio que `2^(log N) = N`.
Aun así, factorizar un módulo RSA de 2048 bits sigue siendo
**intratable** con la tecnología clásica actual.

### 1.2. RSA y la criptografía de clave pública

La importancia práctica de la factorización viene determinada por el
protocolo **RSA**. Su esquema, en versión esquemática, es:

1. Se eligen dos primos grandes `p` y `q`.
2. Se publica `N = p · q` y un exponente público `e`.
3. La clave privada `d` se calcula como el inverso de `e` módulo
   `φ(N) = (p-1)(q-1)`.
4. Cifrar un mensaje `m`: `c = m^e mod N`.
5. Descifrar: `m = c^d mod N`.

La seguridad de RSA descansa enteramente sobre la dificultad de
**factorizar `N`**: quien pueda recuperar `p` y `q` a partir de `N`
calcula `φ(N)`, invierte `e` y obtiene `d`. La criptografía moderna
basada en clave pública (RSA, DSA con primos seguros, etc.) se hunde
si la factorización deja de ser difícil.

### 1.3. Complejidad cuántica

En 1994 Peter Shor descubrió un algoritmo **cuántico** que resuelve la
factorización con complejidad:

```
O( (log N)^3 )   en puertas elementales
O( (log N)^2 )   en profundidad si se permite paralelización
```

Para un `N` de 2048 bits, el coste deja de ser astronómico y pasa a
ser polinómico. Esa diferencia exponencial es lo que convierte al
algoritmo de Shor en la **amenaza cuántica fundamental** para la
criptografía clásica y la motivación principal del campo de la
criptografía post-cuántica.

| Algoritmo | Coste asintótico |
|-----------|------------------|
| GNFS (clásico) | sub-exponencial en `log N` |
| Shor (cuántico) | polinómico en `log N`, `O((log N)^3)` |

---

## 2. Reducción de la factorización al *Order Finding*

El núcleo cuántico del algoritmo de Shor **no factoriza directamente**:
encuentra el **orden** de un elemento en un grupo multiplicativo
modular. La factorización es una consecuencia clásica de ese cálculo.

### 2.1. ¿Qué es el orden de un elemento?

Sea `Z_N* = { a ∈ {1, ..., N-1} : gcd(a, N) = 1 }` el grupo
multiplicativo de unidades módulo `N`. Para cada `a ∈ Z_N*`, el
**orden** de `a` es el menor entero positivo `r` tal que:

```
a^r ≡ 1   (mod N)
```

Ese `r` existe porque `Z_N*` es un grupo finito (de orden `φ(N)`) y
por el teorema de Lagrange `r` divide a `φ(N)`. Determinar `r`
clásicamente es esencialmente equivalente a calcular el logaritmo
discreto, problema considerado igual de difícil que la factorización.

### 2.2. De *order finding* a factorización

La reducción matemática completa funciona así:

1. **Elegir `a`** aleatorio en `{2, ..., N-1}`.
2. **Calcular `d = gcd(a, N)`**. Si `d > 1`, ya hemos encontrado un
   factor no trivial (caso "lucky guess").
3. Si `d = 1`, calcular **cuánticamente** el orden `r` de `a` módulo
   `N`.
4. Si `r` es **impar**, descartar y empezar con otro `a`.
5. Si `a^(r/2) ≡ -1 (mod N)`, descartar y empezar con otro `a`.
6. Si `r` es par y `a^(r/2) ≢ -1 (mod N)`, entonces:

```
a^r ≡ 1   (mod N)
⇒ a^r - 1 ≡ 0   (mod N)
⇒ (a^(r/2) - 1) · (a^(r/2) + 1) ≡ 0   (mod N)
```

Esto significa que `N` divide al producto `(a^(r/2) - 1)(a^(r/2) + 1)`,
pero **no** divide a ninguno de los dos factores por separado (porque
`a^(r/2) ≢ 1` por minimalidad de `r`, y `a^(r/2) ≢ -1` por hipótesis).
Por tanto, los dos enteros:

```
d1 = gcd( a^(r/2) - 1,  N )
d2 = gcd( a^(r/2) + 1,  N )
```

son **divisores no triviales** de `N`. Hemos factorizado.

Se puede demostrar que la probabilidad de que un `a` elegido al azar
caiga en uno de los casos "buenos" (`r` par y `a^(r/2) ≢ -1`) es al
menos `1/2` para casi todos los `N` compuestos no triviales. Por
tanto, con un puñado de intentos el algoritmo termina con alta
probabilidad.

### 2.3. Casos triviales detectados clásicamente

Antes de llamar al subrutina cuántica, el algoritmo clásico realiza
dos comprobaciones baratas pero importantes:

- Si `N` es par, devuelve `2` inmediatamente.
- Si `N = p^k` para algún primo `p` y `k ≥ 2`, devuelve `p`.
  Esto se comprueba por fuerza bruta para `k = 2, 3, ..., ⌊log₂ N⌋`.

Estas dos podas evitan invocar la maquinaria cuántica en casos donde
la salida es inmediata. En el código de `find_factor` se ven así:

```python
# Check if N is even or a non-trivial power.
if N % 2 == 0:
    print("Even number")
    return 2

for k in range(2, round(math.log(N, 2)) + 1):
    d = int(round(N ** (1 / k)))
    if d**k == N:
        factor_found = True
        print(f"{N} is {d} to the power {k}")
        return d
```

---

## 3. Transformada de Fourier Cuántica (QFT)

La Quantum Fourier Transform es la pieza algebraica que permite a
Shor extraer la periodicidad de la función `f(x) = a^x mod N`. Sin
QFT no hay algoritmo de Shor.

### 3.1. Definición formal

Sobre el espacio de `n` qubits, `N_q = 2^n`, la QFT es la
transformación lineal:

```
QFT |j⟩ = (1/√N_q) · Σ_{k=0}^{N_q - 1}  e^{2πi · j · k / N_q} · |k⟩
```

Es decir, mapea cada elemento de la base computacional a una
superposición uniforme cuyas fases son las raíces `N_q`-ésimas de la
unidad evaluadas en `j·k`. Es exactamente la **DFT clásica** salvo
porque vive sobre los amplitudes de un estado cuántico.

Si descomponemos `j = j_{n-1} ... j_1 j_0` en binario, la QFT admite
la factorización en producto tensorial:

```
QFT |j⟩ =  (1/√N_q) ⊗_{l=1}^{n}
           [ |0⟩ + e^{2πi · 0.j_{n-l+1}...j_n} |1⟩ ]
```

Esta forma producto es la que se traduce de manera directa a un
circuito eficiente.

### 3.2. Implementación en puertas

La construcción canónica consiste en:

1. Sobre cada qubit `q_k`, aplicar una **Hadamard**.
2. Aplicar **rotaciones de fase controladas** `CPhase(2π / 2^l)`
   entre el qubit recién Hadamardizado y todos los qubits posteriores,
   con ángulos cada vez más pequeños conforme aumenta la distancia.
3. Finalmente, **invertir el orden** de los qubits con `SWAP`s, porque
   la construcción natural produce el resultado en orden invertido.

El conteo de puertas es:

- `n` Hadamards
- `n(n-1)/2` rotaciones controladas
- `⌊n/2⌋` SWAPs finales

Total: **O(n²)** puertas, frente a las **O(n · 2^n)** que requeriría
una FFT clásica trabajando sobre vectores de amplitud explícita.
Esto sí: el speedup viene con un asterisco — la QFT no permite
**leer** todas las amplitudes resultantes, sólo medir una de ellas.

### 3.3. QFT aproximada

Las rotaciones de fase con ángulos extremadamente pequeños (por
ejemplo, `2π / 2^n` para `n` grande) son ruidosas y poco fiables en
hardware real. La QFT **aproximada** descarta las rotaciones con
ángulos menores que un umbral `π / 2^d`, donde `d` es el "grado de
aproximación". Coppersmith demostró que el error introducido es
acotado y, para `d ≈ log n`, despreciable a efectos prácticos.

### 3.4. El código `qft.py` línea a línea

El archivo `qft.py` del proyecto es un envoltorio muy delgado sobre
las primitivas de Qiskit que añade el soporte de la QFT aproximada y
otras opciones de síntesis. Su contenido íntegro es:

```python
from qiskit.circuit.library import QFTGate
from qiskit.synthesis import synth_qft_full


class QFTFullGate(QFTGate):
    """QFTGate supporting all the arguments of synth_qft_full."""

    # Whether to add swap gates in the end, to obtain the full QFT transformation.
    do_swaps: bool = True
    # The degree of approximation 0 <= d < n (0 for no approximation). Phase rotations
    # with angles smaller than π/2^{n-d} will be dropped.
    approximation_degree: int = 0
    # Whether to insert barrier gates in the circuit for better visualization.
    insert_barriers: bool = False

    def __init__(
        self,
        num_qubits: int,
        do_swaps: bool = True,
        approximation_degree: int = 0,
        insert_barriers: bool = False,
    ):
        super().__init__(num_qubits=num_qubits)
        assert 0 <= approximation_degree < num_qubits, (
            f"The approximation degree d must satisfy 0 <= d < {num_qubits}, "
            f"got d={approximation_degree}."
        )
        self.do_swaps = do_swaps
        self.approximation_degree = approximation_degree
        self.insert_barriers = insert_barriers

    def _define(self):
        """Provide a specific decomposition of the QFTGate into a quantum circuit."""
        self.definition = synth_qft_full(
            num_qubits=self.num_qubits,
            do_swaps=self.do_swaps,
            approximation_degree=self.approximation_degree,
            insert_barriers=self.insert_barriers,
        )
```

Interpretación detallada:

- **`class QFTFullGate(QFTGate)`** — Hereda de `QFTGate`, la
  representación simbólica de la QFT en Qiskit. El simbolismo permite
  al transpiler tratarla como una sola "puerta" lógica antes de
  decidir cómo descomponerla.
- **`do_swaps: bool = True`** — Atributo de clase con el flag por
  defecto. Si está activo, al final del circuito se invierten los
  qubits con `SWAP`s. Si lo desactivamos, la salida de la QFT está
  en orden **bit-reversed**, lo que a veces es útil porque podemos
  absorber esa permutación en el resto del circuito sin coste real.
- **`approximation_degree: int = 0`** — El grado `d` de la QFT
  aproximada. `0` significa QFT exacta. Cuanto mayor `d`, más
  rotaciones se eliminan.
- **`insert_barriers: bool = False`** — Marcadores visuales para que
  el `draw()` muestre separadores entre capas. No afecta a la
  semántica del circuito.
- **Constructor `__init__`** — Llama al `super().__init__` con
  `num_qubits` y guarda los flags. La aserción exige que `0 ≤ d < n`;
  si no, eliminaríamos las rotaciones triviales (que no aportan
  nada) o todas (que destruiría la QFT).
- **`_define(self)`** — Método estándar de Qiskit para "expandir" la
  puerta simbólica en un sub-circuito concreto. Se delega a
  `synth_qft_full`, que aplica el patrón estándar:

  Para `k = n-1, n-2, ..., 0`:
    1. `H(q_k)`
    2. Para cada `j = k-1, k-2, ..., 0`, si la rotación no se descarta
       por aproximación, `CPhase(2π/2^(k-j+1))` con control `q_j` y
       target `q_k`.
  Y finalmente, si `do_swaps`, swap `q_i ↔ q_{n-1-i}`.

Con `QFTFullGate` el resto del proyecto puede pedir la QFT exacta o
aproximada con un único parámetro, y deja que Qiskit transpile cada
puerta `CPhase` y `H` a las nativas del backend.

---

## 4. Estimación de Fase Cuántica (QPE)

Order finding se resuelve cuánticamente reduciéndolo a un problema
todavía más fundamental: **estimar la fase** de un autovalor de un
operador unitario.

### 4.1. El problema de la estimación de fase

Sea `U` un operador unitario sobre `n` qubits y sea `|ψ⟩` uno de sus
autovectores con autovalor `e^{2πi φ}` (todo autovalor de un unitario
tiene módulo 1):

```
U |ψ⟩ = e^{2πi φ} |ψ⟩,   φ ∈ [0, 1)
```

El problema de **Quantum Phase Estimation (QPE)** consiste en obtener
una aproximación binaria de `φ` con `m` bits de precisión:

```
φ ≈ 0.b_{m-1} b_{m-2} ... b_0   (binario)
```

con probabilidad alta y usando del orden de `2^m` aplicaciones del
operador `U` (sin que `U` sea diagonalizado explícitamente).

### 4.2. El circuito de QPE

QPE utiliza dos registros:

- **Registro de control** de `m` qubits, inicializado a `|0...0⟩`.
- **Registro de trabajo** de `n` qubits, inicializado al autoestado
  `|ψ⟩`.

Pasos:

1. **Superposición sobre control**: aplicamos Hadamard a cada qubit
   del registro de control, obteniendo:

   ```
   (1/√2^m) · Σ_{x=0}^{2^m - 1} |x⟩ ⊗ |ψ⟩
   ```

2. **Potencias controladas de U**: por cada qubit de control `k`,
   aplicamos `U^{2^k}` controlada por ese qubit. El estado se
   transforma en:

   ```
   (1/√2^m) · Σ_x  e^{2πi · x · φ}  |x⟩ ⊗ |ψ⟩
   ```

   Es decir, **la fase `φ` queda codificada en el registro de control**
   en forma de "diente de sierra" exponencial sobre `x`. El registro
   de trabajo sigue conteniendo `|ψ⟩` (autoestado), no se ve
   modificado en su estructura, solo se usa como anclaje para
   aplicar `U`.

3. **QFT inversa** sobre el registro de control. Esta es exactamente
   la operación necesaria para convertir esa codificación exponencial
   en una base donde la información de `φ` es **legible por medición
   directa**.

4. **Medir** el registro de control. Si `φ = y/2^m` es exactamente
   diádico, la medida devuelve `y` con probabilidad 1. En general,
   devuelve un valor cercano a `y = ⌊2^m · φ⌋` con probabilidad
   `≥ 4/π² ≈ 0.405`.

### 4.3. ¿Cómo QPE extrae el orden?

En el algoritmo de Shor, el operador unitario que usamos es:

```
U_a |y⟩ = |a · y mod N⟩
```

(es unitario porque `gcd(a, N) = 1`, así que la multiplicación por
`a` es una permutación). Los autovalores de `U_a` son los `e^{2πi
s/r}` para `s = 0, 1, ..., r-1`, donde `r` es justamente el orden de
`a` en `Z_N*`. Los autovectores correspondientes son:

```
|ψ_s⟩ = (1/√r) · Σ_{k=0}^{r-1}  e^{-2πi · s · k / r}  |a^k mod N⟩
```

Es un hecho clave que:

```
(1/√r) · Σ_s |ψ_s⟩ = |1⟩
```

es decir, **el estado `|1⟩` es una superposición uniforme de todos
los autovectores `|ψ_s⟩`**. Por tanto, **no hace falta saber preparar
ningún `|ψ_s⟩` específico**: basta con inicializar el registro de
trabajo en `|1⟩`. QPE entonces estimará uno de los `φ = s/r`
elegidos al azar entre `s = 0, ..., r-1`.

El último paso es **fracciones continuas**: tras medir un valor
`x ∈ {0, ..., 2^m - 1}` y dividir por `2^m`, obtenemos una
aproximación `x/2^m ≈ s/r`. Aplicando el algoritmo de fracciones
continuas a `x/2^m` y truncando convergentes cuyo denominador
exceda `N - 1`, recuperamos `r` (o un múltiplo de `r`). En Python
estándar esto es una línea con `fractions.Fraction.limit_denominator`,
como veremos en el código.

---

## 5. Exponenciación Modular Cuántica

### 5.1. El operador U_a

El operador unitario que QPE necesita aplicar en sus potencias
`U_a^{2^k}` actúa sobre el registro de trabajo así:

```
U_a |y⟩ = |a · y mod N⟩
```

Aplicar `U_a^{2^k}` equivale a multiplicar por `a^{2^k} mod N`. La
versión **controlada** que necesitamos en QPE realiza:

```
|c⟩ |y⟩ → |c⟩ · |a^{c · 2^k} · y mod N⟩
```

con `c ∈ {0, 1}` el bit de control. Combinando una de estas para
cada uno de los `m` qubits del registro de control y multiplicando
sucesivamente sobre el mismo registro de trabajo, obtenemos:

```
|x⟩_m |1⟩_n → |x⟩_m |a^x mod N⟩_n
```

Esa es la **exponenciación modular cuántica**: el corazón
computacional del algoritmo. Su coste asintótico domina el coste
total de Shor.

### 5.2. Estrategia general de Beauregard

La implementación que utiliza este proyecto (en `adder.py`) sigue el
diseño de Beauregard (`quant-ph/0205095`), que descompone la
exponenciación en una cascada de **multiplicaciones modulares
controladas**, cada una en una cascada de **sumas modulares
controladas**, y cada suma se realiza en el **espacio de Fourier**
mediante el sumador de Draper. Las ventajas:

- Usa pocos qubits auxiliares: `n + 2` ancillas además del registro
  de control y el de trabajo. En total `4n + 2` para la versión
  estándar y `2n + 3` para la versión "un solo qubit de control"
  con medidas iterativas.
- Las sumas en QFT no necesitan acarreo clásico explícito; el
  acarreo está implícito en las fases.

### 5.3. El sumador de Draper

#### 5.3.1. Idea geométrica

Si interpretamos el registro `|y⟩` con `n` qubits como un entero
binario, sumar un clásico `X` se puede hacer:

1. Aplicar QFT al registro `|y⟩` → estado en "espacio de fases".
2. En ese espacio, sumar `X` equivale a multiplicar por `e^{2πi X /
   2^n}`, lo cual se logra **aplicando rotaciones individuales** a
   cada qubit: `P(2π · X · 2^{i-n})` sobre el qubit `i`.
3. Aplicar QFT inversa.

El resultado es `|y + X mod 2^n⟩`. El sumador no necesita ningún bit
de acarreo entre qubits: las rotaciones son **independientes** unas
de otras. Esta es la magia del adder de Draper.

#### 5.3.2. `add_classical`: la primitiva más básica

Veamos el código de `add_classical` en `adder.py`:

```python
def add_classical(
    self,
    X: int,
    y_reg: QuantumRegister | list[Qubit],
    include_QFT: bool = True,
) -> None:
    """
    Adds the classical integer X to the quantum integer y.
    Operation: |y> -> |X + y >
    """
    y_bits = self.get_qubits(y_reg)
    n = len(y_bits)

    if include_QFT:
        qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n))
        self.compose(qft_gate, y_bits, inplace=True)

    # Phase gates (= addition in Fourier space)
    for i in range(n):
        self.p(2 * np.pi * X * (2 ** (i - n)), y_bits[i])

    if include_QFT:
        inv_qft_gate = QFTFullGate(
            n, approximation_degree=self.qft_approx_degree(n)
        ).inverse()
        self.compose(inv_qft_gate, y_bits, inplace=True)
```

Explicación detallada:

- **Argumentos**: `X` es un Python int (clásico), `y_reg` es el
  registro cuántico que se modifica in-place, `include_QFT` permite
  desactivar la QFT envolvente cuando varias sumas se encadenan y
  podemos amortizar QFT/QFT⁻¹.
- **`n = len(y_bits)`**: número de qubits del registro.
- **`qft_gate = QFTFullGate(n, approximation_degree=...)`**: la QFT
  exacta o aproximada. La elección del grado de aproximación se
  decide por la heurística `qft_approx_degree`, definida así en la
  clase:

  ```python
  def qft_approx_degree(self, n: int):
      return max(0, n - math.ceil(np.log2(n)) - 2) if self.approx_QFT else 0
  ```

  Es decir, descarta rotaciones con ángulo menor que `π / 2^(log n + 2)`,
  un umbral con error acotado por arriba.
- **Bucle de rotaciones de fase**: para cada qubit `i`, aplicamos
  `P(θ)` con `θ = 2π · X · 2^{i-n}`. Importante: el bit menos
  significativo recibe el ángulo más pequeño y el bit más significativo
  recibe `2π · X · 1/2 = π X` (módulo `2π`). Esto refleja directamente
  la descomposición de la suma binaria en el espacio de Fourier.
- **QFT inversa**: deshace la transformada, dejando el estado en la
  base computacional como `|X + y mod 2^n⟩`.

#### 5.3.3. `c_add_classical`: versión controlada

La versión controlada simplemente sustituye cada `P(θ)` por una
`PhaseGate(θ).control(k)` con `k` el número de bits de control. La
QFT envolvente **no se controla** porque hacerlo sería costoso y
porque, si el control está en `|0⟩`, las rotaciones internas son
identidad y la QFT–QFT⁻¹ se cancelan automáticamente: el efecto
neto es la identidad sobre `|y⟩`.

```python
def c_add_classical(
    self,
    control_reg,
    X: int,
    y_reg,
    include_QFT: bool = True,
) -> None:
    control_bits = self.get_qubits(control_reg)
    y_bits = y_reg[:]
    n = len(y_bits)
    k = len(control_bits)

    if include_QFT:
        qft_gate = QFTFullGate(n, approximation_degree=self.qft_approx_degree(n))
        self.compose(qft_gate, y_bits, inplace=True)

    for i in range(n):
        theta = 2 * np.pi * X * (2 ** (i - n))
        cp_gate = PhaseGate(theta).control(k)
        self.append(cp_gate, control_bits + [y_bits[i]])

    if include_QFT:
        inv_qft_gate = QFTFullGate(
            n, approximation_degree=self.qft_approx_degree(n)
        ).inverse()
        self.compose(inv_qft_gate, y_bits, inplace=True)
```

#### 5.3.4. `add_classical_modulo` (φ_add_mod): suma modular

Sumar módulo `N` es mucho más delicado: si `X + y ≥ N`, hay que
restar `N`. La idea de Beauregard:

```
|y>|0>  → |X + y>|0>                       (suma "libre" en n+1 bits)
       → comparar con N:
              si X + y >= N → setear ancilla bit alto
              si X + y <  N → ancilla queda en 0
       → restar N condicionalmente para mantener el resultado en [0, N)
       → restaurar la ancilla a |0> para que sea reutilizable
```

El código exacto:

```python
def add_classical_modulo(
    self,
    X: int,
    y_reg,
    ancilla_bit: Qubit,
    N: int,
    reset_ancilla: bool = True,
) -> None:
    y_bits = y_reg[:]
    n = math.ceil(math.log2(N))

    assert 0 <= X and X < N, "X must be smaller than N."
    assert len(y_bits) == n + 1, "The y register must have n+1 qubits."

    self.add_classical(X - N, y_bits)          # (1) y -> y + X - N
    self.cx(y_bits[n], ancilla_bit)            # (2) si overflow, marca ancilla
    self.c_add_classical(ancilla_bit, N, y_bits)  # (3) suma N si era negativo

    if reset_ancilla:
        # Limpieza para que ancilla vuelva a |0> sin afectar a y_bits
        self.add_classical(-X, y_bits)
        self.cx(y_bits[n], ancilla_bit)
        self.x(ancilla_bit)
        self.add_classical(X, y_bits)
```

Análisis paso a paso:

1. **`add_classical(X - N, y_bits)`** — Suma `X - N` (que puede ser
   negativo) al registro `y`. Trabajamos en `n+1` bits, así que la
   suma se hace módulo `2^(n+1)`. Si `y + X < N`, el resultado es
   negativo y, en complemento a dos, su **bit más significativo
   `y_bits[n]` está activo**. Si `y + X ≥ N`, el resultado cabe sin
   problema en `n` bits y el bit alto es 0.
2. **`cx(y_bits[n], ancilla_bit)`** — Copiamos el bit de signo a la
   ancilla: ahora `ancilla = 1` si el resultado fue negativo,
   `ancilla = 0` en caso contrario.
3. **`c_add_classical(ancilla_bit, N, y_bits)`** — Sumamos `N` al
   registro condicionalmente. Si el resultado fue negativo,
   recuperamos `(y + X - N) + N = y + X`. Si no, no hacemos nada.
4. **Bloque de reset** — Después de los pasos 1-3, `y_bits` contiene
   `(y + X) mod N`, pero `ancilla` todavía está en `|1⟩` si hubo
   underflow. Para usarla en una próxima suma debemos limpiarla. La
   técnica clásica es **uncomputar el comparador**: comparar
   `(y + X) mod N` con `X` en lugar de con `N`. El truco:
   `add_classical(-X)` produce un resultado negativo si y solo si
   `(y + X) mod N < X`, condición equivalente a "hubo módulo". Eso
   se usa para hacer `cx + x` y dejar la ancilla en `|0⟩`. Finalmente
   `add_classical(X)` restaura el valor original.

Esto es exactamente la receta de Beauregard `φ_add_mod`. Es densa,
pero **completamente reversible** y deja la ancilla limpia para
reusarse.

#### 5.3.5. `c_add_classical_modulo` (c_φ_add_mod)

La versión controlada de la anterior. La diferencia clave es que
solo el primer paso (`c_add_classical(control_bits, X, y_bits)`) se
controla; los siguientes son siempre incondicionales. Esto es
correcto porque, si el control está en `|0⟩`, el primer paso es la
identidad y todos los demás se compensan automáticamente en la fase
de reset.

```python
def c_add_classical_modulo(
    self,
    control_reg,
    X: int,
    y_reg,
    ancilla_bit: Qubit,
    N: int,
    reset_ancilla: bool = True,
) -> None:
    control_bits = self.get_qubits(control_reg)
    y_bits = y_reg[:]
    n = math.ceil(math.log2(N))
    k = len(control_bits)

    assert 0 <= X and X < N, "X must be smaller than N."
    assert len(y_bits) == n + 1, "The y register must have n+1 qubits."

    self.c_add_classical(control_bits, X, y_bits)
    self.add_classical(-N, y_bits)
    self.cx(y_bits[n], ancilla_bit)
    self.c_add_classical(ancilla_bit, N, y_bits)

    if reset_ancilla:
        self.add_classical(-X, y_bits)
        ccx_gate = CXGate().control(k)
        self.append(ccx_gate, control_bits + [y_bits[n], ancilla_bit])
        self.x(ancilla_bit)
        self.add_classical(X, y_bits)
```

Observe la `CXGate().control(k)` en el reset: equivale a una
Toffoli generalizada que solo dispara cuando **todos los controles
están a 1 y `y_bits[n]` está a 1**, garantizando que la ancilla se
limpia exactamente igual que en el caso no controlado, pero
preservando la propiedad de control externo.

### 5.4. Multiplicación modular cuántica

Sumando `A·x` al registro `y` qubit por qubit del `x`, con factores
multiplicados por `2^i`, obtenemos:

```
|x⟩ |y⟩ → |x⟩ |(y + A·x) mod N⟩
```

Eso es lo que hace `add_quantum_modulo`:

```python
def add_quantum_modulo(
    self,
    x_reg,
    y_reg,
    ancilla_bit: Qubit,
    N: int,
    A: int = 1,
) -> None:
    x_bits = x_reg[:]
    y_bits = y_reg[:]
    n = math.ceil(math.log2(N))
    m = len(x_bits)

    assert m <= n, "x register may hold too large numbers."
    assert len(y_bits) == n + 1, "The y register must have n+1 qubits."

    for i in range(m):
        self.c_add_classical_modulo(
            control_reg=x_bits[i],
            N=N,
            X=((A % N) * 2**i) % N,
            y_reg=y_bits,
            ancilla_bit=ancilla_bit,
        )
```

Esto es la clave: cada bit `x_bits[i]` controla una suma modular de
`A · 2^i mod N` al registro `y`. Sumando todas, el efecto es agregar
`A · x mod N`. El cálculo se hace bit a bit sin necesidad de un
sumador multi-control complejo.

A partir de ahí, la **multiplicación in-place** `x → A x mod N`
necesita además un registro `y` auxiliar y un par de bits ancilla:

```python
def multiply_modulo(
    self,
    A: int,
    x_reg,
    y_reg,
    overflow_bit: Qubit,
    ancilla_bit: Qubit,
    N: int,
    with_uncomputation: bool = True,
    with_swap: bool = True,
) -> None:
    x_bits = x_reg[:]
    y_bits = y_reg[:]
    n = math.ceil(math.log2(N))

    # Out-of-place a-multiplication stage: |x>|0>|0> -> |x>|ax mod N>|0>
    self.add_quantum_modulo(
        x_reg=x_bits,
        y_reg=y_bits + [overflow_bit],
        ancilla_bit=ancilla_bit,
        N=N,
        A=A,
    )
    if with_swap:
        # Swap stage : |x>|ax mod N>|0> -> |ax mod N>|x>|0>
        for i in range(n):
            self.swap(x_bits[i], y_bits[i])
    if with_uncomputation:
        # Uncomputation stage: |ax mod N>|x>|0> -> |ax mod N>|0>|0>
        B = pow(A, -1, N)  # AB = 1 mod N
        x = x_bits if with_swap else y_bits
        y = y_bits if with_swap else x_bits
        self.add_quantum_modulo(
            x_reg=x, y_reg=y + [overflow_bit], ancilla_bit=ancilla_bit, N=N, A=-B
        )
```

Las tres etapas son:

1. **Multiplicación out-of-place**: `|x⟩|0⟩ → |x⟩|A·x mod N⟩`. Aquí
   `y_reg + [overflow_bit]` proporciona los `n+1` qubits requeridos
   por `add_quantum_modulo`.
2. **Swap**: intercambia los dos registros, dejando el resultado en
   `x_reg` y el original en `y_reg`.
3. **Uncomputation con el inverso**: para limpiar `y_reg` (que ahora
   contiene `x`), aplicamos la misma transformación con
   `A → -A^{-1} mod N`. Esto resta `A^{-1} · (A·x) = x` del `y_reg`,
   dejándolo en `|0⟩`. La existencia de `A^{-1}` está garantizada
   porque `gcd(A, N) = 1` (lo verificamos en `order_finding_circuit`).

La versión controlada `c_multiply_modulo` añade controles a los
`add_quantum_modulo` y reemplaza el `swap` por `cswap` (controlled
SWAP). El esqueleto es idéntico.

### 5.5. Exponenciación modular: el ensamblaje final

La exponenciación modular `|x⟩ |y⟩ → |x⟩ |A^x · y mod N⟩` se
construye encadenando `m` multiplicaciones modulares controladas
con bases distintas:

```python
def exponentiate_modulo(
    self,
    A: int,
    x_reg,
    y_reg,
    ancilla_reg,
    N: int,
) -> None:
    x_bits = x_reg[:]
    y_bits = y_reg[:]
    a_bits = ancilla_reg[:]
    n = math.ceil(math.log2(N))
    m = len(x_bits)

    assert len(y_bits) == n, "The y register must have n qubits."
    assert len(ancilla_reg) == n + 2, "The ancilla register must have n+2 qubits."

    for i in range(m):
        self.c_multiply_modulo(
            control_reg=x_bits[i],
            A=pow(A, 2**i, N),
            x_reg=y_bits,
            y_reg=a_bits[:n],
            overflow_bit=a_bits[n],
            ancilla_bit=a_bits[n + 1],
            N=N,
        )
```

El bucle hace:

- Por cada bit `x_bits[i]` del registro de exponente, si está en
  `|1⟩`, multiplica el registro `y_bits` por `A^{2^i} mod N`. Si
  está en `|0⟩`, no hace nada.
- Como cualquier `x = Σ_i x_i · 2^i`, la composición de todas estas
  multiplicaciones controladas equivale a multiplicar por
  `Π_i A^{x_i · 2^i} = A^x`. Exactamente lo que queremos.
- Las ancillas se reutilizan entre iteraciones: cada
  `c_multiply_modulo` deja los bits ancilla limpios al final, así
  que pueden encadenarse sin riesgo.

El truco crucial es **precomputar clásicamente `pow(A, 2^i, N)`**.
No hace falta calcular potencias cuánticamente: el exponente del
`U_a` en QPE es siempre `2^k`, y `A^{2^k} mod N` es un entero
clásico fijo conocido antes de construir el circuito.

---

## 6. El Algoritmo Completo

### 6.1. Estructura del circuito

Para factorizar un `N` de `n = ⌈log₂ N⌉` bits, el algoritmo de Shor
con QPE estándar utiliza:

| Registro | Tamaño | Propósito |
|----------|--------|-----------|
| Control (exponente) | `m = 2n` | Almacena la fase estimada `s/r` |
| Trabajo | `n` | Mantiene `a^x mod N` |
| Ancilla | `n + 2` | Auxiliares para sumas modulares y overflow |
| **Total qubits** | **`4n + 2`** | |

Por qué `m = 2n`: para que las fracciones continuas distingan
fracciones con denominador `≤ N` necesitamos al menos `2n` bits de
precisión (una desigualdad de Hardy-Wright clásica). Más precisión
no perjudica pero aumenta el coste.

### 6.2. Flujo del circuito

```
1. Inicializar control: H^{⊗m} sobre control_register
2. Inicializar trabajo: X sobre target_register[0] → |1⟩
3. Exponenciación modular: |x⟩|1⟩ → |x⟩|a^x mod N⟩
4. QFT⁻¹ sobre control_register
5. Medir control_register → bitstring x
6. Post-procesar:
       φ ≈ x / 2^m
       r = denominador del convergente de φ con denominador ≤ N-1
7. Si r par y a^(r/2) ≢ -1: factor = gcd(a^(r/2) ± 1, N)
```

En Qiskit este flujo se compone así (`order_finding_circuit` en
`shor.py`):

```python
def order_finding_circuit(A: int, N: int, precision: int | None = None) -> AdderCircuit:
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return 0

    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    control_register = QuantumRegister(m)
    target_register = QuantumRegister(n)
    ancilla_register = QuantumRegister(n + 2)
    output_register = ClassicalRegister(m, name="output_bits")
    qc = AdderCircuit(
        control_register, target_register, ancilla_register, output_register
    )

    # Prepare control state in "all quantum integers" superposition state
    for i in range(m):
        qc.h(control_register[i])

    # Prepare target state in |1> state
    qc.x(target_register[0])

    # Apply modular exponential operator
    qc.exponentiate_modulo(
        A=A,
        x_reg=control_register,
        y_reg=target_register,
        ancilla_reg=ancilla_register,
        N=N,
    )

    # Apply inverse QFT
    qc.compose(QFTGate(m).inverse(), qubits=control_register, inplace=True)

    # Measure control state
    qc.measure(control_register, output_register)

    return qc
```

### 6.3. Por qué se necesitan varios intentos: `num_tries`

Después de medir el registro de control, la estimación de fase puede
**fallar** por varios motivos:

- La fase medida es `0` (corresponde a `s = 0`). No nos da
  información: `r` queda indeterminado. El código salta ese caso.
- `s` puede ser tal que `gcd(s, r) > 1`. En ese caso, la fracción
  continua reduce `s/r` y obtenemos un divisor de `r`, no `r`. La
  verificación `a^r ≡ 1 mod N` falla.
- Aun obteniendo un `r` válido, este puede ser **impar** o cumplir
  `a^(r/2) ≡ -1`, en cuyo caso debemos descartar y probar otro `a`.

Por todo eso, el algoritmo hace varios intentos (`num_tries`) con
distintos `a`, y para cada `a` toma varias muestras
(`num_shots_per_trial`). En el código:

```python
while (not factor_found) and i < num_tries:
    a = random.randint(2, N - 1)
    d = math.gcd(a, N)
    if d > 1:
        factor_found = True
        print(f"Lucky guess of {a}, found factor {d}")
        return d
    # Run order finding circuit
    r, _ = find_order(
        a,
        N,
        sampler,
        pass_manager,
        num_shots=num_shots_per_trial,
        one_control_circuit=one_control_circuit,
    )
    if r == 0:
        continue
    if r % 2 == 0:
        x = pow(a, r // 2, N) - 1
        d = math.gcd(x, N)
        if d > 1 and d < N:
            factor_found = True
    i += 1
```

### 6.4. Valores de `N` y qubits en este proyecto

El proyecto utiliza una lista concreta de números compuestos:

| N | log₂N ≈ | n = ⌈log₂N⌉ | m = 2n | Total qubits 4n+2 |
|---|---------|-------------|--------|-------------------|
| 15 | 3.91 | 4 | 8 | 18 |
| 21 | 4.39 | 5 | 10 | 22 |
| 35 | 5.13 | 6 | 12 | 26 |
| 77 | 6.27 | 7 | 14 | 30 |
| 143 | 7.16 | 8 | 16 | 34 |

Es interesante notar que el coste crece **muy rápido**. Pasar de
`N=15` a `N=143` casi duplica el conteo de qubits y multiplica
drásticamente la profundidad del circuito (que crece como `n^3`).
Esto explica por qué incluso simulaciones clásicas de Shor están
fuera del alcance para `N` algo mayores.

### 6.5. La versión "un solo qubit de control"

`shor.py` ofrece una variante eficiente en qubits:
`order_finding_circuit_one_control`. La idea es usar **un solo
qubit de control** que se mide y resetea iterativamente, en lugar
de `m` qubits de control simultáneos. Esto cambia el conteo de
qubits totales de `4n + 2` a `2n + 3`, pagando profundidad en
medidas intermedias y control clásico condicionado:

```python
def order_finding_circuit_one_control(A: int, N: int, precision=None):
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return 0

    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    control_register = QuantumRegister(1)
    target_register = QuantumRegister(n)
    ancilla_register = QuantumRegister(n + 2)
    output_register = ClassicalRegister(m, name="output_bits")
    qc = AdderCircuit(
        control_register, target_register, ancilla_register, output_register
    )

    qc.x(target_register[0])

    c_bit = control_register[0]
    for i in range(m):
        # Preparación del bit de control
        qc.h(c_bit)
        # Multiplicación modular controlada
        qc.c_multiply_modulo(
            control_reg=c_bit,
            A=pow(A, 2 ** (m - i - 1), N),
            x_reg=target_register,
            y_reg=ancilla_register[:n],
            overflow_bit=ancilla_register[n],
            ancilla_bit=ancilla_register[n + 1],
            N=N,
        )
        # i-ésimo bloque de la QFT inversa: rotaciones de fase condicionadas
        # a las medidas previas
        for j in range(i):
            with qc.if_test((output_register[j], 1)):
                qc.p(-math.pi / 2 ** (i - j), c_bit)
        qc.h(c_bit)
        # Medida
        qc.measure(c_bit, output_register[i])
        # Reset condicionado al resultado de la medida
        with qc.if_test((output_register[i], 1)):
            qc.x(c_bit)
    return qc
```

Esto requiere que el backend soporte **mediciones intermedias y
operaciones clásicas condicionadas**, una capacidad disponible solo
en hardware moderno (los sistemas de "dynamic circuits"). Es una
optimización fundamental para experimentos con pocos qubits físicos
disponibles.

---

## 7. Implementación en Código: `shor.py` de Qiskit

### 7.1. La orquestación de alto nivel

`shor.py` integra todo lo construido en `adder.py` y `qft.py`. Sus
componentes principales:

- **`order_finding_circuit(A, N, precision)`** — construye el
  circuito de QPE con exponenciación modular.
- **`order_finding_circuit_one_control(A, N, precision)`** — su
  variante de un solo qubit de control con medidas iterativas.
- **`_get_order_from_dist(dist, A, N, precision)`** — post-procesa
  el histograma de medidas para extraer `r` por fracciones
  continuas.
- **`find_order(A, N, sampler, pass_manager, ...)`** — orquesta la
  ejecución de uno o varios shots y devuelve el orden.
- **`find_factor(N, sampler, pass_manager, ...)`** — bucle externo
  clásico: elección de `a`, podas baratas, llamada a `find_order` y
  recuperación de los factores.

### 7.2. La detección de "lucky guess"

Una observación matemática elegante: **si `a² mod N = 1`**, entonces
el orden de `a` es 1 o 2. Si es 1, `a = 1` (caso degenerado). Si es
2, entonces `(a-1)(a+1) ≡ 0 mod N`, y podemos extraer los factores
*sin necesidad de circuito cuántico*: basta con `gcd(a-1, N)` y
`gcd(a+1, N)`.

En el código del proyecto, antes de entrar al circuito se hace una
poda equivalente: si `gcd(a, N) > 1`, ya tenemos un factor común y
devolvemos directamente:

```python
a = random.randint(2, N - 1)
d = math.gcd(a, N)
if d > 1:
    factor_found = True
    print(f"Lucky guess of {a}, found factor {d}")
    return d
```

Este caso es raro cuando `N` es producto de dos primos grandes
(probabilidad `≈ 1/p + 1/q − 1/N` aproximadamente, despreciable
para primos grandes), pero relativamente frecuente para `N`
pequeños como los del proyecto.

### 7.3. La función `_get_order_from_dist`

Es donde ocurre la magia de las fracciones continuas:

```python
def _get_order_from_dist(dist: dict, A: int, N: int, precision: int) -> int:
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    # Look for the order in the 10 most frequent outputs.
    for i in range(min(10, len(sorted_outputs))):
        if sorted_outputs[i] == "0" * precision:
            continue
        x = int(sorted_outputs[i], 2)
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            print(
                f"Found value {r} for the order of {A} in Z_{N}. "
                f"If running on noisy quantum hardware, {r} might be "
                f"a multiple of the order instead."
            )
            return r
    print(f"Failed to find order of {A} in Z_{N}")
    return 0
```

Línea a línea:

- **`sorted_outputs = sorted(dist, key=dist.get, reverse=True)`** —
  Ordena los bitstrings medidos de más frecuente a menos. La
  probabilidad de medir cerca de un `s/r · 2^m` correcto es alta,
  pero los picos finitos del histograma reciben varios resultados
  cercanos. Tomar los más frecuentes maximiza la probabilidad de
  acertar.
- **`for i in range(min(10, len(sorted_outputs)))`** — Probamos
  hasta los 10 resultados más frecuentes. Si ninguno funciona, se
  devuelve 0 (fallo).
- **`if sorted_outputs[i] == "0" * precision: continue`** — El
  resultado `x = 0` corresponde a `s = 0`, fase nula. No da
  información sobre `r`. Se ignora.
- **`x = int(sorted_outputs[i], 2)`** — Convierte el bitstring a
  entero.
- **`r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator`** —
  La clave. `x / 2^m` es nuestra aproximación racional a `s / r`.
  `limit_denominator(N - 1)` encuentra el convergente de la
  fracción continua de `x / 2^m` con denominador máximo `N - 1`.
  Si la estimación de fase ha sido correcta, ese convergente es
  precisamente `s / r` y su denominador es `r`.
- **`if pow(A, r, N) == 1`** — Verificación clásica: si `a^r ≡ 1
  mod N`, hemos encontrado el orden (o un múltiplo). Si no, el
  candidato `r` es incorrecto y probamos el siguiente.

### 7.4. La función `find_order`

```python
def find_order(
    A: int,
    N: int,
    sampler,
    pass_manager,
    precision: int | None = None,
    num_shots: int = 10,
    one_control_circuit: bool = False,
) -> tuple[int, dict[str, int]]:
    m = precision if precision is not None else 2 * math.ceil(math.log2(N))
    if one_control_circuit:
        qc = order_finding_circuit_one_control(A, N, precision=m)
    else:
        qc = order_finding_circuit(A, N, precision=m)
    if qc == 0:
        return 0, {}
    qc_isa = pass_manager.run(qc)

    print(f"Start search for the order of {A} in Z_{N}")
    dist = (
        sampler.run([qc_isa], shots=num_shots).result()[0]
        .data.output_bits.get_counts()
    )
    r = _get_order_from_dist(dist, A, N, precision=m)
    return r, dist
```

Roles:

- **`pass_manager.run(qc)`** — Pasa el circuito por el transpiler
  configurado (en backends reales suele incluir mapeo a la
  topología, descomposición a puertas nativas, optimización de
  niveles). En simuladores se usa también para uniformidad.
- **`sampler.run([qc_isa], shots=num_shots)`** — La primitiva
  `Sampler` de Qiskit que ejecuta el circuito y devuelve el
  histograma. `num_shots` es el número de mediciones repetidas.
- **`.data.output_bits.get_counts()`** — Accede al `ClassicalRegister`
  llamado `output_bits` que se definió en `order_finding_circuit`.

### 7.5. La función `find_factor`

El bucle externo completo:

```python
def find_factor(
    N: int,
    sampler,
    pass_manager,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    one_control_circuit: bool = False,
    seed: int | None = None,
) -> int:
    # Podas baratas
    if N % 2 == 0:
        print("Even number")
        return 2

    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            factor_found = True
            print(f"{N} is {d} to the power {k}")
            return d

    i = 0
    factor_found = False
    if seed is not None:
        random.seed(seed)
    while (not factor_found) and i < num_tries:
        a = random.randint(2, N - 1)
        d = math.gcd(a, N)
        if d > 1:
            factor_found = True
            print(f"Lucky guess of {a}, found factor {d}")
            return d
        r, _ = find_order(
            a, N, sampler, pass_manager,
            num_shots=num_shots_per_trial,
            one_control_circuit=one_control_circuit,
        )
        if r == 0:
            continue
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if d > 1 and d < N:
                factor_found = True
        i += 1

    if factor_found:
        print(f"Factor found: {d}")
        return d

    print("No factor found")
    return 1
```

Observaciones:

- El bucle solo incrementa `i` cuando se completa una ejecución
  cuántica; los "lucky guess" salen antes del while sin gastar uno
  de los `num_tries`.
- Si `r == 0`, el círculo cuántico falló (ningún convergente de
  fracciones continuas funcionó) y se intenta de nuevo sin
  incrementar `i`. Esto evita penalizar al algoritmo por ruido
  estadístico en la medida.
- Se usa `pow(a, r // 2, N)` — exponenciación modular de Python en
  tiempo polinómico. **Solo** se calcula `gcd(x, N)` con `x =
  a^(r/2) - 1`. No se calcula también `gcd(a^(r/2) + 1, N)`: es una
  simplificación que reduce un poco la probabilidad de éxito por
  intento, pero el bucle compensa con más intentos.

### 7.6. Vista comparativa: la versión Cirq

El archivo `python/cirq/shor/shor.py` implementa el mismo algoritmo
en Google Cirq. La estructura externa es esencialmente idéntica:

```python
def order_finding_circuit(A: int, N: int, precision: int | None = None) -> cirq.Circuit:
    if math.gcd(A, N) > 1:
        print(f"Error: gcd({A},{N}) > 1")
        return cirq.Circuit()

    n = math.ceil(math.log2(N))
    m = precision if precision is not None else 2 * n

    exponent_qubits = cirq.LineQubit.range(m)
    target_qubits = cirq.LineQubit.range(m, m + n)

    circuit = cirq.Circuit()

    # Prepare exponent register in superposition
    circuit.append(cirq.H.on_each(*exponent_qubits))

    # Prepare target register in |1> state
    circuit.append(cirq.X(target_qubits[0]))

    # Apply modular exponentiation gate
    mod_exp = ModularExp(
        target_size=n,
        exponent_size=m,
        base=A,
        modulus=N,
    )
    circuit.append(mod_exp.on(*target_qubits, *exponent_qubits))

    # Apply inverse QFT on the exponent register
    circuit.append(cirq.qft(*exponent_qubits, inverse=True))

    # Measure the exponent register
    circuit.append(cirq.measure(*exponent_qubits, key="result"))

    return circuit
```

Diferencias notables respecto a Qiskit:

- **`cirq.LineQubit.range(...)`** — los qubits son objetos
  identificados por su posición en una línea. No hay distinción
  entre "registros" como tales: los grupos son listas de qubits.
- **`cirq.H.on_each(*qubits)`** — aplica Hadamard a todos los
  qubits de un solo trazo.
- **`ModularExp`** — Cirq trae una abstracción de "puerta compuesta"
  para la exponenciación modular en `cirq.contrib.shor` o en el
  módulo `modular_exp` del propio proyecto. Internamente sigue
  diseños equivalentes al de Beauregard.
- **`cirq.qft(*qubits, inverse=True)`** — la QFT inversa viene
  como primitiva de Cirq, sin necesidad de envoltorio.
- **`cirq.measure(*qubits, key="result")`** — clave de medida en
  lugar de registro clásico explícito.

El post-procesado clásico es prácticamente idéntico:

```python
def _get_order_from_dist(dist: dict[int, int], A: int, N: int, precision: int) -> int:
    sorted_outputs = sorted(dist, key=dist.get, reverse=True)
    for i in range(min(10, len(sorted_outputs))):
        x = sorted_outputs[i]
        if x == 0:
            continue
        r = Fraction(x / 2**precision).limit_denominator(N - 1).denominator
        if pow(A, r, N) == 1:
            print(
                f"Found value {r} for the order of {A} in Z_{N}. "
                f"If running on noisy quantum hardware, {r} might be "
                f"a multiple of the order instead."
            )
            return r
    print(f"Failed to find order of {A} in Z_{N}")
    return 0
```

La diferencia más sutil es que **`dist` aquí es un dict de enteros
a contadores** (porque `result.histogram(...)` ya devuelve enteros),
mientras que Qiskit lo devuelve como dict de strings binarias a
contadores. Por eso en Cirq la línea `x = sorted_outputs[i]` es
directa, y en Qiskit hay un `int(sorted_outputs[i], 2)` extra.

---

## 8. Recursos del Circuito

### 8.1. Número total de qubits

Resumiendo lo ya visto:

| Variante | Qubits totales |
|----------|----------------|
| Estándar (`order_finding_circuit`) | `4n + 2` |
| Un solo control (`order_finding_circuit_one_control`) | `2n + 3` |

donde `n = ⌈log₂ N⌉`.

Para los `N` del proyecto:

| N | n | Estándar 4n+2 | Un control 2n+3 |
|---|---|---------------|-----------------|
| 15 | 4 | 18 | 11 |
| 21 | 5 | 22 | 13 |
| 35 | 6 | 26 | 15 |
| 77 | 7 | 30 | 17 |
| 143 | 8 | 34 | 19 |

### 8.2. Profundidad del circuito

El análisis de coste, en profundidad de puertas:

- **Una suma de Draper**: O(n) rotaciones + 2 QFTs de O(n²) cada
  una. Total: O(n²) sin QFT amortizada, O(n) si encadenamos sumas
  bajo una sola QFT.
- **Una suma modular** (`c_φ_add_mod`): O(1) sumas → O(n²) puertas.
- **Una multiplicación modular controlada**: hace `n` sumas modulares
  + un swap de O(n) → O(n³) puertas (con QFT amortizable a O(n²
  log n)).
- **Exponenciación modular**: hace `m = 2n` multiplicaciones
  modulares → O(n⁴) puertas en el peor caso, O(n³ log n) con
  optimizaciones.
- **QFT inversa al final**: O(n²) adicional, despreciable.

En la literatura (y para nuestro código con `approx_QFT`
activado) la profundidad efectiva es aproximadamente **O(n³)**.

### 8.3. Por qué Shor es mucho más costoso de simular que Grover

Comparado con Grover (`O(√N)` aplicaciones del oráculo, cada una
relativamente simple), Shor presenta varios desafíos
fundamentalmente más duros para la simulación clásica:

1. **Mayor número de qubits**: para factorizar un `N` de 8 bits
   (`N=143`), usamos 34 qubits. Para Grover sobre 8 bits, basta
   con 8 qubits. La memoria de la simulación crece como `2^q`, así
   que pasar de 8 a 34 qubits multiplica el coste por `2^26`.
2. **Profundidad mucho mayor**: las potencias controladas `U_a^{2^k}`
   suman O(n³) puertas cada una, comparado con el oráculo
   relativamente plano de Grover.
3. **No es simulable con métodos de aproximación de baja
   entrelazamiento**: el estado en mitad de la QPE de Shor está
   altamente entrelazado entre el registro de control y el de
   trabajo. Métodos como tensor networks o MPS (Matrix Product
   States) que funcionan bien para circuitos con poco
   entrelazamiento son ineficaces.
4. **Sensibilidad al ruido**: las rotaciones de fase de la QFT y
   del sumador de Draper tienen amplitudes muy pequeñas; la
   simulación con precisión limitada introduce errores que se
   acumulan rápidamente.

Por eso, factorizar un `N` de 15 bits ya está en el límite de lo
simulable en una estación de trabajo razonable. Y por eso la
ejecución real en hardware cuántico actual sigue restringida a
demostraciones de `N=15, 21, 35` con técnicas adicionales de
mitigación de errores.

---

## Apéndice A. Convenciones de bits y qubits

Una nota técnica importante para reproducir el código: la
convención de orden de bits en `AdderCircuit` (definida al inicio
de `adder.py`) es **little-endian del qubit más bajo al más alto**:

> *quantum integer `x` (small letter): quantum state `|x⟩ = |x_0⟩|x_1⟩
> ... |x_m⟩`, with `x_k ∈ {0, 1}`, where `x = Σ_k x_k 2^k`. It
> corresponds to the binary string "`x_m ... x_1 x_0`" and it is
> ordered in the qubit register as `[qubit_0, qubit_1, ..., qubit_m]`,
> where `qubit_k` is in the state `|x_k⟩`.*

Es decir, `qubit_0` lleva el bit menos significativo y `qubit_n-1`
el más significativo. Esta convención afecta directamente a los
ángulos `2π · X · 2^{i-n}` en el adder de Draper: el bit `i` recibe
la rotación con peso `2^{i-n}`, donde `n = len(y_bits)`. Para `i = n-1`
(el bit más significativo) el peso es `1/2`, y para `i = 0` (el menos
significativo) el peso es `2^{-n}`.

## Apéndice B. Esqueleto mínimo de uso

Para ejecutar el factorizador en una sesión típica:

```python
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from python.qiskit.shor.shor import find_factor

# Configura sampler simulado
sampler = SamplerV2()

# Configura un pass manager para el backend / simulador
pass_manager = generate_preset_pass_manager(optimization_level=1)

# Factoriza N=15 con semilla reproducible
factor = find_factor(
    N=15,
    sampler=sampler,
    pass_manager=pass_manager,
    num_tries=5,
    num_shots_per_trial=20,
    seed=42,
)
print("Factor encontrado:", factor)
```

Con `N=15`, el algoritmo terminará rápido. Con `N=143` o mayores,
la simulación puede tardar varios minutos por intento debido al
crecimiento exponencial de la memoria.

## Apéndice C. Mapa conceptual de archivos

```
python/qiskit/shor/
├── qft.py        →  Clase QFTFullGate, envoltorio sobre QFTGate de Qiskit
│                    con soporte para QFT aproximada y opciones de swap/barriers
│
├── adder.py      →  Clase AdderCircuit con:
│                    - add_classical / c_add_classical (sumas Draper)
│                    - add_classical_modulo / c_add_classical_modulo
│                    - add_quantum / c_add_quantum
│                    - add_quantum_modulo / c_add_quantum_modulo
│                    - multiply_modulo / c_multiply_modulo
│                    - exponentiate_modulo
│
└── shor.py       →  Orquestación de alto nivel:
                     - order_finding_circuit (4n+2 qubits)
                     - order_finding_circuit_one_control (2n+3 qubits)
                     - _get_order_from_dist (post-procesado con fracciones continuas)
                     - find_order (ejecución QPE + extracción de r)
                     - find_factor (bucle clásico externo)

python/cirq/shor/
├── modular_exp.py →  Implementación de ModularExp como cirq.Gate compuesta
└── shor.py        →  Estructura paralela a la de Qiskit, adaptada a la API de Cirq
```

Con este mapa, el lector puede navegar el código sabiendo
exactamente qué hace cada pieza y por qué.

---

## Resumen ejecutivo

El algoritmo de Shor es una composición elegante de tres ideas
profundas:

1. **Una reducción algebraica**: factorizar `N` ≡ encontrar el orden
   de `a ∈ Z_N*`.
2. **Una observación espectral**: el orden `r` aparece como
   denominadores de los autovalores `e^{2πi s/r}` del operador
   "multiplicar por `a` módulo `N`".
3. **Una herramienta cuántica**: la Quantum Phase Estimation,
   construida sobre la QFT, extrae esos autovalores con
   precisión exponencialmente más eficiente que cualquier método
   clásico conocido.

El coste de implementación se concentra casi enteramente en la
**exponenciación modular cuántica**: cómo aplicar `|x⟩|1⟩ → |x⟩|a^x
mod N⟩` con un número manejable de qubits y profundidad polinómica.
La construcción de Beauregard usada en este proyecto resuelve
elegantemente ese problema descomponiéndolo en sumas modulares y,
finalmente, en sumas de Draper en el espacio de Fourier.

Quien comprenda en profundidad las tres ideas de arriba **puede
reescribir el algoritmo de Shor desde cero**. El código de
`qft.py`, `adder.py` y `shor.py` es una de las muchas
posibles traducciones de esas ideas a un framework concreto
(Qiskit), y la versión en Cirq demuestra que el algoritmo es
agnóstico al framework: lo importante son las matemáticas y la
estructura del circuito, no el SDK.
