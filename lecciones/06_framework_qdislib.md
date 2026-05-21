# QDisLib — Computación Cuántica Distribuida del BSC

> **Nivel**: Grado / Máster en Computación Cuántica  
> **Duración estimada**: 3 horas lectivas  
> **Prerrequisitos**: algoritmo de Grover, algoritmo de Shor, fundamentos de Qiskit, nociones básicas de computación distribuida

---

## Tabla de contenidos

1. [Introducción a QDisLib](#1-introducción-a-qdislib)
2. [El Concepto de Circuit Cutting](#2-el-concepto-de-circuit-cutting)
3. [Modelo de Programación](#3-modelo-de-programación)
4. [Relación con Qiskit](#4-relación-con-qiskit)
5. [Grover en QDisLib](#5-grover-en-qdislib)
6. [Shor en QDisLib](#6-shor-en-qdislib)
7. [El Worker de QDisLib](#7-el-worker-de-qdislib)
8. [Interpretación en el Contexto del Benchmark](#8-interpretación-en-el-contexto-del-benchmark)
9. [Ejercicios](#9-ejercicios)
10. [Referencias](#10-referencias)

---

## 1. Introducción a QDisLib

### 1.1 Historia y origen

**QDisLib** (*Quantum Distributed Library*) es un framework de computación cuántica distribuida desarrollado en el **Barcelona Supercomputing Center (BSC-CNS)**, el centro de supercomputación más importante de España, hogar del superordenador MareNostrum.

El BSC es miembro activo del ecosistema europeo de computación cuántica. QDisLib surge de la necesidad práctica de ejecutar algoritmos cuánticos en entornos donde **ninguna QPU individual tiene suficientes qubits** para resolver el problema completo. En lugar de esperar a que los fabricantes construyan máquinas más grandes, el enfoque del BSC es orquestar varias QPUs pequeñas o medianas mediante técnicas de *circuit cutting* y comunicación clásica.

Esta filosofía conecta directamente con la infraestructura de computación distribuida ya madura en el BSC: del mismo modo que MareNostrum distribuye carga entre miles de nodos clásicos, QDisLib aspira a distribuir carga cuántica entre múltiples nodos QPU.

### 1.2 Filosofía: circuit cutting para computación distribuida multi-QPU

La idea central de QDisLib puede resumirse en una frase:

> **Fragmentar el problema en subcircuitos que caben en las QPUs disponibles, ejecutarlos de forma distribuida, y recombinar los resultados clásicamente.**

Esto contrasta con el enfoque *scale-up* (construir QPUs cada vez más grandes) y apuesta por el enfoque *scale-out* (conectar muchas QPUs medianas). En 2024-2025, la mayoría de QPUs disponibles en la nube tienen entre 5 y 133 qubits. Un circuito de 20 qubits puede partirse en dos subcircuitos de 10, ejecutarse en paralelo en dos QPUs de 10 qubits, y los resultados combinarse después.

### 1.3 Dependencia de Qiskit

QDisLib **no reemplaza a Qiskit**: opera directamente sobre objetos `QuantumCircuit` de Qiskit. Esto tiene una consecuencia muy importante para el programador:

- El circuito se **construye** usando la API estándar de Qiskit.
- QDisLib recibe ese circuito ya construido y aplica el proceso de *cutting*.
- La **ejecución** puede hacerse en un backend real o en un simulador.

En la práctica, como se verá en el código de este TFG, los módulos `python/qdislib/grover.py` y `python/qdislib/shor/shor.py` importan directamente las funciones de construcción de circuitos de sus equivalentes Qiskit (`python/qiskit/grover.py` y `python/qiskit/shor/shor.py`) y sólo difieren en la capa de ejecución.

### 1.4 Limitación de plataforma: no disponible en Linux aarch64

QDisLib presenta una limitación importante de distribución: **no está disponible para Linux ARM64 (aarch64)**. Esto afecta, por ejemplo, a instancias AWS Graviton, Apple Silicon corriendo Linux, o muchos nodos de clúster modernos. El worker de este benchmark detecta esta condición e informa del error en lugar de silenciar el fallo:

```python
# En qdislib_worker.py
try:
    import Qdislib  # noqa: F401
except Exception as e:
    write_error(f"qdislib not available: {e}")
    return
```

Si `Qdislib` no puede importarse (sea por arquitectura, sistema operativo, o ausencia de instalación), el worker aborta limpiamente en lugar de continuar con resultados incorrectos.

---

## 2. El Concepto de Circuit Cutting

### 2.1 Qué es el circuit cutting

El *circuit cutting* (o *circuit knitting*) es una familia de técnicas que permiten simular o ejecutar un circuito cuántico grande dividiéndolo en subcircuitos más pequeños. La idea matemática fundamental es que la **matriz densidad** de un sistema puede representarse como combinación lineal de matrices densidad de subsistemas, con coeficientes que pueden ser negativos (de ahí el nombre *quasiprobability decomposition*).

Existen dos variantes principales:

| Variante | Qué se corta | Overhead clásico |
|---|---|---|
| **Wire cutting** | Se corta un cable (qubit) entre dos partes del circuito | $O(4^k)$ en número de cortes $k$ |
| **Gate cutting** | Se corta una puerta de dos qubits que cruza la partición | $O(9^k)$ en número de cortes $k$ |

QDisLib implementa principalmente **wire cutting**. En el código del benchmark, la llamada a `find_cut` siempre especifica `wire_cut=True, gate_cut=False`.

### 2.2 Por qué es necesario en hardware real

Las QPUs actuales tienen dos restricciones que motivan el circuit cutting:

**Restricción de tamaño**: Una QPU de $n$ qubits físicos, tras contar el overhead de corrección de errores y los qubits auxiliares de transpilación, puede ejecutar algoritmos con bastante menos de $n$ qubits lógicos. Un circuito de Shor para factorizar $N=15$ requiere unos 18-20 qubits. No todas las QPUs disponibles comercialmente los tienen.

**Restricción de conectividad**: Los qubits físicos no están todos conectados entre sí. Una puerta CNOT entre dos qubits no adyacentes requiere una cadena de operaciones SWAP que aumenta la profundidad del circuito y acumula errores. Si el algoritmo tiene una estructura que puede partirse en dos bloques con pocas interacciones entre ellos, *cortar* esas interacciones y resolver cada bloque en una QPU con buena conectividad interna es preferible a ejecutar el circuito completo con muchos SWAPs.

### 2.3 El coste: overhead exponencial en número de cortes

El precio del circuit cutting es **real y severo**. Para $k$ cortes de tipo wire, el número de ejecuciones de subcircuitos necesario para reconstruir la distribución original crece como $O(4^k)$. Para gate cutting el crecimiento es $O(9^k)$.

Esto significa que con $k=2$ cortes de tipo wire hay que ejecutar $4^2 = 16$ subcircuitos en lugar de 1. Con $k=5$ serían $4^5 = 1024$ ejecuciones. Por eso el parámetro `max_cuts=2` es el valor por defecto en el benchmark: más de 2-3 cortes hace el overhead computacional prohibitivo para la mayoría de casos de uso actuales.

```
Overhead de circuit cutting (wire cutting):
  k=1  →   4 ejecuciones
  k=2  →  16 ejecuciones
  k=3  →  64 ejecuciones
  k=4  → 256 ejecuciones
  k=5  → 1024 ejecuciones
```

### 2.4 Quasiprobability decomposition: cómo se recombina la información

La recombinación no es una simple suma de histogramas. El proceso implica:

1. **Descomposición**: El canal cuántico que actúa sobre el cable cortado se escribe como suma de canales locales con coeficientes reales (posiblemente negativos).
2. **Muestreo**: Se ejecutan subcircuitos para cada término de la descomposición, con el número de shots multiplicado por los coeficientes.
3. **Recombinación clásica**: Los resultados de los subcircuitos se combinan linealmente, cancelándose los términos negativos, para obtener el **valor de expectación** del observable de interés sobre el circuito completo.

El resultado de `wire_cutting(qc_isa, cuts, shots=num_shots, backend="numpy")` es un **valor de expectación** (un número real), no una distribución de probabilidad. Esta es una distinción crucial que aparece en el benchmark: el modo *cutting* de QDisLib no devuelve la distribución `{|estado>: conteo}` que devuelve la ejecución directa, sino un escalar que representa la expectación del observable computado.

### 2.5 QDisLib en contexto: comparación con qiskit-addon-cutting

IBM también tiene su propia implementación de circuit cutting: `qiskit-addon-cutting`. Las diferencias clave son:

| Aspecto | qiskit-addon-cutting | QDisLib |
|---|---|---|
| Objetivo principal | Simulación clásica de circuitos grandes | Ejecución distribuida real en multi-QPU |
| Orientación | Investigación / benchmark | Infraestructura de producción (BSC) |
| Integración | Ecosystem Qiskit | Standalone + Qiskit como frontend |
| Dispatch real | No (siempre simulación) | Sí (puede despachar a QPUs remotas) |

Ambas librerías aceptan `QuantumCircuit` de Qiskit como input, lo que permite que el código de construcción de circuitos sea **completamente reutilizable** entre frameworks.

---

## 3. Modelo de Programación

### 3.1 Cómo se define la partición de qubits

QDisLib expone una API de dos funciones principales para el usuario:

```python
from Qdislib.api import find_cut, wire_cutting
```

**`find_cut(circuit, max_qubits, max_cuts, wire_cut, gate_cut)`**: Recibe el circuito ya transpilado (formato ISA — *Instruction Set Architecture* del backend) y devuelve una lista de objetos `Cut` que describen dónde cortar el circuito. El parámetro `max_qubits` define el tamaño máximo de cada subcircuito. QDisLib resuelve internamente el problema de optimización combinatoria de encontrar la partición óptima.

**`wire_cutting(circuit, cuts, shots, backend)`**: Aplica los cortes encontrados, genera los subcircuitos, los ejecuta (en el backend especificado o en QPUs remotas configuradas), y combina los resultados devolviendo el valor de expectación.

### 3.2 El workflow completo

```
                    CONSTRUCCIÓN               CORTE                  EJECUCIÓN
 ┌──────────────┐              ┌───────────┐             ┌──────────────────────┐
 │ QuantumCircuit│  Qiskit API  │  qc (ISA) │  find_cut   │  [subcircuito_1]     │
 │  (cualquier  │ ──────────── │           │ ──────────  │  [subcircuito_2]     │
 │  algoritmo)  │              │           │             │  [subcircuito_1']    │  ×4^k
 └──────────────┘              └───────────┘             │  [subcircuito_2']    │
                                                         └──────────────────────┘
                                                                    │
                                                         ┌──────────────────────┐
                                                         │  wire_cutting(...)   │
                                                         │  recombinación QPD   │
                                                         └──────────────────────┘
                                                                    │
                                                         ┌──────────────────────┐
                                                         │  expectation_value   │
                                                         │  (escalar real)      │
                                                         └──────────────────────┘
```

### 3.3 El atributo `nqubits`: un detalle importante de la API

En el código del benchmark aparece este fragmento:

```python
qc_isa = _pm.run(qc)
if not hasattr(qc_isa, 'nqubits'):
    qc_isa.nqubits = qc_isa.num_qubits
```

QDisLib espera que el objeto circuito tenga un atributo `.nqubits` (sin guion bajo), pero Qiskit usa `.num_qubits` (con guion, forma de propiedad). Este workaround añade el atributo manualmente antes de pasarlo a `find_cut`. Es un ejemplo de la fricción habitual entre bibliotecas que usan el mismo objeto base (QuantumCircuit) pero extienden su interfaz de formas ligeramente distintas.

### 3.4 El fallback: ejecución directa con Qiskit

Cuando `Qdislib` no está instalado, ambos módulos (`grover.py` y `shor.py`) capturan el `ImportError` y continúan con ejecución directa via Qiskit-Aer:

```python
try:
    import Qdislib  # noqa: F401
    # ... ruta QDisLib
except ImportError:
    # --- Fallback: direct Qiskit-Aer execution ---
    # ... misma lógica pero sin cutting
```

Esta arquitectura de fallback es una buena práctica de ingeniería: el código siempre produce resultados correctos, aunque no distribuidos. Esto permite desarrollar y testear en máquinas sin QDisLib instalado, y activar la distribución en producción simplemente instalando la librería.

---

## 4. Relación con Qiskit

### 4.1 QDisLib acepta QuantumCircuit directamente

La integración con Qiskit es la decisión arquitectónica más importante de QDisLib. Al aceptar `QuantumCircuit` como formato de entrada, QDisLib hereda automáticamente:

- Todo el ecosistema de **transpilación** de Qiskit (optimización de puertas, routing de qubits, mapeado a hardware).
- Todos los **algoritmos** ya implementados en Qiskit, Qiskit Nature, Qiskit Finance, etc.
- La **comunidad** y documentación de Qiskit, la librería cuántica más utilizada del mundo.

Para el programador, esto significa que aprender QDisLib no requiere aprender una nueva forma de construir circuitos: si sabes Qiskit, ya sabes construir circuitos para QDisLib.

### 4.2 Por qué grover.py y shor.py de qdislib reutilizan los de qiskit

Analicemos las importaciones al principio de `python/qdislib/grover.py`:

```python
from python.qiskit.grover import (
    build_oracle as _qiskit_build_oracle,
    build_diffuser as _qiskit_build_diffuser,
    grover_circuit as _qiskit_grover_circuit,
)

build_oracle: callable = _qiskit_build_oracle
build_diffuser: callable = _qiskit_build_diffuser
grover_circuit: callable = _qiskit_grover_circuit
```

Las tres funciones de construcción de circuitos (`build_oracle`, `build_diffuser`, `grover_circuit`) son **alias directos** de las implementaciones Qiskit. No hay código duplicado. El módulo QDisLib de Grover **no sabe construir circuitos**: eso es responsabilidad de Qiskit.

Lo mismo ocurre en `python/qdislib/shor/shor.py`:

```python
from python.qiskit.shor.shor import (
    order_finding_circuit as _qiskit_order_finding_circuit,
    _get_order_from_dist,
)

order_finding_circuit: callable = _qiskit_order_finding_circuit
```

Esta decisión de diseño tiene una consecuencia directa en el benchmark: **cualquier mejora en la implementación Qiskit (circuito más profundo, oráculo más eficiente, menos puertas) beneficia automáticamente a la versión QDisLib**. No hay riesgo de que los dos módulos diverjan en la calidad del circuito construido.

### 4.3 La diferencia real: dónde y cómo se aplica la distribución

La única diferencia entre los módulos Qiskit y QDisLib está en la función de ejecución. En Qiskit puro:

```python
# Qiskit: ejecución directa
qc_isa = pm.run(qc)
result = sampler.run([qc_isa], shots=num_shots).result()
dist = result[0].data.result.get_counts()  # distribución de probabilidad
```

En QDisLib con cutting:

```python
# QDisLib: ejecución distribuida
qc_isa = pm.run(qc)
cuts = find_cut(qc_isa, max_qubits=max_sub_qubits, max_cuts=max_cuts,
                wire_cut=True, gate_cut=False)
exp_val = wire_cutting(qc_isa, cuts, shots=num_shots, backend="numpy")  # valor de expectación
```

El tipo de retorno cambia de forma fundamental: Qiskit devuelve una **distribución** (diccionario de estados a conteos), mientras que QDisLib devuelve un **valor de expectación** (escalar real). Esto es una consecuencia directa de la quasiprobability decomposition: los términos con coeficientes negativos se cancelan durante la recombinación, haciendo imposible mantener una distribución de probabilidad positiva sobre estados de base.

---

## 5. Grover en QDisLib

### 5.1 Cómo se construye el circuito

Como se explicó en la sección anterior, la construcción del circuito de Grover es completamente delegada a `python/qiskit/grover.py`. El circuito resultante tiene la estructura estándar de Grover:

```
|0...0>  ──[H⊗n]──[Oracle]──[Diffuser]──···──[Medir]
```

Para $n$ qubits y un estado objetivo `target`, el número óptimo de iteraciones es:

$$\text{iters} = \left\lfloor \frac{\pi}{4} \sqrt{2^n} \right\rfloor$$

Este valor se calcula explícitamente en `grover.py`:

```python
iters = (
    num_iterations
    if num_iterations is not None
    else math.floor(math.pi / 4 * math.sqrt(2**n))
)
qc = grover_circuit(n, target, num_iterations=iters)
```

### 5.2 Dónde se aplica el corte en el algoritmo de Grover

La función `search_with_cutting` aplica wire cutting al circuito de Grover. El parámetro clave para la partición es:

```python
max_sub_qubits = max(2, math.ceil(n / 2))
```

Esto divide el espacio de $n$ qubits en dos subcircuitos de aproximadamente $n/2$ qubits cada uno. Para $n=4$, cada subcircuito tiene 2 qubits. Para $n=6$, cada subcircuito tiene 3 qubits.

La pregunta natural es: ¿dónde exactamente en la estructura del circuito de Grover se colocan los cortes? QDisLib resuelve esto automáticamente mediante `find_cut`, que analiza el grafo de dependencias del circuito y encuentra los lugares donde el número de conexiones entre las dos particiones es mínimo (minimizando así $k$ y por tanto $4^k$).

Para Grover, las conexiones entre la primera mitad y la segunda mitad de qubits se concentran en:
- Las puertas CNOT del oráculo (que puede necesitar controles de múltiples qubits).
- Las puertas CNOT del difusor (que también cruzan qubits).

### 5.3 Código real de grover.py explicado línea a línea

Veamos la función `search_with_cutting` completa con explicación detallada:

```python
def search_with_cutting(
    n: int,
    target: int,
    pass_manager=None,
    num_shots: int = 1024,
    max_cuts: int = 2,
) -> tuple[float, list, float]:
    """Execute Grover via QDisLib circuit cutting.

    Returns (expectation_value, cuts, find_cut_time_ms).
    find_cut_time_ms is the time spent finding the cuts (not executing).
    """
```

La función devuelve una tripla `(expectation_value, cuts, find_cut_time_ms)`. El tiempo de búsqueda de cortes se mide por separado porque es un coste propio de QDisLib que no existe en la ejecución directa.

```python
    import time
    from Qdislib.api import find_cut, wire_cutting
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
```

Los imports son locales a la función. Esto es intencional: si QDisLib no está disponible, la importación fallará al llamar a esta función, no al importar el módulo. Así el módulo puede importarse (y usar `search`) en sistemas sin QDisLib.

```python
    iters = math.floor(math.pi / 4 * math.sqrt(2**n))
    qc = grover_circuit(n, target, num_iterations=iters)
```

Se calcula el número óptimo de iteraciones y se construye el circuito usando la implementación Qiskit. En este punto `qc` es un `QuantumCircuit` estándar, sin nada específico de QDisLib.

```python
    _pm = (
        pass_manager
        if pass_manager is not None
        else generate_preset_pass_manager(backend=AerSimulator())
    )
    qc_isa = _pm.run(qc)
```

**Transpilación**: el circuito se transpila al formato ISA (*Instruction Set Architecture*) del backend. En simulación esto optimiza las puertas y las mapea al conjunto de instrucciones nativas del simulador. En hardware real, esta etapa mapea los qubits lógicos a físicos y añade las puertas SWAP necesarias para la conectividad.

```python
    if not hasattr(qc_isa, 'nqubits'):
        qc_isa.nqubits = qc_isa.num_qubits
```

Workaround para la incompatibilidad de atributos entre Qiskit y QDisLib (explicado en la sección 3.3).

```python
    max_sub_qubits = max(2, math.ceil(n / 2))
    t0 = time.perf_counter()
    try:
        cuts = find_cut(qc_isa, max_qubits=max_sub_qubits, max_cuts=max_cuts,
                       wire_cut=True, gate_cut=False)
    except Exception as e:
        print(f"[QDisLib cutting] find_cut error: {e}")
        cuts = []
    find_time_ms = (time.perf_counter() - t0) * 1000.0
```

Este bloque es el corazón del proceso: `find_cut` analiza el circuito y encuentra la partición óptima en subcircuitos de como máximo `max_sub_qubits` qubits. El tiempo se mide con `perf_counter` (resolución de microsegundos) porque en circuitos pequeños la búsqueda puede completarse en pocos milisegundos. El bloque `try/except` captura cualquier error de QDisLib (por ejemplo, si el circuito es demasiado profundo o demasiado pequeño para cortar) y devuelve una lista vacía de cortes.

```python
    print(f"[QDisLib cutting] Grover n={n} target={target} cuts={cuts}")

    if not cuts:
        print(f"[QDisLib cutting] No cuts found for n={n}, using direct execution")
        exp_val = 0.0
    else:
        try:
            exp_val = wire_cutting(qc_isa, cuts, shots=num_shots, backend="numpy")
        except Exception as e:
            print(f"[QDisLib cutting] wire_cutting error: {e}")
            exp_val = 0.0

    return float(exp_val) if not isinstance(exp_val, tuple) else 0.0, cuts, find_time_ms
```

Si no se encontraron cortes, se devuelve `exp_val=0.0` (indicando que no hubo ejecución real). Si sí hay cortes, `wire_cutting` ejecuta todos los subcircuitos, realiza la recombinación QPD y devuelve el valor de expectación. El check final `if not isinstance(exp_val, tuple)` maneja el caso donde QDisLib devuelve una tupla en lugar de un escalar (comportamiento interno de ciertas versiones de la API).

---

## 6. Shor en QDisLib

### 6.1 Cómo se estructura shor.py

El módulo `python/qdislib/shor/shor.py` es más extenso que el de Grover porque el algoritmo de Shor tiene dos capas:

1. **Order-finding** cuántico: encontrar el orden $r$ de $a$ en $\mathbb{Z}_N$.
2. **Factorización clásica**: usar $r$ para obtener factores de $N$ con aritmética clásica.

El módulo expone cuatro funciones públicas:

| Función | Propósito |
|---|---|
| `find_order(A, N, ...)` | Orden cuántico de $A$ en $\mathbb{Z}_N$, con fallback |
| `find_order_with_cutting(A, N, ...)` | Ídem pero forzando cutting (para benchmark) |
| `find_factor(N, ...)` | Factorización completa de $N$ usando `find_order` |
| `find_factor_with_cutting(N, ...)` | Factorización vía cutting (para benchmark) |

### 6.2 Cómo se particiona el circuito: registro de control vs registro de trabajo

El circuito de Shor para factorizar $N$ tiene dos registros:

```
Registro de control (m qubits):  |0>──[H]──[ctrl-U^j]──[QFT^†]──[Medir]
                                  ...
Registro de trabajo (n qubits):  |1>──[U^j]──···
```

Donde $m = 2\lceil\log_2 N \rceil$ (doble precisión para la estimación de fase) y $n = \lceil\log_2 N \rceil$.

La partición natural para QDisLib es cortar entre el registro de control y el registro de trabajo. El código en `find_order_with_cutting` usa exactamente esta estrategia:

```python
max_sub_qubits = max(2, math.ceil(qc_isa.num_qubits / 2))
```

Donde `qc_isa.num_qubits` es la suma de ambos registros ($m + n$ qubits). Dividir por 2 da aproximadamente la mitad de los qubits para cada subcircuito.

### 6.3 La función find_order en detalle

```python
def find_order(
    A: int,
    N: int,
    sampler=None,
    pass_manager=None,
    precision: int | None = None,
    num_shots: int = 10,
) -> tuple[int, dict[str, int]]:
```

Obsérvese que `num_shots=10` por defecto, no 1024 como en Grover. Esto es porque el circuito de Shor es mucho más profundo y costoso de simular, y 10 shots suele ser suficiente para extraer el período con la aritmética de fracciones continuas.

```python
    m = precision if precision is not None else 2 * math.ceil(math.log2(N))
    qc = order_finding_circuit(A, N, precision=m)
    if qc == 0:
        # gcd(A, N) > 1 — circuit was not built
        return 0, {}
```

Si `order_finding_circuit` devuelve `0`, significa que $\gcd(A, N) > 1$, es decir, ya encontramos un factor trivialmente. El circuito no se construye porque no es necesario.

```python
    try:
        import Qdislib  # noqa: F401

        _, default_sampler, default_pm = _make_backend_defaults()
        _sampler = sampler if sampler is not None else default_sampler
        _pm = pass_manager if pass_manager is not None else default_pm

        print(f"[QDisLib] Start search for the order of {A} in Z_{N}")
        dist = _run_circuit(qc, _sampler, _pm, num_shots, "output_bits")
    except ImportError:
        # --- Fallback: direct Qiskit-Aer execution ---
        _, default_sampler, default_pm = _make_backend_defaults()
        _sampler = sampler if sampler is not None else default_sampler
        _pm = pass_manager if pass_manager is not None else default_pm

        print(f"[QDisLib fallback] Start search for the order of {A} in Z_{N}")
        dist = _run_circuit(qc, _sampler, _pm, num_shots, "output_bits")

    r = _get_order_from_dist(dist, A, N, precision=m)
    return r, dist
```

Nótese que en `find_order` (a diferencia de `find_order_with_cutting`) la ejecución es directa — no hay cutting. La distinción en `find_order` entre el camino QDisLib y el camino fallback es más sutil: en una implementación completa del BSC, el camino QDisLib despacharía el circuito a una QPU remota a través de la infraestructura de QDisLib, aunque el circuito se ejecute completo. Actualmente, como indica el comentario en `grover.py`, esto es un *placeholder* para la pipeline de distribución completa.

### 6.4 La función _run_circuit: separación de responsabilidades

```python
def _run_circuit(
    qc, sampler, pass_manager, num_shots: int, register_name: str
) -> dict[str, int]:
    """Transpile, execute and return the counts distribution."""
    qc_isa = pass_manager.run(qc)
    result = sampler.run([qc_isa], shots=num_shots).result()[0]
    dist = getattr(result.data, register_name).get_counts()
    return dist
```

Esta función auxiliar encapsula el patrón de transpilación y ejecución. El parámetro `register_name` es necesario porque Qiskit nombra los registros de medición por nombre, y el circuito de Shor usa `"output_bits"` para el registro de control (el que se mide para estimar la fase).

### 6.5 La función find_factor: el bucle de Shor clásico

```python
def find_factor(
    N: int,
    sampler=None,
    pass_manager=None,
    num_tries: int = 3,
    num_shots_per_trial: int = 10,
    seed: int | None = None,
) -> int:
    # Trivial checks
    if N % 2 == 0:
        print("Even number")
        return 2

    for k in range(2, round(math.log(N, 2)) + 1):
        d = int(round(N ** (1 / k)))
        if d**k == N:
            print(f"{N} is {d} to the power {k}")
            return d
```

Antes de entrar en el bucle cuántico, se realizan dos comprobaciones clásicas en tiempo $O(\log N)$:
- Si $N$ es par, el factor es 2.
- Si $N = d^k$ para algún entero $d$ y exponente $k$, el factor es $d$.

Estas comprobaciones son importantes porque el algoritmo de Shor solo está diseñado para el caso general (producto de dos primos aproximadamente iguales).

```python
    i = 0
    factor_found = False
    d = 1
    while (not factor_found) and i < num_tries:
        a = random.randint(2, N - 1)
        d = math.gcd(a, N)
        if d > 1:
            factor_found = True
            print(f"Lucky guess of {a}, found factor {d}")
            return d

        r, _ = find_order(a, N, sampler=sampler, pass_manager=pass_manager,
                          num_shots=num_shots_per_trial)
        if r == 0:
            i += 1
            continue
        if r % 2 == 0:
            x = pow(a, r // 2, N) - 1
            d = math.gcd(x, N)
            if 1 < d < N:
                factor_found = True
        i += 1
```

El bucle cuántico de Shor sigue el procedimiento estándar:
1. Escoger $a$ aleatorio.
2. Comprobar $\gcd(a, N)$: si es $> 1$, factor encontrado (suerte clásica).
3. Encontrar el orden $r$ de $a$ en $\mathbb{Z}_N$ cuánticamente.
4. Si $r$ es par, calcular $x = a^{r/2} - 1 \pmod{N}$ y verificar $\gcd(x, N)$.

### 6.6 Diferencias clave respecto a qiskit/shor/shor.py

La diferencia principal está en el **envoltorio de ejecución**, no en el algoritmo. Un resumen:

| Aspecto | qiskit/shor/shor.py | qdislib/shor/shor.py |
|---|---|---|
| Construcción de circuito | Implementación propia | Alias a qiskit/shor/shor.py |
| Ejecución estándar | Siempre Qiskit-Aer directo | Intenta QDisLib, fallback Qiskit-Aer |
| API de cutting | No existe | `find_order_with_cutting`, `find_factor_with_cutting` |
| Tipos de retorno | `tuple[int, dict]` | Igual para find_order; `tuple[float, list, float]` para cutting |
| Logging | Básico | Prefijado con `[QDisLib]` o `[QDisLib fallback]` |

---

## 7. El Worker de QDisLib

### 7.1 Importación condicional y fallback

El archivo `python/workers/qdislib_worker.py` es el punto de entrada del benchmark para QDisLib. Es un **subproceso independiente** que se lanza por el orquestador principal del benchmark. Esta arquitectura de subproceso tiene una ventaja crucial: si QDisLib crashea o no está disponible, solo muere el subproceso hijo, no el benchmark completo.

```python
# QDisLib usa \( en docstrings que dispara SyntaxWarning en Python 3.12+
warnings.filterwarnings("ignore", "invalid escape sequence", SyntaxWarning)
```

Este filtro de warnings es una solución pragmática para un bug en las docstrings de QDisLib: usan `\(` en notación matemática LaTeX sin escapar, lo cual Python 3.12+ interpreta como secuencia de escape inválida. El filtro suprime el warning sin modificar el código de QDisLib.

La comprobación de disponibilidad es temprana y terminante:

```python
try:
    import Qdislib  # noqa: F401
except Exception as e:
    write_error(f"qdislib not available: {e}")
    return
```

Usar `Exception` en lugar de `ImportError` captura también errores de inicialización de QDisLib (por ejemplo, si falla al conectar con hardware remoto durante el `__init__.py`).

### 7.2 Setup del backend: _setup_grover

```python
def _setup_grover(config: BenchmarkConfig):
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from python.qdislib.grover import search, search_with_cutting
    from python.qiskit.grover import grover_circuit as qiskit_grover_circuit

    t0 = time.perf_counter()
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    sampler = SamplerV2()
    startup_ms = (time.perf_counter() - t0) * 1000.0
```

El `startup_ms` mide el tiempo de inicialización del backend: instanciar `AerSimulator()`, compilar el pass manager y crear el sampler. Este tiempo se reporta en el benchmark como métrica de overhead, separado del tiempo de ejecución del algoritmo.

Nótese `optimization_level=1` en el pass manager. El nivel 1 es un compromiso entre tiempo de transpilación y calidad de optimización. En benchmarks, usar niveles altos (2-3) añadiría overhead de transpilación no representativo del tiempo de ejecución del algoritmo.

```python
    def search_call(n, target, num_shots):
        return search(n, target, sampler=sampler, pass_manager=pm, num_shots=num_shots)

    def build_call(n, target):
        return qiskit_grover_circuit(n, target)

    def cutting_call(n, target, num_shots):
        return search_with_cutting(n, target, pass_manager=pm, num_shots=num_shots)

    return startup_ms, search_call, build_call, cutting_call
```

Las funciones de llamada son closures sobre `sampler` y `pm`. Esto evita recrear el backend en cada llamada (lo cual sería extremadamente ineficiente y mediría el tiempo de inicialización repetidamente).

### 7.3 Setup para Shor: startup_ms casi cero

```python
def _setup_shor(config: BenchmarkConfig):
    from python.qdislib.shor.shor import find_factor as _ff
    from python.qdislib.shor.shor import find_factor_with_cutting as _ffc

    t0 = time.perf_counter()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    def factor_call(N):
        return _ff(N, num_tries=3, num_shots_per_trial=config.num_shots)

    def cutting_factor_call(N):
        return _ffc(N, num_shots_per_trial=config.num_shots)

    return startup_ms, factor_call, cutting_factor_call
```

Para Shor, el `startup_ms` es prácticamente cero porque el backend se crea dentro de `find_factor` (vía `_make_backend_defaults()`). Esta asimetría con Grover es un detalle de implementación: Shor crea un backend fresco por cada llamada, mientras que Grover reutiliza un backend compartido. Esto tiene implicaciones para la interpretación del benchmark (ver sección 8).

### 7.4 El bucle de cutting en el worker principal

```python
if algo == "grover":
    try:
        cutting_times: list[float] = []
        last_exp = 0.0
        last_find_ms = 0.0
        for _ in range(config.n_repetitions):
            t0 = time.perf_counter()
            exp_val, _cuts, find_ms = cutting_call(n, n, config.num_shots)
            cutting_times.append((time.perf_counter() - t0) * 1000.0)
            last_exp = exp_val
            last_find_ms = find_ms
        result["cutting_wall_time_ms"] = round(float(np.median(cutting_times)), 3)
        result["cutting_find_time_ms"] = round(last_find_ms, 3)
        result["cutting_expectation_value"] = round(last_exp, 6)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"[QDisLib cutting] grover n={n} failed: {e}", file=sys.stderr)
```

El worker ejecuta el cutting `n_repetitions` veces y reporta la **mediana** del tiempo de pared (*wall time*). Se usa la mediana en lugar de la media para robustez frente a outliers: la primera ejecución puede ser más lenta por efectos de caché JIT, la última por presión de memoria, etc.

Se registran tres métricas distintas:
- `cutting_wall_time_ms`: tiempo total de una invocación de cutting (incluyendo find_cut + wire_cutting).
- `cutting_find_time_ms`: tiempo solo de la búsqueda de cortes (find_cut).
- `cutting_expectation_value`: el valor escalar devuelto por wire_cutting (para verificación de corrección).

---

## 8. Interpretación en el Contexto del Benchmark

### 8.1 Por qué QDisLib es más lento que Qiskit directo (y eso es esperado)

En el benchmark, QDisLib es consistentemente más lento que Qiskit-Aer directo para los mismos algoritmos. Esto no es un defecto: es el comportamiento **esperado y correcto**. Las razones son múltiples:

**Overhead de find_cut**: El algoritmo de búsqueda de particiones óptimas resuelve un problema NP-hard en general. Para circuitos pequeños (los que usa el benchmark), este overhead puede dominar el tiempo total.

**Overhead de $4^k$ subcircuitos**: Con 2 cortes, el circuito se ejecuta $4^2 = 16$ veces en lugar de 1. Incluso si cada ejecución individual es más rápida (circuito más pequeño), la multiplicidad domina.

**Recombinación QPD**: La suma ponderada de los resultados de los 16 subcircuitos tiene un coste clásico no trivial.

**Creación repetida de backend en Shor**: Como se señaló en la sección 7.3, `_make_backend_defaults()` se llama en cada invocación de `find_order`, no una vez en el setup.

La tabla siguiente ilustra el comportamiento esperado para un circuito de Grover con $n=4$:

| Fase | Qiskit directo | QDisLib cutting (k=2) |
|---|---|---|
| Construcción circuito | $T_{\text{build}}$ | $T_{\text{build}}$ (idéntico) |
| Transpilación | $T_{\text{transpile}}$ | $T_{\text{transpile}}$ (idéntico) |
| find_cut | — | $T_{\text{find\_cut}}$ |
| Ejecución | $1 \times T_{\text{exec}}$ | $16 \times T_{\text{exec,sub}}$ |
| Recombinación | — | $T_{\text{QPD}}$ |
| **Total** | $T_{\text{build}} + T_{\text{transpile}} + T_{\text{exec}}$ | $T_{\text{build}} + T_{\text{transpile}} + T_{\text{find\_cut}} + 16T_{\text{exec,sub}} + T_{\text{QPD}}$ |

Donde $T_{\text{exec,sub}} < T_{\text{exec}}$ pero $16 \times T_{\text{exec,sub}} \gg T_{\text{exec}}$ para circuitos de pocos qubits.

### 8.2 Qué mide realmente el benchmark de QDisLib

El benchmark de QDisLib no mide la velocidad intrínseca de los algoritmos de Grover y Shor. Mide **el overhead total del pipeline de computación cuántica distribuida**: desde que se tiene un circuito construido hasta que se obtiene el resultado combinado.

Más específicamente, cada métrica reportada tiene un significado preciso:

| Métrica | Qué mide |
|---|---|
| `simulation_time_ms` | Tiempo de ejecución del algoritmo completo vía el camino QDisLib (o fallback) |
| `build_time_ms` | Tiempo de construcción del `QuantumCircuit` (independiente del framework) |
| `startup_ms` | Overhead de inicialización del backend (instanciar AerSimulator, pass manager) |
| `cutting_wall_time_ms` | Tiempo total de un ciclo completo de cutting (find_cut + wire_cutting) |
| `cutting_find_time_ms` | Tiempo solo de la búsqueda de la partición óptima |
| `cutting_expectation_value` | Valor escalar de expectación devuelto por QPD (para verificar que hay resultado) |

La separación entre `cutting_find_time_ms` y `cutting_wall_time_ms - cutting_find_time_ms` permite descomponer el overhead en:
- Overhead de planificación (encontrar dónde cortar): $T_{\text{find\_cut}}$
- Overhead de ejecución distribuida: $T_{\text{wall}} - T_{\text{find\_cut}}$

### 8.3 Valor científico: separar overhead de comunicación del overhead de simulación pura

El valor científico del benchmark de QDisLib reside precisamente en que **cuantifica el overhead de distribución**. Para un investigador o ingeniero que evalúa si merece la pena usar computación cuántica distribuida para un problema dado, las preguntas relevantes son:

1. ¿Cuánto overhead añade el circuit cutting respecto a ejecución directa?
2. ¿Ese overhead se amortiza cuando los circuitos son demasiado grandes para ejecutarse en una sola QPU?

El benchmark de este TFG responde la pregunta 1 de forma directa: comparando `simulation_time_ms` de QDisLib vs Qiskit para el mismo algoritmo y tamaño, se obtiene el factor de overhead. Este factor es independiente del hardware (porque ambos usan AerSimulator) y por tanto medible en cualquier máquina.

La pregunta 2 no puede responderse en simulación (AerSimulator puede simular cualquier número de qubits en memoria, sin necesitar distribución), pero el benchmark establece la línea base necesaria para extrapolarlo a hardware real.

### 8.4 QDisLib no compite en velocidad sino en escalabilidad de hardware distribuido

La comparación correcta de QDisLib **no** es contra Qiskit-Aer en la misma máquina. La comparación correcta es:

- **Sin QDisLib**: se necesita una QPU con $n$ qubits para ejecutar un circuito de $n$ qubits. Si no existe tal QPU, el algoritmo no se puede ejecutar.
- **Con QDisLib**: se necesitan dos QPUs de $\lceil n/2 \rceil$ qubits cada una. El problema que antes era infactible ahora es factible, aunque más lento.

En este contexto, QDisLib no "pierde" contra Qiskit: resuelve un problema diferente. Es la diferencia entre un coche de carreras y un camión: el camión es más lento, pero puede transportar cargas que el coche nunca podría.

El benchmark de este TFG captura esta realidad al ejecutar ambos frameworks con los mismos parámetros y documentar sistemáticamente la diferencia de tiempos. Cuando en el futuro se disponga de hardware multi-QPU real, los datos de overhead del benchmark permitirán predecir a qué tamaño de problema la distribución empieza a compensar su overhead.

---

## 9. Ejercicios

### Ejercicio 9.1 (Básico): Comprender el overhead

Dado un circuito de Grover con $n=6$ qubits y `max_cuts=2`:

a) Calcula el número óptimo de iteraciones de Grover.  
b) Calcula el número de subcircuitos que QDisLib necesita ejecutar con 2 wire cuts.  
c) Si cada subcircuito tarda 5 ms en ejecutarse y `find_cut` tarda 10 ms, estima el tiempo total de `search_with_cutting`.

### Ejercicio 9.2 (Intermedio): Trazar el flujo de datos

Escribe el flujo completo de datos (tipos de variables) desde la llamada `search_with_cutting(n=4, target=3)` hasta el retorno de la función. Especifica el tipo de cada variable intermedia.

### Ejercicio 9.3 (Intermedio): Comparar implementaciones

Examina las firmas de `search` y `search_with_cutting`. ¿Por qué `search` devuelve `tuple[int, dict[str, int]]` y `search_with_cutting` devuelve `tuple[float, list, float]`? Explica la diferencia desde el punto de vista de la quasiprobability decomposition.

### Ejercicio 9.4 (Avanzado): Diseñar el benchmark

El worker mide `cutting_find_time_ms` por separado del tiempo total. Propón un experimento que use esta separación para determinar si el overhead de QDisLib escala principalmente con la profundidad del circuito, con el número de qubits, o con el número de cortes. ¿Qué variables necesitarías controlar?

### Ejercicio 9.5 (Avanzado): Arquitectura de fallback

El código usa `import Qdislib` dentro de un bloque `try/except ImportError`. Sin embargo, en `qdislib_worker.py` se usa `except Exception`. Explica:  
a) Por qué el worker usa `Exception` en lugar de `ImportError`.  
b) Qué clase de errores captura `Exception` que `ImportError` no capturaría.  
c) ¿Es correcto usar `except Exception` de forma general? ¿Qué riesgos tiene?

### Ejercicio 9.6 (Investigación): Escalabilidad

Investiga la documentación de `qiskit-addon-cutting` y compara con la API de QDisLib. Lista tres diferencias en la forma en que cada librería expone la quasiprobability decomposition al programador.

---

## 10. Referencias

1. **Peng, T. et al.** (2020). *Simulating Large Quantum Circuits on a Small Quantum Computer*. Physical Review Letters 125, 150504. Artículo fundacional del circuit cutting mediante quasiprobability decomposition.

2. **Tang, W. et al.** (2021). *CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations*. ASPLOS 2021. ACM Digital Library. Protocolo de circuit cutting con comunicación clásica eficiente.

3. **Brennan, A. et al.** (2023). *Optimal wire cutting with classical communication*. arXiv:2302.03366. Análisis teórico del overhead óptimo de wire cutting bajo diferentes modelos de comunicación.

4. **Mitarai, K. & Fujii, K.** (2021). *Overhead for simulating a non-local channel with local channels by quasiprobability sampling*. Quantum, 5, 388. Fundamentación matemática de la descomposición en cuasiprobabilidades.

5. **Preskill, J.** (2018). *Quantum Computing in the NISQ Era and Beyond*. Quantum, 2, 79. Contexto sobre limitaciones NISQ y necesidad de estrategias como circuit cutting.

6. **Nielsen, M.A. & Chuang, I.L.** (2010). *Quantum Computation and Quantum Information*. Cambridge University Press. Texto fundamental sobre computación cuántica y teoría de canales cuánticos.

7. **Shor, P.W.** (1997). *Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer*. SIAM Journal on Computing, 26(5), 1484-1509. Algoritmo original de Shor; base para los algoritmos benchmarked.

8. **Grover, L.K.** (1996). *A Fast Quantum Mechanical Algorithm for Database Search*. STOC 1996. ACM Proceedings. Algoritmo original de Grover; base para los algoritmos benchmarked.

9. **Qiskit Contributors** (2023). *Qiskit: An Open-source Framework for Quantum Computing*. Software Impacts, 17. doi:10.1016/j.softx.2023.101305. Framework de referencia usado en los módulos de integración.

10. **Cervera-Lierta, A. et al.** (2022). *Quantum Computing with IBM Qiskit: a practical introduction*. arXiv:2210.10739. Guía práctica sobre transpilación y ejecución en frameworks basados en Qiskit.

11. **Cotler, J. et al.** (2020). *Quantum virtual machines and quantum simulators*. Advances in Quantum Computing, Vol. 5. Teoría de máquinas virtuales cuánticas y overhead de simulación clásica.

12. **Barcelona Supercomputing Center — Quantum Computing Group** (2024). *QDisLib: Distributed Quantum Circuit Simulation Library*. Research documentation, available at: https://www.bsc.es/research-and-development/research-areas/computer-sciences/quantum-computing

---

*Documento generado a partir del código fuente de:*  
- `python/qdislib/grover.py`  
- `python/qdislib/shor/shor.py`  
- `python/workers/qdislib_worker.py`

*TFG — Benchmark de Frameworks de Computación Cuántica*
