# Rust para Simulación Cuántica: Motivación y Ventajas

> **Contexto de esta lección.** Las lecciones anteriores presentaron cuatro
> frameworks Python (Qiskit, Cirq, CUDA-Q, QDisLib) y cuatro backends Rust
> (q1tsim, quantr, quantrs2/quantrs, qcgpu) implementando los algoritmos de
> Grover y Shor. Esta lección analiza *por qué* Rust es una alternativa
> relevante para simulación cuántica clásica: cuándo sus propiedades
> arquitectónicas suponen una ventaja real frente a Python, y cuándo no.

---

## 1. El Problema del Rendimiento en Simulación Cuántica Clásica

### 1.1 Crecimiento exponencial del espacio de estados

Simular un circuito cuántico de *n* qubits requiere mantener en memoria el
vector de estado completo: un vector de 2ⁿ amplitudes complejas. Cada amplitud
es un número complejo de doble precisión, es decir, 16 bytes (dos `f64`). La
tabla siguiente muestra el tamaño de memoria necesario en función de *n*:

| Qubits (*n*) | Estados (2ⁿ) | Memoria (bytes) | Representación |
|:---:|---:|---:|---|
| 10 | 1 024 | ~16 KB | cacheable en L1 |
| 20 | 1 048 576 | ~16 MB | cabe en RAM fácilmente |
| 30 | 1 073 741 824 | ~16 GB | límite de un servidor estándar |
| 40 | 1 099 511 627 776 | ~16 TB | requiere clusters distribuidos |
| 50 | ~10¹⁵ | ~16 PB | computación cuántica real necesaria |

El salto de 30 a 40 qubits es de ×1 000 en memoria. A 30 qubits ya estamos al
borde de lo que puede hacer un nodo de cómputo estándar; a 50 qubits, la
simulación clásica completa es computacionalmente inviable para casi cualquier
configuración.

En el contexto de este proyecto, los benchmarks llegan hasta *n* = 20 qubits
para Grover y *N* pequeño para Shor, precisamente porque el objetivo es
comparar el rendimiento relativo de los frameworks en rangos donde todos pueden
ejecutarse.

### 1.2 El cuello de botella: operaciones matriciales masivas

Aplicar una puerta cuántica de un qubit a un sistema de *n* qubits es una
operación matricial sobre el vector de estado completo. Para una puerta que
actúa sobre el qubit *k*, la operación es equivalente a multiplicar por una
matriz 2ⁿ × 2ⁿ que es el producto tensorial de la identidad *n*−1 veces y la
matriz 2×2 de la puerta. En la práctica, los simuladores evitan construir esa
matriz explícita y la aplican mediante accesos a pares de amplitudes separados
por una distancia binaria en el índice. Aun así, para *n* = 20 esto implica
iterar sobre 524 288 pares de números complejos por cada puerta aplicada.

Un circuito de Grover con *n* = 20 qubits tiene del orden de O(√2²⁰) ≈ 1 024
iteraciones, cada una con un oráculo y un difusor de profundidad O(*n*) = O(20).
El número total de operaciones de puerta asciende a decenas de millones. Cada
operación accede a pares no contiguos en memoria, lo que convierte la eficiencia
del acceso a caché en un factor determinante del rendimiento.

### 1.3 Python y el GIL: overhead del intérprete

CPython (la implementación de referencia de Python) incluye el *Global
Interpreter Lock* (GIL): un mutex que impide la ejecución simultánea de
bytecode Python en múltiples hilos del mismo proceso. Las consecuencias para
simulación son directas:

- **Un solo hilo de Python a la vez.** Aunque se lancen múltiples threads, solo
  uno ejecuta bytecode Python en cada instante.
- **Overhead del intérprete.** Cada instrucción Python implica despacho de
  opcode, comprobaciones de tipo dinámico, gestión de refcounts. Para bucles
  sobre millones de amplitudes complejas en Python puro, este overhead puede
  suponer factores de 100–1 000× respecto a C o Rust.

El overhead del intérprete no es una limitación teórica: es medible y fue
confirmado en los benchmarks de este proyecto, donde el tiempo de *startup* de
Qiskit (importación de módulos, compilación JIT de transpilación, inicialización
del sampler) constituye una fracción significativa del tiempo total para
circuitos pequeños.

### 1.4 NumPy/BLAS: cuándo Python "escapa" del GIL y cuándo no

La clave del rendimiento de los frameworks Python modernos es que **el trabajo
pesado no se ejecuta en Python**. Qiskit-Aer, el simulador estándar de Qiskit,
implementa su motor en C++ con operaciones BLAS altamente optimizadas. Cuando
Qiskit llama a `sampler.run(...)`, Python simplemente invoca código nativo que
libera el GIL internamente. El tiempo de simulación real es comparable a
cualquier implementación nativa bien optimizada.

Sin embargo, esta escapatoria tiene límites:

1. **Overhead de empaquetado.** La conversión entre estructuras Python
   (`QuantumCircuit`, `SparsePauliOp`, etc.) y las representaciones internas del
   simulador nativo implica serialización y copia de datos.
2. **Startup time.** Importar `qiskit`, `qiskit_aer` y `qiskit_ibm_runtime`
   tarda varios segundos. Para benchmarks que miden circuitos pequeños de forma
   repetida, este overhead domina el tiempo total de proceso.
3. **Control de flujo dinámico.** Si el circuito incluye lógica clásica
   mid-circuit (condicionales basados en medición), el control vuelve a Python
   entre operaciones, reintroduciendo el overhead.
4. **Paralelismo de datos externo.** NumPy y BLAS no son parte del código del
   usuario; el usuario no puede paralelizar su propio código de construcción de
   circuitos con threads Python (por el GIL) sin recurrir a multiprocessing.

---

## 2. Rust vs Python para Cómputo Intensivo

### 2.1 Rust sin GIL: paralelismo real

Rust no tiene GIL. Los programas Rust pueden lanzar múltiples threads que
ejecuten código de forma genuinamente paralela en todos los núcleos disponibles.
El modelo de ownership y borrowing de Rust garantiza en tiempo de compilación
que no existan data races: dos threads no pueden modificar la misma memoria
simultáneamente sin sincronización explícita. El compilador rechaza el programa
si se viola esta propiedad.

```rust
// Rust: paralelismo real con Rayon (del ecosistema)
use rayon::prelude::*;

fn apply_gate_parallel(statevec: &mut Vec<Complex<f64>>, gate: [[Complex<f64>; 2]; 2], target: usize) {
    let n = statevec.len();
    let mask = 1 << target;
    // Procesar todos los pares en paralelo — sin GIL, sin locks
    statevec.par_chunks_mut(2 * mask)
        .for_each(|chunk| {
            for i in 0..mask {
                let a = chunk[i];
                let b = chunk[i + mask];
                chunk[i]        = gate[0][0] * a + gate[0][1] * b;
                chunk[i + mask] = gate[1][0] * a + gate[1][1] * b;
            }
        });
}
```

```python
# Python: paralelismo con threads — bloqueado por GIL
import threading

def apply_gate_threaded(statevec, gate, target):
    # Los threads se crean pero solo uno ejecuta Python a la vez
    # El trabajo real hay que delegarlo a NumPy para escapar del GIL
    mask = 1 << target
    # Este bucle en Python puro es ~100x más lento que Rust
    for i in range(0, len(statevec), 2 * mask):
        for j in range(mask):
            a = statevec[i + j]
            b = statevec[i + j + mask]
            statevec[i + j]        = gate[0][0] * a + gate[0][1] * b
            statevec[i + j + mask] = gate[1][0] * a + gate[1][1] * b
```

### 2.2 Gestión de memoria sin GC: sin pausas, latencia predecible

Python usa un recolector de basura (GC) basado en conteo de referencias con un
GC cíclico adicional. Los efectos en simulación cuántica son:

- **Pausas GC.** Cuando el recolector cíclico se activa (por defecto cada
  pocos cientos de asignaciones), puede interrumpir la ejecución durante
  milisegundos. Para benchmarks de tiempo de pared este efecto es ruido; para
  sistemas en tiempo real es inaceptable.
- **Overhead de refcount.** Cada asignación y desasignación de objeto en Python
  incrementa/decrementa un contador. Con millones de objetos complejos, el
  overhead acumulado es medible.

Rust utiliza ownership determinista: la memoria se libera en el momento en que
el propietario sale de scope. No hay GC, no hay pausas no deterministas, no hay
overhead de refcount en el hot path.

```rust
// La memoria se libera exactamente cuando `qc` sale de scope
// sin GC, sin overhead de runtime
{
    let mut qc = Circuit::new(total)?;
    // ... construcción del circuito
    let sim = qc.simulate();
    let counts = sim.measure_all(args.shots);
    // qc y sim se liberan aquí, determinísticamente
}
```

### 2.3 Cero coste de abstracción

El principio de *zero-cost abstractions* de Rust significa que las abstracciones
de alto nivel (iteradores, genéricos, traits) no añaden overhead en runtime
respecto al código de bajo nivel equivalente. El compilador monomorfa los
genéricos y optimiza los iteradores a bucles equivalentes a C. Esto contrasta
con Python, donde cada llamada a método implica lookup dinámico en el diccionario
del objeto.

```rust
// Iterador de alto nivel — se compila a un bucle sin overhead
let probability_sum: f64 = statevec
    .iter()
    .map(|amp| amp.norm_sqr())
    .sum();

// Es equivalente al bucle manual:
let mut probability_sum = 0.0f64;
for amp in &statevec {
    probability_sum += amp.norm_sqr();
}
// El compilador genera el mismo código para ambos
```

### 2.4 SIMD y vectorización automática

El compilador de Rust (LLVM backend) aplica vectorización automática: convierte
bucles sobre arrays de datos a instrucciones SIMD (SSE, AVX, AVX-512) que
procesan múltiples elementos en paralelo dentro de un solo ciclo de CPU. Para
operaciones sobre vectores de amplitudes complejas (`f64` real + `f64` imag),
AVX-256 puede procesar dos amplitudes complejas por instrucción.

La vectorización automática requiere que los datos sean contiguos en memoria y
que el compilador pueda garantizar ausencia de aliasing. El modelo de ownership
de Rust garantiza la segunda condición; los `Vec<T>` garantizan la primera.

### 2.5 Cuándo NO vale la pena Rust

Rust no es siempre la mejor opción. Si el trabajo real lo realiza NumPy o una
biblioteca BLAS (como en Qiskit-Aer, Cirq/qsim o CUDA-Q), entonces Python es
un frontend perfectamente eficiente. La regla práctica es:

- **Python gana** cuando el 95%+ del tiempo de CPU lo consume código nativo
  (NumPy, Aer, qsim, cuBLAS).
- **Rust gana** cuando el código de usuario es el cuello de botella: bucles de
  simulación propios, análisis de resultados a gran escala, código de
  infraestructura de orquestación que se ejecuta millones de veces.
- **Rust gana siempre** en startup time y overhead de proceso: importar Qiskit
  toma 2–5 s; un binario Rust compilado inicia en < 10 ms.

---

## 3. El Modelo de Memoria y su Impacto en Simulación

### 3.1 `Vec<Complex<f64>>`: array contiguo de amplitudes complejas

La representación canónica del vector de estado en Rust es:

```rust
use num_complex::Complex;
type Amplitude = Complex<f64>;
type StateVector = Vec<Amplitude>;
```

`Vec<Amplitude>` es un bloque contiguo de memoria heap: las 2ⁿ amplitudes se
almacenan en posiciones consecutivas de memoria, cada una ocupando 16 bytes
(8 bytes para la parte real + 8 bytes para la imaginaria, en orden `re, im`).
Este layout garantiza:

- **Sin fragmentación.** No hay indirecciones adicionales ni punteros dispersos.
- **Prefetch eficiente.** El hardware prefetcher de la CPU puede predecir
  accesos secuenciales y cargar las líneas de caché antes de que se necesiten.
- **Vectorización nativa.** LLVM puede emitir instrucciones AVX que cargan dos
  `Complex<f64>` (32 bytes) en un registro ymm en una sola instrucción.

Comparar con Python: un array NumPy `dtype=complex128` tiene el mismo layout
de memoria, pero el overhead de despacho Python añade latencia antes de llegar
al dato.

### 3.2 Cache-friendliness para operaciones en el statevector

La aplicación de una puerta de un qubit en el qubit *k* accede a pares de
amplitudes separados por una distancia `2^k` en el índice. Para *k* pequeño
(qubits de bajo orden), los pares son adyacentes: acceso secuencial, óptimo
para caché. Para *k* grande (qubits de alto orden), los pares están separados
por `2^k × 16` bytes, lo que puede exceder el tamaño de la caché L2/L3.

Los simuladores de alto rendimiento (incluyendo el Aer de Qiskit) reordenan los
qubits internamente para maximizar localidad de caché. En Rust esto se implementa
con acceso indexado directo sin overhead adicional:

```rust
fn apply_single_qubit_gate(sv: &mut [Complex<f64>], gate: [[Complex<f64>; 2]; 2], qubit: usize) {
    let half = sv.len() / 2;
    let mask = 1 << qubit;
    for i in 0..half {
        // Reconstruir índices i0, i1 sin el bit `qubit`
        let i0 = (i & (mask - 1)) | ((i >> qubit) << (qubit + 1));
        let i1 = i0 | mask;
        let a = sv[i0];
        let b = sv[i1];
        sv[i0] = gate[0][0] * a + gate[0][1] * b;
        sv[i1] = gate[1][0] * a + gate[1][1] * b;
    }
}
```

### 3.3 Rust ownership: sin aliasing inesperado → mejor optimización

El compilador de Rust garantiza que no existen dos referencias mutables al mismo
dato simultáneamente (la regla de exclusividad). Esta garantía permite al
compilador — y al CPU — aplicar optimizaciones agresivas:

- **Sin dependencias de memoria falsas.** El compilador sabe que dos punteros a
  `&mut [Complex<f64>]` no solapan, y puede reordenar instrucciones libremente.
- **Auto-vectorización más agresiva.** LLVM aplica vectorización SIMD solo
  cuando puede garantizar ausencia de aliasing. En C/C++ esto requiere
  `__restrict__`; en Rust es la garantía por defecto.
- **Eliminación de loads redundantes.** Si el compilador sabe que nadie más
  modifica un valor, puede mantenerlo en registro sin revalidar desde memoria.

### 3.4 Sin boxing ni allocaciones dinámicas en el hot path

En Python, cada número es un objeto heap: un entero Python ocupa 28 bytes, un
float 24 bytes, un complex 32 bytes (más el overhead del objeto). En Rust, los
tipos primitivos son valores directos (stack o en el array contiguo), sin
boxing, sin overhead de objeto.

```rust
// Rust: Complex<f64> son 16 bytes en el array, sin overhead de objeto
let sv: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 1 << n];

// El acceso sv[i] es un load de 16 bytes desde dirección base + i*16
// sin indirección, sin refcount, sin verificación de tipo
```

```python
# Python: numpy.complex128 array también es contiguo en memoria
# pero la interfaz Python añade overhead en cada llamada
import numpy as np
sv = np.zeros(1 << n, dtype=np.complex128)
# sv[i] en Python devuelve un objeto Python temporal con refcount
```

---

## 4. Paralelismo sin Miedo ("Fearless Concurrency")

### 4.1 El compilador previene data races en tiempo de compilación

"Fearless Concurrency" es el término con el que el equipo de Rust describe su
modelo de concurrencia: el compilador garantiza estáticamente que los programas
multi-thread no tienen data races. Un data race ocurre cuando dos threads
acceden al mismo dato simultáneamente y al menos uno escribe. En Rust, esto es
un error de compilación, no un bug en tiempo de ejecución.

El mecanismo: los tipos `Send` y `Sync` del sistema de traits marcan qué tipos
pueden transferirse entre threads (`Send`) o compartirse por referencia
(`Sync`). El compilador rechaza código que intente compartir un `&mut T` entre
threads sin sincronización. No es un sanitizador en runtime (como ThreadSanitizer
en C++): es una garantía en tiempo de compilación.

### 4.2 Rayon: paralelismo de datos idiomático

Rayon es la biblioteca estándar de facto para paralelismo de datos en Rust.
Reemplaza los iteradores estándar con iteradores paralelos que distribuyen el
trabajo entre un pool de threads (work-stealing scheduler). La API es casi
idéntica a los iteradores secuenciales: cambiar `.iter()` por `.par_iter()`
es suficiente en muchos casos.

```rust
use rayon::prelude::*;

// Aplicar una transformación a todas las amplitudes en paralelo
fn normalize(sv: &mut Vec<Complex<f64>>) {
    let norm: f64 = sv.par_iter()
        .map(|a| a.norm_sqr())
        .sum::<f64>()
        .sqrt();
    sv.par_iter_mut()
        .for_each(|a| *a /= norm);
}
```

Rayon usa un pool de threads nativo del sistema operativo. En un sistema con
*k* núcleos físicos, un bucle Rayon se ejecuta genuinamente en *k* threads en
paralelo, con una aceleración teórica de ×k y práctica de ×3–5 para operaciones
sobre arrays grandes (limitado por el ancho de banda de memoria).

### 4.3 Por qué la aplicación de puertas cuánticas es paralelizable

La aplicación de una puerta de un qubit a los 2ⁿ estados del vector de estado
consiste en 2^(n-1) operaciones independientes: cada par de amplitudes (i0, i1)
se transforma independientemente del resto. No hay dependencias entre pares.
Esto lo hace trivialmente paralelizable: es un problema *embarrassingly parallel*.

```rust
fn apply_gate_rayon(sv: &mut Vec<Complex<f64>>, gate: [[Complex<f64>; 2]; 2], qubit: usize) {
    let mask = 1usize << qubit;
    let half = sv.len() / 2;
    // Dividir el trabajo entre todos los núcleos disponibles
    (0..half).into_par_iter().for_each(|i| {
        let i0 = (i & (mask - 1)) | ((i >> qubit) << (qubit + 1));
        let i1 = i0 | mask;
        // SAFETY: i0 y i1 son siempre distintos — Rayon puede dividir sin aliasing
        unsafe {
            let a = *sv.get_unchecked(i0);
            let b = *sv.get_unchecked(i1);
            *sv.get_unchecked_mut(i0) = gate[0][0] * a + gate[0][1] * b;
            *sv.get_unchecked_mut(i1) = gate[1][0] * a + gate[1][1] * b;
        }
    });
}
```

Para que Rayon pueda paralelizar accesos mutables al mismo Vec, los índices i0
y i1 deben ser siempre distintos (lo son por construcción: difieren en el bit
`qubit`). El bloque `unsafe` es necesario porque el compilador no puede probar
estáticamente que dos índices computados dinámicamente sean distintos; el
programador asume esa responsabilidad.

### 4.4 Comparación con Python threading y multiprocessing

| Mecanismo | Python (GIL) | Python (multiprocessing) | Rust (Rayon) |
|---|---|---|---|
| Paralelismo real | No | Sí | Sí |
| Overhead de arranque | Bajo | Alto (fork/spawn) | Nulo (pool reutilizable) |
| Compartir memoria | No sin copia | No sin serialización | Sí, garantías estáticas |
| Data races | Imposibles (GIL lo previene) | Posibles en shared memory | Imposibles (compilador) |
| Overhead por iteración | GIL contention | Serialización pickle | Zero |

Python multiprocessing puede ejecutar código en paralelo, pero compartir el
vector de estado (que puede ser de GB) entre procesos requiere memoria compartida
explícita (`multiprocessing.shared_memory`) o copia completa, lo que elimina
la ventaja para arrays grandes.

---

## 5. Integración con GPU desde Rust

### 5.1 Bindings CUDA en Rust

El ecosistema GPU de Rust existe pero está menos maduro que Python/CUDA:

- **cudarc** (v0.10+): bindings safe de alto nivel para CUDA runtime,
  cuBLAS y cuFFT. Mantenido activamente, usado por Candle (framework ML de
  Hugging Face en Rust). Permite escribir kernels CUDA en CUDA C y cargarlos
  desde Rust, o usar operaciones cuBLAS directamente.
- **rust-cuda**: proyecto experimental que busca compilar shaders/kernels
  directamente desde Rust (sin código CUDA C). Usa `nvptx` como target de
  compilación de rustc. Estado: experimental, no production-ready.

La diferencia respecto a Python/CUDA es significativa: en Python, `cupy` y
`numba.cuda` permiten escribir kernels GPU directamente en Python o NumPy-like
arrays en GPU con una sola línea. El ecosistema Python GPU es mucho más maduro.

### 5.2 OpenCL en Rust

- **ocl** (v0.19): bindings de alto nivel para OpenCL 1.2+. Es la dependencia
  de `qcgpu` en este proyecto (a través de `ocl-core`). El estado de ocl es
  semi-mantenido; no sigue activamente los nuevos estándares OpenCL 3.0.
- **opencl3** (v0.9+): bindings más modernos para OpenCL 3.0, pero menos usados.

En este proyecto, `qcgpu` usa OpenCL a través de `ocl-core` (vendorizado con
parches para compilar con Rust moderno), lo que ilustra la fragilidad del
ecosistema: el crate principal `ocl-core` 0.9.0 no compila sin parches con
toolchains Rust recientes, de ahí la necesidad de vendorizar (`Cargo.toml`:
`[patch.crates-io] ocl-core = { path = "rust/qcgpu/vendor/ocl-core" }`).

### 5.3 Por qué el ecosistema GPU Rust es menos maduro

La razón es estructural: NVIDIA invierte masivamente en Python (cuDNN, cuBLAS,
CUDA-Q, TensorRT-Python, triton), mientras que el ecosistema Rust GPU depende
principalmente de la comunidad open-source. En simulación cuántica, el único
framework con integración GPU seria en Rust es `quantrs2` (ex-quantrs), que
usa bindings a CUDA para acelerar operaciones de statevector.

Dicho esto, el *futuro* es prometedor: cudarc está siendo adoptado por proyectos
serios (Candle, burn), y NVIDIA ha publicado guías oficiales para usar CUDA desde
Rust.

### 5.4 quantrs2 (CUDA) y qcgpu (OpenCL) en este proyecto

En el workspace de este proyecto:

```
[workspace]
members = [
    "rust/qcgpu",      // OpenCL — simulación en GPU via OpenCL
    "rust/quantrs",    // quantrs2 — CPU + opción CUDA
    "rust/quantr",     // CPU puro, sin GPU
    "rust/q1tsim",     // CPU puro, crate vendorizado
]
```

`qcgpu` intenta usar OpenCL en runtime: si no hay dispositivo OpenCL disponible
(como en macOS con Metal y sin OpenCL habilitado), el binario informa el error en
su JSON de salida. `quantrs` (quantrs2) compila con soporte CUDA opcional
dependiendo de si CUDA toolkit está disponible en el sistema de compilación.

---

## 6. El Ecosistema de Simulación Cuántica en Rust

### 6.1 Por qué está más fragmentado

El ecosistema de simulación cuántica en Python tiene detrás inversiones de IBM
(Qiskit, >500 contribuidores), Google (Cirq, qsim), NVIDIA (CUDA-Q), BSC
(QDisLib). Cada framework tiene decenas de ingenieros a tiempo completo, soporte
para hardware real y años de desarrollo.

El ecosistema Rust de simulación cuántica es, por contraste, principalmente obra
de individuos o grupos pequeños sin respaldo institucional comparable. Esto
tiene consecuencias directas en madurez, documentación, soporte y longevidad.

### 6.2 Los cuatro crates del proyecto

| Crate | Estado | Característica diferencial |
|---|---|---|
| **q1tsim** | Abandonado (último commit ~2021) | Simulación statevector pura, buena cobertura de puertas |
| **quantr** | Activo pequeño (1-2 mantenedores) | API limpia, MSB-first, buena documentación inline |
| **quantrs2** (quantrs) | Release Candidate | Multi-backend: CPU, CUDA, OpenCL |
| **qcgpu** | Sin mantenimiento activo | Pionero en GPU OpenCL para Rust |

La heterogeneidad de estados es representativa del ecosistema: de los cuatro,
solo `quantr` puede considerarse activamente mantenido con releases regulares.
`q1tsim` requirió vendorizar el crate base para que compilara
(`rust/q1tsim/vendor/q1tsim`). `qcgpu` requirió vendorizar y parchear `ocl-core`.

### 6.3 La tendencia: backends Rust en proyectos Python

La tendencia más significativa no es Rust *reemplazando* a Python en simulación
cuántica, sino Rust siendo adoptado como *backend de rendimiento* dentro de
proyectos Python. El ejemplo más notable es **qiskit-terra** (ahora parte del
repositorio principal de Qiskit): desde la versión 0.45 (2023), el transpilador
de Qiskit usa un backend escrito en Rust para las pasadas de optimización más
costosas, expuesto a Python mediante PyO3. El resultado fue una mejora de 10–50×
en el tiempo de transpilación para circuitos grandes.

Este patrón — Python como API de usuario, Rust como motor de rendimiento — es
probablemente el modelo dominante para los próximos 5–10 años en el ecosistema
cuántico.

---

## 7. Arquitectura de los Binarios Rust en Este Proyecto

### 7.1 El workspace de Cargo: compilación unificada de 4 crates

El fichero `Cargo.toml` en la raíz del proyecto define un workspace Cargo:

```toml
[workspace]
members = [
    "rust/qcgpu",
    "rust/quantrs",
    "rust/quantr",
    "rust/q1tsim",
]
resolver = "2"

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
rand = "0.8"
```

Un workspace Cargo permite:
- **Compilación incremental compartida.** Las dependencias comunes (serde, clap,
  rand) se compilan una sola vez y se comparten entre todos los miembros.
- **`cargo build --release`** desde la raíz compila los cuatro crates y deposita
  todos los binarios en `target/release/`.
- **Lockfile único** (`Cargo.lock`): versiones de dependencias idénticas para
  todos los crates, reproducibilidad garantizada.

Las dependencias compartidas de workspace se declaran en
`[workspace.dependencies]` y los crates individuales las referencian sin
especificar versión (`serde = { workspace = true }`), evitando divergencia de
versiones.

### 7.2 La interfaz CLI (clap) + JSON (serde_json): comparación justa con Python

Cada binario Rust implementa la misma interfaz que los frameworks Python:
argumentos `--n`, `--target`, `--shots`, `--iterations` vía `clap`, y emisión
de un objeto JSON en stdout vía `serde_json`. Esto permite que `run.py` invoque
los binarios Rust como subprocesos y consuma sus resultados de forma uniforme:

```python
# run.py: invocar un binario Rust como subproceso
result = subprocess.run(
    [str(binary), "--n", str(n), "--target", str(target), "--shots", str(shots)],
    capture_output=True, text=True, timeout=300
)
data = json.loads(result.stdout)
# data["time_ms"] es el tiempo reportado por Rust (sin overhead Python)
```

El tiempo reportado en `time_ms` es medido *dentro* del proceso Rust con
`std::time::Instant`, excluyendo el overhead de inicialización del subproceso
Python. Esto garantiza que la comparación mide el tiempo de simulación real, no
el overhead de invocación.

### 7.3 Compilación estática: todo incluido en el binario

Por defecto, los binarios Rust en `--release` enlazan estáticamente la
biblioteca estándar de Rust y todas sus dependencias puras (serde, clap, etc.).
El resultado es un binario autocontenido que no requiere ninguna instalación
adicional en el sistema destino, excepto las bibliotecas del sistema (libc).

Esto contrasta con Python, donde ejecutar `python run.py` requiere:
- Python 3.12+ instalado
- Un entorno virtual con todas las dependencias (`uv sync`)
- Las dependencias nativas de cada framework (PyTorch, CUDA runtime, etc.)

### 7.4 startup_time: binarios Rust inician ~10–100× más rápido que Qiskit

El tiempo de arranque de cada framework es relevante para benchmarks que miden
circuitos pequeños o ejecutan muchas instancias en secuencia. Mediciones
representativas en hardware estándar:

| Framework | Startup aproximado |
|---|---|
| quantr (Rust) | < 5 ms |
| q1tsim (Rust) | < 10 ms |
| Python puro | < 50 ms |
| Qiskit (import + sampler init) | 2 000–5 000 ms |
| Cirq (import) | 500–1 500 ms |
| CUDA-Q (import + GPU init) | 1 000–3 000 ms |

Para un benchmark que ejecuta 100 instancias de circuitos de 5 qubits, el
startup de Qiskit puede dominar el tiempo total si no se amortiza
correctamente. El diseño del runner en `run.py` amortiza este costo
inicializando cada framework una sola vez por sesión de benchmark.

---

## 8. Comparación Directa: Grover en Qiskit vs Grover en quantr

### 8.1 Side-by-side: construcción del oráculo

El oráculo de fase para el estado |target> es la misma operación matemática
en ambos frameworks: X en qubits donde target tiene bit 0, luego MCZ, luego
X de nuevo. La diferencia está en cómo se expresa:

**Python (Qiskit) — `python/qiskit/grover.py`:**
```python
def build_oracle(n: int, target: int) -> QuantumCircuit:
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    # Multi-controlled Z: ZGate().control(n-1) construye MCZ automáticamente
    qc.append(ZGate().control(n - 1), qr[:])

    for i in range(n):
        if not (target >> i) & 1:
            qc.x(qr[i])

    return qc
```

**Rust (quantr) — `rust/quantr/src/grover.rs`:**
```rust
pub fn build_oracle(
    qc: &mut Circuit,
    n: usize,
    target: u64,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            qc.add_gate(Gate::X, i)?;
        }
    }
    add_mcz(qc, n, ancillas)?;  // descomposición manual con ancillas
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            qc.add_gate(Gate::X, i)?;
        }
    }
    Ok(())
}
```

### 8.2 La diferencia clave: MCZ

La diferencia más significativa está en la puerta multi-controlled Z (MCZ):

- **Qiskit**: `ZGate().control(n - 1)` construye la puerta MCZ genérica y la
  transpila automáticamente a la base de puertas del backend. El programador
  no necesita conocer la descomposición.

- **quantr**: Solo tiene `Gate::Toffoli(c1, c2)` (2 controles). Para MCZ con
  más controles hay que implementar manualmente la descomposición con ancillas:
  la estrategia de escalera de Toffoli que computa la AND de todos los controles
  en qubits ancilla, aplica la puerta central, y descomputa las ancillas.

```rust
// quantr: descomposición manual de MCX con k controles
pub fn add_mcx(qc: &mut Circuit, controls: &[usize], target: usize, ancillas: &[usize])
    -> Result<(), QuantrError>
{
    match controls.len() {
        0 => { qc.add_gate(Gate::X, target)?; }
        1 => { qc.add_gate(Gate::CNot(controls[0]), target)?; }
        2 => { qc.add_gate(Gate::Toffoli(controls[0], controls[1]), target)?; }
        k => {
            // Escalera forward: AND de todos los controles en ancillas
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
            for i in 2..(k - 1) {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i-2]), ancillas[i-1])?;
            }
            // Puerta central
            qc.add_gate(Gate::Toffoli(controls[k-1], ancillas[k-3]), target)?;
            // Escalera inversa: descomputar ancillas a |0>
            for i in (2..(k - 1)).rev() {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i-2]), ancillas[i-1])?;
            }
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
        }
    }
    Ok(())
}
```

Esta diferencia refleja el nivel de madurez de cada ecosistema: Qiskit tiene
una biblioteca de puertas y descomposiciones muy completa (resultado de años
de desarrollo e investigación). quantr, siendo un proyecto pequeño, expone
solo las puertas fundamentales y delega las descomposiciones al usuario.

### 8.3 Manejo de errores

Qiskit usa excepciones Python (`assert`, `QiskitError`). quantr usa `Result<T, E>`
de Rust: cada operación que puede fallar devuelve un `Result`, y el operador `?`
propaga el error hacia arriba en la call stack. No hay excepciones, no hay
unwinding no determinista:

```rust
// Rust: propagación explícita de errores con ?
pub fn grover_circuit(n: usize, target: u64, iterations: Option<usize>)
    -> Result<Circuit, QuantrError>
{
    let mut qc = Circuit::new(total)?;          // si falla, retorna Err
    qc.add_repeating_gate(Gate::H, &search)?;   // idem
    for _ in 0..iterations {
        build_oracle(&mut qc, n, target, &ancillas)?;
        build_diffuser(&mut qc, n, &ancillas)?;
    }
    Ok(qc)  // retorno explícito del valor exitoso
}
```

```python
# Python: excepciones implícitas
def grover_circuit(n: int, target: int, num_iterations=None) -> QuantumCircuit:
    qc = QuantumCircuit(qr, cr)  # puede lanzar QiskitError
    for _ in range(num_iterations):
        qc.compose(oracle, qubits=qr, inplace=True)  # puede lanzar
        qc.compose(diffuser, qubits=qr, inplace=True)
    qc.measure(qr, cr)
    return qc
```

### 8.4 Verbosidad de Rust y lo que refleja

El código Rust es notablemente más verboso que el Python equivalente. La función
`grover_circuit` en Rust requiere ~30 líneas para expresar lo mismo que ~15 en
Python. Esta verbosidad no es un defecto de diseño, sino una consecuencia de
las garantías que Rust ofrece:

- Tipos explícitos en todas las firmas de función
- Manejo de errores explícito con `Result`
- Gestión manual de qubits ancilla (que Qiskit oculta)
- Sin constructores "mágicos" como `ZGate().control(n-1)`

La verbosidad adicional de Rust es el precio de las garantías: el programa que
compila es correcto respecto a memoria y concurrencia. En Python, muchos errores
solo aparecen en runtime.

---

## 9. Cuándo Usar Rust vs Python

### 9.1 Python: prototipado rápido y acceso a hardware real

Python es la opción correcta cuando:

- **Prototipado**: la velocidad de iteración importa más que el rendimiento.
  Python permite probar un nuevo algoritmo en horas; Rust requiere compilación
  y gestión de tipos explícita.
- **Hardware cuántico real**: solo Qiskit, Cirq y CUDA-Q tienen clientes
  certificados para hardware IBM, Google y IonQ/Quantinuum respectivamente.
  No existe ningún framework Rust con acceso directo a hardware cuántico real
  en producción.
- **Integración con ML**: las librerías de machine learning cuántico (PennyLane,
  TorchQuantum) están en Python.
- **Visualización y análisis**: matplotlib, qiskit.visualization, etc.
- **Comunidad y documentación**: Stack Overflow, tutoriales, cursos, libros.

### 9.2 Rust: rendimiento máximo en CPU y producción

Rust es la opción correcta cuando:

- **Rendimiento máximo en CPU**: para simuladores que no dependen de BLAS
  externos, Rust nativo bien optimizado supera a Python por 10–100×.
- **Latencia predecible**: sistemas en tiempo real que no pueden tolerar pausas
  GC ni overhead del intérprete.
- **Concurrencia masiva**: servidores que manejan miles de solicitudes de
  simulación en paralelo.
- **Integración en sistemas embebidos o de bajo nivel**: Rust compila a código
  nativo sin runtime, compatible con sistemas sin SO.
- **Backend de rendimiento para Python**: usando PyO3, los módulos Rust se
  exponen como módulos Python nativos. Esta es la arquitectura que usa Qiskit
  para su transpilador.

### 9.3 El futuro híbrido: backends Rust con APIs en Python

El modelo que está consolidándose en el ecosistema cuántico es el mismo que
ya domina en otros dominios (Polars, Pydantic v2, Ruff, uv):

1. El *hot path* se implementa en Rust: parsing, compilación de circuitos,
   optimización, simulación del statevector.
2. La *API de usuario* vive en Python: sintaxis ergonómica, notebooks Jupyter,
   integración con el ecosistema científico.
3. El puente es PyO3: permite exportar tipos y funciones Rust como módulos
   Python con overhead mínimo.

En Qiskit, esto ya es una realidad: el transpilador Rust (qiskit-terra) es
invocado transparentemente cuando se llama a `transpile()` desde Python. El
usuario no necesita conocer Rust; se beneficia del rendimiento automáticamente.

Para los frameworks del proyecto, `quantrs2` (quantrs) es el más cercano a este
modelo: tiene una API Python en desarrollo y ya usa CUDA para acelerar
simulaciones en GPU. Es probable que frameworks como quantr sigan una evolución
similar si el proyecto crece.

### 9.4 Resumen de criterios de selección

| Criterio | Python | Rust |
|---|---|---|
| Velocidad de prototipado | Alta | Baja |
| Rendimiento CPU (código propio) | Bajo | Alto |
| Rendimiento con BLAS/CUDA | Comparable | Comparable |
| Acceso a hardware real | Sí (IBM, Google, IonQ) | No |
| Startup time | 1–5 s | < 10 ms |
| Paralelismo sin GIL | No (threads) / Sí (multiprocessing) | Sí |
| Garantías de memoria | Runtime (refcount, GC) | Compile-time |
| Madurez del ecosistema cuántico | Alta | Baja |
| Integración ML/viz | Excelente | Limitada |
| Binarios distribuibles | No (entorno Python) | Sí (estático) |

---

## 10. Referencias

[1] **Jung, R., Jourdan, J.-H., Krebbers, R., & Dreyer, D.** (2021).
*RustBelt: Securing the Foundations of the Rust Programming Language.*
Journal of the ACM (JACM), 69(1), 1–34.
https://doi.org/10.1145/3486476
> Formalización de las garantías de seguridad de tipos y memoria de Rust.

[2] **Bauer, B., Bravyi, S., Motta, M., & Chan, G. K.-L.** (2020).
*Quantum Algorithms for Quantum Chemistry and Quantum Materials Science.*
Chemical Reviews, 120(22), 12685–12717.
https://doi.org/10.1021/acs.chemrev.9b00829
> Revisión exhaustiva de algoritmos cuánticos para simulación molecular, contexto
> de por qué la simulación cuántica eficiente importa.

[3] **Guerreschi, G. G., Hogaboam, J., Baruffa, F., & Sawaya, N. P. D.** (2020).
*Intel Quantum Simulator: A cloud-ready high-performance simulator for
quantum circuits.* Quantum Science and Technology, 5(3), 034007.
https://doi.org/10.1088/2058-9565/ab8505
> Análisis detallado de optimizaciones de rendimiento en simuladores de
> statevector: cache locality, SIMD, paralelismo.

[4] **Häner, T., & Steiger, D. S.** (2017).
*0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit.*
SC'17: Proceedings of the International Conference for High Performance
Computing, Networking, Storage and Analysis.
https://doi.org/10.1145/3126908.3126947
> Límites de la simulación cuántica clásica a gran escala; relevante para la
> sección de crecimiento exponencial.

[5] **Nielsen, M. A., & Chuang, I. L.** (2010).
*Quantum Computation and Quantum Information* (10th anniversary ed.).
Cambridge University Press.
ISBN: 978-1107002173
> Referencia fundamental para el formalismo de circuitos cuánticos, puertas y
> algoritmos (Grover cap. 6, Shor cap. 5).

[6] **Grover, L. K.** (1996).
*A Fast Quantum Mechanical Algorithm for Database Search.*
Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC),
212–219.
https://doi.org/10.1145/237814.237866
> Paper original del algoritmo de búsqueda de Grover.

[7] **Shor, P. W.** (1997).
*Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms
on a Quantum Computer.* SIAM Journal on Computing, 26(5), 1484–1509.
https://doi.org/10.1137/S0097539795293172
> Paper original del algoritmo de factorización de Shor.

[8] **Barenco, A., et al.** (1995).
*Elementary gates for quantum computation.*
Physical Review A, 52(5), 3457–3467.
https://doi.org/10.1103/PhysRevA.52.3457
> Descomposición de puertas multi-controladas; base teórica de la descomposición
> MCX/MCZ usada en ambos frameworks del proyecto.

[9] **Javadi-Abhari, A., et al.** (2024).
*Quantum computing with Qiskit.*
arXiv:2405.08810 [quant-ph].
https://arxiv.org/abs/2405.08810
> Descripción actualizada de la arquitectura de Qiskit, incluyendo el
> transpilador en Rust.

[10] **Villalonga, B., et al.** (2020).
*Establishing the quantum supremacy frontier with a 281 Pflop/s simulation.*
npj Quantum Information, 6, 84.
https://doi.org/10.1038/s41534-020-00322-4
> Simulación cuántica de alto rendimiento (qsim de Google); relevante para
> comparar con el enfoque Rust.

[11] **Matsakis, N. D., & Klock II, F. S.** (2014).
*The Rust programming language.*
Proceedings of the ACM SIGPLAN Workshop on Systems Programming and
Application Software (SPLASH), 103–104.
https://doi.org/10.1145/2663171.2663188
> Presentación original de los objetivos de diseño de Rust: seguridad, rendimiento
> y concurrencia sin GIL.

[12] **Devitt, S. J., Munro, W. J., & Nemoto, K.** (2013).
*Quantum error correction for beginners.*
Reports on Progress in Physics, 76(7), 076001.
https://doi.org/10.1088/0034-4885/76/7/076001
> Contexto sobre por qué los simuladores cuánticos necesitan ser eficientes:
> la simulación de corrección de errores es exponencialmente más costosa.

---

*Lección parte del TFG "Benchmarking de Frameworks de Computación Cuántica".*
*Implementaciones de referencia en `/rust/quantr/src/grover.rs` (Rust) y*
*`/python/qiskit/grover.py` (Python).*
