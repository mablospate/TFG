# qcgpu — Simulación Cuántica GPU via OpenCL en Rust

> **Nivel**: Universidad — Máster en Computación Cuántica  
> **Prerrequisitos**: álgebra lineal compleja, puertas cuánticas básicas, nociones de GPU computing  
> **Duración estimada**: 3 horas de lectura + laboratorio

---

## 1. Contexto e Historia

### 1.1 El crate qcgpu v0.1.0

`qcgpu` es un simulador cuántico de statevector escrito en Rust y publicado en
[crates.io](https://crates.io/crates/qcgpu) en 2018 bajo licencia MIT. Su
nombre abrevia *Quantum Computing GPU* y fue desarrollado por Adam Kelly como
prueba de concepto de simulación cuántica acelerada por GPU usando el estándar
**OpenCL** en lugar del ecosistema CUDA exclusivo de NVIDIA.

El crate implementa el modelo de circuito cuántico estándar:

- Representación de estado como **vector de amplitudes complejas** (`2^n`
  entradas de `f32` complejos) almacenado en un **buffer GPU OpenCL**.
- Un conjunto reducido pero funcional de puertas: H, X, Y, Z, S, T, CX
  (CNOT), Toffoli (CCX), SWAP, y la puerta de fase parametrizada `r(θ)`.
- Función `measure_many(shots)` que colapsa el estado y devuelve un histograma
  de mediciones.

Su diseño es minimalista a propósito: el kernel OpenCL que aplica una puerta
unitaria arbitraria de un qubit es unas pocas decenas de líneas de C, y el
wrapper Rust es de menos de 500 líneas. Esto lo convierte en un caso de estudio
ideal para entender cómo funciona la aceleración GPU en simulación cuántica sin
la complejidad de frameworks de producción.

### 1.2 OpenCL vs CUDA: estándar abierto frente a ecosistema propietario

La elección de OpenCL sobre CUDA es deliberada y tiene implicaciones profundas:

| Característica | OpenCL | CUDA |
|---|---|---|
| Promotor | Khronos Group (consorcio) | NVIDIA (propietario) |
| Hardware soportado | AMD, Intel, NVIDIA, FPGAs | Solo NVIDIA |
| Portabilidad del kernel | Compila en cualquier plataforma | Sólo en GPUs NVIDIA |
| Rendimiento pico | 10–20 % menor en NVIDIA | Óptimo en NVIDIA |
| Adopción en HPC cuántico | Minoritaria | Dominante (cuQuantum) |
| Soporte en macOS | Sí (hasta macOS 12 sin deprecar) | No nativo |

OpenCL es la opción correcta cuando se busca **portabilidad real**: el mismo
código fuente compilará y ejecutará en una AMD Radeon, en una Intel Iris, en
una NVIDIA GeForce y en determinados FPGAs. El precio que se paga es que en
hardware NVIDIA el driver OpenCL rara vez alcanza el rendimiento del driver
CUDA nativo.

### 1.3 Limitación a Linux x86\_64

El crate qcgpu v0.1.0 sólo está garantizado en **Linux x86\_64** por razones
técnicas concretas:

1. **Drivers OpenCL de producción**: en Linux, los drivers AMD ROCm y NVIDIA
   ofrecen implementaciones OpenCL 1.2/2.0 estables. En Windows existen pero
   el ecosistema Rust para FFI con `ocl` (el binding OpenCL de qcgpu) tiene
   menos pruebas. En macOS, Apple deprecó OpenCL en 2018 en favor de Metal y
   aunque sigue presente hasta macOS 14, la ABI tiene diferencias.

2. **Dependencia transitiva `ocl`**: el crate `ocl` (binding seguro de OpenCL
   para Rust) enlaza contra `libOpenCL.so` en Linux o `OpenCL.framework` en
   macOS. La compilación en Linux es trivial; en otros sistemas requiere ajustes
   en `build.rs`.

3. **Medición de memoria RSS**: el código usa `/proc/self/status` para leer
   `VmHWM` (peak RSS), API exclusiva de Linux. En macOS/Windows retorna 0.

En el contexto de este TFG el entorno de ejecución del pipeline de benchmarks
es Linux (Ubuntu 22.04 en contenedor Docker), lo que hace que las restricciones
de qcgpu sean irrelevantes para la comparativa.

---

## 2. OpenCL para Simulación Cuántica

### 2.1 Qué es OpenCL

OpenCL (*Open Computing Language*) es una API y un lenguaje de shaders (basado
en C99) para programación heterogénea. Define el modelo de plataforma siguiente:

```
Host (CPU, código Rust)
  └── Platform (driver OpenCL instalado)
        └── Device (GPU / CPU OpenCL)
              └── Context
                    ├── CommandQueue
                    ├── Buffer (memoria del dispositivo)
                    └── Program → Kernels (funciones GPU)
```

El host (Rust) compila el programa OpenCL en tiempo de ejecución a partir de
código fuente en texto, crea buffers en la memoria del dispositivo, encola
ejecuciones de kernels y lee resultados de vuelta a la RAM del host.

El binding Rust `ocl` encapsula toda esta maquinaria con RAII y tipos seguros,
ocultando los punteros C de la API nativa.

### 2.2 Kernels OpenCL para puertas cuánticas

El kernel fundamental en qcgpu aplica una **puerta unitaria U de un qubit** al
qubit `q` de un statevector de `2^n` amplitudes. El truco estándar es que las
amplitudes que interactúan bajo la puerta forman pares: para cada estado base
`|i⟩`, la pareja es `|i XOR (1 << q)⟩`. El kernel se ejecuta con `2^(n-1)`
hilos, cada uno procesando un par:

```c
// Kernel OpenCL conceptual para una puerta de 1 qubit
__kernel void apply_gate(
    __global float2* state,   // amplitudes complejas (re, im)
    int qubit,
    float4 gate               // matriz 2×2: (a, b, c, d) donde U = [[a,b],[c,d]]
) {
    int tid = get_global_id(0);
    int lo = (tid & ~((1 << qubit) - 1)) << 1 | (tid & ((1 << qubit) - 1));
    int hi = lo | (1 << qubit);

    float2 amp_lo = state[lo];
    float2 amp_hi = state[hi];

    // Multiplicación de vector por matriz 2×2 compleja
    state[lo] = complex_mul_add(gate.xy, amp_lo, gate.zw, amp_hi); // a*lo + b*hi
    state[hi] = complex_mul_add(gate.zw, amp_lo, gate.xyzw, amp_hi); // c*lo + d*hi
}
```

Este patrón **no requiere ninguna sincronización entre hilos** porque cada par
`(lo, hi)` es disjunto. La GPU puede ejecutar los `2^(n-1)` pares en paralelo
masivo, lo que da la aceleración característica de los simuladores GPU.

Para puertas de **dos qubits** (CX, Toffoli) el kernel es análogo pero filtra
los estados según la condición del control.

### 2.3 El statevector en buffer GPU

La representación de estado es:

```
|ψ⟩ = Σ α_i |i⟩,  i = 0 … 2^n - 1
```

donde cada `α_i` es un número complejo de 32 bits (`float2` en OpenCL, dos
`f32` en Rust). Para `n` qubits el buffer ocupa:

```
tamaño = 2^n × 2 × 4 bytes = 2^(n+3) bytes
```

| n (qubits) | Amplitudes | Memoria |
|---|---|---|
| 10 | 1 024 | 8 KB |
| 20 | 1 048 576 | 8 MB |
| 25 | 33 554 432 | 256 MB |
| 28 | 268 435 456 | 2 GB |
| 30 | 1 073 741 824 | 8 GB |

Para `n ≥ 30` el statevector ya no cabe en la memoria de la mayoría de GPUs
de consumo (8–16 GB VRAM). qcgpu hereda esta limitación directamente: el
simulador es **exacto** (no tiene aproximaciones) pero acotado en `n` por la
VRAM disponible.

### 2.4 Transferencias CPU–GPU y su overhead

El ciclo de vida de un cálculo en qcgpu es:

```
[CPU] Crear State::new(n, 0)
        │
        ▼
[GPU] Alocar buffer de 2^n amplitudes, inicializar |0…0⟩
        │
        ▼
[CPU] Llamar a state.h(qubit)
        │
        ▼
[GPU] Ejecutar kernel apply_gate (sin copia CPU→GPU de la matriz: es un argumento pequeño)
        │
        ▼  (repetir para cada puerta)
        │
[CPU] Llamar a state.measure_many(shots)
        │
        ▼
[GPU] Ejecutar kernel de muestreo, copiar histograma a CPU (copia GPU→CPU)
```

La copia CPU→GPU de las amplitudes sólo ocurre en la inicialización. Las
matrices de puertas se pasan como argumentos de kernel (escalares), no como
buffers. La única copia GPU→CPU de datos grandes es la del histograma final,
que es `O(shots)` entradas. Este diseño minimiza el overhead de transferencia.

Sin embargo, la **inicialización del contexto OpenCL** (compilación JIT del
programa, selección de plataforma) añade una latencia fija de 50–500 ms que
domina el tiempo total para circuitos pequeños.

### 2.5 Comparación CUDA vs OpenCL para statevector

En simulación de statevector el cuello de botella es el ancho de banda de
memoria, no la velocidad de cómputo. Los kernels son simples (multiplicaciones
complejas), así que lo que importa es cuántos GB/s puede el hardware transferir
desde la VRAM al tejido de cómputo.

En una misma GPU NVIDIA A100:

| Framework | Backend | Rendimiento relativo |
|---|---|---|
| cuQuantum / cuStateVec | CUDA nativo | 100 % (referencia) |
| Qiskit Aer (GPU) | CUDA via cuStateVec | ~95 % |
| qcgpu v0.1 | OpenCL | ~60–70 % |
| qcgpu v0.1 | OpenCL en AMD Radeon | ~85 % del rendimiento CUDA teórico de la Radeon |

La diferencia en NVIDIA se debe principalmente al overhead del driver OpenCL
vs el driver CUDA: la compilación JIT del kernel OpenCL no usa los mismos
caminos de optimización que `nvcc`.

---

## 3. API de qcgpu en Rust

### 3.1 Creación del estado y el contexto GPU

El punto de entrada principal es `State::new`:

```rust
use qcgpu::State;

// Crea un estado de n qubits iniciado en |0...0⟩.
// El segundo argumento es el índice de la plataforma OpenCL (0 = primera).
let mut state = State::new(4, 0);
```

Internamente, `State::new` hace:
1. Enumerar las plataformas OpenCL disponibles (`ocl::Platform::list()`).
2. Seleccionar la plataforma por índice.
3. Crear un contexto y una cola de comandos para la primera GPU de esa
   plataforma.
4. Compilar el programa OpenCL (los kernels) desde código fuente embebido en
   la librería.
5. Alocar el buffer GPU de `2^n × 8` bytes e inicializar `state[0] = 1+0i`,
   `state[i] = 0` para `i > 0`.

Si no hay ninguna plataforma OpenCL disponible, el constructor hace `panic!`.

### 3.2 Puertas disponibles

```rust
// Puertas de un qubit
state.h(qubit: i32);           // Hadamard
state.x(qubit: i32);           // Puerta X (NOT)
state.y(qubit: i32);           // Puerta Y
state.z(qubit: i32);           // Puerta Z
state.s(qubit: i32);           // Phase gate S (Z^(1/2))
state.t(qubit: i32);           // T gate (Z^(1/4))
state.r(qubit: i32, theta: f32); // Phase rotation R(θ) = diag(1, e^{iθ})

// Puertas de dos qubits
state.cx(control: i32, target: i32);    // CNOT
state.swap(a: i32, b: i32);             // SWAP

// Puerta de tres qubits
state.toffoli(c1: i32, c2: i32, target: i32); // CCX (Toffoli)

// Puerta arbitraria controlada de un qubit
state.apply_controlled_gate(control: i32, target: i32, gate: Gate);
```

El tipo `Gate` es una matriz 2×2 compleja construida con `qcgpu::gates::r(θ)`.
Esta última función es la que permite implementar la IQFT manualmente (ver
sección 6.1).

### 3.3 Aplicación de puertas sobre el buffer GPU

Cada llamada a `state.h(q)` encola de forma **síncrona** (o con flush
implícito) la ejecución del kernel correspondiente. Desde la perspectiva de
Rust, la llamada retorna cuando el kernel ha terminado. No hay un modelo de
circuito compilado: las puertas se aplican secuencialmente conforme se invocan.

Este diseño simplifica el código pero impide optimizaciones de alto nivel como
la fusión de puertas (gate fusion), que sí hacen frameworks como cuQuantum.

### 3.4 Medición

```rust
// Realiza `shots` mediciones del estado y devuelve un HashMap<String, i32>
// donde la clave es la cadena de bits y el valor es el número de ocurrencias.
let results: std::collections::HashMap<String, i32> = state.measure_many(shots);
```

El formato de la cadena de bits sigue la convención **LSB-first** de qcgpu:
el qubit 0 es el carácter más a la derecha de la cadena. Por ejemplo, el
estado `|5⟩ = |101⟩` (n=3) aparece como `"101"` con qubit 0 = 1, qubit 1 = 0,
qubit 2 = 1.

Esta convención es la opuesta a la mayoría de frameworks (Qiskit usa MSB-first)
y es fuente habitual de confusión cuando se migra código entre frameworks.

El muestreo se realiza en la GPU: se genera una distribución de probabilidades
acumuladas y se muestrean `shots` valores usando números aleatorios. El
histograma resultante se copia de GPU a CPU.

### 3.5 Manejo de errores OpenCL

qcgpu v0.1 no propaga errores de OpenCL de forma idiomática con `Result<T, E>`;
los errores internos de `ocl` hacen `panic!`. En producción esto significa que
hay que capturar los panics con `std::panic::catch_unwind`:

```rust
use std::panic;

let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
    let mut state = State::new(20, 0);
    state.h(0);
    state.measure_many(100)
}));

match result {
    Ok(dist) => { /* usar dist */ }
    Err(e) => {
        let msg = e
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "OpenCL not available".to_string());
        eprintln!("Error OpenCL: {}", msg);
    }
}
```

Este patrón se usa en ambos binarios del TFG (`grover.rs`, `shor/mod.rs`) para
emitir errores en JSON en lugar de terminar el proceso con un crash.

---

## 4. Integración con el Pipeline

### 4.1 Estructura del crate

```
rust/qcgpu/
├── Cargo.toml
└── src/
    ├── lib.rs           ← expone los módulos grover y shor
    ├── grover.rs        ← implementación completa de Grover
    ├── bin/
    │   ├── grover.rs    ← main() trivial: qcgpu_bench::grover::run()
    │   └── shor.rs      ← main() trivial: qcgpu_bench::shor::run()
    └── shor/
        ├── mod.rs       ← driver de Shor y entrypoint run()
        ├── classical.rs ← aritmética modular y post-procesamiento
        ├── permutation.rs ← red de permutaciones para U_f
        └── qft.rs       ← IQFT manual via gate r(θ)
```

El diseño separa la lógica de la implementación del punto de entrada binario.
Los módulos `grover` y `shor` son bibliotecas reutilizables con sus propios
tests unitarios; los binarios son envoltorios triviales de una línea.

### 4.2 CLI con clap

Ambos binarios exponen una CLI homogénea usando el crate `clap` con la macro de
derivación:

```rust
#[derive(Parser, Debug)]
#[command(name = "grover", about = "Grover's search on qcgpu (OpenCL)")]
pub struct Args {
    #[arg(long)]
    pub n: u32,

    #[arg(long)]
    pub target: u64,

    #[arg(long, default_value_t = 1024)]
    pub shots: usize,

    #[arg(long)]
    pub iterations: Option<usize>,
}
```

Uso desde el pipeline de benchmarks:

```bash
./qcgpu-grover --n 10 --target 42 --shots 1024
./qcgpu-shor --N 15 --shots 10 --tries 3 --seed 42
```

### 4.3 JSON de salida con serde\_json

Toda la salida se produce como JSON en stdout, facilitando el parsing
automatizado por el pipeline Python del TFG:

```rust
#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,   // "qcgpu"
    pub algorithm: &'static str,   // "grover" | "shor"
    pub n: u32,
    pub target: u64,
    pub shots: usize,
    pub found: u64,
    pub time_ms: f64,
    pub distribution: HashMap<String, usize>,
}
```

En caso de error, se emite un `ErrorOutput` con el campo `"error"` y el proceso
termina con código 0 (para que el pipeline Python pueda parsear el JSON sin
confundir el error cuántico con un error de sistema).

### 4.4 Detección de disponibilidad OpenCL

qcgpu no ofrece una función de comprobación previa. La única forma de detectar
la ausencia de OpenCL es intentar crear un `State` y capturar el panic. El
pipeline del TFG usa `catch_unwind` para esto y, si falla, emite el error JSON
correspondiente y el orquestador Python lo registra como `framework_unavailable`.

### 4.5 Manejo del campo `"error"` en `_run_rust_binary()`

El binario Rust ya emitía `{"error": "..."}` en stdout con exit code 0 cuando
OpenCL no está disponible (gracias a `catch_unwind`). Sin embargo, la función
`_run_rust_binary()` del pipeline Python no comprobaba este campo: parseaba el
JSON y continuaba con las 11 repeticiones de benchmarking, todas con datos cero,
generando 11 mensajes de error en el log en lugar de uno.

El fix corrige dos problemas encadenados:

1. **Comprobación del campo `"error"`**: inmediatamente tras parsear el JSON de
   la primera repetición, se verifica `if "error" in payload`. Si existe, se
   captura el stderr del proceso (que contiene el mensaje de panic de OpenCL) y
   se lanza `RuntimeError`. Esto aborta el framework completo tras el primer
   fallo y el llamador lo registra como `ERROR`/`SKIP`.

2. **Supresión del spam de OpenCL**: se cambió `stderr=sys.stderr` por
   `stderr=subprocess.PIPE` en la llamada al subproceso. OpenCL emite el mensaje
   `thread 'main' panicked` en stderr incluso cuando el proceso termina con exit
   code 0 (porque el panic es capturado por `catch_unwind`). Con
   `stderr=subprocess.PIPE` ese spam ya no contamina la salida del pipeline; el
   stderr se guarda en el objeto `CompletedProcess` y sólo se muestra si se
   produce un error real.

El efecto combinado es que un entorno sin OpenCL produce exactamente **1 mensaje
de error** (en lugar de 11) y el resto de frameworks del benchmark continúan con
normalidad.

### 4.6 Fallback si no hay GPU OpenCL

qcgpu v0.1 no tiene modo CPU de fallback. Si no hay plataforma OpenCL, el
crate no puede funcionar. Esto contrasta con frameworks como `quantrs2`, que
puede usar CUDA opcionalmente pero funciona en CPU si no hay GPU disponible.

En el TFG, cuando qcgpu falla por falta de OpenCL, el benchmark se marca como
`"error": "OpenCL not available on this platform"` y se excluye de la
comparativa de rendimiento.

### 4.7 Soporte NVIDIA en Docker

La imagen Docker `base-amd64` usa `nvidia/cuda:12.6.0-runtime-ubuntu22.04`
como imagen base e incluye el paquete `ocl-icd-libopencl1`, que proporciona el
ICD loader de OpenCL. Con **NVIDIA Container Toolkit** instalado en el host, el
flag `--gpus all` hace que el runtime de Docker inyecte automáticamente el ICD
de NVIDIA dentro del contenedor, lo que permite a qcgpu descubrir la GPU a
través de OpenCL.

Para ejecutar qcgpu con GPU NVIDIA en Docker se necesita:

| Requisito | Descripción |
|---|---|
| Host | NVIDIA Container Toolkit instalado (`nvidia-container-toolkit`) |
| Imagen | `mablospate/tfg-bench` (ya incluye `ocl-icd-libopencl1`) |
| Flag Docker | `--gpus all` (inyecta el ICD de NVIDIA) |

Invocación directa:

```bash
docker run --gpus all mablospate/tfg-bench <argumentos>
```

Los scripts `bench.ps1` (Windows/PowerShell) y `bench` (Linux/macOS) del
repositorio detectan automáticamente si hay GPU NVIDIA disponible en el host y
añaden el flag `--gpus all` cuando corresponde, sin necesidad de intervención
manual.

Si la imagen detecta GPU NVIDIA en el host pero `libOpenCL` no está accesible
dentro del contenedor (por ejemplo, si el ICD no se inyectó correctamente),
`run.py` muestra un **warning al inicio** indicando que qcgpu puede fallar, en
lugar de producir un error silencioso más adelante.

---

## 5. Implementación de Grover en qcgpu

### 5.1 La estructura general

El algoritmo de Grover para buscar un elemento marcado `target` en un espacio
de `2^n` estados consiste en:

1. Preparar la superposición uniforme: `H^⊗n |0...0⟩`
2. Aplicar `k ≈ (π/4)√(2^n)` iteraciones, cada una compuesta por:
   - **Oráculo** `O`: invierte la fase del estado marcado `|target⟩`
   - **Difusor** `D`: inversión sobre la media

En qcgpu el estado tiene `n + n_anc` qubits en total:

```rust
pub fn grover_state(n: u32, target: u64, num_iterations: usize) -> State {
    let n_anc: u32 = if n >= 3 { n - 2 } else { 0 };
    let total = n + n_anc;
    let mut state = State::new(total, 0);
    let ancillas: Vec<i32> = (n as i32..(n + n_anc) as i32).collect();

    // Superposición uniforme en el registro de búsqueda
    for i in 0..n as i32 {
        state.h(i);
    }

    for _ in 0..num_iterations {
        build_oracle(&mut state, n, target, &ancillas);
        build_diffuser(&mut state, n, &ancillas);
    }
    state
}
```

### 5.2 Ancillas y la descomposición MCX

qcgpu sólo ofrece Toffoli como puerta de múltiple control. Para implementar
una puerta **multi-controlled X (MCX)** con `k ≥ 3` controles, se usa la
estrategia B de la sección 7.10 del implementation guide: una escalera de
puertas Toffoli que combina los controles sucesivamente en qubits ancilla.

```rust
pub fn mcx(state: &mut State, controls: &[i32], target: i32, ancillas: &[i32]) {
    let k = controls.len();
    match k {
        0 => state.x(target),
        1 => state.cx(controls[0], target),
        2 => state.toffoli(controls[0], controls[1], target),
        _ => {
            // Escalera hacia adelante: combinar controles en ancillas
            state.toffoli(controls[0], controls[1], ancillas[0]);
            for i in 2..(k - 1) {
                state.toffoli(controls[i], ancillas[i - 2], ancillas[i - 1]);
            }
            // Toffoli final: escribe el resultado MCX en target
            state.toffoli(controls[k - 1], ancillas[k - 3], target);
            // Descompute: restituir ancillas a |0⟩
            for i in (2..(k - 1)).rev() {
                state.toffoli(controls[i], ancillas[i - 2], ancillas[i - 1]);
            }
            state.toffoli(controls[0], controls[1], ancillas[0]);
        }
    }
}
```

La escalera requiere `k - 2` ancillas para `k` controles. Como Grover con `n`
qubits de búsqueda necesita MCX sobre los `n` qubits, se necesitan `n - 2`
ancillas. Por eso el estado total tiene `n + (n - 2) = 2n - 2` qubits para
`n ≥ 3`.

### 5.3 El oráculo

El oráculo marca el estado `|target⟩` invirtiendo su fase. La técnica estándar
es:

1. Aplicar X a los qubits donde `target` tiene un bit a 0 (así `|target⟩` se
   mapea a `|11...1⟩`).
2. Aplicar MCZ (multi-controlled Z) a todos los qubits de búsqueda.
3. Revertir los X del paso 1.

```rust
pub fn build_oracle(state: &mut State, n: u32, target: u64, ancillas: &[i32]) {
    let n_i = n as i32;
    let qubits: Vec<i32> = (0..n_i).collect();

    // Paso 1: X en bits donde target tiene 0
    for i in 0..n_i {
        if (target >> i) & 1 == 0 {
            state.x(i);
        }
    }

    mcz(state, &qubits, ancillas);  // Paso 2: MCZ sobre |11...1⟩

    // Paso 3: Deshacer los X
    for i in 0..n_i {
        if (target >> i) & 1 == 0 {
            state.x(i);
        }
    }
}
```

Y `mcz` se implementa como H + MCX + H sobre el último qubit (equivalencia
estándar entre Z controlada y X controlada):

```rust
pub fn mcz(state: &mut State, qubits: &[i32], ancillas: &[i32]) {
    let n = qubits.len();
    match n {
        0 => {}
        1 => state.z(qubits[0]),
        2 => cz(state, qubits[0], qubits[1]),
        _ => {
            let target = qubits[n - 1];
            let controls: Vec<i32> = qubits[..n - 1].to_vec();
            state.h(target);
            mcx(state, &controls, target, ancillas);
            state.h(target);
        }
    }
}
```

### 5.4 El difusor

El difusor implementa la **inversión sobre la media**: `D = 2|ψ⟩⟨ψ| - I`
donde `|ψ⟩` es la superposición uniforme. En términos de circuito:

```
D = H^⊗n · (2|0⟩⟨0| - I) · H^⊗n
```

y `(2|0⟩⟨0| - I)` es MCZ sobre el estado `|00...0⟩`, implementado con X en
todos los qubits antes y después del MCZ sobre `|11...1⟩`:

```rust
pub fn build_diffuser(state: &mut State, n: u32, ancillas: &[i32]) {
    let n_i = n as i32;
    let qubits: Vec<i32> = (0..n_i).collect();

    // H en todos los qubits de búsqueda
    for i in 0..n_i { state.h(i); }
    // X en todos los qubits de búsqueda
    for i in 0..n_i { state.x(i); }

    mcz(state, &qubits, ancillas);  // MCZ sobre |11...1⟩

    // Deshacer X y H
    for i in 0..n_i { state.x(i); }
    for i in 0..n_i { state.h(i); }
}
```

### 5.5 Iteraciones óptimas

El número de iteraciones que maximiza la probabilidad de éxito es:

```
k = floor( (π/4) · √(2^n) )
```

En el código:

```rust
let iters = iterations.unwrap_or_else(|| {
    ((std::f64::consts::PI / 4.0) * (2u64.pow(n) as f64).sqrt()).floor() as usize
});
```

Para `n = 3` → `k = 2`; para `n = 4` → `k = 3`; para `n = 10` → `k = 25`.
Con estas iteraciones la probabilidad del estado marcado es `≥ 1 - 1/N ≈ 1`.

### 5.6 Particularidades de ejecutar Grover en GPU via OpenCL

Cada llamada a `state.h(i)`, `state.x(i)`, `state.toffoli(...)`, etc., lanza
un kernel OpenCL. Para Grover con `n = 10` y `k = 25` iteraciones:

- Cada iteración tiene: `n` X condicionales + MCZ (2H + escalera de Toffoli)
  + difusor (2n H/X + MCZ) ≈ `O(n)` kernels.
- Total: `O(k·n) = O(n·√N)` lanzamientos de kernels.

Cada lanzamiento de kernel en OpenCL tiene una latencia de 1–10 μs. Para
`n = 20`, `k ≈ 800` iteraciones, y la escalera MCX tiene `n - 2 = 18` puertas
Toffoli → el número total de lanzamientos de kernels supera los 50 000. El
overhead acumulado puede ser significativo en comparación con simuladores que
fusionan puertas antes de enviarlas a la GPU.

---

## 6. Implementación de Shor en qcgpu

### 6.1 QFT manual via OpenCL

qcgpu no tiene un operador QFT incorporado. La **Transformada Cuántica de
Fourier inversa (IQFT)** se implementa explícitamente usando la puerta `r(θ)`:

```rust
use qcgpu::gates;
use qcgpu::State;
use std::f64::consts::PI;

pub fn apply_inverse_qft(state: &mut State, qubits: &[i32]) {
    let n = qubits.len();

    // Paso 1: deshacer el orden de bits (swap)
    for i in 0..n / 2 {
        state.swap(qubits[i], qubits[n - 1 - i]);
    }

    // Paso 2: aplicar QFT inversa en orden descendente
    for i in (0..n).rev() {
        // Rotaciones de fase controladas con ángulo negativo (inversa)
        for j in ((i + 1)..n).rev() {
            let angle = -(PI / (1u64 << (j - i)) as f64) as f32;
            state.apply_controlled_gate(qubits[j], qubits[i], gates::r(angle));
        }
        state.h(qubits[i]);
    }
}
```

Nótese que los ángulos se calculan en `f64` y se convierten a `f32` para la
API de qcgpu. Para `m = 8` qubits de control (usado en N = 15) la precisión
`f32` es suficiente: el ángulo más pequeño es `π/128 ≈ 0.0245`, que `f32`
representa con 7 dígitos significativos de precisión.

Para `m > 20` la situación cambiaría: `π/2^20 ≈ 3×10^{-6}`, que está cerca del
límite de precisión de `f32` (≈ 1.2×10^{-7}). Para N grandes sería necesario
un simulador con `f64` o una representación de ángulos exacta.

### 6.2 La exponenciación modular

La estrategia implementada en el TFG es la **exponenciación modular via red de
permutaciones** (no un circuito de aritmética cuántica). La unidad `U_f` que
realiza `|y⟩ → |a^power · y mod N⟩` se implementa como:

1. **Calcular la permutación clásica** `π: y → (a^power · y) mod N` para
   `y = 0, …, N-1`.
2. **Descomponer en ciclos disjuntos**.
3. **Cada ciclo se descompone en transposiciones** (intercambios de pares de
   estados base).
4. **Cada transposición** se implementa como una puerta MCX controlada por el
   qubit de control de la estimación de fase.

```rust
pub fn build_mod_exp_permutation(a: u64, n_val: u64, power: u64) -> HashMap<u64, u64> {
    let a_power = mod_pow(a, power, n_val);
    let mut perm: HashMap<u64, u64> = HashMap::new();
    for y in 0..n_val {
        let target = (a_power * y) % n_val;
        if y != target {                 // omitir puntos fijos
            perm.insert(y, target);
        }
    }
    perm
}
```

La ventaja de esta aproximación es que **no requiere diseñar un circuito
aritmético modular** (que es complejo y dependiente de N). La desventaja es que
la descomposición en transposiciones puede generar un número exponencial de
puertas MCX en el peor caso.

### 6.3 Estimación de fase cuántica (QPE)

El algoritmo de Shor completo usa QPE para encontrar el orden:

```rust
pub fn order_finding_state(a: u64, n_val: u64, precision: usize) -> Option<(State, usize)> {
    let width = bit_width(n_val) as usize;
    let m = precision;    // m qubits de control
    let n_anc = width.max(2);
    let total = m + width + n_anc;

    let mut state = State::new(total as u32, 0);

    // Hadamard en el registro de control
    for &q in &ctrl_qubits {
        state.h(q);
    }

    // Inicializar registro objetivo a |1⟩
    state.x(tgt_qubits[0]);

    // Exponenciación modular controlada
    for i in 0..m {
        let power = 1u64 << (m - 1 - i);
        let perm = build_mod_exp_permutation(a, n_val, power);
        if !perm.is_empty() {
            controlled_swap_permutation(&mut state, ctrl_qubits[i],
                                        &tgt_qubits, &perm, &ancillas);
        }
    }

    apply_inverse_qft(&mut state, &ctrl_qubits);
    Some((state, m))
}
```

La precisión por defecto es `m = 2 · ⌈log₂(N)⌉`, que coincide con el número
de qubits de control que usa la referencia de Qiskit para N = 15 (`m = 8`).

### 6.4 Post-procesamiento clásico: fracciones continuas

Una vez medido el registro de control, el resultado `x/2^m` es una
aproximación a `s/r` donde `r` es el orden buscado. Para recuperar `r` se usa
el algoritmo de **fracción continua**:

```rust
pub fn limit_denominator(
    numerator: i128,
    denominator: i128,
    max_denominator: i128,
) -> (i128, i128) {
    // Implementación fiel de Fraction.limit_denominator de CPython
    // usando i128 para evitar overflow con N grandes
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let (mut n, mut d) = (numerator, denominator);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_denominator { break; }
        let p2 = p0 + a * p1;
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
        let new_d = n - a * d;
        n = d;
        d = new_d;
        if d == 0 { break; }
    }
    // Seleccionar el mejor convergente
    let k = (max_denominator - q0) / q1;
    let (b1n, b1d) = (p0 + k * p1, q0 + k * q1);
    let (b2n, b2d) = (p1, q1);
    if (b2n * denominator - numerator * b2d).abs() * b1d
        <= (b1n * denominator - numerator * b1d).abs() * b2d
    {
        (b2n, b2d)
    } else {
        (b1n, b1d)
    }
}
```

El uso de `i128` (en lugar de `i64`) es crucial: para N = 15 con `m = 8`,
`denominator = 2^8 = 256` y los cálculos intermedios se mantienen en rango.
Pero para N grandes (N ~ 10^6, m ~ 40) los intermedios necesitan `i128`.

### 6.5 Limitaciones específicas de qcgpu para Shor

1. **Precisión f32**: para N > 100, `m > 14` qubits de control, los ángulos de
   la IQFT empiezan a perder precisión relevante.

2. **Escala de transposiciones**: el número de puertas MCX generadas por la red
   de permutaciones crece con el número de ciclos de la permutación. Para N = 15
   esto es manejable (N-1 = 14 valores no fijos). Para N = 35 serían 34 valores.
   El número de lanzamientos de kernels OpenCL puede ser prohibitivo.

3. **Sin QFT nativa**: la ausencia de un kernel QFT optimizado significa que
   qcgpu aplica las rotaciones una a una (O(m²) kernels), frente a un kernel
   QFT fusionado que haría todo en O(1) lanzamientos.

4. **Restricción de plataforma**: solo Linux x86_64 con driver OpenCL instalado.
   En el TFG, Shor con qcgpu sólo se ejecuta en la máquina de benchmarks Linux.

---

## 7. Rendimiento y Comparación

### 7.1 El umbral de qubits: cuándo qcgpu supera a CPU

En simulación de statevector, la GPU es ventajosa cuando el tamaño del estado
es suficientemente grande para amortzar el overhead de lanzamiento de kernels.
Para qcgpu, el umbral aproximado es:

- **n < 12**: la CPU (numpy, ndarray) puede ser más rápida por el overhead de
  OpenCL y el bajo paralelismo.
- **12 ≤ n ≤ 20**: qcgpu comienza a superar a simuladores CPU de un solo hilo.
- **n > 20**: la ventaja GPU es clara en tiempo de wall-clock, pero la memoria
  se convierte en el cuello de botella.

### 7.2 Comparación con quantrs2 (CUDA)

`quantrs2` (framework CUDA del TFG, basado en cuQuantum/cuStateVec) usa el
backend CUDA con puertas fusionadas y kernels optimizados para NVIDIA:

| Métrica | qcgpu (OpenCL) | quantrs2 (CUDA) |
|---|---|---|
| Backend | OpenCL 1.2 | CUDA + cuStateVec |
| Fusión de puertas | No | Sí (automática) |
| Precisión amplitudes | f32 | f64 |
| Lanzamientos por puerta | 1 kernel por puerta | Bloques fusionados |
| Overhead inicialización | 50–500 ms | 100–300 ms |
| Rendimiento relativo (Grover n=20) | Referencia | 5–15× más rápido |
| Portabilidad hardware | AMD+Intel+NVIDIA | Solo NVIDIA |
| Soporte macOS | Sí (deprecado) | No |

La diferencia de rendimiento de 5–15× viene principalmente de:
1. La fusión de puertas de cuStateVec: combina secuencias de puertas en una
   sola multiplicación matricial de mayor tamaño.
2. El compilador CUDA optimiza mejor el acceso a memoria VRAM que OpenCL.
3. cuStateVec usa f64 con las mismas técnicas de vectorización SIMD.

### 7.3 El overhead de OpenCL: inicialización y platform selection

Al iniciar, qcgpu llama a `ocl::Platform::list()` que interroga a todos los
drivers OpenCL instalados. En un sistema con varios drivers (NVIDIA + Intel
onboard), esto puede tardar 200–500 ms. Este costo fijo aparece como overhead
en los benchmarks de circuitos pequeños.

En el pipeline del TFG se mide el tiempo desde que comienza `run_grover` hasta
que termina `measure_many`, incluyendo este overhead. Para circuitos de n = 5
qubits, el overhead de inicialización domina completamente el tiempo medido.

### 7.4 Por qué qcgpu es relevante a pesar de ser más lento que CUDA

A pesar de sus limitaciones de rendimiento, qcgpu es valioso por:

1. **Portabilidad real**: es el único framework del TFG que puede ejecutar en
   hardware AMD y Intel sin modificaciones. Esto es relevante en entornos HPC
   donde NVIDIA no es el proveedor exclusivo.

2. **Simplicidad de implementación**: su API minimalista lo hace ideal para
   entender los principios de la simulación GPU. No hay abstracciones de alto
   nivel que oculten lo que ocurre en la GPU.

3. **Código abierto y auditabilidad**: el código del kernel OpenCL es legible y
   modificable. Frameworks como cuStateVec son de código cerrado.

4. **Benchmark de referencia OpenCL**: en el ecosistema de simuladores cuánticos
   hay pocos puntos de datos para OpenCL. qcgpu llena ese hueco.

5. **Demostración académica**: para un TFG que compara frameworks, incluir un
   simulador OpenCL frente a uno CUDA ilustra cuantitativamente la diferencia
   entre ambos ecosistemas en la misma tarea.

---

## 8. Ejemplo Integrado: Grover de n=4, target=11

El siguiente bloque muestra el flujo completo tal y como lo ejecuta el pipeline
del TFG:

```rust
use qcgpu_bench::grover::run_grover;

fn main() {
    // Búsqueda de |11⟩ = |1011⟩ en un espacio de 16 estados
    let n = 4u32;
    let target = 11u64;
    let shots = 1024;

    // k = floor(π/4 · √16) = floor(π) = 3 iteraciones
    let (found, dist) = run_grover(n, target, None, shots);

    println!("Encontrado: {} (esperado: {})", found, target);
    println!("Distribución de las 5 cadenas más frecuentes:");
    let mut entries: Vec<_> = dist.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    for (bitstring, count) in entries.iter().take(5) {
        let prob = **count as f64 / shots as f64;
        println!("  |{}⟩ : {:.1}%", bitstring, prob * 100.0);
    }
}
```

Salida esperada (ejemplo con OpenCL disponible):

```
Encontrado: 11 (esperado: 11)
Distribución de las 5 cadenas más frecuentes:
  |1011⟩ : 94.7%
  |0011⟩ :  0.9%
  |1010⟩ :  0.8%
  |1001⟩ :  0.7%
  |0101⟩ :  0.6%
```

El estado `|1011⟩` (que corresponde a 11 en binario con LSB a la derecha)
domina con ~95% de probabilidad después de 3 iteraciones de Grover.

---

## 9. Convención LSB-first: una trampa habitual

La convención de indexación de qcgpu merece un análisis detallado porque es una
fuente frecuente de bugs al migrar código.

En qcgpu, el qubit 0 es el bit **menos significativo** (rightmost) de la
cadena de bits. Por tanto:

| Estado cuántico | Representación Qiskit (MSB-first) | Representación qcgpu (LSB-first) |
|---|---|---|
| `|5⟩ = |101⟩` | `"101"` | `"101"` (coincide para palíndromos) |
| `|6⟩ = |110⟩` | `"110"` | `"011"` |
| `|1⟩ = |001⟩` | `"001"` | `"100"` |

Para extraer correctamente el registro de búsqueda de `n` qubits de la medición
completa (que incluye ancillas), hay que tomar los **últimos** `n` caracteres
de la cadena (los menos significativos):

```rust
// En qcgpu, qubit 0 es el carácter más a la DERECHA de la cadena
let key = bs[total_len.saturating_sub(n as usize)..].to_string();
```

Y al interpretar bitstrings de medición del registro de control en QPE, hay que
**revertir** la cadena antes de parsear porque ctrl[0] es el MSB del resultado:

```rust
// En shor/classical.rs
let reversed: String = bs.chars().rev().collect();
let x = u64::from_str_radix(&reversed, 2).unwrap_or(0);
```

---

## 10. Tests y Calidad del Código

### 10.1 Separación de tests con y sin OpenCL

El codebase separa explícitamente los tests que requieren OpenCL de los que no:

```rust
// Tests pure-Rust (siempre ejecutan en CI):
#[test]
fn test_default_iterations_formula() {
    assert_eq!(default_iterations(3), 2);
    assert_eq!(default_iterations(4), 3);
}

#[test]
fn test_ancilla_count() {
    let count = |n: u32| -> u32 { if n >= 3 { n - 2 } else { 0 } };
    assert_eq!(count(8), 6);
}

// Tests dependientes de OpenCL (ignorados en CI):
#[test]
#[ignore]
fn test_grover_finds_target_n3() {
    let (found, _) = run_grover(3, 5, None, 200);
    assert_eq!(found, 5);
}
```

Los tests ignorados se pueden ejecutar en la máquina de benchmarks con:

```bash
cargo test -- --include-ignored
```

### 10.2 Cobertura de la aritmética clásica

Los módulos `classical.rs` y `permutation.rs` tienen cobertura de tests
exhaustiva porque no dependen de OpenCL:

```rust
#[test]
fn test_limit_denominator_exact() {
    // Verifica que 64/256 con max_denom=14 da 1/4
    let (n, d) = limit_denominator(64, 256, 14);
    assert_eq!((n, d), (1, 4));
}

#[test]
fn test_build_mod_exp_permutation() {
    // 4 * y mod 5: 1→4, 2→3, 3→2, 4→1
    let perm = build_mod_exp_permutation(2, 5, 2);
    assert_eq!(perm[&1], 4);
    assert_eq!(perm[&3], 2);
}
```

---

## 11. Resumen Comparativo de Frameworks del TFG

| Framework | Lenguaje impl. | Backend GPU | Grover | Shor | Portabilidad |
|---|---|---|---|---|---|
| Qiskit | Python/C++ | CUDA (Aer-GPU) | Sí | Sí | Linux/macOS/Win |
| Cirq | Python | No (CPU) | Sí | Sí | Universal |
| cudaq | C++/Python | CUDA nativo | Sí | Sí | Linux (NVIDIA) |
| qdislib | Python | CPU distribuida | Sí | Sí | Linux cluster |
| q1tsim | Rust | CPU | Sí | Parcial | Universal |
| quantr | Rust | CPU | Sí | No | Universal |
| quantrs2 | Rust | CUDA (cuStateVec) | Sí | Sí | Linux (NVIDIA) |
| **qcgpu** | **Rust** | **OpenCL** | **Sí** | **Sí** | **Linux (AMD/Intel/NVIDIA)** |

qcgpu es el único framework del TFG con backend OpenCL, lo que lo hace único
en el ecosistema de simuladores cuánticos en Rust.

---

## Referencias

1. **Nielsen, M. A., & Chuang, I. L.** (2010). *Quantum Computation and
   Quantum Information* (10th Anniversary Edition). Cambridge University Press.
   — Referencia canónica para las puertas cuánticas, algoritmo de Grover
   (cap. 6) y algoritmo de Shor (cap. 5).

2. **Khronos Group** (2021). *OpenCL 3.0 Reference Guide*.
   https://www.khronos.org/opencl/ — Especificación oficial de OpenCL 3.0.

3. **Haner, T., & Steiger, D. S.** (2017). 0.5 Petabyte simulation of a
   45-qubit quantum circuit. *Proceedings of SC'17*, ACM.
   https://doi.org/10.1145/3126908.3126947 — Técnicas de simulación distribuida
   de statevector a escala de petabyte.

4. **Efthymiou, S., et al.** (2022). Qibo: A framework for quantum simulation
   with hardware acceleration. *Quantum Science and Technology*, 7(1), 015018.
   https://doi.org/10.1088/2058-9565/ac39f5 — Comparativa de backends GPU
   (CUDA, OpenCL, TensorFlow) para simulación cuántica.

5. **Bayraktar, H., et al.** (2023). cuQuantum SDK: A high-performance library
   for accelerating quantum science. *IEEE International Conference on Quantum
   Computing and Engineering (QCE)*, IEEE.
   https://doi.org/10.1109/QCE57702.2023.00039 — Presentación oficial de
   cuStateVec (backend CUDA de NVIDIA para statevector).

6. **Suzuki, Y., et al.** (2021). Qulacs: A fast and versatile quantum circuit
   simulator for research purpose. *Quantum*, 5, 559.
   https://doi.org/10.22331/q-2021-10-06-559 — Técnicas de optimización en
   simuladores de statevector de alta velocidad (CPU y GPU).

7. **McCaskey, A., et al.** (2020). Quantum chemistry as a benchmark for
   near-term quantum computers. *npj Quantum Information*, 5(1), 99.
   https://doi.org/10.1038/s41534-019-0209-0 — Análisis de rendimiento de
   simuladores cuánticos en aplicaciones reales.

8. **Smelyanskiy, M., et al.** (2016). qHiPSTER: The Quantum High Performance
   Software Testing Environment. *arXiv:1601.07195*.
   https://arxiv.org/abs/1601.07195 — Técnicas de paralelización GPU/CPU para
   statevector, incluyendo el patrón de kernel "pair-wise".

9. **De Raedt, H., et al.** (2019). Massively parallel quantum computer
   simulator, eleven years later. *Computer Physics Communications*, 237,
   47–61. https://doi.org/10.1016/j.cpc.2018.11.005 — Evolución histórica y
   técnica de simuladores cuánticos paralelos durante once años.

10. **Kelly, A.** (2018). *qcgpu: GPU-accelerated quantum computer simulation*.
    https://github.com/AQIS/qcgpu — Repositorio original del crate qcgpu,
    incluyendo el paper de acompañamiento *"Simulating Quantum Computers Using
    OpenCL"* (arXiv:1805.00988, https://arxiv.org/abs/1805.00988).

---

*Esta lección forma parte del Trabajo de Fin de Grado "Comparativa de
Frameworks de Simulación Cuántica" — Universidad Politécnica de Madrid, 2026.*
