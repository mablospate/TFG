# quantrs2 — Simulador Cuántico en Rust con Soporte CUDA

> Lección universitaria — nivel de posgrado  
> Área: Computación Cuántica · Ingeniería de Software de Alto Rendimiento

---

## Tabla de Contenidos

1. [Contexto e Historia](#1-contexto-e-historia)
2. [Arquitectura: CPU y GPU](#2-arquitectura-cpu-y-gpu)
3. [API de quantrs2](#3-api-de-quantrs2)
4. [Implementación de Grover en quantrs2](#4-implementación-de-grover-en-quantrs2)
5. [Implementación de Shor en quantrs2](#5-implementación-de-shor-en-quantrs2)
6. [Inestabilidades y Limitaciones](#6-inestabilidades-y-limitaciones)
7. [Potencial y Comparación](#7-potencial-y-comparación)
8. [Referencias](#8-referencias)

---

## 1. Contexto e Historia

### 1.1 El ecosistema de simulación cuántica en Rust

Rust se ha convertido en un lenguaje de elección para el software de infraestructura cuántica por razones bien fundadas: su modelo de propiedad elimina una clase entera de errores de memoria que son especialmente peligrosos cuando se manipulan vectores de estado de alta dimensión, su sistema de tipos con genéricos const permite expresar restricciones de tamaño de circuito en tiempo de compilación en lugar de tiempo de ejecución, y su rendimiento se equipara al de C++ sin el coste cognitivo de la gestión manual de memoria.

La familia `quantrs` nació como exploración de ese espacio. El crate original **quantrs** fue un simulador modesto de circuitos cuánticos con API ergonómica pero sin aceleración hardware. La versión **quantrs2** representa una reescritura con una arquitectura multicrate más ambiciosa y, crucialmente, con la promesa de soporte CUDA para la simulación de vectores de estado en GPU.

### 1.2 quantrs2 v0.1.3: Release Candidate inestable

En el momento en que este proyecto fue desarrollado, quantrs2 se encontraba en la versión **0.1.3**, clasificada internamente como Release Candidate. El término "Release Candidate" en el ecosistema de crates.io tiene una connotación especial: indica que la API pública puede cambiar entre versiones menores sin garantía de compatibilidad semántica, que algunas funcionalidades pueden estar parcialmente implementadas, y que la documentación puede no reflejar el estado real del código.

Esta inestabilidad se manifiesta de forma práctica en el proyecto: en la declaración de dependencias del `Cargo.toml` del benchmark se usa `quantrs2 = "0.1"` en lugar de fijar una versión exacta, lo que significa que cualquier versión compatible dentro de la serie 0.1.x se considera aceptable:

```toml
[dependencies]
quantrs2 = "0.1"
quantrs2-circuit = "0.1"
quantrs2-sim = "0.1"
quantrs2-core = "0.1"
```

### 1.3 La diferencia conceptual entre quantrs y quantrs2

**quantrs** (versión 1) era un crate monolítico. Toda la lógica —definición de puertas, construcción de circuitos, simulación de vectores de estado— vivía en un único árbol de módulos. Esto simplifica la compilación incremental pero limita la reutilización: un usuario que solo necesite la representación del circuito sin la simulación debe igualmente compilar todo el crate.

**quantrs2** adopta una arquitectura multicrate que separa responsabilidades:

- `quantrs2-core`: tipos primitivos compartidos, en particular `QubitId` y las matrices de puertas.
- `quantrs2-circuit`: el tipo `Circuit<N>` para la construcción de circuitos.
- `quantrs2-sim`: el simulador de vectores de estado, separable del constructor de circuitos.
- `quantrs2`: crate paraguas que reexporta lo esencial.

Esta separación permite sustituir el backend de simulación sin modificar el código de construcción del circuito, que es exactamente la abstracción necesaria para soportar CUDA como backend alternativo al CPU.

### 1.4 Soporte CUDA: solo Linux x86\_64

La aceleración GPU en quantrs2 está restringida a **Linux x86_64**. Esta restricción no es arbitraria: el kit de desarrollo CUDA de NVIDIA (CUDA Toolkit) solo ofrece soporte oficial para Linux en la mayoría de sus componentes de bajo nivel, las bibliotecas de enlace dinámico de CUDA no existen en macOS desde 2019 (NVIDIA dejó de fabricar drivers para macOS tras la serie Kepler), y el subsistema WSL2 de Windows, aunque técnicamente soportado, presenta latencias de transferencia de memoria que anulan buena parte de la ganancia de rendimiento.

Para el proyecto de benchmarking de esta tesis, los experimentos de quantrs2 se ejecutan **sin la feature CUDA activa**, ya que el entorno de desarrollo es macOS. El framework se usa exclusivamente con el simulador CPU basado en vector de estado.

### 1.5 Por qué es interesante pese a la inestabilidad

quantrs2 representa un hito: es el primer framework de simulación cuántica en Rust que integra un path de aceleración GPU real (no una promesa o un stub). La comunidad cuántica ha visto varios intentos de simuladores GPU en Python —qcgpu, cuQuantum, cuStateVec— pero en Rust los precedentes son casi inexistentes. Incluso en estado RC, el framework demuestra que es posible construir una API ergonómica de generics const que se compile a código eficiente tanto en CPU como en GPU, lo cual tiene valor didáctico y técnico independientemente de su estabilidad de producción.

---

## 2. Arquitectura: CPU y GPU

### 2.1 Los dos backends de simulación

quantrs2 separa el frontend (construcción del circuito) del backend (ejecución física). Esta separación se materializa en el trait `Simulator` de `quantrs2-sim`, del que existen al menos dos implementaciones:

- **`StateVectorSimulator`** (CPU): simula el estado cuántico como un vector complejo de longitud `2^N` en memoria RAM, aplicando matrices unitarias mediante multiplicación densa.
- **Backend CUDA** (Linux x86_64 con feature `cuda`): delega las multiplicaciones matriciales al dispositivo GPU, manteniendo el vector de estado en memoria de dispositivo (VRAM) y minimizando las transferencias de datos.

En el código del proyecto, toda la ejecución ocurre a través del backend CPU:

```rust
use quantrs2_sim::statevector::StateVectorSimulator;

let sim = StateVectorSimulator::new();
let reg = c.run(sim).expect("simulation failed");
let probs = reg.probabilities();
```

La API es idéntica tanto para CPU como para GPU: `c.run(sim)` devuelve un registro cuántico del que se extraen probabilidades. Cambiar de backend es, en teoría, solo cambiar el tipo del simulador.

### 2.2 Feature flags de Cargo que habilitan CUDA

En Rust, las características opcionales se declaran como `features` en `Cargo.toml`. Para el backend CUDA de quantrs2-sim, la estructura hipotética sería:

```toml
[features]
default = []
cuda = ["dep:quantrs2-cuda-sys", "dep:cuda-runtime-sys"]
```

La feature `cuda` no se activa en el proyecto de esta tesis porque el entorno de compilación es macOS. Para activarla en Linux:

```bash
cargo build --release --features cuda
```

Esto instruye al compilador a incluir el código condicionalmente compilado bajo `#[cfg(feature = "cuda")]` y a enlazar contra las bibliotecas CUDA del sistema.

### 2.3 Selección en tiempo de compilación vs. tiempo de ejecución

La distinción entre estos dos modos de selección es fundamental en sistemas de alto rendimiento:

**Tiempo de compilación**: El compilador genera código para un único backend. Las rutas no usadas son eliminadas por dead code elimination. El costo de despacho es cero en tiempo de ejecución. Esta es la aproximación que quantrs2 toma con sus features.

**Tiempo de ejecución**: Existe un dispatch dinámico que selecciona el backend en función de la disponibilidad del hardware. Permite un único binario que funciona tanto en máquinas con GPU como sin ella, pero introduce una indirección (vtable) en cada llamada al simulador.

El diseño de quantrs2 con `Circuit<N>::run(simulator)` donde `simulator` es genérico sobre el trait `Simulator` permite ambos enfoques: si el tipo concreto se resuelve en tiempo de compilación (monomorfización), el compilador puede inlinar la llamada. Si se usa `Box<dyn Simulator>`, se obtiene dispatch dinámico.

### 2.4 La abstracción CPU/GPU en el código de benchmarking

El código de benchmarking del proyecto es agnóstico al backend. La función `run_grover` construye el circuito, llama a `StateVectorSimulator::new()`, y ejecuta `c.run(sim)`. Si en el futuro se quisiera ejecutar en GPU, bastaría con reemplazar `StateVectorSimulator::new()` por la variante CUDA sin modificar nada más en la lógica del algoritmo.

Esta separación limpia entre algoritmo y hardware es una de las contribuciones de diseño más valiosas de quantrs2.

---

## 3. API de quantrs2

### 3.1 El tipo central: `Circuit<N>`

La característica más llamativa de la API de quantrs2 es el uso de **genéricos const** para codificar el número de qubits en el tipo del circuito:

```rust
let mut c: Circuit<8> = Circuit::new();
```

El parámetro `N = 8` es un entero conocido en tiempo de compilación. El compilador puede por tanto:

- Verificar estáticamente que los índices de qubit están dentro del rango.
- Dimensionar el vector de estado (`2^N` amplitudes) sin asignaciones dinámicas.
- Generar código especializado para cada tamaño de circuito mediante monomorfización.

El precio es que no existe una construcción dinámica como `Circuit::new(n_qubits)`. Para algoritmos cuyo tamaño solo se conoce en tiempo de ejecución —como Grover con `n` configurable por argumento de línea de comandos— es necesario un dispatcher que traduzca del mundo dinámico al mundo estático.

### 3.2 Identificadores de qubit: `QubitId`

Los qubits se referencian mediante el tipo `QubitId` del crate `quantrs2-core`:

```rust
use quantrs2_core::qubit::QubitId;

let q0 = QubitId::new(0);
let q1 = QubitId::new(1);
```

El uso de un tipo envolvente en lugar de enteros desnudos tiene varias ventajas: el compilador puede distinguir qubits de otros enteros, los errores de tipo aparecen en tiempo de compilación, y la API queda autodocumentada.

### 3.3 Puertas disponibles y sintaxis

quantrs2 expone las puertas cuánticas fundamentales como métodos en `Circuit<N>`:

| Puerta         | Sintaxis                                       | Notas                                        |
|----------------|------------------------------------------------|----------------------------------------------|
| Hadamard       | `c.h(q)`                                       | Pone qubit en superposición uniforme         |
| Pauli-X        | `c.x(q)`                                       | Negación cuántica (NOT)                      |
| Pauli-Z        | `c.z(q)`                                       | Inversión de fase del estado `\|1>`          |
| CZ             | `c.cz(q0, q1)`                                 | Z controlado                                 |
| CNOT           | `c.cnot(control, target)`                      | X controlado                                 |
| Toffoli        | `c.toffoli(c0, c1, target)`                    | X doblemente controlado                      |
| RZ(θ)          | `c.rz(q, theta)`                               | Rotación sobre eje Z                         |
| CRZ(θ)         | `c.crz(control, target, theta)`                | Rotación controlada sobre Z                  |
| SWAP           | `c.swap(q0, q1)`                               | Intercambio de estados                       |

Todos los métodos devuelven `Result<(), _>`, de modo que los errores (por ejemplo, índice fuera de rango) se capturan explícitamente mediante `.unwrap()` o manejo de errores estructurado.

**Nota importante**: quantrs2 **no tiene MCX nativo** (multi-controlled X de aridad arbitraria). Esto obliga a descomponer manualmente las puertas multicontroladas en escaleras de Toffoli, como veremos en detalle en las secciones 4 y 5.

### 3.4 Ejecución y obtención de probabilidades

El flujo de ejecución sigue tres pasos:

```rust
// 1. Construir el circuito
let mut c: Circuit<8> = Circuit::new();
c.h(QubitId::new(0)).unwrap();
// ... más puertas ...

// 2. Ejecutar con el simulador
let sim = StateVectorSimulator::new();
let reg = c.run(sim).expect("simulation failed");

// 3. Obtener distribución de probabilidad
let probs: &[f64] = reg.probabilities();
// probs[i] = |<i|psi>|^2 para i en 0..2^N
```

El vector `probs` tiene longitud `2^N` y suma exactamente 1.0. No hay un método de medición estocástica incorporado en quantrs2; el muestreo se implementa manualmente mediante `WeightedIndex` del crate `rand`:

```rust
use rand::distributions::WeightedIndex;
use rand::prelude::*;

let dist = WeightedIndex::new(probs).expect("invalid probability distribution");
let mut rng = thread_rng();
let sample = dist.sample(&mut rng);
```

Esta decisión de diseño —exponer probabilidades brutas en lugar de simular mediciones— es coherente con el enfoque educativo y de benchmarking, pero puede sorprender a usuarios acostumbrados a simuladores que devuelven histogramas de conteo directamente.

### 3.5 Comparación de API con quantr y q1tsim

| Aspecto                  | quantr            | q1tsim             | quantrs2                         |
|--------------------------|-------------------|--------------------|----------------------------------|
| Tamaño de circuito       | Dinámico          | Dinámico           | Const-generic (estático)         |
| MCX nativa               | No (ancilla manual) | Parcial           | No (Toffoli ladder manual)       |
| Backend GPU              | No                | No                 | Si (CUDA, Linux x86_64)          |
| Medición integrada       | Si                | Si                 | No (probabilidades brutas)       |
| Estado inicial           | \|0...0>          | \|0...0>           | \|0...0>                         |
| Versión semántica estable| Si                | Si                 | RC (0.1.x inestable)             |

---

## 4. Implementación de Grover en quantrs2

### 4.1 El problema del dispatch estático

El algoritmo de Grover requiere `n` qubits para el espacio de búsqueda. En el proyecto, `n` es un parámetro de línea de comandos que solo se conoce en tiempo de ejecución. Pero `Circuit<N>` requiere `N` en tiempo de compilación. Esta tensión se resuelve mediante un **dispatcher estático** que enumera todos los valores soportados:

```rust
pub fn run_grover(
    n: usize,
    target: u64,
    shots: u32,
    iterations: Option<usize>,
) -> (u64, HashMap<String, usize>) {
    match n {
        3 => grover_impl::<3>(n, target, shots, iterations),
        4 => grover_impl::<6>(n, target, shots, iterations),
        5 => grover_impl::<8>(n, target, shots, iterations),
        6 => grover_impl::<10>(n, target, shots, iterations),
        7 => grover_impl::<12>(n, target, shots, iterations),
        8 => grover_impl::<14>(n, target, shots, iterations),
        _ => panic!("unsupported n={n}; supported range is 3..=8"),
    }
}
```

La razón por la que el parámetro `TOTAL` es mayor que `n` se explica en la siguiente sección: los ancillas. Para `n=4` se necesitan `n-2=2` ancillas, por lo que `TOTAL = n + (n-2) = 6`. El patrón general es `TOTAL = 2*n - 2` para `n >= 4`, y `TOTAL = n = 3` para el caso base.

### 4.2 Número óptimo de iteraciones

Antes de construir el circuito, se calcula el número óptimo de iteraciones de Grover. Para un espacio de `N = 2^n` elementos con un único elemento marcado, el número de iteraciones que maximiza la probabilidad de éxito es:

$$k^* = \left\lfloor \frac{\pi}{4} \sqrt{N} \right\rfloor$$

En el código:

```rust
let num_iter = args.iterations.unwrap_or_else(|| {
    let space = (1u64 << args.n) as f64;
    ((std::f64::consts::PI / 4.0) * space.sqrt()).floor() as usize
});
```

Este cálculo es una aplicación directa del análisis de Grover: cada iteración rota el estado en un ángulo `2θ` donde `sin(θ) = 1/√N`, y después de `k*` rotaciones el ángulo acumulado está próximo a `π/2`, maximizando la proyección sobre el estado objetivo.

### 4.3 Estructura del circuito principal

La función `grover_impl` construye el circuito en tres fases:

```rust
fn grover_impl<const TOTAL: usize>(
    n: usize,
    target: u64,
    shots: u32,
    iterations: Option<usize>,
) -> (u64, HashMap<String, usize>) {
    let mut c: Circuit<TOTAL> = Circuit::new();

    // Fase 1: Superposición uniforme sobre los n qubits de búsqueda
    for i in 0..n {
        c.h(QubitId::new(i as u32)).unwrap();
    }

    // Fase 2: Iteraciones de Grover
    for _ in 0..num_iter {
        apply_oracle(&mut c, n, target);   // Oráculo de fase
        apply_diffuser(&mut c, n);         // Difusor (2|s><s| - I)
    }

    // Fase 3: Simulación y muestreo
    let sim = StateVectorSimulator::new();
    let reg = c.run(sim).expect("simulation failed");
    let probs = reg.probabilities();
    let dist = sample_search_qubits::<TOTAL>(&probs, n, shots);
    // ...
}
```

### 4.4 El oráculo de fase

El oráculo de fase aplica una inversión de signo `|target> -> -|target>` sin perturbar ningún otro estado de la base. Se implementa mediante el patrón X–MCZ–X:

```rust
fn apply_oracle<const TOTAL: usize>(c: &mut Circuit<TOTAL>, n: usize, target: u64) {
    // X en los bits donde target tiene 0: convierte |target> en |11..1>
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(QubitId::new(i as u32)).unwrap();
        }
    }
    // MCZ: invierte la fase solo cuando todos los qubits son |1>
    apply_mcz(c, n);
    // Deshace las X para restaurar la base computacional
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(QubitId::new(i as u32)).unwrap();
        }
    }
}
```

La idea es que la puerta MCZ (Z multicontrolada) aplica un factor de fase `-1` únicamente cuando todos sus qubits de entrada son `|1>`. Al precondicionar con puertas X que convierten el patrón de bits de `target` en `|111...1>`, el MCZ actúa efectivamente sobre `|target>`.

### 4.5 El difusor: 2|s><s| - I

El difusor refleja el estado actual respecto a la superposición uniforme `|s>`. Su implementación sigue el patrón H–X–MCZ–X–H:

```rust
fn apply_diffuser<const TOTAL: usize>(c: &mut Circuit<TOTAL>, n: usize) {
    // H: lleva |s> a |0...0>
    for i in 0..n { c.h(QubitId::new(i as u32)).unwrap(); }
    // X: convierte |0...0> en |1...1> (para que MCZ actúe sobre él)
    for i in 0..n { c.x(QubitId::new(i as u32)).unwrap(); }
    // MCZ: fase -1 sobre |1...1>, equivalente a -|0...0><0...0|
    apply_mcz(c, n);
    // X y H: deshace la preparación
    for i in 0..n { c.x(QubitId::new(i as u32)).unwrap(); }
    for i in 0..n { c.h(QubitId::new(i as u32)).unwrap(); }
}
```

El resultado neto es la transformación `2|s><s| - I`, que amplifica la amplitud del estado marcado por el oráculo a expensas del resto.

### 4.6 La escalera de Toffoli para MCZ: el corazón técnico

La función `apply_mcz` implementa una Z multicontrolada sobre los primeros `n` qubits. Para `n <= 3` existen casos base directos; para `n >= 4` se despliega una escalera de Toffoli con `n-2` qubits ancilla:

```rust
fn apply_mcz<const TOTAL: usize>(c: &mut Circuit<TOTAL>, n: usize) {
    match n {
        0 => {}
        1 => { c.z(QubitId::new(0)).unwrap(); }
        2 => { c.cz(QubitId::new(0), QubitId::new(1)).unwrap(); }
        3 => {
            // CCZ = H · Toffoli · H en el qubit objetivo
            c.h(QubitId::new(2)).unwrap();
            c.toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2)).unwrap();
            c.h(QubitId::new(2)).unwrap();
        }
        _ => {
            let num_anc = n - 2;
            let anc_base = n;  // Los ancillas empiezan justo después de los qubits de búsqueda
            let target_q = n - 1;

            // Paso 1: Escalera forward — cada ancilla acumula el AND de más controles
            c.toffoli(
                QubitId::new(0),
                QubitId::new(1),
                QubitId::new(anc_base as u32),
            ).unwrap();
            for k in 1..num_anc {
                c.toffoli(
                    QubitId::new((k + 1) as u32),
                    QubitId::new((anc_base + k - 1) as u32),
                    QubitId::new((anc_base + k) as u32),
                ).unwrap();
            }

            // Paso 2: CZ sobre (last_anc, target_q) implementado como H·CNOT·H
            let last_anc = anc_base + num_anc - 1;
            c.h(QubitId::new(target_q as u32)).unwrap();
            c.cnot(
                QubitId::new(last_anc as u32),
                QubitId::new(target_q as u32),
            ).unwrap();
            c.h(QubitId::new(target_q as u32)).unwrap();

            // Paso 3: Escalera reverse — restaura los ancillas a |0>
            for k in (1..num_anc).rev() {
                c.toffoli(
                    QubitId::new((k + 1) as u32),
                    QubitId::new((anc_base + k - 1) as u32),
                    QubitId::new((anc_base + k) as u32),
                ).unwrap();
            }
            c.toffoli(
                QubitId::new(0),
                QubitId::new(1),
                QubitId::new(anc_base as u32),
            ).unwrap();
        }
    }
}
```

**Por qué la escalera de Toffoli es necesaria**: Una puerta MCZ de `n` controles no existe como primitiva en la mayoría de los procesadores y simuladores. Se descompone en puertas de 2 qubits (Toffoli) mediante la técnica de "AND tree": el primer Toffoli calcula `ancilla[0] = qubit[0] AND qubit[1]`, el segundo `ancilla[1] = qubit[2] AND ancilla[0]`, etc. Al final, `ancilla[n-3]` vale `1` si y solo si todos los controles son `1`. Entonces se aplica CZ (como H·CNOT·H) entre `ancilla[n-3]` y el qubit objetivo. La escalera se recorre en sentido inverso para descomponer los ancillas (devolverlos a `|0>`), lo que es imprescindible para no "contaminar" el estado cuántico con información basura en los qubits ancilla.

**Comparación con quantr**: El framework quantr (lección 08) utiliza exactamente la misma técnica de escalera de Toffoli. Ninguno de los dos tiene MCX nativo de aridad arbitraria. La diferencia está en el estilo de API: quantr usa un constructor de circuitos de estilo fluido con tamaño dinámico, mientras que quantrs2 requiere el const-generic en tiempo de compilación.

### 4.7 Muestreo: solo los qubits de búsqueda

El vector de estado tiene `2^TOTAL` amplitudes, pero solo los primeros `n` qubits son el registro de búsqueda; los `TOTAL - n` restantes son ancillas que deberían estar en `|0>` tras la computación. La función `sample_search_qubits` colapsa marginalmente los ancillas:

```rust
fn sample_search_qubits<const TOTAL: usize>(
    probs: &[f64],
    n: usize,
    shots: u32,
) -> HashMap<String, usize> {
    let mut rng = thread_rng();
    let dist = WeightedIndex::new(probs).expect("invalid probability distribution");
    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..shots {
        let idx = dist.sample(&mut rng);
        let mut bits = String::with_capacity(n);
        for q in (0..n).rev() {
            let b = (idx >> q) & 1;
            bits.push(if b == 1 { '1' } else { '0' });
        }
        *counts.entry(bits).or_insert(0) += 1;
    }
    counts
}
```

El bit `q` en el índice `idx` se extrae con `(idx >> q) & 1`. Los qubits ancilla (posiciones `n..TOTAL`) simplemente no se incluyen en el bitstring resultado; como están en `|0>` su contribución al índice es nula.

La convención de ordenamiento es que el qubit 0 es el bit menos significativo (LSB), lo que produce un bitstring con el qubit más significativo a la izquierda, coherente con la convención de Qiskit.

### 4.8 Tests integrados

El módulo incluye cuatro tests que verifican el comportamiento del algoritmo:

```rust
#[test]
fn test_grover_finds_target_n3() {
    let (found, dist) = run_grover(3, 5, 200, None);
    assert_eq!(found, 5, "Expected to find target 5");
    let count = dist.get("101").copied().unwrap_or(0);
    assert!(count > 100, "Expected >100/200 shots to be '101', got {}", count);
}

#[test]
fn test_grover_finds_target_n4() {
    let (found, dist) = run_grover(4, 11, 200, None);
    assert_eq!(found, 11);
    let count = dist.get("1011").copied().unwrap_or(0);
    assert!(count > 100, "Expected '1011' to dominate, got {}", count);
}
```

El test para `n=4` verifica que se encuentra el estado `|1011>` (decimal 11) con más del 50% de las 200 mediciones, lo que es un umbral conservador dada la probabilidad teórica superior al 95% tras el número óptimo de iteraciones.

---

## 5. Implementación de Shor en quantrs2

### 5.1 Estructura del algoritmo de Shor en el proyecto

La implementación de Shor en quantrs2 sigue una arquitectura modular distribuida en cuatro archivos:

- `shor/mod.rs`: Orquestación principal, circuito de estimación de fase cuántica (QPE), dispatcher const-generic.
- `shor/qft.rs`: Transformada de Fourier Cuántica inversa, construida desde puertas primitivas.
- `shor/classical.rs`: Partes clásicas del algoritmo: exponenciación modular, fracciones continuas, extracción del orden.
- `shor/permutation.rs`: Red de permutación para la multiplicación modular controlada.

### 5.2 El layout del registro cuántico

El circuito de Shor usa un layout preciso de tres regiones:

```
[0..m)            Registro de control (m = 2·ceil(log2 N) qubits)
[m..m+n)          Registro objetivo |y> (n = ceil(log2 N) qubits)
[m+n..m+n+(n-2))  Ancillas para descomponer MCX
```

Para factorizar `N=15` (`n=4` bits, `m=8`): el circuito tiene `8 + 4 + 2 = 14` qubits en total.

Este layout está documentado en el encabezado de `mod.rs`:

```rust
//! Layout of the `Circuit<TOTAL>` register (qubit 0 = LSB):
//!   [0..m)                  - control register, written into via QFT-based PE
//!   [m..m+n)                - target register holding |y> for the order finder
//!   [m+n..m+n+(n-2))        - ancillas used to decompose multi-controlled X
```

### 5.3 El dispatcher const-generic para Shor

Al igual que en Grover, el número de qubits total se conoce solo en tiempo de ejecución pero `Circuit<N>` lo requiere en tiempo de compilación. El dispatcher de Shor es más complejo porque el total depende de `N` de forma no trivial:

```rust
pub fn find_order(a: u64, n_val: u64, num_shots: usize) -> (u64, HashMap<String, usize>) {
    let n_bits = (n_val as f64).log2().ceil() as usize;
    let m = 2 * n_bits;
    let anc = if n_bits >= 3 { n_bits - 2 } else { 0 };
    let total = m + n_bits + anc;

    match total {
        4  => order_finding::<4>(a, n_val, n_bits, m, num_shots),
        6  => order_finding::<6>(a, n_val, n_bits, m, num_shots),
        8  => order_finding::<8>(a, n_val, n_bits, m, num_shots),
        10 => order_finding::<10>(a, n_val, n_bits, m, num_shots),
        // ... hasta 20
        _ => panic!("unsupported total qubit count {total} (N={n_val})"),
    }
}
```

El rango 4–20 permite factorizar números desde los más pequeños hasta `N` de unos pocos bits, suficiente para benchmarking educativo.

### 5.4 El circuito de estimación de fase cuántica

La función `order_finding` construye el circuito QPE completo:

```rust
fn order_finding<const TOTAL: usize>(
    a: u64,
    n_val: u64,
    n_bits: usize,
    m: usize,
    num_shots: usize,
) -> (u64, HashMap<String, usize>) {
    let mut c: Circuit<TOTAL> = Circuit::new();

    // 1. Superposición en el registro de control
    for i in 0..m {
        c.h(QubitId::new(i as u32)).unwrap();
    }
    // 2. Inicializar registro objetivo: |y> = |1>
    c.x(QubitId::new(m as u32)).unwrap();

    // 3. Multiplicaciones modulares controladas: ctrl i aplica a^(2^(m-1-i))
    for i in 0..m {
        let power = 1u64 << (m - 1 - i);
        let a_power = mod_pow(a, power, n_val);
        if a_power == 1 { continue; }  // Identidad: no hace nada
        let perm = build_mod_exp_permutation(a_power, n_val);
        apply_controlled_permutation::<TOTAL>(&mut c, i, m, n_bits, &perm);
    }

    // 4. QFT inversa sobre el registro de control
    inverse_qft(&mut c, 0, m);

    // 5. Simulación y muestreo
    let sim = StateVectorSimulator::new();
    let reg = c.run(sim).expect("simulation failed");
    let probs = reg.probabilities();
    let dist = sample_control_register::<TOTAL>(&probs, m, num_shots);

    // 6. Extraer el orden mediante fracciones continuas
    let r = get_order_from_dist(&dist, a, n_val, m);
    (r, dist)
}
```

### 5.5 La QFT inversa en quantrs2

La transformada de Fourier cuántica inversa se implementa manualmente en `qft.rs` porque quantrs2 no incluye una puerta QFT de alto nivel. El algoritmo sigue la descomposición estándar en puertas de fase controladas y Hadamard:

```rust
pub fn inverse_qft<const TOTAL: usize>(c: &mut Circuit<TOTAL>, start: usize, len: usize) {
    // Paso 1: SWAP para invertir el orden de los qubits
    for i in 0..len / 2 {
        c.swap(
            QubitId::new((start + i) as u32),
            QubitId::new((start + len - 1 - i) as u32),
        ).unwrap();
    }
    // Paso 2: Puertas de fase controladas + Hadamard
    for i in (0..len).rev() {
        for j in (i + 1..len).rev() {
            let angle = -PI / (1u64 << (j - i)) as f64;
            controlled_phase::<TOTAL>(c, start + j, start + i, angle);
        }
        c.h(QubitId::new((start + i) as u32)).unwrap();
    }
}
```

La **fase controlada** `CP(λ)` no existe como puerta nativa en quantrs2, por lo que se sintetiza mediante la identidad:

```
CP(λ) = RZ_control(λ/2) · CRZ(control, target, -λ)
```

Esta descomposición es correcta salvo una fase global irrelevante:

```rust
pub fn controlled_phase<const TOTAL: usize>(
    c: &mut Circuit<TOTAL>,
    control: usize,
    target: usize,
    lambda: f64,
) {
    // RZ en el control da la fase |1>_c: e^{iλ/2}
    c.rz(QubitId::new(control as u32), lambda / 2.0).unwrap();
    // CRZ(-λ): aplica RZ(-λ) al target condicionado al control
    c.crz(
        QubitId::new(control as u32),
        QubitId::new(target as u32),
        -lambda,
    ).unwrap();
}
```

El resultado neto sobre los 4 estados de la base es:
- `|00>` → `|00>` (fase 1)
- `|01>` → `|01>` (fase 1)
- `|10>` → `e^{iλ/2} · e^{iλ/2} |10>` = `e^{iλ}|10>` (que es una fase global)
- `|11>` → `e^{iλ/2} · e^{-iλ/2} · e^{iλ}|11>` = `e^{iλ}|11>`

La puerta `diag(1, 1, 1, e^{iλ})` es precisamente la puerta de fase controlada `CP(λ)`.

### 5.6 La exponenciación modular como red de permutación

La parte más compleja de Shor es la multiplicación modular controlada: la puerta `CU_a` que aplica `y -> a·y mod N` en el registro objetivo condicionalmente al qubit de control. En lugar de construirla desde cero mediante aritmética cuántica completa, el proyecto usa el enfoque de **permutación**: para valores pequeños de `N`, `a·y mod N` es simplemente una permutación de los estados `{0, 1, ..., N-1}`, y cualquier permutación puede descomponerse en transposiciones (SWAPs) de dos elementos.

```rust
pub fn build_mod_exp_permutation(a_power: u64, n_val: u64) -> Vec<(u64, u64)> {
    let mut perm = Vec::new();
    for y in 0..n_val {
        let target = ((a_power as u128) * (y as u128) % n_val as u128) as u64;
        if y != target {
            perm.push((y, target));
        }
    }
    perm
}
```

Esta función calcula la lista de pares `(y, a^p · y mod N)` donde `y != a^p · y mod N`. Cada par es un arco en el grafo de permutación. Los ciclos de este grafo se identifican y cada ciclo se implementa como una secuencia de transposiciones controladas.

La función `apply_controlled_permutation` descompone los ciclos:

```rust
pub fn apply_controlled_permutation<const TOTAL: usize>(
    c: &mut Circuit<TOTAL>,
    ctrl: usize,
    m: usize,
    n_bits: usize,
    perm: &[(u64, u64)],
) {
    // Construir el mapa de permutación y encontrar los ciclos
    let map: Map<u64, u64> = perm.iter().copied().collect();
    let mut visited = std::collections::HashSet::new();
    // Para cada ciclo, aplicar transposiciones controladas secuenciales
    for &start in &keys {
        if visited.contains(&start) { continue; }
        let mut cycle = Vec::new();
        let mut current = start;
        while !visited.contains(&current) {
            visited.insert(current);
            cycle.push(current);
            current = *map.get(&current).unwrap_or(&current);
        }
        // Un ciclo (a0, a1, a2, ...) se implementa como
        // swap(a0,a1), swap(a0,a2), ... (todas controladas)
        for idx in 1..cycle.len() {
            controlled_transposition::<TOTAL>(c, ctrl, m, n_bits, cycle[0], cycle[idx]);
        }
    }
}
```

Cada transposición controlada `|ctrl>|a> <-> |ctrl>|b>` entre dos estados que difieren en exactamente un bit se implementa como un MCX (X multicontrolado) con el qubit de control externo más los qubits del registro objetivo que coinciden entre `a` y `b`.

### 5.7 Extracción del orden mediante fracciones continuas

Tras la QPE, el registro de control contiene una aproximación a `k/r · 2^m` para algún entero `k`, donde `r` es el orden buscado. El algoritmo de fracciones continuas recupera `r` como el denominador de la fracción racional más cercana a la medición:

```rust
pub fn get_order_from_dist(
    dist: &HashMap<String, usize>,
    a: u64,
    n_val: u64,
    precision: usize,
) -> u64 {
    let two_m = 1u64 << precision;
    let limit = if n_val > 1 { n_val - 1 } else { 1 };

    for (bs, _) in sorted.iter().take(10) {
        if bs.chars().all(|ch| ch == '0') { continue; }
        let x = u64::from_str_radix(bs, 2).unwrap();
        let frac = Ratio::new(x as i128, two_m as i128);
        let approx = limit_denominator(frac, limit as i128);
        let r = *approx.denom() as u64;
        if mod_pow(a, r, n_val) == 1 {
            return reduce_to_min_order(r, a, n_val);
        }
    }
    0
}
```

La función `limit_denominator` es un puerto fiel del algoritmo de CPython para `Fraction.limit_denominator`, basado en el desarrollo en fracciones continuas:

```rust
pub fn limit_denominator(frac: Ratio<i128>, max_denom: i128) -> Ratio<i128> {
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let mut n = *frac.numer();
    let mut d = *frac.denom();
    loop {
        if d == 0 { break; }
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_denom { break; }
        // ... actualiza convergentes ...
    }
    // Devuelve el convergente más cercano
}
```

### 5.8 Factorización clásica final

Una vez obtenido el orden `r`, la factorización sigue el método estándar de Shor:

```rust
if r % 2 == 0 {
    let x = mod_pow(a, r / 2, n_val);  // x = a^(r/2) mod N
    if x > 1 {
        let d1 = (x - 1).gcd(&n_val);  // gcd(a^(r/2) - 1, N)
        let d2 = (x + 1).gcd(&n_val);  // gcd(a^(r/2) + 1, N)
        // Si d1 o d2 es no trivial, es un factor
    }
}
```

El método funciona porque si el orden `r` es par, entonces `(a^(r/2))^2 ≡ 1 (mod N)`, lo que significa que `a^(r/2) ≡ ±1 (mod N)` o `N | (a^(r/2) - 1)(a^(r/2) + 1)`. En el caso general (cuando `a^(r/2) ≢ ±1`), `N` debe compartir un factor no trivial con `a^(r/2) - 1` o `a^(r/2) + 1`.

---

## 6. Inestabilidades y Limitaciones

### 6.1 Por qué está en RC

La versión 0.1.x de quantrs2 se clasifica como Release Candidate por varias razones estructurales:

**API no estabilizada**: La interfaz pública de `Circuit<N>` puede cambiar. En particular, la firma del método `run()` y el tipo de retorno del registro cuántico no tienen garantías de compatibilidad semántica entre 0.1.x y 0.2.x. El proyecto actual usa `reg.probabilities()` para obtener el vector de probabilidades, pero versiones futuras podrían exponer amplitudes complejas directamente o cambiar el nombre del método.

**Backend CUDA parcialmente implementado**: La documentación de quantrs2 menciona soporte CUDA pero los bindings de bajo nivel (probablemente basados en `cuda-sys` o una versión propia) no han alcanzado la madurez suficiente para garantizar resultados correctos en todos los casos. Los tests solo cubren el backend CPU.

**Ausencia de puertas de alto nivel**: La falta de MCX nativa, de una puerta QFT directa, y de otras puertas compuestas habituales indica que el framework todavía no ha completado su biblioteca de puertas.

### 6.2 Cómo el proyecto maneja posibles fallos

El código del proyecto es defensivo ante las inestabilidades de quantrs2. Se usan dos estrategias:

**Estrategia 1: `.expect()` con mensajes descriptivos**
```rust
let reg = c.run(sim).expect("simulation failed");
```
Esto convierte los errores internos de quantrs2 en panics informativos en lugar de comportamientos indefinidos.

**Estrategia 2: Validación explícita de precondiciones**
```rust
assert!(anc_base + num_anc <= TOTAL, "not enough qubits for ancillas");
```
Las aserciones en tiempo de ejecución detectan configuraciones inválidas que el sistema de tipos no puede prevenir.

**Estrategia 3: Tests ignorados para código lento/inestable**
```rust
#[test]
#[ignore] // slow: full quantum Shor
fn test_find_factor_15() {
    let f = find_factor(15, 5, 20, Some(42));
    assert!(f == 3 || f == 5, "Expected 3 or 5, got {}", f);
}
```
Los tests que requieren ejecutar el circuito cuántico completo están marcados con `#[ignore]` para que `cargo test` los omita por defecto, permitiendo un ciclo de CI rápido sin sacrificar la verificabilidad.

### 6.3 Restricciones de plataforma estrictas

La restricción Linux x86_64 para CUDA no es solo una limitación operativa: tiene implicaciones de reproducibilidad en investigación. Un experimento ejecutado en CPU en macOS y en GPU en Linux puede producir resultados numéricamente equivalentes pero temporalmente muy distintos, lo que complica la comparación directa de benchmarks. El proyecto gestiona esto documentando explícitamente el entorno de ejecución en la metadata JSON de cada experimento.

La restricción del dispatcher a `n ∈ {3..8}` para Grover y a `total ∈ {4..20}` para Shor es otra limitación práctica: ampliarla requeriría añadir casos al `match`, lo que aumenta el tiempo de compilación (cada brazo genera una instancia distinta del circuito por monomorfización).

---

## 7. Potencial y Comparación

### 7.1 Speedup esperado con GPU para simulación de vectores de estado

La simulación de vectores de estado para `n` qubits requiere mantener y operar sobre un vector de `2^n` amplitudes complejas (números de punto flotante de 64 bits). Para `n=30`, esto representa `2^30 × 16 bytes ≈ 16 GB` de memoria. Las operaciones dominantes son multiplicaciones de vectores densos por matrices dispersas (para puertas de 1 qubit) o permutaciones de bloques (para puertas de 2 qubits).

El speedup de GPU sobre CPU para estas operaciones depende del tamaño del problema:

| Qubits | Tamaño del estado | Speedup típico GPU vs CPU (single node) |
|--------|-------------------|-----------------------------------------|
| 20     | 16 MB             | 2–5×                                    |
| 25     | 512 MB            | 10–30×                                  |
| 30     | 16 GB             | 20–50× (limitado por ancho de banda)   |
| 35+    | > 512 GB          | Requiere multi-GPU o distribución        |

La razón por la que el speedup aumenta con el tamaño es que para circuitos pequeños el overhead de transferencia de datos CPU→GPU domina sobre el cómputo. Para estados grandes, el paralelismo masivo de la GPU (miles de núcleos CUDA operando sobre el vector simultáneamente) compensa holgadamente el overhead.

### 7.2 Comparación con qcgpu (OpenCL): CUDA vs OpenCL

**qcgpu** es un simulador Python de vectores de estado que usa OpenCL para aceleración GPU, soportando tanto tarjetas NVIDIA como AMD e Intel. La comparación con quantrs2-CUDA es instructiva:

| Aspecto                  | qcgpu (OpenCL)                    | quantrs2 (CUDA)                          |
|--------------------------|-----------------------------------|------------------------------------------|
| Lenguaje del simulador   | Python + OpenCL kernel            | Rust + CUDA kernel                       |
| Portabilidad GPU         | NVIDIA + AMD + Intel              | Solo NVIDIA                              |
| Overhead del entorno     | GC de Python, llamadas ctypes     | Cero (bindings directos en Rust)         |
| API cuántica             | Dinámica, Pythónica               | Const-generic estática                   |
| Estado de producción     | Abandonado (~2019)                | RC activo (2024)                         |
| Benchmark típico (n=25)  | ~5× CPU Python                    | ~20–40× CPU Rust (estimado)              |

### 7.3 Por qué CUDA generalmente supera a OpenCL para statevector

La simulación de vectores de estado implica operaciones que se benefician especialmente de las características específicas de CUDA:

1. **Memoria compartida y texturas**: Los kernels de simulación acceden al vector de estado con patrones de acceso no contiguos que se benefician de la jerarquía de cache de CUDA. OpenCL tiene equivalentes pero los compiladores de NVIDIA optimizan más agresivamente para sus propias GPUs cuando el código es CUDA nativo.

2. **`cuStateVec`**: La biblioteca cuantum de NVIDIA incluye `cuStateVec`, una biblioteca de CUDA altamente optimizada para la simulación de vectores de estado que explota instrucciones warp-level y técnicas de fusión de operadores. No existe un equivalente OpenCL oficial con el mismo nivel de optimización.

3. **Latencia de operaciones single-qubit**: Para puertas que afectan a un solo qubit, el kernel debe modificar `2^(n-1)` pares de amplitudes. Con CUDA se puede lanzar exactamente `2^(n-1)` hilos, cada uno operando sobre un par. OpenCL permite lo mismo en teoría, pero la sobrecarga del compilador JIT (Just-In-Time) de OpenCL añade latencia en la primera ejecución.

4. **Soporte del fabricante**: NVIDIA invierte activamente en optimizar CUDA para HPC cuántico (proyectos cuQuantum, CUDA-Q). El ecosistema OpenCL para computación cuántica no tiene el mismo nivel de soporte industrial.

### 7.4 El valor de quantrs2 en el panorama actual

A pesar de sus inestabilidades, quantrs2 ocupa un nicho genuino: es el único framework de simulación cuántica en Rust con un path demostrado hacia la aceleración GPU. Las alternativas existentes son:

- **Frameworks Python con CUDA** (cuQuantum, Qiskit Aer GPU): maduros y potentes, pero con el overhead del intérprete Python y sin las garantías de tipos que ofrece Rust.
- **Frameworks Rust CPU** (quantr, q1tsim): estables y ergonómicos, pero limitados a simulación en CPU.
- **Frameworks C++/CUDA** (cuStateVec): máximo rendimiento, pero API de muy bajo nivel.

quantrs2 apunta a llenar el espacio entre Rust ergonómico y aceleración GPU real, una combinación que ningún otro framework ofrece actualmente.

---

## 8. Referencias

1. **Nielsen, M. A. y Chuang, I. L.** (2010). *Quantum Computation and Quantum Information* (10th Anniversary Edition). Cambridge University Press. ISBN 978-1-107-00217-3. — Referencia canónica para fundamentos de algoritmos cuánticos, incluyendo los análisis de Grover (§6.1) y Shor (§5.3).

2. **Grover, L. K.** (1996). A Fast Quantum Mechanical Algorithm for Database Search. *Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC)*, pp. 212–219. ACM. DOI: 10.1145/237814.237866. — Paper original del algoritmo de búsqueda cuántica.

3. **Shor, P. W.** (1994). Algorithms for Quantum Computation: Discrete Logarithms and Factoring. *Proceedings of the 35th Annual Symposium on Foundations of Computer Science (FOCS)*, pp. 124–134. IEEE. DOI: 10.1109/SFCS.1994.365700. — Paper fundacional del algoritmo de factorización cuántica.

4. **Bayraktar, H., Charara, A., Clark, D., Cohen, S., Costa, T., Fang, Y.-L. L., Gao, J., Guan, J., Gunnels, J., Haug, T., Hasan, A., Hoefler, T., Huang, E., Kürekçi, S., Liu, D., Lyakh, D., Morvan, A., Mulligan, P., Paulson, E., Rossi, F., Schiefer, E., Sellers, J., Stanwyck, S., Steiger, D., Tomesh, T., Venturelli, D., Webber, M. y de Jong, W.** (2023). cuQuantum SDK: A High-Performance Library for Accelerating Quantum Science. *2023 IEEE International Conference on Quantum Computing and Engineering (QCE)*, pp. 1050–1061. IEEE. DOI: 10.1109/QCE57702.2023.00119. — Descripción de cuStateVec y cuTensorNet, las bibliotecas CUDA de NVIDIA para simulación cuántica.

5. **Kelly, J.** (2018). Preview of Bristlecone, Google's New Quantum Processor. *Google AI Blog*. Disponible en: https://ai.googleblog.com/2018/03/a-preview-of-bristlecone-googles-new.html. — Contexto sobre la escala de simuladores necesarios para verificar hardware cuántico real.

6. **Häner, T. y Steiger, D. S.** (2017). 0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit. *Proceedings of SC17: International Conference for High Performance Computing, Networking, Storage and Analysis*, Article 33. ACM. DOI: 10.1145/3126908.3126947. — Análisis de costes de memoria y comunicación para simulación distribuida de vectores de estado.

7. **Suzuki, Y., Kawase, Y., Masumura, Y., Hiraga, Y., Nakadai, M., Chen, J., Nakanishi, K. M., Mitarai, K., Imai, R., Tamiya, S., Yamamoto, T., Yan, T., Aspuru-Guzik, A., Fujii, K. y Mitarai, K.** (2021). Qulacs: a fast and versatile quantum circuit simulator for research purpose. *Quantum*, 5, 559. DOI: 10.22331/q-2021-10-06-559. — Benchmark de referencia para simuladores de alto rendimiento en CPU y GPU; contexto comparativo con quantrs2.

8. **Shende, V. V., Markov, I. L. y Bullock, S. S.** (2006). Synthesis of Quantum Logic Circuits. *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 25(6), pp. 1000–1010. IEEE. DOI: 10.1109/TCAD.2005.855930. — Fundamentos de descomposición de puertas multicontroladas en compuertas primitivas; base teórica de las escaleras de Toffoli usadas en el proyecto.

9. **Barenco, A., Bennett, C. H., Cleve, R., DiVincenzo, D. P., Margolus, N., Shor, P., Sleator, T., Smolin, J. A. y Weinfurter, H.** (1995). Elementary gates for quantum computation. *Physical Review A*, 52(5), pp. 3457–3467. APS. DOI: 10.1103/PhysRevA.52.3457. — Artículo seminal sobre la descomposición de puertas cuánticas arbitrarias en puertas elementales, incluyendo la técnica de ancilla para puertas multicontroladas.

10. **Brandhofer, S., Devitt, S. y Polian, I.** (2021). Special-Purpose Quantum Circuit Synthesis with Respect to IBM Quantum Experience. *arXiv preprint*, arXiv:2104.12311 [quant-ph]. https://arxiv.org/abs/2104.12311. — Análisis moderno de la síntesis de circuitos cuánticos con backends reales, relevante para entender las restricciones de puertas disponibles en frameworks como quantrs2.

---

*Fin de la lección. Los ejemplos de código Rust son extractos reales del repositorio del proyecto. Para ejecutar los benchmarks: `cargo run --bin quantrs2-grover -- --n 8 --target 42` y `cargo run --bin quantrs2-shor -- --N 15`.*
