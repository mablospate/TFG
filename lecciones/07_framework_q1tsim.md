# q1tsim — Simulador Cuántico en Rust (Abandonado)

> **Lección 7 del TFG: Benchmarking de Simuladores Cuánticos**
>
> Esta lección cubre el crate `q1tsim` (versión 0.5.0), su API, y la
> implementación concreta de los algoritmos de Grover y Shor usada en el
> pipeline de benchmarking del proyecto. Se incluye código Rust real extraído
> del repositorio y se explica línea a línea.

---

## 1. Contexto e Historia

### 1.1 El crate q1tsim

`q1tsim` es una biblioteca de simulación cuántica escrita íntegramente en Rust,
publicada en [crates.io](https://crates.io/crates/q1tsim) por la empresa Q1t BV.
La versión utilizada en este proyecto es la **0.5.0**, cuya última actualización
data de 2019. El crate nació con un objetivo modesto pero bien definido: ser un
simulador **fácil de usar y eficiente** para el desarrollo y prueba de algoritmos
cuánticos, sin pretender competir con los grandes frameworks industriales.

El propio `lib.rs` del vendored source lo declara:

```
//! q1tsim is a simulator library for a quantum computer, written in Rust.
//! Its goal is to be an easy to use, efficient simulator for the development
//! and testing of quantum algorithms.
```

Las características documentadas en la versión 0.5.0 son:

- Implementación y simulación de circuitos cuánticos mediante una API orientada
  a métodos en un `struct Circuit`.
- Soporte para creación de puertas arbitrarias mediante el trait `Gate`.
- Las puertas más comunes incluidas de serie: H, X, Y, Z, S, T, V, RX, RY, RZ,
  U1, U2, U3, CX, CY, CZ, CCX, CCZ, Swap, CU1 y variantes.
- Medición en bases X, Y y Z.
- Medición no destructiva (`peek`) que no colapsa el estado cuántico.
- Histogramas de resultados de medición sobre múltiples ejecuciones.
- Operaciones condicionales sobre bits clásicos.
- Exportación a OpenQASM, cQASM y LaTeX.
- Simulación eficiente de circuitos estabilizadores mediante el formalismo de
  Gottesman-Knill.

### 1.2 Estado del mantenimiento

El crate lleva **sin actualizaciones desde 2019**. En el contexto de Rust, donde
la edición 2021 trajo cambios sintácticos y el ecosistema de dependencias
(especialmente `ndarray`, `rand`) ha evolucionado con breaking changes, esto
genera problemas de compatibilidad. Para aislar el proyecto de estos problemas,
el repositorio utiliza un directorio `vendor/` con el código fuente del crate
fijado a la versión exacta 0.5.0, gestionado por Cargo's vendoring:

```
rust/q1tsim/
├── Cargo.toml          ← crate wrapper con los binarios
├── build.rs            ← detecta la versión para inyectarla en env!()
├── src/
│   ├── lib.rs          ← pub mod grover; pub mod shor;
│   ├── grover.rs       ← implementación del algoritmo de Grover
│   ├── bin/grover.rs   ← punto de entrada del binario
│   ├── shor/
│   │   ├── mod.rs      ← orquestador de Shor
│   │   ├── classical.rs← aritmética clásica (mod_pow, fracciones continuas)
│   │   ├── qft.rs      ← QFT e IQFT sobre q1tsim
│   │   └── permutation.rs ← red de transposiciones controladas
│   └── bin/shor.rs     ← punto de entrada del binario
└── vendor/q1tsim/      ← código fuente fijado del crate original
```

### 1.3 Por qué incluirlo en el benchmark

A pesar de su estado abandonado, q1tsim tiene un papel legítimo en el
benchmarking:

1. **Baseline histórica**: representa el estado del arte de los simuladores
   cuánticos en Rust en 2019, antes de la aparición de Qiskit Aer, PennyLane o
   CUDA-Q.

2. **Representatividad de época**: permite cuantificar cuánto ha mejorado el
   ecosistema. Comparar su rendimiento con Qiskit o Qibo en 2025 revela la
   distancia tecnológica recorrida en seis años.

3. **Simulador statevector puro, CPU-only, sin GPU**: no tiene ninguna
   optimización especial. Es un caso de referencia limpio, sin transpilación
   avanzada, sin fusión de puertas, sin paralelismo explícito. El tiempo que
   mide es el tiempo mínimo de un simulador sencillo correcto.

4. **Resultados verificables**: el crate pasa todos sus tests unitarios en el
   proyecto con `cargo test`. Sus resultados son estadísticamente correctos; lo
   que es lento es la ejecución, no la física.

### 1.4 El modelo de simulación: statevector puro

q1tsim implementa exclusivamente simulación por **vector de estado** (también
llamado statevector). El estado de un sistema de `n` qubits se representa como
un vector complejo de `2^n` amplitudes. Cada puerta es una multiplicación
matricial sobre ese vector.

La memoria requerida escala exponencialmente: `2^n` números complejos de 64
bits (dos `f64` por número = 16 bytes cada uno). Para `n = 20` qubits, el
vector ocupa 16 MB; para `n = 30` qubits, 16 GB. En la práctica, el límite
práctico de q1tsim en un portátil con 16 GB de RAM es aproximadamente **25-27
qubits**.

El crate también implementa el **formalismo de estabilizadores** (Gottesman-
Knill) para circuitos compuestos solo de puertas del grupo de Clifford (H, S,
CNOT, Pauli), que permite simular hasta cientos de qubits en tiempo polinomial.
Sin embargo, Grover y Shor requieren puertas fuera del grupo de Clifford (la
puerta T en la descomposición de Toffoli, o las rotaciones de fase de la QFT),
por lo que ambos usan el backend de statevector.

---

## 2. API de q1tsim

### 2.1 Creación de un circuito

La unidad central de q1tsim es `q1tsim::circuit::Circuit`. Se crea con:

```rust
use q1tsim::circuit::Circuit;

// nr_qbits: número de qubits cuánticos
// nr_cbits: número de bits clásicos para almacenar mediciones
let mut circuit = Circuit::new(nr_qbits, nr_cbits);
```

La distinción entre qubits y bits clásicos es importante: el registro clásico
solo recibe resultados de medición. No tiene efecto sobre la simulación hasta
que se usa en una puerta condicional.

### 2.2 Puertas de un qubit (métodos directos)

El método más ergonómico es usar los métodos integrados en `Circuit`:

```rust
circuit.h(qubit)?;          // Puerta Hadamard
circuit.x(qubit)?;          // Pauli-X (NOT cuántico)
circuit.y(qubit)?;          // Pauli-Y
circuit.z(qubit)?;          // Pauli-Z
circuit.s(qubit)?;          // Phase gate S (sqrt(Z))
circuit.sdg(qubit)?;        // S-dagger
circuit.t(qubit)?;          // T gate (sqrt(S) = pi/8 gate)
circuit.tdg(qubit)?;        // T-dagger
circuit.rx(theta, qubit)?;  // Rotación alrededor del eje X
circuit.ry(theta, qubit)?;  // Rotación alrededor del eje Y
circuit.rz(lambda, qubit)?; // Rotación alrededor del eje Z
circuit.u1(lambda, qubit)?; // Phase rotation U1(λ) = diag(1, e^{iλ})
circuit.u2(phi, lambda, qubit)?; // Puerta U2
circuit.u3(theta, phi, lambda, qubit)?; // Puerta universal U3
```

Todos devuelven `crate::error::Result<()>`. El operador `?` propaga el error si
el índice de qubit está fuera de rango.

### 2.3 Puertas de dos qubits (métodos directos)

```rust
circuit.cx(control, target)?;  // CNOT: Controlled-X
```

`CX` es el único gate de dos qubits con método directo. El resto requieren
`add_gate`.

### 2.4 Puertas adicionales via `add_gate`

Para puertas que no tienen método directo, se usa `add_gate<G>` con una
instancia del struct correspondiente:

```rust
use q1tsim::gates::{CCX, CCZ, CZ, CU1, Swap};

// Toffoli (CCX): tres qubits, control0, control1, target
circuit.add_gate(CCX::new(), &[ctrl0, ctrl1, target])?;

// CCZ: Controlled-Controlled-Z
circuit.add_gate(CCZ::new(), &[ctrl0, ctrl1, target])?;

// CZ: Controlled-Z
circuit.add_gate(CZ::new(), &[control, target])?;

// CU1: Controlled phase rotation con ángulo theta
circuit.add_gate(CU1::new(theta), &[control, target])?;

// Swap: intercambia dos qubits
circuit.add_gate(Swap::new(), &[qubit_a, qubit_b])?;
```

La firma de `add_gate` es genérica sobre cualquier tipo que implemente el trait
`CircuitGate` (que a su vez requiere `Gate`):

```rust
pub fn add_gate<G: 'static>(&mut self, gate: G, bits: &[usize])
    -> crate::error::Result<()>
where G: CircuitGate
```

### 2.5 Medición

```rust
// Medir qubit i en bit clásico j (base Z por defecto)
circuit.measure(qubit, cbit)?;

// Medir en base X o Y
circuit.measure_x(qubit, cbit)?;
circuit.measure_y(qubit, cbit)?;
circuit.measure_z(qubit, cbit)?; // equivalente a measure()

// Medir sin colapsar el estado (no físico, útil para debugging)
circuit.peek_basis(qubit, cbit, Basis::Z)?;
```

### 2.6 Ejecución

```rust
// Ejecutar el circuito con nr_shots disparos
circuit.execute(nr_shots)?;
```

`execute` realiza `nr_shots` ejecuciones del circuito completo, reiniciando el
estado cuántico a `|0...0>` antes de cada shot. Internamente, detecta si el
circuito es estabilizador (todas las puertas son Clifford) y usa el backend
apropiado. Para Grover y Shor, usa siempre el backend de statevector.

### 2.7 Histograma de resultados

```rust
// Devuelve HashMap<String, usize>: clave es el bitstring, valor es el conteo
let dist: HashMap<String, usize> = circuit.histogram_string()?;
```

El método `histogram_string()` agrega los resultados de los `nr_shots` disparos
en un histograma. Las claves son bitstrings binarios con `nr_cbits` caracteres,
formateados como `format!("{:0width$b}", key, width = nr_cbits)`. El **bit más
significativo aparece a la izquierda** (notación big-endian en la cadena), pero
el bit clásico 0 corresponde al qubit 0. Esto tiene implicaciones para la
decodificación de la QFT en Shor (ver sección 5).

### 2.8 Resumen del flujo completo

```rust
use q1tsim::circuit::Circuit;
use q1tsim::gates::{CCX, CZ};

fn ejemplo_minimo() -> q1tsim::error::Result<()> {
    // 1. Crear circuito: 3 qubits cuánticos, 3 bits clásicos
    let mut c = Circuit::new(3, 3);

    // 2. Construir el circuito (puertas)
    c.h(0)?;
    c.cx(0, 1)?;
    c.cx(0, 2)?;

    // 3. Medir todos los qubits
    c.measure(0, 0)?;
    c.measure(1, 1)?;
    c.measure(2, 2)?;

    // 4. Ejecutar con 1024 shots
    c.execute(1024)?;

    // 5. Obtener distribución
    let dist = c.histogram_string()?;
    // dist contiene aprox. {"000": ~512, "111": ~512}

    Ok(())
}
```

---

## 3. Integración con el Pipeline

### 3.1 Estructura de binarios

El `Cargo.toml` del crate wrapper define dos binarios:

```toml
[package]
name = "q1tsim-bench"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "q1tsim-grover"
path = "src/bin/grover.rs"

[[bin]]
name = "q1tsim-shor"
path = "src/bin/shor.rs"
```

Cada binario es un delegador mínimo al módulo correspondiente:

```rust
// src/bin/grover.rs
fn main() {
    q1tsim_bench::grover::run();
}

// src/bin/shor.rs
fn main() {
    q1tsim_bench::shor::run();
}
```

La lógica real reside en `src/grover.rs` y `src/shor/mod.rs`. Esta separación
permite importar las funciones como biblioteca en tests unitarios, sin necesidad
de levantar un proceso separado.

### 3.2 CLI con clap

El parser de argumentos usa el crate `clap` con el macro derive:

**Grover:**

```rust
#[derive(Parser, Debug)]
#[command(name = "grover", about = "q1tsim Grover benchmark")]
pub struct Args {
    /// Número de qubits de búsqueda.
    #[arg(long)]
    pub n: usize,

    /// Estado objetivo (entero en [0, 2^n)).
    #[arg(long)]
    pub target: u64,

    /// Número de shots.
    #[arg(long, default_value_t = 1024)]
    pub shots: usize,

    /// Override del número de iteraciones (por defecto: floor(pi/4 * sqrt(2^n))).
    #[arg(long)]
    pub iterations: Option<usize>,
}
```

Ejemplo de invocación:

```bash
./q1tsim-grover --n 4 --target 11 --shots 2048
./q1tsim-grover --n 3 --target 5 --shots 1024 --iterations 2
```

**Shor:**

```rust
#[derive(Parser, Debug)]
#[command(name = "shor", about = "q1tsim Shor benchmark")]
pub struct Args {
    #[arg(long = "N")]
    pub n: u64,           // número a factorizar

    #[arg(long, default_value_t = 10)]
    pub shots: usize,     // shots por intento de order-finding

    #[arg(long, default_value_t = 3)]
    pub tries: u32,       // máximo de intentos con base aleatoria

    #[arg(long)]
    pub seed: Option<u64>,// semilla opcional para reproducibilidad
}
```

Ejemplo de invocación:

```bash
./q1tsim-shor --N 15 --shots 30 --tries 5
./q1tsim-shor --N 21 --tries 10 --seed 42
```

### 3.3 Salida JSON por stdout

Ambos algoritmos emiten un objeto JSON por `stdout` al terminar, usando
`serde_json`. El runner Python (que llama a estos binarios) lee exactamente este
stdout. Los mensajes de progreso van a `stderr` y no interfieren.

**Grover Output:**

```rust
#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,           // "q1tsim"
    pub framework_version: &'static str,   // "0.1.0" (del Cargo.toml wrapper)
    pub algorithm: &'static str,           // "grover"
    pub n: usize,
    pub target: u64,
    pub shots: usize,
    pub iterations: usize,
    pub found: u64,                        // estado más frecuente
    pub time_ms: f64,
    pub mem_mb: f64,
    pub distribution: HashMap<String, usize>,
}
```

Ejemplo de salida:

```json
{
  "framework": "q1tsim",
  "framework_version": "0.1.0",
  "algorithm": "grover",
  "n": 3,
  "target": 5,
  "shots": 1024,
  "iterations": 2,
  "found": 5,
  "time_ms": 142.7,
  "mem_mb": 0.0,
  "distribution": {"101": 918, "000": 12, "001": 11, "010": 14, "011": 9, "100": 17, "110": 13, "111": 30}
}
```

**Shor Output:**

```rust
#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,
    pub framework_version: &'static str,
    pub algorithm: &'static str,
    #[serde(rename = "N")]
    pub n: u64,
    pub factor: u64,
    pub time_ms: f64,
    pub mem_mb: f64,
}
```

Ejemplo:

```json
{
  "framework": "q1tsim",
  "framework_version": "0.1.0",
  "algorithm": "shor",
  "N": 15,
  "factor": 3,
  "time_ms": 1843.5,
  "mem_mb": 0.0
}
```

### 3.4 Medición de tiempo

El tiempo se mide con `std::time::Instant`, que en la plataforma Linux/macOS
llama al reloj de alta resolución del sistema operativo. La medición **incluye**
tanto la construcción del circuito como su ejecución:

```rust
let start = Instant::now();
let mut circuit = grover_circuit(args.n, args.target, iterations)?;
circuit.execute(args.shots)?;
let dist = circuit.histogram_string()?;
let elapsed = start.elapsed();
// tiempo en milisegundos:
let time_ms = elapsed.as_secs_f64() * 1000.0;
```

### 3.5 Medición de memoria

La función `peak_rss_mb()` lee el pico de memoria residente del proceso desde
`/proc/self/status` en Linux:

```rust
fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmHWM:") {
                if let Some(kb) = line.split_whitespace().nth(1)
                    .and_then(|s| s.parse::<u64>().ok()) {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }
    0.0  // macOS: devuelve 0 (no disponible en /proc)
}
```

`VmHWM` es el "High Water Mark" del RSS (Resident Set Size), es decir, el
máximo de memoria física usada en cualquier momento de la vida del proceso. En
macOS, donde no existe `/proc`, la función devuelve `0.0`.

---

## 4. Implementación de Grover en q1tsim

### 4.1 El algoritmo de Grover: recordatorio

El algoritmo de Grover busca un elemento marcado en un espacio de `N = 2^n`
estados aplicando repetidamente dos operaciones:

1. **Oráculo**: invierte la fase del estado objetivo `|t>`:
   `U_f |x> = -|x>` si `x == t`, `|x>` en caso contrario.

2. **Difusor** (operador de Grover): reflexión alrededor de la superposición
   uniforme `|s> = H^{⊗n}|0>`:
   `D = 2|s><s| - I`

Tras `k ≈ (π/4) √N` iteraciones, la probabilidad de medir el estado `|t>` es
cercana a 1.

### 4.2 El problema de las puertas multi-control en q1tsim

El oráculo y el difusor requieren una **puerta Z multi-controlada** (MCZ o
CnZ): una puerta que aplica Z al qubit objetivo solo cuando todos los `n`
qubits de control están en `|1>`. En la base computacional, esto se reduce a
negar la fase solo del estado `|11...1>`.

q1tsim proporciona de serie:
- `CZ`: Controlled-Z (2 qubits total: 1 control + 1 objetivo)
- `CCZ`: Doubly-Controlled-Z (3 qubits total: 2 controles + 1 objetivo)

Para `n >= 4` controles no existe una puerta nativa. La solución estándar es
la **descomposición con ancillas de Toffoli** (ladder de Toffoli):

### 4.3 La escalera de Toffoli (Toffoli ladder)

Para implementar MCX de `k` controles sobre un objetivo `t`, con `k-2` qubits
ancilla en `|0>`, el circuito es:

```
CCX(ctrl[0], ctrl[1], anc[0])
CCX(ctrl[2], anc[0], anc[1])
CCX(ctrl[3], anc[1], anc[2])
...
CCX(ctrl[k-1], anc[k-3], tgt)    ← puerta central
...inversa en orden contrario:
CCX(ctrl[3], anc[1], anc[2])
CCX(ctrl[2], anc[0], anc[1])
CCX(ctrl[0], ctrl[1], anc[0])    ← descomputa ancillas a |0>
```

La función `apply_mcx` implementa exactamente esto:

```rust
pub fn apply_mcx(
    c: &mut Circuit,
    ctrls: &[usize],     // índices de los qubits de control
    tgt: usize,          // índice del qubit objetivo
    ancillas: &[usize],  // qubits ancilla en |0>, longitud = k-2 para k>2 controles
) -> QResult<()> {
    let k = ctrls.len();
    match k {
        // Sin controles: X incondicionalmente
        0 => c.x(tgt)?,
        // Un control: CNOT nativo
        1 => c.cx(ctrls[0], tgt)?,
        // Dos controles: Toffoli (CCX) nativo
        2 => c.add_gate(CCX::new(), &[ctrls[0], ctrls[1], tgt])?,
        // Tres o más controles: escalera de Toffoli
        _ => {
            // Fase de computación: propagar control hacia el objetivo
            c.add_gate(CCX::new(), &[ctrls[0], ctrls[1], ancillas[0]])?;
            for i in 1..(k - 2) {
                c.add_gate(CCX::new(), &[ctrls[i + 1], ancillas[i - 1], ancillas[i]])?;
            }
            // Puerta central que aplica X al objetivo
            c.add_gate(CCX::new(), &[ctrls[k - 1], ancillas[k - 3], tgt])?;
            // Fase de descomputación (idéntica a la computación en orden inverso)
            for i in (1..(k - 2)).rev() {
                c.add_gate(CCX::new(), &[ctrls[i + 1], ancillas[i - 1], ancillas[i]])?;
            }
            c.add_gate(CCX::new(), &[ctrls[0], ctrls[1], ancillas[0]])?;
        }
    }
    Ok(())
}
```

La simetría del circuito (computación + descomputación) es esencial: garantiza
que las ancillas vuelvan a `|0>` al finalizar, de modo que no entren en
superposición ni contaminan las iteraciones siguientes.

### 4.4 La puerta MCZ (multi-controlled Z)

MCZ se implementa sobre MCX mediante la identidad `Z = H X H`:

```rust
pub fn apply_mcz(c: &mut Circuit, qubits: &[usize], ancillas: &[usize]) -> QResult<()> {
    let n = qubits.len();
    match n {
        0 => {}
        // Un qubit: Z nativo
        1 => c.z(qubits[0])?,
        // Dos qubits: CZ nativo de q1tsim
        2 => c.add_gate(CZ::new(), &[qubits[0], qubits[1]])?,
        // Tres qubits: CCZ nativo de q1tsim
        3 => c.add_gate(CCZ::new(), &[qubits[0], qubits[1], qubits[2]])?,
        // Cuatro o más: MCX con H en el objetivo para convertir X en Z
        _ => {
            let target = qubits[n - 1];
            let ctrls: Vec<usize> = qubits[..n - 1].to_vec();
            c.h(target)?;            // convierte la base: |0>->(|0>+|1>)/√2
            apply_mcx(c, &ctrls, target, ancillas)?;  // aplica X condicionalmente
            c.h(target)?;            // vuelve a la base Z
        }
    }
    Ok(())
}
```

El truco `H MCX H = MCZ` se basa en que `H X H = Z`, y que la Hadamard es su
propia inversa. Al poner `H` antes y después del objetivo, transformamos la
condición "flipa el bit objetivo" en "invierte la fase del objetivo".

### 4.5 El oráculo de Grover

El oráculo para el objetivo `t` funciona así:
- Los qubits donde el bit correspondiente de `t` es `0` se niegan con X (para
  que el estado `|t>` sea el único donde todos esos qubits están en `|1>`).
- Se aplica MCZ a todos los qubits.
- Se deshacen las negaciones (segunda ronda de X).

```rust
pub fn build_oracle(
    c: &mut Circuit,
    n: usize,         // número de qubits de búsqueda
    target: u64,      // estado objetivo como entero
    ancillas: &[usize],
) -> QResult<()> {
    let qubits: Vec<usize> = (0..n).collect();

    // Primera ronda de X: negar los qubits que son 0 en el objetivo
    // Objetivo: (target >> i) & 1 == 0 -> el bit i debe ser 1 para activar MCZ
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(i)?;   // invierte el qubit para que |0> pase a |1>
        }
    }

    // MCZ: marca (fase -1) el estado donde todos los qubits son |1>
    // Tras la ronda de X, ese estado es precisamente |target>
    apply_mcz(c, &qubits, ancillas)?;

    // Segunda ronda de X: deshacer la inversión (mismo patrón)
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(i)?;
        }
    }
    Ok(())
}
```

**Ejemplo para `target = 5` (binario `101`) con `n = 3`:**
- `i=0`: bit 0 de 5 es 1, no aplicar X.
- `i=1`: bit 1 de 5 es 0, aplicar X al qubit 1.
- `i=2`: bit 2 de 5 es 1, no aplicar X.
- Después de las X, el estado `|101>` se convierte en `|111>`.
- MCZ sobre `{0,1,2}` aplica fase -1 a `|111>`.
- Las X finales restauran `|111>` a `|101>`.
- Resultado neto: solo `|101>` tiene fase -1.

### 4.6 El difusor de Grover

El difusor `D = 2|s><s| - I` se implementa como:
`D = H^n (2|0><0| - I) H^n`

Y `(2|0><0| - I)` se implementa como X^n MCZ X^n (fase -1 al estado `|0...0>`
= negarlo a `|1...1>`, aplicar MCZ, volver a negar):

```rust
pub fn build_diffuser(
    c: &mut Circuit,
    n: usize,
    ancillas: &[usize],
) -> QResult<()> {
    let qubits: Vec<usize> = (0..n).collect();

    // Paso 1: H en todos los qubits -> base de superposición uniforme
    for i in 0..n { c.h(i)?; }

    // Paso 2: X en todos los qubits -> convierte |0...0> en |1...1>
    for i in 0..n { c.x(i)?; }

    // Paso 3: MCZ -> fase -1 al estado |1...1> (que es el antiguo |0...0>)
    apply_mcz(c, &qubits, ancillas)?;

    // Paso 4: X en todos los qubits -> deshace la conversión
    for i in 0..n { c.x(i)?; }

    // Paso 5: H en todos los qubits -> vuelve a la base computacional
    for i in 0..n { c.h(i)?; }

    Ok(())
}
```

### 4.7 El circuito completo de Grover

```rust
pub fn grover_circuit(n: usize, target: u64, iterations: usize) -> QResult<Circuit> {
    // Calcular cuántas ancillas se necesitan
    let n_anc = ancilla_count(n);  // = max(0, n-2)
    let total = n + n_anc;         // qubits totales del circuito

    // nr_qbits = total, nr_cbits = n (solo medimos los qubits de búsqueda)
    let mut circuit = Circuit::new(total, n);
    let ancillas: Vec<usize> = (n..n + n_anc).collect();

    // Inicialización: superposición uniforme sobre los n qubits de búsqueda
    for i in 0..n {
        circuit.h(i)?;
    }

    // Iteraciones de Grover
    for _ in 0..iterations {
        build_oracle(&mut circuit, n, target, &ancillas)?;
        build_diffuser(&mut circuit, n, &ancillas)?;
    }

    // Medir solo los n qubits de búsqueda (las ancillas quedan en |0> y no se miden)
    for q in 0..n {
        circuit.measure(q, q)?;
    }
    Ok(circuit)
}
```

**Cálculo del número de iteraciones:**

```rust
let iterations = args
    .iterations
    .unwrap_or_else(|| {
        // Fórmula exacta: floor(π/4 * √(2^n))
        ((PI / 4.0) * (2f64.powi(args.n as i32)).sqrt()).floor() as usize
    })
    .max(1);  // al menos 1 iteración
```

Para `n = 3`: `floor(π/4 * √8) ≈ floor(2.22) = 2` iteraciones.
Para `n = 4`: `floor(π/4 * √16) = floor(3.14) = 3` iteraciones.
Para `n = 10`: `floor(π/4 * √1024) ≈ floor(25.1) = 25` iteraciones.

### 4.8 El `run_with_args`: orquestación y extracción de resultados

```rust
pub fn run_with_args(args: &Args) -> Result<Output, Box<dyn Error>> {
    // Validar que target está en rango
    let max_target = 1u64 << args.n;
    if args.target >= max_target {
        return Err(format!("target {} out of range", args.target).into());
    }

    let start = Instant::now();

    // Construir y ejecutar el circuito
    let mut circuit = grover_circuit(args.n, args.target, iterations)?;
    circuit.execute(args.shots)?;

    // Obtener el histograma como HashMap<String, usize>
    let dist = circuit.histogram_string()?;
    let elapsed = start.elapsed();

    // Encontrar el estado más frecuente (el "found")
    let (best_bs, _) = dist
        .iter()
        .max_by_key(|(_, c)| *c)
        .ok_or("empty histogram")?;

    // Convertir el bitstring a entero (desde_str_radix con base 2)
    let found = u64::from_str_radix(best_bs, 2)?;

    Ok(Output { found, time_ms: elapsed.as_secs_f64() * 1000.0, distribution: dist, ... })
}
```

---

## 5. Implementación de Shor en q1tsim

### 5.1 Arquitectura general

La implementación de Shor en q1tsim sigue el mismo esquema conceptual que la
versión Python de CUDA-Q: **estimación cuántica de fase (QPE)** para encontrar
el orden `r` tal que `a^r ≡ 1 (mod N)`, seguida de post-procesamiento clásico
para extraer factores.

El diseño se distribuye en cuatro módulos:

| Módulo | Responsabilidad |
|--------|-----------------|
| `shor/mod.rs` | Orquestación, CLI, QPE, loop de factorización |
| `shor/qft.rs` | QFT directa e inversa sobre q1tsim |
| `shor/permutation.rs` | Red de permutaciones controladas para U_a |
| `shor/classical.rs` | Aritmética clásica: mod_pow, fracciones continuas, reducción de orden |

### 5.2 Layout de qubits

```
[0 .. m)              : registro de control de fase (m = 2*width bits)
[m .. m + width)      : registro objetivo |y> para la multiplicación modular
[m + width .. total)  : ancillas para la escalera de Toffoli
```

Donde `width = ceil(log2(N))` es el número de bits necesarios para representar
números menores que `N`.

### 5.3 La QFT en q1tsim

La Transformada de Fourier Cuántica sobre `n` qubits se define como:

```
QFT |j> = (1/√N) Σ_k e^{2πi jk/N} |k>
```

Su circuito canónico consiste en Hadamards seguidos de rotaciones de fase
controladas con ángulos `π/2^k`.

La implementación en `shor/qft.rs` usa la puerta `CU1` de q1tsim:

```rust
use q1tsim::gates::{CU1, Swap};

pub fn apply_qft(c: &mut Circuit, qubits: &[usize]) -> QResult<()> {
    let n = qubits.len();
    for i in 0..n {
        // Hadamard en el qubit i
        c.h(qubits[i])?;
        // Rotaciones de fase controladas desde los qubits j > i
        for j in (i + 1)..n {
            let k = (j - i + 1) as i32;
            // CU1 aplica la fase e^{iθ} al qubit objetivo cuando el control es |1>
            // Ángulo: θ = π / 2^{k-1}
            let angle = PI / 2f64.powi(k - 1);
            c.add_gate(CU1::new(angle), &[qubits[j], qubits[i]])?;
        }
    }
    // Intercambios de bits para corregir el orden de los qubits de salida
    for i in 0..n / 2 {
        c.add_gate(Swap::new(), &[qubits[i], qubits[n - 1 - i]])?;
    }
    Ok(())
}
```

La **QFT inversa** (IQFT) es la adjunta de la QFT: se aplican primero los
intercambios de bits, y luego las rotaciones con ángulos negados seguidas de
Hadamards, en orden inverso:

```rust
pub fn apply_inverse_qft(c: &mut Circuit, qubits: &[usize]) -> QResult<()> {
    let n = qubits.len();
    // Paso 1: intercambios de bits (igual que en QFT)
    for i in 0..n / 2 {
        c.add_gate(Swap::new(), &[qubits[i], qubits[n - 1 - i]])?;
    }
    // Paso 2: aplicar en orden inverso H y rotaciones negadas
    for i in (0..n).rev() {
        // Rotaciones con ángulo negativo (conjugadas)
        for j in ((i + 1)..n).rev() {
            let k = (j - i + 1) as i32;
            let angle = -PI / 2f64.powi(k - 1);  // ángulo negado = adjunta
            c.add_gate(CU1::new(angle), &[qubits[j], qubits[i]])?;
        }
        c.h(qubits[i])?;
    }
    Ok(())
}
```

La IQFT es la que se usa en QPE: primero se aplica la exponenciación controlada
(que construye el estado de fase), y luego la IQFT extrae la fase al registro
clásico.

### 5.4 La exponenciación modular controlada

En QPE para Shor, necesitamos implementar `CU_a^{2^k}`, la puerta que aplica
`|y> -> |a^{2^k} * y mod N>` condicionalmente sobre el qubit de control `k`.

La implementación en q1tsim toma un enfoque funcional: calcula la **permutación**
que representa `x -> (a^power * x) mod N` sobre los valores `{0, 1, ..., N-1}`,
y luego implementa esa permutación mediante una **red de transposiciones
controladas**:

```rust
// En shor/mod.rs, bucle principal de QPE:
for i in 0..m {
    // power = 2^{m-1-i}: potencia correspondiente al qubit de control i
    let power: u64 = 1u64 << (m - 1 - i);

    // Calcular la permutación: x -> (a^power * x) mod N
    let perm = build_mod_exp_permutation(a, n_mod, power);

    if perm.is_empty() {
        continue;  // permutación identidad: no hace nada
    }

    // Aplicar la permutación controlada por el qubit i
    controlled_swap_permutation(&mut c, i, &target_qubits, &ancillas, &perm)?;
}
```

La función `build_mod_exp_permutation` es puramente clásica:

```rust
pub fn build_mod_exp_permutation(a: u64, n_mod: u64, power: u64) -> HashMap<u64, u64> {
    // Calcular a^power mod n_mod usando exponenciación modular rápida
    let a_pow = mod_pow(a as u128, power as u128, n_mod as u128) as u64;

    let mut perm = HashMap::new();
    for y in 0..n_mod {
        let tgt = ((a_pow as u128 * y as u128) % n_mod as u128) as u64;
        if y != tgt {
            // Solo almacenar los puntos no fijos
            perm.insert(y, tgt);
        }
    }
    perm
}
```

### 5.5 Descomposición de permutaciones en transposiciones

Una permutación se descompone en ciclos, y cada ciclo en transposiciones. La
función `controlled_swap_permutation` encuentra los ciclos de la permutación y
los implementa mediante transposiciones controladas:

```rust
pub fn controlled_swap_permutation(
    c: &mut Circuit,
    ctrl: usize,             // qubit de control
    target_qubits: &[usize], // registro objetivo
    ancillas: &[usize],
    permutation: &HashMap<u64, u64>,
) -> QResult<()> {
    let mut visited: HashSet<u64> = HashSet::new();

    for start in sorted_keys {
        // Encontrar el ciclo que contiene 'start'
        let mut cycle = Vec::new();
        let mut current = start;
        while !visited.contains(&current) {
            visited.insert(current);
            cycle.push(current);
            current = *permutation.get(&current).unwrap_or(&current);
        }
        if cycle.len() <= 1 { continue; }

        // Descomponer el ciclo (a0, a1, ..., ak) en transposiciones:
        // (a0, a1, ..., ak) = (a0, a1)(a0, a2)...(a0, ak)
        for idx in 1..cycle.len() {
            controlled_transposition(c, ctrl, target_qubits, ancillas,
                                     cycle[0], cycle[idx])?;
        }
    }
    Ok(())
}
```

Cada transposición controlada intercambia los estados `|a>` y `|b>` del registro
objetivo condicionalmente sobre el qubit de control. Si `a` y `b` difieren solo
en un bit, se implementa con un MCX (multi-controlled-X); si difieren en varios
bits, se aplica recursivamente usando un pivote:

```rust
pub fn controlled_transposition(..., a: u64, b: u64) -> QResult<()> {
    let diff_bits = a ^ b;
    if diff_bits == 0 { return Ok(()); }

    let diff_positions: Vec<usize> = (0..n).filter(|&i| (diff_bits >> i) & 1 == 1).collect();

    if diff_positions.len() == 1 {
        // Un solo bit diferente: transposición directa con MCX
        controlled_single_bit_transposition(c, ctrl, target_qubits, ancillas, a, b)?;
    } else {
        // Varios bits diferentes: descomponer vía pivote
        // (a, b) = (a, a') (a', b) (a, a')  donde a' = a XOR (1 << pivot)
        let pivot = diff_positions[0];
        let a_prime = a ^ (1u64 << pivot);
        controlled_transposition(c, ctrl, target_qubits, ancillas, a, a_prime)?;
        controlled_transposition(c, ctrl, target_qubits, ancillas, a_prime, b)?;
        controlled_transposition(c, ctrl, target_qubits, ancillas, a, a_prime)?;
    }
    Ok(())
}
```

### 5.6 El circuito QPE completo

```rust
pub fn order_finding_circuit(a: u64, n_mod: u64, precision: usize) -> QResult<Circuit> {
    // Validar que gcd(a, N) = 1 (requisito del algoritmo)
    if gcd(a, n_mod) > 1 {
        return Err(Error::InternalError(format!("gcd({}, {}) > 1", a, n_mod)));
    }

    let width = ((n_mod as f64).log2().ceil() as usize).max(1);
    let m = precision;           // bits de precisión del registro de control
    let n_anc = width.saturating_sub(1);
    let total = m + width + n_anc;

    let mut c = Circuit::new(total, m);
    let target_qubits: Vec<usize> = (m..m + width).collect();
    let ancillas: Vec<usize> = (m + width..total).collect();

    // 1. Hadamard sobre el registro de control: superposición uniforme
    for i in 0..m {
        c.h(i)?;
    }

    // 2. Inicializar el registro objetivo en |1> (estado "1 mod N")
    c.x(target_qubits[0])?;

    // 3. Para cada qubit de control i, aplicar U_a^{2^{m-1-i}} controlado por i
    for i in 0..m {
        let power: u64 = 1u64 << (m - 1 - i);
        let perm = build_mod_exp_permutation(a, n_mod, power);
        if perm.is_empty() { continue; }
        controlled_swap_permutation(&mut c, i, &target_qubits, &ancillas, &perm)?;
    }

    // 4. QFT inversa sobre el registro de control
    let ctrl_qubits: Vec<usize> = (0..m).collect();
    apply_inverse_qft(&mut c, &ctrl_qubits)?;

    // 5. Medir el registro de control
    for i in 0..m {
        c.measure(i, i)?;
    }
    Ok(c)
}
```

### 5.7 Post-procesamiento clásico: fracciones continuas

El histograma del circuito QPE contiene picos en valores `x ≈ j * 2^m / r`
para distintos enteros `j`. Para extraer `r`, se usa el algoritmo de fracciones
continuas: `x / 2^m ≈ j/r`, y el denominador de la fracción reducida es `r`.

La función `get_order_from_dist` implementa esto:

```rust
pub fn get_order_from_dist(
    dist: &HashMap<String, usize>,
    a: u64, n_mod: u64, precision: usize,
) -> u64 {
    let two_to_m: u64 = 1u64 << precision;

    // Ordenar por frecuencia decreciente, probar los 10 picos más altos
    for (bs, _count) in top_10_entries {
        if bs.chars().all(|c| c == '0') { continue; }  // estado |0...0> es ambiguo

        // IMPORTANTE: q1tsim formatea con el bit 0 a la derecha,
        // pero el ctrl[0] es el MSB de la fase. Hay que invertir el bitstring.
        let reversed: String = bs.chars().rev().collect();
        let x = u64::from_str_radix(&reversed, 2)?;

        // Fracciones continuas: aproximar x/2^m por p/q con q <= N-1
        let (_, q) = limit_denominator(x, two_to_m, n_mod.saturating_sub(1).max(1));
        let r = q;

        // Verificar que r es realmente el orden: a^r ≡ 1 (mod N)
        if mod_pow(a as u128, r as u128, n_mod as u128) == 1 {
            // Reducir al orden mínimo (por si r es un múltiplo del orden real)
            return reduce_to_min_order(r, a, n_mod);
        }
    }
    0  // no se encontró orden válido en los 10 picos
}
```

La inversión del bitstring (`bs.chars().rev().collect()`) merece atención: como
se documenta en el comentario del código, `histogram_string()` formatea la clave
con el bit clásico `m-1` a la izquierda y el bit 0 a la derecha. El ctrl[0] es
el qubit 0, que corresponde al bit más significativo del valor de fase (por
cómo se construye el circuito QPE). La inversión alinea la representación.

### 5.8 La exponenciación modular clásica (mod_pow)

```rust
pub fn mod_pow(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
    if modulus == 1 { return 0; }
    let mut result: u128 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;  // cuadrado iterativo
    }
    result
}
```

Esta es la implementación estándar de exponenciación modular en tiempo
`O(log exp)`. Se usa en múltiples lugares: `build_mod_exp_permutation`,
`get_order_from_dist`, y la verificación final del factor en `find_factor`.

### 5.9 El loop de factorización

La función `find_factor` integra todos los componentes:

```rust
pub fn find_factor(args: &Args) -> QResult<u64> {
    let n = args.n;

    // Atajos clásicos rápidos
    if n % 2 == 0 { return Ok(2); }           // números pares
    // Test de potencia perfecta (n = p^k)
    for k in 2..=max_k { ... }

    // Bucle cuántico con reintentos
    for _ in 0..args.tries {
        let a: u64 = rng.gen_range(2..n);

        // Suerte clásica: a y N tienen factor común
        let g = a.gcd(&n);
        if g > 1 { return Ok(g); }

        // QPE para encontrar el orden r de a mod N
        let (r, _dist) = find_order(a, n, None, args.shots);

        // El orden debe ser par
        if r == 0 || r % 2 != 0 { continue; }

        // Intentar extraer factor de gcd(a^{r/2} ± 1, N)
        let half = mod_pow(a as u128, (r / 2) as u128, n as u128) as u64;
        if half == 0 { continue; }

        let d = half.wrapping_sub(1).gcd(&n);
        if d > 1 && d < n { return Ok(d); }

        let d2 = (half + 1).gcd(&n);
        if d2 > 1 && d2 < n { return Ok(d2); }
    }
    Ok(1)  // no se encontró factor no trivial
}
```

### 5.10 Diferencias con la implementación Python

| Aspecto | Python (CUDA-Q) | Rust (q1tsim) |
|---------|-----------------|---------------|
| Backend de simulación | CUDA-Q statevector (GPU opcional) | q1tsim statevector (CPU) |
| Exponenciación modular | Puertas U rotativas sobre el estado | Red de permutaciones/transposiciones |
| QFT | Implementada explícitamente con puertas phase | Misma estructura, con CU1 de q1tsim |
| Fracciones continuas | `Fraction(x, 2**m).limit_denominator(N)` de Python | Implementación propia de Stern-Brocot/Euclides en Rust |
| Post-procesamiento | Separado en módulos Python | En `classical.rs` |
| Semilla RNG | `random.seed()` global | `StdRng::seed_from_u64()` con inyección explícita |

La diferencia más significativa es el enfoque de la exponenciación modular.
La implementación de q1tsim descompone la operación `|y> -> |ay mod N>` en
transposiciones de estados de la base computacional, lo que resulta en circuitos
con un número de puertas que crece con el número de ciclos de la permutación,
no con la precisión de la QFT. Esto es eficiente para N pequeño pero no escala
bien para N grande.

---

## 6. Limitaciones y Consideraciones

### 6.1 El límite práctico de qubits

La memoria para el statevector crece como `16 * 2^n` bytes (16 bytes por
amplitud compleja con dos `f64`):

| n qubits | Memoria statevector | Nota |
|----------|---------------------|------|
| 10 | 16 KB | trivial |
| 15 | 512 KB | trivial |
| 20 | 16 MB | cómodo |
| 25 | 512 MB | factible en 16 GB de RAM |
| 27 | 2 GB | límite práctico en portátil |
| 30 | 16 GB | máquina de escritorio |
| 33 | 128 GB | servidor |

Para Grover con `n` qubits de búsqueda y `n-2` ancillas, el total de qubits
del circuito es `2n - 2`. Para `n = 14`, se necesitan 26 qubits en total, lo
que requiere un statevector de `2^26 * 16 ≈ 1 GB`. Este es el límite superior
razonable para este proyecto.

Para Shor, el conteo de qubits es `m + width + n_anc = 2*width + width + (width-1)
= 4*width - 1`. Para `N = 15` (width = 4), se necesitan 15 qubits; para `N = 21`
(width = 5), 19 qubits; para `N = 35` (width = 6), 23 qubits.

### 6.2 Incompatibilidades con Rust moderno

La versión vendorizada 0.5.0 fue escrita con Rust 2018 edition y versiones
antiguas de `ndarray` y `rand`. Los problemas conocidos encontrados durante el
desarrollo incluyen:

1. **Trait `rand::Rng` vs `rand::RngCore`**: La API de `rand` cambió entre 0.7
   y 0.8. El vendored code usa `rand::Rng` en los genéricos donde `rand::RngCore`
   sería más correcto. Se resuelve manteniendo la dependencia de `rand` fijada
   en el workspace.

2. **`ndarray` API changes**: `ndarray` 0.15+ cambió algunos tipos. El vendored
   code está fijado a la versión compatible.

3. **`u64::from_str_radix` en contexto `?`**: El tipo de error de
   `ParseIntError` debe ser compatible con `Box<dyn Error>` en la cadena de
   propagación. Se resuelve con `.into()`.

4. **Warnings de dead code**: `apply_qft` en `qft.rs` no se llama directamente
   (solo se usa IQFT), lo que genera un warning. Se silencia con un dummy:
   ```rust
   #[allow(dead_code)]
   fn _keep_apply_qft_alive() {
       let _ = apply_qft as fn(&mut Circuit, &[usize]) -> QResult<()>;
   }
   ```

### 6.3 El formato del histograma y el orden de bits

Una particularidad de q1tsim que requiere atención es el formato de
`histogram_string()`. La clave para el estado medido como el entero `k`
con `n` bits clásicos es `format!("{:0width$b}", k, width = n)`. Esto
significa que:

- El bit clásico más significativo (`cbit n-1`) aparece a la **izquierda**
  de la cadena.
- El bit clásico 0 aparece a la **derecha**.

En Grover esto no importa, porque `circuit.measure(q, q)` mapea el qubit `q`
al cbit `q`, y el entero resultante `u64::from_str_radix(bitstring, 2)`
reproduce el entero target correctamente.

En Shor sí importa: el qubit de control `i=0` es el de mayor potencia
(`2^{m-1}`), que corresponde al bit más significativo de la fase. Sin embargo,
`cbit 0` aparece a la **derecha** del bitstring, que es la posición de menor
peso en la interpretación `from_str_radix`. La corrección es invertir el
bitstring antes de parsear, como se ve en `get_order_from_dist`:

```rust
let reversed: String = bs.chars().rev().collect();
let x = u64::from_str_radix(&reversed, 2)?;
```

### 6.4 Validez de los resultados para el benchmark

A pesar del estado abandonado del crate, los resultados de q1tsim son válidos
para la comparación por las siguientes razones:

1. **Correctitud matemática verificada**: La suite de tests unitarios cubre
   casos conocidos (orden de 2 mod 15, búsqueda de targets en n=3 y n=4) y
   pasan consistentemente.

2. **Simulación exacta**: q1tsim no hace aproximaciones. El statevector se
   representa con precisión `f64` estándar (doble precisión IEEE 754), igual
   que todos los demás simuladores del benchmark.

3. **El crate funciona en el commit fijado**: El código se compila sin errores
   con la edición 2021 de Rust y las dependencias del workspace. Los problemas
   de compatibilidad fueron resueltos durante la integración y están encapsulados
   en el directorio `vendor/`.

4. **Lo que el benchmark mide es real**: El tiempo de ejecución refleja
   genuinamente el coste de simular el circuito con este framework. La lentitud
   de q1tsim frente a frameworks modernos es un dato valioso, no un artefacto.

### 6.5 Ausencia de optimizaciones

q1tsim no implementa ninguna de las optimizaciones que sí tienen los frameworks
modernos:

- **Sin fusión de puertas**: cada puerta se aplica como una multiplicación
  matricial independiente.
- **Sin paralelismo**: las operaciones son secuenciales sobre el vector de estado.
- **Sin GPU**: únicamente CPU.
- **Sin transpilación**: no optimiza la secuencia de puertas antes de ejecutar.
- **Sin circuit cutting**: no divide el circuito en partes más pequeñas.

Esto hace que q1tsim sea un **lower bound** del rendimiento posible para un
simulador statevector correcto en Rust, y un punto de referencia excelente para
medir el impacto de cada una de esas optimizaciones en los frameworks modernos.

---

## Resumen

| Característica | q1tsim 0.5.0 |
|----------------|--------------|
| Lenguaje | Rust (edición 2018, compilable con 2021) |
| Modelo de simulación | Statevector puro + Estabilizadores (Clifford) |
| Hardware | CPU exclusivamente |
| Última actualización | 2019 |
| API principal | `Circuit::new(n_q, n_c)` + métodos `h()`, `x()`, `cx()`, `add_gate()`, `execute()`, `histogram_string()` |
| Puertas multi-control | Via escalera de Toffoli con ancillas |
| Exportación | OpenQASM, cQASM, LaTeX |
| Grover | MCZ via H-MCX-H; ancillas descomputadas en cada iteración |
| Shor | Permutaciones modulares controladas + IQFT + fracciones continuas |
| Límite práctico | ~25-27 qubits en 16 GB RAM |
| Rol en el benchmark | Baseline histórica; cuantifica el progreso del ecosistema desde 2019 |
