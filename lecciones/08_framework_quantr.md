# quantr — Simulador Cuántico en Rust (Statevector Moderno)

> **Lección 08 — Computación Cuántica Aplicada**
> Prerrequisitos: álgebra lineal básica, Rust nivel intermedio, lectura previa de la lección sobre q1tsim.

---

## 1. Contexto e Historia

### El ecosistema de simuladores cuánticos en Rust

El ecosistema Rust para simulación cuántica es joven pero activo. A diferencia de Python, donde Qiskit y Cirq tienen miles de contribuidores, el espacio Rust está dominado por una docena de crates pequeños mantenidos por individuos o grupos universitarios. Dos de los más maduros son **q1tsim** (versión 0.5) y **quantr** (versión 0.6).

**quantr** emerge de la necesidad de tener una interfaz ergonómica para Rust moderno. Frente a q1tsim, que adoptó una API de estilo más imperativo (operaciones sobre un objeto `Qubit` directamente), quantr adopta el modelo de *circuit builder* que ya es familiar en Qiskit y Cirq: se construye un `Circuit`, se añaden puertas, y luego se simula de una vez. Esta separación entre construcción y simulación es más limpia conceptualmente y más fácil de razonar sobre.

### Posición en el benchmark de este TFG

En el benchmark comparativo de este proyecto, quantr es el representante de referencia para Rust en modo statevector puro. El crate se consume en el workspace como:

```toml
[dependencies]
quantr = "0.6"
```

Las razones para incluirlo como referencia son:
1. API moderna y ergonómica que hace que el código sea legible.
2. Simulación correcta del statevector completo (amplitudes complejas exactas).
3. Mantenimiento activo en 2024-2025.
4. Comportamiento predecible y determinista, útil para comparaciones justas.

### Filosofía del simulador: statevector denso

quantr implementa una simulación **statevector densa**: mantiene en memoria el vector completo de 2^n amplitudes complejas (`Complex<f64>`, 16 bytes cada una) y aplica cada puerta como una transformación unitaria exacta sobre ese vector. Es el mismo modelo que usan Qiskit's `statevector_simulator` y Cirq's `Simulator`.

Esta filosofía tiene ventajas e inconvenientes claros:

| Aspecto | Ventaja | Inconveniente |
|---|---|---|
| Exactitud | Resultado exacto, no estadístico | — |
| Memoria | — | Crece exponencialmente: 2^n × 16 bytes |
| Velocidad para n pequeño | Muy rápido (todo en caché L3) | — |
| Velocidad para n grande | — | Lento por la presión de memoria |
| Paralelismo | — | Single-thread en quantr 0.6 |

El límite práctico de **~16 qubits** se deriva directamente de la memoria disponible en RAM de consumidor, como se analiza en detalle en la Sección 6.

---

## 2. API de quantr

### Creación de un circuito

Un circuito en quantr se crea con `Circuit::new(n)`, donde `n` es el número total de qubits (incluyendo ancillas, si las hay). El resultado es un `Result<Circuit, QuantrError>`:

```rust
use quantr::{Circuit, Gate, Measurement, QuantrError};

fn main() -> Result<(), QuantrError> {
    // Circuito de 3 qubits, todos inicializados en |0>
    let mut qc = Circuit::new(3)?;
    Ok(())
}
```

El estado inicial implícito es siempre |0...0⟩. No hay constructor que acepte un estado inicial arbitrario: si se necesita inicializar en otro estado, se añaden puertas X al principio.

### Añadir puertas: Gate y métodos

Las puertas disponibles en la versión 0.6 se representan con el enum `Gate`. Los más relevantes son:

```rust
Gate::H             // Hadamard
Gate::X             // Pauli-X (NOT)
Gate::Y             // Pauli-Y
Gate::Z             // Pauli-Z
Gate::CNot(control) // CNOT con control en `control`
Gate::CZ(control)   // CZ con control en `control`
Gate::Toffoli(c1, c2) // Toffoli (CCNOT): 2 controles, 1 target
Gate::Swap(other)   // SWAP entre el qubit actual y `other`
Gate::CRk(k, ctrl)  // Rotación controlada de fase 2π/2^k (k es i32, signo negativo = rotación inversa)
Gate::RZ(theta)     // Rotación Z por ángulo theta (f64, radianes)
Gate::RX(theta)     // Rotación X
Gate::RY(theta)     // Rotación Y
```

Para aplicar una puerta a un qubit específico se usa `add_gate`:

```rust
qc.add_gate(Gate::H, 0)?;           // Hadamard en qubit 0
qc.add_gate(Gate::CNot(0), 1)?;     // CNOT: control=0, target=1
qc.add_gate(Gate::Toffoli(0, 1), 2)?; // Toffoli: controls=0,1, target=2
```

Para aplicar la misma puerta a varios qubits en paralelo (misma capa del circuito):

```rust
let qubits = vec![0, 1, 2];
qc.add_repeating_gate(Gate::H, &qubits)?;
// Equivale a H en qubit 0, H en qubit 1, H en qubit 2, en la misma capa
```

### Encadenamiento y sintaxis ergonómica

A diferencia de q1tsim, donde cada operación requería gestionar mutabilidad de estados intermedios, en quantr todas las operaciones devuelven `Result<(), QuantrError>` y se pueden encadenar con `?`:

```rust
fn build_bell_pair(qc: &mut Circuit) -> Result<(), QuantrError> {
    qc.add_gate(Gate::H, 0)?;
    qc.add_gate(Gate::CNot(0), 1)?;
    Ok(())
}
```

El `?` propaga cualquier `QuantrError` al llamador, lo cual es idiomático en Rust. Un `QuantrError` puede producirse, por ejemplo, si se intenta crear un circuito de 0 qubits (`Circuit::new(0)` devuelve `Err`).

### Simulación y medida

Una vez construido el circuito, la simulación se invoca con `.simulate()`, que devuelve un objeto simulado. Sobre ese objeto se solicita la medida con `.measure_all(shots)`:

```rust
let sim = qc.simulate();
let counts = match sim.measure_all(1024) {
    Measurement::Observable(c) => c,
    Measurement::NonObservable(c) => c,
};
```

El resultado es un `HashMap<ProductState, usize>` donde cada `ProductState` representa un estado base computacional (una cadena de bits) y el `usize` es el número de veces que ese estado fue observado en las `shots` medidas.

Para obtener la cadena de bits como `String`:

```rust
for (state, count) in counts.into_iter() {
    let bitstring: String = state.to_string();
    println!("{}: {}", bitstring, count);
}
```

**Punto crítico**: quantr es **MSB-first** (Most Significant Bit first). En `ProductState::to_string()`, el qubit 0 es el carácter más a la izquierda. Esto es lo opuesto a la convención de Qiskit (LSB-first). Si el resto del benchmark interpreta el qubit 0 como el bit menos significativo, hay que invertir el bitstring:

```rust
let bs_lsb: String = state.to_string().chars().rev().collect();
let value = u64::from_str_radix(&bs_lsb, 2).unwrap();
```

### Obtener el statevector

Para acceso directo a las amplitudes del statevector (sin muestreo), se usa el resultado de `simulate()` directamente. Sin embargo, para la mayoría de los algoritmos, la ruta `measure_all` + post-procesado del histograma es suficiente y más natural.

### Comparación con la API de q1tsim

| Aspecto | quantr 0.6 | q1tsim 0.5 |
|---|---|---|
| Modelo de construcción | Circuit builder | Circuit builder similar |
| Inicialización | `Circuit::new(n)` | `Circuit::new(n, ...)` con más opciones |
| Aplicar puerta | `qc.add_gate(Gate::H, i)` | `circuit.h(i)` (métodos directos) |
| Puertas multi-qubit | Enum `Gate` con parámetros | Métodos específicos (`toffoli`, `cx`, etc.) |
| Medida | `sim.measure_all(shots)` → `HashMap<ProductState, usize>` | `circuit.measure(...)` integrado |
| Endianness | MSB-first (qubit 0 = izquierda) | LSB-first (qubit 0 = derecha) |
| MCX nativo | Solo hasta 2 controles (`Toffoli`) | Similar, hasta Toffoli nativo |
| Paralelismo | Single-thread | Single-thread |
| Límite práctico | ~16 qubits | ~25 qubits |
| Mantenimiento | Activo 2024-2025 | Menos activo |

La diferencia de límite práctico (~16 vs ~25 qubits) se debe principalmente a la eficiencia de la implementación interna: q1tsim tiene optimizaciones de memoria en su simulador, mientras que quantr prioriza claridad de código sobre rendimiento extremo.

---

## 3. El Problema de las Puertas Multi-Controladas en quantr

### Por qué quantr no tiene MCX/MCZ nativo para n >= 3 controles

La puerta más compleja nativa de quantr es **Toffoli**, que implementa un CCNOT: 2 controles, 1 target. Cualquier operación que requiera 3 o más controles debe descomponerse manualmente.

Esto no es una omisión de la librería por descuido: es una decisión de diseño. Implementar una puerta MCX nativa para n controles arbitrarios requiere aplicar una transformación unitaria de tamaño 2^(n+1) × 2^(n+1) directamente sobre el subespacio de qubits involucrados, lo cual es complejo de implementar eficientemente en un simulador statevector de propósito general. En su lugar, quantr expone los bloques primitivos (Toffoli, CNOT, X) y delega la descomposición al programador.

Para algoritmos como Grover y Shor, donde se necesitan puertas MCZ y MCX con hasta n-1 controles, esta limitación obliga a implementar el **ancilla ladder** (escalera de ancillas).

### El ancilla ladder: concepto fundamental

El **ancilla ladder** (también llamado Toffoli ladder o V-chain en la literatura) es una descomposición de una puerta multi-controlada en una cadena de puertas Toffoli que usan qubits auxiliares (ancillas) para acumular el resultado lógico AND de los controles.

La idea central: en lugar de aplicar MCX(c0, c1, ..., c_{k-1}, target) directamente, se calcula el AND booleano de los controles de forma incremental, almacenando resultados parciales en ancillas.

**Para MCX con 3 controles** (c0, c1, c2, target):
- Se necesita 1 ancilla (k-2 = 3-2 = 1)
- Paso 1 (forward): `Toffoli(c0, c1) → anc[0]`  ← anc[0] = c0 AND c1
- Paso 2 (central): `Toffoli(c2, anc[0]) → target` ← target XOR= (c0 AND c1 AND c2)
- Paso 3 (reverse): `Toffoli(c0, c1) → anc[0]` ← restaurar anc[0] = 0

**Para MCX con 4 controles** (c0, c1, c2, c3, target):
- Se necesitan 2 ancillas (k-2 = 4-2 = 2)
- Paso 1: `Toffoli(c0, c1) → anc[0]` ← anc[0] = c0 AND c1
- Paso 2: `Toffoli(c2, anc[0]) → anc[1]` ← anc[1] = c0 AND c1 AND c2
- Paso 3 (central): `Toffoli(c3, anc[1]) → target`
- Paso 4: `Toffoli(c2, anc[0]) → anc[1]` ← restaurar anc[1] = 0
- Paso 5: `Toffoli(c0, c1) → anc[0]` ← restaurar anc[0] = 0

El patrón es simétrico: la primera mitad ("forward ladder") acumula el AND, la segunda mitad ("reverse ladder") deshace ese cálculo para dejar las ancillas en |0⟩ exactamente como estaban.

### Formalización del escalado

Para MCX con k controles:
- Ancillas necesarias: **k - 2**
- Puertas Toffoli: **2(k-1) - 1** = 2k - 3
  - k-2 Toffolies en el forward
  - 1 Toffoli central
  - k-2 Toffolies en el reverse

| k (controles) | Ancillas | Toffolies totales |
|---|---|---|
| 2 | 0 | 1 (Toffoli nativo) |
| 3 | 1 | 3 |
| 4 | 2 | 5 |
| 5 | 3 | 7 |
| n | n-2 | 2n-3 |

El coste en qubits es lineal en k, y el coste en puertas también es lineal. Esto es importante: el enfoque alternativo de descomposición en puertas universales (sin ancillas) tiene coste O(k²) en puertas. El ancilla ladder es más económico en puertas a cambio de más qubits.

### Coste en el registro del circuito

El punto clave, que es fuente de errores frecuentes en estudiantes, es que **las ancillas deben estar incluidas en el circuito desde el principio**. No se pueden añadir qubits a un `Circuit` una vez creado. El circuito se crea con `Circuit::new(n + n_anc)` donde `n_anc = max(0, k - 2)` y `k` es el máximo número de controles que va a necesitar cualquier puerta del circuito.

Para el algoritmo de Grover con n qubits de búsqueda, la puerta más exigente es MCZ sobre los n qubits. Esta se implementa como H + MCX(n-1 controles, 1 target) + H. MCX con n-1 controles necesita n-3 ancillas. Por tanto, el tamaño total del circuito es:

```
n_total = n (búsqueda) + n_anc (ancillas) = n + max(0, n-2)
```

Para n=3: n_total = 3 + 1 = 4 qubits
Para n=5: n_total = 5 + 3 = 8 qubits
Para n=7: n_total = 7 + 5 = 12 qubits
Para n=11: n_total = 11 + 9 = 20 qubits

Este crecimiento cuasi-lineal en qubits totales, combinado con el crecimiento exponencial de la memoria (2^n_total), explica por qué el benchmark de este TFG se limita a n_values = [3, 5, 7, 9, 11].

### El código Rust real del ancilla ladder

La función `add_mcx` en `src/grover.rs` implementa el ancilla ladder de forma limpia:

```rust
/// Multi-controlled X gate usando el ancilla + Toffoli-ladder.
///
/// `controls`: índices de los qubits de control (longitud k)
/// `target`:   índice del qubit objetivo
/// `ancillas`: k-2 qubits auxiliares en estado |0> al entrar
///             (garantizados en |0> al salir, gracias al reverse ladder)
pub fn add_mcx(
    qc: &mut Circuit,
    controls: &[usize],
    target: usize,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let k = controls.len();
    match k {
        0 => {
            // 0 controles: X incondicional
            qc.add_gate(Gate::X, target)?;
        }
        1 => {
            // 1 control: CNOT nativo
            qc.add_gate(Gate::CNot(controls[0]), target)?;
        }
        2 => {
            // 2 controles: Toffoli nativo
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), target)?;
        }
        _ => {
            // k >= 3: ancilla ladder
            assert!(
                ancillas.len() >= k - 2,
                "MCX ladder necesita k-2 ancillas (k={}, dado {})",
                k, ancillas.len()
            );

            // Forward ladder: acumular AND de controles
            // Toffoli(c[0], c[1]) -> anc[0]
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
            // Para i = 2, 3, ..., k-2:
            //   Toffoli(c[i], anc[i-2]) -> anc[i-1]
            for i in 2..(k - 1) {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i - 2]), ancillas[i - 1])?;
            }

            // Puerta central: flip target si anc[k-3] == 1
            qc.add_gate(Gate::Toffoli(controls[k - 1], ancillas[k - 3]), target)?;

            // Reverse ladder: deshacer las ancillas (mismo orden invertido)
            for i in (2..(k - 1)).rev() {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i - 2]), ancillas[i - 1])?;
            }
            // Restaurar anc[0]
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
        }
    }
    Ok(())
}
```

**Ejemplo concreto para k=4** (4 controles: c0, c1, c2, c3; 2 ancillas: anc0, anc1):

```
Forward:
  Toffoli(c0, c1)    → anc0    [anc0 = c0 AND c1]
  Toffoli(c2, anc0)  → anc1    [anc1 = c0 AND c1 AND c2]

Central:
  Toffoli(c3, anc1)  → target  [target ^= c0 AND c1 AND c2 AND c3]

Reverse:
  Toffoli(c2, anc0)  → anc1    [anc1 restaurado a 0]
  Toffoli(c0, c1)    → anc0    [anc0 restaurado a 0]
```

La simetría es perfecta: el reverse ladder es el forward ejecutado al revés. Aplicar el mismo Toffoli dos veces sobre el mismo target deshace la operación, porque Toffoli es su propia inversa (es una puerta unitaria y hermítica: T† = T).

---

## 4. Implementación de Grover en quantr

### Estructura general del circuito

El algoritmo de Grover en quantr tiene la siguiente arquitectura:

```
Registro: [q0, q1, ..., q_{n-1}] [anc0, anc1, ..., anc_{n-3}]
           ↑ qubits de búsqueda    ↑ ancillas para MCZ

1. Inicializar: H^⊗n sobre [q0..q_{n-1}]
2. Repetir √(2^n) / 4 * π veces:
   a. Oráculo: invierte fase del estado objetivo
   b. Difusor: inversión sobre la media (amplifica)
3. Medir [q0..q_{n-1}] (las ancillas están en |0>)
```

La función `grover_circuit` en el código real implementa exactamente esto:

```rust
pub fn grover_circuit(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
) -> Result<Circuit, QuantrError> {
    assert!(n >= 1);
    assert!(target < (1u64 << n));

    // Calcular número de iteraciones: floor(π/4 * √(2^n))
    let iterations = num_iterations.unwrap_or_else(|| {
        let n_states = (1u64 << n) as f64;
        ((std::f64::consts::PI / 4.0) * n_states.sqrt()).floor() as usize
    });

    // Ancillas necesarias: max(0, n - 2)
    let n_anc = if n >= 3 { n - 2 } else { 0 };
    let total = n + n_anc;
    let ancillas: Vec<usize> = (n..total).collect();

    // Crear circuito con n + n_anc qubits
    let mut qc = Circuit::new(total)?;

    // Inicialización: superposición uniforme en los qubits de búsqueda
    let search: Vec<usize> = (0..n).collect();
    qc.add_repeating_gate(Gate::H, &search)?;

    // Bucle de Grover
    for _ in 0..iterations {
        build_oracle(&mut qc, n, target, &ancillas)?;
        build_diffuser(&mut qc, n, &ancillas)?;
    }

    Ok(qc)
}
```

### El oráculo de fase

El oráculo invierte la fase del estado |target⟩ sin perturbarlo visualmente. Implementación:

1. Para cada bit i del target que sea 0, aplicar X al qubit i (mapea |0⟩ → |1⟩).
2. Aplicar MCZ sobre todos los qubits de búsqueda (invierte la fase de |11...1⟩).
3. Deshacer los X del paso 1.

Esto funciona porque después del paso 1, el único estado que tiene todos los qubits en |1⟩ es exactamente |target⟩ (con los bits cero "corregidos" por las puertas X). MCZ invierte la fase de ese único estado. Las X del paso 3 restauran el mapeo.

```rust
pub fn build_oracle(
    qc: &mut Circuit,
    n: usize,
    target: u64,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    // Paso 1: X en posiciones donde target tiene un 0
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            qc.add_gate(Gate::X, i)?;
        }
    }

    // Paso 2: MCZ sobre todos los n qubits de búsqueda
    add_mcz(qc, n, ancillas)?;

    // Paso 3: deshacer X
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            qc.add_gate(Gate::X, i)?;
        }
    }
    Ok(())
}
```

### MCZ via conjugación H

La puerta MCZ (multi-controlled Z) se implementa a través de la identidad:
**MCZ = H · MCX · H** (aplicada al qubit target)

Esto se deriva de que Z = H X H: conjugar X con Hadamard da Z. La misma identidad escala a multi-control:

```rust
pub fn add_mcz(qc: &mut Circuit, n: usize, ancillas: &[usize]) -> Result<(), QuantrError> {
    match n {
        0 => {}
        1 => {
            qc.add_gate(Gate::Z, 0)?;
        }
        2 => {
            // CZ nativo: control en 0, target en 1
            qc.add_gate(Gate::CZ(0), 1)?;
        }
        _ => {
            // Para n >= 3: H · MCX(c0..c_{n-2}, t_{n-1}) · H
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

En el caso `n >= 3`, el target es el qubit `n-1` y los controles son todos los qubits anteriores (0..n-2). MCX sobre esos controles y target se implementa con el ancilla ladder. Los qubits ancilla son los que están después del registro de búsqueda.

### El difusor de Grover

El operador de inversión sobre la media (difusor) se implementa como:
**D = H^⊗n · (2|0⟩⟨0| - I) · H^⊗n**

La matriz `2|0⟩⟨0| - I` invierte la fase de todos los estados excepto |0...0⟩. En términos de circuito:

1. H en todos los qubits (cambiar base).
2. X en todos los qubits (mapear |0...0⟩ → |1...1⟩).
3. MCZ sobre todos los qubits (invertir fase de |1...1⟩, que corresponde a |0...0⟩ original).
4. X en todos los qubits (restaurar).
5. H en todos los qubits (volver a base computacional).

```rust
pub fn build_diffuser(
    qc: &mut Circuit,
    n: usize,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let qs: Vec<usize> = (0..n).collect();

    qc.add_repeating_gate(Gate::H, &qs)?;   // Base change
    qc.add_repeating_gate(Gate::X, &qs)?;   // Flip zeros

    add_mcz(qc, n, ancillas)?;              // Phase flip of |11...1>

    qc.add_repeating_gate(Gate::X, &qs)?;   // Restore
    qc.add_repeating_gate(Gate::H, &qs)?;   // Base change back

    Ok(())
}
```

### Ejecución y post-procesado

```rust
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let qc = grover_circuit(args.n, args.target, Some(args.iterations))?;
    let sim = qc.simulate();

    let counts = match sim.measure_all(args.shots) {
        Measurement::Observable(c) => c,
        Measurement::NonObservable(c) => c,
    };

    // quantr es MSB-first: qubit 0 es el carácter más a la izquierda.
    // Las ancillas están en los qubits n..total y vuelven a |0> siempre.
    // Extraemos solo los primeros n caracteres (registro de búsqueda),
    // y los invertimos para obtener la convención LSB-first del benchmark.
    let mut distribution: HashMap<String, usize> = HashMap::new();
    for (state, count) in counts.into_iter() {
        let raw = state.to_string();
        let search_msb: String = raw.chars().take(args.n).collect();
        let search_lsb: String = search_msb.chars().rev().collect();
        *distribution.entry(search_lsb).or_insert(0) += count;
    }

    let (best, _) = distribution.iter().max_by_key(|(_, c)| *c).unwrap();
    let found = u64::from_str_radix(best, 2)?;
    // ...
    Ok(())
}
```

**Puntos clave del post-procesado**:
1. Las ancillas están en los qubits `n..total`. Dado que el ancilla ladder siempre las restaura a |0⟩, su contribución al bitstring medido es siempre "0...0" y se puede descartar tomando solo los primeros `n` caracteres.
2. quantr es MSB-first: `raw.chars().take(n)` da el registro de búsqueda en orden MSB.
3. La inversión `.chars().rev().collect()` convierte a LSB-first para compatibilidad con el resto del benchmark.

---

## 5. Implementación de Shor en quantr

### Arquitectura del circuito de order-finding

El circuito de order-finding en quantr sigue el esquema estándar de la QPE (Quantum Phase Estimation):

```
Registro de control (qubits 0..m):    fase del periodo
Registro objetivo (qubits m..m+n):    valor modular
Registro ancilla (qubits m+n..total): para las MCX ladders

1. H^⊗m sobre registro de control
2. X en el LSB del registro objetivo (inicializar en |1>)
3. Exponenciación modular controlada:
   Para cada ctrl_bit i (0 a m-1):
     Aplicar A^(2^(m-1-i)) * y mod N
     controlado por ctrl_bit i
4. QFT^-1 sobre registro de control
5. Medir registro de control
```

### Layout de qubits y convención big-endian

Un detalle importante es cómo el código mapea bits lógicos a qubits físicos. quantr es MSB-first (qubit 0 a la izquierda en la cadena de bits). Para que el post-procesado sea compatible con la referencia en Python, se usa el siguiente mapeo:

- El qubit de control i=0 es el MSB de la fase medida (qubit 0 del circuito).
- El bit lógico k del registro objetivo se mapea al qubit físico `m + (n-1-k)`.

Esto invierte el orden del registro objetivo: el LSB lógico está en el qubit físico más a la derecha, igual que en Python. El código en `order_finding_circuit`:

```rust
let target_qubits: Vec<usize> = (0..n_bits)
    .map(|k| m + (n_bits - 1 - k))
    .collect();
```

### QFT inversa con Gate::CRk

La QFT inversa se implementa en `src/shor/qft.rs` usando la puerta `Gate::CRk(k, ctrl)`, que aplica una rotación de fase controlada `e^{2πi/2^k}`. La clave está en que `k` es un `i32` con signo: **k negativo invierte la rotación** (`e^{-2πi/2^|k|}`), lo cual es exactamente lo que se necesita para la QFT inversa sin duplicar código.

```rust
pub fn apply_inverse_qft(
    qc: &mut Circuit,
    start: usize,
    len: usize,
) -> Result<(), QuantrError> {
    // Primero invertir el orden de los qubits con SWAPs
    for i in 0..(len / 2) {
        qc.add_gate(Gate::Swap(start + len - 1 - i), start + i)?;
    }
    // Aplicar QFT^-1: orden inverso de QFT, con rotaciones negativas
    for i in (0..len).rev() {
        for j in ((i + 1)..len).rev() {
            let k = (j - i + 1) as i32;
            // k negativo → rotación de -2π/2^k (inversa)
            qc.add_gate(Gate::CRk(-k, start + j), start + i)?;
        }
        qc.add_gate(Gate::H, start + i)?;
    }
    Ok(())
}
```

### Exponenciación modular: redes de permutación

La parte más creativa de la implementación de Shor en quantr es cómo implementar la exponenciación modular controlada. quantr no tiene ninguna primitiva de alto nivel para esto; hay que descomponerlo completamente en puertas elementales.

La estrategia adoptada es la **permutation network** (red de permutaciones):

1. La multiplicación modular `y → A^p * y mod N` es una permutación de los estados base del registro objetivo (para los estados `0..N-1`; los estados `N..2^n-1` son inválidos y se ignoran).

2. Toda permutación se descompone en ciclos disjuntos.

3. Cada ciclo se descompone en transposiciones de la forma `(cycle[0], cycle[i])`.

4. Cada transposición `|a⟩ ↔ |b⟩` se implementa como una secuencia de transposiciones de un solo bit (donde a XOR b es una potencia de 2).

5. Cada transposición de un solo bit se implementa como una MCX con X-wraps en los qubits que deben ser |1⟩ como condición.

En `src/shor/permutation.rs`:

```rust
pub fn build_mod_exp_permutation(a: u64, n_val: u64, power: u64) -> HashMap<u64, u64> {
    let a_power = mod_pow(a, power, n_val);
    let mut perm: HashMap<u64, u64> = HashMap::new();
    for y in 0..n_val {
        let target = (a_power * y) % n_val;
        if y != target {
            perm.insert(y, target);  // Solo entradas no-identidad
        }
    }
    perm
}
```

La transposición de un solo bit funciona así: si `a` y `b` difieren solo en el bit `flip_bit`, entonces para intercambiar |a⟩ y |b⟩ se aplica una MCX controlada por todos los demás bits (con X-wraps donde el bit es 0 en el estado `a`):

```rust
pub fn controlled_single_bit_transposition(
    qc: &mut Circuit,
    ctrl: usize,      // qubit de control exterior (del registro de control)
    target_qubits: &[usize],
    a: u64,
    b: u64,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let diff = a ^ b;
    let flip_bit = diff.trailing_zeros() as usize;

    // Los otros bits de `a` son condiciones adicionales de control
    let other_positions: Vec<usize> = (0..target_qubits.len())
        .filter(|&i| i != flip_bit)
        .collect();

    // X-wraps: convertir 0-controles en 1-controles
    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            qc.add_gate(Gate::X, target_qubits[pos])?;
        }
    }

    // MCX con todos los controles + flip del bit diferente
    let mut controls = vec![ctrl];
    for &pos in &other_positions {
        controls.push(target_qubits[pos]);
    }
    add_mcx(qc, &controls, target_qubits[flip_bit], ancillas)?;

    // Deshacer X-wraps
    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            qc.add_gate(Gate::X, target_qubits[pos])?;
        }
    }
    Ok(())
}
```

### Post-procesado clásico: fracciones continuas

Una vez medido el registro de control, el valor obtenido `x` se interpreta como la aproximación de la fase `x/2^m ≈ k/r` (donde `r` es el orden buscado). Se aplica el algoritmo de fracciones continuas para encontrar `k/r` como fracción irreducible:

```rust
// classical.rs: port fiel de Fraction.limit_denominator de CPython
pub fn limit_denominator(
    numerator: i128,
    denominator: i128,
    max_denominator: i128,
) -> (i128, i128) {
    // Devuelve la fracción p/q más cercana a numerator/denominator
    // con q <= max_denominator, usando el algoritmo de fracciones continuas.
    // ...
}
```

El denominador resultante es un candidato al orden `r`. Se verifica que `a^r ≡ 1 (mod N)` y se intenta extraer factores con `gcd(a^(r/2) ± 1, N)`.

### Diferencias respecto a q1tsim

| Aspecto | quantr | q1tsim |
|---|---|---|
| QFT | `Gate::CRk(k, ctrl)` con k signed | Método `cr2(ctrl, target)` separado |
| Exp. modular | Permutation network (MCX ladders) | Idem (mismo enfoque) |
| Ancillas | Explícitas en el circuito desde el inicio | Idem |
| Endianness del control | Qubit 0 = MSB de la fase | Qubit 0 = LSB (requiere inversión diferente) |
| Post-procesado | `limit_denominator` en Rust puro | Similar, usando `num-rational` |

---

## 6. El Límite de ~16 Qubits

### Fundamento: crecimiento exponencial del statevector

El estado de un sistema de n qubits es un vector en un espacio de Hilbert de dimensión 2^n. Para almacenar ese vector completo, un simulador statevector necesita 2^n números complejos. Usando `Complex<f64>` (dos f64 de 8 bytes cada uno):

```
memoria = 2^n × 16 bytes
```

| n qubits | Dimensión | Memoria |
|---|---|---|
| 10 | 1 024 | 16 KB |
| 14 | 16 384 | 256 KB |
| 16 | 65 536 | 1 MB |
| 20 | 1 048 576 | 16 MB |
| 24 | 16 777 216 | 256 MB |
| 26 | 67 108 864 | 1 GB |
| 28 | 268 435 456 | 4 GB |
| 30 | 1 073 741 824 | 16 GB |

Con 8 GB de RAM disponibles (máquina de desarrollo típica), el límite teórico está en torno a n=26. Pero hay un efecto de segundo orden importante: las operaciones de puerta requieren espacio de trabajo adicional, la localidad de caché degrada fuertemente para vectores grandes, y el sistema operativo necesita memoria propia.

En la práctica, quantr 0.6 en single-thread empieza a ser lento con n=18 y se vuelve impracticable para el benchmark (tiempo > 60 s) con n=20 aproximadamente. El límite de **~16 qubits** para ejecución rápida es un valor empírico bien establecido.

### El efecto de las ancillas en el benchmark de Grover

En el benchmark de este TFG, los valores son `n_values = [3, 5, 7, 9, 11]` (qubits de búsqueda). Pero el circuito real tiene más qubits porque incluye las ancillas:

| n búsqueda | n ancillas | n total | Dimensión del SV | Memoria |
|---|---|---|---|---|
| 3 | 1 | 4 | 16 | 256 B |
| 5 | 3 | 8 | 256 | 4 KB |
| 7 | 5 | 12 | 4 096 | 64 KB |
| 9 | 7 | 16 | 65 536 | 1 MB |
| 11 | 9 | 20 | 1 048 576 | 16 MB |

Esto explica por qué el proyecto no supera n=11: el siguiente paso sería n=13, que requeriría n_total = 13+11 = 24 qubits y 256 MB solo para el statevector, más el tiempo de simulación correspondiente.

### Por qué no se paraleliza en quantr 0.6

quantr 0.6 es single-thread por diseño. La paralelización del statevector requiere acceso concurrente al vector completo durante la aplicación de puertas de dos qubits, lo cual introduce complejidad de sincronización. Frameworks como qcgpu resuelven esto con OpenCL (GPU), y quantrs2 usa cuQuantum de NVIDIA para operaciones paralelas masivas. En quantr, la prioridad es la correctitud y la simplicidad del código.

---

## 7. Comparación con q1tsim

### Tabla de diferencias detallada

| Criterio | quantr 0.6 | q1tsim 0.5 |
|---|---|---|
| **Versión actual** | 0.6.0 | 0.5.0 |
| **Estado de mantenimiento** | Activo (2024-2025) | Menos activo |
| **API de construcción** | `add_gate(Gate::Variant, qubit)` | Métodos como `.h(i)`, `.cx(c, t)` |
| **Endianness** | MSB-first (qubit 0 = izquierda) | LSB-first (qubit 0 = derecha) |
| **Puertas nativas máximas** | `Toffoli(c1, c2)` = 2 controles | `toffoli(c1, c2, t)` = 2 controles |
| **Puertas de rotación** | `Gate::CRk(k, ctrl)`, `Gate::RZ(θ)` | `cr2(ctrl, t)`, rotaciones manuales |
| **MCX para k≥3** | Ancilla ladder manual | Ancilla ladder manual |
| **Limit práctico** | ~16 qubits | ~25 qubits |
| **Paralelismo** | Single-thread | Single-thread |
| **Medida** | `sim.measure_all(shots)` externa | Integrada en el circuito |
| **Crate de fracciones** | Implementación propia (classical.rs) | `num-rational` |
| **Tratamiento de errores** | `Result<_, QuantrError>` idiomático | Similar |
| **Documentación** | Buena, con ejemplos | Razonable |

### Cuándo usar uno u otro

**Usar quantr cuando**:
- Se quiere una API limpia y moderna en el estilo "circuit builder" de Rust.
- Se trabaja con n ≤ 15 qubits y la prioridad es claridad de código.
- Se necesita `Gate::CRk` con rotaciones negativas (QFT inversa elegante).
- Se quiere el proyecto más activo.

**Usar q1tsim cuando**:
- Se necesitan más qubits (~20-25) con la misma RAM disponible.
- Se prefiere una API estilo método directo (`circuit.h(i)`) en lugar de enum.
- Se necesita acceso al statevector intermediario durante la construcción.

### Ambos comparten la misma limitación fundamental

Ni quantr ni q1tsim resuelven el problema de las puertas multi-controladas nativamente. Los dos crates exponen Toffoli como máximo nativo, y los dos requieren que el programador implemente el ancilla ladder manualmente para k≥3 controles. Esta es la limitación arquitectural más importante del ecosistema Rust de simulación cuántica de bajo nivel frente a frameworks de alto nivel como Qiskit, que proporcionan `MCXGate` directamente.

---

## Apéndice A: Ejemplo mínimo completo

El siguiente programa crea un circuito de Grover para n=3 qubits, busca el estado |5⟩ = |101⟩, simula y muestra la distribución:

```rust
use std::collections::HashMap;
use quantr::{Circuit, Gate, Measurement, QuantrError};

fn add_mcx(
    qc: &mut Circuit,
    controls: &[usize],
    target: usize,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    match controls.len() {
        0 => qc.add_gate(Gate::X, target)?,
        1 => qc.add_gate(Gate::CNot(controls[0]), target)?,
        2 => qc.add_gate(Gate::Toffoli(controls[0], controls[1]), target)?,
        k => {
            let ancs = ancillas;
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancs[0])?;
            for i in 2..(k - 1) {
                qc.add_gate(Gate::Toffoli(controls[i], ancs[i-2]), ancs[i-1])?;
            }
            qc.add_gate(Gate::Toffoli(controls[k-1], ancs[k-3]), target)?;
            for i in (2..(k-1)).rev() {
                qc.add_gate(Gate::Toffoli(controls[i], ancs[i-2]), ancs[i-1])?;
            }
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancs[0])?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 3usize;
    let target_state = 5u64;   // |101>
    let n_anc = n - 2;         // = 1 ancilla para n=3
    let total = n + n_anc;     // = 4 qubits en total
    let ancillas: Vec<usize> = (n..total).collect();  // [3]

    let mut qc = Circuit::new(total)?;

    // Superposición inicial
    let search: Vec<usize> = (0..n).collect();
    qc.add_repeating_gate(Gate::H, &search)?;

    // 1 iteración de Grover para n=3 (floor(π/4 * √8) = 2, pero 1 funciona)
    let iterations = ((std::f64::consts::PI / 4.0) * (1u64 << n) as f64).sqrt().floor() as usize;

    for _ in 0..iterations {
        // Oráculo: invertir fase de |5> = |101>
        // Bit 0 = 1, bit 1 = 0, bit 2 = 1
        qc.add_gate(Gate::X, 1)?;  // bit 1 es 0 en el target
        {
            let controls: Vec<usize> = (0..n-1).collect();
            let t = n - 1;
            qc.add_gate(Gate::H, t)?;
            add_mcx(&mut qc, &controls, t, &ancillas)?;
            qc.add_gate(Gate::H, t)?;
        }
        qc.add_gate(Gate::X, 1)?;

        // Difusor
        qc.add_repeating_gate(Gate::H, &search)?;
        qc.add_repeating_gate(Gate::X, &search)?;
        {
            let controls: Vec<usize> = (0..n-1).collect();
            let t = n - 1;
            qc.add_gate(Gate::H, t)?;
            add_mcx(&mut qc, &controls, t, &ancillas)?;
            qc.add_gate(Gate::H, t)?;
        }
        qc.add_repeating_gate(Gate::X, &search)?;
        qc.add_repeating_gate(Gate::H, &search)?;
    }

    // Simular y medir
    let sim = qc.simulate();
    let counts = match sim.measure_all(1024) {
        Measurement::Observable(c) => c,
        Measurement::NonObservable(c) => c,
    };

    // Post-procesar: solo registro de búsqueda, convertir a LSB-first
    let mut dist: HashMap<String, usize> = HashMap::new();
    for (state, count) in counts {
        let raw = state.to_string();
        let search_bits: String = raw.chars().take(n).collect();
        let lsb: String = search_bits.chars().rev().collect();
        *dist.entry(lsb).or_insert(0) += count;
    }

    println!("Distribución (LSB-first, solo registro de búsqueda):");
    let mut sorted: Vec<_> = dist.iter().collect();
    sorted.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (bs, count) in sorted.iter().take(5) {
        let val = u64::from_str_radix(bs, 2).unwrap();
        println!("  |{}> (={}) : {} veces ({:.1}%)", bs, val, count, *count as f64 / 1024.0 * 100.0);
    }

    Ok(())
}
```

---

## Apéndice B: Resumen de conceptos clave

1. **quantr 0.6** es un simulador statevector denso en Rust con API circuit-builder ergonómica.

2. **MSB-first**: el qubit 0 está a la izquierda en el bitstring. Siempre invertir el bitstring para obtener convención LSB-first.

3. **Toffoli es el máximo nativo**: para MCX con 3+ controles, usar el ancilla ladder.

4. **El ancilla ladder** requiere k-2 qubits auxiliares para MCX con k controles. Usa puertas Toffoli en patrón forward-central-reverse. El reverse ladder deja las ancillas en |0⟩.

5. **MCZ = H · MCX · H**: conjugar MCX con Hadamard en el target da MCZ.

6. **El circuito de Grover** necesita `n + max(0, n-2)` qubits en total. Para n=11, son 20 qubits.

7. **Límite ~16 qubits** de uso cómodo: 2^16 × 16 bytes = 1 MB de statevector, más overhead. n=20 ya son 16 MB; n=24 son 256 MB.

8. **QFT con Gate::CRk**: k positivo da rotación directa; k negativo da rotación inversa. Esto permite implementar QFT e IQFT con la misma primitiva.

9. **La exponenciación modular en Shor** se implementa como permutation network: la multiplicación modular es una permutación de estados base, que se descompone en transposiciones de un solo bit, que a su vez se implementan con MCX + X-wraps.

---

## Referencias

[1] Nielsen, M.A. & Chuang, I.L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.

[2] Barenco, A., et al. (1995). "Elementary gates for quantum computation." *Physical Review A*, 52(5), 3457-3467.

[3] Selinger, P. (2013). "Quantum circuits of T-depth one." *Physical Review A*, 87, 042302.

[4] Amy, M., Maslov, D., Mosca, M., & Roetteler, M. (2013). "A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum circuits." *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 32(6), 818-830.

[5] Häner, T. & Steiger, D.S. (2017). "0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit." In *Proceedings of SC17: The International Conference for High Performance Computing, Networking, Storage and Analysis*. ACM/IEEE.

[6] Preskill, J. (2018). "Quantum Computing in the NISQ Era and Beyond." *Quantum*, 2, 79.

[7] Grover, L.K. (1996). "A fast quantum mechanical algorithm for database search." In *Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC)*, pp. 212-219.

[8] Shor, P.W. (1997). "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer." *SIAM Journal on Computing*, 26(5), 1484-1509.

[9] Shende, V.V., Markov, I.L., & Bullock, S.S. (2004). "Minimal Universal Two-Qubit Controlled-NOT-Based Circuits." *Physical Review A*, 69(6), 062321.

[10] Dueck, G.W. & Maslov, D. (2005). "Reversible function synthesis with minimum garbage outputs." In *6th International Symposium on Representations and Methodology of Future Computing Technologies*, pp. 154-161.

[11] Gheorghiu, V. & Mosca, M. (2017). "Classical simulation of quantum many-body systems with a tree tensor network." *Physical Review B*, 91(23), 235430.

[12] Aharonov, D. & Ben-Or, M. (2008). "Fault-tolerant quantum computation with constant error rate." *SIAM Journal on Computing*, 38(4), 1207-1282.
