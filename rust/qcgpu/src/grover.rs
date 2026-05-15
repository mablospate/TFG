//! Grover's algorithm on the qcgpu OpenCL-based simulator.
//!
//! qcgpu only exposes single-qubit gates, single-control gates, Toffoli, and
//! swap. Multi-controlled Z (MCZ) for n >= 4 must be decomposed using
//! Toffoli gates plus ancilla qubits, following section 7.10 (strategy B) of
//! the implementation guide.

use std::collections::HashMap;
use std::panic;
use std::time::Instant;

use clap::Parser;
use qcgpu::State;
use serde::Serialize;

/// CLI arguments for the Grover binary.
#[derive(Parser, Debug)]
#[command(name = "grover", about = "Grover's search on qcgpu (OpenCL)")]
pub struct Args {
    /// Number of search qubits.
    #[arg(long)]
    pub n: u32,

    /// Target basis state (0 <= target < 2^n).
    #[arg(long)]
    pub target: u64,

    /// Number of shots for sampling.
    #[arg(long, default_value_t = 1024)]
    pub shots: usize,

    /// Override the number of Grover iterations (default: floor(pi/4 * sqrt(2^n))).
    #[arg(long)]
    pub iterations: Option<usize>,
}

/// JSON output schema (success case).
#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,
    pub algorithm: &'static str,
    pub n: u32,
    pub target: u64,
    pub shots: usize,
    pub found: u64,
    pub time_ms: f64,
    pub distribution: HashMap<String, usize>,
}

/// JSON output schema (error case).
#[derive(Serialize)]
pub struct ErrorOutput {
    pub framework: &'static str,
    pub algorithm: &'static str,
    pub n: u32,
    pub target: u64,
    pub error: String,
}

/// Apply a controlled-Z to (control, target) using H + CX + H.
pub fn cz(state: &mut State, control: i32, target: i32) {
    state.h(target);
    state.cx(control, target);
    state.h(target);
}

/// Apply a multi-controlled X on `controls -> target` using the Toffoli +
/// ancilla ladder (strategy B). `ancillas` must hold at least `k - 2` qubits
/// initialized to |0> when `k >= 3`, where k = controls.len().
pub fn mcx(state: &mut State, controls: &[i32], target: i32, ancillas: &[i32]) {
    let k = controls.len();
    match k {
        0 => state.x(target),
        1 => state.cx(controls[0], target),
        2 => state.toffoli(controls[0], controls[1], target),
        _ => {
            assert!(
                ancillas.len() >= k - 2,
                "mcx needs k-2 ancillas, got {} for k={}",
                ancillas.len(),
                k
            );

            // Forward ladder: combine controls into ancillas.
            state.toffoli(controls[0], controls[1], ancillas[0]);
            for i in 2..(k - 1) {
                state.toffoli(controls[i], ancillas[i - 2], ancillas[i - 1]);
            }
            // Final Toffoli writes the MCX result onto `target`.
            state.toffoli(controls[k - 1], ancillas[k - 3], target);
            // Uncompute the ladder so ancillas return to |0>.
            for i in (2..(k - 1)).rev() {
                state.toffoli(controls[i], ancillas[i - 2], ancillas[i - 1]);
            }
            state.toffoli(controls[0], controls[1], ancillas[0]);
        }
    }
}

/// Apply MCZ on `qubits` using H + MCX + H around the last qubit.
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

/// Oracle: flip the phase of |target> on the first n qubits using ancilla qubits if needed.
pub fn build_oracle(state: &mut State, n: u32, target: u64, ancillas: &[i32]) {
    let n_i = n as i32;
    let qubits: Vec<i32> = (0..n_i).collect();

    // X on qubits where the target bit is 0, mapping |target> to |11...1>.
    for i in 0..n_i {
        if (target >> i) & 1 == 0 {
            state.x(i);
        }
    }

    mcz(state, &qubits, ancillas);

    // Undo the X flips.
    for i in 0..n_i {
        if (target >> i) & 1 == 0 {
            state.x(i);
        }
    }
}

/// Diffuser: inversion about the mean on the first n qubits.
pub fn build_diffuser(state: &mut State, n: u32, ancillas: &[i32]) {
    let n_i = n as i32;
    let qubits: Vec<i32> = (0..n_i).collect();

    for i in 0..n_i {
        state.h(i);
    }
    for i in 0..n_i {
        state.x(i);
    }

    mcz(state, &qubits, ancillas);

    for i in 0..n_i {
        state.x(i);
    }
    for i in 0..n_i {
        state.h(i);
    }
}

/// Build the full Grover state. Returns the state after all gates have been
/// applied (no measurement). Ancilla qubits live at indices [n, n + n_anc).
pub fn grover_state(n: u32, target: u64, num_iterations: usize) -> State {
    let n_anc: u32 = if n >= 3 { n - 2 } else { 0 };
    let total = n + n_anc;

    let mut state = State::new(total, 0);
    let ancillas: Vec<i32> = (n as i32..(n + n_anc) as i32).collect();

    // Uniform superposition on the search register.
    for i in 0..n as i32 {
        state.h(i);
    }

    for _ in 0..num_iterations {
        build_oracle(&mut state, n, target, &ancillas);
        build_diffuser(&mut state, n, &ancillas);
    }

    state
}

/// Run Grover and return (most-frequent search-register value, distribution).
pub fn run_grover(
    n: u32,
    target: u64,
    iterations: Option<usize>,
    shots: usize,
) -> (u64, HashMap<String, usize>) {
    let iters = iterations
        .unwrap_or_else(|| ((std::f64::consts::PI / 4.0) * (2u64.pow(n) as f64).sqrt()).floor() as usize);

    let mut state = grover_state(n, target, iters);
    let raw = state.measure_many(shots as i32);

    // Keep only the search-register bits (the rightmost n bits, since qubit 0
    // is the rightmost character in qcgpu bitstrings).
    let mut dist: HashMap<String, usize> = HashMap::new();
    for (bs, count) in raw {
        let total = bs.len();
        let start = total.saturating_sub(n as usize);
        let key = bs[start..].to_string();
        *dist.entry(key).or_insert(0) += count as usize;
    }

    let best_bs = dist
        .iter()
        .max_by_key(|(_, &c)| c)
        .map(|(k, _)| k.clone())
        .unwrap_or_default();
    let found = u64::from_str_radix(&best_bs, 2).unwrap_or(0);
    (found, dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Pure-Rust tests (no OpenCL required) ----

    /// Default iteration count matches floor(pi/4 * sqrt(2^n)).
    fn default_iterations(n: u32) -> usize {
        ((std::f64::consts::PI / 4.0) * (2u64.pow(n) as f64).sqrt()).floor() as usize
    }

    #[test]
    fn test_default_iterations_formula() {
        // n=3 → floor(pi/4 * sqrt(8)) = floor(2.221...) = 2
        assert_eq!(default_iterations(3), 2);
        // n=4 → floor(pi/4 * sqrt(16)) = floor(3.1415...) = 3
        assert_eq!(default_iterations(4), 3);
        // n=2 → floor(pi/4 * sqrt(4)) = floor(1.5707...) = 1
        assert_eq!(default_iterations(2), 1);
    }

    #[test]
    fn test_ancilla_count() {
        // qcgpu Grover uses n - 2 ancillas for n >= 3, else 0.
        let count = |n: u32| -> u32 { if n >= 3 { n - 2 } else { 0 } };
        assert_eq!(count(1), 0);
        assert_eq!(count(2), 0);
        assert_eq!(count(3), 1);
        assert_eq!(count(4), 2);
        assert_eq!(count(8), 6);
    }

    #[test]
    fn test_bitstring_to_int_conversion() {
        // The conversion logic used by run_grover.
        assert_eq!(u64::from_str_radix("101", 2).unwrap(), 5);
        assert_eq!(u64::from_str_radix("000", 2).unwrap(), 0);
        assert_eq!(u64::from_str_radix("1011", 2).unwrap(), 11);
    }

    // ---- OpenCL-dependent tests (require GPU + OpenCL runtime) ----

    #[test]
    #[ignore]
    fn test_grover_finds_target_n3() {
        let (found, dist) = run_grover(3, 5, None, 200);
        assert_eq!(found, 5);
        let count = dist.get("101").copied().unwrap_or(0);
        assert!(count > 100, "Expected '101' to dominate, got {}", count);
    }

    #[test]
    #[ignore]
    fn test_grover_finds_target_zero() {
        let (found, dist) = run_grover(3, 0, None, 200);
        assert_eq!(found, 0);
        let count = dist.get("000").copied().unwrap_or(0);
        assert!(count > 100, "Expected '000' to dominate (>100/200 shots), got {}", count);
    }

    #[test]
    #[ignore]
    fn test_grover_finds_n4() {
        let (found, _) = run_grover(4, 11, None, 200);
        assert_eq!(found, 11);
    }
}

/// Binary entrypoint: parse CLI args, run Grover, and emit JSON output.
pub fn run() -> ! {
    let args = Args::parse();

    let n = args.n;
    let target = args.target;
    let shots = args.shots;

    if target >= (1u64 << n) {
        let err = ErrorOutput {
            framework: "qcgpu",
            algorithm: "grover",
            n,
            target,
            error: format!("target {} out of range for n = {}", target, n),
        };
        println!("{}", serde_json::to_string(&err).unwrap());
        std::process::exit(0);
    }

    let start = Instant::now();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_grover(n, target, args.iterations, shots)
    }));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok((found, dist)) => {
            let out = Output {
                framework: "qcgpu",
                algorithm: "grover",
                n,
                target,
                shots,
                found,
                time_ms: elapsed_ms,
                distribution: dist,
            };
            println!("{}", serde_json::to_string(&out).unwrap());
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "OpenCL not available or runtime error".to_string()
            };
            let err = ErrorOutput {
                framework: "qcgpu",
                algorithm: "grover",
                n,
                target,
                error: msg,
            };
            println!("{}", serde_json::to_string(&err).unwrap());
        }
    }

    std::process::exit(0);
}
