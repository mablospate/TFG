//! Grover's algorithm for the quantr Rust crate.
//!
//! quantr is MSB-first: in `ProductState::to_string()`, qubit 0 is the leftmost
//! character. We reverse measured bitstrings to match the LSB-first convention
//! used by the rest of the benchmark suite.
//!
//! MCZ decomposition: quantr exposes only `Gate::Toffoli(c1, c2)` (2 controls).
//! For n >= 3 controls we use the ancilla + Toffoli-ladder strategy (Strategy B
//! in implementation_guide.md section 7.10) wrapped with H on the target to
//! upgrade MCX -> MCZ.

use std::collections::HashMap;
use std::time::Instant;

use clap::Parser;
use quantr::{Circuit, Gate, Measurement, QuantrError};
use serde::Serialize;

/// CLI arguments for the Grover binary.
#[derive(Parser, Debug)]
#[command(name = "grover", about = "Grover search on the quantr backend")]
pub struct Args {
    /// Number of search qubits.
    #[arg(long)]
    pub n: usize,
    /// Target basis state encoded as a non-negative integer.
    #[arg(long)]
    pub target: u64,
    /// Number of measurement shots.
    #[arg(long, default_value_t = 1024)]
    pub shots: usize,
    /// Override the number of Grover iterations.
    #[arg(long)]
    pub iterations: Option<usize>,
}

/// Per-run JSON record emitted on stdout.
#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,
    pub framework_version: &'static str,
    pub algorithm: &'static str,
    pub n: usize,
    pub target: u64,
    pub shots: usize,
    pub iterations: usize,
    pub found: u64,
    pub time_ms: f64,
    pub mem_mb: f64,
    pub distribution: HashMap<String, usize>,
}

/// Multi-controlled X gate using the ancilla + Toffoli-ladder decomposition.
///
/// Controls live at the indices in `controls` (length n_ctrl) and the target
/// at `target`. For n_ctrl >= 3 we expect `ancillas` to contain n_ctrl - 2
/// indices that are guaranteed to be in state |0>. The ladder is uncomputed
/// after the central gate so ancillas are returned to |0>.
pub fn add_mcx(
    qc: &mut Circuit,
    controls: &[usize],
    target: usize,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let k = controls.len();
    match k {
        0 => {
            qc.add_gate(Gate::X, target)?;
        }
        1 => {
            qc.add_gate(Gate::CNot(controls[0]), target)?;
        }
        2 => {
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), target)?;
        }
        _ => {
            assert!(
                ancillas.len() >= k - 2,
                "MCX ladder needs k-2 ancillas (k={}, given {})",
                k,
                ancillas.len()
            );
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

/// Multi-controlled Z on `qubits[0..n]` implemented as H(target) MCX H(target).
pub fn add_mcz(qc: &mut Circuit, n: usize, ancillas: &[usize]) -> Result<(), QuantrError> {
    match n {
        0 => {}
        1 => {
            qc.add_gate(Gate::Z, 0)?;
        }
        2 => {
            qc.add_gate(Gate::CZ(0), 1)?;
        }
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

/// Apply the Grover oracle that flips the phase of |target> on the first n qubits.
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
    add_mcz(qc, n, ancillas)?;
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            qc.add_gate(Gate::X, i)?;
        }
    }
    Ok(())
}

/// Apply the Grover diffusion operator (inversion about the mean) on n qubits.
pub fn build_diffuser(qc: &mut Circuit, n: usize, ancillas: &[usize]) -> Result<(), QuantrError> {
    let qs: Vec<usize> = (0..n).collect();
    qc.add_repeating_gate(Gate::H, &qs)?;
    qc.add_repeating_gate(Gate::X, &qs)?;
    add_mcz(qc, n, ancillas)?;
    qc.add_repeating_gate(Gate::X, &qs)?;
    qc.add_repeating_gate(Gate::H, &qs)?;
    Ok(())
}

/// Build the full Grover circuit. The first n qubits are the search register;
/// any additional qubits are ancillas used by the MCZ decomposition.
pub fn grover_circuit(
    n: usize,
    target: u64,
    num_iterations: Option<usize>,
) -> Result<Circuit, QuantrError> {
    assert!(n >= 1, "n must be >= 1");
    assert!(target < (1u64 << n), "target out of range for n qubits");

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

fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmHWM:") {
                if let Some(kb) = line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()) {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        // getrusage(RUSAGE_SELF).ru_maxrss = peak RSS histórico (bytes en macOS)
        #[repr(C)]
        struct MinRusage {
            _utime: [u8; 16],
            _stime: [u8; 16],
            ru_maxrss: i64,
            _rest: [u8; 104],
        }
        extern "C" {
            fn getrusage(who: i32, usage: *mut MinRusage) -> i32;
        }
        let mut usage = unsafe { std::mem::zeroed::<MinRusage>() };
        if unsafe { getrusage(0, &mut usage) } == 0 {
            return usage.ru_maxrss as f64 / (1024.0 * 1024.0);
        }
    }
    0.0
}

/// Entry point used by the thin `bin/grover.rs` wrapper.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let iterations = args.iterations.unwrap_or_else(|| {
        let n_states = (1u64 << args.n) as f64;
        ((std::f64::consts::PI / 4.0) * n_states.sqrt()).floor() as usize
    });
    eprintln!(
        "Start Grover search for |{}> in {}-qubit space ({} iterations)",
        args.target, args.n, iterations
    );
    let start = Instant::now();
    let qc = grover_circuit(args.n, args.target, Some(iterations))?;
    let sim = qc.simulate();
    let counts = match sim.measure_all(args.shots) {
        Measurement::Observable(c) => c,
        Measurement::NonObservable(c) => c,
    };
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // quantr is MSB-first; the search register lives on qubits 0..n.
    // Build the LSB-first bitstring for the search register only (ancillas
    // are guaranteed to be back in |0> and are dropped from the distribution).
    let mut distribution: HashMap<String, usize> = HashMap::new();
    for (state, count) in counts.into_iter() {
        let raw = state.to_string();
        let search_msb: String = raw.chars().take(args.n).collect();
        let search_lsb: String = search_msb.chars().rev().collect();
        *distribution.entry(search_lsb).or_insert(0) += count;
    }

    let (best, _) = distribution
        .iter()
        .max_by_key(|(_, c)| *c)
        .ok_or("no measurement outcomes recorded")?;
    let found = u64::from_str_radix(best, 2)?;
    let mem_mb = peak_rss_mb();

    let total_shots: usize = distribution.values().sum();
    let target_bs = format!("{:0width$b}", args.target, width = args.n);
    let prob = distribution.get(&target_bs).copied().unwrap_or(0) as f64 / total_shots.max(1) as f64;
    eprintln!("Found target state |{}> with probability {:.2}%", found, prob * 100.0);

    let out = Output {
        framework: "quantr",
        framework_version: env!("CARGO_PKG_VERSION"),
        algorithm: "grover",
        n: args.n,
        target: args.target,
        shots: args.shots,
        iterations,
        found,
        time_ms: elapsed,
        mem_mb,
        distribution,
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate the circuit produced by `grover_circuit` and reduce the result
    /// to an LSB-first distribution over the n-qubit search register. This
    /// replicates the same post-processing that `main()` performs.
    fn run_grover(n: usize, target: u64, iterations: Option<usize>, shots: usize) -> HashMap<String, usize> {
        let qc = grover_circuit(n, target, iterations).expect("grover_circuit failed");
        let sim = qc.simulate();
        let counts = match sim.measure_all(shots) {
            Measurement::Observable(c) => c,
            Measurement::NonObservable(c) => c,
        };
        let mut distribution: HashMap<String, usize> = HashMap::new();
        for (state, count) in counts.into_iter() {
            let raw = state.to_string();
            let search_msb: String = raw.chars().take(n).collect();
            let search_lsb: String = search_msb.chars().rev().collect();
            *distribution.entry(search_lsb).or_insert(0) += count;
        }
        distribution
    }

    fn most_frequent(dist: &HashMap<String, usize>) -> u64 {
        let (best, _) = dist
            .iter()
            .max_by_key(|(_, c)| *c)
            .expect("distribution should not be empty");
        u64::from_str_radix(best, 2).expect("best bitstring is binary")
    }

    #[test]
    fn test_grover_n3_target5() {
        let dist = run_grover(3, 5, None, 1024);
        assert_eq!(most_frequent(&dist), 5, "expected most frequent state to be 5, dist = {:?}", dist);
    }

    #[test]
    fn test_grover_n3_target0() {
        let dist = run_grover(3, 0, None, 1024);
        assert_eq!(most_frequent(&dist), 0, "expected most frequent state to be 0, dist = {:?}", dist);
    }

    #[test]
    fn test_grover_n4_target11() {
        let dist = run_grover(4, 11, None, 1024);
        assert_eq!(most_frequent(&dist), 11, "expected most frequent state to be 11, dist = {:?}", dist);
    }

    #[test]
    fn test_grover_explicit_iterations() {
        let dist = run_grover(3, 5, Some(2), 1024);
        assert_eq!(
            most_frequent(&dist),
            5,
            "expected most frequent state to be 5 even with 2 iterations, dist = {:?}",
            dist
        );
    }
}
