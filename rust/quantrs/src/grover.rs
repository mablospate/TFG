//! Grover's algorithm implemented on top of `quantrs2`.
//!
//! `quantrs2` uses const-generic `Circuit<N>`, so a runtime-dispatched search
//! must enumerate every supported qubit count at compile time. We dispatch on
//! the search-space width `n` (3..=8). For `n >= 4` the circuit allocates
//! `n - 2` ancilla qubits to implement the multi-controlled Z required by the
//! oracle and the diffuser via a Toffoli ladder.

use std::collections::HashMap;
use std::time::Instant;

use clap::Parser;
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::statevector::StateVectorSimulator;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(about = "Grover search on quantrs2")]
pub struct Args {
    #[arg(long)]
    pub n: usize,
    #[arg(long)]
    pub target: u64,
    #[arg(long, default_value_t = 1024)]
    pub shots: u32,
    #[arg(long)]
    pub iterations: Option<usize>,
}

#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,
    pub framework_version: &'static str,
    pub algorithm: &'static str,
    pub n: usize,
    pub target: u64,
    pub shots: u32,
    pub iterations: usize,
    pub found: u64,
    pub time_ms: f64,
    pub mem_mb: f64,
    pub distribution: HashMap<String, usize>,
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
    0.0
}

/// Binary entry point. Parses CLI args, runs Grover, prints JSON, then exits.
pub fn run() -> ! {
    let args = Args::parse();
    let num_iter = args.iterations.unwrap_or_else(|| {
        let space = (1u64 << args.n) as f64;
        ((std::f64::consts::PI / 4.0) * space.sqrt()).floor() as usize
    });
    eprintln!(
        "Start Grover search for |{}> in {}-qubit space ({} iterations)",
        args.target, args.n, num_iter
    );
    let start = Instant::now();
    let (found, dist) = run_grover(args.n, args.target, args.shots, Some(num_iter));
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mem_mb = peak_rss_mb();

    let total_shots: usize = dist.values().sum();
    let target_bs = format!("{:0width$b}", args.target, width = args.n);
    let prob = dist.get(&target_bs).copied().unwrap_or(0) as f64 / total_shots.max(1) as f64;
    eprintln!("Found target state |{}> with probability {:.2}%", found, prob * 100.0);

    let out = Output {
        framework: "quantrs2",
        framework_version: env!("CARGO_PKG_VERSION"),
        algorithm: "grover",
        n: args.n,
        target: args.target,
        shots: args.shots,
        iterations: num_iter,
        found,
        time_ms,
        mem_mb,
        distribution: dist,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
    std::process::exit(0);
}

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

fn grover_impl<const TOTAL: usize>(
    n: usize,
    target: u64,
    shots: u32,
    iterations: Option<usize>,
) -> (u64, HashMap<String, usize>) {
    assert!(n <= TOTAL, "n must fit in TOTAL qubits");
    let num_iter = iterations.unwrap_or_else(|| {
        let space = (1u64 << n) as f64;
        ((std::f64::consts::PI / 4.0) * space.sqrt()).floor() as usize
    });

    let mut c: Circuit<TOTAL> = Circuit::new();

    // Superposition over the n search qubits.
    for i in 0..n {
        c.h(QubitId::new(i as u32)).unwrap();
    }

    for _ in 0..num_iter {
        // Oracle: flip phase of |target>.
        apply_oracle(&mut c, n, target);
        // Diffuser: 2|s><s| - I.
        apply_diffuser(&mut c, n);
    }

    let sim = StateVectorSimulator::new();
    let reg = c.run(sim).expect("simulation failed");
    let probs = reg.probabilities();

    let dist = sample_search_qubits::<TOTAL>(&probs, n, shots);

    let (found_bs, _) = dist
        .iter()
        .max_by_key(|(_, count)| *count)
        .expect("empty distribution");
    // The bitstring is built with q0 as the rightmost char (Qiskit-style).
    let found = u64::from_str_radix(found_bs, 2).unwrap();

    (found, dist)
}

fn apply_oracle<const TOTAL: usize>(c: &mut Circuit<TOTAL>, n: usize, target: u64) {
    // X gates where target has 0 bits, so |target> maps to |11..1>.
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(QubitId::new(i as u32)).unwrap();
        }
    }
    apply_mcz(c, n);
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(QubitId::new(i as u32)).unwrap();
        }
    }
}

fn apply_diffuser<const TOTAL: usize>(c: &mut Circuit<TOTAL>, n: usize) {
    for i in 0..n {
        c.h(QubitId::new(i as u32)).unwrap();
    }
    for i in 0..n {
        c.x(QubitId::new(i as u32)).unwrap();
    }
    apply_mcz(c, n);
    for i in 0..n {
        c.x(QubitId::new(i as u32)).unwrap();
    }
    for i in 0..n {
        c.h(QubitId::new(i as u32)).unwrap();
    }
}

/// Multi-controlled Z on the first n qubits. Uses ancillas at positions n..2n-2
/// for n >= 4. Ancillas are returned to |0> by reversing the Toffoli ladder.
fn apply_mcz<const TOTAL: usize>(c: &mut Circuit<TOTAL>, n: usize) {
    match n {
        0 => {}
        1 => {
            c.z(QubitId::new(0)).unwrap();
        }
        2 => {
            c.cz(QubitId::new(0), QubitId::new(1)).unwrap();
        }
        3 => {
            // H + Toffoli + H == CCZ.
            c.h(QubitId::new(2)).unwrap();
            c.toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2))
                .unwrap();
            c.h(QubitId::new(2)).unwrap();
        }
        _ => {
            // Toffoli ladder using n-2 ancilla qubits at indices n..2n-2.
            // Final ladder output sits at qubit (2n - 3). Convert it to a Z on
            // target qubit (n - 1) by sandwiching with Hadamards on (n - 1).
            let num_anc = n - 2;
            let anc_base = n;
            let target_q = n - 1;
            assert!(anc_base + num_anc <= TOTAL, "not enough qubits for ancillas");

            // Compute ladder forward: ancilla[k] = AND(controls[..k+2]).
            c.toffoli(
                QubitId::new(0),
                QubitId::new(1),
                QubitId::new(anc_base as u32),
            )
            .unwrap();
            for k in 1..num_anc {
                c.toffoli(
                    QubitId::new((k + 1) as u32),
                    QubitId::new((anc_base + k - 1) as u32),
                    QubitId::new((anc_base + k) as u32),
                )
                .unwrap();
            }
            // CZ-equivalent on (last_anc, target_q) via H + CNOT + H.
            let last_anc = anc_base + num_anc - 1;
            c.h(QubitId::new(target_q as u32)).unwrap();
            c.cnot(
                QubitId::new(last_anc as u32),
                QubitId::new(target_q as u32),
            )
            .unwrap();
            c.h(QubitId::new(target_q as u32)).unwrap();
            // Uncompute the ladder in reverse to restore ancillas to |0>.
            for k in (1..num_anc).rev() {
                c.toffoli(
                    QubitId::new((k + 1) as u32),
                    QubitId::new((anc_base + k - 1) as u32),
                    QubitId::new((anc_base + k) as u32),
                )
                .unwrap();
            }
            c.toffoli(
                QubitId::new(0),
                QubitId::new(1),
                QubitId::new(anc_base as u32),
            )
            .unwrap();
        }
    }
}

/// Sample `shots` bitstrings from the full-state probability distribution and
/// keep only the first `n` qubits (the search register). Qubit 0 is the LSB:
/// the bitstring is written with q0 at the rightmost position so it matches the
/// Qiskit / Python-CUDAQ convention used elsewhere in the project.
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
            // q is the qubit index; bit position in idx is q (qubit 0 = LSB).
            let b = (idx >> q) & 1;
            bits.push(if b == 1 { '1' } else { '0' });
        }
        *counts.entry(bits).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover_finds_target_n3() {
        let (found, dist) = run_grover(3, 5, 200, None);
        assert_eq!(found, 5, "Expected to find target 5");
        let count = dist.get("101").copied().unwrap_or(0);
        assert!(count > 100, "Expected >100/200 shots to be '101', got {}", count);
    }

    #[test]
    fn test_grover_finds_target_zero() {
        let (found, dist) = run_grover(3, 0, 200, None);
        assert_eq!(found, 0);
        let count = dist.get("000").copied().unwrap_or(0);
        assert!(count > 100, "Expected '000' to dominate, got {}", count);
    }

    #[test]
    fn test_grover_finds_target_n4() {
        let (found, dist) = run_grover(4, 11, 200, None);
        assert_eq!(found, 11);
        let count = dist.get("1011").copied().unwrap_or(0);
        assert!(count > 100, "Expected '1011' to dominate, got {}", count);
    }

    #[test]
    fn test_grover_explicit_iterations() {
        // 2 is the optimal iteration count for n=3 (~ (π/4)·√8).
        let (found, _) = run_grover(3, 5, 100, Some(2));
        assert_eq!(found, 5);
    }
}
