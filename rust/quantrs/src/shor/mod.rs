//! Shor's algorithm on top of `quantrs2`.
//!
//! Layout of the `Circuit<TOTAL>` register (qubit 0 = LSB):
//!   [0..m)                  - control register, written into via QFT-based PE
//!   [m..m+n)                - target register holding |y> for the order finder
//!   [m+n..m+n+(n-2))        - ancillas used to decompose multi-controlled X
//!
//! The Python Qiskit reference (`python/qiskit/shor/shor.py`) defines the
//! public API (`find_order` → `(u64, HashMap<String, usize>)`,
//! `find_factor` → `u64`).  The classical helpers live in `classical.rs`,
//! mirroring the Python module layout.

pub mod classical;
pub mod permutation;
pub mod qft;

use std::collections::HashMap;
use std::time::Instant;

use clap::Parser;
use num_integer::Integer;
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::statevector::StateVectorSimulator;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use classical::{get_order_from_dist, mod_pow};
use permutation::{apply_controlled_permutation, build_mod_exp_permutation};
use qft::inverse_qft;

#[derive(Parser, Debug)]
#[command(about = "Shor factoring on quantrs2")]
pub struct Args {
    #[arg(long = "N")]
    pub n: u64,
    #[arg(long, default_value_t = 10)]
    pub shots: usize,
    #[arg(long, default_value_t = 3)]
    pub tries: usize,
    #[arg(long)]
    pub seed: Option<u64>,
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
        let pid = std::process::id();
        if let Ok(out) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(kb) = s.trim().parse::<u64>() {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }
    0.0
}

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

/// Binary entry point. Parses CLI args, runs Shor, prints JSON, then exits.
pub fn run() -> ! {
    let args = Args::parse();
    eprintln!("Shor: factoring N={} (tries={}, shots={})", args.n, args.tries, args.shots);
    let start = Instant::now();
    let factor = find_factor(args.n, args.tries, args.shots, args.seed);
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mem_mb = peak_rss_mb();
    eprintln!("Shor: factor={} for N={} in {:.1}ms", factor, args.n, time_ms);
    let out = Output {
        framework: "quantrs2",
        framework_version: env!("CARGO_PKG_VERSION"),
        algorithm: "shor",
        n: args.n,
        factor,
        time_ms,
        mem_mb,
    };
    println!("{}", serde_json::to_string(&out).unwrap());
    std::process::exit(0);
}

// =============================================================================
// Classical body (mirrors python/qiskit/shor/shor.py:find_factor).
// =============================================================================

pub fn find_factor(n_val: u64, num_tries: usize, num_shots: usize, seed: Option<u64>) -> u64 {
    if n_val % 2 == 0 {
        return 2;
    }
    // Perfect power check.
    let max_k = (n_val as f64).log2().round() as u32;
    for k in 2..=max_k {
        let d = (n_val as f64).powf(1.0 / k as f64).round() as u64;
        if d > 1 {
            let mut p: u64 = 1;
            for _ in 0..k {
                p = p.saturating_mul(d);
                if p > n_val {
                    break;
                }
            }
            if p == n_val {
                return d;
            }
        }
    }

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(StdRng::from_entropy()),
    };

    let mut tries_done = 0;
    while tries_done < num_tries {
        let a = rng.gen_range(2..n_val);
        let d = a.gcd(&n_val);
        if d > 1 {
            return d;
        }
        let (r, _dist) = find_order(a, n_val, num_shots);
        tries_done += 1;
        if r == 0 {
            continue;
        }
        if r % 2 == 0 {
            let x = mod_pow(a, r / 2, n_val);
            if x > 1 {
                let d1 = (x - 1).gcd(&n_val);
                if d1 > 1 && d1 < n_val {
                    return d1;
                }
                let d2 = (x + 1).gcd(&n_val);
                if d2 > 1 && d2 < n_val {
                    return d2;
                }
            }
        }
    }
    1
}

// =============================================================================
// Runtime → compile-time dispatcher.
// =============================================================================

/// Find the multiplicative order of `a` in Z_N using quantum phase estimation.
///
/// Returns `(order, distribution)` where `distribution` maps measurement
/// bitstrings to shot counts (mirrors the `find_order` return type in
/// `python/qiskit/shor/shor.py`).  Returns `(0, HashMap::new())` when
/// `gcd(a, n_val) > 1`.
pub fn find_order(a: u64, n_val: u64, num_shots: usize) -> (u64, HashMap<String, usize>) {
    let n_bits = (n_val as f64).log2().ceil() as usize;
    let m = 2 * n_bits;
    // Ancillas to decompose MCX with up to n_bits controls.
    let anc = if n_bits >= 3 { n_bits - 2 } else { 0 };
    let total = m + n_bits + anc;

    match total {
        4 => order_finding::<4>(a, n_val, n_bits, m, num_shots),
        6 => order_finding::<6>(a, n_val, n_bits, m, num_shots),
        8 => order_finding::<8>(a, n_val, n_bits, m, num_shots),
        10 => order_finding::<10>(a, n_val, n_bits, m, num_shots),
        12 => order_finding::<12>(a, n_val, n_bits, m, num_shots),
        13 => order_finding::<13>(a, n_val, n_bits, m, num_shots),
        14 => order_finding::<14>(a, n_val, n_bits, m, num_shots),
        15 => order_finding::<15>(a, n_val, n_bits, m, num_shots),
        16 => order_finding::<16>(a, n_val, n_bits, m, num_shots),
        17 => order_finding::<17>(a, n_val, n_bits, m, num_shots),
        18 => order_finding::<18>(a, n_val, n_bits, m, num_shots),
        19 => order_finding::<19>(a, n_val, n_bits, m, num_shots),
        20 => order_finding::<20>(a, n_val, n_bits, m, num_shots),
        _ => panic!("unsupported total qubit count {total} (N={n_val})"),
    }
}

// =============================================================================
// Order-finding circuit.
// =============================================================================

fn order_finding<const TOTAL: usize>(
    a: u64,
    n_val: u64,
    n_bits: usize,
    m: usize,
    num_shots: usize,
) -> (u64, HashMap<String, usize>) {
    if a.gcd(&n_val) > 1 {
        return (0, HashMap::new());
    }

    let mut c: Circuit<TOTAL> = Circuit::new();

    // Hadamards on control register.
    for i in 0..m {
        c.h(QubitId::new(i as u32)).unwrap();
    }
    // Initialize target |y> = |1>.
    c.x(QubitId::new(m as u32)).unwrap();

    // Controlled modular multiplication ladder: for i in 0..m, the control
    // qubit i (MSB-first ordering matches Python:
    //     `ctrl_qubits[i]` controls `A^(2^(m-1-i))`)
    for i in 0..m {
        let power = 1u64 << (m - 1 - i);
        let a_power = mod_pow(a, power, n_val);
        if a_power == 1 {
            continue;
        }
        let perm = build_mod_exp_permutation(a_power, n_val);
        apply_controlled_permutation::<TOTAL>(&mut c, i, m, n_bits, &perm);
    }

    // Inverse QFT on the control register.
    inverse_qft(&mut c, 0, m);

    let sim = StateVectorSimulator::new();
    let reg = c.run(sim).expect("simulation failed");
    let probs = reg.probabilities();

    // Sample only the control register (qubits 0..m). Bitstring uses
    // ctrl_qubits[0] as the MSB so it matches the Python convention.
    let dist = sample_control_register::<TOTAL>(&probs, m, num_shots);

    let r = get_order_from_dist(&dist, a, n_val, m);
    (r, dist)
}

// =============================================================================
// Measurement sampling.
// =============================================================================

fn sample_control_register<const TOTAL: usize>(
    probs: &[f64],
    m: usize,
    num_shots: usize,
) -> HashMap<String, usize> {
    let mut rng = thread_rng();
    let dist = WeightedIndex::new(probs).expect("invalid probability distribution");
    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..num_shots {
        let idx = dist.sample(&mut rng);
        // Control register lives at qubits 0..m. Python uses ctrl_qubits[0] as MSB,
        // i.e. control qubit i contributes bit position (m-1-i). Reproduce here
        // by writing qubit 0 as the leftmost char.
        let mut bits = String::with_capacity(m);
        for q in 0..m {
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
    use classical::{mod_pow, reduce_to_min_order};

    // -------------------------------------------------------------------------
    // Classical helpers (re-tested here for integration coverage).
    // -------------------------------------------------------------------------

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 4, 15), 1);
        assert_eq!(mod_pow(7, 4, 15), 1);
        assert_eq!(mod_pow(2, 0, 15), 1);
        assert_eq!(mod_pow(2, 1, 15), 2);
    }

    #[test]
    fn test_reduce_to_min_order() {
        assert_eq!(reduce_to_min_order(8, 2, 15), 4);
        assert_eq!(reduce_to_min_order(12, 7, 15), 4);
    }

    // -------------------------------------------------------------------------
    // Permutation helpers.
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_mod_exp_permutation() {
        let a_power = mod_pow(2, 2, 5);
        assert_eq!(a_power, 4);
        let perm = build_mod_exp_permutation(a_power, 5);
        let pairs: std::collections::HashSet<(u64, u64)> = perm.iter().copied().collect();
        assert_eq!(pairs.len(), 4);
        assert!(pairs.contains(&(1, 4)));
        assert!(pairs.contains(&(4, 1)));
        assert!(pairs.contains(&(2, 3)));
        assert!(pairs.contains(&(3, 2)));
        assert!(!perm.iter().any(|&(y, _)| y == 0));
    }

    #[test]
    fn test_build_mod_exp_permutation_identity() {
        let perm = build_mod_exp_permutation(1, 7);
        assert!(perm.is_empty());
    }

    // -------------------------------------------------------------------------
    // find_factor — fast classical short-circuits.
    // -------------------------------------------------------------------------

    #[test]
    fn test_find_factor_even() {
        let f = find_factor(14, 1, 1, Some(0));
        assert_eq!(f, 2, "14 is even, factor should be 2");
    }

    #[test]
    fn test_find_factor_prime_power() {
        assert_eq!(find_factor(9, 1, 1, Some(0)), 3);
        assert_eq!(find_factor(25, 1, 1, Some(0)), 5);
        assert_eq!(find_factor(27, 1, 1, Some(0)), 3);
    }

    // -------------------------------------------------------------------------
    // find_order — gcd guard (no quantum circuit needed).
    // -------------------------------------------------------------------------

    /// Mirrors `test_find_order_gcd_not_one` in the qiskit test suite.
    #[test]
    fn test_find_order_gcd_not_one() {
        let (r, dist) = find_order(6, 15, 1);
        assert_eq!(r, 0, "gcd(6,15)=3 > 1, expected order 0, got {r}");
        assert!(dist.is_empty(), "expected empty distribution when gcd > 1");
    }

    /// The const-generic dispatcher should return (0, {}) when gcd(a,N) > 1.
    #[test]
    fn test_order_finding_circuit_gcd() {
        // gcd(3, 15) = 3 > 1 → early return from order_finding<…>
        let (r, dist) = find_order(3, 15, 1);
        assert_eq!(r, 0);
        assert!(dist.is_empty());
    }

    // -------------------------------------------------------------------------
    // Slow quantum tests (ignored by default).
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // slow: full quantum Shor
    fn test_find_factor_15() {
        let f = find_factor(15, 5, 20, Some(42));
        assert!(f == 3 || f == 5, "Expected 3 or 5, got {}", f);
    }

    #[test]
    #[ignore] // slow: full quantum Shor
    fn test_find_factor_multiple_tries() {
        let f = find_factor(15, 5, 20, Some(7));
        assert!(
            f == 3 || f == 5,
            "Expected a non-trivial factor of 15, got {f}"
        );
    }

    #[test]
    #[ignore] // slow: quantum order finding
    fn test_find_order() {
        let (r, dist) = find_order(2, 15, 20);
        assert_eq!(r, 4, "Order of 2 mod 15 = 4, got {r}");
        assert!(!dist.is_empty(), "distribution should be non-empty");
    }

    #[test]
    #[ignore] // slow: quantum order finding
    fn test_find_order_2_mod_15() {
        let (r, _dist) = find_order(2, 15, 20);
        assert_eq!(r, 4, "Order of 2 mod 15 = 4, got {}", r);
    }
}
