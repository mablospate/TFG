//! Shor's algorithm (order finding + classical post-processing) on quantr.
//!
//! Uses the **permutation-network** flavour of order finding (same as the
//! CUDA-Q reference in `python/cudaq/shor/`). The modular multiplication is
//! decomposed into disjoint cycles, each cycle into transpositions of basis
//! states that differ in exactly one bit, and each single-bit controlled
//! transposition into a multi-controlled X (with X-wraps for the 0-controls)
//! built from Toffoli ladders + ancillas.
//!
//! quantr is MSB-first: the bitstring returned by `ProductState::to_string()`
//! has qubit 0 as the leftmost character.
//!
//! Qubit layout:
//! - 0..m              : control register (qubit 0 is MSB of the phase).
//! - m..m+n            : target register. To keep the Python permutation
//!                       logic intact we map *logical* bit k of the target
//!                       to *physical* qubit `m + (n - 1 - k)`. That way
//!                       `target_qubits[0]` (logical, LSB) maps to the
//!                       rightmost physical qubit, just like Python does
//!                       with `tgt_qubits_lsb = list(reversed(tgt_qubits))`.
//! - m+n..              : ancilla register for the MCX ladder. The number
//!                       of ancillas is computed from the maximum control
//!                       count produced by `add_mcx`.

pub mod classical;
pub mod permutation;
pub mod qft;

use std::collections::HashMap;
use std::time::Instant;

use clap::Parser;
use num_integer::Integer;
use quantr::{Circuit, Measurement, QuantrError};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::shor::classical::{get_order_from_dist, mod_pow};
use crate::shor::permutation::{
    ancilla_count, build_mod_exp_permutation, controlled_swap_permutation,
};
use crate::shor::qft::apply_inverse_qft;

/// CLI arguments for the Shor binary.
#[derive(Parser, Debug)]
#[command(name = "shor", about = "Shor factoring on the quantr backend")]
pub struct Args {
    /// Composite integer to factor.
    #[arg(long = "N")]
    pub n_val: u64,
    /// Shots per order-finding trial.
    #[arg(long, default_value_t = 10)]
    pub shots: usize,
    /// Maximum number of order-finding attempts.
    #[arg(long, default_value_t = 3)]
    pub tries: usize,
    /// Optional RNG seed for the classical loop.
    #[arg(long)]
    pub seed: Option<u64>,
}

fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(kb) = line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()) {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }
    0.0
}

/// Per-run JSON record emitted on stdout.
#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,
    pub framework_version: &'static str,
    pub algorithm: &'static str,
    #[serde(rename = "N")]
    pub n_val: u64,
    pub factor: u64,
    pub time_ms: f64,
    pub mem_mb: f64,
}

/// Build the order-finding circuit for base `a` modulo `n_val` with `m` bits
/// of phase precision. Returns the assembled circuit along with `m`.
///
/// Returns an error early if `gcd(a, n_val) > 1` (mirrors the qiskit reference
/// which refuses to build the circuit in that case).
pub fn order_finding_circuit(
    a: u64,
    n_val: u64,
    precision: Option<usize>,
) -> Result<(Circuit, usize), QuantrError> {
    if a.gcd(&n_val) > 1 {
        // QuantrError's `message` field is `pub(crate)` upstream, so we can't
        // construct one directly. Produce one via the public API by asking
        // for a zero-qubit circuit, which upstream rejects with a QuantrError.
        let err = Circuit::new(0).err().expect(
            "Circuit::new(0) must return Err so the gcd>1 guard can propagate a QuantrError",
        );
        return Err(err);
    }

    let n_bits = (64 - (n_val - 1).leading_zeros()) as usize;
    let m = precision.unwrap_or(2 * n_bits);

    let n_anc = ancilla_count(n_bits);
    let total = m + n_bits + n_anc;

    // Target register: logical bit k <-> physical qubit `m + (n_bits - 1 - k)`.
    let target_qubits: Vec<usize> = (0..n_bits).map(|k| m + (n_bits - 1 - k)).collect();
    let ancillas: Vec<usize> = (m + n_bits..total).collect();

    let mut qc = Circuit::new(total)?;
    // Hadamards on the control register.
    let ctrl: Vec<usize> = (0..m).collect();
    qc.add_repeating_gate(quantr::Gate::H, &ctrl)?;
    // Initialise the target register to |1> (logical bit 0 set).
    qc.add_gate(quantr::Gate::X, target_qubits[0])?;

    // Controlled modular exponentiation. ctrl_qubits[0] is the MSB of the
    // measured phase, so it must drive A^(2^(m-1)).
    for i in 0..m {
        let power = 1u64 << (m - 1 - i);
        let perm = build_mod_exp_permutation(a, n_val, power);
        if !perm.is_empty() {
            controlled_swap_permutation(&mut qc, i, &target_qubits, &perm, &ancillas)?;
        }
    }

    apply_inverse_qft(&mut qc, 0, m)?;

    Ok((qc, m))
}

/// Run the order-finding circuit and pull the order out of the measurement
/// distribution.
///
/// Returns `(order, distribution)` where `distribution` is keyed by LSB-first
/// bitstrings of the control register (the same convention as the qiskit
/// reference). When `gcd(a, n_val) > 1` the circuit cannot be built and we
/// return `(0, empty)`.
pub fn find_order(a: u64, n_val: u64, shots: usize) -> (u64, HashMap<String, usize>) {
    let (qc, m) = match order_finding_circuit(a, n_val, None) {
        Ok(v) => v,
        Err(_) => return (0, HashMap::new()),
    };
    let sim = qc.simulate();
    let counts = match sim.measure_all(shots) {
        Measurement::Observable(c) => c,
        Measurement::NonObservable(c) => c,
    };
    // Reduce to the control register only (qubits 0..m). quantr's
    // `ProductState::to_string()` is MSB-first with qubit 0 as the leftmost
    // character, and in `order_finding_circuit` qubit 0 is the MSB of the
    // phase. So the first `m` characters of `raw` already form the phase
    // integer in standard binary (parseable directly with
    // `u64::from_str_radix(_, 2)`), matching qiskit's `int(bitstring, 2)`.
    let mut dist: HashMap<String, usize> = HashMap::new();
    for (state, count) in counts.into_iter() {
        let raw = state.to_string();
        let ctrl_bits: String = raw.chars().take(m).collect();
        *dist.entry(ctrl_bits).or_insert(0) += count;
    }
    let r = get_order_from_dist(&dist, a, n_val, m);
    (r, dist)
}

/// Classical wrapper that retries until a non-trivial factor of `n_val` is
/// discovered or `tries` attempts are exhausted.
pub fn find_factor(
    n_val: u64,
    shots: usize,
    tries: usize,
    seed: Option<u64>,
) -> Result<u64, QuantrError> {
    if n_val % 2 == 0 {
        return Ok(2);
    }
    // Quick perfect-power check.
    let max_k = ((n_val as f64).log2() as u64).max(2);
    for k in 2..=max_k {
        let root = (n_val as f64).powf(1.0 / k as f64).round() as u64;
        if root > 1 && root.pow(k as u32) == n_val {
            return Ok(root);
        }
    }

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    for _ in 0..tries {
        let a = rng.gen_range(2..n_val);
        let d = a.gcd(&n_val);
        if d > 1 {
            return Ok(d);
        }
        let (r, _dist) = find_order(a, n_val, shots);
        if r == 0 || r % 2 != 0 {
            continue;
        }
        let x = mod_pow(a, r / 2, n_val);
        if x == 0 {
            continue;
        }
        let candidate = (x + n_val - 1) % n_val;
        let factor = candidate.gcd(&n_val);
        if factor > 1 && factor < n_val {
            return Ok(factor);
        }
        let other = (x + 1).gcd(&n_val);
        if other > 1 && other < n_val {
            return Ok(other);
        }
    }
    Ok(1)
}

/// Entry point used by the thin `bin/shor.rs` wrapper.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    eprintln!("Shor: factoring N={} (tries={}, shots={})", args.n_val, args.tries, args.shots);
    let start = Instant::now();
    let factor = find_factor(args.n_val, args.shots, args.tries, args.seed)?;
    let time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mem_mb = peak_rss_mb();
    eprintln!("Shor: factor={} for N={} in {:.1}ms", factor, args.n_val, time_ms);
    let out = Output {
        framework: "quantr",
        framework_version: env!("CARGO_PKG_VERSION"),
        algorithm: "shor",
        n_val: args.n_val,
        factor,
        time_ms,
        mem_mb,
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shor::permutation::build_mod_exp_permutation;

    #[test]
    fn test_build_mod_exp_permutation() {
        // 4 * y mod 5 for y in [0, 5):
        // y=0 -> 0 (identity, omitted)
        // y=1 -> 4
        // y=2 -> 3
        // y=3 -> 2
        // y=4 -> 1
        let perm = build_mod_exp_permutation(2, 5, 2);
        assert_eq!(perm.get(&1), Some(&4));
        assert_eq!(perm.get(&2), Some(&3));
        assert_eq!(perm.get(&3), Some(&2));
        assert_eq!(perm.get(&4), Some(&1));
        // y=0 is the identity and must be omitted from the permutation map.
        assert!(!perm.contains_key(&0));
    }

    #[test]
    fn test_order_finding_circuit_gcd_not_one() {
        // gcd(6, 15) = 3, so the circuit cannot be built.
        let res = order_finding_circuit(6, 15, None);
        assert!(res.is_err(), "expected Err when gcd(a, n_val) > 1");
    }

    #[test]
    fn test_find_order() {
        // 2^4 = 16 = 1 mod 15, so the order of 2 mod 15 is 4.
        let (order, dist) = find_order(2, 15, 30);
        assert_eq!(order, 4, "expected order 4 for 2 mod 15, got {}", order);
        assert!(!dist.is_empty(), "distribution must be non-empty");
    }

    #[test]
    fn test_find_order_gcd_not_one() {
        // gcd(6, 15) > 1: find_order must short-circuit to (0, empty).
        let (order, dist) = find_order(6, 15, 1);
        assert_eq!(order, 0);
        assert!(dist.is_empty(), "distribution must be empty when gcd > 1");
    }

    #[test]
    fn test_find_factor_even() {
        // Even N: the classical shortcut returns 2 immediately.
        let f = find_factor(14, 1, 1, Some(0)).expect("find_factor failed");
        assert_eq!(f, 2);
    }

    #[test]
    fn test_find_factor_prime_power() {
        // Perfect-power shortcut: 9 = 3^2, 25 = 5^2.
        let f = find_factor(9, 1, 1, Some(0)).expect("find_factor failed");
        assert_eq!(f, 3);
        let f = find_factor(25, 1, 1, Some(0)).expect("find_factor failed");
        assert_eq!(f, 5);
    }

    #[test]
    #[ignore]
    fn test_find_factor_multiple_tries() {
        // Slow: runs the full quantum order-finding circuit. Marked ignored so
        // it is only exercised via `cargo test -- --ignored`.
        let f = find_factor(15, 5, 3, Some(42)).expect("find_factor failed");
        assert!(f == 3 || f == 5, "Expected factor 3 or 5, got {}", f);
    }
}
