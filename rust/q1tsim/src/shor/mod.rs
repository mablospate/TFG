// Shor's algorithm for q1tsim 0.5 using a permutation-network approach.
//
// Mirrors the CUDA-Q Python reference in python/cudaq/shor/.
//
// Qubit layout:
//   [0 .. m)              : phase-estimation control register
//   [m .. m + w)          : target register |y> (mod N)
//   [m + w .. total)      : ancillas for the multi-controlled X ladder
//
// Only the control qubits are measured (the classical register has m bits).

pub mod classical;
pub mod permutation;
pub mod qft;

use std::error::Error;
use std::time::Instant;

use clap::Parser;
use num_integer::Integer;
use q1tsim::circuit::Circuit;
use q1tsim::error::Result as QResult;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use classical::{get_order_from_dist, mod_pow};
use num_integer::gcd;
use permutation::{build_mod_exp_permutation, controlled_swap_permutation};
use qft::apply_inverse_qft;

#[derive(Parser, Debug)]
#[command(name = "shor", about = "q1tsim Shor benchmark")]
pub struct Args {
    /// Number N to factor.
    #[arg(long = "N")]
    pub n: u64,

    /// Shots per order-finding attempt.
    #[arg(long, default_value_t = 10)]
    pub shots: usize,

    /// Maximum number of random base attempts.
    #[arg(long, default_value_t = 3)]
    pub tries: u32,

    /// Optional RNG seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,
}

#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,
    pub algorithm: &'static str,
    #[serde(rename = "N")]
    pub n: u64,
    pub factor: u64,
    pub time_ms: f64,
}

// ----- order-finding circuit -----

pub fn order_finding_circuit(a: u64, n_mod: u64, precision: usize) -> QResult<Circuit> {
    if gcd(a, n_mod) > 1 {
        return Err(q1tsim::error::Error::InternalError(format!(
            "gcd({}, {}) > 1: a and n_mod must be coprime",
            a, n_mod
        )));
    }
    let width = ((n_mod as f64).log2().ceil() as usize).max(1);
    let m = precision;
    // controlled_single_bit_transposition uses 1 (external ctrl) + (width - 1)
    // qubits of the target register as controls -> width total -> width - 2
    // ancillas. We allocate width - 1 to be safe for any rounding edge case.
    let n_anc = width.saturating_sub(1);
    let total = m + width + n_anc;

    let mut c = Circuit::new(total, m);
    let target_qubits: Vec<usize> = (m..m + width).collect();
    let ancillas: Vec<usize> = (m + width..total).collect();

    for i in 0..m {
        c.h(i)?;
    }
    c.x(target_qubits[0])?;

    for i in 0..m {
        let power: u64 = 1u64 << (m - 1 - i);
        let perm = build_mod_exp_permutation(a, n_mod, power);
        if perm.is_empty() {
            continue;
        }
        controlled_swap_permutation(&mut c, i, &target_qubits, &ancillas, &perm)?;
    }

    let ctrl_qubits: Vec<usize> = (0..m).collect();
    apply_inverse_qft(&mut c, &ctrl_qubits)?;

    for i in 0..m {
        c.measure(i, i)?;
    }
    Ok(c)
}

// ----- find_order / find_factor -----

pub fn find_order(
    a: u64,
    n_mod: u64,
    precision: Option<usize>,
    shots: usize,
) -> (u64, std::collections::HashMap<String, usize>) {
    if gcd(a, n_mod) > 1 {
        return (0, std::collections::HashMap::new());
    }
    let width = ((n_mod as f64).log2().ceil() as usize).max(1);
    let m = precision.unwrap_or(2 * width);
    let mut c = match order_finding_circuit(a, n_mod, m) {
        Ok(c) => c,
        Err(_) => return (0, std::collections::HashMap::new()),
    };
    if c.execute(shots).is_err() {
        return (0, std::collections::HashMap::new());
    }
    let dist = match c.histogram_string() {
        Ok(d) => d,
        Err(_) => return (0, std::collections::HashMap::new()),
    };
    let order = get_order_from_dist(&dist, a, n_mod, m);
    (order, dist)
}

pub fn find_factor(args: &Args) -> QResult<u64> {
    let n = args.n;
    if n % 2 == 0 {
        return Ok(2);
    }
    // Perfect-power test.
    let max_k = ((n as f64).log2().floor() as u32).max(2);
    for k in 2..=max_k {
        let d = (n as f64).powf(1.0 / k as f64).round() as u64;
        for cand in [d.saturating_sub(1), d, d + 1] {
            if cand < 2 {
                continue;
            }
            let mut p: u128 = 1;
            let mut ok = true;
            for _ in 0..k {
                p = match p.checked_mul(cand as u128) {
                    Some(v) => v,
                    None => {
                        ok = false;
                        break;
                    }
                };
                if p > n as u128 {
                    ok = false;
                    break;
                }
            }
            if ok && p == n as u128 {
                return Ok(cand);
            }
        }
    }

    let mut rng: StdRng = match args.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    for _ in 0..args.tries {
        let a: u64 = rng.gen_range(2..n);
        let g = a.gcd(&n);
        if g > 1 {
            return Ok(g);
        }
        let (r, _dist) = find_order(a, n, None, args.shots);
        if r == 0 || r % 2 != 0 {
            continue;
        }
        let half = mod_pow(a as u128, (r / 2) as u128, n as u128) as u64;
        if half == 0 {
            continue;
        }
        let d = half.wrapping_sub(1).gcd(&n);
        if d > 1 && d < n {
            return Ok(d);
        }
        let d2 = (half + 1).gcd(&n);
        if d2 > 1 && d2 < n {
            return Ok(d2);
        }
    }
    Ok(1)
}

pub fn run() -> ! {
    let args = Args::parse();
    let start = Instant::now();
    let factor = match find_factor(&args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();
    let out = Output {
        framework: "q1tsim",
        algorithm: "shor",
        n: args.n,
        factor,
        time_ms: elapsed.as_secs_f64() * 1000.0,
    };
    println!("{}", serde_json::to_string(&out).expect("serialize"));
    // suppress unused-import warning for Error
    let _ = std::any::type_name::<Box<dyn Error>>();
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_order_2_mod_15() {
        // Order of 2 mod 15 is 4 (2^4 = 16 == 1 mod 15).
        let (order, _dist) = find_order(2, 15, Some(8), 30);
        assert_eq!(order, 4, "Order of 2 mod 15 should be 4, got {}", order);
    }

    #[test]
    fn test_find_order_7_mod_15() {
        // Order of 7 mod 15 is 4 (7^2 = 49 == 4, 7^4 == 1 mod 15).
        let (order, _dist) = find_order(7, 15, Some(8), 30);
        assert_eq!(order, 4, "Order of 7 mod 15 should be 4, got {}", order);
    }

    // --- new tests matching qiskit coverage ---

    #[test]
    fn test_find_order() {
        // Canonical: find_order(2, 15, None, shots) returns (4, non-empty dist).
        let (order, dist) = find_order(2, 15, None, 30);
        assert_eq!(order, 4, "Order of 2 mod 15 should be 4, got {}", order);
        assert!(!dist.is_empty(), "distribution should be non-empty");
    }

    #[test]
    fn test_find_order_gcd_not_one() {
        // gcd(6, 15) = 3 > 1: should return (0, empty map).
        let (order, dist) = find_order(6, 15, None, 1);
        assert_eq!(order, 0, "Expected order 0 when gcd > 1, got {}", order);
        assert!(dist.is_empty(), "Expected empty distribution when gcd > 1");
    }

    #[test]
    fn test_order_finding_circuit_gcd_not_one() {
        // gcd(6, 15) = 3 > 1: circuit builder must return Err.
        let result = order_finding_circuit(6, 15, 4);
        assert!(result.is_err(), "Expected Err when gcd(a, n_mod) > 1");
    }

    #[test]
    fn test_find_factor_even() {
        // Even-N shortcut: should return 2 instantly without quantum work.
        let args = Args { n: 14, shots: 1, tries: 1, seed: Some(0) };
        assert_eq!(find_factor(&args).expect("find_factor failed"), 2);
    }

    #[test]
    fn test_find_factor_even_number() {
        // Even-N shortcut: should return 2 instantly without quantum work.
        let args = Args { n: 14, shots: 1, tries: 1, seed: Some(0) };
        assert_eq!(find_factor(&args).expect("find_factor failed"), 2);
    }

    #[test]
    fn test_find_factor_even_large() {
        let args = Args { n: 100, shots: 1, tries: 1, seed: Some(0) };
        assert_eq!(find_factor(&args).expect("find_factor failed"), 2);
    }

    #[test]
    fn test_find_factor_prime_power() {
        // 9 = 3^2: perfect-power shortcut returns 3.
        let args9 = Args { n: 9, shots: 1, tries: 1, seed: Some(0) };
        assert_eq!(find_factor(&args9).expect("find_factor failed"), 3);
        // 25 = 5^2: perfect-power shortcut returns 5.
        let args25 = Args { n: 25, shots: 1, tries: 1, seed: Some(0) };
        assert_eq!(find_factor(&args25).expect("find_factor failed"), 5);
    }

    #[test]
    fn test_find_factor_prime_power_9() {
        // 9 = 3^2: perfect-power shortcut returns 3.
        let args = Args { n: 9, shots: 1, tries: 1, seed: Some(0) };
        assert_eq!(find_factor(&args).expect("find_factor failed"), 3);
    }

    #[test]
    fn test_find_factor_prime_power_25() {
        // 25 = 5^2: perfect-power shortcut returns 5.
        let args = Args { n: 25, shots: 1, tries: 1, seed: Some(0) };
        assert_eq!(find_factor(&args).expect("find_factor failed"), 5);
    }

    #[test]
    #[ignore] // slow: full quantum Shor -- run with `cargo test -- --ignored`
    fn test_find_factor_multiple_tries() {
        // find_factor(15) should yield a factor in {3, 5}.
        let args = Args { n: 15, shots: 30, tries: 10, seed: Some(42) };
        let f = find_factor(&args).expect("find_factor failed");
        assert!(f == 3 || f == 5, "expected factor 3 or 5, got {}", f);
    }

    #[test]
    #[ignore] // slow: full quantum Shor -- run with `cargo test -- --ignored`
    fn test_find_factor_15() {
        let args = Args { n: 15, shots: 10, tries: 5, seed: Some(42) };
        let f = find_factor(&args).expect("find_factor failed");
        assert!(f == 3 || f == 5, "expected 3 or 5, got {}", f);
    }
}
