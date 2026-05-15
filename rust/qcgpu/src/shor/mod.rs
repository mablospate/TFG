//! Shor's algorithm on the qcgpu OpenCL-based simulator.
//!
//! Strategy: phase-estimation order finding with a permutation-network
//! implementation of controlled modular exponentiation (no quantum arithmetic
//! circuit — the unitary is realized as a controlled permutation of the
//! computational basis on the target register).
//!
//! The QFT is implemented manually using qcgpu's `r(angle: f32)` phase gate,
//! since qcgpu has no built-in QFT. All angles are `f32`, which means accuracy
//! degrades for large `m`; for N = 15 with m = 8 the accuracy is sufficient.

pub mod classical;
pub mod permutation;
pub mod qft;

use std::collections::HashMap;
use std::panic;
use std::time::Instant;

use clap::Parser;
use num_integer::Integer;
use qcgpu::State;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

pub use classical::{get_order_from_dist, limit_denominator, mod_pow, reduce_to_min_order};
pub use permutation::{
    build_mod_exp_permutation, controlled_single_bit_transposition, controlled_swap_permutation,
    controlled_transposition, mcx,
};
pub use qft::apply_inverse_qft;

#[derive(Parser, Debug)]
#[command(name = "shor", about = "Shor's factoring on qcgpu (OpenCL)")]
pub struct Args {
    /// Composite integer to factor.
    #[arg(long = "N")]
    pub n: u64,

    /// Shots per order-finding trial.
    #[arg(long, default_value_t = 10)]
    pub shots: usize,

    /// Maximum number of base-`a` trials.
    #[arg(long, default_value_t = 3)]
    pub tries: usize,

    /// Optional RNG seed for the classical base selection.
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

#[derive(Serialize)]
pub struct ErrorOutput {
    pub framework: &'static str,
    pub algorithm: &'static str,
    #[serde(rename = "N")]
    pub n: u64,
    pub error: String,
}

// ---------------------------------------------------------------------------
// Modular arithmetic helpers (kept in mod.rs for backward compat)
// ---------------------------------------------------------------------------

/// Number of bits needed to represent values 0..N.
pub fn bit_width(n: u64) -> u32 {
    let mut w = 0u32;
    let mut x = n - 1;
    while x > 0 {
        w += 1;
        x >>= 1;
    }
    w.max(1)
}

// ---------------------------------------------------------------------------
// Order finding
// ---------------------------------------------------------------------------

/// Build the order-finding state for the pair (a, N). Returns the final
/// statevector ready for measurement of the m control qubits.
///
/// Returns `None` if gcd(a, n_val) > 1 (no quantum work needed).
pub fn order_finding_state(a: u64, n_val: u64, precision: usize) -> Option<(State, usize)> {
    if a.gcd(&n_val) > 1 {
        return None;
    }

    let width = bit_width(n_val) as usize;
    let m = precision;

    // Phase-estimation control register: m qubits at indices 0..m.
    // Target register: width qubits at indices m..m+width.
    // Ancillas (for multi-control X decomposition): we need at most
    // (1 + (width - 1)) - 2 = width - 2 ancillas for transpositions, where the
    // external phase-estimation control adds 1 to the control count. Use
    // `width` ancillas to be safe (covers single-bit transpositions which
    // have at most `width` controls including the external one).
    let n_anc = width.max(2);
    let total = m + width + n_anc;

    let mut state = State::new(total as u32, 0);

    let ctrl_qubits: Vec<i32> = (0..m as i32).collect();
    let tgt_qubits: Vec<i32> = (m as i32..(m + width) as i32).collect();
    let ancillas: Vec<i32> = ((m + width) as i32..total as i32).collect();

    // Hadamard on control register.
    for &q in &ctrl_qubits {
        state.h(q);
    }

    // Initialize target register to |1>: set qubit 0 of the target register.
    state.x(tgt_qubits[0]);

    // Controlled modular exponentiation: ctrl_qubits[0] is the MSB of the
    // output phase bitstring (after the IQFT swap), so it controls the
    // largest power A^(2^(m-1)).
    for i in 0..m {
        let power = 1u64 << (m - 1 - i);
        let perm = build_mod_exp_permutation(a, n_val, power);
        if !perm.is_empty() {
            controlled_swap_permutation(
                &mut state,
                ctrl_qubits[i],
                &tgt_qubits,
                &perm,
                &ancillas,
            );
        }
    }

    apply_inverse_qft(&mut state, &ctrl_qubits);

    Some((state, m))
}

/// Run the order-finding circuit for (a, N) and return `(order, distribution)`.
///
/// * `precision` — number of QPE control qubits. If `None`, defaults to
///   `2 * ceil(log2(N))`, matching the Python qiskit reference.
/// * Returns `(0, HashMap::new())` when `gcd(a, N) > 1` or no order is found.
pub fn find_order(
    a: u64,
    n_val: u64,
    precision: Option<usize>,
    shots: usize,
) -> (u64, HashMap<String, usize>) {
    // Compute default precision: 2 * ceil(log2(N)).
    let m = precision.unwrap_or_else(|| 2 * (u64::BITS - n_val.leading_zeros()) as usize);

    let (mut state, m) = match order_finding_state(a, n_val, m) {
        Some(s) => s,
        None => return (0, HashMap::new()),
    };

    let raw = state.measure_many(shots as i32);

    // Keep only the m control qubits (rightmost m chars: qubit 0 is rightmost).
    let mut dist: HashMap<String, usize> = HashMap::new();
    for (bs, count) in raw {
        let total = bs.len();
        let start = total.saturating_sub(m);
        let key = bs[start..].to_string();
        *dist.entry(key).or_insert(0) += count as usize;
    }

    let r = get_order_from_dist(&dist, a, n_val, m);
    (r, dist)
}

// ---------------------------------------------------------------------------
// Classical factoring driver
// ---------------------------------------------------------------------------

pub fn integer_kth_root(n: u64, k: u32) -> u64 {
    if k == 0 {
        return 1;
    }
    if k == 1 {
        return n;
    }
    let approx = (n as f64).powf(1.0 / k as f64).round() as u64;
    for cand in [approx.saturating_sub(1), approx, approx + 1] {
        if cand == 0 {
            continue;
        }
        if let Some(p) = checked_pow(cand, k) {
            if p == n {
                return cand;
            }
        }
    }
    0
}

pub fn checked_pow(base: u64, exp: u32) -> Option<u64> {
    let mut result: u64 = 1;
    for _ in 0..exp {
        result = result.checked_mul(base)?;
    }
    Some(result)
}

pub fn find_factor(n_val: u64, tries: usize, shots: usize, seed: Option<u64>) -> u64 {
    if n_val.is_even() {
        return 2;
    }

    // Test for perfect powers.
    let bits = (n_val as f64).log2().ceil() as u32;
    for k in 2..=bits {
        let d = integer_kth_root(n_val, k);
        if d > 1 {
            if let Some(p) = checked_pow(d, k) {
                if p == n_val {
                    return d;
                }
            }
        }
    }

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    for _ in 0..tries {
        let a = rng.gen_range(2..n_val);
        let g = a.gcd(&n_val);
        if g > 1 {
            return g;
        }
        // Precision defaults to None (2 * ceil(log2(N)) computed inside find_order).
        let (r, _dist) = find_order(a, n_val, None, shots);
        if r != 0 && r % 2 == 0 {
            let half = mod_pow(a, r / 2, n_val);
            if half != n_val - 1 {
                let x = (half + n_val - 1) % n_val;
                let d = (x as u64).gcd(&n_val);
                if d > 1 && d < n_val {
                    return d;
                }
            }
        }
    }

    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;

    // ---- Pure-Rust tests (no OpenCL required) ----

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 4, 15), 1);
        assert_eq!(mod_pow(7, 4, 15), 1);
        assert_eq!(mod_pow(2, 0, 15), 1);
        assert_eq!(mod_pow(2, 1, 15), 2);
        assert_eq!(mod_pow(3, 5, 7), 5);
        assert_eq!(mod_pow(5, 0, 1), 0);
    }

    #[test]
    fn test_build_mod_exp_permutation() {
        // 4 * x mod 5: 1->4, 2->3, 3->2, 4->1 (0 is a fixed point, not in map).
        let perm = build_mod_exp_permutation(2, 5, 2);
        assert_eq!(perm[&1], 4);
        assert_eq!(perm[&2], 3);
        assert_eq!(perm[&3], 2);
        assert_eq!(perm[&4], 1);
    }

    #[test]
    fn test_build_mod_exp_permutation_identity() {
        // power=0 ⇒ A=1 ⇒ permutation is identity ⇒ no non-fixed entries.
        let perm = build_mod_exp_permutation(2, 5, 0);
        for x in 1..5u64 {
            assert_eq!(perm.get(&x).copied().unwrap_or(x), x);
        }
    }

    #[test]
    fn test_limit_denominator_below_bound() {
        // If denom already <= max_denom, the fraction is returned unchanged.
        let (n, d) = limit_denominator(1, 4, 15);
        assert_eq!((n, d), (1, 4));
    }

    #[test]
    fn test_limit_denominator_approximation() {
        // 3/8 with cap 4 → best convergent has denominator <= 4.
        let (_, d) = limit_denominator(3, 8, 4);
        assert!(d <= 4, "denominator {d} exceeds cap 4");
    }

    #[test]
    fn test_limit_denominator_i128() {
        // Verify i128 version handles a large-N scenario without overflow.
        // x = 2^15 = 32768 out of 2^30 = 1073741824, max_denom = 14.
        // Should recover 1/... (some fraction with denom <= 14).
        let (_, d) = limit_denominator(32768, 1_073_741_824, 14);
        assert!(d <= 14, "denominator {d} exceeds cap 14");
        assert!(d >= 1);
    }

    #[test]
    fn test_bit_width() {
        assert_eq!(bit_width(2), 1);
        assert_eq!(bit_width(15), 4);
        assert_eq!(bit_width(16), 4);
        assert_eq!(bit_width(17), 5);
    }

    #[test]
    fn test_integer_kth_root() {
        assert_eq!(integer_kth_root(9, 2), 3);
        assert_eq!(integer_kth_root(27, 3), 3);
        assert_eq!(integer_kth_root(25, 2), 5);
        assert_eq!(integer_kth_root(10, 2), 0); // not a perfect square
    }

    // ---- Classical shortcuts in find_factor (no OpenCL needed) ----

    #[test]
    fn test_find_factor_even() {
        assert_eq!(find_factor(14, 1, 0, Some(0)), 2);
        assert_eq!(find_factor(100, 1, 0, Some(0)), 2);
    }

    #[test]
    fn test_find_factor_prime_power() {
        assert_eq!(find_factor(9, 1, 0, Some(0)), 3);
        assert_eq!(find_factor(25, 1, 0, Some(0)), 5);
        assert_eq!(find_factor(27, 1, 0, Some(0)), 3);
    }

    #[test]
    fn test_reduce_to_min_order() {
        assert_eq!(reduce_to_min_order(8, 2, 15), 4);
        assert_eq!(reduce_to_min_order(12, 7, 15), 4);
        assert_eq!(reduce_to_min_order(4, 2, 15), 4);
    }

    // ---- New API tests ----

    /// gcd guard: find_order(6, 15, ...) → (0, {}) because gcd(6,15)=3.
    #[test]
    fn test_find_order_gcd_not_one() {
        let (r, dist) = find_order(6, 15, None, 1);
        assert_eq!(r, 0, "expected r=0 when gcd(a,N)>1");
        assert!(dist.is_empty(), "expected empty dist when gcd(a,N)>1");
    }

    // ---- OpenCL-dependent tests ----
    // These are marked #[ignore] because:
    //   a) OpenCL may not be available in all CI environments, and
    //   b) even when available the probabilistic circuit may not converge
    //      in a fixed shot budget without a tuned seed.
    // Run with `cargo test -- --include-ignored` to exercise them.

    #[test]
    #[ignore]
    fn test_find_order_2_mod_15() {
        let (r, dist) = find_order(2, 15, None, 20);
        assert!(!dist.is_empty(), "distribution should not be empty");
        assert_eq!(r, 4, "order of 2 mod 15 should be 4, got {r}");
    }

    #[test]
    #[ignore]
    fn test_find_order_explicit_precision() {
        let (r, dist) = find_order(2, 15, Some(8), 20);
        assert!(!dist.is_empty());
        assert_eq!(r, 4, "order of 2 mod 15 should be 4, got {r}");
    }

    #[test]
    #[ignore]
    fn test_find_factor_multiple_tries() {
        let f = find_factor(15, 5, 10, Some(42));
        assert!(f == 3 || f == 5, "expected 3 or 5, got {f}");
    }

    #[test]
    #[ignore]
    fn test_find_factor_15() {
        let f = find_factor(15, 5, 10, Some(42));
        assert!(f == 3 || f == 5, "Expected 3 or 5, got {}", f);
    }
}

/// Binary entrypoint: parse CLI args, run Shor's factoring, and emit JSON.
pub fn run() -> ! {
    let args = Args::parse();
    let n_val = args.n;

    if n_val < 2 {
        let err = ErrorOutput {
            framework: "qcgpu",
            algorithm: "shor",
            n: n_val,
            error: "N must be >= 2".to_string(),
        };
        println!("{}", serde_json::to_string(&err).unwrap());
        std::process::exit(0);
    }

    let start = Instant::now();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        find_factor(n_val, args.tries, args.shots, args.seed)
    }));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(factor) => {
            let out = Output {
                framework: "qcgpu",
                algorithm: "shor",
                n: n_val,
                factor,
                time_ms: elapsed_ms,
            };
            println!("{}", serde_json::to_string(&out).unwrap());
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "OpenCL not available on this platform".to_string()
            };
            let err = ErrorOutput {
                framework: "qcgpu",
                algorithm: "shor",
                n: n_val,
                error: msg,
            };
            println!("{}", serde_json::to_string(&err).unwrap());
        }
    }

    std::process::exit(0);
}
