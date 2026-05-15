//! Classical helpers for Shor's algorithm.
//!
//! Mirrors the classical body of `python/qiskit/shor/shor.py` and
//! `python/cudaq/shor/shor.py`: modular exponentiation, continued-fractions
//! order extraction, and order minimisation.

use std::collections::HashMap;

use num_rational::Ratio;
use num_traits::sign::Signed;

// =============================================================================
// Modular exponentiation.
// =============================================================================

pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    let m = modulus as u128;
    base %= modulus;
    let mut b = base as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m;
        }
        exp >>= 1;
        b = (b * b) % m;
    }
    result as u64
}

// =============================================================================
// Continued-fractions helpers (CPython Fraction.limit_denominator port).
// =============================================================================

/// Faithful port of `Fraction(x).limit_denominator(max_denominator)`. The
/// algorithm is the one documented in CPython's `fractions` module.
pub fn limit_denominator(frac: Ratio<i128>, max_denom: i128) -> Ratio<i128> {
    if max_denom < 1 {
        return frac;
    }
    if *frac.denom() <= max_denom {
        return frac;
    }
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let mut n = *frac.numer();
    let mut d = *frac.denom();
    loop {
        if d == 0 {
            break;
        }
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_denom {
            break;
        }
        let new_p1 = p0 + a * p1;
        p0 = p1;
        q0 = q1;
        p1 = new_p1;
        q1 = q2;
        let new_d = n - a * d;
        n = d;
        d = new_d;
    }
    let k = (max_denom - q0) / q1;
    let bound1 = Ratio::new(p0 + k * p1, q0 + k * q1);
    let bound2 = Ratio::new(p1, q1);
    let diff1 = (bound1 - frac).abs();
    let diff2 = (bound2 - frac).abs();
    if diff2 <= diff1 {
        bound2
    } else {
        bound1
    }
}

// =============================================================================
// Order extraction from measurement distribution.
// =============================================================================

/// Mirrors `_get_order_from_dist` in `python/qiskit/shor/shor.py`.
///
/// Sorts bitstrings by frequency (descending), skips all-zeros, tries the top
/// 10, applies `limit_denominator` to extract a candidate order, and verifies
/// it before returning.  Returns 0 if no valid order is found.
pub fn get_order_from_dist(
    dist: &HashMap<String, usize>,
    a: u64,
    n_val: u64,
    precision: usize,
) -> u64 {
    let mut sorted: Vec<(&String, &usize)> = dist.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    let two_m = 1u64 << precision;
    let limit = if n_val > 1 { n_val - 1 } else { 1 };

    for (bs, _) in sorted.iter().take(10) {
        if bs.chars().all(|ch| ch == '0') {
            continue;
        }
        let x = match u64::from_str_radix(bs, 2) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let frac = Ratio::new(x as i128, two_m as i128);
        let approx = limit_denominator(frac, limit as i128);
        let r = *approx.denom() as u64;
        if r == 0 {
            continue;
        }
        if mod_pow(a, r, n_val) == 1 {
            return reduce_to_min_order(r, a, n_val);
        }
    }
    0
}

// =============================================================================
// Order minimisation.
// =============================================================================

pub fn reduce_to_min_order(mut r: u64, a: u64, n_val: u64) -> u64 {
    let mut primes = Vec::new();
    let mut temp = r;
    let mut d: u64 = 2;
    while d.saturating_mul(d) <= temp {
        while temp % d == 0 {
            primes.push(d);
            temp /= d;
        }
        d += 1;
    }
    if temp > 1 {
        primes.push(temp);
    }
    for p in primes {
        if r % p == 0 && mod_pow(a, r / p, n_val) == 1 {
            r /= p;
        }
    }
    r
}

// =============================================================================
// Tests.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 4, 15), 1); // 2^4 = 16 ≡ 1 (mod 15)
        assert_eq!(mod_pow(7, 4, 15), 1);
        assert_eq!(mod_pow(2, 0, 15), 1);
        assert_eq!(mod_pow(2, 1, 15), 2);
    }

    #[test]
    fn test_limit_denominator_basic() {
        // 1/4 with max_denom=15: denom 4 <= 15, returned as-is.
        let r = limit_denominator(Ratio::new(1i128, 4i128), 15);
        assert_eq!((*r.numer(), *r.denom()), (1, 4));

        // 1/3 with max_denom=15: denom 3 <= 15, returned as-is.
        let r2 = limit_denominator(Ratio::new(1i128, 3i128), 15);
        assert_eq!((*r2.numer(), *r2.denom()), (1, 3));
    }

    #[test]
    fn test_reduce_to_min_order() {
        // 2^4 = 1 mod 15, order is 4. reduce_to_min_order(8, 2, 15) should give 4
        // because 2^4 ≡ 1 mod 15 and 4 divides 8.
        assert_eq!(reduce_to_min_order(8, 2, 15), 4);

        // 7^4 = 1 mod 15, order is 4. reduce_to_min_order(12, 7, 15) should give 4
        // because 7^4 ≡ 1 mod 15 and 4 divides 12.
        assert_eq!(reduce_to_min_order(12, 7, 15), 4);
    }

    #[test]
    fn test_get_order_from_dist_synthetic() {
        // Bitstring "01000000" with m=8: qubit 0 is leftmost character.
        // Parsed as binary: "01000000" = 64. Phase = 64/256 = 1/4.
        // Continued fractions → denominator = 4, which is the order of 2 mod 15.
        let mut dist: HashMap<String, usize> = HashMap::new();
        dist.insert("01000000".to_string(), 150);
        dist.insert("00000000".to_string(), 10);
        let r = get_order_from_dist(&dist, 2, 15, 8);
        assert_eq!(r, 4, "Expected order 4 for phase 1/4 (a=2, N=15), got {}", r);
    }

    #[test]
    fn test_get_order_from_dist_skips_zeros() {
        // Only all-zeros entry — should return 0.
        let mut dist: HashMap<String, usize> = HashMap::new();
        dist.insert("0000".to_string(), 100);
        let r = get_order_from_dist(&dist, 2, 15, 4);
        assert_eq!(r, 0);
    }
}
