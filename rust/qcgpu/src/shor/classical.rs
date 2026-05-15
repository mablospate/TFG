//! Classical number-theory helpers and post-processing for Shor's algorithm.
//!
//! Uses `i128` throughout to avoid overflow for large N that would occur with
//! `i64`. The `limit_denominator` implementation faithfully mirrors CPython's
//! `Fraction.limit_denominator`.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Modular arithmetic
// ---------------------------------------------------------------------------

pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    base %= modulus;
    let mut b: u128 = base as u128;
    let m: u128 = modulus as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m;
        }
        b = (b * b) % m;
        exp >>= 1;
    }
    result as u64
}

// ---------------------------------------------------------------------------
// Continued-fraction approximation
// ---------------------------------------------------------------------------

/// Approximate `numerator / denominator` by a rational whose denominator is
/// <= `max_denominator`, using the continued-fraction convergents.
///
/// This is a faithful i128 port of CPython's `Fraction.limit_denominator`.
/// Key differences from the old i64 version:
///   - Uses `i128` to prevent overflow for large N.
///   - The early `if r == 0 { break; }` has been removed; the only
///     termination condition inside the loop is `if d == 0 { break; }`,
///     matching CPython's reference implementation.
pub fn limit_denominator(
    numerator: i128,
    denominator: i128,
    max_denominator: i128,
) -> (i128, i128) {
    if max_denominator < 1 {
        panic!("max_denominator must be >= 1");
    }
    if denominator <= max_denominator {
        return (numerator, denominator);
    }
    // Faithful i128 port of CPython's Fraction.limit_denominator.
    // The tuple assignment (p0,q0,p1,q1) = (p1,q1,p2,q2) is done atomically.
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let (mut n, mut d) = (numerator, denominator);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_denominator {
            break;
        }
        let p2 = p0 + a * p1;
        // Atomic update: (p0,q0,p1,q1) <- (p1,q1,p2,q2)
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        let new_d = n - a * d;
        n = d;
        d = new_d;
        if d == 0 {
            break;
        }
    }
    let k = (max_denominator - q0) / q1;
    let (b1n, b1d) = (p0 + k * p1, q0 + k * q1);
    let (b2n, b2d) = (p1, q1);
    let orig_n = numerator;
    let orig_d = denominator;
    // Choose the candidate closest to orig_n/orig_d.
    if (b2n * orig_d - orig_n * b2d).abs() * b1d
        <= (b1n * orig_d - orig_n * b1d).abs() * b2d
    {
        (b2n, b2d)
    } else {
        (b1n, b1d)
    }
}

// ---------------------------------------------------------------------------
// Order post-processing
// ---------------------------------------------------------------------------

/// Reduce `r` to the smallest positive integer such that `a^r ≡ 1 (mod n)`.
pub fn reduce_to_min_order(mut r: u64, a: u64, n_val: u64) -> u64 {
    let mut temp = r;
    let mut primes: Vec<u64> = Vec::new();
    let mut divisor = 2u64;
    while divisor.saturating_mul(divisor) <= temp {
        while temp % divisor == 0 {
            primes.push(divisor);
            temp /= divisor;
        }
        divisor += 1;
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

/// Extract the order from a measurement distribution.
///
/// Mirrors Python's `_get_order_from_dist`. Returns 0 if the order cannot be
/// determined from the top-10 most frequent bitstrings.
pub fn get_order_from_dist(
    dist: &HashMap<String, usize>,
    a: u64,
    n_val: u64,
    precision: usize,
) -> u64 {
    let mut entries: Vec<(&String, &usize)> = dist.iter().collect();
    entries.sort_by(|x, y| y.1.cmp(x.1));
    let two_to_m: u64 = 1u64 << precision;
    for (bs, _) in entries.iter().take(10) {
        if bs.chars().all(|c| c == '0') {
            continue;
        }
        // qcgpu stores qubit 0 at the rightmost position (LSB-first), but
        // ctrl[0] is the MSB of the QPE phase, so reverse before parsing.
        let reversed: String = bs.chars().rev().collect();
        let x = match u64::from_str_radix(&reversed, 2) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let (_, q) = limit_denominator(
            x as i128,
            two_to_m as i128,
            (n_val.saturating_sub(1).max(1)) as i128,
        );
        let r = q as u64;
        if r == 0 {
            continue;
        }
        if mod_pow(a, r, n_val) == 1 {
            return reduce_to_min_order(r, a, n_val);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_limit_denominator_pi_approx() {
        // CPython: Fraction(314159, 100000).limit_denominator(100) == 311/99.
        let (n, d) = limit_denominator(314159, 100000, 100);
        assert!(d <= 100, "denominator {d} exceeds cap 100");
        assert_eq!((n, d), (311, 99));
    }

    #[test]
    fn test_limit_denominator_exact() {
        // x/2^m for order-finding: x=64 out of 2^8=256, max_denom=14.
        // CPython: Fraction(64, 256).limit_denominator(14) == Fraction(1,4).
        let (n, d) = limit_denominator(64, 256, 14);
        assert_eq!((n, d), (1, 4));
    }

    #[test]
    fn test_reduce_to_min_order() {
        assert_eq!(reduce_to_min_order(8, 2, 15), 4);
        assert_eq!(reduce_to_min_order(12, 7, 15), 4);
        assert_eq!(reduce_to_min_order(4, 2, 15), 4);
    }
}
