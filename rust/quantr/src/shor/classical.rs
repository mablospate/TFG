//! Classical number-theory helpers and post-processing for Shor's algorithm
//! on the quantr backend.
//!
//! The continued-fraction step uses a faithful port of CPython's
//! `Fraction.limit_denominator`, operating on `i128` for overflow safety.

use std::collections::HashMap;

/// Modular exponentiation `base^exp mod modulus` using `u128` intermediates.
pub fn mod_pow(base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let modulus_u128 = modulus as u128;
    let mut result: u128 = 1;
    let mut base_u128 = (base as u128) % modulus_u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base_u128) % modulus_u128;
        }
        exp >>= 1;
        base_u128 = (base_u128 * base_u128) % modulus_u128;
    }
    result as u64
}

/// Port of CPython `Fraction.limit_denominator`.
///
/// Returns the closest rational `p/q` to `numerator/denominator` with
/// `q <= max_denominator`. Uses `i128` for overflow safety.
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
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    let (mut n, mut d) = (numerator, denominator);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_denominator {
            break;
        }
        let p2 = p0 + a * p1;
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
    // Two candidate bounds:
    //   bound1 = (p0 + k*p1) / (q0 + k*q1)
    //   bound2 = p1 / q1
    // Choose the one closer to numerator/denominator without using floats.
    let (bound1_n, bound1_d) = (p0 + k * p1, q0 + k * q1);
    let (bound2_n, bound2_d) = (p1, q1);
    let orig_n = numerator;
    let orig_d = denominator;
    let diff1 = (bound1_n * orig_d - orig_n * bound1_d).abs();
    let diff2 = (bound2_n * orig_d - orig_n * bound2_d).abs();
    // |bound2 - orig| <= |bound1 - orig|  <=> diff2/bound2_d <= diff1/bound1_d
    // Multiply through by the (positive) denominators.
    if diff1 * bound2_d <= diff2 * bound1_d {
        (bound1_n, bound1_d)
    } else {
        (bound2_n, bound2_d)
    }
}

/// Reduce `r` to the minimum r' such that `a^r' == 1 (mod n_val)` by dividing
/// out prime factors of `r`.
pub fn reduce_to_min_order(mut r: u64, a: u64, n_val: u64) -> u64 {
    let mut temp = r;
    let mut primes: Vec<u64> = Vec::new();
    let mut d = 2u64;
    while d * d <= temp {
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

/// Recover the order from a measurement distribution.
///
/// `dist` maps standard (LSB-first, the same convention as the qiskit
/// reference) bitstrings to their counts. `precision` is `m`, the number of
/// control-register bits, so each phase candidate `x` is interpreted as
/// `x / 2^m`.
pub fn get_order_from_dist(
    dist: &HashMap<String, usize>,
    a: u64,
    n_val: u64,
    precision: usize,
) -> u64 {
    let mut sorted: Vec<(&String, &usize)> = dist.iter().collect();
    sorted.sort_by(|x, y| y.1.cmp(x.1));
    let two_m: i128 = 1i128 << precision;
    let max_den: i128 = (n_val as i128).saturating_sub(1).max(1);
    for (bitstring, _) in sorted.into_iter().take(10) {
        if bitstring.chars().all(|c| c == '0') {
            continue;
        }
        let x = match u64::from_str_radix(bitstring, 2) {
            Ok(v) => v as i128,
            Err(_) => continue,
        };
        if x == 0 {
            continue;
        }
        let (_, denom) = limit_denominator(x, two_m, max_den);
        if denom <= 0 {
            continue;
        }
        let r = denom as u64;
        if r > 0 && r < n_val && mod_pow(a, r, n_val) == 1 {
            return reduce_to_min_order(r, a, n_val);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 4, 15), 1); // 2^4 = 16 = 1 mod 15
        assert_eq!(mod_pow(7, 4, 15), 1); // 7^4 = 2401 = 1 mod 15
        assert_eq!(mod_pow(2, 1, 15), 2);
        assert_eq!(mod_pow(2, 0, 15), 1);
        assert_eq!(mod_pow(0, 5, 7), 0);
    }

    #[test]
    fn test_limit_denominator_already_within_cap() {
        // 1/4 with max_den >= 4 returns 1/4 verbatim (denominator <= max).
        assert_eq!(limit_denominator(1, 4, 15), (1, 4));
    }

    #[test]
    fn test_limit_denominator_quarter() {
        // 64/256 with max_den 14 must reduce to 1/4 (the convergent below 14).
        let (p, q) = limit_denominator(64, 256, 14);
        assert_eq!((p, q), (1, 4));
    }

    #[test]
    fn test_limit_denominator_pi_approx() {
        // CPython doctest example:
        // Fraction(3141592653589793, 1000000000000000).limit_denominator(10) == 22/7.
        let (p, q) = limit_denominator(3_141_592_653_589_793, 1_000_000_000_000_000, 10);
        assert_eq!((p, q), (22, 7));
    }

    #[test]
    fn test_limit_denominator_capped() {
        // 3/10 with max_den=4: the best approximation is 1/3 (|1/3 - 3/10| < |1/4 - 3/10|).
        let (p, q) = limit_denominator(3, 10, 4);
        assert!(q <= 4);
        assert_eq!((p, q), (1, 3));
    }

    #[test]
    fn test_reduce_to_min_order() {
        // 2^4 == 1 mod 15. reduce_to_min_order(8, 2, 15) -> 4 (4 | 8).
        assert_eq!(reduce_to_min_order(8, 2, 15), 4);
        // 7^4 == 1 mod 15. reduce_to_min_order(12, 7, 15) -> 4 (4 | 12).
        assert_eq!(reduce_to_min_order(12, 7, 15), 4);
        // Already minimal: reduce_to_min_order(4, 2, 15) -> 4.
        assert_eq!(reduce_to_min_order(4, 2, 15), 4);
    }
}
