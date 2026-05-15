// Classical number-theory helpers and post-processing for Shor's algorithm.

use std::collections::HashMap;

use num_integer::Integer;
use num_rational::Ratio;

pub fn mod_pow(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u128 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

pub fn limit_denominator(num: u64, den: u64, max_den: u64) -> (u64, u64) {
    if den == 0 || max_den == 0 {
        return (0, 1);
    }
    let g = num.gcd(&den);
    let mut p0: i128 = 0;
    let mut q0: i128 = 1;
    let mut p1: i128 = 1;
    let mut q1: i128 = 0;
    let mut n = (num / g) as i128;
    let mut d = (den / g) as i128;
    while d != 0 {
        let a = n / d;
        let p2 = a * p1 + p0;
        let q2 = a * q1 + q0;
        if q2 > max_den as i128 {
            let k = if q1 > 0 { (max_den as i128 - q0) / q1 } else { 0 };
            let p_sc = k * p1 + p0;
            let q_sc = (k * q1 + q0).max(1);
            let target = Ratio::new(num as i128, den as i128);
            let r1 = Ratio::new(p1, q1.max(1));
            let r_sc = Ratio::new(p_sc.max(0), q_sc);
            let d1 = if r1 > target {
                r1 - target.clone()
            } else {
                target.clone() - r1
            };
            let d_sc = if r_sc > target {
                r_sc - target.clone()
            } else {
                target - r_sc
            };
            if d_sc <= d1 {
                return (p_sc.max(0) as u64, q_sc as u64);
            } else {
                return (p1.max(0) as u64, q1.max(1) as u64);
            }
        }
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        let nn = d;
        d = n - a * d;
        n = nn;
    }
    (p1.max(0) as u64, q1.max(1) as u64)
}

pub fn reduce_to_min_order(mut r: u64, a: u64, n_mod: u64) -> u64 {
    let mut temp = r;
    let mut primes: Vec<u64> = Vec::new();
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
        if r % p == 0 && mod_pow(a as u128, (r / p) as u128, n_mod as u128) == 1 {
            r /= p;
        }
    }
    r
}

pub fn get_order_from_dist(
    dist: &HashMap<String, usize>,
    a: u64,
    n_mod: u64,
    precision: usize,
) -> u64 {
    let mut entries: Vec<(&String, &usize)> = dist.iter().collect();
    entries.sort_by(|x, y| y.1.cmp(x.1));
    let max_try = entries.len().min(10);
    let two_to_m: u64 = 1u64 << precision;
    for (bs, _count) in entries.iter().take(max_try) {
        if bs.chars().all(|c| c == '0') {
            continue;
        }
        // histogram_string is written with format!("{:0width$b}", key), so the
        // leftmost character is the highest classical bit (cbit m-1) and the
        // rightmost is cbit 0. Since cbit i mirrors ctrl qubit i, and ctrl[0]
        // is the MSB of the phase (matching the Python order_finding_circuit),
        // we need to reverse the bitstring before parsing to recover the phase
        // integer x in the standard big-endian representation.
        let reversed: String = bs.chars().rev().collect();
        let x = match u64::from_str_radix(&reversed, 2) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let (_, q) = limit_denominator(x, two_to_m, n_mod.saturating_sub(1).max(1));
        let r = q;
        if r == 0 {
            continue;
        }
        if mod_pow(a as u128, r as u128, n_mod as u128) == 1 {
            return reduce_to_min_order(r, a, n_mod);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_denominator_basic() {
        // 1/4 with max_den >= 4 reduces to 1/4 directly.
        let (p, q) = limit_denominator(1, 4, 15);
        assert_eq!((p, q), (1, 4));
    }

    #[test]
    fn test_limit_denominator_reduces() {
        // 2/8 reduces to 1/4.
        let (p, q) = limit_denominator(2, 8, 15);
        assert_eq!((p, q), (1, 4));
    }

    #[test]
    fn test_limit_denominator_capped() {
        // 3/10 with max_den=4 should approximate to 1/3 (closest with den<=4).
        // We only assert the denominator respects the cap.
        let (_p, q) = limit_denominator(3, 10, 4);
        assert!(q <= 4, "denominator {} exceeds cap 4", q);
        assert!(q >= 1);
    }

    #[test]
    fn test_mod_pow_basic() {
        // 2^10 = 1024 mod 1000 = 24
        assert_eq!(mod_pow(2, 10, 1000), 24);
        // 2^4 mod 15 = 1 (verifies order computations downstream)
        assert_eq!(mod_pow(2, 4, 15), 1);
        // 7^4 mod 15 = 1
        assert_eq!(mod_pow(7, 4, 15), 1);
    }

    #[test]
    fn test_reduce_to_min_order() {
        // 2^4 == 1 mod 15, so order is 4. reduce_to_min_order(8, 2, 15) -> 4 (4 | 8).
        assert_eq!(reduce_to_min_order(8, 2, 15), 4);
        // 7^4 == 1 mod 15, so order is 4. reduce_to_min_order(12, 7, 15) -> 4 (4 | 12).
        assert_eq!(reduce_to_min_order(12, 7, 15), 4);
        // Already minimal: reduce_to_min_order(4, 2, 15) -> 4.
        assert_eq!(reduce_to_min_order(4, 2, 15), 4);
    }
}
