// Permutation network for controlled modular multiplication.
//
// Builds the permutation x -> (a^power * x) mod N as a product of controlled
// transpositions on the target register. Multi-bit transpositions are
// decomposed into single-bit ones via the standard conjugation trick.

use std::collections::HashMap;

use q1tsim::circuit::Circuit;
use q1tsim::error::Result as QResult;
use q1tsim::gates::CCX;

use super::classical::mod_pow;

// ----- multi-controlled X with Toffoli ladder -----

pub fn apply_mcx(c: &mut Circuit, ctrls: &[usize], tgt: usize, ancillas: &[usize]) -> QResult<()> {
    let k = ctrls.len();
    match k {
        0 => c.x(tgt)?,
        1 => c.cx(ctrls[0], tgt)?,
        2 => c.add_gate(CCX::new(), &[ctrls[0], ctrls[1], tgt])?,
        _ => {
            c.add_gate(CCX::new(), &[ctrls[0], ctrls[1], ancillas[0]])?;
            for i in 1..(k - 2) {
                c.add_gate(CCX::new(), &[ctrls[i + 1], ancillas[i - 1], ancillas[i]])?;
            }
            c.add_gate(CCX::new(), &[ctrls[k - 1], ancillas[k - 3], tgt])?;
            for i in (1..(k - 2)).rev() {
                c.add_gate(CCX::new(), &[ctrls[i + 1], ancillas[i - 1], ancillas[i]])?;
            }
            c.add_gate(CCX::new(), &[ctrls[0], ctrls[1], ancillas[0]])?;
        }
    }
    Ok(())
}

// ----- permutation network -----

pub fn controlled_single_bit_transposition(
    c: &mut Circuit,
    ctrl: usize,
    target_qubits: &[usize],
    ancillas: &[usize],
    a: u64,
    b: u64,
) -> QResult<()> {
    let n = target_qubits.len();
    let diff = a ^ b;
    debug_assert!(diff != 0 && diff & (diff - 1) == 0, "a,b must differ in 1 bit");
    let flip_bit = diff.trailing_zeros() as usize;

    let other_positions: Vec<usize> = (0..n).filter(|&i| i != flip_bit).collect();

    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            c.x(target_qubits[pos])?;
        }
    }

    let mut all_ctrls: Vec<usize> = Vec::with_capacity(1 + other_positions.len());
    all_ctrls.push(ctrl);
    for &pos in &other_positions {
        all_ctrls.push(target_qubits[pos]);
    }
    apply_mcx(c, &all_ctrls, target_qubits[flip_bit], ancillas)?;

    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            c.x(target_qubits[pos])?;
        }
    }
    Ok(())
}

pub fn controlled_transposition(
    c: &mut Circuit,
    ctrl: usize,
    target_qubits: &[usize],
    ancillas: &[usize],
    a: u64,
    b: u64,
) -> QResult<()> {
    let diff_bits = a ^ b;
    if diff_bits == 0 {
        return Ok(());
    }
    let n = target_qubits.len();
    let diff_positions: Vec<usize> = (0..n).filter(|&i| (diff_bits >> i) & 1 == 1).collect();

    if diff_positions.len() == 1 {
        controlled_single_bit_transposition(c, ctrl, target_qubits, ancillas, a, b)?;
    } else {
        let pivot = diff_positions[0];
        let a_prime = a ^ (1u64 << pivot);
        controlled_transposition(c, ctrl, target_qubits, ancillas, a, a_prime)?;
        controlled_transposition(c, ctrl, target_qubits, ancillas, a_prime, b)?;
        controlled_transposition(c, ctrl, target_qubits, ancillas, a, a_prime)?;
    }
    Ok(())
}

pub fn controlled_swap_permutation(
    c: &mut Circuit,
    ctrl: usize,
    target_qubits: &[usize],
    ancillas: &[usize],
    permutation: &HashMap<u64, u64>,
) -> QResult<()> {
    let mut keys: Vec<u64> = permutation.keys().copied().collect();
    keys.sort();
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for start in keys {
        if visited.contains(&start) {
            continue;
        }
        let mut cycle: Vec<u64> = Vec::new();
        let mut current = start;
        while !visited.contains(&current) {
            visited.insert(current);
            cycle.push(current);
            current = *permutation.get(&current).unwrap_or(&current);
        }
        if cycle.len() <= 1 {
            continue;
        }
        for idx in 1..cycle.len() {
            controlled_transposition(c, ctrl, target_qubits, ancillas, cycle[0], cycle[idx])?;
        }
    }
    Ok(())
}

pub fn build_mod_exp_permutation(a: u64, n_mod: u64, power: u64) -> HashMap<u64, u64> {
    let a_pow = mod_pow(a as u128, power as u128, n_mod as u128) as u64;
    let mut perm = HashMap::new();
    for y in 0..n_mod {
        let tgt = ((a_pow as u128 * y as u128) % n_mod as u128) as u64;
        if y != tgt {
            perm.insert(y, tgt);
        }
    }
    perm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_exp_permutation_a2_n5_pow2() {
        // 2^2 = 4 mod 5: permutation x -> 4x mod 5
        // 1 -> 4, 2 -> 3, 3 -> 2, 4 -> 1 (0 is fixed and omitted)
        let perm = build_mod_exp_permutation(2, 5, 2);
        assert_eq!(perm.get(&1), Some(&4));
        assert_eq!(perm.get(&2), Some(&3));
        assert_eq!(perm.get(&3), Some(&2));
        assert_eq!(perm.get(&4), Some(&1));
        // 0 is a fixed point and is not stored.
        assert!(perm.get(&0).is_none());
    }

    #[test]
    fn test_mod_exp_permutation_identity() {
        // a^0 mod n = 1: perm is x -> x, hence empty (no non-fixed points).
        let perm = build_mod_exp_permutation(2, 5, 0);
        assert!(perm.is_empty(), "identity permutation should be empty, got {:?}", perm);
    }

    #[test]
    fn test_mod_exp_permutation_a7_n15_pow1() {
        // 7 mod 15: x -> 7x mod 15
        let perm = build_mod_exp_permutation(7, 15, 1);
        assert_eq!(perm.get(&1), Some(&7));
        assert_eq!(perm.get(&2), Some(&14));
        assert_eq!(perm.get(&3), Some(&6));
    }
}
