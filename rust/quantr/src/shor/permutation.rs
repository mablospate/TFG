//! Permutation-network primitives for the controlled modular multiplication
//! in Shor's order-finding circuit on quantr.
//!
//! A modular multiplication y -> A^p * y mod N is decomposed into disjoint
//! cycles, each cycle into transpositions (cycle[0], cycle[i]), and each
//! transposition into single-bit-difference controlled transpositions
//! implemented via a multi-controlled X (Toffoli ladder with ancillas).

use std::collections::{HashMap, HashSet};

use quantr::{Circuit, Gate, QuantrError};

use crate::shor::classical::mod_pow;

/// Required ancilla count for the MCX ladders used by
/// `controlled_single_bit_transposition`. The worst-case control count is
/// `1 + (n - 1) = n` (one outer control plus all but one target qubit), so
/// we need `n - 2` ancillas. Returns 0 when `n < 3`.
pub fn ancilla_count(n: usize) -> usize {
    if n >= 3 {
        n - 2
    } else {
        0
    }
}

/// Apply an MCX with the given control qubits onto `target`. For k >= 3
/// controls we need k - 2 ancillas (all in |0>); ancillas are restored to |0>
/// before returning. For k <= 2 we use the native gates and `ancillas` is
/// ignored.
pub fn add_mcx(
    qc: &mut Circuit,
    controls: &[usize],
    target: usize,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let k = controls.len();
    match k {
        0 => {
            qc.add_gate(Gate::X, target)?;
        }
        1 => {
            qc.add_gate(Gate::CNot(controls[0]), target)?;
        }
        2 => {
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), target)?;
        }
        _ => {
            assert!(
                ancillas.len() >= k - 2,
                "MCX ladder needs k-2 ancillas (k={}, given {})",
                k,
                ancillas.len()
            );
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
            for i in 2..(k - 1) {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i - 2]), ancillas[i - 1])?;
            }
            qc.add_gate(Gate::Toffoli(controls[k - 1], ancillas[k - 3]), target)?;
            for i in (2..(k - 1)).rev() {
                qc.add_gate(Gate::Toffoli(controls[i], ancillas[i - 2]), ancillas[i - 1])?;
            }
            qc.add_gate(Gate::Toffoli(controls[0], controls[1]), ancillas[0])?;
        }
    }
    Ok(())
}

/// Controlled single-bit transposition |a> <-> |b> where a XOR b is a power of
/// two. `target_qubits[i]` is the qubit holding *logical* bit i of the
/// target register (LSB at index 0).
pub fn controlled_single_bit_transposition(
    qc: &mut Circuit,
    ctrl: usize,
    target_qubits: &[usize],
    a: u64,
    b: u64,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let n = target_qubits.len();
    let diff = a ^ b;
    debug_assert!(diff != 0 && diff & (diff - 1) == 0);
    let flip_bit = diff.trailing_zeros() as usize;

    let other_positions: Vec<usize> = (0..n).filter(|&i| i != flip_bit).collect();

    // X-wrap any positions where a has a 0 bit so those qubits act as
    // |1>-controls.
    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            qc.add_gate(Gate::X, target_qubits[pos])?;
        }
    }

    let mut controls: Vec<usize> = Vec::with_capacity(1 + other_positions.len());
    controls.push(ctrl);
    for &pos in &other_positions {
        controls.push(target_qubits[pos]);
    }
    add_mcx(qc, &controls, target_qubits[flip_bit], ancillas)?;

    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            qc.add_gate(Gate::X, target_qubits[pos])?;
        }
    }
    Ok(())
}

/// Controlled transposition |a> <-> |b> (general multi-bit difference)
/// recursively reduced to single-bit transpositions.
pub fn controlled_transposition(
    qc: &mut Circuit,
    ctrl: usize,
    target_qubits: &[usize],
    a: u64,
    b: u64,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let diff_bits = a ^ b;
    if diff_bits == 0 {
        return Ok(());
    }
    let n = target_qubits.len();
    let diff_positions: Vec<usize> = (0..n).filter(|&i| (diff_bits >> i) & 1 == 1).collect();
    if diff_positions.len() == 1 {
        controlled_single_bit_transposition(qc, ctrl, target_qubits, a, b, ancillas)?;
    } else {
        let pivot = diff_positions[0];
        let a_prime = a ^ (1u64 << pivot);
        controlled_transposition(qc, ctrl, target_qubits, a, a_prime, ancillas)?;
        controlled_transposition(qc, ctrl, target_qubits, a_prime, b, ancillas)?;
        controlled_transposition(qc, ctrl, target_qubits, a, a_prime, ancillas)?;
    }
    Ok(())
}

/// Apply a controlled permutation of basis states. The permutation is given
/// as a list of (input, output) pairs, which is decomposed into disjoint
/// cycles and each cycle into transpositions of the form (cycle[0], cycle[i]).
pub fn controlled_swap_permutation(
    qc: &mut Circuit,
    ctrl: usize,
    target_qubits: &[usize],
    permutation: &HashMap<u64, u64>,
    ancillas: &[usize],
) -> Result<(), QuantrError> {
    let mut visited: HashSet<u64> = HashSet::new();
    let mut keys: Vec<u64> = permutation.keys().copied().collect();
    keys.sort();
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
            controlled_transposition(qc, ctrl, target_qubits, cycle[0], cycle[idx], ancillas)?;
        }
    }
    Ok(())
}

/// Build the modular-multiplication permutation y -> A^power * y mod N for
/// 0 <= y < N. States with y == target are omitted (identity entries).
pub fn build_mod_exp_permutation(a: u64, n_val: u64, power: u64) -> HashMap<u64, u64> {
    let a_power = mod_pow(a, power, n_val);
    let mut perm: HashMap<u64, u64> = HashMap::new();
    for y in 0..n_val {
        let target = (a_power * y) % n_val;
        if y != target {
            perm.insert(y, target);
        }
    }
    perm
}
