//! Permutation-network implementation of controlled modular exponentiation
//! on qcgpu. The unitary is realized as a controlled permutation of the
//! computational basis on the target register, decomposed into disjoint
//! cycles, then into single-bit transpositions implemented via
//! multi-controlled X gates (Toffoli + ancilla ladder, strategy B from
//! section 7.10 of the implementation guide).

use std::collections::HashMap;

use qcgpu::State;

use super::classical::mod_pow;

// ---------------------------------------------------------------------------
// Multi-controlled X via ancilla ladder (strategy B from guide section 7.10)
// ---------------------------------------------------------------------------

/// Multi-controlled X. `ancillas` must hold at least `controls.len() - 2`
/// fresh |0> qubits when there are >= 3 controls.
pub fn mcx(state: &mut State, controls: &[i32], target: i32, ancillas: &[i32]) {
    let k = controls.len();
    match k {
        0 => state.x(target),
        1 => state.cx(controls[0], target),
        2 => state.toffoli(controls[0], controls[1], target),
        _ => {
            assert!(
                ancillas.len() >= k - 2,
                "mcx needs k-2 ancillas, got {} for k={}",
                ancillas.len(),
                k
            );

            state.toffoli(controls[0], controls[1], ancillas[0]);
            for i in 2..(k - 1) {
                state.toffoli(controls[i], ancillas[i - 2], ancillas[i - 1]);
            }
            state.toffoli(controls[k - 1], ancillas[k - 3], target);
            for i in (2..(k - 1)).rev() {
                state.toffoli(controls[i], ancillas[i - 2], ancillas[i - 1]);
            }
            state.toffoli(controls[0], controls[1], ancillas[0]);
        }
    }
}

// ---------------------------------------------------------------------------
// Permutation network
// ---------------------------------------------------------------------------

/// Controlled transposition where `a` and `b` differ in exactly one bit.
/// Implemented as a multi-controlled X targeting the differing bit, with the
/// other bits forced to match `a` (== `b` on those positions) via X-flips
/// where needed.
pub fn controlled_single_bit_transposition(
    state: &mut State,
    ctrl: i32,
    target_qubits: &[i32],
    a: u64,
    b: u64,
    ancillas: &[i32],
) {
    let n = target_qubits.len();
    let diff = a ^ b;
    debug_assert!(diff != 0 && (diff & (diff - 1)) == 0);
    let flip_bit = diff.trailing_zeros() as usize;

    let other_positions: Vec<usize> = (0..n).filter(|&i| i != flip_bit).collect();

    // X-flip target qubits that are 0 in `a` so the MCX trigger condition
    // becomes "all controls are 1".
    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            state.x(target_qubits[pos]);
        }
    }

    // Build the full controls list: external `ctrl` + the "shared" target
    // qubits (everything except the flipping one).
    let mut controls: Vec<i32> = Vec::with_capacity(1 + other_positions.len());
    controls.push(ctrl);
    for &pos in &other_positions {
        controls.push(target_qubits[pos]);
    }

    mcx(state, &controls, target_qubits[flip_bit], ancillas);

    // Undo the X flips.
    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            state.x(target_qubits[pos]);
        }
    }
}

/// Recursively decompose a controlled |a> <-> |b> transposition into
/// single-bit transpositions.
pub fn controlled_transposition(
    state: &mut State,
    ctrl: i32,
    target_qubits: &[i32],
    a: u64,
    b: u64,
    ancillas: &[i32],
) {
    let diff_bits = a ^ b;
    if diff_bits == 0 {
        return;
    }
    let n = target_qubits.len();
    let diff_positions: Vec<usize> = (0..n).filter(|&i| (diff_bits >> i) & 1 == 1).collect();

    if diff_positions.len() == 1 {
        controlled_single_bit_transposition(state, ctrl, target_qubits, a, b, ancillas);
    } else {
        let pivot = diff_positions[0];
        let a_prime = a ^ (1u64 << pivot);
        controlled_transposition(state, ctrl, target_qubits, a, a_prime, ancillas);
        controlled_transposition(state, ctrl, target_qubits, a_prime, b, ancillas);
        controlled_transposition(state, ctrl, target_qubits, a, a_prime, ancillas);
    }
}

/// Apply a controlled permutation of basis states on the target register by
/// decomposing into disjoint cycles, then into transpositions.
pub fn controlled_swap_permutation(
    state: &mut State,
    ctrl: i32,
    target_qubits: &[i32],
    permutation: &HashMap<u64, u64>,
    ancillas: &[i32],
) {
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();

    let mut keys: Vec<u64> = permutation.keys().copied().collect();
    keys.sort_unstable();

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
            controlled_transposition(state, ctrl, target_qubits, cycle[0], cycle[idx], ancillas);
        }
    }
}

/// Build the permutation |y> -> |A^power * y mod N> for 0 <= y < N.
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
