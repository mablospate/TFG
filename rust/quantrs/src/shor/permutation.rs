//! Permutation network for controlled modular exponentiation.
//!
//! Mirrors `python/cudaq/shor/permutation.py`: build the directed transposition
//! list for `y -> a^p * y mod N`, decompose into cycles, then realize each
//! cycle as a sequence of controlled transpositions implemented with MCX.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;

pub fn build_mod_exp_permutation(a_power: u64, n_val: u64) -> Vec<(u64, u64)> {
    let mut perm = Vec::new();
    for y in 0..n_val {
        let target = ((a_power as u128) * (y as u128) % n_val as u128) as u64;
        if y != target {
            perm.push((y, target));
        }
    }
    perm
}

pub fn apply_controlled_permutation<const TOTAL: usize>(
    c: &mut Circuit<TOTAL>,
    ctrl: usize,
    m: usize,
    n_bits: usize,
    perm: &[(u64, u64)],
) {
    use std::collections::HashMap as Map;
    let map: Map<u64, u64> = perm.iter().copied().collect();
    let mut visited = std::collections::HashSet::new();
    let mut keys: Vec<u64> = map.keys().copied().collect();
    keys.sort_unstable();
    for &start in &keys {
        if visited.contains(&start) {
            continue;
        }
        let mut cycle = Vec::new();
        let mut current = start;
        while !visited.contains(&current) {
            visited.insert(current);
            cycle.push(current);
            current = *map.get(&current).unwrap_or(&current);
        }
        if cycle.len() <= 1 {
            continue;
        }
        for idx in 1..cycle.len() {
            controlled_transposition::<TOTAL>(c, ctrl, m, n_bits, cycle[0], cycle[idx]);
        }
    }
}

pub fn controlled_transposition<const TOTAL: usize>(
    c: &mut Circuit<TOTAL>,
    ctrl: usize,
    m: usize,
    n_bits: usize,
    a: u64,
    b: u64,
) {
    let diff = a ^ b;
    if diff == 0 {
        return;
    }
    let diff_positions: Vec<usize> = (0..n_bits).filter(|&i| (diff >> i) & 1 == 1).collect();
    if diff_positions.len() == 1 {
        single_bit_controlled_transposition::<TOTAL>(c, ctrl, m, n_bits, a, b);
    } else {
        // Same recursive decomposition as the Python reference.
        let pivot = diff_positions[0];
        let a_prime = a ^ (1 << pivot);
        controlled_transposition::<TOTAL>(c, ctrl, m, n_bits, a, a_prime);
        controlled_transposition::<TOTAL>(c, ctrl, m, n_bits, a_prime, b);
        controlled_transposition::<TOTAL>(c, ctrl, m, n_bits, a, a_prime);
    }
}

/// Controlled transposition |a> <-> |b> when a and b differ in exactly one bit.
/// Implements `cx([ctrl] + others_qubits, flip_qubit)` sandwiched by X gates
/// that select the |a> basis pattern on the unchanged bits.
fn single_bit_controlled_transposition<const TOTAL: usize>(
    c: &mut Circuit<TOTAL>,
    ctrl: usize,
    m: usize,
    n_bits: usize,
    a: u64,
    b: u64,
) {
    let diff = a ^ b;
    let flip_bit = diff.trailing_zeros() as usize;
    let other_positions: Vec<usize> = (0..n_bits).filter(|&i| i != flip_bit).collect();

    // X-prep: flip the data qubits where a has a 0 bit, so the desired pattern
    // becomes all-ones (the standard MCX activation pattern).
    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            c.x(QubitId::new((m + pos) as u32)).unwrap();
        }
    }

    // Build MCX controls: external control + the (n_bits - 1) target-register
    // qubits that share their bit with both a and b.
    let mut controls = Vec::with_capacity(other_positions.len() + 1);
    controls.push(ctrl);
    for &pos in &other_positions {
        controls.push(m + pos);
    }
    let target_q = m + flip_bit;
    apply_mcx::<TOTAL>(c, &controls, target_q, m, n_bits);

    // Undo the X-prep.
    for &pos in &other_positions {
        if (a >> pos) & 1 == 0 {
            c.x(QubitId::new((m + pos) as u32)).unwrap();
        }
    }
}

/// Multi-controlled X using a Toffoli ladder with `controls.len() - 2`
/// ancillas. Ancillas live at indices `m + n_bits .. m + n_bits + (n_bits - 2)`
/// (allocated once for the whole circuit). The ladder is uncomputed in reverse
/// so the ancillas return to |0>.
fn apply_mcx<const TOTAL: usize>(
    c: &mut Circuit<TOTAL>,
    controls: &[usize],
    target: usize,
    m: usize,
    n_bits: usize,
) {
    let k = controls.len();
    match k {
        0 => {
            c.x(QubitId::new(target as u32)).unwrap();
        }
        1 => {
            c.cnot(
                QubitId::new(controls[0] as u32),
                QubitId::new(target as u32),
            )
            .unwrap();
        }
        2 => {
            c.toffoli(
                QubitId::new(controls[0] as u32),
                QubitId::new(controls[1] as u32),
                QubitId::new(target as u32),
            )
            .unwrap();
        }
        _ => {
            let anc_base = m + n_bits;
            let num_anc = k - 2;
            assert!(
                anc_base + num_anc <= TOTAL,
                "ancillas overflow: anc_base={anc_base} num_anc={num_anc} TOTAL={TOTAL}"
            );

            // Forward ladder: anc[0] = AND(c0, c1), anc[i] = AND(c_{i+1}, anc[i-1]).
            c.toffoli(
                QubitId::new(controls[0] as u32),
                QubitId::new(controls[1] as u32),
                QubitId::new(anc_base as u32),
            )
            .unwrap();
            for i in 1..num_anc {
                c.toffoli(
                    QubitId::new(controls[i + 1] as u32),
                    QubitId::new((anc_base + i - 1) as u32),
                    QubitId::new((anc_base + i) as u32),
                )
                .unwrap();
            }
            // Final activation gate on target.
            c.toffoli(
                QubitId::new(controls[k - 1] as u32),
                QubitId::new((anc_base + num_anc - 1) as u32),
                QubitId::new(target as u32),
            )
            .unwrap();
            // Reverse ladder to restore ancillas to |0>.
            for i in (1..num_anc).rev() {
                c.toffoli(
                    QubitId::new(controls[i + 1] as u32),
                    QubitId::new((anc_base + i - 1) as u32),
                    QubitId::new((anc_base + i) as u32),
                )
                .unwrap();
            }
            c.toffoli(
                QubitId::new(controls[0] as u32),
                QubitId::new(controls[1] as u32),
                QubitId::new(anc_base as u32),
            )
            .unwrap();
        }
    }
}
