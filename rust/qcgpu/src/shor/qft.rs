//! Inverse QFT for the qcgpu OpenCL-based simulator.
//!
//! qcgpu has no built-in QFT, so we implement it manually using the
//! `r(angle: f32)` phase gate. All angles are `f32`, which is sufficient
//! for the small circuit sizes used in the benchmarks (N ≤ 15 with m ≤ 8).

use std::f64::consts::PI;

use qcgpu::gates;
use qcgpu::State;

/// Apply the inverse QFT on the given qubit list.
///
/// qcgpu uses LSB-first indexing: qubit 0 is the rightmost character of the
/// bitstring. The convention here is that `qubits[0]` holds the MSB of the
/// phase after the IQFT swap.
pub fn apply_inverse_qft(state: &mut State, qubits: &[i32]) {
    let n = qubits.len();

    // Inverse-QFT begins by un-swapping the bit order.
    for i in 0..n / 2 {
        state.swap(qubits[i], qubits[n - 1 - i]);
    }

    // Reverse the QFT gates.
    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            let angle = -(PI / (1u64 << (j - i)) as f64) as f32;
            state.apply_controlled_gate(qubits[j], qubits[i], gates::r(angle));
        }
        state.h(qubits[i]);
    }
}
