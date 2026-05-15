//! Manual QFT helpers built from controlled-phase decomposition.
//!
//! The simulator dispatches `crz` natively, so controlled-phase is synthesized
//! as `RZ(λ/2) · CRZ(-λ)` (correct up to a global phase).

use std::f64::consts::PI;

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;

/// Controlled phase `diag(1,1,1,e^{i*lambda})` on (control, target) built from
/// `crz` (which the simulator natively dispatches). The identity used:
///
/// ```text
///   P(λ/2)_control · CRZ(control, target, -λ)
///     = diag(1, 1, 1, e^{iλ})
/// ```
pub fn controlled_phase<const TOTAL: usize>(
    c: &mut Circuit<TOTAL>,
    control: usize,
    target: usize,
    lambda: f64,
) {
    // Up to a global phase: CP(λ) = RZ_c(λ/2) · CRZ(c, t, -λ).
    // RZ rather than P because the StateVectorSimulator only dispatches RZ.
    c.rz(QubitId::new(control as u32), lambda / 2.0).unwrap();
    c.crz(
        QubitId::new(control as u32),
        QubitId::new(target as u32),
        -lambda,
    )
    .unwrap();
}

pub fn inverse_qft<const TOTAL: usize>(c: &mut Circuit<TOTAL>, start: usize, len: usize) {
    // Mirror of python/cudaq/shor/qft.py:apply_inverse_qft.
    for i in 0..len / 2 {
        c.swap(
            QubitId::new((start + i) as u32),
            QubitId::new((start + len - 1 - i) as u32),
        )
        .unwrap();
    }
    for i in (0..len).rev() {
        for j in (i + 1..len).rev() {
            let angle = -PI / (1u64 << (j - i)) as f64;
            controlled_phase::<TOTAL>(c, start + j, start + i, angle);
        }
        c.h(QubitId::new((start + i) as u32)).unwrap();
    }
}
