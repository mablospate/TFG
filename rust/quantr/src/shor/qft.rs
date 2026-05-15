//! QFT and inverse QFT for the quantr Shor backend.
//!
//! Uses `Gate::CRk(k, ctrl)` with a signed `k`: negative `k` applies the
//! inverse rotation, matching quantr's `CRk` semantics.

use quantr::{Circuit, Gate, QuantrError};

/// Apply a standard QFT to qubits `[start, start+len)`.
#[allow(dead_code)]
pub fn apply_qft(qc: &mut Circuit, start: usize, len: usize) -> Result<(), QuantrError> {
    for i in 0..len {
        qc.add_gate(Gate::H, start + i)?;
        for j in (i + 1)..len {
            let k = (j - i + 1) as i32;
            // CRk(k, ctrl) applied at target rotates by 2*pi / 2^k.
            qc.add_gate(Gate::CRk(k, start + j), start + i)?;
        }
    }
    for i in 0..(len / 2) {
        qc.add_gate(Gate::Swap(start + len - 1 - i), start + i)?;
    }
    Ok(())
}

/// Apply an inverse QFT to qubits `[start, start+len)`.
pub fn apply_inverse_qft(qc: &mut Circuit, start: usize, len: usize) -> Result<(), QuantrError> {
    for i in 0..(len / 2) {
        qc.add_gate(Gate::Swap(start + len - 1 - i), start + i)?;
    }
    for i in (0..len).rev() {
        for j in ((i + 1)..len).rev() {
            let k = (j - i + 1) as i32;
            // Negative k -> rotation of -2*pi / 2^k (quantr CRk reads i32 with sign).
            qc.add_gate(Gate::CRk(-k, start + j), start + i)?;
        }
        qc.add_gate(Gate::H, start + i)?;
    }
    Ok(())
}
