// QFT and inverse QFT for the order-finding control register.

use std::f64::consts::PI;

use q1tsim::circuit::Circuit;
use q1tsim::error::Result as QResult;
use q1tsim::gates::{CU1, Swap};

pub fn apply_qft(c: &mut Circuit, qubits: &[usize]) -> QResult<()> {
    let n = qubits.len();
    for i in 0..n {
        c.h(qubits[i])?;
        for j in (i + 1)..n {
            let k = (j - i + 1) as i32;
            let angle = PI / 2f64.powi(k - 1);
            c.add_gate(CU1::new(angle), &[qubits[j], qubits[i]])?;
        }
    }
    for i in 0..n / 2 {
        c.add_gate(Swap::new(), &[qubits[i], qubits[n - 1 - i]])?;
    }
    Ok(())
}

pub fn apply_inverse_qft(c: &mut Circuit, qubits: &[usize]) -> QResult<()> {
    let n = qubits.len();
    for i in 0..n / 2 {
        c.add_gate(Swap::new(), &[qubits[i], qubits[n - 1 - i]])?;
    }
    for i in (0..n).rev() {
        for j in ((i + 1)..n).rev() {
            let k = (j - i + 1) as i32;
            let angle = -PI / 2f64.powi(k - 1);
            c.add_gate(CU1::new(angle), &[qubits[j], qubits[i]])?;
        }
        c.h(qubits[i])?;
    }
    Ok(())
}

#[allow(dead_code)]
fn _keep_apply_qft_alive() {
    // Silences dead_code warning for apply_qft which is exported for symmetry.
    let _ = apply_qft as fn(&mut Circuit, &[usize]) -> QResult<()>;
}
