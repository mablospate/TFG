// Grover's search for q1tsim 0.5.
//
// q1tsim natively provides h, x, z, cx as Circuit methods and CCX, CCZ, CZ,
// Swap, CU1 as gate structs that we register via add_gate.
//
// Multi-controlled-Z (n >= 4 controls) is realised with the Toffoli-ladder
// ancilla decomposition (Strategy B in implementation guide section 7.10).
// Ancilla qubits live at indices [n .. n + (n-2)] and are uncomputed after each
// use so they remain in |0> and do not pollute the histogram.

use std::collections::HashMap;
use std::error::Error;
use std::f64::consts::PI;
use std::time::Instant;

use clap::Parser;
use q1tsim::circuit::Circuit;
use q1tsim::error::Result as QResult;
use q1tsim::gates::{CCX, CCZ, CZ};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "grover", about = "q1tsim Grover benchmark")]
pub struct Args {
    /// Number of search qubits.
    #[arg(long)]
    pub n: usize,

    /// Target state (integer in [0, 2^n)).
    #[arg(long)]
    pub target: u64,

    /// Number of shots.
    #[arg(long, default_value_t = 1024)]
    pub shots: usize,

    /// Override Grover iteration count (default: floor(pi/4 * sqrt(2^n))).
    #[arg(long)]
    pub iterations: Option<usize>,
}

#[derive(Serialize)]
pub struct Output {
    pub framework: &'static str,
    pub framework_version: &'static str,
    pub algorithm: &'static str,
    pub n: usize,
    pub target: u64,
    pub shots: usize,
    pub iterations: usize,
    pub found: u64,
    pub time_ms: f64,
    pub mem_mb: f64,
    pub distribution: HashMap<String, usize>,
}

pub fn ancilla_count(n_controls: usize) -> usize {
    if n_controls <= 2 { 0 } else { n_controls - 2 }
}

pub fn apply_mcx(
    c: &mut Circuit,
    ctrls: &[usize],
    tgt: usize,
    ancillas: &[usize],
) -> QResult<()> {
    let k = ctrls.len();
    match k {
        0 => c.x(tgt)?,
        1 => c.cx(ctrls[0], tgt)?,
        2 => c.add_gate(CCX::new(), &[ctrls[0], ctrls[1], tgt])?,
        _ => {
            // Toffoli ladder.
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

pub fn apply_mcz(c: &mut Circuit, qubits: &[usize], ancillas: &[usize]) -> QResult<()> {
    let n = qubits.len();
    match n {
        0 => {}
        1 => c.z(qubits[0])?,
        2 => c.add_gate(CZ::new(), &[qubits[0], qubits[1]])?,
        3 => c.add_gate(CCZ::new(), &[qubits[0], qubits[1], qubits[2]])?,
        _ => {
            let target = qubits[n - 1];
            let ctrls: Vec<usize> = qubits[..n - 1].to_vec();
            c.h(target)?;
            apply_mcx(c, &ctrls, target, ancillas)?;
            c.h(target)?;
        }
    }
    Ok(())
}

pub fn build_oracle(c: &mut Circuit, n: usize, target: u64, ancillas: &[usize]) -> QResult<()> {
    let qubits: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(i)?;
        }
    }
    apply_mcz(c, &qubits, ancillas)?;
    for i in 0..n {
        if (target >> i) & 1 == 0 {
            c.x(i)?;
        }
    }
    Ok(())
}

pub fn build_diffuser(c: &mut Circuit, n: usize, ancillas: &[usize]) -> QResult<()> {
    let qubits: Vec<usize> = (0..n).collect();
    for i in 0..n {
        c.h(i)?;
    }
    for i in 0..n {
        c.x(i)?;
    }
    apply_mcz(c, &qubits, ancillas)?;
    for i in 0..n {
        c.x(i)?;
    }
    for i in 0..n {
        c.h(i)?;
    }
    Ok(())
}

pub fn grover_circuit(n: usize, target: u64, iterations: usize) -> QResult<Circuit> {
    let n_anc = ancilla_count(n);
    let total = n + n_anc;
    let mut circuit = Circuit::new(total, n);
    let ancillas: Vec<usize> = (n..n + n_anc).collect();

    for i in 0..n {
        circuit.h(i)?;
    }
    for _ in 0..iterations {
        build_oracle(&mut circuit, n, target, &ancillas)?;
        build_diffuser(&mut circuit, n, &ancillas)?;
    }
    for q in 0..n {
        circuit.measure(q, q)?;
    }
    Ok(circuit)
}

fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmHWM:") {
                if let Some(kb) = line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()) {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        let pid = std::process::id();
        if let Ok(out) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(kb) = s.trim().parse::<u64>() {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }
    0.0
}

pub fn run_with_args(args: &Args) -> Result<Output, Box<dyn Error>> {
    let max_target = 1u64 << args.n;
    if args.target >= max_target {
        return Err(format!(
            "target {} out of range for n={} qubits (max {})",
            args.target,
            args.n,
            max_target - 1
        )
        .into());
    }
    let iterations = args
        .iterations
        .unwrap_or_else(|| {
            ((PI / 4.0) * (2f64.powi(args.n as i32)).sqrt()).floor() as usize
        })
        .max(1);

    eprintln!(
        "Start Grover search for |{}> in {}-qubit space ({} iterations)",
        args.target, args.n, iterations
    );
    let start = Instant::now();
    let mut circuit = grover_circuit(args.n, args.target, iterations)
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })?;
    circuit
        .execute(args.shots)
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })?;
    let dist = circuit
        .histogram_string()
        .map_err(|e| -> Box<dyn Error> { e.to_string().into() })?;
    let elapsed = start.elapsed();

    let (best_bs, _) = dist
        .iter()
        .max_by_key(|(_, c)| *c)
        .ok_or("empty histogram")?;
    let found = u64::from_str_radix(best_bs, 2)?;
    let mem_mb = peak_rss_mb();

    let total_shots: usize = dist.values().sum();
    let target_bs = format!("{:0width$b}", args.target, width = args.n);
    let prob = dist.get(&target_bs).copied().unwrap_or(0) as f64 / total_shots.max(1) as f64;
    eprintln!("Found target state |{}> with probability {:.2}%", found, prob * 100.0);

    Ok(Output {
        framework: "q1tsim",
        framework_version: env!("CARGO_PKG_VERSION"),
        algorithm: "grover",
        n: args.n,
        target: args.target,
        shots: args.shots,
        iterations,
        found,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        mem_mb,
        distribution: dist,
    })
}

pub fn run() -> ! {
    let args = Args::parse();
    match run_with_args(&args) {
        Ok(out) => {
            println!("{}", serde_json::to_string(&out).expect("serialize"));
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a Grover circuit and return the histogram of measurement outcomes.
    fn run_grover(n: usize, target: u64, iterations: usize, shots: usize) -> HashMap<String, usize> {
        let mut circuit = grover_circuit(n, target, iterations).expect("circuit build failed");
        circuit.execute(shots).expect("execute failed");
        circuit.histogram_string().expect("histogram failed")
    }

    /// Convert an integer target to the q1tsim histogram bitstring.
    /// histogram_string() formats with format!("{:0width$b}", key) where the
    /// classical register has `n` bits; cbit i mirrors qubit i, so the
    /// rightmost char in the string corresponds to qubit 0 (LSB-first layout
    /// when read right-to-left, MSB-first when read left-to-right as the
    /// formatted integer key).
    fn target_bitstring(n: usize, target: u64) -> String {
        format!("{:0width$b}", target, width = n)
    }

    #[test]
    fn test_grover_finds_target_n3() {
        // target=5 (binary 101) with 3 qubits, iterations near pi/4 * sqrt(8) ~= 2
        let shots = 200;
        let dist = run_grover(3, 5, 2, shots);
        let expected = target_bitstring(3, 5);
        let (best_bs, best_count) = dist
            .iter()
            .max_by_key(|(_, c)| *c)
            .expect("non-empty histogram");
        assert_eq!(
            best_bs, &expected,
            "expected mode {} got {} (dist={:?})",
            expected, best_bs, dist
        );
        // For n=3, Grover with 2 iterations has > 90% amplitude on the target,
        // but allow a generous lower bound for randomness.
        assert!(
            *best_count * 2 > shots,
            "target {} only got {} of {} shots (dist={:?})",
            expected,
            best_count,
            shots,
            dist
        );
    }

    #[test]
    fn test_grover_finds_target_zero() {
        let shots = 200;
        let dist = run_grover(3, 0, 2, shots);
        let expected = target_bitstring(3, 0);
        let (best_bs, best_count) = dist
            .iter()
            .max_by_key(|(_, c)| *c)
            .expect("non-empty histogram");
        assert_eq!(best_bs, &expected);
        assert!(*best_count * 2 > shots);
    }

    #[test]
    fn test_grover_finds_target_n4() {
        // n=4, target=11 (binary 1011), iterations near pi/4 * sqrt(16) ~= 3
        let shots = 200;
        let dist = run_grover(4, 11, 3, shots);
        let expected = target_bitstring(4, 11);
        let (best_bs, best_count) = dist
            .iter()
            .max_by_key(|(_, c)| *c)
            .expect("non-empty histogram");
        assert_eq!(
            best_bs, &expected,
            "expected mode {} got {} (dist={:?})",
            expected, best_bs, dist
        );
        // For n=4 success probability is ~96% with 3 iterations.
        assert!(
            *best_count * 2 > shots,
            "target {} only got {} of {} shots",
            expected,
            best_count,
            shots
        );
    }
}
