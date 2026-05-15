//! qcgpu-bench library: shared implementations of Grover and Shor for the
//! qcgpu OpenCL-based simulator. The `bin/` entrypoints are thin wrappers
//! around `grover::run` and `shor::run`.

pub mod grover;
pub mod shor;
