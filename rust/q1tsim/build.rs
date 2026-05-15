//! Build script for q1tsim-bench.
//!
//! Upstream `q1tsim 0.5` is declared as `crate-type = ["dylib"]`, which forces
//! the bin to dynamically link `libstd-*.dylib`. The Rust toolchain does not
//! emit an rpath for the sysroot, so without help the binary aborts at startup
//! with "Library not loaded: @rpath/libstd-...dylib".
//!
//! We resolve the sysroot via `rustc --print sysroot` at build time and embed
//! it as an rpath in each [[bin]]. This keeps the workaround self-contained
//! and avoids polluting workspace-wide RUSTFLAGS.

use std::process::Command;

fn main() {
    let sysroot = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string()))
        .arg("--print")
        .arg("sysroot")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    let target = std::env::var("TARGET").unwrap_or_default();

    if let Some(sysroot) = sysroot {
        let rustlib = format!("{}/lib/rustlib/{}/lib", sysroot, target);
        // Apply rpath to both bins in this crate.
        for bin in ["q1tsim-grover", "q1tsim-shor"] {
            println!("cargo:rustc-link-arg-bin={}=-Wl,-rpath,{}", bin, rustlib);
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
