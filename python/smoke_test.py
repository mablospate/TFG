"""Smoke test suite — verifies that all framework measurements are captured correctly."""
from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).parent.parent

PYTHON_FRAMEWORKS = ["qiskit", "cirq", "cudaq", "qdislib"]
RUST_GROVER_BINS  = ["q1tsim-grover", "quantr-grover", "quantrs2-grover", "qcgpu-grover"]
RUST_SHOR_BINS    = ["q1tsim-shor", "quantr-shor", "quantrs2-shor"]

ARCH = platform.machine().lower()

PYTHON_GROVER_NONZERO = {
    "wall_time_median_ms", "wall_time_mean_ms",
    "peak_memory_rss_mb", "startup_time_ms", "simulation_time_ms",
    "num_shots", "n_repetitions", "n_qubits",
}
PYTHON_GROVER_PRESENT = {
    "status", "framework", "algorithm", "contributor_name",
    "hostname", "os", "cpu_model", "cpu_freq_mhz", "ram_total_gb",
    "runtime_version", "framework_version",
    "wall_time_iqr_ms", "wall_time_std_ms", "cv",
    "build_time_ms", "raw_times_ms", "cpu_percent_mean",
}

PYTHON_SHOR_EXTRA_NONZERO = {"n_to_factor"}
PYTHON_SHOR_EXTRA_PRESENT = {"factor_found"}

QDISLIB_CUTTING_NONZERO = {"cutting_wall_time_ms", "cutting_find_time_ms"}
QDISLIB_CUTTING_PRESENT = {"cutting_expectation_value"}

RUST_GROVER_NONZERO = {"time_ms", "mem_mb"}
RUST_GROVER_PRESENT = {"distribution", "framework_version"}
RUST_SHOR_NONZERO = {"time_ms", "mem_mb"}
RUST_SHOR_PRESENT = {"factor", "framework_version"}


def _is_zero(val) -> bool:
    return val in (0, 0.0, "", [], {}, None)


def _abbrev(val) -> str:
    if isinstance(val, list):
        return str(val[:3])
    if isinstance(val, dict):
        keys = list(val.keys())[:3]
        return str({k: val[k] for k in keys})
    if isinstance(val, str) and len(val) > 80:
        return val[:80]
    return str(val)


def _is_available(name: str) -> bool:
    if name == "cudaq":
        return ARCH not in ("aarch64", "arm64")
    if "qcgpu" in name:
        return ARCH == "x86_64"
    if name == "qdislib":
        return ARCH in ("x86_64", "amd64")
    return True


def _rust_bin(name: str) -> pathlib.Path | None:
    found = shutil.which(name)
    if found:
        p = pathlib.Path(found)
        if p.exists() and os.access(p, os.X_OK):
            return p
    for candidate in [ROOT / "bin" / name, ROOT / "target/release" / name]:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _invoke_python_worker(framework: str, algo: str, n: int) -> dict | None:
    config = {
        "n_repetitions": 3,
        "num_shots": 10,
        "algo": algo,
        "n": n,
        "contributor": "smoke_test",
        "cudaq_target": "qpp-cpu",
    }
    config_json = json.dumps(config)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", f"python.workers.{framework}_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
            text=True,
        )
        stdout, stderr = proc.communicate(config_json, timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  FAIL TIMEOUT")
        return None

    if stderr.strip():
        print(f"  [stderr] {stderr.strip()[:400]}")

    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return None

    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None

    if result.get("status") == "error":
        print(f"  FAIL worker error: {result.get('error', '?')}")
        return None

    return result


def _invoke_rust_grover(binary: pathlib.Path, n: int) -> dict | None:
    try:
        proc = subprocess.run(
            [str(binary), "--n", str(n), "--target", str(n), "--shots", "10"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None

    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        return None

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _invoke_rust_shor(binary: pathlib.Path) -> dict | None:
    try:
        proc = subprocess.run(
            [str(binary), "--N", "15", "--shots", "10", "--tries", "3"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None

    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        return None

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _check_fields(
    result: dict,
    present: set,
    nonzero: set,
    issues: list[str],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> int:
    critical = 0
    for field in sorted(present | nonzero):
        val = result.get(field)
        if val is None:
            print(f"  X {field}: MISSING")
            issues.append(f"missing:{field}")
            critical += 1
        elif field in nonzero and _is_zero(val):
            print(f"  ! {field} = {val!r}  [zero]")
            issues.append(f"zero:{field}")
            critical += 1
        else:
            print(f"  OK {field} = {_abbrev(val)}")
    if ranges:
        for field, (lo, hi) in ranges.items():
            val = result.get(field)
            if val is None:
                print(f"  X {field}: MISSING")
                issues.append(f"missing:{field}")
                critical += 1
            elif not lo <= float(val) <= hi:
                print(f"  ! {field} = {val:.4f}  [out of range {lo}–{hi}]")
                issues.append(f"range:{field}")
            else:
                print(f"  OK {field} = {val:.4f}  [in range]")
    return critical


def _test_python_framework(fw: str) -> list[tuple[str, str, str, int]]:
    rows: list[tuple[str, str, str, int]] = []

    for algo, n in [("grover", 3), ("shor", 15)]:
        print(f"\n--- {fw}/{algo} ---")
        result = _invoke_python_worker(fw, algo, n)
        if result is None:
            rows.append((fw, algo, "FAIL", 1))
            continue

        issues: list[str] = []

        present = set(PYTHON_GROVER_PRESENT)
        nonzero = set(PYTHON_GROVER_NONZERO)
        if algo == "shor":
            nonzero |= PYTHON_SHOR_EXTRA_NONZERO
            present |= PYTHON_SHOR_EXTRA_PRESENT
            present.discard("build_time_ms")
        if fw == "qdislib":
            nonzero |= QDISLIB_CUTTING_NONZERO
            present |= QDISLIB_CUTTING_PRESENT

        ranges = {"jsd": (0.0, 1.0)} if algo == "grover" else {"success_rate": (0.0, 1.0)}
        crit = _check_fields(result, present, nonzero, issues, ranges=ranges)

        status = "FAIL" if crit > 0 else ("WARN" if issues else "PASS")
        rows.append((fw, algo, status, len(issues)))

    return rows


def _test_rust_binary(name: str, algo: str) -> tuple[str, str, str, int]:
    print(f"\n--- {name}/{algo} ---")
    bin_path = _rust_bin(name)
    if bin_path is None:
        print("  - binary not found or not executable (SKIP)")
        return (name, algo, "SKIP", 0)

    result = _invoke_rust_grover(bin_path, 3) if algo == "grover" else _invoke_rust_shor(bin_path)
    if result is None:
        return (name, algo, "FAIL", 1)

    issues: list[str] = []
    if algo == "shor":
        nonzero = set(RUST_SHOR_NONZERO)
        present = set(RUST_SHOR_PRESENT)
    else:
        nonzero = set(RUST_GROVER_NONZERO)
        present = set(RUST_GROVER_PRESENT)

    crit = _check_fields(result, present, nonzero, issues)
    status = "FAIL" if crit > 0 else ("WARN" if issues else "PASS")
    return (name, algo, status, len(issues))


def _test_supabase() -> bool:
    print("\n--- Supabase connectivity ---")
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        print("  - SUPABASE_URL/KEY not set (SKIP)")
        return True
    try:
        req = urllib.request.Request(
            url,
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  OK HTTP {resp.status} reachable")
            return True
    except urllib.error.HTTPError as e:
        print(f"  OK HTTP {e.code} reachable (API returned {e.code})")
        return True
    except Exception as e:
        print(f"  X unreachable: {e}")
        return False


def _print_summary(rows: list, supabase_ok: bool) -> None:
    print("\n============================================================")
    print("  Quantum Benchmarking Smoke Test — SUMMARY")
    print("============================================================")
    print(f"  {'Framework':<22} {'Algo':<8} {'Status':<8} {'Issues':>6}")
    print("------------------------------------------------------------")

    for fw, algo, status, n_issues in rows:
        if status == "PASS":
            sym = "OK"
        elif status == "WARN":
            sym = "! "
        elif status == "SKIP":
            sym = "- "
        else:
            sym = "X "
        print(f"  {sym} {fw:<20} {algo:<8} {status:<8} {n_issues:>6}")

    sym = "OK" if supabase_ok else "X "
    print(f"  {sym} {'supabase':<20} {'http':<8} {'PASS' if supabase_ok else 'FAIL':<8} {'':>6}")
    print("============================================================")


def main() -> None:
    print("=" * 60)
    print("  Quantum Benchmarking Smoke Test")
    print(f"  Arch: {ARCH} | Python: {sys.version.split()[0]}")
    print("=" * 60)

    rows: list[tuple[str, str, str, int]] = []

    for fw in PYTHON_FRAMEWORKS:
        if not _is_available(fw):
            print(f"\n[SKIP] {fw} — not available on {ARCH}")
            rows += [(fw, "grover", "SKIP", 0), (fw, "shor", "SKIP", 0)]
        else:
            rows += _test_python_framework(fw)

    for name in RUST_GROVER_BINS:
        if not _is_available(name):
            print(f"\n[SKIP] {name} — arch restricted")
            rows.append((name, "grover", "SKIP", 0))
        else:
            rows.append(_test_rust_binary(name, "grover"))

    for name in RUST_SHOR_BINS:
        rows.append(_test_rust_binary(name, "shor"))

    supabase_ok = _test_supabase()
    _print_summary(rows, supabase_ok)

    has_failures = any(status == "FAIL" for _, _, status, _ in rows) or not supabase_ok
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
