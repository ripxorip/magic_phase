#!/usr/bin/env python3
"""Run a VST3 integration test and print a results table.

Usage:
    python run_test.py                          # Run default test
    python run_test.py path/to/test.json        # Run specific test
"""

import argparse
import json
import subprocess
import sys
import os
import platform
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
if platform.system() == "Windows":
    HARNESS = ROOT / "build" / "bin" / "Release" / "VST3TestHarness.exe"
else:
    HARNESS = ROOT / "build" / "bin" / "VST3TestHarness"
DEFAULT_TEST = ROOT / "tests" / "integration" / "lfwh_sm57_vs_u87.json"


def run(test_file: Path):
    if not HARNESS.exists():
        print(f"Harness not found: {HARNESS}")
        print("Build with: cmake --build build --config Release --target VST3TestHarness")
        return 1

    if not test_file.exists():
        print(f"Test definition not found: {test_file}")
        return 1

    # Output dir based on test name
    test_name = test_file.stem
    out_dir = ROOT / "results" / test_name
    result_file = out_dir / "result.json"

    # Run harness
    cmd = [str(HARNESS), "--test", str(test_file), "--output-dir", str(out_dir), "--result", str(result_file)]
    print(f"Running VST3 integration test: {test_name}\n")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Show harness output (filter to key lines)
    for line in proc.stdout.splitlines():
        if any(k in line for k in ["Found plugin", "Alignment", "Timing", "===", "Triggered", "Waiting", "Loaded:", "Set instance"]):
            print(f"  {line.strip()}")

    if proc.returncode != 0 and not result_file.exists():
        print(f"\nHarness failed:\n{proc.stderr}")
        return 1

    # Load results
    result = json.loads(result_file.read_text())
    test_def = json.loads(test_file.read_text())
    expected = test_def.get("expected", {})

    # Find all target tracks
    targets = [t for t in result["tracks"] if t["role"] == "target" and "results" in t]

    if not targets:
        print("\nNo target track results found.")
        return 1

    timing = result.get("timing", {})

    # Print table
    W = 66
    print()
    print("=" * W)
    print(f"  MAGIC PHASE  -  VST3 Integration Test")
    print("=" * W)
    print(f"  Test:    {result['test']}")
    print(f"  Config:  {int(result['config']['sample_rate'])} Hz / {int(result['config']['buffer_size'])} buf")
    print(f"  Tracks:  {len(targets)} target(s)")

    all_pass = True

    for idx, target in enumerate(targets):
        r = target["results"]
        track_name = Path(target.get("input_file", f"Track {idx+1}")).stem

        print("-" * W)
        print(f"  TARGET: {track_name}")
        print("-" * W)
        print(f"  {'Metric':<22} {'Actual':>10} {'Expected':>12}  {'':>6}")
        print("-" * W)

        sub = r.get('delay_sub_sample', r['delay_samples'])  # Fallback for older results
        rows = [
            ("Alignment",      r["alignment_state"],          expected.get("alignment_state", ""),       None),
            ("Delay (samples)", f"{r['delay_samples']:.1f} ({sub:.2f})",  fmt_expected(expected.get("delay_samples")), check(r["delay_samples"], expected.get("delay_samples"))),
            ("Delay (ms)",      f"{r['delay_ms']:.2f}",       fmt_expected(expected.get("delay_ms")),      check(r["delay_ms"], expected.get("delay_ms"))),
            ("Correlation",     f"{r['correlation']:.3f}",     fmt_min(expected.get("correlation")),        check_min(r["correlation"], expected.get("correlation"))),
            ("Coherence",       f"{r['coherence']:.3f}",       fmt_min(expected.get("coherence")),          check_min(r["coherence"], expected.get("coherence"))),
            ("Phase (deg)",     f"{r['phase_degrees']:.1f}",   "",                                         None),
            ("Polarity",        "INV" if r["polarity_inverted"] else "normal", expected.get("polarity", ""), check_eq("inverted" if r["polarity_inverted"] else "normal", expected.get("polarity"))),
            ("Time corr",       "ON" if r["time_correction_on"] else "off",    "",                          None),
            ("Phase corr",      "ON" if r["phase_correction_on"] else "off",   "",                          None),
        ]

        for label, actual, exp_str, passed in rows:
            mark = ""
            if passed is True:
                mark = "PASS"
            elif passed is False:
                mark = "FAIL"
                all_pass = False
            print(f"  {label:<22} {actual:>10} {exp_str:>12}  {mark:>6}")

    print("-" * W)
    print(f"  Timing: load={timing.get('plugin_load_ms',0):.0f}ms"
          f"  play={timing.get('playback_ms',0):.0f}ms"
          f"  analysis={timing.get('analysis_wait_ms',0):.0f}ms"
          f"  total={timing.get('total_ms',0):.0f}ms")
    print("=" * W)

    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED")

    print("=" * W)
    print(f"  Output: {out_dir}")
    print("=" * W)
    return 0 if all_pass else 1


def fmt_expected(spec):
    if not spec:
        return ""
    if "value" in spec:
        return f"{spec['value']} +/-{spec.get('tolerance', 0)}"
    return ""


def fmt_min(spec):
    if not spec:
        return ""
    if "min" in spec:
        return f">= {spec['min']}"
    return ""


def check(actual, spec):
    if not spec or "value" not in spec:
        return None
    tol = spec.get("tolerance", 0)
    return abs(actual - spec["value"]) <= tol


def check_min(actual, spec):
    if not spec or "min" not in spec:
        return None
    return actual >= spec["min"]


def check_eq(actual, expected):
    if not expected:
        return None
    return actual == expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VST3 integration test")
    parser.add_argument("test_file", nargs="?", type=Path, default=DEFAULT_TEST,
                        help="Path to test definition JSON (default: lfwh_sm57_vs_u87.json)")
    args = parser.parse_args()
    sys.exit(run(args.test_file))
