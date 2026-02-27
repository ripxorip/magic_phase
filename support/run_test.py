#!/usr/bin/env python3
"""Run a VST3 integration test and print a results table."""

import json
import subprocess
import sys
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
HARNESS = ROOT / "build" / "bin" / "Release" / "VST3TestHarness.exe"
TEST_DEF = ROOT / "tests" / "integration" / "lfwh_sm57_vs_u87.json"
OUT_DIR = ROOT / "results" / "latest"
RESULT = OUT_DIR / "result.json"


def run():
    if not HARNESS.exists():
        print(f"Harness not found: {HARNESS}")
        print("Build with: cmake --build build --config Release --target VST3TestHarness")
        return 1

    if not TEST_DEF.exists():
        print(f"Test definition not found: {TEST_DEF}")
        return 1

    # Run harness
    cmd = [str(HARNESS), "--test", str(TEST_DEF), "--output-dir", str(OUT_DIR), "--result", str(RESULT)]
    print("Running VST3 integration test...\n")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Show harness output (filter to key lines)
    for line in proc.stdout.splitlines():
        if any(k in line for k in ["Found plugin", "Alignment", "Timing", "===", "Triggered", "Waiting", "Loaded:", "Set instance"]):
            print(f"  {line.strip()}")

    if proc.returncode != 0 and not RESULT.exists():
        print(f"\nHarness failed:\n{proc.stderr}")
        return 1

    # Load results
    result = json.loads(RESULT.read_text())
    test_def = json.loads(TEST_DEF.read_text())
    expected = test_def.get("expected", {})

    # Find target track
    target = None
    for t in result["tracks"]:
        if t["role"] == "target":
            target = t
            break

    if not target or "results" not in target:
        print("\nNo target track results found.")
        return 1

    r = target["results"]
    timing = result.get("timing", {})

    # Print table
    W = 58
    print()
    print("=" * W)
    print(f"  MAGIC PHASE  -  VST3 Integration Test")
    print("=" * W)
    print(f"  Test:    {result['test']}")
    print(f"  Config:  {int(result['config']['sample_rate'])} Hz / {int(result['config']['buffer_size'])} buf")
    print("-" * W)
    print(f"  {'Metric':<22} {'Actual':>10} {'Expected':>12}  {'':>6}")
    print("-" * W)

    rows = [
        ("Alignment",      r["alignment_state"],          expected.get("alignment_state", ""),       None),
        ("Delay (samples)", f"{r['delay_samples']:.1f}",  fmt_expected(expected.get("delay_samples")), check(r["delay_samples"], expected.get("delay_samples"))),
        ("Delay (ms)",      f"{r['delay_ms']:.2f}",       fmt_expected(expected.get("delay_ms")),      check(r["delay_ms"], expected.get("delay_ms"))),
        ("Correlation",     f"{r['correlation']:.3f}",     fmt_min(expected.get("correlation")),        check_min(r["correlation"], expected.get("correlation"))),
        ("Coherence",       f"{r['coherence']:.3f}",       fmt_min(expected.get("coherence")),          check_min(r["coherence"], expected.get("coherence"))),
        ("Phase (deg)",     f"{r['phase_degrees']:.1f}",   "",                                         None),
        ("Polarity",        "INV" if r["polarity_inverted"] else "normal", expected.get("polarity", ""), check_eq("inverted" if r["polarity_inverted"] else "normal", expected.get("polarity"))),
        ("Time corr",       "ON" if r["time_correction_on"] else "off",    "",                          None),
        ("Phase corr",      "ON" if r["phase_correction_on"] else "off",   "",                          None),
    ]

    all_pass = True
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
    sys.exit(run())
