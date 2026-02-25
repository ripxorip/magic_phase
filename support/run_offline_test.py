#!/usr/bin/env python3
"""
Offline test harness: runs both Python and C++ on audio file sets,
compares results, and generates a summary report.

Usage:
    # Run all built-in test sets
    python support/run_offline_test.py

    # Custom: first file is reference, rest are targets
    python support/run_offline_test.py input/kick.wav input/snare_top.wav input/snare_bot.wav

    # With comparison plots
    python support/run_offline_test.py --plot

Output structure:
    output/offline_test/<ref_stem>/<target_stem>/python/
    output/offline_test/<ref_stem>/<target_stem>/cpp/
    output/offline_test/summary.txt
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Project root is one level up from support/ ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CPP_BIN = ROOT / "build" / "bin" / "MagicPhaseTest"
PY_ALIGN = ROOT / "python" / "align_files.py"
PY_PLOT = ROOT / "python" / "plot_cpp_results.py"
OUTPUT_BASE = ROOT / "output" / "offline_test"

# ── Built-in test sets ──────────────────────────────────────────────────
# Each entry: list of files.  First file = reference, rest = targets.
BUILTIN_SETS = [
    [
        "input/lfwh_sm57_front.wav",
        "input/lfwh_sm57_back.wav",
        "input/lfwh_u87.wav",
    ],
    [
        "input/temperance_sm57_front.wav",
        "input/temperance_sm57_back.wav",
    ],
]


def run_cmd(cmd, label=""):
    """Run a shell command, return subprocess result."""
    print(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  [{label}] FAILED (rc={result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f"    {line}")
    return result


def parse_cpp_stdout(text):
    """Extract key metrics from C++ MagicPhaseTest console output."""
    metrics = {}
    patterns = {
        'delay_samples': r'Delay:\s+([-\d.]+)\s+samples',
        'delay_ms':      r'\(([-\d.]+)\s+ms\)',
        'correlation':   r'Correlation:\s*([-\d.]+)',
        'polarity':      r'Polarity:\s+(\S+)',
        'coherence':     r'Overall coherence:\s*([-\d.]+)',
        'phase_avg_deg': r'Average phase correction:\s*([-\d.]+)',
        'energy_gain_db': r'Sum energy gain:\s*([-\d.]+)',
        'num_frames':    r'Accumulated\s+(\d+)\s+ref frames',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def parse_python_stdout(text):
    """Extract key metrics from Python align_files.py console output."""
    metrics = {}
    patterns = {
        'delay_samples': r'Detected delay:\s+([-\d.]+)\s+samples',
        'delay_ms':      r'Detected delay:.*\(([-\d.]+)\s+ms\)',
        'correlation':   r'Correlation:\s*([-\d.]+)',
        'polarity':      r'Polarity:\s+(\S+)',
        'coherence':     r'Average coherence:\s*([-\d.]+)',
        'energy_gain_db': r'Sum energy gain:\s*([-\d.]+)',
        'corr_before':   r'Correlation:\s*([-\d.]+)\s*->',
        'corr_after':    r'Correlation:.*->\s*([-\d.]+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def load_csv_stats(csv_path):
    """Load analysis.csv and compute summary stats."""
    if not csv_path.exists():
        return {}
    corrs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            corrs.append(float(row['phase_correction_deg']))
    if not corrs:
        return {}
    import numpy as np
    arr = np.array(corrs)
    return {
        'csv_phase_min': float(np.min(arr)),
        'csv_phase_max': float(np.max(arr)),
        'csv_phase_mean': float(np.mean(arr)),
        'csv_phase_std': float(np.std(arr)),
        'csv_nonzero_bins': int(np.count_nonzero(arr)),
    }


def run_pair(name, ref_path, tar_path, output_base, skip_python=False, skip_cpp=False):
    """Run both Python and C++ on a single ref/target pair."""
    pair_dir = output_base / name
    cpp_dir = pair_dir / "cpp"
    py_dir = pair_dir / "python"
    cpp_dir.mkdir(parents=True, exist_ok=True)
    py_dir.mkdir(parents=True, exist_ok=True)

    ref_abs = Path(ref_path) if Path(ref_path).is_absolute() else (ROOT / ref_path).resolve()
    tar_abs = Path(tar_path) if Path(tar_path).is_absolute() else (ROOT / tar_path).resolve()

    if not ref_abs.exists():
        print(f"  SKIP: reference not found: {ref_abs}")
        return None
    if not tar_abs.exists():
        print(f"  SKIP: target not found: {tar_abs}")
        return None

    result = {'name': name, 'ref': str(ref_path), 'target': str(tar_path)}

    # ── Run C++ ──
    cpp_metrics = {}
    if not skip_cpp:
        if not CPP_BIN.exists():
            print(f"  SKIP C++: binary not found at {CPP_BIN}")
            print(f"  Build with: nix develop --command bash -c "
                  f"\"cmake -B build -DMP_BUILD_CLI_TEST=ON && cmake --build build --target MagicPhaseTest\"")
        else:
            r = run_cmd([str(CPP_BIN), str(ref_abs), str(tar_abs), "-o", str(cpp_dir)], "C++")
            cpp_metrics = parse_cpp_stdout(r.stdout)
            csv_stats = load_csv_stats(cpp_dir / "analysis.csv")
            cpp_metrics.update(csv_stats)
    result['cpp'] = cpp_metrics

    # ── Run Python ──
    py_metrics = {}
    if not skip_python:
        r = run_cmd([
            sys.executable, str(PY_ALIGN),
            str(ref_abs), str(tar_abs),
            "-o", str(py_dir),
            "--no-plot",
        ], "Python")
        py_metrics = parse_python_stdout(r.stdout)
    result['python'] = py_metrics

    return result


def format_val(val, fmt=".3f"):
    """Format a value for the table."""
    if val is None or val == '':
        return '-'
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def print_comparison(results, output_file=None):
    """Print a side-by-side comparison table."""
    lines = []

    def p(s=""):
        lines.append(s)

    p("=" * 90)
    p("MAGIC PHASE OFFLINE TEST RESULTS")
    p("=" * 90)

    for r in results:
        if r is None:
            continue

        p(f"\n{'─' * 90}")
        p(f"  {r['name']}")
        p(f"  ref: {r['ref']}")
        p(f"  tar: {r['target']}")
        p(f"{'─' * 90}")

        cpp = r.get('cpp', {})
        py = r.get('python', {})

        rows = [
            ("Delay (samples)",  'delay_samples', '.1f'),
            ("Delay (ms)",       'delay_ms',      '.3f'),
            ("Correlation",      'correlation',    '.4f'),
            ("Polarity",         'polarity',       's'),
            ("Coherence",        'coherence',      '.4f'),
            ("Energy gain (dB)", 'energy_gain_db', '.2f'),
            ("Frames analyzed",  'num_frames',     '.0f'),
        ]

        p(f"  {'Metric':<25s} {'C++':>15s} {'Python':>15s} {'Delta':>12s}")
        p(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*12}")

        for label, key, fmt in rows:
            cv = cpp.get(key)
            pv = py.get(key)

            cs = format_val(cv, fmt) if fmt != 's' else format_val(cv)
            ps = format_val(pv, fmt) if fmt != 's' else format_val(pv)

            delta = ''
            if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
                d = cv - pv
                delta = f"{d:+.3f}"

            p(f"  {label:<25s} {cs:>15s} {ps:>15s} {delta:>12s}")

        if cpp.get('csv_nonzero_bins') is not None:
            p(f"\n  C++ CSV: phase correction range "
              f"[{cpp.get('csv_phase_min', 0):.1f}, {cpp.get('csv_phase_max', 0):.1f}] deg, "
              f"std={cpp.get('csv_phase_std', 0):.2f}, nonzero={cpp.get('csv_nonzero_bins', 0)} bins")

    # ── Quick pass/fail heuristics ──
    p(f"\n{'=' * 90}")
    p("QUICK CHECKS")
    p("=" * 90)

    for r in results:
        if r is None:
            continue
        cpp = r.get('cpp', {})
        py = r.get('python', {})
        name = r['name']

        checks = []

        # Delay should be in same ballpark
        cd = cpp.get('delay_samples')
        pd = py.get('delay_samples')
        if cd is not None and pd is not None:
            diff = abs(cd - pd)
            ok = diff < 10
            checks.append(("Delay within 10 samples", ok, f"delta={diff:.1f}"))

        # Polarity should match
        cp = str(cpp.get('polarity', '')).upper()
        pp = str(py.get('polarity', '')).upper()
        if cp and pp:
            cp_inv = 'INVERT' in cp
            pp_inv = 'INVERT' in pp
            ok = cp_inv == pp_inv
            checks.append(("Polarity match", ok, f"C++={cp}, Py={pp}"))

        # Energy gain should be positive (alignment helped)
        ce = cpp.get('energy_gain_db')
        if ce is not None:
            ok = ce > -1.0
            checks.append(("C++ energy gain > -1dB", ok, f"{ce:.2f} dB"))

        p(f"\n  {name}:")
        for label, ok, detail in checks:
            status = "PASS" if ok else "FAIL"
            p(f"    [{status}] {label} ({detail})")

    p("")

    text = '\n'.join(lines)
    print(text)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(text)
        print(f"\nSummary written to: {output_file}")


def expand_sets_to_pairs(file_sets):
    """Expand file sets into (name, ref, target) pairs.
    First file in each set is reference, rest are targets."""
    pairs = []
    for files in file_sets:
        ref = files[0]
        ref_stem = Path(ref).stem
        for tar in files[1:]:
            tar_stem = Path(tar).stem
            name = f"{ref_stem}/{tar_stem}"
            pairs.append((name, ref, tar))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Run offline C++ vs Python comparison tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Run all built-in test sets
              python support/run_offline_test.py

              # Custom set: first file = reference, rest = targets
              python support/run_offline_test.py input/kick.wav input/snare_top.wav input/snare_bot.wav

              # Multiple sets via --set (can repeat)
              python support/run_offline_test.py \\
                  --set input/kick.wav input/snare_top.wav input/snare_bot.wav \\
                  --set input/ref.wav input/target.wav

              # With plots
              python support/run_offline_test.py --plot

              # C++ only
              python support/run_offline_test.py --cpp-only
        """))
    parser.add_argument('files', nargs='*',
                        help='Audio files: first = reference, rest = targets')
    parser.add_argument('--set', action='append', nargs='+', metavar='FILE',
                        dest='extra_sets',
                        help='Additional file set (first = ref, rest = targets). Repeatable.')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots for each pair')
    parser.add_argument('--cpp-only', action='store_true',
                        help='Only run C++ (skip Python)')
    parser.add_argument('--python-only', action='store_true',
                        help='Only run Python (skip C++)')
    parser.add_argument('--output', '-o', type=Path, default=OUTPUT_BASE,
                        help=f'Output base directory (default: {OUTPUT_BASE.relative_to(ROOT)})')
    args = parser.parse_args()

    output_base = args.output
    output_base.mkdir(parents=True, exist_ok=True)

    # Collect file sets
    file_sets = []

    if args.files:
        if len(args.files) < 2:
            parser.error("Need at least 2 files (reference + target)")
        file_sets.append(args.files)

    if args.extra_sets:
        for s in args.extra_sets:
            if len(s) < 2:
                parser.error(f"Each --set needs at least 2 files, got: {s}")
            file_sets.append(s)

    # Fall back to built-in sets if nothing provided
    if not file_sets:
        file_sets = BUILTIN_SETS

    pairs = expand_sets_to_pairs(file_sets)

    print(f"Running {len(pairs)} pair(s) from {len(file_sets)} set(s)...\n")

    results = []
    for name, ref, tar in pairs:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        r = run_pair(name, ref, tar, output_base,
                     skip_python=args.cpp_only,
                     skip_cpp=args.python_only)
        results.append(r)

    # ── Summary ──
    summary_path = output_base / "summary.txt"
    print_comparison(results, summary_path)

    # ── Save JSON for programmatic access ──
    json_path = output_base / "results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"JSON results: {json_path}")

    # ── Plots ──
    if args.plot:
        print(f"\nGenerating comparison plots...")
        for r in results:
            if r is None:
                continue
            pair_dir = output_base / r['name']
            cpp_dir = pair_dir / "cpp"
            py_dir = pair_dir / "python"
            plot_path = pair_dir / "comparison.png"

            cmd = [sys.executable, str(PY_PLOT), str(cpp_dir)]
            if py_dir.exists() and (py_dir / "ref_mono.wav").exists():
                cmd.append(str(py_dir))
            cmd.extend(["--save", str(plot_path)])

            run_cmd(cmd, f"Plot {r['name']}")
            if plot_path.exists():
                print(f"  -> {plot_path}")


if __name__ == '__main__':
    main()
