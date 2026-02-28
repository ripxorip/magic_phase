#!/usr/bin/env python3
"""Run a Magic Phase integration test and print a results table.

Usage:
    python run_test.py path/to/test.json        # Run with VST3 harness
    python run_test.py path/to/test.json --py   # Run with Python DSP engine
    python run_test.py path/to/test.json --no-plot  # Skip plot generation

The --py flag uses the Python DSP engine instead of the C++ VST3 harness.
Outputs go to results/<test_name>_py/ with the same structure (WAVs, plots,
Reaper project) for direct comparison. Faster iteration for DSP prototyping.
"""

import argparse
import json
import subprocess
import sys
import os
import platform
from pathlib import Path

# Import plotting (same directory)
from plot_test_results import main as generate_plots
from create_reaper_project import create_rpp

import shutil

import numpy as np
import soundfile as sf
from datetime import datetime

# Python DSP engine path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


def create_normalized_audio(out_dir: Path):
    """Create normalized versions of sum.wav and raw_sum.wav for easy listening.

    Normalizes sum.wav to -1dB peak, applies same gain to raw_sum.wav
    so the relative difference is preserved for fair A/B comparison.
    """
    sum_path = out_dir / "sum.wav"
    raw_sum_path = out_dir / "raw_sum.wav"

    if not sum_path.exists() or not raw_sum_path.exists():
        return

    # Load both files
    sum_audio, sr = sf.read(str(sum_path))
    raw_sum_audio, _ = sf.read(str(raw_sum_path))

    # Calculate gain to normalize sum.wav to -1dB peak
    sum_peak = np.max(np.abs(sum_audio))
    target_peak = 10 ** (-1.0 / 20)  # -1 dB
    gain = target_peak / (sum_peak + 1e-10)

    # Apply same gain to both (keeps relative difference fair)
    sum_norm = sum_audio * gain
    raw_sum_norm = raw_sum_audio * gain

    # Clip raw_sum if it would exceed 0dB (unlikely but safe)
    raw_sum_norm = np.clip(raw_sum_norm, -1.0, 1.0)

    # Write normalized versions
    sf.write(str(out_dir / "sum_norm.wav"), sum_norm, sr)
    sf.write(str(out_dir / "raw_sum_norm.wav"), raw_sum_norm, sr)

    gain_db = 20 * np.log10(gain)
    print(f"  Created normalized audio (gain: {gain_db:+.1f} dB):")
    print(f"    - sum_norm.wav (aligned, -1dB peak)")
    print(f"    - raw_sum_norm.wav (raw, same gain for fair A/B)")

def run_python_engine(test_file: Path, out_dir: Path, result_file: Path,
                      analyze_window_s: float = 7.5):
    """Run alignment using Python DSP engine (prototyping mode).

    Produces the same output structure as the VST3 harness:
    - Per-track <name>_out.wav files
    - raw_sum.wav (reference + raw targets)
    - sum.wav (reference + aligned targets)
    - result.json (same schema as C++ harness)
    """
    import time
    from align_files import (detect_delay_xcorr, correct_delay_subsample,
                             analyze_phase_spectral, apply_phase_spectral)

    test_def = json.loads(test_file.read_text())
    test_name = test_file.stem
    sr_config = test_def.get("sample_rate", 48000)
    buffer_size = test_def.get("buffer_size", 64)

    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # Separate reference and targets
    ref_def = None
    target_defs = []
    for t in test_def["tracks"]:
        if t["role"] == "reference":
            ref_def = t
        else:
            target_defs.append(t)

    if not ref_def:
        print("ERROR: No reference track in test definition")
        return False

    # Load all audio
    def load_mono(path):
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr

    ref_path = ROOT / ref_def["file"]
    ref_audio, sr = load_mono(ref_path)
    print(f"  Reference: {ref_path.name} ({len(ref_audio)/sr:.2f}s @ {sr}Hz)")

    target_audios = []
    for td in target_defs:
        tp = ROOT / td["file"]
        ta, _ = load_mono(tp)
        target_audios.append(ta)
        print(f"  Target:    {tp.name} ({len(ta)/sr:.2f}s)")

    # Common length across all tracks
    common_len = min(len(ref_audio), *[len(t) for t in target_audios])
    ref_audio = ref_audio[:common_len]
    target_audios = [t[:common_len] for t in target_audios]

    # Analysis window (match C++ VST3 harness default)
    analyze_samples = min(int(analyze_window_s * sr), common_len)
    ref_analyze = ref_audio[:analyze_samples]

    t_load = time.perf_counter()
    print(f"\n  Loaded {1 + len(target_defs)} tracks in {(t_load - t_start)*1000:.0f}ms"
          f" (common length: {common_len/sr:.2f}s, {common_len} samples)")
    print(f"  Analysis window: {analyze_samples/sr:.1f}s / {common_len/sr:.1f}s"
          f" ({analyze_samples} samples)")

    # Write reference output (pass-through, same as VST3 harness)
    ref_out = out_dir / f"{ref_path.stem}_out.wav"
    sf.write(str(ref_out), ref_audio, sr)

    # Initialize sums
    raw_sum = ref_audio.copy()
    aligned_sum = ref_audio.copy()

    # Result tracks (reference first)
    result_tracks = [{
        "name": "Track 0",
        "role": "reference",
        "input_file": ref_def["file"],
        "output_file": str(ref_out.resolve()),
        "slot": 0
    }]

    # Process each target
    for idx, (td, tar) in enumerate(zip(target_defs, target_audios)):
        track_name = Path(td["file"]).stem
        mode = td.get("mode", "phi")

        print(f"\n  {'-'*54}")
        print(f"  TARGET {idx+1}/{len(target_defs)}: {track_name} (mode={mode})")
        print(f"  {'-'*54}")

        # ── Delay Detection (on analysis window) ──
        tar_analyze = tar[:analyze_samples]
        t0 = time.perf_counter()
        delay, corr, pol = detect_delay_xcorr(ref_analyze, tar_analyze, sr)
        delay_ms = delay / sr * 1000
        t1 = time.perf_counter()

        pol_str = 'INVERTED' if pol < 0 else 'Normal'
        print(f"    Delay:       {delay} samples ({delay_ms:.3f} ms)")
        print(f"    Correlation: {corr:.4f}")
        print(f"    Polarity:    {pol_str}")
        print(f"    (xcorr: {(t1-t0)*1000:.0f}ms)")

        # ── Time Correction (full file, analyzed on window) ──
        corrected = correct_delay_subsample(tar, delay, sr)
        corrected_analyze = correct_delay_subsample(tar_analyze, delay, sr)
        if pol < 0:
            corrected = -corrected
            corrected_analyze = -corrected_analyze

        # ── Spectral Phase Correction ──
        bands_48 = [0.0] * 48
        phase_deg = 0.0
        avg_coh = 0.0
        phase_on = False

        if mode == "phi":
            t2 = time.perf_counter()

            # Analyze on window, apply to full file
            f_bins, phase_corr, coh = analyze_phase_spectral(
                ref_analyze, corrected_analyze, sr,
                coherence_threshold=0.4, max_correction_deg=120
            )
            corrected = apply_phase_spectral(corrected, phase_corr, sr)
            corrected = corrected[:common_len]

            t3 = time.perf_counter()
            phase_on = True

            # Metrics
            mask20 = f_bins > 20
            avg_coh = float(np.mean(coh[mask20]))
            phase_deg = float(np.mean(np.abs(np.degrees(phase_corr[mask20]))))

            # 48-band coherence for result (matches C++ GUI bands)
            edges = np.logspace(np.log10(20), np.log10(sr/2), 49)
            for b in range(48):
                m = (f_bins >= edges[b]) & (f_bins < edges[b+1])
                if np.any(m):
                    bands_48[b] = float(np.mean(coh[m]))

            print(f"    Phase corr:  {np.degrees(np.min(phase_corr)):.1f}° to "
                  f"{np.degrees(np.max(phase_corr)):.1f}°")
            print(f"    Avg cohere:  {avg_coh:.3f}")
            print(f"    (spectral: {(t3-t2)*1000:.0f}ms)")

            # Detailed per-region breakdown
            print(f"\n    Coherence / correction by frequency region:")
            regions = [
                ("Sub/Bass", 20, 200),
                ("Low-Mid",  200, 800),
                ("Mid",      800, 2500),
                ("Hi-Mid",   2500, 6000),
                ("High",     6000, sr/2),
            ]
            for rname, lo, hi in regions:
                m = (f_bins >= lo) & (f_bins < hi)
                if np.any(m):
                    rc = np.mean(coh[m])
                    rp = np.mean(np.abs(np.degrees(phase_corr[m])))
                    filled = int(rc * 20)
                    bar = '#' * filled + '.' * (20 - filled)
                    print(f"      {rname:>8}: {bar} {rc:.3f}  corr={rp:.1f}°")

        # ── Accumulate sums ──
        raw_sum += tar
        aligned_sum += corrected[:common_len]

        # ── Quality metrics ──
        e_raw = np.sum((ref_audio + tar) ** 2)
        e_aligned = np.sum((ref_audio + corrected[:common_len]) ** 2)
        gain_db = 10 * np.log10(e_aligned / (e_raw + 1e-20))
        print(f"\n    Sum energy gain: {gain_db:+.1f} dB")

        # ── Write output ──
        out_path = out_dir / f"{track_name}_out.wav"
        sf.write(str(out_path), corrected[:common_len], sr)

        # ── Result entry ──
        result_tracks.append({
            "name": "Track 0",
            "role": "target",
            "input_file": td["file"],
            "output_file": str(out_path.resolve()),
            "slot": idx + 1,
            "mode": mode,
            "results": {
                "alignment_state": "ALIGNED",
                "delay_samples": float(delay),
                "delay_sub_sample": float(delay),
                "delay_ms": float(delay_ms),
                "correlation": float(corr),
                "coherence": float(avg_coh),
                "phase_degrees": float(phase_deg),
                "polarity_inverted": pol < 0,
                "time_correction_on": True,
                "phase_correction_on": phase_on,
                "spectral_bands": bands_48,
            }
        })

    # ── Write sum files ──
    sf.write(str(out_dir / "raw_sum.wav"), raw_sum, sr)
    sf.write(str(out_dir / "sum.wav"), aligned_sum, sr)

    t_end = time.perf_counter()
    t_process = t_end - t_load

    # ── Overall summary ──
    total_e_raw = np.sum(raw_sum ** 2)
    total_e_aligned = np.sum(aligned_sum ** 2)
    total_gain = 10 * np.log10(total_e_aligned / (total_e_raw + 1e-20))
    print(f"\n  {'='*54}")
    print(f"  TOTAL sum energy gain: {total_gain:+.1f} dB")
    print(f"  Processing time: {t_process*1000:.0f}ms")
    print(f"  {'='*54}")

    # ── Write result.json ──
    result = {
        "test": test_name,
        "timestamp": datetime.now().astimezone().isoformat(),
        "config": {
            "plugin_path": "Python DSP engine",
            "sample_rate": float(sr),
            "buffer_size": buffer_size,
            "plugin_loaded": True,
            "num_instances": len(test_def["tracks"]),
        },
        "tracks": result_tracks,
        "timing": {
            "plugin_load_ms": (t_load - t_start) * 1000,
            "playback_ms": 0,
            "analysis_wait_ms": t_process * 1000,
            "total_ms": (t_end - t_start) * 1000,
        },
    }
    result_file.write_text(json.dumps(result, indent=2))

    return True


# Paths
ROOT = Path(__file__).parent.parent
if platform.system() == "Windows":
    HARNESS = ROOT / "build" / "bin" / "Release" / "VST3TestHarness.exe"
else:
    HARNESS = ROOT / "build" / "bin" / "VST3TestHarness"
DEFAULT_TEST = ROOT / "tests" / "integration" / "lfwh_sm57_vs_u87.json"


def run(test_file: Path, generate_plot: bool = True, use_python: bool = False,
        analyze_window: float = 7.5):
    if not test_file.exists():
        print(f"Test definition not found: {test_file}")
        return 1

    # Output dir based on test name (+ _py suffix for Python engine)
    test_name = test_file.stem
    out_dir = ROOT / "results" / (f"{test_name}_py" if use_python else test_name)
    result_file = out_dir / "result.json"

    # Clean previous results to avoid stale/corrupt files
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"  Cleaned previous results: {out_dir}")

    if use_python:
        # ── Python DSP engine ──
        print(f"Running Python DSP engine: {test_name}\n")
        if not run_python_engine(test_file, out_dir, result_file, analyze_window):
            return 1
    else:
        # ── C++ VST3 harness ──
        if not HARNESS.exists():
            print(f"Harness not found: {HARNESS}")
            print("Build with: cmake --build build --config Release --target VST3TestHarness")
            return 1

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
    engine_label = "Python DSP Engine" if use_python else "VST3 Integration Test"

    print()
    print("=" * W)
    print(f"  MAGIC PHASE  -  {engine_label}")
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

    # Generate analysis plots
    if generate_plot:
        print("\nGenerating analysis plots...")
        # Temporarily override sys.argv for the plot script
        old_argv = sys.argv
        sys.argv = ['plot_test_results.py', str(out_dir)]
        try:
            generate_plots()
        except SystemExit:
            pass  # plot script calls exit()
        finally:
            sys.argv = old_argv
        print(f"\n  View: {out_dir / 'plot_overview.png'}")

    # Create normalized audio for easy listening
    print("\nCreating normalized audio for A/B listening...")
    create_normalized_audio(out_dir)

    # Generate Reaper project for A/B listening
    print("\nCreating Reaper project...")
    try:
        rpp_path = create_rpp(out_dir)
        print(f"  Reaper project: {rpp_path}")
    except Exception as e:
        print(f"  Warning: Could not create Reaper project: {e}")

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
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating analysis plots")
    parser.add_argument("--py", action="store_true",
                        help="Use Python DSP engine instead of VST3 harness (faster prototyping)")
    parser.add_argument("--analyze-window", type=float, default=7.5,
                        help="Analysis window in seconds (default: 7.5, matching C++ harness)")
    args = parser.parse_args()
    sys.exit(run(args.test_file, generate_plot=not args.no_plot, use_python=args.py,
                 analyze_window=args.analyze_window))
