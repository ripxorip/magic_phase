#!/usr/bin/env python3
"""Compare VST3 test harness results against golden references."""

import json
import sys
from pathlib import Path


def compare_values(actual, expected):
    """Compare a result value against an expected spec."""
    errors = []

    if "value" in expected:
        tolerance = expected.get("tolerance", 0)
        if abs(actual - expected["value"]) > tolerance:
            errors.append(
                f"  Expected {expected['value']} +/- {tolerance}, got {actual}"
            )

    if "min" in expected and actual < expected["min"]:
        errors.append(f"  Expected >= {expected['min']}, got {actual}")

    if "max" in expected and actual > expected["max"]:
        errors.append(f"  Expected <= {expected['max']}, got {actual}")

    return errors


def compare_result_json(result_path, test_def_path):
    """Compare result JSON against test definition expected values."""
    result = json.loads(Path(result_path).read_text())
    test_def = json.loads(Path(test_def_path).read_text())

    expected = test_def.get("expected", {})
    errors = []

    for track in result["tracks"]:
        if track["role"] == "reference":
            continue

        r = track.get("results", {})

        for key, spec in expected.items():
            if key == "alignment_state":
                if r.get("alignment_state") != spec:
                    errors.append(
                        f"  alignment_state: expected {spec}, "
                        f"got {r.get('alignment_state')}"
                    )
            elif key == "polarity":
                actual_pol = "inverted" if r.get("polarity_inverted") else "normal"
                if actual_pol != spec:
                    errors.append(f"  polarity: expected {spec}, got {actual_pol}")
            elif key in r:
                errors.extend(compare_values(r[key], spec))

    return errors


def analyze_output_audio(ref_wav, target_wav):
    """Independent audio analysis -- don't trust the plugin's numbers."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        print("  SKIP audio analysis (numpy/soundfile not installed)")
        return None

    ref, sr = sf.read(ref_wav)
    tar, sr2 = sf.read(target_wav)
    assert sr == sr2, "Sample rate mismatch"

    # Trim to same length
    min_len = min(len(ref), len(tar))
    ref = ref[:min_len]
    tar = tar[:min_len]

    # Cross-correlation (verify delay)
    correlation = np.correlate(ref, tar, mode="full")
    peak_idx = np.argmax(np.abs(correlation))
    measured_delay = peak_idx - (min_len - 1)

    # RMS comparison
    sum_signal = ref + tar
    ref_rms = np.sqrt(np.mean(ref**2))
    tar_rms = np.sqrt(np.mean(tar**2))
    sum_rms = np.sqrt(np.mean(sum_signal**2))

    sum_gain_db = 0.0
    if max(ref_rms, tar_rms) > 0:
        sum_gain_db = 20 * np.log10(sum_rms / max(ref_rms, tar_rms))

    result = {
        "measured_delay_samples": int(measured_delay),
        "sum_gain_db": float(sum_gain_db),
        "ref_rms": float(ref_rms),
        "tar_rms": float(tar_rms),
        "sum_rms": float(sum_rms),
    }

    # Optional coherence analysis
    try:
        from scipy.signal import coherence as scipy_coherence

        freqs, coh = scipy_coherence(ref, tar, fs=sr, nperseg=4096)
        avg_coherence = np.mean(coh[(freqs > 100) & (freqs < 8000)])
        result["avg_coherence_100_8k"] = float(avg_coherence)
    except ImportError:
        pass

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_results.py <result.json> <test_definition.json>")
        sys.exit(1)

    result_path = sys.argv[1]
    test_def_path = sys.argv[2]

    print(f"Comparing {result_path} against {test_def_path}")
    print("=" * 60)

    # Value comparison
    errors = compare_result_json(result_path, test_def_path)
    if errors:
        print("FAIL -- Value comparison errors:")
        for e in errors:
            print(e)
    else:
        print("PASS -- All values within tolerance")

    # Audio analysis
    result = json.loads(Path(result_path).read_text())
    ref_track = next((t for t in result["tracks"] if t["role"] == "reference"), None)
    tar_tracks = [t for t in result["tracks"] if t["role"] == "target"]

    if ref_track and ref_track.get("output_file"):
        for tar_track in tar_tracks:
            if not tar_track.get("output_file"):
                continue

            ref_file = ref_track["output_file"]
            tar_file = tar_track["output_file"]

            if not Path(ref_file).exists() or not Path(tar_file).exists():
                print(f"\n  SKIP audio analysis (output files not found)")
                continue

            print(f"\nAudio analysis: {tar_track.get('name', 'unknown')}")
            analysis = analyze_output_audio(ref_file, tar_file)

            if analysis is None:
                continue

            for k, v in analysis.items():
                print(f"  {k}: {v}")

            # Sum gain > 3dB means constructive summing (good phase alignment)
            if analysis["sum_gain_db"] > 3.0:
                print("  PASS -- Constructive summing (phase-aligned)")
            else:
                print(f"  WARN -- Sum gain only {analysis['sum_gain_db']:.1f} dB")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
