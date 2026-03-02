#!/usr/bin/env python3
"""Run a Magic Phase integration test and print a results table.

Usage:
    python run_test.py path/to/test.json              # Run with VST3 harness
    python run_test.py path/to/test.json --py         # Run with Python DSP engine
    python run_test.py path/to/test.json --py --tree  # Magic align (cluster+star)
    python run_test.py path/to/test.json --no-plot    # Skip plot generation

The --py flag uses the Python DSP engine instead of the C++ VST3 harness.
The --tree flag uses magic align (cluster+star, auto reference).
Auto-detected when no reference track is defined in the test JSON.
Outputs go to results/<test_name>_magic/ with the same structure (WAVs, plots,
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
                             analyze_phase_spectral, apply_phase_spectral,
                             analyze_and_plot)

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
        spectral_info = None

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

            spectral_info = (f_bins, phase_corr, coh)

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

        # ── Analysis plot (align_files.py style with correction curve) ──
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        title = f"Phase Alignment: {track_name}" + (" + SPECTRAL" if phase_on else "")
        fig = analyze_and_plot(ref_audio, tar, corrected[:common_len], sr,
                               title=title, spectral_info=spectral_info)
        plot_path = out_dir / f"analysis_{track_name}.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"    Analysis plot: {plot_path.name}")

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


def _print_matrix(matrix, track_names, indent="  ", fmt=".2f"):
    """Print a correlation matrix as a formatted text table."""
    N = matrix.shape[0]
    header = indent + "         " + "".join(f"{n[:8]:>10}" for n in track_names)
    print(header)
    for i, name in enumerate(track_names):
        row = indent + f"{name[:8]:<8}"
        for j in range(N):
            if i == j:
                row += "       ---"
            else:
                row += f"    {matrix[i, j]:{fmt}} "
        print(row)


def run_python_magic_engine(test_file: Path, out_dir: Path, result_file: Path,
                            analyze_window_s: float = 7.5, threshold: float = 0.15,
                            bridge_threshold: float = 0.30):
    """Run alignment using Python DSP engine with cluster+star algorithm.

    Uses all-pairs broadband + envelope correlation to discover clusters,
    star-aligns within each cluster, and bridges orphan clusters via envelope.
    """
    import time
    from align_files import (detect_delay_xcorr, correct_delay_subsample,
                             analyze_phase_spectral, apply_phase_spectral,
                             analyze_and_plot)
    from magic_align import (magic_align, compute_correlation_matrix,
                             detect_delay_envelope_xcorr,
                             detect_transients, compute_windowed_xcorr_matrix,
                             windowed_xcorr_pair,
                             compute_peak_distance_matrix, plot_peak_match_matrix,
                             plot_detected_peaks, plot_correlation_matrix,
                             plot_triple_xcorr, plot_delay_matrix, plot_triple_delays,
                             plot_peak_detail, plot_cluster_overview,
                             plot_lens_overview)

    test_def = json.loads(test_file.read_text())
    test_name = test_file.stem
    sr_config = test_def.get("sample_rate", 48000)
    buffer_size = test_def.get("buffer_size", 64)

    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # Load ALL tracks (no reference/target distinction initially)
    def load_mono(path):
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr

    track_defs = test_def["tracks"]
    track_names = []
    track_audios = []
    track_paths = []

    for td in track_defs:
        tp = ROOT / td["file"]
        ta, sr = load_mono(tp)
        track_names.append(tp.stem)
        track_audios.append(ta)
        track_paths.append(tp)
        print(f"  Loaded: {tp.name} ({len(ta)/sr:.2f}s @ {sr}Hz)")

    N = len(track_audios)

    # Common length across all tracks
    common_len = min(len(t) for t in track_audios)
    track_audios = [t[:common_len] for t in track_audios]

    # Analysis window
    analyze_samples = min(int(analyze_window_s * sr), common_len)
    analyze_audios = [t[:analyze_samples] for t in track_audios]

    t_load = time.perf_counter()
    print(f"\n  Loaded {N} tracks in {(t_load - t_start)*1000:.0f}ms"
          f" (common length: {common_len/sr:.2f}s)")
    print(f"  Analysis window: {analyze_samples/sr:.1f}s ({analyze_samples} samples)")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: MAGIC ALIGN (cluster + star)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  MAGIC ALIGN (threshold={threshold}, bridge={bridge_threshold})")
    print(f"{'═'*60}")

    t_corr_start = time.perf_counter()
    align_result = magic_align(
        analyze_audios, sr, track_names,
        threshold=threshold, bridge_threshold=bridge_threshold, verbose=True
    )
    t_corr = time.perf_counter() - t_corr_start
    print(f"  Alignment plan computed in {t_corr*1000:.0f}ms")

    # Extract matrices from result
    corr_matrix = align_result.broadband_corr_matrix
    delay_matrix = align_result.broadband_delay_matrix
    pol_matrix = align_result.broadband_pol_matrix
    env_corr_matrix = align_result.envelope_corr_matrix
    env_delay_matrix = align_result.envelope_delay_matrix

    # Print broadband matrix
    print(f"\n  Broadband Correlation Matrix:")
    _print_matrix(corr_matrix, track_names, "    ")

    # Main cluster root
    main_cluster = align_result.clusters[0]
    root = main_cluster.root_idx

    # Print row sums
    print(f"\n  Row sums (correlations >={threshold}):")
    row_sums = []
    for i in range(N):
        s = sum(corr_matrix[i, j] for j in range(N)
                if i != j and corr_matrix[i, j] >= threshold)
        row_sums.append((s, i))
    row_sums.sort(reverse=True)
    for s, i in row_sums:
        a = align_result.alignments[i]
        marker = ""
        if a.tier == 0:
            marker = " <- ROOT"
        elif a.tier == -1:
            marker = " <- ORPHAN"
        elif a.tier == 2:
            marker = " <- TIER 2 (bridged)"
        print(f"    {track_names[i]}: {s:.2f}{marker}")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: APPLY CORRECTIONS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  APPLYING CORRECTIONS")
    print(f"{'═'*60}")

    corrected_audios = [None] * N
    corrected_analyze = [None] * N
    result_tracks = []
    track_gains = {}  # idx -> gain_db vs root

    # Process order: first all cluster-internal alignments (roots + tier 1),
    # then apply cluster shifts for bridged clusters.
    # Within each cluster: root first, then non-root members.

    # Phase A: Process each cluster internally (star alignment within cluster)
    for cluster in align_result.clusters:
        c_root = cluster.root_idx

        # Root passes through (no intra-cluster correction)
        corrected_audios[c_root] = track_audios[c_root].copy()
        corrected_analyze[c_root] = analyze_audios[c_root].copy()

        # Non-root members: Tier 1 alignment to cluster root
        for idx in cluster.track_indices:
            if idx == c_root:
                continue

            a = align_result.alignments[idx]
            name = track_names[idx]
            mode = track_defs[idx].get("mode", "phi")
            root_audio = corrected_audios[c_root]
            root_analyze = corrected_analyze[c_root]

            print(f"\n  {'-'*54}")
            print(f"  {name} -> {track_names[c_root]} (Tier 1, cluster {cluster.cluster_id})")
            print(f"  {'-'*54}")

            # Recompute xcorr vs cluster root (fresh, not from matrix)
            delay, recomputed_corr, _xcorr_pol = detect_delay_xcorr(
                root_analyze, analyze_audios[idx], sr, max_delay_ms=50.0
            )
            delay_ms = delay / sr * 1000

            # Time-align
            corrected = correct_delay_subsample(track_audios[idx], delay, sr)
            corrected_ana = correct_delay_subsample(analyze_audios[idx], delay, sr)

            # Empirical polarity: try both, keep whichever sums better with root
            rms_normal = np.sqrt(np.mean((root_analyze + corrected_ana[:len(root_analyze)]) ** 2))
            rms_inv = np.sqrt(np.mean((root_analyze - corrected_ana[:len(root_analyze)]) ** 2))
            polarity = 1 if rms_normal >= rms_inv else -1

            print(f"    Delay:    {delay:+d} samples ({delay_ms:+.2f} ms)")
            print(f"    Polarity: {'INVERTED' if polarity < 0 else 'Normal'} "
                  f"(empirical: normal={20*np.log10(rms_normal+1e-20):+.1f}dB, "
                  f"inv={20*np.log10(rms_inv+1e-20):+.1f}dB)")

            if polarity < 0:
                corrected = -corrected
                corrected_ana = -corrected_ana

            # Spectral phase correction (Tier 1 only)
            bands_48 = [0.0] * 48
            phase_deg = 0.0
            avg_coh = 0.0
            phase_on = False
            spectral_info = None

            if mode == "phi":
                t_phase = time.perf_counter()
                f_bins, phase_corr, coh = analyze_phase_spectral(
                    root_analyze, corrected_ana, sr,
                    coherence_threshold=0.4, max_correction_deg=120
                )
                corrected = apply_phase_spectral(corrected, phase_corr, sr)
                corrected = corrected[:common_len]
                phase_on = True
                t_phase_done = time.perf_counter()

                mask20 = f_bins > 20
                avg_coh = float(np.mean(coh[mask20]))
                phase_deg = float(np.mean(np.abs(np.degrees(phase_corr[mask20]))))
                spectral_info = (f_bins, phase_corr, coh)

                edges_log = np.logspace(np.log10(20), np.log10(sr/2), 49)
                for b in range(48):
                    m = (f_bins >= edges_log[b]) & (f_bins < edges_log[b+1])
                    if np.any(m):
                        bands_48[b] = float(np.mean(coh[m]))

                print(f"    Phase:    {np.degrees(np.min(phase_corr)):.1f} to "
                      f"{np.degrees(np.max(phase_corr)):.1f} deg")
                print(f"    Coherence: {avg_coh:.3f}")
                print(f"    (spectral: {(t_phase_done-t_phase)*1000:.0f}ms)")

                print(f"\n    Coherence by region:")
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
                        print(f"      {rname:>8}: {bar} {rc:.3f}  corr={rp:.1f} deg")

            corrected = corrected[:common_len]

            # Sanity check: RMS sum comparison
            e_raw = np.sum((root_audio[:common_len] + track_audios[idx][:common_len]) ** 2)
            e_aligned = np.sum((root_audio[:common_len] + corrected[:common_len]) ** 2)
            gain_db = 10 * np.log10(e_aligned / (e_raw + 1e-20))
            print(f"\n    Sum gain vs root: {gain_db:+.1f} dB")
            track_gains[idx] = gain_db

            if gain_db < 0:
                print(f"    WARNING: Correction made it worse! Reverting to raw.")
                corrected = track_audios[idx][:common_len].copy()
                corrected_ana = analyze_audios[idx].copy()
                delay = 0
                delay_ms = 0.0
                polarity = 1
                phase_on = False
                avg_coh = 0.0
                phase_deg = 0.0
                bands_48 = [0.0] * 48
                spectral_info = None

            corrected_audios[idx] = corrected[:common_len]
            corrected_analyze[idx] = corrected_ana[:analyze_samples] if len(corrected_ana) >= analyze_samples else corrected_ana

            # Plot
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            title = f"{name} -> {track_names[c_root]}" + (" + SPECTRAL" if phase_on else "")
            fig = analyze_and_plot(root_audio, track_audios[idx], corrected[:common_len],
                                   sr, title=title, spectral_info=spectral_info)
            plot_path = out_dir / f"analysis_{name}.png"
            fig.savefig(str(plot_path), dpi=150)
            plt.close(fig)
            print(f"    Analysis plot: {plot_path.name}")

    # Phase B: Apply Tier 2 cluster shifts (bridged orphan clusters)
    bridged_edges = {e.child_idx: e for e in align_result.edges if e.tier == 2}
    for cluster in align_result.clusters[1:]:
        bridge_edge = bridged_edges.get(cluster.root_idx)
        if not bridge_edge:
            continue  # True orphan cluster, no shift

        shift_delay = bridge_edge.delay_samples
        shift_pol = bridge_edge.polarity
        target_name = track_names[bridge_edge.root_idx]
        print(f"\n  {'-'*54}")
        print(f"  TIER 2 BRIDGE: Cluster {cluster.cluster_id} -> {target_name}")
        print(f"  Shift: {shift_delay:+.1f} samples ({shift_delay/sr*1000:+.2f} ms)"
              f"{', INV' if shift_pol < 0 else ''}")
        print(f"  {'-'*54}")

        for idx in cluster.track_indices:
            name = track_names[idx]
            # Apply bridge shift on top of intra-cluster correction
            shifted = correct_delay_subsample(corrected_audios[idx], shift_delay, sr)
            if shift_pol < 0:
                shifted = -shifted
            shifted = shifted[:common_len]

            # Sanity check vs main root
            main_root_audio = corrected_audios[root]
            e_before = np.sum((main_root_audio[:common_len] + corrected_audios[idx][:common_len]) ** 2)
            e_after = np.sum((main_root_audio[:common_len] + shifted[:common_len]) ** 2)
            bridge_gain = 10 * np.log10(e_after / (e_before + 1e-20))
            print(f"    {name}: bridge gain vs main root: {bridge_gain:+.1f} dB")
            track_gains[idx] = track_gains.get(idx, 0.0) + bridge_gain

            if bridge_gain < 0:
                print(f"    WARNING: Bridge shift made {name} worse! Keeping pre-bridge audio.")
                track_gains[idx] = track_gains.get(idx, 0.0) - bridge_gain  # undo
            else:
                corrected_audios[idx] = shifted[:common_len]

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD RESULT TRACKS
    # ═══════════════════════════════════════════════════════════════════════
    for idx in range(N):
        a = align_result.alignments[idx]
        name = track_names[idx]
        mode = track_defs[idx].get("mode", "phi")

        if a.tier == 0:
            role = "reference"
            state = "ALIGNED"
        elif a.tier == -1:
            role = "orphan"
            state = "ORPHAN"
        elif a.tier == 2:
            role = "target"
            state = "BRIDGED"
        else:
            role = "target"
            state = "ALIGNED"

        result_tracks.append({
            "name": name,
            "role": role,
            "input_file": track_defs[idx]["file"],
            "output_file": str((out_dir / f"{name}_out.wav").resolve()),
            "slot": idx,
            "tier": a.tier,
            "aligned_to": track_names[a.aligned_to] if a.aligned_to is not None else None,
            "cluster_id": a.cluster_id,
            "results": {
                "alignment_state": state,
                "delay_samples": float(a.delay_samples),
                "delay_sub_sample": float(a.delay_samples),
                "delay_ms": float(a.delay_samples / sr * 1000),
                "correlation": float(corr_matrix[root, idx]) if idx != root else 1.0,
                "coherence": 0.0,
                "phase_degrees": 0.0,
                "polarity_inverted": a.polarity < 0,
                "time_correction_on": a.tier > 0,
                "phase_correction_on": a.tier == 1,
                "is_root": a.tier == 0,
                "is_orphan": a.tier == -1,
                "spectral_bands": [0.0] * 48,
            }
        })

    # ═══════════════════════════════════════════════════════════════════════
    # WRITE OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  WRITING OUTPUTS")
    print(f"{'═'*60}")

    for i in range(N):
        out_path = out_dir / f"{track_names[i]}_out.wav"
        sf.write(str(out_path), corrected_audios[i], sr)

    raw_sum = np.sum(track_audios, axis=0)[:common_len]
    aligned_sum = np.sum(corrected_audios, axis=0)

    sf.write(str(out_dir / "raw_sum.wav"), raw_sum, sr)
    sf.write(str(out_dir / "sum.wav"), aligned_sum, sr)

    total_e_raw = np.sum(raw_sum ** 2)
    total_e_aligned = np.sum(aligned_sum ** 2)
    total_gain = 10 * np.log10(total_e_aligned / (total_e_raw + 1e-20))

    t_end = time.perf_counter()
    print(f"\n  TOTAL sum energy gain: {total_gain:+.1f} dB")
    print(f"  Processing time: {(t_end-t_load)*1000:.0f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n  Generating cluster visualizations...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Result overview (matrix + alignment table with gains)
    fig = plot_cluster_overview(
        align_result, track_names, threshold=threshold, sr=sr,
        total_gain_db=total_gain, track_gains=track_gains,
        output_path=out_dir / "result.png"
    )
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # XCORR DIAGNOSTICS — 3 correlation matrices
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  XCORR DIAGNOSTICS")
    print(f"{'═'*60}")

    broadband_matrix = corr_matrix
    print(f"\n  Broadband XCorr (correlation):")
    _print_matrix(broadband_matrix, track_names, "    ")

    broadband_delay_ms = delay_matrix / sr * 1000
    print(f"\n  Broadband XCorr (delay in ms):")
    _print_matrix(broadband_delay_ms, track_names, "    ", fmt="+.2f")

    print(f"\n  Envelope XCorr (correlation):")
    _print_matrix(env_corr_matrix, track_names, "    ")

    env_delay_ms = env_delay_matrix / sr * 1000
    print(f"\n  Envelope XCorr (delay in ms):")
    _print_matrix(env_delay_ms, track_names, "    ", fmt="+.2f")

    # Windowed XCorr — needs transient detection
    print(f"\n  Detecting transients per track...")
    all_peaks = []
    for i in range(N):
        peaks = detect_transients(analyze_audios[i], sr)
        all_peaks.append(peaks)
        print(f"    {track_names[i]}: {len(peaks)} peaks")

    print(f"\n  Computing windowed correlation matrix...")
    t_win_start = time.perf_counter()
    win_corr_matrix, win_delay_matrix, win_polarity_matrix = compute_windowed_xcorr_matrix(
        analyze_audios, sr, all_peaks
    )
    t_win = time.perf_counter() - t_win_start
    print(f"  Windowed matrix computed in {t_win*1000:.0f}ms")
    print(f"\n  Windowed XCorr (correlation):")
    _print_matrix(win_corr_matrix, track_names, "    ")

    win_delay_ms = win_delay_matrix / sr * 1000
    print(f"\n  Windowed XCorr (delay in ms):")
    _print_matrix(win_delay_ms, track_names, "    ", fmt="+.2f")

    # Peak-Match
    print(f"\n  Computing peak-match matrix (histogram vote)...")
    peak_delay_matrix, peak_conf_matrix = compute_peak_distance_matrix(all_peaks, sr)
    print(f"\n  Peak-Match Delay (ms):")
    _print_matrix(peak_delay_matrix, track_names, "    ", fmt="+.1f")
    print(f"\n  Peak-Match Confidence:")
    _print_matrix(peak_conf_matrix, track_names, "    ", fmt=".0%")

    # Save heatmaps
    print(f"\n  Saving xcorr heatmaps...")

    fig_bb = plot_correlation_matrix(broadband_matrix, track_names,
                                     title="Broadband XCorr", full_scale=True)
    fig_bb.savefig(str(out_dir / "xcorr_broadband.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_bb)
    print(f"    xcorr_broadband.png")

    fig_env = plot_correlation_matrix(env_corr_matrix, track_names,
                                      title="RMS Envelope XCorr", full_scale=True)
    fig_env.savefig(str(out_dir / "xcorr_envelope.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_env)
    print(f"    xcorr_envelope.png")

    fig_win = plot_correlation_matrix(win_corr_matrix, track_names,
                                      title="Windowed XCorr", full_scale=True)
    fig_win.savefig(str(out_dir / "xcorr_windowed.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_win)
    print(f"    xcorr_windowed.png")

    fig_bb_d = plot_delay_matrix(broadband_delay_ms, track_names, title="Broadband Delay (ms)")
    fig_bb_d.savefig(str(out_dir / "xcorr_broadband_delay.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_bb_d)
    print(f"    xcorr_broadband_delay.png")

    fig_env_d = plot_delay_matrix(env_delay_ms, track_names, title="Envelope Delay (ms)")
    fig_env_d.savefig(str(out_dir / "xcorr_envelope_delay.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_env_d)
    print(f"    xcorr_envelope_delay.png")

    fig_win_d = plot_delay_matrix(win_delay_ms, track_names, title="Windowed Delay (ms)")
    fig_win_d.savefig(str(out_dir / "xcorr_windowed_delay.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_win_d)
    print(f"    xcorr_windowed_delay.png")

    fig_triple_d = plot_triple_delays(broadband_delay_ms, env_delay_ms, win_delay_ms,
                                       track_names, output_path=out_dir / "xcorr_triple_delay.png")
    plt.close(fig_triple_d)

    fig_pmatch = plot_peak_match_matrix(peak_delay_matrix, peak_conf_matrix, track_names,
                                         output_path=out_dir / "xcorr_peak_match.png")
    plt.close(fig_pmatch)

    fig_peaks = plot_detected_peaks(analyze_audios, all_peaks, sr, track_names,
                                     output_path=out_dir / "xcorr_peaks.png")
    plt.close(fig_peaks)

    fig_triple = plot_triple_xcorr(broadband_matrix, env_corr_matrix, win_corr_matrix,
                                    track_names, output_path=out_dir / "xcorr_triple.png")
    plt.close(fig_triple)

    # 4x2 lens overview: all delays + all correlations/confidences
    fig_overview = plot_lens_overview(
        broadband_matrix, broadband_delay_ms,
        env_corr_matrix, env_delay_ms,
        win_corr_matrix, win_delay_ms,
        peak_conf_matrix, peak_delay_matrix,
        track_names, output_path=out_dir / "lens_overview.png"
    )
    plt.close(fig_overview)

    # ═══════════════════════════════════════════════════════════════════════
    # PER-PEAK DETAIL for interesting pairs
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n  Per-peak detail plots...")

    # Find tracks of interest: kick, orphans, bridged tracks
    kick_indices = [i for i, n in enumerate(track_names) if 'kick' in n.lower()]
    if not kick_indices:
        orphan_indices = [i for i in range(N) if align_result.alignments[i].tier in (-1, 2)]
        kick_indices = orphan_indices[:1] if orphan_indices else []

    detail_dir = out_dir / "peak_detail"
    detail_dir.mkdir(exist_ok=True)

    for ki in kick_indices:
        ki_name = track_names[ki]
        print(f"\n    {ki_name} vs all others:")
        for j in range(N):
            if j == ki:
                continue
            j_name = track_names[j]
            _, _, _, detail = windowed_xcorr_pair(
                analyze_audios[ki], analyze_audios[j],
                all_peaks[ki], all_peaks[j],
                sr, return_detail=True
            )
            n_peaks = len(detail) if detail else 0
            delays = [d.delay_ms for d in detail] if detail else []
            corrs_d = [d.correlation for d in detail] if detail else []
            mean_d = np.mean(delays) if delays else 0
            med_d = np.median(delays) if delays else 0
            std_d = np.std(delays) if delays else 0
            mean_c = np.mean(corrs_d) if corrs_d else 0
            print(f"      vs {j_name:>14}: {n_peaks:2d} peaks, "
                  f"delay={mean_d:+.2f}ms (median={med_d:+.2f}, std={std_d:.2f}), "
                  f"corr={mean_c:.3f}")

            fig_d = plot_peak_detail(
                analyze_audios[ki], analyze_audios[j],
                detail, sr, ki_name, j_name,
                output_path=detail_dir / f"detail_{ki_name}_vs_{j_name}.png"
            )
            plt.close(fig_d)

    # ═══════════════════════════════════════════════════════════════════════
    # RESULT JSON
    # ═══════════════════════════════════════════════════════════════════════
    cluster_info = []
    for c in align_result.clusters:
        bridge_edge = bridged_edges.get(c.root_idx)
        cluster_info.append({
            "cluster_id": c.cluster_id,
            "root": track_names[c.root_idx],
            "members": [track_names[i] for i in c.track_indices],
            "is_main": c.is_main,
            "bridged_to": track_names[bridge_edge.root_idx] if bridge_edge else None,
            "bridge_delay_samples": bridge_edge.delay_samples if bridge_edge else None,
            "bridge_correlation": bridge_edge.correlation if bridge_edge else None,
        })

    result = {
        "test": test_name,
        "timestamp": datetime.now().astimezone().isoformat(),
        "algorithm": "cluster_star",
        "config": {
            "plugin_path": "Python DSP engine (cluster+star)",
            "sample_rate": float(sr),
            "buffer_size": buffer_size,
            "plugin_loaded": True,
            "num_instances": N,
            "cluster_threshold": threshold,
            "bridge_threshold": bridge_threshold,
        },
        "clusters": cluster_info,
        "correlation_matrix": corr_matrix.tolist(),
        "xcorr_diagnostics": {
            "broadband": {"correlation": corr_matrix.tolist(), "delay_ms": (delay_matrix / sr * 1000).tolist()},
            "envelope": {"correlation": env_corr_matrix.tolist(), "delay_ms": (env_delay_matrix / sr * 1000).tolist()},
            "windowed": {"correlation": win_corr_matrix.tolist(), "delay_ms": (win_delay_matrix / sr * 1000).tolist()},
            "peak_match": {"delay_ms": peak_delay_matrix.tolist(), "confidence": peak_conf_matrix.tolist()},
            "peaks_per_track": {track_names[i]: len(all_peaks[i]) for i in range(N)},
        },
        "tracks": result_tracks,
        "timing": {
            "plugin_load_ms": (t_load - t_start) * 1000,
            "alignment_plan_ms": t_corr * 1000,
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
        analyze_window: float = 7.5, use_tree: bool = False, tree_threshold: float = 0.15):
    if not test_file.exists():
        print(f"Test definition not found: {test_file}")
        return 1

    # Auto-detect magic align mode: --py with no reference track → magic align
    if use_python and not use_tree:
        test_def = json.loads(test_file.read_text())
        has_reference = any(t.get("role") == "reference" for t in test_def.get("tracks", []))
        if not has_reference:
            print(f"  No reference track found -> auto-enabling magic align mode")
            use_tree = True

    # Output dir based on test name (+ _py/_magic suffix)
    test_name = test_file.stem
    if use_tree:
        out_dir = ROOT / "results" / f"{test_name}_magic"
    elif use_python:
        out_dir = ROOT / "results" / f"{test_name}_py"
    else:
        out_dir = ROOT / "results" / test_name
    result_file = out_dir / "result.json"

    # Clean previous results to avoid stale/corrupt files
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"  Cleaned previous results: {out_dir}")

    if use_tree:
        # ── Magic Align (cluster + star) ──
        print(f"Running Magic Align (cluster+star): {test_name}\n")
        if not run_python_magic_engine(test_file, out_dir, result_file, analyze_window, tree_threshold):
            return 1
    elif use_python:
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
    if use_tree:
        engine_label = "Magic Align (Cluster+Star)"
    elif use_python:
        engine_label = "Python DSP Engine"
    else:
        engine_label = "VST3 Integration Test"

    print()
    print("=" * W)
    print(f"  MAGIC PHASE  -  {engine_label}")
    print("=" * W)
    print(f"  Test:    {result['test']}")
    print(f"  Config:  {int(result['config']['sample_rate'])} Hz / {int(result['config']['buffer_size'])} buf")
    print(f"  Tracks:  {len(targets)} target(s)")

    # Cluster info
    if "clusters" in result:
        for ci in result["clusters"]:
            tag = " (MAIN)" if ci["is_main"] else ""
            bridge = f" -> {ci['bridged_to']}" if ci.get("bridged_to") else ""
            print(f"  Cluster {ci['cluster_id']}{tag}: root={ci['root']}, "
                  f"members=[{', '.join(ci['members'])}]{bridge}")

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
    parser.add_argument("--tree", action="store_true",
                        help="Use magic align: cluster+star (auto reference, strongest pairs)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Correlation threshold for tree edges (default: 0.15)")
    parser.add_argument("--analyze-window", type=float, default=7.5,
                        help="Analysis window in seconds (default: 7.5, matching C++ harness)")
    args = parser.parse_args()
    sys.exit(run(args.test_file, generate_plot=not args.no_plot, use_python=args.py,
                 analyze_window=args.analyze_window, use_tree=args.tree,
                 tree_threshold=args.threshold))
