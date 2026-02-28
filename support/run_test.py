#!/usr/bin/env python3
"""Run a Magic Phase integration test and print a results table.

Usage:
    python run_test.py path/to/test.json              # Run with VST3 harness
    python run_test.py path/to/test.json --py         # Run with Python DSP engine
    python run_test.py path/to/test.json --py --tree  # Spanning tree alignment
    python run_test.py path/to/test.json --no-plot    # Skip plot generation

The --py flag uses the Python DSP engine instead of the C++ VST3 harness.
The --tree flag uses spanning tree alignment (auto reference, strongest pairs).
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


def run_python_tree_engine(test_file: Path, out_dir: Path, result_file: Path,
                           analyze_window_s: float = 7.5, threshold: float = 0.15):
    """Run alignment using Python DSP engine with spanning tree algorithm.

    Uses all-pairs correlation matrix to build maximum spanning tree.
    Root (reference) is auto-selected. Each track aligns to its tree parent.
    """
    import time
    from align_files import (detect_delay_xcorr, correct_delay_subsample,
                             analyze_phase_spectral, apply_phase_spectral,
                             analyze_and_plot)
    from graph_align import (compute_correlation_matrix, max_spanning_tree,
                             select_root, build_directed_tree, compute_corrections,
                             plot_alignment_overview)

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
    # STAGE 1: SPANNING TREE ALIGNMENT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  SPANNING TREE ALIGNMENT (threshold={threshold})")
    print(f"{'═'*60}")

    # Compute all-pairs correlation matrix
    print(f"\n  Computing {N}×{N} correlation matrix...")
    t_corr_start = time.perf_counter()
    corr_matrix, delay_matrix, polarity_matrix = compute_correlation_matrix(
        analyze_audios, sr, max_delay_ms=50.0
    )
    t_corr = time.perf_counter() - t_corr_start
    print(f"  Matrix computed in {t_corr*1000:.0f}ms")

    # Print matrix
    print(f"\n  Correlation Matrix:")
    header = "         " + "".join(f"{n[:8]:>10}" for n in track_names)
    print(f"  {header}")
    for i, name in enumerate(track_names):
        row = f"  {name[:8]:<8}"
        for j in range(N):
            if i == j:
                row += "       ---"
            else:
                val = corr_matrix[i, j]
                marker = "*" if val >= threshold else " "
                row += f"    {val:.2f}{marker}"
        print(row)

    # Build spanning tree
    print(f"\n  Building maximum spanning tree...")
    tree_edges = max_spanning_tree(corr_matrix, threshold)
    root = select_root(corr_matrix, threshold)
    directed_tree = build_directed_tree(tree_edges, root, delay_matrix, polarity_matrix)
    corrections = compute_corrections(directed_tree, root, N)

    # Print row sums
    print(f"\n  Row sums (correlations ≥{threshold}):")
    row_sums = []
    for i in range(N):
        s = sum(corr_matrix[i, j] for j in range(N)
                if i != j and corr_matrix[i, j] >= threshold)
        row_sums.append((s, i))
    row_sums.sort(reverse=True)
    for s, i in row_sums:
        marker = " ← ROOT" if i == root else (" ← ORPHAN" if corrections[i].is_orphan else "")
        print(f"    {track_names[i]}: {s:.2f}{marker}")

    # Print tree structure
    print(f"\n  Tree structure:")
    children = {i: [] for i in range(N)}
    for child_idx, edge in directed_tree.items():
        children[edge.parent].append((child_idx, edge))

    def print_tree(idx, prefix="    ", is_last=True):
        corr = corrections[idx]
        if corr.parent_idx is None and not corr.is_orphan:
            print(f"{prefix}{track_names[idx]} (ROOT, untouched)")
        else:
            edge = directed_tree.get(idx)
            if edge:
                delay_ms = edge.delay / sr * 1000
                pol_str = ", INV" if edge.polarity < 0 else ""
                print(f"{prefix}{track_names[idx]} ({edge.correlation:.2f}) "
                      f"→ delay={edge.delay:+d} ({delay_ms:+.2f}ms){pol_str}")

        child_list = children.get(idx, [])
        for i, (child_idx, _) in enumerate(sorted(child_list)):
            is_last_child = (i == len(child_list) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            connector = "└── " if is_last_child else "├── "
            print_tree(child_idx, prefix + connector, is_last_child)

    print_tree(root)

    # Print orphans
    for i in range(N):
        if corrections[i].is_orphan:
            print(f"    {track_names[i]} (ORPHAN, untouched)")

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: APPLY CORRECTIONS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  APPLYING CORRECTIONS")
    print(f"{'═'*60}")

    # Process tracks in BFS order (parents before children)
    corrected_audios = [None] * N
    corrected_analyze = [None] * N
    result_tracks = []

    # Root is untouched
    corrected_audios[root] = track_audios[root].copy()
    corrected_analyze[root] = analyze_audios[root].copy()

    # BFS order
    from collections import deque
    queue = deque([root])
    process_order = []
    visited = {root}

    while queue:
        idx = queue.popleft()
        process_order.append(idx)
        for child_idx in [c for c, _ in children.get(idx, [])]:
            if child_idx not in visited:
                visited.add(child_idx)
                queue.append(child_idx)

    # Add orphans
    for i in range(N):
        if corrections[i].is_orphan:
            process_order.append(i)

    # Process each track
    for idx in process_order:
        corr = corrections[idx]
        name = track_names[idx]
        mode = track_defs[idx].get("mode", "phi")

        print(f"\n  {'-'*54}")
        if idx == root:
            print(f"  {name} (ROOT) - pass through")
            print(f"  {'-'*54}")
            result_tracks.append({
                "name": name,
                "role": "reference",
                "input_file": track_defs[idx]["file"],
                "output_file": str((out_dir / f"{name}_out.wav").resolve()),
                "slot": idx,
                "results": {
                    "alignment_state": "ALIGNED",
                    "delay_samples": 0.0,
                    "delay_sub_sample": 0.0,
                    "delay_ms": 0.0,
                    "correlation": 1.0,
                    "coherence": 1.0,
                    "phase_degrees": 0.0,
                    "polarity_inverted": False,
                    "time_correction_on": False,
                    "phase_correction_on": False,
                    "is_root": True,
                    "spectral_bands": [1.0] * 48,
                }
            })
            continue
        elif corr.is_orphan:
            print(f"  {name} (ORPHAN) - pass through")
            print(f"  {'-'*54}")
            corrected_audios[idx] = track_audios[idx].copy()
            corrected_analyze[idx] = analyze_audios[idx].copy()
            result_tracks.append({
                "name": name,
                "role": "orphan",
                "input_file": track_defs[idx]["file"],
                "output_file": str((out_dir / f"{name}_out.wav").resolve()),
                "slot": idx,
                "results": {
                    "alignment_state": "ORPHAN",
                    "delay_samples": 0.0,
                    "delay_sub_sample": 0.0,
                    "delay_ms": 0.0,
                    "correlation": 0.0,
                    "coherence": 0.0,
                    "phase_degrees": 0.0,
                    "polarity_inverted": False,
                    "time_correction_on": False,
                    "phase_correction_on": False,
                    "is_orphan": True,
                    "spectral_bands": [0.0] * 48,
                }
            })
            continue

        # Get parent's corrected audio
        parent_idx = corr.parent_idx
        parent_audio = corrected_audios[parent_idx]
        parent_analyze = corrected_analyze[parent_idx]
        edge = directed_tree[idx]

        print(f"  {name} → align to {track_names[parent_idx]} (corr={edge.correlation:.2f})")
        print(f"  {'-'*54}")

        # Recompute xcorr against CORRECTED parent (not raw!)
        # The matrix delay was raw-vs-raw, but parent has been shifted
        delay, recomputed_corr, polarity = detect_delay_xcorr(
            parent_analyze, analyze_audios[idx], sr, max_delay_ms=50.0
        )
        delay_ms = delay / sr * 1000

        print(f"    Delay:    {delay:+d} samples ({delay_ms:+.2f} ms)")
        print(f"    Polarity: {'INVERTED' if polarity < 0 else 'Normal'}")
        print(f"    (recomputed vs corrected parent, was {edge.delay:+d} vs raw)")

        corrected = correct_delay_subsample(track_audios[idx], delay, sr)
        corrected_ana = correct_delay_subsample(analyze_audios[idx], delay, sr)
        if polarity < 0:
            corrected = -corrected
            corrected_ana = -corrected_ana

        # Spectral phase correction
        bands_48 = [0.0] * 48
        phase_deg = 0.0
        avg_coh = 0.0
        phase_on = False
        spectral_info = None

        if mode == "phi":
            t_phase = time.perf_counter()

            # Analyze against CORRECTED parent
            f_bins, phase_corr, coh = analyze_phase_spectral(
                parent_analyze, corrected_ana, sr,
                coherence_threshold=0.4, max_correction_deg=120
            )
            corrected = apply_phase_spectral(corrected, phase_corr, sr)
            corrected = corrected[:common_len]

            phase_on = True
            t_phase_done = time.perf_counter()

            # Metrics
            mask20 = f_bins > 20
            avg_coh = float(np.mean(coh[mask20]))
            phase_deg = float(np.mean(np.abs(np.degrees(phase_corr[mask20]))))
            spectral_info = (f_bins, phase_corr, coh)

            # 48-band coherence
            edges = np.logspace(np.log10(20), np.log10(sr/2), 49)
            for b in range(48):
                m = (f_bins >= edges[b]) & (f_bins < edges[b+1])
                if np.any(m):
                    bands_48[b] = float(np.mean(coh[m]))

            print(f"    Phase:    {np.degrees(np.min(phase_corr)):.1f}° to "
                  f"{np.degrees(np.max(phase_corr)):.1f}°")
            print(f"    Coherence: {avg_coh:.3f}")
            print(f"    (spectral: {(t_phase_done-t_phase)*1000:.0f}ms)")

            # Per-region breakdown
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
                    print(f"      {rname:>8}: {bar} {rc:.3f}  corr={rp:.1f}°")

        corrected_audios[idx] = corrected[:common_len]
        corrected_analyze[idx] = corrected_ana[:analyze_samples] if len(corrected_ana) >= analyze_samples else corrected_ana

        # Quality: sum with parent
        e_raw = np.sum((parent_audio + track_audios[idx][:common_len]) ** 2)
        e_aligned = np.sum((parent_audio + corrected[:common_len]) ** 2)
        gain_db = 10 * np.log10(e_aligned / (e_raw + 1e-20))
        print(f"\n    Sum gain vs parent: {gain_db:+.1f} dB")

        # Plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        title = f"{name} → {track_names[parent_idx]}" + (" + SPECTRAL" if phase_on else "")
        fig = analyze_and_plot(parent_audio, track_audios[idx], corrected[:common_len],
                               sr, title=title, spectral_info=spectral_info)
        plot_path = out_dir / f"analysis_{name}.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"    Analysis plot: {plot_path.name}")

        result_tracks.append({
            "name": name,
            "role": "target",
            "input_file": track_defs[idx]["file"],
            "output_file": str((out_dir / f"{name}_out.wav").resolve()),
            "slot": idx,
            "mode": mode,
            "parent": track_names[parent_idx],
            "results": {
                "alignment_state": "ALIGNED",
                "delay_samples": float(delay),
                "delay_sub_sample": float(delay),
                "delay_ms": float(delay_ms),
                "correlation": float(edge.correlation),
                "coherence": float(avg_coh),
                "phase_degrees": float(phase_deg),
                "polarity_inverted": polarity < 0,
                "time_correction_on": True,
                "phase_correction_on": phase_on,
                "spectral_bands": bands_48,
            }
        })

    # ═══════════════════════════════════════════════════════════════════════
    # WRITE OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  WRITING OUTPUTS")
    print(f"{'═'*60}")

    # Write individual tracks
    for i in range(N):
        out_path = out_dir / f"{track_names[i]}_out.wav"
        sf.write(str(out_path), corrected_audios[i], sr)

    # Sum files
    raw_sum = np.sum(track_audios, axis=0)[:common_len]
    aligned_sum = np.sum(corrected_audios, axis=0)

    sf.write(str(out_dir / "raw_sum.wav"), raw_sum, sr)
    sf.write(str(out_dir / "sum.wav"), aligned_sum, sr)

    # Overall energy gain
    total_e_raw = np.sum(raw_sum ** 2)
    total_e_aligned = np.sum(aligned_sum ** 2)
    total_gain = 10 * np.log10(total_e_aligned / (total_e_raw + 1e-20))

    t_end = time.perf_counter()
    print(f"\n  TOTAL sum energy gain: {total_gain:+.1f} dB")
    print(f"  Processing time: {(t_end-t_load)*1000:.0f}ms")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n  Generating tree visualizations...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Overview plot (matrix + tree)
    fig = plot_alignment_overview(
        corr_matrix, directed_tree, corrections, root, track_names,
        threshold=threshold, output_path=out_dir / "tree_overview.png"
    )
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # RESULT JSON
    # ═══════════════════════════════════════════════════════════════════════
    result = {
        "test": test_name,
        "timestamp": datetime.now().astimezone().isoformat(),
        "algorithm": "spanning_tree",
        "config": {
            "plugin_path": "Python DSP engine (spanning tree)",
            "sample_rate": float(sr),
            "buffer_size": buffer_size,
            "plugin_loaded": True,
            "num_instances": N,
            "tree_threshold": threshold,
        },
        "tree": {
            "root": track_names[root],
            "root_idx": root,
            "edges": [
                {
                    "parent": track_names[e.parent],
                    "child": track_names[e.child],
                    "correlation": e.correlation,
                    "delay": e.delay,
                    "polarity": e.polarity
                }
                for e in directed_tree.values()
            ],
            "orphans": [track_names[i] for i in range(N) if corrections[i].is_orphan],
        },
        "correlation_matrix": corr_matrix.tolist(),
        "tracks": result_tracks,
        "timing": {
            "plugin_load_ms": (t_load - t_start) * 1000,
            "correlation_matrix_ms": t_corr * 1000,
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

    # Output dir based on test name (+ _py/_tree suffix)
    test_name = test_file.stem
    if use_tree:
        out_dir = ROOT / "results" / f"{test_name}_tree"
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
        # ── Spanning Tree Python engine ──
        print(f"Running Spanning Tree alignment: {test_name}\n")
        if not run_python_tree_engine(test_file, out_dir, result_file, analyze_window, tree_threshold):
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
        engine_label = "Spanning Tree Alignment"
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

    # Tree-specific info
    if "tree" in result:
        tree_info = result["tree"]
        print(f"  Root:    {tree_info['root']} (auto-selected)")
        if tree_info.get("orphans"):
            print(f"  Orphans: {', '.join(tree_info['orphans'])}")

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
                        help="Use spanning tree alignment (auto reference, strongest pairs)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Correlation threshold for tree edges (default: 0.15)")
    parser.add_argument("--analyze-window", type=float, default=7.5,
                        help="Analysis window in seconds (default: 7.5, matching C++ harness)")
    args = parser.parse_args()
    sys.exit(run(args.test_file, generate_plot=not args.no_plot, use_python=args.py,
                 analyze_window=args.analyze_window, use_tree=args.tree,
                 tree_threshold=args.threshold))
