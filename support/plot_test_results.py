#!/usr/bin/env python3
"""
Plot VST3 integration test results - matches Python prototype analysis style.

Generates visual comparison plots showing before/after phase alignment.

Usage:
    python plot_test_results.py results/lfwh_3way/
    python plot_test_results.py results/lfwh_3way/ --show  # Interactive display
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).parent.parent


def load_wav_mono(path):
    """Load WAV, return mono float array + sample rate."""
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def find_audio_start(data, threshold=0.001):
    """Find the first sample where audio exceeds threshold."""
    nonzero = np.where(np.abs(data) > threshold)[0]
    return nonzero[0] if len(nonzero) > 0 else 0


def detect_phase_spectral(reference, target, sr, n_fft=4096):
    """Detect per-frequency phase differences using STFT."""
    hop_length = n_fft // 4
    window = signal.windows.hann(n_fft)

    f, t, Zref = signal.stft(reference, sr, window=window, nperseg=n_fft,
                              noverlap=n_fft-hop_length)
    _, _, Ztar = signal.stft(target, sr, window=window, nperseg=n_fft,
                              noverlap=n_fft-hop_length)

    # Phase difference per bin, per frame
    phase_diff = np.angle(Ztar * np.conj(Zref))

    # Weight by magnitude product
    magnitude_product = np.abs(Zref) * np.abs(Ztar)

    # Weighted circular mean
    weighted_complex = magnitude_product * np.exp(1j * phase_diff)
    avg_complex = np.sum(weighted_complex, axis=1)
    avg_phase_diff = np.angle(avg_complex)

    return f, avg_phase_diff


def compute_coherence(reference, target, sr, n_fft=4096):
    """Compute magnitude-squared coherence between signals."""
    f, Cxy = signal.coherence(reference, target, sr, nperseg=n_fft)
    return f, Cxy


def find_test_definition(test_name):
    """Find the test definition JSON for a given test name."""
    test_file = ROOT / "tests" / "integration" / f"{test_name}.json"
    if test_file.exists():
        return test_file
    return None


def plot_overview(result_dir, result, test_def, show=False):
    """Generate overview plot: raw_sum vs sum comparison."""

    raw_sum_path = result_dir / "raw_sum.wav"
    sum_path = result_dir / "sum.wav"

    if not raw_sum_path.exists() or not sum_path.exists():
        print(f"  Missing sum files in {result_dir}")
        return None

    raw_sum, sr = load_wav_mono(raw_sum_path)
    aligned_sum, _ = load_wav_mono(sum_path)

    # Find where audio starts (they have different latencies)
    raw_start = max(0, find_audio_start(raw_sum) - int(sr * 0.002))
    aligned_start = max(0, find_audio_start(aligned_sum) - int(sr * 0.002))

    # Create 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    test_name = result.get("test", result_dir.name)
    fig.suptitle(f'Magic Phase VST3 - {test_name}\nSum Before vs After Alignment',
                 fontsize=14, fontweight='bold')

    # Panel 1: Time domain (100ms window, each aligned to its own start)
    zoom_len = int(sr * 0.1)
    raw_end = min(raw_start + zoom_len, len(raw_sum))
    aligned_end = min(aligned_start + zoom_len, len(aligned_sum))
    t_ms = np.arange(zoom_len) / sr * 1000

    axes[0].plot(t_ms[:raw_end-raw_start], raw_sum[raw_start:raw_end],
                 label='Raw Sum (before)', alpha=0.7, color='C0')
    axes[0].plot(t_ms[:aligned_end-aligned_start], aligned_sum[aligned_start:aligned_end],
                 label='Aligned Sum (after)', alpha=0.7, color='C1')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain (100ms window)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Spectrum comparison (THE MONEY SHOT)
    n_fft = 8192
    f = np.fft.rfftfreq(n_fft, 1/sr)

    # Use full audio for spectrum (skip initial silence)
    raw_audio = raw_sum[raw_start:]
    aligned_audio = aligned_sum[aligned_start:]

    mag_before = 20 * np.log10(np.abs(np.fft.rfft(raw_audio, n_fft)) + 1e-10)
    mag_after = 20 * np.log10(np.abs(np.fft.rfft(aligned_audio, n_fft)) + 1e-10)

    axes[1].semilogx(f[1:], mag_before[1:], label='Sum before (comb filtering)', alpha=0.8, color='C0')
    axes[1].semilogx(f[1:], mag_after[1:], label='Sum after (coherent)', alpha=0.8, color='C2')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Summed Signal Spectrum - Look for comb filtering notches!')
    axes[1].set_xlim([20, sr/2])
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Per-band energy gain
    n_bands = 48
    band_edges = np.logspace(np.log10(20), np.log10(sr/2), n_bands + 1)
    band_centers = np.sqrt(band_edges[:-1] * band_edges[1:])

    spec_before = np.abs(np.fft.rfft(raw_audio, n_fft))**2
    spec_after = np.abs(np.fft.rfft(aligned_audio, n_fft))**2

    gain_db = []
    for i in range(n_bands):
        mask = (f >= band_edges[i]) & (f < band_edges[i+1])
        if np.any(mask):
            e_before = np.sum(spec_before[mask])
            e_after = np.sum(spec_after[mask])
            gain_db.append(10 * np.log10((e_after + 1e-20) / (e_before + 1e-20)))
        else:
            gain_db.append(0)

    colors = ['C2' if g > 0 else 'C3' for g in gain_db]
    axes[2].bar(range(len(gain_db)), gain_db, color=colors, alpha=0.7, width=0.8)
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Frequency Band')
    axes[2].set_ylabel('Energy Gain (dB)')
    axes[2].set_title('Per-Band Energy Gain (positive = more energy after alignment)')

    tick_positions = [0, len(gain_db)//4, len(gain_db)//2, 3*len(gain_db)//4, len(gain_db)-1]
    tick_labels = [f'{int(band_centers[i])} Hz' for i in tick_positions]
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels)
    axes[2].grid(True, alpha=0.3, axis='y')

    total_gain = 10 * np.log10(np.sum(aligned_audio**2) / (np.sum(raw_audio**2) + 1e-20))
    axes[2].annotate(f'Total: {total_gain:+.1f} dB', xy=(0.98, 0.95),
                     xycoords='axes fraction', ha='right', va='top',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    out_path = result_dir / "plot_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return out_path


def plot_track_pair(result_dir, result, test_def, track_result, track_idx, show=False):
    """Generate analysis plot for a single target track - matching Python style."""

    # Get paths
    output_file = Path(track_result.get("output_file", ""))
    input_file_rel = track_result.get("input_file", "")
    input_file = ROOT / input_file_rel

    # Find reference track
    ref_result = None
    ref_input_rel = None
    for t in result.get("tracks", []):
        if t.get("role") == "reference":
            ref_result = t
            ref_input_rel = t.get("input_file", "")
            break

    if not ref_result:
        print(f"  No reference track found")
        return None

    ref_input = ROOT / ref_input_rel
    ref_output = Path(ref_result.get("output_file", ""))

    # Check files exist
    if not output_file.exists() or not input_file.exists():
        print(f"  Missing files for track")
        return None

    # Load audio
    target_in, sr = load_wav_mono(input_file)
    target_out, _ = load_wav_mono(output_file)
    ref_in, _ = load_wav_mono(ref_input)
    ref_out, _ = load_wav_mono(ref_output)

    # Get results
    results = track_result.get("results", {})
    track_name = Path(input_file_rel).stem

    # Find audio starts
    in_start = max(0, min(find_audio_start(ref_in), find_audio_start(target_in)) - int(sr * 0.002))
    out_start = max(0, min(find_audio_start(ref_out), find_audio_start(target_out)) - int(sr * 0.002))

    # Align lengths for analysis
    min_in_len = min(len(ref_in), len(target_in))
    min_out_len = min(len(ref_out), len(target_out))

    ref_in_aligned = ref_in[in_start:min_in_len]
    target_in_aligned = target_in[in_start:min_in_len]
    ref_out_aligned = ref_out[out_start:min_out_len]
    target_out_aligned = target_out[out_start:min_out_len]

    # Create 5-panel figure like Python
    fig, axes = plt.subplots(5, 1, figsize=(14, 15))

    delay_ms = results.get("delay_ms", 0)
    delay_samples = results.get("delay_sub_sample", results.get("delay_samples", 0))
    corr = results.get("correlation", 0)
    coh = results.get("coherence", 0)
    polarity = "INVERTED" if results.get("polarity_inverted", False) else "Normal"

    fig.suptitle(f'Magic Phase VST3 - {track_name}\n'
                 f'Delay: {delay_ms:.2f}ms ({delay_samples:.2f} samples) | '
                 f'Correlation: {corr:.3f} | Coherence: {coh:.3f} | Polarity: {polarity}',
                 fontsize=12, fontweight='bold')

    # === Panel 1: Time Domain ===
    zoom_len = int(sr * 0.05)
    t_ms = np.arange(zoom_len) / sr * 1000

    in_end = min(zoom_len, len(ref_in_aligned), len(target_in_aligned))
    out_end = min(zoom_len, len(ref_out_aligned), len(target_out_aligned))

    axes[0].plot(t_ms[:in_end], ref_in_aligned[:in_end], label='Reference', alpha=0.8, color='C0')
    axes[0].plot(t_ms[:in_end], target_in_aligned[:in_end], label='Target (input)', alpha=0.8, color='C1')
    axes[0].plot(t_ms[:out_end], target_out_aligned[:out_end], label='Corrected (output)',
                 alpha=0.8, color='C2', linestyle='--')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'{track_name} - Time Domain (First 50ms)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # === Panel 2: Cross-Correlation ===
    # Compute cross-correlation before and after
    xcorr_before = signal.correlate(target_in_aligned, ref_in_aligned, mode='same')
    xcorr_after = signal.correlate(target_out_aligned, ref_out_aligned, mode='same')

    # Compute lags for each (they may differ in length)
    lag_before = np.arange(-len(xcorr_before)//2, len(xcorr_before)//2)
    lag_after = np.arange(-len(xcorr_after)//2, len(xcorr_after)//2)
    lag_ms_before = lag_before / sr * 1000
    lag_ms_after = lag_after / sr * 1000

    zoom_before = np.abs(lag_ms_before) < 10
    zoom_after = np.abs(lag_ms_after) < 10

    axes[1].plot(lag_ms_before[zoom_before], xcorr_before[zoom_before], label='Before', alpha=0.8, color='C0')
    axes[1].plot(lag_ms_after[zoom_after], xcorr_after[zoom_after], label='After', alpha=0.8, color='C1')
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag (ms)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Cross-Correlation')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # === Panel 3: Phase Difference Spectrum ===
    freqs_before, phase_before = detect_phase_spectral(ref_in_aligned, target_in_aligned, sr)
    freqs_after, phase_after = detect_phase_spectral(ref_out_aligned, target_out_aligned, sr)

    axes[2].semilogx(freqs_before[1:], phase_before[1:], label='Before', alpha=0.8, color='C0')
    axes[2].semilogx(freqs_after[1:], phase_after[1:], label='After', alpha=0.8, color='C1')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Phase Diff (rad)')
    axes[2].set_title('Phase Difference Spectrum')
    axes[2].set_xlim([20, sr/2])
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # === Panel 4: Sum Spectrum (comb filtering check) ===
    sum_before = ref_in_aligned + target_in_aligned
    sum_after = ref_out_aligned + target_out_aligned

    n_fft = 8192
    f_sum = np.fft.rfftfreq(n_fft, 1/sr)
    mag_ref = 20 * np.log10(np.abs(np.fft.rfft(ref_in_aligned, n_fft)) + 1e-10)
    mag_before = 20 * np.log10(np.abs(np.fft.rfft(sum_before, n_fft)) + 1e-10)
    mag_after = 20 * np.log10(np.abs(np.fft.rfft(sum_after, n_fft)) + 1e-10)

    axes[3].semilogx(f_sum[1:], mag_ref[1:], label='Reference only', alpha=0.6, color='C0')
    axes[3].semilogx(f_sum[1:], mag_before[1:], label='Sum before (comb filtering)', alpha=0.8, color='C1')
    axes[3].semilogx(f_sum[1:], mag_after[1:], label='Sum after (coherent)', alpha=0.8, color='C2')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Magnitude (dB)')
    axes[3].set_title('Summed Signal Spectrum - Look for comb filtering notches!')
    axes[3].set_xlim([20, sr/2])
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    # === Panel 5: Coherence Before/After ===
    f_coh_before, coh_before = compute_coherence(ref_in_aligned, target_in_aligned, sr)
    f_coh_after, coh_after = compute_coherence(ref_out_aligned, target_out_aligned, sr)

    ax5 = axes[4]
    ax5.semilogx(f_coh_before[1:], coh_before[1:], label='Coherence Before', alpha=0.7, color='C0')
    ax5.semilogx(f_coh_after[1:], coh_after[1:], label='Coherence After', alpha=0.7, color='C2')
    ax5.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='Good (>0.9)')
    ax5.axhline(0.4, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.4)')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Coherence')
    ax5.set_title('Magnitude-Squared Coherence Before vs After')
    ax5.set_xlim([20, sr/2])
    ax5.set_ylim([0, 1.05])
    ax5.legend(fontsize=8, loc='lower right')
    ax5.grid(True, alpha=0.3)

    # Add average coherence annotation
    avg_coh_before = np.mean(coh_before[f_coh_before > 20])
    avg_coh_after = np.mean(coh_after[f_coh_after > 20])
    ax5.annotate(f'Avg: {avg_coh_before:.3f} → {avg_coh_after:.3f}',
                 xy=(0.02, 0.95), xycoords='axes fraction', ha='left', va='top',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    out_path = result_dir / f"plot_{track_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot VST3 integration test results')
    parser.add_argument('result_dir', type=Path, help='Results directory (e.g., results/lfwh_3way/)')
    parser.add_argument('--show', '-s', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    result_file = result_dir / "result.json"

    if not result_file.exists():
        print(f"No result.json found in {result_dir}")
        return 1

    result = json.loads(result_file.read_text())
    test_name = result.get("test", result_dir.name)

    # Find test definition JSON
    test_def_file = find_test_definition(test_name)
    test_def = None
    if test_def_file:
        test_def = json.loads(test_def_file.read_text())

    print(f"\n{'='*60}")
    print(f"  MAGIC PHASE VST3 - Generating Analysis Plots")
    print(f"{'='*60}")
    print(f"  Test: {test_name}")
    print(f"  Dir:  {result_dir}")
    print(f"{'='*60}\n")

    plots = []

    # Overview plot (sum comparison)
    print("Generating overview plot...")
    p = plot_overview(result_dir, result, test_def, show=args.show)
    if p:
        plots.append(p)

    # Per-track plots
    targets = [t for t in result.get("tracks", []) if t.get("role") == "target" and "results" in t]
    for idx, track in enumerate(targets):
        track_name = Path(track.get("input_file", f"track_{idx}")).stem
        print(f"Generating plot for {track_name}...")
        p = plot_track_pair(result_dir, result, test_def, track, idx, show=args.show)
        if p:
            plots.append(p)

    print(f"\n{'='*60}")
    print(f"  Generated {len(plots)} plot(s)")
    for p in plots:
        print(f"    - {p.name}")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    exit(main())
