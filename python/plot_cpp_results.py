#!/usr/bin/env python3
"""
Compare C++ Magic Phase output against Python reference output.

Usage:
    python plot_cpp_results.py output_cpp/ output_python/
    python plot_cpp_results.py output_cpp/              # C++ only (no comparison)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import csv
from pathlib import Path
from scipy import signal


def load_wav_mono(path):
    """Load WAV, return mono float array + sample rate."""
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def load_csv(path):
    """Load analysis.csv, return dict of arrays."""
    bins, freqs, corr_rad, corr_deg, coherence = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bins.append(int(row['bin']))
            freqs.append(float(row['freq_hz']))
            corr_rad.append(float(row['phase_correction_rad']))
            corr_deg.append(float(row['phase_correction_deg']))
            coherence.append(float(row['coherence']))
    return {
        'bin': np.array(bins),
        'freq_hz': np.array(freqs),
        'phase_correction_rad': np.array(corr_rad),
        'phase_correction_deg': np.array(corr_deg),
        'coherence': np.array(coherence),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare C++ vs Python Magic Phase results')
    parser.add_argument('cpp_dir', help='C++ output directory')
    parser.add_argument('python_dir', nargs='?', help='Python output directory (optional)')
    parser.add_argument('--save', '-s', help='Save plot to file instead of showing')
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    py_dir = Path(args.python_dir) if args.python_dir else None
    has_python = py_dir is not None

    # Load C++ outputs
    print(f"Loading C++ results from {cpp_dir}/")
    ref_cpp, sr = load_wav_mono(cpp_dir / 'ref_mono.wav')
    tar_cpp, _ = load_wav_mono(cpp_dir / 'unaligned.wav')
    aligned_cpp, _ = load_wav_mono(cpp_dir / 'aligned.wav')
    sum_before_cpp, _ = load_wav_mono(cpp_dir / 'sum_before.wav')
    sum_after_cpp, _ = load_wav_mono(cpp_dir / 'sum_after.wav')
    csv_data = load_csv(cpp_dir / 'analysis.csv')

    # Load Python outputs if available
    if has_python:
        print(f"Loading Python results from {py_dir}/")
        ref_py, _ = load_wav_mono(py_dir / 'ref_mono.wav')
        aligned_py, _ = load_wav_mono(py_dir / 'aligned.wav')
        sum_before_py, _ = load_wav_mono(py_dir / 'sum_before.wav')
        sum_after_py, _ = load_wav_mono(py_dir / 'sum_after.wav')

    # Create figure
    n_plots = 5
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
    fig.suptitle('Magic Phase: C++ vs Python Comparison' if has_python else 'Magic Phase: C++ Results',
                 fontsize=14, fontweight='bold')

    # --- Panel 1: Time domain overlay (first 50ms) ---
    zoom_samples = min(int(sr * 0.05), len(ref_cpp))
    t_ms = np.arange(zoom_samples) / sr * 1000

    axes[0].plot(t_ms, ref_cpp[:zoom_samples], label='Reference', alpha=0.8)
    axes[0].plot(t_ms, tar_cpp[:zoom_samples], label='Target (unaligned)', alpha=0.6)
    axes[0].plot(t_ms, aligned_cpp[:zoom_samples], label='C++ aligned', alpha=0.8, linestyle='--')
    if has_python:
        n = min(zoom_samples, len(aligned_py))
        axes[0].plot(t_ms[:n], aligned_py[:n], label='Python aligned', alpha=0.6, linestyle=':')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain (First 50ms)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Cross-correlation before/after ---
    def xcorr_zoom(sig, ref, sr, max_ms=10):
        xc = signal.correlate(sig, ref, mode='same')
        lags = np.arange(-len(xc)//2, len(xc)//2)
        lag_ms = lags / sr * 1000
        mask = np.abs(lag_ms) < max_ms
        return lag_ms[mask], xc[mask]

    n_common = min(len(ref_cpp), len(tar_cpp), len(aligned_cpp))
    lag_ms, xc_before = xcorr_zoom(tar_cpp[:n_common], ref_cpp[:n_common], sr)
    _, xc_after_cpp = xcorr_zoom(aligned_cpp[:n_common], ref_cpp[:n_common], sr)

    axes[1].plot(lag_ms, xc_before, label='Before', alpha=0.8)
    axes[1].plot(lag_ms, xc_after_cpp, label='After (C++)', alpha=0.8)
    if has_python:
        n_py = min(len(ref_py), len(aligned_py))
        _, xc_after_py = xcorr_zoom(aligned_py[:n_py], ref_py[:n_py], sr)
        axes[1].plot(lag_ms[:len(xc_after_py)], xc_after_py, label='After (Python)',
                     alpha=0.6, linestyle='--')
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag (ms)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Cross-Correlation')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Sum magnitude spectrum (comb filtering check) ---
    n_fft = 8192
    f_sum = np.fft.rfftfreq(n_fft, 1/sr)

    mag_before_cpp = 20 * np.log10(np.abs(np.fft.rfft(sum_before_cpp, n_fft)) + 1e-10)
    mag_after_cpp = 20 * np.log10(np.abs(np.fft.rfft(sum_after_cpp, n_fft)) + 1e-10)
    mag_ref = 20 * np.log10(np.abs(np.fft.rfft(ref_cpp, n_fft)) + 1e-10)

    axes[2].semilogx(f_sum[1:], mag_ref[1:], label='Reference only', alpha=0.5)
    axes[2].semilogx(f_sum[1:], mag_before_cpp[1:], label='Sum before (comb)', alpha=0.7)
    axes[2].semilogx(f_sum[1:], mag_after_cpp[1:], label='Sum after C++ (coherent)', alpha=0.8)
    if has_python:
        mag_after_py = 20 * np.log10(np.abs(np.fft.rfft(sum_after_py, n_fft)) + 1e-10)
        axes[2].semilogx(f_sum[1:], mag_after_py[1:], label='Sum after Python',
                         alpha=0.6, linestyle='--')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude (dB)')
    axes[2].set_title('Summed Signal Spectrum')
    axes[2].set_xlim([20, sr/2])
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # --- Panel 4: Phase correction curve (C++ from CSV) ---
    freqs = csv_data['freq_hz']
    phase_deg = csv_data['phase_correction_deg']

    mask = freqs > 0
    axes[3].semilogx(freqs[mask], phase_deg[mask], label='C++ phase correction', linewidth=1.5)
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Phase Correction (degrees)')
    axes[3].set_title('Phase Correction Curve')
    axes[3].set_xlim([20, sr/2])
    axes[3].set_ylim([-180, 180])
    axes[3].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # --- Panel 5: Energy comparison ---
    # Compare sum_before vs sum_after energy in frequency bands
    n_bands = 48
    band_edges = np.logspace(np.log10(20), np.log10(sr/2), n_bands + 1)
    band_centers = np.sqrt(band_edges[:-1] * band_edges[1:])

    spec_before = np.abs(np.fft.rfft(sum_before_cpp, n_fft))**2
    spec_after_cpp_arr = np.abs(np.fft.rfft(sum_after_cpp, n_fft))**2

    gain_db = []
    for i in range(n_bands):
        mask = (f_sum >= band_edges[i]) & (f_sum < band_edges[i+1])
        if np.any(mask):
            e_before = np.sum(spec_before[mask])
            e_after = np.sum(spec_after_cpp_arr[mask])
            gain_db.append(10 * np.log10((e_after + 1e-20) / (e_before + 1e-20)))
        else:
            gain_db.append(0)

    axes[4].semilogx(band_centers, gain_db, 'o-', markersize=3, label='C++ energy gain')
    axes[4].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[4].set_xlabel('Frequency (Hz)')
    axes[4].set_ylabel('Energy Gain (dB)')
    axes[4].set_title('Per-Band Energy Gain (sum_after / sum_before)')
    axes[4].set_xlim([20, sr/2])
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
