#!/usr/bin/env python3
"""
Phase Alignment Lab
===================
Prototype for phase alignment with spectral corrections.

Techniques implemented:
1. Cross-correlation based time-delay estimation
2. Spectral phase analysis (per-frequency bin phase difference)
3. Phase-linear correction via all-pass filtering
4. Frequency-dependent phase alignment
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# Test Signal Generation
# =============================================================================

def generate_test_signals(sr=48000, duration=2.0, delay_samples=47):
    """
    Generate a pair of signals where one is delayed (out of phase).

    Returns:
        reference: Original signal
        delayed: Phase-shifted version
        delay_samples: Ground truth delay
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Rich harmonic content - simulates a real instrument
    signal_clean = (
        0.5 * np.sin(2 * np.pi * 110 * t) +           # Fundamental
        0.3 * np.sin(2 * np.pi * 220 * t + 0.3) +     # 2nd harmonic
        0.2 * np.sin(2 * np.pi * 330 * t + 0.5) +     # 3rd harmonic
        0.15 * np.sin(2 * np.pi * 440 * t + 0.7) +    # 4th harmonic
        0.1 * np.sin(2 * np.pi * 550 * t + 0.9)       # 5th harmonic
    )

    # Add some transients (like drum hits)
    for hit_time in [0.2, 0.7, 1.2, 1.7]:
        hit_idx = int(hit_time * sr)
        env = np.exp(-30 * np.abs(t - hit_time))
        signal_clean += 0.4 * env * np.sin(2 * np.pi * 1000 * t)

    # Normalize
    signal_clean = signal_clean / np.max(np.abs(signal_clean)) * 0.8

    # Create delayed version (simulates mic placement difference)
    delayed = np.zeros_like(signal_clean)
    delayed[delay_samples:] = signal_clean[:-delay_samples]

    return signal_clean, delayed, delay_samples


def generate_frequency_dependent_phase_shift(sr=48000, duration=2.0):
    """
    Generate signals with frequency-dependent phase shifts.
    This simulates real-world scenarios like all-pass filters or room reflections.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_samples = len(t)

    # Create a broadband signal
    np.random.seed(42)
    noise = np.random.randn(n_samples)

    # Bandpass filter it
    sos = signal.butter(4, [100, 8000], btype='band', fs=sr, output='sos')
    reference = signal.sosfilt(sos, noise)
    reference = reference / np.max(np.abs(reference)) * 0.8

    # Apply frequency-dependent phase shift in spectral domain
    spectrum = fft(reference)
    freqs = fftfreq(n_samples, 1/sr)

    # Phase shift that increases with frequency (simulates dispersive medium)
    phase_shift = np.zeros(n_samples)
    for i, f in enumerate(freqs):
        if abs(f) > 0:
            # Non-linear phase: more shift at higher frequencies
            phase_shift[i] = 0.5 * np.sign(f) * (abs(f) / 1000) ** 0.5

    # Apply phase shift
    shifted_spectrum = spectrum * np.exp(1j * phase_shift)
    shifted = np.real(ifft(shifted_spectrum))

    return reference, shifted, phase_shift


# =============================================================================
# Phase Detection Algorithms
# =============================================================================

def detect_delay_xcorr(reference, target, sr=48000, max_delay_ms=50):
    """
    Detect time delay using cross-correlation.

    This is the classic approach - works well for simple time delays.
    """
    max_delay_samples = int(max_delay_ms * sr / 1000)

    # Normalized cross-correlation
    correlation = signal.correlate(target, reference, mode='full')
    lags = signal.correlation_lags(len(target), len(reference), mode='full')

    # Find peak within reasonable range
    center = len(correlation) // 2
    search_range = slice(center - max_delay_samples, center + max_delay_samples)

    peak_idx = np.argmax(np.abs(correlation[search_range]))
    delay_samples = lags[search_range][peak_idx]

    correlation_coef = correlation[search_range][peak_idx] / (
        np.sqrt(np.sum(reference**2) * np.sum(target**2))
    )

    return delay_samples, correlation_coef


def detect_phase_spectral(reference, target, sr=48000, n_fft=4096):
    """
    Detect per-frequency phase differences using STFT.

    Returns phase difference spectrum - useful for frequency-dependent corrections.
    """
    # STFT parameters
    hop_length = n_fft // 4
    window = signal.windows.hann(n_fft)

    # Compute STFTs
    f, t, Zref = signal.stft(reference, sr, window=window, nperseg=n_fft,
                              noverlap=n_fft-hop_length)
    _, _, Ztar = signal.stft(target, sr, window=window, nperseg=n_fft,
                              noverlap=n_fft-hop_length)

    # Phase difference (averaged across time frames with weighting by magnitude)
    phase_diff = np.angle(Ztar) - np.angle(Zref)

    # Weighted average (weight by magnitude coherence)
    magnitude_weight = np.abs(Zref) * np.abs(Ztar)
    magnitude_weight = magnitude_weight / (np.sum(magnitude_weight, axis=1, keepdims=True) + 1e-10)

    avg_phase_diff = np.sum(phase_diff * magnitude_weight, axis=1)

    # Unwrap for continuous phase
    avg_phase_diff = np.unwrap(avg_phase_diff)

    return f, avg_phase_diff


def estimate_group_delay(frequencies, phase_diff):
    """
    Estimate group delay from phase difference spectrum.
    Group delay = -d(phase)/d(omega)
    """
    # Numerical derivative
    d_phase = np.gradient(phase_diff)
    d_freq = np.gradient(frequencies)

    # Avoid division by zero
    d_freq[d_freq == 0] = 1e-10

    group_delay = -d_phase / (2 * np.pi * d_freq)

    return group_delay


# =============================================================================
# Phase Correction Algorithms
# =============================================================================

def correct_delay_simple(signal_in, delay_samples):
    """
    Simple time-domain delay correction by shifting samples.
    """
    if delay_samples > 0:
        corrected = np.zeros_like(signal_in)
        corrected[:-delay_samples] = signal_in[delay_samples:]
    elif delay_samples < 0:
        corrected = np.zeros_like(signal_in)
        corrected[-delay_samples:] = signal_in[:delay_samples]
    else:
        corrected = signal_in.copy()

    return corrected


def correct_delay_subsample(signal_in, delay_samples_fractional, sr=48000):
    """
    Sub-sample accurate delay correction using sinc interpolation.

    This is crucial for high-quality phase alignment!
    """
    n = len(signal_in)

    # FFT-based fractional delay
    spectrum = fft(signal_in)
    freqs = fftfreq(n, 1/sr)

    # Phase shift for delay
    phase_shift = np.exp(2j * np.pi * freqs * delay_samples_fractional / sr)

    corrected_spectrum = spectrum * phase_shift
    corrected = np.real(ifft(corrected_spectrum))

    return corrected


def correct_phase_spectral(reference, target, sr=48000, n_fft=4096,
                           smooth_bands=True, n_bands=32):
    """
    Frequency-dependent phase correction.

    Analyzes phase differences per frequency band and applies corrections.
    """
    n = len(target)

    # Get spectral phase differences
    freqs, phase_diff = detect_phase_spectral(reference, target, sr, n_fft)

    # Optionally smooth into bands (reduces noise in correction)
    if smooth_bands:
        band_edges = np.logspace(np.log10(20), np.log10(sr/2), n_bands + 1)
        smoothed_phase = np.zeros_like(phase_diff)

        for i in range(n_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            if np.any(mask):
                smoothed_phase[mask] = np.mean(phase_diff[mask])

        phase_diff = smoothed_phase

    # Apply correction in frequency domain (use full signal length!)
    spectrum = fft(target)

    # Interpolate phase correction to full signal FFT size
    full_freqs = fftfreq(n, 1/sr)
    correction_phase = np.interp(
        np.abs(full_freqs),  # Use absolute freq for interpolation
        freqs,
        phase_diff,
        left=0, right=0
    )
    # Handle negative frequencies (conjugate symmetry)
    correction_phase = np.sign(full_freqs) * correction_phase

    # Apply inverse phase shift
    corrected_spectrum = spectrum * np.exp(-1j * correction_phase)

    # Back to time domain
    corrected = np.real(ifft(corrected_spectrum))

    return corrected


def adaptive_phase_alignment(reference, target, sr=48000,
                             block_size=2048, hop_size=512):
    """
    Adaptive phase alignment that tracks time-varying phase differences.

    Useful for live/streaming applications or non-stationary signals.
    """
    n = len(target)
    n_blocks = (n - block_size) // hop_size + 1

    corrected = np.zeros(n)
    window = signal.windows.hann(block_size)
    normalization = np.zeros(n)

    for i in range(n_blocks):
        start = i * hop_size
        end = start + block_size

        if end > n:
            break

        ref_block = reference[start:end] * window
        tar_block = target[start:end] * window

        # Detect local phase difference
        delay, _ = detect_delay_xcorr(ref_block, tar_block, sr, max_delay_ms=10)

        # Apply sub-sample correction
        corrected_block = correct_delay_subsample(tar_block, delay, sr)

        # Overlap-add
        corrected[start:end] += corrected_block
        normalization[start:end] += window

    # Normalize overlap-add
    normalization[normalization < 1e-10] = 1
    corrected = corrected / normalization

    return corrected


# =============================================================================
# Analysis & Visualization
# =============================================================================

def analyze_phase_alignment(reference, target, corrected, sr=48000, title=""):
    """
    Comprehensive visualization of phase alignment results.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Time domain comparison (zoomed)
    t = np.arange(len(reference)) / sr * 1000  # ms
    zoom_range = slice(0, min(2000, len(reference)))  # First ~40ms at 48kHz

    axes[0].plot(t[zoom_range], reference[zoom_range], label='Reference', alpha=0.8)
    axes[0].plot(t[zoom_range], target[zoom_range], label='Target (misaligned)', alpha=0.8)
    axes[0].plot(t[zoom_range], corrected[zoom_range], label='Corrected', alpha=0.8, linestyle='--')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'{title} - Time Domain (Zoomed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cross-correlation before/after
    xcorr_before = signal.correlate(target, reference, mode='same')
    xcorr_after = signal.correlate(corrected, reference, mode='same')
    lag_samples = np.arange(-len(xcorr_before)//2, len(xcorr_before)//2)
    lag_ms = lag_samples / sr * 1000

    zoom_lag = np.abs(lag_ms) < 5  # +/- 5ms

    axes[1].plot(lag_ms[zoom_lag], xcorr_before[zoom_lag], label='Before correction')
    axes[1].plot(lag_ms[zoom_lag], xcorr_after[zoom_lag], label='After correction')
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5, label='Zero lag')
    axes[1].set_xlabel('Lag (ms)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Cross-Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Phase spectrum comparison
    n_fft = 4096
    f_ref, t_ref, Z_ref = signal.stft(reference, sr, nperseg=n_fft)
    f_tar, t_tar, Z_tar = signal.stft(target, sr, nperseg=n_fft)
    f_cor, t_cor, Z_cor = signal.stft(corrected, sr, nperseg=n_fft)

    # Average phase difference across time
    phase_diff_before = np.mean(np.angle(Z_tar) - np.angle(Z_ref), axis=1)
    phase_diff_after = np.mean(np.angle(Z_cor) - np.angle(Z_ref), axis=1)

    axes[2].semilogx(f_ref[1:], np.unwrap(phase_diff_before[1:]), label='Before', alpha=0.8)
    axes[2].semilogx(f_ref[1:], np.unwrap(phase_diff_after[1:]), label='After', alpha=0.8)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Phase Difference (rad)')
    axes[2].set_title('Phase Difference Spectrum')
    axes[2].set_xlim([20, sr/2])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Magnitude spectrum (check for artifacts)
    axes[3].semilogx(f_ref[1:], 20*np.log10(np.mean(np.abs(Z_ref[1:]), axis=1) + 1e-10),
                     label='Reference')
    axes[3].semilogx(f_tar[1:], 20*np.log10(np.mean(np.abs(Z_tar[1:]), axis=1) + 1e-10),
                     label='Target', alpha=0.8)
    axes[3].semilogx(f_cor[1:], 20*np.log10(np.mean(np.abs(Z_cor[1:]), axis=1) + 1e-10),
                     label='Corrected', alpha=0.8, linestyle='--')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Magnitude (dB)')
    axes[3].set_title('Magnitude Spectrum (check for artifacts)')
    axes[3].set_xlim([20, sr/2])
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compute_metrics(reference, target, corrected):
    """
    Compute quality metrics for phase alignment.
    """
    # Correlation coefficient
    corr_before = np.corrcoef(reference, target)[0, 1]
    corr_after = np.corrcoef(reference, corrected)[0, 1]

    # Sum (comb filtering check)
    sum_before = reference + target
    sum_after = reference + corrected

    # Energy ratio (perfect alignment should give 2x energy, 180 phase = 0)
    energy_ref = np.sum(reference**2)
    energy_sum_before = np.sum(sum_before**2)
    energy_sum_after = np.sum(sum_after**2)

    # Ideal sum energy = 4 * single signal energy (coherent sum)
    coherence_before = energy_sum_before / (4 * energy_ref)
    coherence_after = energy_sum_after / (4 * energy_ref)

    return {
        'correlation_before': corr_before,
        'correlation_after': corr_after,
        'coherence_before': coherence_before,
        'coherence_after': coherence_after,
    }


# =============================================================================
# Main Demo
# =============================================================================

def main():
    sr = 48000
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase Alignment Lab - Demo")
    print("=" * 60)

    # Test 1: Simple time delay
    print("\n[Test 1] Simple Time Delay")
    print("-" * 40)

    ref, delayed, true_delay = generate_test_signals(sr=sr, delay_samples=47)

    # Detect delay
    detected_delay, corr = detect_delay_xcorr(ref, delayed, sr)
    print(f"True delay: {true_delay} samples ({true_delay/sr*1000:.3f} ms)")
    print(f"Detected delay: {detected_delay} samples ({detected_delay/sr*1000:.3f} ms)")
    print(f"Correlation: {corr:.4f}")

    # Correct
    corrected = correct_delay_subsample(delayed, detected_delay, sr)

    # Metrics
    metrics = compute_metrics(ref, delayed, corrected)
    print(f"\nCorrelation: {metrics['correlation_before']:.4f} -> {metrics['correlation_after']:.4f}")
    print(f"Coherence: {metrics['coherence_before']:.4f} -> {metrics['coherence_after']:.4f}")

    # Save audio
    sf.write(output_dir / 'test1_reference.wav', ref, sr)
    sf.write(output_dir / 'test1_delayed.wav', delayed, sr)
    sf.write(output_dir / 'test1_corrected.wav', corrected, sr)

    # Visualize
    fig1 = analyze_phase_alignment(ref, delayed, corrected, sr, "Test 1: Simple Delay")
    fig1.savefig(output_dir / 'test1_analysis.png', dpi=150)
    print(f"Saved analysis to {output_dir / 'test1_analysis.png'}")

    # Test 2: Frequency-dependent phase shift
    print("\n[Test 2] Frequency-Dependent Phase Shift")
    print("-" * 40)

    ref2, shifted2, true_phase = generate_frequency_dependent_phase_shift(sr=sr)

    # Spectral phase analysis
    freqs, phase_diff = detect_phase_spectral(ref2, shifted2, sr)
    print(f"Phase difference range: {np.min(phase_diff):.2f} to {np.max(phase_diff):.2f} rad")

    # Spectral correction
    corrected2 = correct_phase_spectral(ref2, shifted2, sr)

    # Metrics
    metrics2 = compute_metrics(ref2, shifted2, corrected2)
    print(f"\nCorrelation: {metrics2['correlation_before']:.4f} -> {metrics2['correlation_after']:.4f}")
    print(f"Coherence: {metrics2['coherence_before']:.4f} -> {metrics2['coherence_after']:.4f}")

    # Save
    sf.write(output_dir / 'test2_reference.wav', ref2, sr)
    sf.write(output_dir / 'test2_shifted.wav', shifted2, sr)
    sf.write(output_dir / 'test2_corrected.wav', corrected2, sr)

    fig2 = analyze_phase_alignment(ref2, shifted2, corrected2, sr, "Test 2: Freq-Dependent Phase")
    fig2.savefig(output_dir / 'test2_analysis.png', dpi=150)
    print(f"Saved analysis to {output_dir / 'test2_analysis.png'}")

    # Test 3: Adaptive alignment
    print("\n[Test 3] Adaptive Phase Alignment")
    print("-" * 40)

    corrected3 = adaptive_phase_alignment(ref, delayed, sr)
    metrics3 = compute_metrics(ref, delayed, corrected3)
    print(f"Correlation: {metrics3['correlation_before']:.4f} -> {metrics3['correlation_after']:.4f}")
    print(f"Coherence: {metrics3['coherence_before']:.4f} -> {metrics3['coherence_after']:.4f}")

    sf.write(output_dir / 'test3_adaptive_corrected.wav', corrected3, sr)

    print("\n" + "=" * 60)
    print("Done! Check the 'output' directory for results.")
    print("=" * 60)

    plt.show()


if __name__ == '__main__':
    main()
