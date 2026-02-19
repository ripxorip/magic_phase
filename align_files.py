#!/usr/bin/env python3
"""
Align two audio files and output comparison.

Usage:
    python align_files.py reference.wav target.wav [--output-dir ./output]

Outputs:
    - aligned.wav          : Phase-corrected target
    - sum_before.wav       : reference + target (hear the comb filtering!)
    - sum_after.wav        : reference + aligned (should be fuller/louder)
    - comparison.png       : Visual analysis
"""

import argparse
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path


def detect_delay_xcorr(reference, target, sr, max_delay_ms=50):
    """Detect time delay using cross-correlation."""
    max_delay_samples = int(max_delay_ms * sr / 1000)

    correlation = signal.correlate(target, reference, mode='full')
    lags = signal.correlation_lags(len(target), len(reference), mode='full')

    center = len(correlation) // 2
    search_range = slice(center - max_delay_samples, center + max_delay_samples)

    peak_idx = np.argmax(np.abs(correlation[search_range]))
    delay_samples = lags[search_range][peak_idx]

    # Also check for polarity flip (negative correlation)
    peak_val = correlation[search_range][peak_idx]
    polarity = 1 if peak_val > 0 else -1

    correlation_coef = np.abs(peak_val) / (
        np.sqrt(np.sum(reference**2) * np.sum(target**2))
    )

    return delay_samples, correlation_coef, polarity


def correct_delay_subsample(signal_in, delay_samples, sr):
    """Sub-sample accurate delay correction using FFT."""
    n = len(signal_in)
    spectrum = fft(signal_in)
    freqs = fftfreq(n, 1/sr)
    phase_shift = np.exp(2j * np.pi * freqs * delay_samples / sr)
    corrected_spectrum = spectrum * phase_shift
    return np.real(ifft(corrected_spectrum))


def detect_phase_spectral(reference, target, sr, n_fft=4096):
    """Detect per-frequency phase differences."""
    hop_length = n_fft // 4
    window = signal.windows.hann(n_fft)

    f, t, Zref = signal.stft(reference, sr, window=window, nperseg=n_fft,
                              noverlap=n_fft-hop_length)
    _, _, Ztar = signal.stft(target, sr, window=window, nperseg=n_fft,
                              noverlap=n_fft-hop_length)

    phase_diff = np.angle(Ztar) - np.angle(Zref)
    magnitude_weight = np.abs(Zref) * np.abs(Ztar)
    magnitude_weight = magnitude_weight / (np.sum(magnitude_weight, axis=1, keepdims=True) + 1e-10)
    avg_phase_diff = np.sum(phase_diff * magnitude_weight, axis=1)
    avg_phase_diff = np.unwrap(avg_phase_diff)

    return f, avg_phase_diff


def correct_phase_spectral(reference, target, sr, n_fft=4096, n_bands=64,
                           low_freq=20, high_freq=20000, smoothing=0.5,
                           coherence_threshold=0.4, max_correction_deg=120):
    """
    Frequency-dependent phase correction using STFT.

    This is the GOOD STUFF - corrects phase independently per frequency band!

    Parameters:
        reference: Reference signal (target will be aligned to this)
        target: Signal to be corrected
        sr: Sample rate
        n_fft: FFT size for STFT analysis
        n_bands: Number of frequency bands for smoothed correction
        low_freq: Lowest frequency to correct (Hz)
        high_freq: Highest frequency to correct (Hz)
        smoothing: Smoothing factor (0=none, 1=heavy)
        coherence_threshold: Minimum coherence to apply correction (0-1)
        max_correction_deg: Maximum phase correction in degrees (limits artifacts)

    Returns:
        corrected: Phase-corrected signal
        phase_correction: The phase correction applied per frequency
    """
    hop_length = n_fft // 4
    window = signal.windows.hann(n_fft)

    # STFT of both signals
    f, t_frames, Zref = signal.stft(reference, sr, window=window, nperseg=n_fft,
                                     noverlap=n_fft-hop_length)
    _, _, Ztar = signal.stft(target, sr, window=window, nperseg=n_fft,
                              noverlap=n_fft-hop_length)

    # Calculate phase difference per bin, per frame
    phase_diff = np.angle(Ztar * np.conj(Zref))  # More stable than angle(Ztar) - angle(Zref)

    # Weight by coherence (how consistent is the phase relationship?)
    # High coherence = reliable phase estimate
    magnitude_product = np.abs(Zref) * np.abs(Ztar)

    # Compute weighted circular mean of phase difference per frequency bin
    # Using complex averaging for proper circular statistics
    weighted_complex = magnitude_product * np.exp(1j * phase_diff)
    avg_complex = np.sum(weighted_complex, axis=1)
    avg_phase_diff = np.angle(avg_complex)

    # Compute coherence (magnitude of averaged complex = consistency)
    coherence = np.abs(avg_complex) / (np.sum(magnitude_product, axis=1) + 1e-10)

    # Create smoothed phase correction curve using frequency bands
    # This prevents noisy corrections in areas with low signal
    band_edges = np.logspace(np.log10(max(low_freq, 1)),
                              np.log10(min(high_freq, sr/2)),
                              n_bands + 1)

    smoothed_phase = np.zeros_like(avg_phase_diff)
    smoothed_coherence = np.zeros_like(coherence)

    for i in range(n_bands):
        mask = (f >= band_edges[i]) & (f < band_edges[i+1])
        if np.any(mask):
            band_coherence = coherence[mask]
            band_phase = avg_phase_diff[mask]

            # Weight by coherence within each band
            weights = band_coherence ** 2  # Square to emphasize high-coherence bins
            if np.sum(weights) > 1e-10:
                # Circular weighted mean
                weighted_complex_band = weights * np.exp(1j * band_phase)
                band_avg_phase = np.angle(np.sum(weighted_complex_band))
                band_avg_coherence = np.mean(band_coherence)
            else:
                band_avg_phase = 0
                band_avg_coherence = 0

            smoothed_phase[mask] = band_avg_phase
            smoothed_coherence[mask] = band_avg_coherence

    # Always use smoothed for stability (raw per-bin is too noisy)
    phase_correction = smoothed_phase

    # Apply coherence-weighted correction (don't correct where we're uncertain)
    confidence = np.clip((smoothed_coherence - coherence_threshold) / (1 - coherence_threshold), 0, 1)
    # Smooth the confidence curve to avoid sudden jumps
    from scipy.ndimage import gaussian_filter1d
    confidence = gaussian_filter1d(confidence, sigma=3)

    phase_correction = phase_correction * confidence

    # Limit maximum correction to prevent artifacts
    max_correction_rad = np.deg2rad(max_correction_deg)
    phase_correction = np.clip(phase_correction, -max_correction_rad, max_correction_rad)

    # Smooth the final phase correction curve to prevent discontinuities
    phase_correction = gaussian_filter1d(phase_correction, sigma=2)

    # Apply the correction to each STFT frame
    correction_matrix = np.exp(-1j * phase_correction[:, np.newaxis])
    Zcorrected = Ztar * correction_matrix

    # Inverse STFT
    _, corrected = signal.istft(Zcorrected, sr, window=window, nperseg=n_fft,
                                 noverlap=n_fft-hop_length)

    # Match length to original
    if len(corrected) > len(target):
        corrected = corrected[:len(target)]
    elif len(corrected) < len(target):
        corrected = np.pad(corrected, (0, len(target) - len(corrected)))

    return corrected, (f, phase_correction, smoothed_coherence)


def analyze_and_plot(reference, target, corrected, sr, title="", spectral_info=None):
    """Generate analysis plot."""
    n_plots = 5 if spectral_info else 4
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))

    # Time domain (zoomed to ~50ms)
    t = np.arange(len(reference)) / sr * 1000
    zoom_samples = min(int(sr * 0.05), len(reference))
    zoom = slice(0, zoom_samples)

    axes[0].plot(t[zoom], reference[zoom], label='Reference', alpha=0.8)
    axes[0].plot(t[zoom], target[zoom], label='Target', alpha=0.8)
    axes[0].plot(t[zoom], corrected[zoom], label='Corrected', alpha=0.8, linestyle='--')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'{title} - Time Domain (First 50ms)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cross-correlation
    xcorr_before = signal.correlate(target, reference, mode='same')
    xcorr_after = signal.correlate(corrected, reference, mode='same')
    lag_samples = np.arange(-len(xcorr_before)//2, len(xcorr_before)//2)
    lag_ms = lag_samples / sr * 1000
    zoom_lag = np.abs(lag_ms) < 10

    axes[1].plot(lag_ms[zoom_lag], xcorr_before[zoom_lag], label='Before')
    axes[1].plot(lag_ms[zoom_lag], xcorr_after[zoom_lag], label='After')
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag (ms)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Cross-Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Phase difference
    freqs, phase_before = detect_phase_spectral(reference, target, sr)
    _, phase_after = detect_phase_spectral(reference, corrected, sr)

    axes[2].semilogx(freqs[1:], phase_before[1:], label='Before', alpha=0.8)
    axes[2].semilogx(freqs[1:], phase_after[1:], label='After', alpha=0.8)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Phase Diff (rad)')
    axes[2].set_title('Phase Difference Spectrum')
    axes[2].set_xlim([20, sr/2])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Sum comparison (magnitude spectrum)
    sum_before = reference + target
    sum_after = reference + corrected

    n_fft = 8192
    f_sum = np.fft.rfftfreq(n_fft, 1/sr)
    mag_before = 20 * np.log10(np.abs(np.fft.rfft(sum_before, n_fft)) + 1e-10)
    mag_after = 20 * np.log10(np.abs(np.fft.rfft(sum_after, n_fft)) + 1e-10)
    mag_ref = 20 * np.log10(np.abs(np.fft.rfft(reference, n_fft)) + 1e-10)

    axes[3].semilogx(f_sum[1:], mag_ref[1:], label='Reference only', alpha=0.6)
    axes[3].semilogx(f_sum[1:], mag_before[1:], label='Sum before (comb filtering)', alpha=0.8)
    axes[3].semilogx(f_sum[1:], mag_after[1:], label='Sum after (coherent)', alpha=0.8)
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Magnitude (dB)')
    axes[3].set_title('Summed Signal Spectrum - Look for comb filtering notches!')
    axes[3].set_xlim([20, sr/2])
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # Spectral correction visualization
    if spectral_info is not None:
        freq_bins, phase_corr, coherence = spectral_info

        ax4 = axes[4]
        ax4_twin = ax4.twinx()

        # Phase correction applied
        color1 = 'tab:blue'
        ax4.semilogx(freq_bins[1:], np.degrees(phase_corr[1:]), color=color1,
                     label='Phase correction', linewidth=1.5)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Phase Correction (degrees)', color=color1)
        ax4.tick_params(axis='y', labelcolor=color1)
        ax4.set_xlim([20, sr/2])
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylim([-180, 180])

        # Coherence overlay
        color2 = 'tab:orange'
        ax4_twin.semilogx(freq_bins[1:], coherence[1:], color=color2,
                          label='Coherence', alpha=0.7, linewidth=1.5)
        ax4_twin.set_ylabel('Coherence', color=color2)
        ax4_twin.tick_params(axis='y', labelcolor=color2)
        ax4_twin.set_ylim([0, 1])

        ax4.set_title('Spectral Phase Correction Applied (blue) + Coherence (orange)')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Align two audio files')
    parser.add_argument('reference', help='Reference audio file (the "correct" timing)')
    parser.add_argument('target', help='Target audio file (to be aligned)')
    parser.add_argument('--output-dir', '-o', default='./output', help='Output directory')
    parser.add_argument('--max-delay', '-d', type=float, default=50.0,
                        help='Maximum delay to search for (ms)')
    parser.add_argument('--spectral', '-s', action='store_true',
                        help='Enable frequency-dependent phase correction (the GOOD STUFF)')
    parser.add_argument('--bands', '-b', type=int, default=64,
                        help='Number of frequency bands for spectral correction')
    parser.add_argument('--coherence-threshold', '-c', type=float, default=0.4,
                        help='Minimum coherence to apply correction (0-1, higher=safer)')
    parser.add_argument('--max-correction', '-m', type=float, default=120,
                        help='Maximum phase correction in degrees (lower=safer)')
    parser.add_argument('--no-plot', action='store_true', help='Skip showing plot')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load files
    print(f"Loading {args.reference}...")
    ref, sr_ref = sf.read(args.reference)
    print(f"Loading {args.target}...")
    tar, sr_tar = sf.read(args.target)

    # Handle stereo -> mono
    if ref.ndim > 1:
        print(f"  Reference is {ref.shape[1]}ch, converting to mono")
        ref = np.mean(ref, axis=1)
    if tar.ndim > 1:
        print(f"  Target is {tar.shape[1]}ch, converting to mono")
        tar = np.mean(tar, axis=1)

    # Check sample rates
    if sr_ref != sr_tar:
        print(f"WARNING: Sample rates differ ({sr_ref} vs {sr_tar})")
        print("Resampling target to match reference...")
        from scipy.signal import resample
        tar = resample(tar, int(len(tar) * sr_ref / sr_tar))
    sr = sr_ref

    # Match lengths
    min_len = min(len(ref), len(tar))
    ref = ref[:min_len]
    tar = tar[:min_len]

    print(f"\nAudio: {min_len/sr:.2f}s @ {sr}Hz")

    # Detect delay
    print(f"\nDetecting delay (searching +/- {args.max_delay}ms)...")
    delay_samples, corr, polarity = detect_delay_xcorr(ref, tar, sr, args.max_delay)
    delay_ms = delay_samples / sr * 1000

    print(f"  Detected delay: {delay_samples} samples ({delay_ms:.3f} ms)")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Polarity: {'INVERTED (180°)' if polarity < 0 else 'Normal'}")

    # Apply correction
    print("\nApplying correction...")
    corrected = correct_delay_subsample(tar, delay_samples, sr)
    if polarity < 0:
        print("  Flipping polarity...")
        corrected = -corrected

    # Spectral phase correction (the GOOD STUFF!)
    spectral_info = None
    if args.spectral:
        print(f"\n*** SPECTRAL PHASE CORRECTION ***")
        print(f"  Bands: {args.bands}")
        print(f"  Coherence threshold: {args.coherence_threshold}")
        print(f"  Max correction: {args.max_correction}°")
        print("  Analyzing per-frequency phase differences...")

        corrected, spectral_info = correct_phase_spectral(
            ref, corrected, sr,
            n_bands=args.bands,
            coherence_threshold=args.coherence_threshold,
            max_correction_deg=args.max_correction
        )

        freq_bins, phase_corr, coherence = spectral_info
        print(f"  Phase correction range: {np.min(phase_corr):.2f} to {np.max(phase_corr):.2f} rad")
        print(f"  Average coherence: {np.mean(coherence):.3f}")
        print("  Applied frequency-dependent phase correction!")

    # Compute quality metrics
    corr_before = np.corrcoef(ref, tar)[0, 1]
    corr_after = np.corrcoef(ref, corrected)[0, 1]

    sum_before = ref + tar
    sum_after = ref + corrected

    energy_before = np.sum(sum_before**2)
    energy_after = np.sum(sum_after**2)
    energy_ref = np.sum(ref**2)

    print(f"\nResults:")
    print(f"  Correlation: {corr_before:.4f} -> {corr_after:.4f}")
    print(f"  Sum energy gain: {10*np.log10(energy_after/energy_before):.2f} dB")
    print(f"  Coherence ratio: {energy_after/(4*energy_ref):.3f} (1.0 = perfect)")

    # Normalize outputs - each file normalized independently to prevent clipping
    def normalize(sig, headroom_db=-1.0):
        """Normalize to peak with headroom."""
        peak = np.max(np.abs(sig))
        if peak > 0:
            target = 10 ** (headroom_db / 20)  # -1dB = 0.89
            return sig * (target / peak)
        return sig

    # Save files - normalize each independently
    sf.write(output_dir / 'aligned.wav', normalize(corrected), sr)
    sf.write(output_dir / 'sum_before.wav', normalize(sum_before), sr)
    sf.write(output_dir / 'sum_after.wav', normalize(sum_after), sr)

    # Also output reference and aligned at -6dB for easy mixing/comparison
    sf.write(output_dir / 'ref_mix.wav', normalize(ref, -6), sr)
    sf.write(output_dir / 'aligned_mix.wav', normalize(corrected, -6), sr)
    print(f"\nSaved to {output_dir}/:")
    correction_type = "Time + Spectral corrected" if args.spectral else "Time corrected"
    print(f"  aligned.wav      - {correction_type} target (normalized to -1dB)")
    print(f"  sum_before.wav   - ref + target (LISTEN FOR COMB FILTERING)")
    print(f"  sum_after.wav    - ref + aligned (should sound fuller)")
    print(f"  ref_mix.wav      - reference at -6dB (for mixing)")
    print(f"  aligned_mix.wav  - aligned at -6dB (for mixing)")

    # Plot
    title = "Phase Alignment Analysis" + (" + SPECTRAL" if args.spectral else "")
    fig = analyze_and_plot(ref, tar, corrected, sr, title, spectral_info)
    fig.savefig(output_dir / 'analysis.png', dpi=150)
    print(f"  analysis.png    - Visual analysis")

    if not args.no_plot:
        plt.show()


if __name__ == '__main__':
    main()
