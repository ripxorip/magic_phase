#!/usr/bin/env python3
"""
Phase Audit Tool - Analyze multi-mic recordings for phase issues
Independent from Magic Phase C++ implementation
"""

import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from pathlib import Path


def load_audio(path):
    """Load audio file, return (sample_rate, mono_float_data)"""
    sr, data = wavfile.read(path)

    # Convert to float
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        # 24-bit is stored in int32, scaled to 32-bit range
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        pass  # already float
    else:
        raise ValueError(f"Unsupported dtype: {data.dtype}")

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data[:, 0]  # Take first channel

    return sr, data


def find_delay_and_polarity(ref, tar, sr, max_delay_ms=50):
    """Find delay and polarity between two signals using cross-correlation"""
    max_delay_samples = int(max_delay_ms * sr / 1000)

    # Cross-correlation
    xcorr = correlate(tar, ref, mode='full')
    mid = len(xcorr) // 2

    # Search within ±max_delay
    search_start = mid - max_delay_samples
    search_end = mid + max_delay_samples + 1
    search = xcorr[search_start:search_end]

    # Find peak (by absolute value)
    peak_idx = np.argmax(np.abs(search))
    peak_val = search[peak_idx]
    delay_samples = peak_idx - max_delay_samples

    # Polarity from sign of peak
    polarity_inverted = peak_val < 0

    # Correlation coefficient
    ref_energy = np.sum(ref ** 2)
    tar_energy = np.sum(tar ** 2)
    corr_coef = np.abs(peak_val) / np.sqrt(ref_energy * tar_energy + 1e-10)

    return {
        'delay_samples': delay_samples,
        'delay_ms': delay_samples / sr * 1000,
        'polarity_inverted': polarity_inverted,
        'correlation': corr_coef,
        'peak_value': peak_val
    }


def compute_coherence(ref, tar, sr, nperseg=4096):
    """Compute magnitude-squared coherence between signals"""
    from scipy.signal import coherence
    f, coh = coherence(ref, tar, fs=sr, nperseg=nperseg)
    return np.mean(coh), f, coh


def compute_sum_energy(ref, tar, delay, invert):
    """Compute energy of summed signal with given delay and polarity"""
    n = len(ref)

    # Shift target
    if delay >= 0:
        tar_aligned = np.concatenate([tar[delay:], np.zeros(delay)])
    else:
        tar_aligned = np.concatenate([np.zeros(-delay), tar[:n+delay]])

    tar_aligned = tar_aligned[:n]

    if invert:
        tar_aligned = -tar_aligned

    sum_signal = ref + tar_aligned
    return np.sum(sum_signal ** 2)


def analyze_pair(ref_path, tar_path, verbose=True):
    """Analyze a pair of recordings for phase issues"""

    # Load files
    sr_ref, ref = load_audio(ref_path)
    sr_tar, tar = load_audio(tar_path)

    if sr_ref != sr_tar:
        raise ValueError(f"Sample rate mismatch: {sr_ref} vs {sr_tar}")

    sr = sr_ref

    # Trim to same length
    n = min(len(ref), len(tar))
    ref = ref[:n]
    tar = tar[:n]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PHASE AUDIT")
        print(f"{'='*60}")
        print(f"  Reference: {Path(ref_path).name}")
        print(f"  Target:    {Path(tar_path).name}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {n/sr:.2f} seconds")
        print(f"{'='*60}")

    # Find delay and polarity
    result = find_delay_and_polarity(ref, tar, sr)

    if verbose:
        print(f"\n  DELAY & POLARITY")
        print(f"  {'-'*40}")
        print(f"  Delay: {result['delay_samples']} samples ({result['delay_ms']:.2f} ms)")
        print(f"  Polarity: {'INVERTED' if result['polarity_inverted'] else 'NORMAL'}")
        print(f"  Correlation: {result['correlation']:.3f}")

    # Compute coherence
    avg_coh, f, coh = compute_coherence(ref, tar, sr)
    result['coherence_avg'] = avg_coh

    if verbose:
        print(f"\n  COHERENCE")
        print(f"  {'-'*40}")
        print(f"  Average coherence: {avg_coh:.3f}")

        # Coherence interpretation
        if avg_coh > 0.7:
            coh_verdict = "HIGH - signals very similar, alignment valuable"
        elif avg_coh > 0.4:
            coh_verdict = "MEDIUM - some shared content, alignment may help"
        else:
            coh_verdict = "LOW - different content, alignment benefit limited"
        print(f"  Interpretation: {coh_verdict}")

    # Compute sum energy before/after alignment
    energy_raw = compute_sum_energy(ref, tar, 0, False)
    energy_delay_only = compute_sum_energy(ref, tar, result['delay_samples'], False)
    energy_delay_and_flip = compute_sum_energy(ref, tar, result['delay_samples'], result['polarity_inverted'])

    # Also try just flipping without delay
    energy_flip_only = compute_sum_energy(ref, tar, 0, True)

    result['energy_raw'] = energy_raw
    result['energy_delay_only'] = energy_delay_only
    result['energy_delay_and_flip'] = energy_delay_and_flip
    result['energy_flip_only'] = energy_flip_only

    # Find best option
    options = {
        'Raw (no correction)': energy_raw,
        'Delay only': energy_delay_only,
        'Flip only': energy_flip_only,
        'Delay + flip': energy_delay_and_flip
    }
    best_option = max(options, key=options.get)
    best_energy = options[best_option]

    if verbose:
        print(f"\n  SUM ENERGY ANALYSIS")
        print(f"  {'-'*40}")
        for name, energy in sorted(options.items(), key=lambda x: -x[1]):
            gain_db = 10 * np.log10(energy / energy_raw + 1e-10)
            marker = " <-- BEST" if name == best_option else ""
            print(f"  {name:20s}: {gain_db:+.1f} dB{marker}")

    # Verdict
    gain_from_correction = 10 * np.log10(best_energy / energy_raw + 1e-10)
    result['potential_gain_db'] = gain_from_correction
    result['best_option'] = best_option

    if verbose:
        print(f"\n  VERDICT")
        print(f"  {'-'*40}")
        if gain_from_correction < 0.5:
            print(f"  No significant phase issues detected.")
            print(f"  Signals already sum well as-is.")
            result['has_phase_issues'] = False
        else:
            print(f"  PHASE ISSUES DETECTED!")
            print(f"  Recommended: {best_option}")
            print(f"  Potential improvement: +{gain_from_correction:.1f} dB")
            result['has_phase_issues'] = True
        print(f"{'='*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-mic recordings for phase issues')
    parser.add_argument('reference', help='Reference audio file')
    parser.add_argument('target', help='Target audio file to analyze against reference')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode (JSON output only)')

    args = parser.parse_args()

    result = analyze_pair(args.reference, args.target, verbose=not args.quiet)

    if args.quiet:
        import json
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
