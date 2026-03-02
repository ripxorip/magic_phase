#!/usr/bin/env python3
"""
Magic Align — Cluster + Star Alignment for Magic Phase.

Uses all-pairs correlation matrices (broadband + envelope) to discover
clusters of related mics, then star-aligns within each cluster.
Orphan clusters are bridged via envelope correlation (Tier 2, time-only).

See docs/MAGIC_ALIGN_ALGORITHM.md for the full spec.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks, butter, sosfilt
from scipy.fft import fft, ifft, fftfreq
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt






@dataclass
class PairResult:
    """Result of pairwise cross-correlation."""
    i: int              # Track index A
    j: int              # Track index B
    correlation: float  # Absolute correlation coefficient
    delay: int          # Delay in samples (positive = j is late)
    polarity: int       # +1 or -1


@dataclass
class ClusterInfo:
    """A cluster of correlated tracks."""
    cluster_id: int
    track_indices: List[int]
    root_idx: int
    is_main: bool              # True for the largest cluster


@dataclass
class AlignmentEdge:
    """An edge in the star alignment graph."""
    child_idx: int
    root_idx: int
    tier: int                  # 1 = full correction, 2 = time-only
    delay_samples: float
    polarity: int
    correlation: float
    source: str                # 'broadband' or 'windowed'


@dataclass
class TrackAlignment:
    """Final alignment decision for a single track."""
    track_idx: int
    cluster_id: int
    tier: int                  # 0=root, 1=full, 2=time-only, -1=true orphan
    delay_samples: float       # Total delay (intra-cluster + cluster shift)
    polarity: int
    aligned_to: Optional[int]
    cluster_shift_samples: float  # Tier 2 cluster shift (0 for main cluster)


@dataclass
class MagicAlignResult:
    """Complete result of the magic_align orchestrator."""
    clusters: List[ClusterInfo]
    alignments: Dict[int, TrackAlignment]
    edges: List[AlignmentEdge]
    main_cluster_id: int
    broadband_corr_matrix: np.ndarray
    broadband_delay_matrix: np.ndarray
    broadband_pol_matrix: np.ndarray
    envelope_corr_matrix: np.ndarray
    envelope_delay_matrix: np.ndarray


def compute_correlation_matrix(
    audios: List[np.ndarray],
    sr: int,
    max_delay_ms: float = 50.0,
    detect_delay_fn=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all-pairs cross-correlation matrix.

    Returns:
        corr_matrix: N×N correlation coefficients (absolute value)
        delay_matrix: N×N delays in samples (delay[i,j] = how much j lags i)
        polarity_matrix: N×N polarity values (+1 or -1)
    """
    from align_files import detect_delay_xcorr

    if detect_delay_fn is None:
        detect_delay_fn = detect_delay_xcorr

    N = len(audios)
    corr_matrix = np.zeros((N, N))
    delay_matrix = np.zeros((N, N))
    polarity_matrix = np.ones((N, N), dtype=int)

    # Compute upper triangle, mirror to lower
    for i in range(N):
        for j in range(i + 1, N):
            delay, corr, pol = detect_delay_fn(audios[i], audios[j], sr, max_delay_ms)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            delay_matrix[i, j] = delay
            delay_matrix[j, i] = -delay  # Opposite direction
            polarity_matrix[i, j] = pol
            polarity_matrix[j, i] = pol

    return corr_matrix, delay_matrix, polarity_matrix


# =============================================================================
# XCORR DIAGNOSTICS — RMS Envelope & Windowed
# =============================================================================

def compute_rms_envelope(audio: np.ndarray, sr: int,
                         hp_freq: float = 900, window_ms: float = 3) -> np.ndarray:
    """Compute RMS envelope of audio after high-pass filtering.

    Args:
        audio: mono audio signal
        sr: sample rate
        hp_freq: high-pass cutoff in Hz (removes low-freq bleed)
        window_ms: RMS window in milliseconds
    Returns:
        RMS envelope (same length as audio)
    """
    # High-pass filter (2nd-order Butterworth)
    sos = butter(2, hp_freq, btype='high', fs=sr, output='sos')
    hp_audio = sosfilt(sos, audio)

    # RMS envelope via sliding window
    window_samples = max(1, int(window_ms * sr / 1000))
    squared = hp_audio ** 2
    kernel = np.ones(window_samples) / window_samples
    rms_sq = np.convolve(squared, kernel, mode='same')
    return np.sqrt(np.maximum(rms_sq, 0))


def detect_delay_envelope_xcorr(ref: np.ndarray, target: np.ndarray, sr: int,
                                max_delay_ms: float = 50) -> Tuple[int, float, int]:
    """Detect delay using RMS envelope cross-correlation.

    Drop-in replacement for detect_delay_xcorr — same signature and return.
    """
    ref_env = compute_rms_envelope(ref, sr)
    tar_env = compute_rms_envelope(target, sr)

    max_delay_samples = int(max_delay_ms * sr / 1000)

    correlation = sp_signal.correlate(tar_env, ref_env, mode='full')
    lags = sp_signal.correlation_lags(len(tar_env), len(ref_env), mode='full')

    center = len(correlation) // 2
    search_range = slice(center - max_delay_samples, center + max_delay_samples)

    peak_idx = np.argmax(np.abs(correlation[search_range]))
    delay_samples = lags[search_range][peak_idx]

    peak_val = correlation[search_range][peak_idx]
    polarity = 1 if peak_val > 0 else -1

    correlation_coef = np.abs(peak_val) / (
        np.sqrt(np.sum(ref_env ** 2) * np.sum(tar_env ** 2)) + 1e-20
    )

    return int(delay_samples), float(correlation_coef), int(polarity)


def detect_transients(audio: np.ndarray, sr: int,
                      rise_ratio: float = 2.5,
                      min_rise: float = 0.04,
                      floor_decay_ms: float = 80,
                      hold_ms: float = 30,
                      ) -> np.ndarray:
    """Detect transient onsets using adaptive-floor Schmitt trigger.

    Tracks a slowly-decaying floor that follows the envelope downward.
    An onset fires when the envelope rises significantly above this floor.
    This handles sustained energy (cymbals) — new hits on top of wash
    still trigger because the floor has settled to the wash level.

    Algorithm:
        1. Compute RMS envelope, normalize to own peak
        2. Adaptive floor tracks envelope downward with exponential decay
           (time constant = floor_decay_ms). Floor never follows UP — that's
           what we're detecting.
        3. Onset when: env > floor * rise_ratio AND (env - floor) > min_rise
        4. After onset: floor resets to current level, hold for hold_ms
        5. During hold: floor follows envelope up (tracks the attack peak)

    Args:
        audio: mono audio
        sr: sample rate
        rise_ratio: envelope must exceed floor * this to trigger (2.5 = 2.5x)
        min_rise: minimum absolute rise above floor (normalized 0-1)
        floor_decay_ms: time constant for floor to follow envelope down
        hold_ms: minimum hold after each onset (ignore new onsets)
    Returns:
        Array of onset sample indices, sorted by time
    """
    env = compute_rms_envelope(audio, sr)

    peak_val = np.max(env)
    if peak_val < 1e-10:
        return np.array([], dtype=int)
    env_norm = env / peak_val

    # Per-sample decay coefficient for floor tracking
    decay_alpha = 1.0 - np.exp(-1.0 / (floor_decay_ms / 1000.0 * sr))
    hold_samples = max(1, int(hold_ms * sr / 1000))
    scan_back_samples = int(15 * sr / 1000)  # 15ms lookback for attack start

    floor = 0.0
    hold_until = 0
    onsets = []

    for i in range(len(env_norm)):
        val = env_norm[i]

        if i < hold_until:
            # In hold period — track envelope UP (follow the attack/peak)
            # so floor settles at the event's sustained level
            floor = max(floor, val)
            continue

        # Outside hold: floor decays toward envelope (follows down, not up)
        if val < floor:
            floor += (val - floor) * decay_alpha
        # If val >= floor, floor stays put — rising above floor = potential onset

        # Check for onset: significant rise above adaptive floor
        rise = val - floor
        is_onset = (rise > min_rise) and (floor < 1e-6 or val > floor * rise_ratio)

        if is_onset:
            # Scan back to find where the rise actually started
            attack_start = i
            scan_threshold = floor * 1.1 + min_rise * 0.2
            for j in range(i - 1, max(0, i - scan_back_samples) - 1, -1):
                if env_norm[j] <= scan_threshold:
                    attack_start = j
                    break

            onsets.append(attack_start)
            floor = val       # reset floor to current level
            hold_until = i + hold_samples

    return np.array(onsets, dtype=int)


def cluster_values(values: List[float], tolerance: float = 0.5
                   ) -> Tuple[List[float], List[int]]:
    """Find the largest cluster of values that are within tolerance of their neighbors.

    Sorts values, walks through grouping consecutive values where the gap
    between neighbors is <= tolerance. Returns the largest group's values
    and their original indices.

    Args:
        values: list of float values to cluster
        tolerance: max gap between consecutive sorted values to be in same cluster

    Returns:
        (cluster_values, cluster_original_indices)
    """
    if not values:
        return [], []

    indexed = sorted(enumerate(values), key=lambda x: x[1])

    # Walk sorted values, split into groups at gaps > tolerance
    groups: List[List[Tuple[int, float]]] = [[indexed[0]]]
    for k in range(1, len(indexed)):
        if indexed[k][1] - indexed[k-1][1] <= tolerance:
            groups[-1].append(indexed[k])
        else:
            groups.append([indexed[k]])

    # Largest group wins
    best = max(groups, key=len)
    cluster_vals = [v for _, v in best]
    cluster_idxs = [i for i, _ in best]

    return cluster_vals, cluster_idxs


@dataclass
class PeakXCorrResult:
    """Per-peak cross-correlation result."""
    peak_sample: int
    peak_time_s: float
    delay_samples: float
    delay_ms: float
    correlation: float
    polarity: int


def windowed_xcorr_pair(audio_a: np.ndarray, audio_b: np.ndarray,
                        peaks_a: np.ndarray, peaks_b: np.ndarray,
                        sr: int,
                        window_ms: float = 25,
                        max_delay_ms: float = 15,
                        return_detail: bool = False
                        ) -> Tuple[float, float, int, Optional[List[PeakXCorrResult]]]:
    """Compute windowed cross-correlation for a pair of tracks.

    Unions all detected peaks from both tracks, windows around each,
    and averages the per-window correlation.

    Returns:
        (mean_delay, mean_corr, dominant_polarity, detail_or_None)
        detail is a list of PeakXCorrResult when return_detail=True
    """
    from align_files import detect_delay_xcorr

    window_samples = int(window_ms * sr / 1000)

    # Union peaks from both tracks, deduplicate nearby (within window_ms/2)
    all_peaks = np.concatenate([peaks_a, peaks_b])
    if len(all_peaks) == 0:
        return 0.0, 0.0, 1, ([] if return_detail else None)

    all_peaks = np.sort(np.unique(all_peaks))

    # Deduplicate: merge peaks within half-window of each other
    min_gap = window_samples // 2
    deduped = [all_peaks[0]]
    for p in all_peaks[1:]:
        if p - deduped[-1] >= min_gap:
            deduped.append(p)
    all_peaks = np.array(deduped)

    audio_len = min(len(audio_a), len(audio_b))
    delays = []
    corrs = []
    pols = []
    detail = [] if return_detail else None

    for peak in all_peaks:
        start = max(0, peak - window_samples)
        end = min(audio_len, peak + window_samples)
        if end - start < window_samples // 2:
            continue  # too short at edges

        win_a = audio_a[start:end]
        win_b = audio_b[start:end]

        # Skip silent windows
        if np.max(np.abs(win_a)) < 1e-8 or np.max(np.abs(win_b)) < 1e-8:
            continue

        delay, corr, pol = detect_delay_xcorr(win_a, win_b, sr, max_delay_ms)
        delays.append(delay)
        corrs.append(corr)
        pols.append(pol)

        if return_detail:
            detail.append(PeakXCorrResult(
                peak_sample=int(peak),
                peak_time_s=float(peak / sr),
                delay_samples=float(delay),
                delay_ms=float(delay / sr * 1000),
                correlation=float(corr),
                polarity=int(pol),
            ))

    if not corrs:
        return 0.0, 0.0, 1, ([] if return_detail else None)

    # Cluster delays: find the dominant group of consistent measurements
    # tolerance = 0.5ms in samples (e.g. 24 samples at 48kHz)
    delays_ms = [d / sr * 1000 for d in delays]
    cluster_vals, cluster_idxs = cluster_values(delays_ms, tolerance=0.5)

    if cluster_vals:
        cluster_delay_ms = float(np.mean(cluster_vals))
        cluster_delay = cluster_delay_ms * sr / 1000
        cluster_corrs = [corrs[i] for i in cluster_idxs]
        cluster_pols = [pols[i] for i in cluster_idxs]
        mean_corr = float(np.mean(cluster_corrs))
        dominant_pol = 1 if sum(cluster_pols) >= 0 else -1
    else:
        cluster_delay = float(np.mean(delays))
        mean_corr = float(np.mean(corrs))
        dominant_pol = 1 if sum(pols) >= 0 else -1

    return cluster_delay, mean_corr, dominant_pol, detail


def compute_windowed_xcorr_matrix(
    audios: List[np.ndarray],
    sr: int,
    all_peaks_per_track: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute N×N windowed cross-correlation matrix.

    For each pair, unions peaks from both tracks and correlates around each.
    """
    N = len(audios)
    corr_matrix = np.zeros((N, N))
    delay_matrix = np.zeros((N, N))
    polarity_matrix = np.ones((N, N), dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            delay, corr, pol, _ = windowed_xcorr_pair(
                audios[i], audios[j],
                all_peaks_per_track[i], all_peaks_per_track[j],
                sr
            )
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            delay_matrix[i, j] = delay
            delay_matrix[j, i] = -delay
            polarity_matrix[i, j] = pol
            polarity_matrix[j, i] = pol

    return corr_matrix, delay_matrix, polarity_matrix


# =============================================================================
# ROBUST POLARITY DETECTION — Transient-windowed GCC-PHAT + dot-product voting
# =============================================================================

@dataclass
class PolarityVote:
    """Single transient window polarity vote."""
    peak_sample: int
    delay_samples: float
    dot_product: float        # normalized dot product after alignment (-1 to +1)
    gcc_sharpness: float      # peak sharpness of GCC-PHAT
    band: str                 # 'broadband' or band label


@dataclass
class PolarityResult:
    """Result of robust polarity detection."""
    polarity: int             # +1 or -1
    confidence: float         # 0.0 to 1.0
    n_votes: int              # number of valid votes
    n_agree: int              # number that agree with majority
    votes: List[PolarityVote]


def _gcc_phat(ref_win: np.ndarray, tar_win: np.ndarray,
              max_lag: int) -> Tuple[float, float]:
    """GCC-PHAT delay estimation for a single window.

    Returns:
        (delay_samples, peak_sharpness)
        peak_sharpness = peak_value / median_value (how sharp the peak is)
    """
    n = len(ref_win)
    nfft = 1 << (2 * n - 1).bit_length()  # next power of 2

    R = fft(ref_win, n=nfft)
    T = fft(tar_win, n=nfft)

    # Cross-power spectrum, whitened
    cross = T * np.conj(R)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-20] = 1e-20
    gcc = np.real(ifft(cross / magnitude))

    # Restrict to allowed lag range
    # gcc[0..max_lag] = positive lags, gcc[-max_lag..] = negative lags
    valid_pos = gcc[:max_lag + 1]
    valid_neg = gcc[-(max_lag):]  # last max_lag samples
    valid = np.concatenate([valid_neg, valid_pos])
    lags = np.arange(-max_lag, max_lag + 1)

    peak_idx = np.argmax(valid)
    delay = float(lags[peak_idx])

    # Peak sharpness: ratio of peak to median
    peak_val = valid[peak_idx]
    median_val = np.median(np.abs(valid)) + 1e-20
    sharpness = float(peak_val / median_val)

    return delay, sharpness


def detect_polarity_robust(
    reference: np.ndarray,
    target: np.ndarray,
    sr: int,
    peaks: np.ndarray,
    delay_hint: float = 0.0,
    max_refine_ms: float = 5.0,
    window_pre_ms: float = 2.0,
    window_post_ms: float = 12.0,
    confidence_threshold: float = 0.2,
    bandpass_range: Tuple[float, float] = (80.0, 3000.0),
    verbose: bool = False,
) -> PolarityResult:
    """Robust polarity detection using transient-windowed GCC-PHAT + dot-product voting.

    Algorithm:
        1. Pre-shift target by delay_hint (known delay from broadband/windowed XCorr)
        2. For each transient onset, extract a short window [-pre, +post] ms
        3. Bandpass filter both windows (80-3000 Hz by default)
        4. GCC-PHAT to refine delay within ±max_refine_ms
        5. Shift target window by refined delay
        6. Normalized dot product → polarity vote
        7. Weight each vote by |dot_product| * gcc_sharpness
        8. Aggregate weighted votes → final polarity + confidence

    Args:
        reference: reference audio signal
        target: target audio signal
        sr: sample rate
        peaks: transient onset indices (in reference track's timeline)
        delay_hint: known delay in samples (target lags reference by this amount)
        max_refine_ms: GCC-PHAT refinement range in ms (small, since delay_hint is close)
        window_pre_ms: window start before onset (ms)
        window_post_ms: window end after onset (ms)
        confidence_threshold: minimum |dot_product| to count a vote
        bandpass_range: (low_hz, high_hz) for bandpass filter
        verbose: print per-vote details

    Returns:
        PolarityResult with polarity, confidence, and vote details
    """
    if len(peaks) == 0:
        return PolarityResult(polarity=1, confidence=0.0, n_votes=0,
                              n_agree=0, votes=[])

    max_refine_lag = int(max_refine_ms * sr / 1000)
    pre_samples = int(window_pre_ms * sr / 1000)
    post_samples = int(window_post_ms * sr / 1000)
    delay_hint_int = int(round(delay_hint))
    audio_len = min(len(reference), len(target))

    # Bandpass filter (applied to full signal once, not per window)
    lo, hi = bandpass_range
    sos = butter(3, [lo, hi], btype='bandpass', fs=sr, output='sos')
    ref_bp = sosfilt(sos, reference)
    tar_bp = sosfilt(sos, target)

    votes: List[PolarityVote] = []

    for peak in peaks:
        # Window in reference timeline
        ref_start = peak - pre_samples
        ref_end = peak + post_samples
        # Corresponding window in target, shifted by delay hint
        tar_start = ref_start + delay_hint_int
        tar_end = ref_end + delay_hint_int

        # Expand target window by refine range for GCC-PHAT search
        tar_start_exp = tar_start - max_refine_lag
        tar_end_exp = tar_end + max_refine_lag

        if ref_start < 0 or ref_end >= audio_len:
            continue
        if tar_start_exp < 0 or tar_end_exp >= audio_len:
            continue

        ref_win = ref_bp[ref_start:ref_end].copy()
        tar_win_exp = tar_bp[tar_start_exp:tar_end_exp].copy()

        # Skip if either window is too quiet
        ref_rms = np.sqrt(np.mean(ref_win ** 2))
        tar_rms = np.sqrt(np.mean(tar_win_exp ** 2))
        if ref_rms < 1e-8 or tar_rms < 1e-8:
            continue

        # GCC-PHAT to refine delay within expanded target window
        refine_delay, sharpness = _gcc_phat(ref_win, tar_win_exp, max_refine_lag)

        # Extract aligned target window using refined delay
        # The expanded window is centered on delay_hint, so refine_delay is
        # the residual offset from center
        center_offset = max_refine_lag  # tar_win_exp[center_offset] = tar_bp[tar_start]
        align_start = center_offset + int(round(refine_delay))
        align_end = align_start + len(ref_win)

        if align_start < 0 or align_end > len(tar_win_exp):
            continue

        aligned_tar = tar_win_exp[align_start:align_end]
        aligned_ref = ref_win

        if len(aligned_ref) < pre_samples or len(aligned_tar) < pre_samples:
            continue

        # Normalized dot product
        norm_ref = np.sqrt(np.sum(aligned_ref ** 2))
        norm_tar = np.sqrt(np.sum(aligned_tar ** 2))
        if norm_ref < 1e-10 or norm_tar < 1e-10:
            continue

        dot = float(np.sum(aligned_ref * aligned_tar) / (norm_ref * norm_tar))

        total_delay = delay_hint + refine_delay
        votes.append(PolarityVote(
            peak_sample=int(peak),
            delay_samples=total_delay,
            dot_product=dot,
            gcc_sharpness=sharpness,
            band='broadband',
        ))

    # Filter votes by confidence threshold
    valid_votes = [v for v in votes if abs(v.dot_product) >= confidence_threshold]

    if not valid_votes:
        # Fall back to all votes if none pass threshold
        if votes:
            valid_votes = votes
        else:
            return PolarityResult(polarity=1, confidence=0.0, n_votes=0,
                                  n_agree=0, votes=[])

    # Weighted vote: weight = |dot_product| * gcc_sharpness
    weighted_sum = 0.0
    total_weight = 0.0
    for v in valid_votes:
        weight = abs(v.dot_product) * max(v.gcc_sharpness, 0.1)
        weighted_sum += np.sign(v.dot_product) * weight
        total_weight += weight

    final_polarity = 1 if weighted_sum >= 0 else -1

    # Confidence: fraction of weight that agrees with majority
    agree_weight = 0.0
    n_agree = 0
    for v in valid_votes:
        if np.sign(v.dot_product) == final_polarity:
            agree_weight += abs(v.dot_product) * max(v.gcc_sharpness, 0.1)
            n_agree += 1
    confidence = float(agree_weight / (total_weight + 1e-20))

    if verbose:
        print(f"    Polarity votes: {len(valid_votes)} valid / {len(votes)} total, "
              f"pol={'INV' if final_polarity < 0 else 'NORMAL'}, "
              f"confidence={confidence:.2f} ({n_agree}/{len(valid_votes)} agree)")

    return PolarityResult(
        polarity=final_polarity,
        confidence=confidence,
        n_votes=len(valid_votes),
        n_agree=n_agree,
        votes=votes,
    )


def compute_polarity_matrix(
    audios: List[np.ndarray],
    sr: int,
    all_peaks: List[np.ndarray],
    delay_matrix: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute N×N robust polarity matrix using transient-windowed GCC-PHAT.

    Uses delay_matrix values as delay hints so GCC-PHAT only needs to
    refine within a small range (works for close mics AND room mics).

    Returns:
        polarity_matrix: N×N polarity values (+1 or -1)
        confidence_matrix: N×N confidence values (0.0 to 1.0)
    """
    N = len(audios)
    pol_matrix = np.ones((N, N), dtype=int)
    conf_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            # Use peaks from the reference track (i)
            # delay_matrix[i,j] = how much j lags i (in samples)
            result = detect_polarity_robust(
                audios[i], audios[j], sr, all_peaks[i],
                delay_hint=float(delay_matrix[i, j]),
                verbose=verbose,
            )
            pol_matrix[i, j] = result.polarity
            pol_matrix[j, i] = result.polarity
            conf_matrix[i, j] = result.confidence
            conf_matrix[j, i] = result.confidence

    return pol_matrix, conf_matrix


@dataclass
class PeakMatchResult:
    """Result of peak-matching between two tracks."""
    delay_ms: float         # consensus delay (mean of largest cluster)
    confidence: float       # fraction of peaks that agree (0-1)
    n_matched: int          # number of peaks in the winning bin
    n_total: int            # total peak pairs considered
    all_distances_ms: List[float]  # all signed nearest-peak distances


def peak_match_pair(peaks_a: np.ndarray, peaks_b: np.ndarray,
                    sr: int, max_dist_ms: float = 20.0) -> PeakMatchResult:
    """Match peaks between two tracks via cluster voting.

    For each peak in A, find the closest peak in B (signed distance).
    Ignore pairs further than max_dist_ms (not a real match).
    Cluster the distances (0.5ms tolerance), largest cluster = consensus delay.
    Confidence = cluster size / total A peaks.

    Direction matters: A is the track being aligned, B is the reference.
    "How many of A's onsets can we find in B at a consistent delay?"

    Args:
        peaks_a, peaks_b: onset sample indices
        sr: sample rate
        max_dist_ms: ignore nearest-peak pairs further than this
    """
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return PeakMatchResult(0.0, 0.0, 0, 0, [])

    # For each peak in A, find closest peak in B (signed)
    # Only keep if within max_dist_ms
    dists_ms = []
    for pa in peaks_a:
        diffs = (peaks_b - pa) / sr * 1000  # signed, in ms
        closest_idx = np.argmin(np.abs(diffs))
        d = float(diffs[closest_idx])
        if abs(d) <= max_dist_ms:
            dists_ms.append(d)

    if not dists_ms:
        return PeakMatchResult(0.0, 0.0, 0, 0, [])

    # Cluster the distances: group values within 0.5ms of each other
    cluster_vals, cluster_idxs = cluster_values(dists_ms, tolerance=0.5)
    delay = float(np.mean(cluster_vals)) if cluster_vals else 0.0
    best_count = len(cluster_vals)

    # Confidence = cluster size / total A-peaks
    # "What fraction of my onsets found a consistent match?"
    confidence = best_count / len(peaks_a) if len(peaks_a) > 0 else 0.0

    return PeakMatchResult(
        delay_ms=delay,
        confidence=float(confidence),
        n_matched=int(best_count),
        n_total=len(peaks_a),
        all_distances_ms=dists_ms,
    )


def compute_peak_distance_matrix(
    all_peaks_per_track: List[np.ndarray],
    sr: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute N*N peak-match matrices via cluster voting.

    Returns:
        delay_matrix: consensus delay in ms (signed, [i,j] = how much j lags i)
        confidence_matrix: fraction of peaks agreeing on the consensus delay
        detail: dict of (i,j) -> PeakMatchResult for inspection
    """
    N = len(all_peaks_per_track)
    delay_matrix = np.zeros((N, N))
    confidence_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            # Forward: i→j (how many of i's peaks match j?)
            fwd = peak_match_pair(
                all_peaks_per_track[i], all_peaks_per_track[j], sr
            )
            # Reverse: j→i (how many of j's peaks match i?)
            rev = peak_match_pair(
                all_peaks_per_track[j], all_peaks_per_track[i], sr
            )
            delay_matrix[i, j] = fwd.delay_ms
            delay_matrix[j, i] = rev.delay_ms
            confidence_matrix[i, j] = fwd.confidence
            confidence_matrix[j, i] = rev.confidence

    return delay_matrix, confidence_matrix


def plot_peak_match_matrix(
    delay_matrix: np.ndarray,
    confidence_matrix: np.ndarray,
    track_names: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot peak-match results: delay heatmap + confidence heatmap side by side."""
    N = delay_matrix.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: delay
    disp_delay = np.ma.masked_where(np.eye(N, dtype=bool), delay_matrix)
    vmax_d = np.max(np.abs(delay_matrix[~np.eye(N, dtype=bool)]))
    vmax_d = max(vmax_d, 0.5)

    cmap_d = plt.cm.RdBu_r.copy()
    cmap_d.set_bad(color='#e0e0e0')
    im1 = ax1.imshow(disp_delay, cmap=cmap_d, vmin=-vmax_d, vmax=vmax_d, aspect='equal')
    fig.colorbar(im1, ax=ax1, shrink=0.8).set_label('Delay (ms)', rotation=270, labelpad=15)

    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    ax1.set_xticklabels(track_names, rotation=45, ha='right')
    ax1.set_yticklabels(track_names)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            val = delay_matrix[i, j]
            color = 'white' if abs(val) > vmax_d * 0.6 else 'black'
            ax1.text(j, i, f'{val:+.1f}', ha='center', va='center',
                     fontsize=8, color=color, fontweight='bold')
    ax1.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax1.grid(which='minor', color='white', linewidth=2)
    ax1.set_title('Peak-Match Delay (ms)', fontsize=12, fontweight='bold')

    # Right: confidence
    disp_conf = np.ma.masked_where(np.eye(N, dtype=bool), confidence_matrix)
    cmap_c = plt.cm.YlGn.copy()
    cmap_c.set_bad(color='#e0e0e0')
    im2 = ax2.imshow(disp_conf, cmap=cmap_c, vmin=0, vmax=1.0, aspect='equal')
    fig.colorbar(im2, ax=ax2, shrink=0.8).set_label('Confidence', rotation=270, labelpad=15)

    ax2.set_xticks(range(N))
    ax2.set_yticks(range(N))
    ax2.set_xticklabels(track_names, rotation=45, ha='right')
    ax2.set_yticklabels(track_names)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            val = confidence_matrix[i, j]
            n_match = 0  # we don't have n_matched here, just show %
            color = 'white' if val > 0.6 else 'black'
            ax2.text(j, i, f'{val:.0%}', ha='center', va='center',
                     fontsize=9, color=color, fontweight='bold')
    ax2.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax2.grid(which='minor', color='white', linewidth=2)
    ax2.set_title('Peak-Match Confidence', fontsize=12, fontweight='bold')

    fig.suptitle('Peak Onset Matching (Histogram Vote)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def _compute_adaptive_floor(env_norm: np.ndarray, sr: int,
                            rise_ratio: float = 2.5,
                            min_rise: float = 0.04,
                            floor_decay_ms: float = 80,
                            hold_ms: float = 30) -> np.ndarray:
    """Recompute the adaptive floor curve for visualization.

    Mirrors detect_transients logic to produce floor[i] at every sample.
    """
    decay_alpha = 1.0 - np.exp(-1.0 / (floor_decay_ms / 1000.0 * sr))
    hold_samples = max(1, int(hold_ms * sr / 1000))

    floor_curve = np.zeros_like(env_norm)
    floor = 0.0
    hold_until = 0

    for i in range(len(env_norm)):
        val = env_norm[i]

        if i < hold_until:
            floor = max(floor, val)
        else:
            if val < floor:
                floor += (val - floor) * decay_alpha

            rise = val - floor
            is_onset = (rise > min_rise) and (floor < 1e-6 or val > floor * rise_ratio)
            if is_onset:
                floor = val
                hold_until = i + hold_samples

        floor_curve[i] = floor

    return floor_curve


def plot_detected_peaks(
    audios: List[np.ndarray],
    all_peaks_per_track: List[np.ndarray],
    sr: int,
    track_names: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Debug plot: each track's envelope + adaptive floor + detected onsets.

    One row per track. Shows how the adaptive floor tracks the envelope
    and where onsets fire.
    """
    N = len(audios)
    fig, axes = plt.subplots(N, 1, figsize=(14, 2.5 * N), sharex=True)
    if N == 1:
        axes = [axes]

    for i, (audio, peaks, name) in enumerate(zip(audios, all_peaks_per_track, track_names)):
        ax = axes[i]
        t = np.arange(len(audio)) / sr

        # Waveform (light)
        ax.plot(t, audio, color='#BBDEFB', linewidth=0.3, alpha=0.7)

        # Envelope + adaptive floor
        env = compute_rms_envelope(audio, sr)
        env_peak = np.max(env)
        if env_peak > 1e-10:
            env_norm = env / env_peak
            display_scale = np.max(np.abs(audio)) * 0.9

            ax.plot(t, env_norm * display_scale, color='#1565C0', linewidth=0.8,
                    label='RMS envelope')

            # Adaptive floor curve
            floor_curve = _compute_adaptive_floor(env_norm, sr)
            ax.plot(t, floor_curve * display_scale, color='#FF9800', linewidth=0.7,
                    linestyle='-', alpha=0.8, label='adaptive floor')

            # Trigger level (floor * rise_ratio) — where onset would fire
            trigger_curve = np.minimum(floor_curve * 2.5, 1.0)
            ax.plot(t, trigger_curve * display_scale, color='#4CAF50', linewidth=0.5,
                    linestyle=':', alpha=0.5, label='trigger level (2.5x floor)')

        # Onsets
        if len(peaks) > 0:
            valid = peaks[peaks < len(env)]
            ax.plot(t[valid], env[valid], 'rv', markersize=6, label=f'{len(peaks)} onsets')
            for p in valid:
                ax.axvline(t[p], color='#F44336', linewidth=0.5, alpha=0.4)

        ax.set_ylabel(name, fontsize=9, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_xlim(0, len(audio) / sr)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Detected Onsets (Adaptive Floor, 900Hz HP RMS Envelope)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_peak_detail(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    detail: List[PeakXCorrResult],
    sr: int,
    name_a: str,
    name_b: str,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot per-peak xcorr detail for a single track pair.

    Top: both waveforms with peak locations marked, colored by correlation.
    Middle: delay scatter (each peak's detected delay in ms vs time).
    Bottom: correlation scatter (each peak's correlation vs time).
    """
    if not detail:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(0.5, 0.5, 'No peaks to display', ha='center', va='center', fontsize=14)
        ax.set_title(f'{name_a} vs {name_b} - No data')
        return fig

    times = [d.peak_time_s for d in detail]
    delays_ms = [d.delay_ms for d in detail]
    corrs = [d.correlation for d in detail]
    pols = [d.polarity for d in detail]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1.2, 1.2]})

    t = np.arange(max(len(audio_a), len(audio_b))) / sr

    # Top: waveforms + peak markers
    ax1.plot(t[:len(audio_a)], audio_a, color='#90CAF9', linewidth=0.4, alpha=0.7, label=name_a)
    ax1.plot(t[:len(audio_b)], audio_b, color='#FFAB91', linewidth=0.4, alpha=0.7, label=name_b)

    # Color peaks by correlation strength
    scatter_colors = ['#4CAF50' if c > 0.3 else '#FF9800' if c > 0.15 else '#F44336' for c in corrs]
    pol_markers = ['v' if p < 0 else '^' for p in pols]
    for t_p, c_color, m in zip(times, scatter_colors, pol_markers):
        idx = int(t_p * sr)
        if idx < len(audio_a):
            ax1.axvline(t_p, color=c_color, alpha=0.3, linewidth=0.8)

    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f'{name_a} vs {name_b} - Per-Peak XCorr Detail', fontsize=12, fontweight='bold')

    # Middle: delay scatter
    colors_arr = ['#1565C0' if p > 0 else '#C62828' for p in pols]
    ax2.scatter(times, delays_ms, c=colors_arr, s=30, zorder=5, edgecolors='white', linewidth=0.5)
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    mean_delay = np.mean(delays_ms)
    median_delay = np.median(delays_ms)
    ax2.axhline(mean_delay, color='#1565C0', linewidth=1.5, linestyle='-', alpha=0.7,
                label=f'mean={mean_delay:.2f}ms')
    ax2.axhline(median_delay, color='#4CAF50', linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'median={median_delay:.2f}ms')
    ax2.set_ylabel('Delay (ms)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Bottom: correlation scatter
    ax3.scatter(times, corrs, c=scatter_colors, s=30, zorder=5, edgecolors='white', linewidth=0.5)
    mean_corr = np.mean(corrs)
    ax3.axhline(mean_corr, color='#1565C0', linewidth=1.5, linestyle='-', alpha=0.7,
                label=f'mean={mean_corr:.3f}')
    ax3.set_ylabel('Correlation')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylim(0, max(corrs) * 1.2 if corrs else 1.0)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_delay_matrix(
    delay_matrix_ms: np.ndarray,
    track_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = "Delay Matrix (ms)"
) -> plt.Figure:
    """Plot delay matrix as a diverging heatmap (blue=negative, red=positive).

    Values are in ms. Diagonal is masked. Annotated with delay values.
    """
    N = delay_matrix_ms.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Mask diagonal
    display = np.ma.masked_where(np.eye(N, dtype=bool), delay_matrix_ms)

    # Symmetric range around zero
    vmax = np.max(np.abs(delay_matrix_ms[~np.eye(N, dtype=bool)]))
    vmax = max(vmax, 0.5)  # at least 0.5ms range

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='#e0e0e0')

    im = ax.imshow(display, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Delay (ms)', rotation=270, labelpad=15)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(track_names, rotation=45, ha='right')
    ax.set_yticklabels(track_names)

    # Annotate
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            val = delay_matrix_ms[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.text(0.02, -0.12, '+ve = column track lags row track',
            transform=ax.transAxes, fontsize=8, color='#666666')

    plt.tight_layout()
    return fig


def plot_lens_overview(
    broadband_corr: np.ndarray,
    broadband_delay_ms: np.ndarray,
    envelope_corr: np.ndarray,
    envelope_delay_ms: np.ndarray,
    windowed_corr: np.ndarray,
    windowed_delay_ms: np.ndarray,
    peak_conf: np.ndarray,
    peak_delay_ms: np.ndarray,
    track_names: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """4x2 overview of all lenses: top row = delays, bottom row = confidence/correlation."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('Alignment Lenses Overview', fontsize=16, fontweight='bold', y=0.98)

    # Top row: delays
    plot_delay_matrix(broadband_delay_ms, track_names, ax=axes[0, 0], title='Broadband Delay')
    plot_delay_matrix(envelope_delay_ms, track_names, ax=axes[0, 1], title='Envelope Delay')
    plot_delay_matrix(windowed_delay_ms, track_names, ax=axes[0, 2], title='Windowed Delay')
    plot_delay_matrix(peak_delay_ms, track_names, ax=axes[0, 3], title='Peak-Match Delay')

    # Bottom row: correlation / confidence
    plot_correlation_matrix(broadband_corr, track_names, ax=axes[1, 0],
                            title='Broadband Corr', full_scale=True)
    plot_correlation_matrix(envelope_corr, track_names, ax=axes[1, 1],
                            title='Envelope Corr', full_scale=True)
    plot_correlation_matrix(windowed_corr, track_names, ax=axes[1, 2],
                            title='Windowed Corr', full_scale=True)
    plot_correlation_matrix(peak_conf, track_names, ax=axes[1, 3],
                            title='Peak-Match Conf', full_scale=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_triple_delays(
    broadband_delay_ms: np.ndarray,
    envelope_delay_ms: np.ndarray,
    windowed_delay_ms: np.ndarray,
    track_names: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot all 3 delay matrices side by side."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

    plot_delay_matrix(broadband_delay_ms, track_names, ax=ax1,
                      title="Broadband Delay (ms)")
    plot_delay_matrix(envelope_delay_ms, track_names, ax=ax2,
                      title="Envelope Delay (ms)")
    plot_delay_matrix(windowed_delay_ms, track_names, ax=ax3,
                      title="Windowed Delay (ms)")

    fig.suptitle('Magic Phase - Delay Matrices (N x N)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_triple_xcorr(
    broadband_matrix: np.ndarray,
    envelope_matrix: np.ndarray,
    windowed_matrix: np.ndarray,
    track_names: List[str],
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Plot all 3 correlation matrices side by side."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

    plot_correlation_matrix(broadband_matrix, track_names, ax=ax1,
                            title="Broadband XCorr", full_scale=True)
    plot_correlation_matrix(envelope_matrix, track_names, ax=ax2,
                            title="RMS Envelope XCorr", full_scale=True)
    plot_correlation_matrix(windowed_matrix, track_names, ax=ax3,
                            title="Windowed XCorr", full_scale=True)

    fig.suptitle('Magic Phase — XCorr Diagnostics (N×N)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


# =============================================================================
# CLUSTER + STAR ALIGNMENT ALGORITHM
# =============================================================================

def find_clusters(
    corr_matrix: np.ndarray,
    threshold: float = 0.15
) -> List[ClusterInfo]:
    """Find connected components in the broadband correlation graph.

    Uses Union-Find. Edges exist where corr >= threshold.
    Root per cluster = highest intra-cluster row-sum.
    Sorted by size descending; largest = main (is_main=True).
    """
    N = corr_matrix.shape[0]

    # Union-Find
    parent = list(range(N))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Connect all pairs above threshold
    for i in range(N):
        for j in range(i + 1, N):
            if corr_matrix[i, j] >= threshold:
                union(i, j)

    # Group tracks by component
    components: Dict[int, List[int]] = {}
    for i in range(N):
        root = find(i)
        components.setdefault(root, []).append(i)

    # Build ClusterInfo list, sorted by size descending
    clusters_raw = sorted(components.values(), key=len, reverse=True)

    clusters = []
    for cid, members in enumerate(clusters_raw):
        # Root = highest intra-cluster broadband row-sum
        best_root = members[0]
        best_sum = -1.0
        for i in members:
            row_sum = sum(
                corr_matrix[i, j]
                for j in members
                if i != j and corr_matrix[i, j] >= threshold
            )
            if row_sum > best_sum:
                best_sum = row_sum
                best_root = i

        clusters.append(ClusterInfo(
            cluster_id=cid,
            track_indices=members,
            root_idx=best_root,
            is_main=(cid == 0),  # Largest cluster is main
        ))

    return clusters


def bridge_orphans(
    clusters: List[ClusterInfo],
    env_corr_matrix: np.ndarray,
    env_delay_matrix: np.ndarray,
    audios: List[np.ndarray],
    sr: int,
    all_peaks: List[np.ndarray],
    bridge_threshold: float = 0.30
) -> Tuple[List[ClusterInfo], List[AlignmentEdge]]:
    """Bridge non-main clusters to the main cluster via envelope correlation.

    For each non-main cluster root: find the main-cluster track with highest
    envelope correlation. If >= bridge_threshold, create a Tier 2 bridge edge
    using windowed XCorr delay (computed on demand, not from the NxN matrix).

    Returns updated clusters and list of Tier 2 bridge edges.
    """
    if len(clusters) <= 1:
        return clusters, []

    main_cluster = clusters[0]
    main_indices = set(main_cluster.track_indices)

    bridge_edges = []
    merged_into_main = []

    for cluster in clusters[1:]:
        orphan_root = cluster.root_idx

        # Find best envelope correlation to any main-cluster track
        best_corr = 0.0
        best_target = -1
        for m_idx in main_cluster.track_indices:
            ec = env_corr_matrix[orphan_root, m_idx]
            if ec > best_corr:
                best_corr = ec
                best_target = m_idx

        if best_corr >= bridge_threshold and best_target >= 0:
            # Compute windowed XCorr for precise delay
            peaks_orphan = all_peaks[orphan_root] if orphan_root < len(all_peaks) else np.array([])
            peaks_target = all_peaks[best_target] if best_target < len(all_peaks) else np.array([])

            if len(peaks_orphan) > 0 or len(peaks_target) > 0:
                w_delay, w_corr, w_pol, _ = windowed_xcorr_pair(
                    audios[best_target], audios[orphan_root],
                    peaks_target, peaks_orphan, sr
                )
            else:
                # Fallback to envelope delay
                w_delay = env_delay_matrix[best_target, orphan_root]
                w_corr = best_corr
                w_pol = 1

            bridge_edges.append(AlignmentEdge(
                child_idx=orphan_root,
                root_idx=best_target,
                tier=2,
                delay_samples=float(w_delay),
                polarity=int(w_pol),
                correlation=float(best_corr),
                source='windowed',
            ))
            merged_into_main.append(cluster)

    return clusters, bridge_edges


def magic_align(
    audios: List[np.ndarray],
    sr: int,
    track_names: Optional[List[str]] = None,
    detect_delay_fn=None,
    threshold: float = 0.15,
    bridge_threshold: float = 0.30,
    verbose: bool = True
) -> MagicAlignResult:
    """Main orchestrator: compute the alignment PLAN (does NOT apply corrections).

    Steps:
      1. Compute broadband NxN matrix
      2. Compute envelope NxN matrix
      3. find_clusters() from broadband
      4. Detect transients (needed for windowed xcorr in bridging)
      5. bridge_orphans() for orphan clusters via envelope
      6. Build AlignmentEdge list (Tier 1 from broadband, Tier 2 from bridges)
      7. Build TrackAlignment per track
      8. Return MagicAlignResult
    """
    N = len(audios)
    if track_names is None:
        track_names = [f"Track {i}" for i in range(N)]

    # Step 1: Broadband NxN
    if verbose:
        print(f"\n  Computing {N}x{N} broadband correlation matrix...")
    bb_corr, bb_delay, bb_pol = compute_correlation_matrix(
        audios, sr, max_delay_ms=50.0, detect_delay_fn=detect_delay_fn
    )

    # Step 2: Envelope NxN
    if verbose:
        print(f"  Computing {N}x{N} envelope correlation matrix...")
    env_corr, env_delay, _ = compute_correlation_matrix(
        audios, sr, max_delay_ms=50.0,
        detect_delay_fn=detect_delay_envelope_xcorr
    )

    # Step 3: Clustering
    if verbose:
        print(f"  Finding clusters (threshold={threshold})...")
    clusters = find_clusters(bb_corr, threshold)

    if verbose:
        for c in clusters:
            names = [track_names[i] for i in c.track_indices]
            tag = " (MAIN)" if c.is_main else ""
            print(f"    Cluster {c.cluster_id}{tag}: root={track_names[c.root_idx]}, "
                  f"members=[{', '.join(names)}]")

    # Step 4: Detect transients for windowed xcorr
    if verbose:
        print(f"  Detecting transients...")
    all_peaks = []
    for i in range(N):
        peaks = detect_transients(audios[i], sr)
        all_peaks.append(peaks)
        if verbose:
            print(f"    {track_names[i]}: {len(peaks)} peaks")

    # Note: Polarity is determined empirically in the correction loop
    # (try both, keep whichever sums better). bb_pol from broadband XCorr
    # is kept in the plan for informational display only.

    # Step 5: Bridge orphans
    if verbose:
        print(f"  Bridging orphan clusters (bridge_threshold={bridge_threshold})...")
    clusters, bridge_edges = bridge_orphans(
        clusters, env_corr, env_delay, audios, sr, all_peaks, bridge_threshold
    )

    if verbose:
        for be in bridge_edges:
            print(f"    BRIDGE: {track_names[be.child_idx]} → {track_names[be.root_idx]} "
                  f"(env_corr={be.correlation:.3f}, delay={be.delay_samples:.1f}, "
                  f"pol={'INV' if be.polarity < 0 else 'normal'})")
        # Check for true orphans
        bridged_roots = {be.child_idx for be in bridge_edges}
        main_indices = set(clusters[0].track_indices)
        for c in clusters[1:]:
            if c.root_idx not in bridged_roots:
                names = [track_names[i] for i in c.track_indices]
                print(f"    TRUE ORPHAN cluster: [{', '.join(names)}] (no bridge found)")

    # Step 6: Build all edges
    edges = list(bridge_edges)  # Start with Tier 2 bridges

    # Tier 1 edges: within each cluster, every non-root → root (star)
    for cluster in clusters:
        for idx in cluster.track_indices:
            if idx == cluster.root_idx:
                continue
            edges.append(AlignmentEdge(
                child_idx=idx,
                root_idx=cluster.root_idx,
                tier=1,
                delay_samples=float(bb_delay[cluster.root_idx, idx]),
                polarity=int(bb_pol[cluster.root_idx, idx]),
                correlation=float(bb_corr[cluster.root_idx, idx]),
                source='broadband',
            ))

    # Step 7: Build TrackAlignment per track
    # Determine which clusters are bridged
    bridged_roots = {be.child_idx: be for be in bridge_edges}
    main_cluster = clusters[0]
    main_indices = set(main_cluster.track_indices)

    alignments: Dict[int, TrackAlignment] = {}

    for cluster in clusters:
        bridge_edge = bridged_roots.get(cluster.root_idx)
        cluster_shift = bridge_edge.delay_samples if bridge_edge else 0.0
        cluster_pol_shift = bridge_edge.polarity if bridge_edge else 1

        for idx in cluster.track_indices:
            if idx == cluster.root_idx and cluster.is_main:
                # Main cluster root: tier 0
                alignments[idx] = TrackAlignment(
                    track_idx=idx,
                    cluster_id=cluster.cluster_id,
                    tier=0,
                    delay_samples=0.0,
                    polarity=1,
                    aligned_to=None,
                    cluster_shift_samples=0.0,
                )
            elif idx == cluster.root_idx and bridge_edge:
                # Bridged orphan cluster root: tier 2
                alignments[idx] = TrackAlignment(
                    track_idx=idx,
                    cluster_id=cluster.cluster_id,
                    tier=2,
                    delay_samples=cluster_shift,
                    polarity=cluster_pol_shift,
                    aligned_to=bridge_edge.root_idx,
                    cluster_shift_samples=cluster_shift,
                )
            elif idx == cluster.root_idx:
                # Unbridged orphan cluster root: true orphan
                alignments[idx] = TrackAlignment(
                    track_idx=idx,
                    cluster_id=cluster.cluster_id,
                    tier=-1,
                    delay_samples=0.0,
                    polarity=1,
                    aligned_to=None,
                    cluster_shift_samples=0.0,
                )
            else:
                # Non-root: Tier 1 within cluster
                intra_delay = float(bb_delay[cluster.root_idx, idx])
                intra_pol = int(bb_pol[cluster.root_idx, idx])

                if bridge_edge:
                    # Bridged cluster member: tier 1 intra + tier 2 shift
                    total_delay = intra_delay + cluster_shift
                    total_pol = intra_pol * cluster_pol_shift
                    tier = 1  # Still gets full spectral within its cluster
                else:
                    total_delay = intra_delay
                    total_pol = intra_pol
                    tier = 1 if cluster.is_main else 1  # Still tier 1 within orphan cluster

                # Check if this is a truly orphan single-track (no bridge)
                if not cluster.is_main and not bridge_edge and len(cluster.track_indices) == 1:
                    tier = -1

                alignments[idx] = TrackAlignment(
                    track_idx=idx,
                    cluster_id=cluster.cluster_id,
                    tier=tier,
                    delay_samples=total_delay,
                    polarity=total_pol,
                    aligned_to=cluster.root_idx,
                    cluster_shift_samples=cluster_shift,
                )

    if verbose:
        print_cluster_structure(
            MagicAlignResult(
                clusters=clusters,
                alignments=alignments,
                edges=edges,
                main_cluster_id=main_cluster.cluster_id,
                broadband_corr_matrix=bb_corr,
                broadband_delay_matrix=bb_delay,
                broadband_pol_matrix=bb_pol,
                envelope_corr_matrix=env_corr,
                envelope_delay_matrix=env_delay,
            ),
            track_names, sr
        )

    return MagicAlignResult(
        clusters=clusters,
        alignments=alignments,
        edges=edges,
        main_cluster_id=main_cluster.cluster_id,
        broadband_corr_matrix=bb_corr,
        broadband_delay_matrix=bb_delay,
        broadband_pol_matrix=bb_pol,
        envelope_corr_matrix=env_corr,
        envelope_delay_matrix=env_delay,
    )


def print_cluster_structure(result: MagicAlignResult, track_names: List[str], sr: int):
    """Print the cluster + star layout with tier info."""
    print(f"\n  Cluster structure:")
    for cluster in result.clusters:
        tag = " (MAIN)" if cluster.is_main else ""
        root_name = track_names[cluster.root_idx]
        print(f"    Cluster {cluster.cluster_id}{tag}:")
        print(f"      {root_name} (ROOT, tier {result.alignments[cluster.root_idx].tier})")

        for idx in cluster.track_indices:
            if idx == cluster.root_idx:
                continue
            a = result.alignments[idx]
            delay_ms = a.delay_samples / sr * 1000
            pol_str = ", INV" if a.polarity < 0 else ""
            tier_str = f"tier {a.tier}"
            shift_str = ""
            if a.cluster_shift_samples != 0:
                shift_ms = a.cluster_shift_samples / sr * 1000
                shift_str = f" [+bridge {shift_ms:+.2f}ms]"
            corr_val = result.broadband_corr_matrix[cluster.root_idx, idx]
            print(f"      └── {track_names[idx]} ({corr_val:.2f}, {tier_str}) "
                  f"→ delay={a.delay_samples:+.1f} ({delay_ms:+.2f}ms){pol_str}{shift_str}")


def plot_cluster_overview(
    result: MagicAlignResult,
    track_names: List[str],
    threshold: float = 0.15,
    sr: int = 48000,
    total_gain_db: float = 0.0,
    track_gains: Optional[Dict[int, float]] = None,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Combined result overview: correlation matrix + alignment results table."""
    fig = plt.figure(figsize=(16, 8))

    # Left: broadband correlation matrix
    ax1 = fig.add_subplot(1, 2, 1)
    plot_correlation_matrix(result.broadband_corr_matrix, track_names,
                            threshold, ax=ax1,
                            title="Broadband Correlation Matrix")

    # Right: cluster alignment table
    ax2 = fig.add_subplot(1, 2, 2)
    _plot_cluster_table(result, track_names, sr=sr,
                        total_gain_db=total_gain_db,
                        track_gains=track_gains, ax=ax2)

    plt.suptitle("Magic Phase — Alignment Result", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def _plot_cluster_table(
    result: MagicAlignResult,
    track_names: List[str],
    sr: int = 48000,
    total_gain_db: float = 0.0,
    track_gains: Optional[Dict[int, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Render cluster+star structure as a colored results table.

    Rows grouped by cluster, color-coded by tier.
    Columns: Track | Tier | Aligned To | Delay (ms) | Pol | Corr | Gain
    Big headline with total sum gain at top.
    """
    N = len(track_names)
    if track_gains is None:
        track_gains = {}

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = ax.figure

    ax.axis('off')

    # ── Big gain headline ──
    gain_color = '#2E7D32' if total_gain_db > 0 else '#C62828' if total_gain_db < 0 else '#666666'
    ax.text(0.50, 0.97, f'Sum Gain: {total_gain_db:+.1f} dB',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=16, fontweight='bold', color=gain_color)

    # Tier colors
    tier_colors = {
        0:  '#C8E6C9',   # Green: root
        1:  '#BBDEFB',   # Blue: Tier 1
        2:  '#FFE0B2',   # Orange: Tier 2
        -1: '#E0E0E0',   # Gray: orphan
    }
    tier_labels = {0: 'ROOT', 1: 'T1 full', 2: 'T2 time', -1: 'orphan'}

    # Build rows: ordered by cluster, root first within each cluster
    rows = []
    bridged_edges = {e.child_idx: e for e in result.edges if e.tier == 2}

    for cluster in result.clusters:
        bridge_edge = bridged_edges.get(cluster.root_idx)
        tag = "MAIN" if cluster.is_main else (
            f"-> {track_names[bridge_edge.root_idx]}" if bridge_edge else "ORPHAN"
        )
        rows.append(('cluster_header', cluster.cluster_id, tag, len(cluster.track_indices)))

        root_a = result.alignments[cluster.root_idx]
        rows.append(('track', cluster.root_idx, root_a))

        non_root = [i for i in cluster.track_indices if i != cluster.root_idx]
        non_root.sort(key=lambda i: result.broadband_corr_matrix[cluster.root_idx, i], reverse=True)
        for idx in non_root:
            rows.append(('track', idx, result.alignments[idx]))

    # Table layout — 7 columns now
    col_headers = ['Track', 'Tier', 'Aligned To', 'Delay (ms)', 'Pol', 'Corr', 'Gain']
    col_widths = [0.24, 0.10, 0.20, 0.14, 0.07, 0.10, 0.11]
    col_x = [0.02]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    n_rows = len(rows)
    row_height = min(0.055, 0.75 / max(n_rows, 1))
    y_start = 0.87

    # Column headers
    for j, header in enumerate(col_headers):
        ax.text(col_x[j] + col_widths[j] / 2, y_start, header,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, fontweight='bold', color='#333333')

    # Header underline
    ax.plot([0.02, 0.98], [y_start - row_height * 0.45, y_start - row_height * 0.45],
            transform=ax.transAxes, color='#333333', linewidth=1.5, clip_on=False)

    y = y_start - row_height

    for row in rows:
        if row[0] == 'cluster_header':
            _, cid, tag, n_members = row
            y -= row_height * 0.3
            label = f"Cluster {cid}  ({tag}, {n_members} tracks)"
            ax.text(0.02, y, label, transform=ax.transAxes, ha='left', va='center',
                    fontsize=7.5, fontweight='bold', color='#555555', style='italic')
            y -= row_height * 0.7
            continue

        _, idx, a = row
        name = track_names[idx]
        tier = a.tier
        bg_color = tier_colors.get(tier, '#FFFFFF')

        # Row background
        rect = plt.Rectangle((0.01, y - row_height * 0.45), 0.98, row_height * 0.9,
                              transform=ax.transAxes, facecolor=bg_color,
                              edgecolor='white', linewidth=0.5, clip_on=False)
        ax.add_patch(rect)

        # Track name
        ax.text(col_x[0] + 0.01, y, name, transform=ax.transAxes,
                ha='left', va='center', fontsize=8.5, fontweight='bold', color='#222222')

        # Tier
        ax.text(col_x[1] + col_widths[1] / 2, y, tier_labels.get(tier, '?'),
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7.5, fontweight='bold',
                color='#2E7D32' if tier == 0 else '#1565C0' if tier == 1 else
                      '#E65100' if tier == 2 else '#757575')

        # Aligned to
        if a.aligned_to is not None:
            aligned_name = track_names[a.aligned_to]
        else:
            aligned_name = '---'
        ax.text(col_x[2] + col_widths[2] / 2, y, aligned_name,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7.5, color='#444444')

        # Delay
        if tier == 0 or tier == -1:
            delay_str = '---'
        else:
            delay_ms = a.delay_samples / sr * 1000
            delay_str = f'{delay_ms:+.2f}'
        ax.text(col_x[3] + col_widths[3] / 2, y, delay_str,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7.5, fontfamily='monospace', color='#444444')

        # Polarity
        pol_str = 'INV' if a.polarity < 0 else ''
        pol_color = '#C62828' if a.polarity < 0 else '#444444'
        ax.text(col_x[4] + col_widths[4] / 2, y, pol_str,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=pol_color)

        # Correlation
        if tier == 0 or tier == -1 or a.aligned_to is None:
            corr_str = '---'
        else:
            corr_val = result.broadband_corr_matrix[a.aligned_to, idx]
            if tier == 2:
                corr_val = result.envelope_corr_matrix[a.aligned_to, idx]
            corr_str = f'{corr_val:.2f}'
        ax.text(col_x[5] + col_widths[5] / 2, y, corr_str,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7.5, color='#444444')

        # Gain (dB)
        if idx in track_gains:
            g = track_gains[idx]
            gain_str = f'{g:+.1f}'
            g_color = '#2E7D32' if g > 0.5 else '#C62828' if g < -0.5 else '#888888'
        elif tier == 0 or tier == -1:
            gain_str = '---'
            g_color = '#444444'
        else:
            gain_str = ''
            g_color = '#444444'
        ax.text(col_x[6] + col_widths[6] / 2, y, gain_str,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7.5, fontweight='bold', fontfamily='monospace', color=g_color)

        y -= row_height

    # Legend at bottom
    legend_y = max(y - row_height * 0.5, 0.02)
    legend_items = [
        ('#C8E6C9', 'Root'),
        ('#BBDEFB', 'Tier 1 (full)'),
        ('#FFE0B2', 'Tier 2 (time)'),
        ('#E0E0E0', 'Orphan'),
    ]
    lx = 0.02
    for color, label in legend_items:
        rect = plt.Rectangle((lx, legend_y - 0.01), 0.03, 0.02,
                              transform=ax.transAxes, facecolor=color,
                              edgecolor='#999999', linewidth=0.5, clip_on=False)
        ax.add_patch(rect)
        ax.text(lx + 0.04, legend_y, label, transform=ax.transAxes,
                ha='left', va='center', fontsize=7, color='#666666')
        lx += 0.15

    return fig


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    track_names: List[str],
    threshold: float = 0.15,
    ax: Optional[plt.Axes] = None,
    title: str = "Correlation Matrix",
    full_scale: bool = False
) -> plt.Figure:
    """
    Plot correlation matrix as heatmap with annotations.

    Cells below threshold are grayed out (unless full_scale=True).
    full_scale=True: fixed 0-1 range, no threshold masking.
    """
    N = corr_matrix.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    if full_scale:
        # Fixed 0-1 range, no masking
        cmap = plt.cm.YlOrRd.copy()
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect='equal')
    else:
        # Create masked array for cells below threshold
        masked = np.ma.masked_where(corr_matrix < threshold, corr_matrix)
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color='#f0f0f0')  # Gray for below threshold
        im = ax.imshow(masked, cmap=cmap, vmin=threshold, vmax=1.0, aspect='equal')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', rotation=270, labelpad=15)

    # Set ticks
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(track_names, rotation=45, ha='right')
    ax.set_yticklabels(track_names)

    # Annotate cells
    for i in range(N):
        for j in range(N):
            if i != j:
                val = corr_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                if not full_scale and val < threshold:
                    color = '#999999'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color,
                        fontweight='bold' if (full_scale or val >= threshold) else 'normal')

    # Grid
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)

    ax.set_title(title, fontsize=12, fontweight='bold')

    if not full_scale:
        ax.text(0.02, -0.15, f'Gray cells: below threshold ({threshold})',
                transform=ax.transAxes, fontsize=9, color='#666666')

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing cluster+star with synthetic correlation matrix...")

    track_names = ["SnTop", "Kick", "SnBot", "OH_L", "OH_R"]
    corr_matrix = np.array([
        [0.00, 0.08, 0.62, 0.40, 0.35],
        [0.08, 0.00, 0.08, 0.09, 0.13],
        [0.62, 0.08, 0.00, 0.32, 0.35],
        [0.40, 0.09, 0.32, 0.00, 0.22],
        [0.35, 0.13, 0.35, 0.22, 0.00],
    ])

    clusters = find_clusters(corr_matrix, threshold=0.15)
    print(f"\nClusters found: {len(clusters)}")
    for c in clusters:
        names = [track_names[i] for i in c.track_indices]
        print(f"  Cluster {c.cluster_id}: root={track_names[c.root_idx]}, "
              f"members=[{', '.join(names)}], main={c.is_main}")
