# Magic Phase Alignment Algorithm v2

*2026-03-01 — Locked algorithm specification*

**Supersedes:** `SPANNING_TREE_ALIGNMENT.md`, `MULTI_TRACK_ALIGNMENT.md`,
`dual_correlation_strategy.md`, `late_night_multi_mic_conclusions.md`

Those documents capture the journey. This one captures the destination.

---

## Philosophy

1. **Only align what correlates.** Correlation = shared physics. No correlation = nothing to sum.
2. **Star topology, never tree.** Every track aligns to one root. No cascading errors.
3. **Tiered correction.** Full spectral where confident, time-only where marginal, untouched where pointless.
4. **Automatic everything.** No manual reference, no grouping, no knobs. 3 clicks, mind blown.

---

## The Four Lenses

Each lens answers a different question. No overlap in responsibilities.

| Lens | Question It Answers | Used For |
|------|---------------------|----------|
| **Broadband XCorr** | "How much gain will alignment produce?" | Clustering + Tier 1 delay |
| **RMS Envelope XCorr** | "Do these mics hear the same events?" | Tier 2 bridging (orphan rescue) |
| **Windowed XCorr** | "What's the direct-sound delay between transients?" | Tier 2 delay estimation + validation |
| **Peak-Match Confidence** | "What fraction of transients are shared?" | Validation + diagnostics |

### Why broadband delay for Tier 1

Broadband XCorr finds the delay tau that maximizes:

    sum_energy = E_x + E_y + 2 * R_xy(tau)

This is literally the delay that makes the sum loudest. Maximum gain, by definition.
That's what we want. That's what the user hears.

Star topology protects us from the edge cases where broadband gets confused
(e.g., OH L vs OH R at -7.5ms). Each track aligns to the root independently —
the problematic pairwise delays are never used directly.

### Why windowed delay for Tier 2

At low broadband correlation (0.08-0.15), the XCorr peak is in the noise.
Broadband delay is unreliable. Windowed XCorr isolates transient onsets
in short windows (25ms) — direct sound before reflections arrive.
It's physically grounded, cross-validated by peak-match.

### Lens computation

**Broadband XCorr** (existing `detect_delay_xcorr`):
- Full waveform cross-correlation on analysis window
- Returns: delay (sub-sample via parabolic interpolation), correlation, polarity

**RMS Envelope XCorr** (`detect_delay_envelope_xcorr`):
- 900Hz 2nd-order Butterworth highpass (tracks attacks, not sub energy)
- 3ms sliding RMS window
- Cross-correlate the envelopes
- Returns: delay, correlation (typically much higher than broadband)

**Windowed XCorr** (`windowed_xcorr_pair`):
- Detect transients per track (adaptive-floor Schmitt trigger)
- Union all peaks from both tracks
- For each peak: 25ms window, broadband XCorr within window, max_delay 15ms
- Returns: mean delay, mean correlation, dominant polarity

**Peak-Match** (`peak_match_pair`):
- For each peak in A, find closest peak in B (signed distance)
- Filter: only pairs within 20ms
- Sliding 3ms density window on distances -> find densest cluster
- Returns: median delay of cluster, confidence = cluster_size / len(peaks_a)
- Directional: A->B confidence != B->A confidence

---

## Algorithm

### Stage 1: Analysis

Compute on the analysis window (default 7.5s from start of audio):

```
1. Load all N tracks, trim to common length
2. Compute N x N broadband correlation matrix
   - For each pair (i,j): detect_delay_xcorr -> delay, corr, polarity
   - Store: corr_matrix[i,j], delay_matrix[i,j], pol_matrix[i,j]
3. Compute N x N envelope correlation matrix
   - For each pair (i,j): detect_delay_envelope_xcorr -> delay, corr
   - Store: env_corr_matrix[i,j], env_delay_matrix[i,j]
```

Cost: N*(N-1)/2 pairs for each matrix. For 8 tracks = 28 pairs. For 24 tracks = 276 pairs.

### Stage 2: Clustering

```
4. Build adjacency graph from broadband correlation
   - Edge (i,j) exists if corr_matrix[i,j] >= CLUSTER_THRESHOLD
   - CLUSTER_THRESHOLD = 0.15 (TODO: validate across diverse material)

5. Find connected components (Union-Find)
   - Each component = one cluster
   - Single-track components = potential orphans

6. Within each cluster:
   - Root = track with highest row sum (broadband corr, edges >= threshold only)
   - This selects the most "connected" track — hears the most from other sources
```

### Stage 3: Tier 2 Bridging (Orphan Rescue)

```
7. For each orphan track O:
   a. Find the cluster root R with highest envelope correlation to O
   b. If env_corr_matrix[O, R] >= BRIDGE_THRESHOLD:
      - Create Tier 2 edge: O -> R
      - O joins R's cluster as a Tier 2 member
   c. If no cluster root has sufficient envelope correlation:
      - O remains untouched (true orphan — different world)

   BRIDGE_THRESHOLD = 0.30 (envelope, TODO: validate)

8. For orphan clusters with multiple tracks (e.g., Kick In + Kick Out):
   a. Internal alignment is Tier 1 (they correlate well broadband)
   b. The cluster root becomes the bridge representative
   c. Bridge the cluster root to the main cluster root (Tier 2)
   d. Entire orphan cluster shifts by the bridge delay
   - This is hierarchical star, NOT tree:
     - Level 1: star within each cluster
     - Level 2: cluster roots star-align to the main cluster root
   - Each track gets at most 2 corrections (intra-cluster + cluster shift)
   - No cascading, no path-following
```

### Stage 4: Correction

```
9. Within each cluster, process star edges (root is untouched):

   For each Tier 1 edge (child -> root):
   a. Time correction:
      - Delay from broadband XCorr (maximizes sum energy)
      - Sub-sample precision via parabolic interpolation
      - Apply via FFT phase ramp (or delay buffer for integer part)
   b. Polarity correction:
      - From broadband XCorr polarity detection
      - TODO: improve with multi-band polarity analysis
   c. Spectral phase correction:
      - STFT analysis of corrected pair (4096 FFT, 75% overlap, Hann)
      - Per-bin phase difference, weighted by coherence
      - Coherence threshold: 0.4
      - Max correction: +/-120 degrees
      - Gaussian smoothing (prevents artifacts)
      - This is the secret sauce. Per-frequency phase rotation.
   d. Sanity check:
      - Compare RMS(root + corrected) vs RMS(root + raw)
      - If correction made it worse: revert to raw

   For each Tier 2 edge (orphan -> cluster root):
   a. Time correction ONLY:
      - Delay from windowed XCorr (broadband unreliable at low correlation)
      - Sub-sample precision
   b. Polarity correction:
      - From windowed XCorr dominant polarity
   c. NO spectral phase correction:
      - Low broadband correlation = low coherence = spectral correction
        would be fitting noise
   d. Sanity check:
      - Same RMS comparison, revert if worse

10. For bridged orphan clusters (e.g., {Kick In, Kick Out}):
    a. First: Tier 1 star alignment within the orphan cluster
       - Kick Out aligns to Kick In (broadband delay, full spectral)
    b. Then: Tier 2 shift of entire cluster
       - Both Kick In and Kick Out shift by the bridge delay
    c. Order matters: intra-cluster first, then cluster shift
```

### Stage 5: Output

```
11. Write corrected audio files
12. Write sum (all corrected tracks) and raw sum (all original tracks)
13. Report per-track:
    - Delay applied (ms, samples, sub-sample)
    - Polarity (normal / inverted)
    - Correlation (broadband, to alignment partner)
    - Coherence (mean, from spectral analysis)
    - Tier (1 = full correction, 2 = time only, uncorrected = orphan/root)
    - Sum gain vs partner (dB)
14. Report overall:
    - Total sum energy gain (dB)
    - Cluster structure (which tracks grouped together)
    - Root track per cluster
```

---

## Polarity Detection

### Current Implementation (v1)

Simple sign of broadband XCorr peak:

```python
polarity = 1 if peak_val > 0 else -1
```

This works when:
- Same source, close mics (snare top/bot, kick in/out)
- Clear waveform similarity

This fails when:
- Low correlation (polarity of noise)
- Mixed sources (which component determines the sign?)
- Frequency-dependent polarity (e.g., bottom mic inverted at low freq but not high)

### Improvement Needed (v2 — future work)

Polarity is a critical decision. Getting it wrong costs 6dB instead of gaining 3dB.
Needs dedicated investigation with diverse test material. Options to explore:

1. **Multi-band polarity**: Check polarity in frequency bands independently.
   If low band says inverted but high band says normal, the low band wins
   (more energy, more audible).

2. **Coherence-weighted polarity**: Only consider frequency bins with
   coherence > threshold for the polarity vote.

3. **Onset polarity**: Check polarity at transient onsets only (first few ms
   of each hit). The direct sound determines the "true" polarity — reflections
   and room ambience can flip apparent polarity at some frequencies.

4. **Sanity check is the safety net**: The RMS sum check (step 9d) catches
   catastrophic polarity errors. If flipping made it worse, revert.
   This is the last line of defense.

---

## Transient Detection

Adaptive-floor Schmitt trigger on 900Hz HP RMS envelope:

```
1. Compute RMS envelope (900Hz HP, 3ms window)
2. Maintain adaptive floor:
   - Decays exponentially toward zero (floor_decay_ms = 80ms)
   - During hold period after onset: floor tracks envelope UPWARD
     (follows the attack peak so re-arm threshold rises)
3. Onset trigger: envelope > floor * rise_ratio AND envelope > min_rise
   - rise_ratio = 2.5 (must be 2.5x above adaptive floor)
   - min_rise = 0.04 (absolute minimum, prevents noise triggers)
4. Hold period: 30ms (prevents double-triggering on same transient)
```

This handles:
- Clean transients (drums, guitar picks)
- Sustained energy with new hits on top (cymbal wash + snare)
- Re-arming during loud passages (adaptive floor tracks the energy)

---

## Thresholds & Parameters

All fixed. No user configuration. "Magic Phase, not Manual Phase."

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Analysis window | 7.5s | Captures enough events for statistics |
| CLUSTER_THRESHOLD | 0.15 | Broadband correlation floor for meaningful alignment |
| BRIDGE_THRESHOLD | 0.30 | Envelope correlation floor for Tier 2 bridging |
| Max delay search | 50ms | ~17m at 343 m/s, covers any studio setup |
| Windowed XCorr window | 25ms | ~8.6m, captures direct sound before reflections |
| Windowed XCorr max delay | 15ms | ~5.1m, max mic distance within window |
| Coherence threshold | 0.4 | Conservative — only correct where confident |
| Max phase correction | +/-120 deg | Prevents artifacts from over-correction |
| Schmitt rise_ratio | 2.5 | Onset sensitivity |
| Schmitt floor_decay | 80ms | Re-arm speed |
| Schmitt hold | 30ms | Double-trigger prevention |
| Peak-match max_dist | 20ms | Max distance for "same event" |
| Peak-match cluster window | 3ms | Onset jitter tolerance |

**TODO**: Validate thresholds across 20+ diverse sessions before shipping.
Some (especially CLUSTER_THRESHOLD and BRIDGE_THRESHOLD) may need
adaptive behavior based on the overall correlation landscape of the session.

---

## Compute Cost

For N tracks:

| Step | Cost |
|------|------|
| Broadband matrix | N*(N-1)/2 xcorr calls |
| Envelope matrix | N*(N-1)/2 xcorr calls (envelope precomputed per track) |
| Transient detection | N tracks (for windowed/peak-match validation) |
| Windowed XCorr matrix | N*(N-1)/2 pairs * P peaks per pair |
| Clustering | O(N^2) Union-Find |
| Star alignment | N-1 edges (one per non-root track) |
| Spectral correction | N-1 STFT analyses |

For 8 tracks: 28 pairs, ~10s in Python. Embarrassingly parallel.
For 24 tracks: 276 pairs, ~120s in Python. C++ will be 10-50x faster.

Note: Windowed XCorr and Peak-Match are currently computed as full N x N
for diagnostics. In production, only the edges used for alignment need
their detailed analysis. The full matrices remain available as diagnostics.

---

## What's Proven

- Star topology outperforms tree (no cascading) and merge (no dilution)
- Broadband correlation correctly identifies alignable pairs
- Envelope correlation catches shared-event pairs that broadband misses (Kick)
- Windowed XCorr avoids reflection artifacts (OH L/R case)
- Spectral phase correction produces audible improvement over time-only
- Adaptive-floor Schmitt trigger detects transients robustly in drums
- Peak-match confidence correctly classifies mic pair relationships
- Tested on: 7sg drums (8 mics), Lucifer (3 mics), guitar sessions

## What Needs Testing Before Ship

- [ ] Threshold robustness across 20+ diverse sessions
- [ ] Non-drum sources (guitar amps, piano, brass, strings, vocals)
- [ ] Pre-processed material (gated, compressed, EQ'd)
- [ ] Scale testing (16, 24, 32 tracks)
- [ ] Polarity detection edge cases and improvement
- [ ] A/B blind test vs Auto-Align 2 on matched material
- [ ] Adaptive thresholding (if fixed thresholds fail across material)
- [ ] Performance benchmarks (C++ implementation)

---

## Architecture Summary

```
                    +-----------------------+
                    |   Load N tracks       |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Broadband N x N     |  -> clustering + Tier 1 delay
                    |   Envelope N x N      |  -> Tier 2 bridging
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Connected Components |
                    |   (Union-Find)         |
                    +-----------+-----------+
                                |
                +---------------+---------------+
                |                               |
    +-----------v-----------+       +-----------v-----------+
    |   Tier 1 Clusters     |       |   Orphan Tracks       |
    |   (broadband >= 0.15) |       |   (broadband < 0.15)  |
    +-----------+-----------+       +-----------+-----------+
                |                               |
                |                   +-----------v-----------+
                |                   |   Envelope check      |
                |                   |   against cluster     |
                |                   |   roots               |
                |                   +-----------+-----------+
                |                               |
                |                   +-----------v-----------+
                |                   |  env >= 0.30?         |
                |                   |  YES: Tier 2 bridge   |
                |                   |  NO:  true orphan     |
                |                   +-----------+-----------+
                |                               |
    +-----------v-------------------------------v-----------+
    |                                                       |
    |   Star Alignment Within Each Cluster                  |
    |                                                       |
    |   Root: untouched                                     |
    |   Tier 1 children: broadband delay + spectral phase   |
    |   Tier 2 children: windowed delay + time only         |
    |                                                       |
    |   Orphan clusters: internal Tier 1, then Tier 2       |
    |   shift as a unit                                     |
    |                                                       |
    +-------------------------------------------------------+
                                |
                    +-----------v-----------+
                    |   Per-track sanity    |
                    |   check (RMS sum)     |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Output corrected    |
                    |   audio + diagnostics |
                    +-----------------------+
```
