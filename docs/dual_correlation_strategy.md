# Dual Correlation Strategy

*2026-03-01 - After multi-lens diagnostic analysis*

## The Insight

We compute the correlation matrix **two ways** and take the best per pair:

1. **Broadband XCorr** — raw waveform cross-correlation (what we have today)
2. **RMS Envelope XCorr** — short-window RMS (3ms) → cross-correlate the energy envelopes

## Why Two Methods

Broadband xcorr compares the raw waveform — every sample, every frequency. This is great when the signals are similar (same mic type, close placement), but struggles when waveforms differ significantly despite sharing the same events.

RMS Envelope strips the waveform detail and compares just the **energy shape over time**. A kick hit creates an energy bump on every mic in the room — different waveform shapes, but the same bump. The envelope captures that shared structure.

## What We Measured (7sg drums, 8 mics)

| Pair | Broadband Corr | RMS Envelope Corr |
|------|---------------|-------------------|
| Snare Top vs OH Left | 0.39 | **0.74** |
| Snare Top vs OH Right | 0.36 | **0.78** |
| Snare Top vs Tom Floor L | 0.54 | **0.84** |
| OH Left vs OH Right | 0.21 | **0.95** |
| OH Left vs Tom Floor L | 0.36 | **0.91** |
| OH Right vs Tom Floor L | 0.34 | **0.94** |
| Tom Floor L vs C | 0.56 | **0.97** |
| Kick vs OH Left | 0.09 | 0.34 |
| Kick vs OH Right | 0.13 | 0.39 |

RMS Envelope gives dramatically higher correlations for pairs that share content. Kick pairs stay low on both — correctly, because the physics says they don't share enough.

## Delay Estimates

Both methods agree on delays when the signal is good:

| Pair | Broadband Delay | RMS Envelope Delay |
|------|----------------|-------------------|
| Snare Top vs OH Left | +3.04ms | +3.44ms |
| Snare Top vs OH Right | +2.85ms | +3.46ms |
| Snare Top vs Tom Floor L | +0.92ms | +0.94ms |
| OH Left vs OH Right | -7.52ms | +0.35ms |

Note OH Left vs OH Right: broadband says -7.52ms (wrong — overheads are roughly equidistant), RMS Envelope says +0.35ms (correct). The waveforms decorrelate due to stereo content differences, but the energy envelope is nearly identical.

**Use broadband xcorr for the delay estimate. Use the better of the two correlations for the cluster graph.**

## Key Refinement: HP Before Envelope

Don't do RMS on the whole band. Low-frequency content (sub, reverb tails) smears the timing.

**The right envelope for alignment:**

1. **High-pass first** — ~900 Hz (range 600-1200Hz to taste)
   - This ensures the envelope tracks **attacks**, not sub energy
   - Our Low-Freq Envelope lens proved this: delay estimates were garbage because the slow sub content dominates
   - HP at 2kHz (our Transient Envelope lens) was probably too aggressive — 900Hz is the sweet spot
2. **Rectify** — `abs(x_hp)` or `x_hp²`
3. **Low-pass smooth** — ~50 Hz (range 30-80Hz, i.e. 3-10ms time constant)
   - Or equivalently: short sliding RMS window (5-15ms) after HP

This gives an "attack-tracking" envelope that works for any source material.

## Key Refinement: Windowing Around Events

**This is potentially the biggest win.** Instead of correlating the entire analysis window (7.5 seconds), find where the interesting stuff happens and correlate only those moments.

### The Problem With Full-Window Correlation

Correlating 7.5 seconds of audio at 48kHz = correlating 360,000 samples. Most of that is silence, room noise, cymbal wash, or sustained content that contributes ambiguity, not information. The result: a correlation function with a forest of similar-height peaks (low sharpness), especially for periodic material like drums.

#### "But silent parts don't accumulate in the correlation anyway?"

Technically correct — silence × silence ≈ 0, so quiet sections don't add false signal to the correlation sum. But the problem was never silence:

1. **Denominator dilution.** Normalized correlation divides by total energy. 7.5 seconds of cymbal wash and room noise inflates the denominator, making the transient-vs-transient peak *shorter* (weaker), not wrong. The signal-to-noise of the peak drops.

2. **Periodic content creates competing peaks.** This is the real killer. Kick at t=0, t=0.5s, t=1.0s — the correlation function gets a peak at every beat interval. The "true" alignment peak isn't corrupted by silence, it's *competing with other legitimate correlation peaks from other hits*. That's why sharpness tanks — there are multiple correct-looking peaks.

3. **Mixed sources at different physical delays.** Snare arrives at +3ms on the overhead, kick arrives at +8ms on the same overhead. Full-window correlation sees BOTH superimposed and produces a smeared compromise. Short windows around individual events give you +3ms for the snare windows and +8ms for the kick windows — clean separation, then the median picks the dominant source.

So silence doesn't hurt. But ambiguity from multiple events and denominator dilution from sustained content do. Event windowing solves both.

### The Solution: Event-Windowed Correlation

```
1. Compute the (HP'd) envelope for both tracks
2. Find the top N peaks in the envelope (e.g. 20-100, spaced out)
   - These are the "events" — hits, transients, note attacks
3. For each event peak:
   - Extract a short window (~80-200ms) around it from BOTH tracks
   - Cross-correlate that window pair → get a per-event delay estimate
4. Aggregate: take the MEDIAN of all per-event delay estimates
5. Confidence: how tightly do the per-event estimates cluster?
   - Low std = high confidence, single dominant source
   - High std = mixed sources or ambiguous content
```

### Why This Is Huge

- **Sharpness goes up**: Short windows have one dominant peak, not a forest
- **Outlier rejection is built in**: Median naturally rejects the few events that don't match
- **It's onset matching done right**: Without being drum-specific — any "event" works (guitar strum, vocal consonant, piano attack, orchestral accent)
- **Confidence becomes meaningful**: Std of per-event delays is a real physical quantity — "how consistently does this delay appear across events?"
- **Works with broadband AND envelope**: Apply this windowing to both correlation methods for even better results

### Example: Kick vs Overhead

Full 7.5s correlation: forest of peaks, confidence ≈ 0, delay estimate is unreliable.

Event-windowed: find the 8 kick hits (they're the biggest envelope peaks on both tracks), correlate 100ms around each one, get 8 delay estimates that all say ~+8ms. Median = +8ms, std = 0.3ms. High confidence, correct answer.

## Sharpness: The Ambiguity Detector

We already compute `confidence = 1 - second_peak / main_peak`. This IS sharpness.

Our lens data proved its importance:
- **Broadband xcorr**: Often decent correlation (0.39) but near-zero sharpness — many similar peaks
- **Onset Matching**: Moderate correlation but sharpness 0.75-0.88 — one clear winner
- **RMS Envelope**: High correlation (0.95) but zero sharpness — the envelope is smooth, so the correlation peak is broad

**Sharpness matters more than correlation magnitude.** A sharp peak at 0.3 correlation is more trustworthy than a broad plateau at 0.8.

## Fusion Logic: Trust Hierarchy

Not "pick one", not "average both". A tiered decision:

### 1. Broadband confident → use it (precision wins)

When broadband sharpness is high, its delay estimate is the most precise.

```
if sharp_bb > S1 and corr_bb > P1 → use τ_broadband
```

Ballpark: S1 ≈ 1.2-1.6, P1 ≈ 0.2-0.4

### 2. Broadband ambiguous, envelope confident → envelope arbitrates

When broadband has multiple plausible peaks but envelope sees one clear answer:

```
if sharp_env > S2 and corr_env > P2 → use τ_envelope
```

### 3. Both okay but disagree → weighted blend

```
if |τ_bb - τ_env| < 1.0ms → weighted average by (corr × sharpness)
if |τ_bb - τ_env| > 1.0ms → trust higher (corr × sharpness) product
```

### 4. Both low confidence → don't force it

Return "no strong evidence". The cluster solver ignores/downweights this pair. This is key to avoiding confidently-wrong alignment.

From the strategy doc: *"correlation < threshold → different sources → leave alone"*

## Algorithm (Revised)

```
For each pair (i, j):
  1. Compute HP'd envelope for both tracks
  2. Find top N event peaks (shared between tracks)
  3. For each event:
     a. Broadband xcorr on short window → τ_bb, corr_bb, sharp_bb
     b. Envelope xcorr on short window → τ_env, corr_env, sharp_env
  4. Aggregate per-event estimates → median τ, std, confidence
  5. Apply fusion logic → final delay + confidence for this pair

Build cluster graph from final confidences
Star align within clusters
```

## Why This Works For All Music

- **Drums**: Envelope peaks = hits. Short-window xcorr nails the timing.
- **Guitars**: Envelope peaks = strums/picks. Same principle.
- **Vocals**: Envelope peaks = consonants, breaths, phrase starts.
- **Orchestra**: Envelope peaks = accents, section entries, bowing changes.
- **Anything with dynamics**: Events exist in all music. If there are no events, there's nothing to align anyway.

## What We Killed

- **Spectral Flux**: Coarse time resolution, misleading correlations, adds nothing
- **Onset Matching (pulse train)**: Drum-specific, fails on sustained sources
- **Band-Limited / Low-Freq Envelope**: Too narrow, inconsistent
- **Low-Band Onset**: Numerically fragile, drum-specific
- **Full-window correlation as the only method**: Still useful, but event-windowed is the upgrade

## Next Steps

1. Implement HP'd envelope (900Hz HP → rectify → 50Hz LP)
2. Implement event-windowed correlation (both broadband + envelope paths)
3. Implement sharpness-based fusion logic
4. Test on 7sg drums + other material (guitar, vocals, mixed sources)
5. Wire into `graph_align.py` cluster pipeline
