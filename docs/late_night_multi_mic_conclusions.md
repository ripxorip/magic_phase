# Late Night Multi-Mic Alignment Conclusions

*2026-02-28 - After a full day of algorithm experimentation*

## The Physics

Cross-correlation finds **"the same sound arriving at different times"**.

- **High correlation** = same source, different mics, different arrival times
- **Low correlation** = different sources, no shared content

You **cannot align what doesn't correlate**. Forcing alignment on low-correlation pairs means fitting noise - the "correction" is meaningless.

## Why Kick vs Snare Doesn't Correlate

1. **Intermittent sources** - Kick hits, then silence. Snare hits at different times.
2. **Different content** - Cross-correlation needs "same sound, different arrival"
3. **Minimal bleed** - Directional mics, distance = kick barely registers in snare mic

Low correlation = low potential gain. Even with perfect alignment, you can't sum what isn't there.

## Algorithms Tested

| Algorithm | Lucifer | 7sg | Verdict |
|-----------|---------|-----|---------|
| Star | +3.7 dB | +4.2 dB | Simple, clean, skips orphans |
| Tree | +3.7 dB | +4.2 dB | Complex, cascading errors possible |
| Merge | +3.4 dB | +3.8 dB | Dilutes reference signal |
| Cluster | +3.8 dB | +4.2 dB | Best potential, needs refinement |

## The Winning Approach: Star Within Clusters

```
1. Build correlation graph (edges where corr ≥ threshold)
2. Find connected components (natural clusters)
3. Within each cluster: star alignment to local root
4. Don't bridge between clusters - leave them independent
```

### Example

```
Cluster 1: Snare Top (root), Snare Bot, OHs, Toms
           → all correlate, all align to Snare Top

Cluster 2: Kick In, Kick Out
           → correlate with each other, not with snare
           → align Kick Out to Kick In

Cluster 3: Room (isolated)
           → no edges above threshold
           → untouched
```

## Why NOT Tree/Merge

**Tree**: Routes through strongest paths, but cascades errors. If A→B has small error, it propagates to B→C→D. More complex for minimal gain.

**Merge**: Sums and averages signals after each pair. Dilutes the reference over time. Aligning to an "average" is suboptimal vs aligning to a pristine single mic.

## Why NOT Forced Inter-Cluster Bridging

We tried connecting isolated tracks via weak edges (e.g., Kick↔OH at 0.13). This violates the physics:

- 0.13 correlation = barely above noise floor
- The "alignment" found is random, not physical
- No constructive interference to gain anyway

**Principle**: Only align what actually correlates.

## Pre-Processed Material

Gating/compression breaks correlation:

| Processing | Effect |
|------------|--------|
| **Gating** | WORST - binary on/off destroys envelope matching |
| **Heavy compression** | Changes dynamics, messes with xcorr peaks |
| **Gentle outboard** | Usually OK - 1176/LA-2A preserve envelope shape |
| **Linear EQ** | OK - changes amplitude, not phase relationships |
| **Min-phase EQ** | Bad - changes both amplitude AND phase |

The TBM drum kit showed classic signs of pre-processing:
- Low coherence (0.2 range vs 0.7+ for raw recordings)
- Weird stereo overhead behavior (different polarities, 12ms difference)
- Low correlations across the board

**Raw recordings are required for accurate alignment.**

## Final Algorithm: `--cluster` (without inter-cluster bridging)

```python
1. Compute N×N correlation matrix
2. Find connected components using Union-Find (edges ≥ threshold)
3. For each component:
   - If single track: leave untouched
   - If multiple tracks:
     - Pick local root (highest row sum within cluster)
     - Star align all others to local root
4. Output: each cluster aligned internally, clusters independent
```

This respects the physics. Tracks only align when they share content. Natural groupings emerge from the correlation structure. No forced connections.

## The Solid Principle

```
correlation ≥ threshold → shared physics → align
correlation < threshold → different sources → leave alone
```

Simple. Principled. Works.
