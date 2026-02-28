# Spanning Tree Alignment — "Magic Align"

## Problem

Pairwise alignment requires choosing a reference track. Wrong choice = garbage corrections on low-correlation tracks (e.g. kick aligned to snare at 0.08 correlation). The user shouldn't have to think about this.

## Solution

Use the all-pairs correlation matrix to build a maximum spanning tree. The tree determines:
- **Which track is the reference** (root = highest row-sum)
- **Which track aligns to which** (tree edges = strongest pairwise connections)
- **Which tracks are left untouched** (orphans = no edges above threshold)

Every alignment uses the strongest available partner. No track ever gets corrected against a weak correlation.

## Algorithm

### Stage 1: Structure (time + polarity)

```
1. All-pairs cross-correlation → N×N matrix
   - correlation (absolute value)
   - delay (samples)
   - polarity (+1 or -1)
   Computed once on the analysis window (default 7.5s).

2. Maximum spanning tree
   - Sort all edges by correlation descending
   - Greedy: add edge if it connects two previously unconnected nodes (Kruskal's)
   - Drop edges below threshold (0.15) — tracks with no edges become orphans

3. Root selection
   - Sum each track's correlations (abs, above threshold) across its row
   - Highest sum = root (most connected track = best reference)

4. Tree walk (BFS, top-down from root)
   At each edge parent → child:
   a. Flip polarity if xcorr says inverted
   b. Apply time correction (delay from xcorr against corrected parent)
   c. Corrections accumulate: child inherits parent's cumulative offset

   Top-down because each child needs its parent corrected first.

5. Per-track sanity check
   - Compare RMS(parent + corrected child) vs RMS(parent + raw child)
   - If correction made pair sum worse → revert to raw (catches bad polarity, catastrophic failures)
   - Orphans pass through untouched
```

### Stage 2: Spectral phase (applied after all time corrections)

```
6. For each tree edge parent → child:
   - analyze_phase_spectral(corrected parent, time-corrected child)
   - apply_phase_spectral(child, correction)
   - Per-bin coherence weighting already gates which frequencies get corrected
   - Coherence threshold 0.4, max correction ±120°, gaussian smoothing
```

Stages are independent. Stage 1 alone beats manual reference selection. Stage 2 is the per-frequency polish.

## Why a tree?

The tree naturally handles the problems that broke other approaches:

**No reference selection needed.** The root falls out of the correlation matrix — track with the most shared signal across other tracks.

**Cascading through strong pairs.** Kick doesn't align directly to Snare Top (0.08 correlation). Instead: Snare Top → OH Left (0.40) → Kick In (0.25). Each link is the strongest available connection.

**Automatic grouping.** After dropping edges below threshold, disconnected sub-trees form natural groups. Each group gets its own root. Orphans pass through untouched. No explicit group detection needed.

**Overlapping groups handled.** If OH Left connects to both the snare group and the kick group, it's a bridge node in the tree. Kick aligns through it. No ambiguity about which group a track belongs to.

## Example: 7sg drums

Correlation matrix:
```
              SnTop   Kick   SnBot   OH_L   OH_R
  SnTop        ---    0.08   0.62    0.40   0.35
  Kick        0.08     ---   0.08    0.09   0.13
  SnBot       0.62    0.08    ---    0.32   0.35
  OH_L        0.40    0.09   0.32     ---   0.22
  OH_R        0.35    0.13   0.35    0.22    ---
```

Row sums (above 0.15):
```
  SnTop: 0.62 + 0.40 + 0.35 = 1.37  ← root
  Kick:  (none above 0.15)  = 0.00  ← orphan
  SnBot: 0.62 + 0.32 + 0.35 = 1.29
  OH_L:  0.40 + 0.32 + 0.22 = 0.94
  OH_R:  0.35 + 0.35 + 0.22 = 0.92
```

Maximum spanning tree (threshold 0.15):
```
  Snare Top (root, untouched)
    ├── Snare Bot (0.62) → align to Snare Top
    ├── OH Left (0.40)   → align to Snare Top
    └── OH Right (0.35)  → align to Snare Top

  Kick (orphan, untouched)
```

Result: Kick stays raw (no shared signal with anyone). Snare Bot, OH Left, OH Right all align to Snare Top through strong connections.

## Example: kick in/out + snare + overheads

```
  Snare Top (root)
    ├── Snare Bot (0.59)  → align to Snare Top
    ├── OH Left (0.40)    → align to Snare Top
    │     └── Kick In (0.25) → align to corrected OH Left
    │           └── Kick Out (0.75) → align to corrected Kick In
    └── OH Right (0.35)   → align to Snare Top
```

Kick In connects through OH Left (the bridge). Kick Out connects through Kick In (strongest partner). The chain: Snare Top → OH Left → Kick In → Kick Out. Each link is the best available pair.

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Edge threshold | 0.15 | Below this, no meaningful shared signal. Pair RMS check catches edge cases anyway |
| Tree algorithm | Kruskal's (max spanning) | Greedy, simple, gives strongest edges. ~15 lines of code |
| Root selection | Highest correlation row-sum | Most connected = most bleed from other sources = best reference |
| Walk order | BFS top-down | Parent must be corrected before child aligns to it |
| Sanity check | Pair RMS (parent + child) | Catches catastrophic failures. Full-mix RMS hides quiet tracks |
| Orphan handling | Pass through untouched | No edge above threshold = no shared signal = nothing to align |
| Phase correction | After all time corrections | Phase analysis needs time-aligned signals to be meaningful |

## Existing code to reuse

All from `python/align_files.py`:
- `detect_delay_xcorr(ref, tar, sr)` → (delay, correlation, polarity)
- `correct_delay_subsample(signal, delay, sr)` → corrected
- `analyze_phase_spectral(ref, tar, sr)` → (f, phase_correction, coherence)
- `apply_phase_spectral(target, phase_correction, sr)` → corrected

From `python/graph_align.py`:
- `connected_components(N, corr_matrix, threshold)` → list of sets

New code needed:
- `max_spanning_tree(corr_matrix, threshold)` → list of (parent, child, weight) edges
- Tree walk (BFS from root, ~20 lines)
- Pair RMS check (~5 lines per track)

## Compute cost

For N tracks:
- Correlation matrix: N*(N-1)/2 xcorr calls (done once)
- Tree walk: N-1 pairwise alignments (one per edge)
- Phase correction: N-1 spectral analyses

For 6-track drums: 15 xcorr + 5 alignments. ~5s in Python. Embarrassingly parallel for future threading (each tree branch is independent after the parent is done).
