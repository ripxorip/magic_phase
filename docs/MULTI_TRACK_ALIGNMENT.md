# Multi-Track Alignment Strategy

## The Problem

When aligning multiple drum mics to a single reference, correlation is low because different mics capture fundamentally different content:

- Kick close mic vs overhead = very different waveforms
- Snare bottom vs room mic = almost no correlation

Current approach: Align everything to one global reference. Works great for guitar (multiple mics on same source), struggles with drums (multiple sources, multiple mics).

## Sound Radix Auto-Align 2 (Competition)

- Industry standard for drum alignment
- Has "spectral phase" mode
- Handles multi-track
- $149 USD

## Ideas to Beat Them

### 1. Smart Reference Auto-Selection

Instead of user picking reference, analyze all tracks and pick the one with highest average correlation to all others.

```
For each track T:
    avg_corr[T] = mean(correlation(T, all_other_tracks))

reference = track with max(avg_corr)
```

Usually this will be overheads or room mic since they capture the whole kit.

### 2. Hierarchical / Multi-Reference Alignment

Allow tracks to align to different references based on what makes sense sonically:

```
Overheads (master reference)
├── Kick OUT → align to overheads
│   └── Kick IN/Sub → align to Kick OUT
├── Snare TOP → align to overheads
│   └── Snare BOTTOM → align to Snare TOP
├── Hi-Hat → align to overheads
├── Tom 1 → align to overheads
├── Tom 2 → align to overheads
└── Room → align to overheads
```

**JSON format:**
```json
{
  "tracks": [
    { "file": "overhead_L.wav", "role": "reference" },
    { "file": "kick_out.wav", "role": "target", "mode": "phi" },
    { "file": "kick_in.wav", "role": "target", "mode": "phi", "align_to": "kick_out.wav" },
    { "file": "snare_top.wav", "role": "target", "mode": "phi" },
    { "file": "snare_btm.wav", "role": "target", "mode": "phi", "align_to": "snare_top.wav" }
  ]
}
```

### 3. Automatic Source Grouping

Detect which mics are capturing similar sources by analyzing:

1. **Transient correlation** - Mics on same source have correlated transients
2. **Spectral similarity** - Kick mics have similar low-end content
3. **Delay clustering** - Mics on same source have similar delays to room/overheads

Algorithm:
```
1. Compute correlation matrix between all tracks
2. Cluster tracks by correlation (hierarchical clustering)
3. Within each cluster, pick local reference (highest avg correlation)
4. Align cluster members to local reference
5. Align local references to global reference (overheads)
```

### 4. Frequency-Band-Aware Alignment

Don't try to align frequencies where a mic has no content:

- Kick mic: Only align < 200 Hz (where it has real content)
- Hi-hat mic: Only align > 3 kHz
- Overheads: Full spectrum

**Implementation:**
```
For each frequency bin:
    if coherence[bin] < threshold:
        phase_correction[bin] = 0  // Don't correct
    else:
        phase_correction[bin] = measured_correction * confidence
```

We already do this with coherence weighting, but could be more aggressive.

### 5. Transient-Focused Alignment

Drums are transient-heavy. Instead of full-signal correlation:

1. Detect transients in reference
2. Window around transients (±10ms)
3. Compute correlation only in transient windows
4. Weight alignment by transient energy

This would give better results for drums where the "meat" is in the attacks.

### 6. Per-Instrument Phase Profiles

Store learned phase relationships for common setups:

- "Kick IN vs Kick OUT typically needs ~2ms delay + polarity flip"
- "Snare bottom typically needs polarity flip vs snare top"

Could be user-saveable presets or ML-learned defaults.

## Implementation Priority

1. **Per-track `align_to` field** - Easy JSON change, big win
2. **Smart reference auto-selection** - Analyze correlation matrix, pick best
3. **Frequency-band awareness** - Improve existing coherence weighting
4. **Automatic grouping** - Clustering algorithm, more complex
5. **Transient-focused** - Requires transient detection, medium complexity

## Success Metrics

- Drums sum with more punch (subjective A/B test)
- Higher correlation in frequency bands that matter
- Fewer artifacts from over-correction
- Works out-of-the-box without manual reference selection

## The Killer Feature

**One-click drum alignment that just works.**

User drops 16 drum tracks, clicks "Align", Magic Phase:
1. Auto-detects overheads as reference
2. Groups kick mics, snare mics, tom mics
3. Aligns within groups, then to master
4. Only corrects frequencies where each mic has real content

No manual setup. No wrong reference selection. Just tight drums.
