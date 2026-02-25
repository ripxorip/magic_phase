# Magic Phase - User Experience Contract

> "The best interface is no interface." - Golden Krishna

This document is a binding agreement for how Magic Phase must behave. Every implementation
decision must serve this UX. If code conflicts with this doc, the code is wrong.

---

## The Promise

**3 clicks. 8 seconds. Mind blown.**

```
Click 1: Load plugin on reference track
Click 2: Load plugin on target track
Click 3: Click MAGIC ALIGN

...play audio for 7.5 seconds...

Done. Tracks are phase-aligned. User hears the magic.
```

---

## The Competition

| Plugin | Steps to align | Time | User must know |
|--------|---------------|------|----------------|
| Waves InPhase | 6+ | 30s+ | What phase is, how to read correlation meter |
| MAutoAlign | 4+ | 20s+ | Which track is reference, when to click |
| Sound Radix Auto-Align | 5+ | 25s+ | Capture mode, detector settings |
| **Magic Phase** | **3** | **8s** | **Nothing. Just click.** |

We win by being stupidly simple.

---

## Core Principles

### 1. Zero Configuration
- No settings dialogs
- No "detector mode" dropdowns
- No threshold adjustments
- Defaults that work for 95% of cases
- Power users can tweak, but never required

### 2. Progressive Disclosure
- Basic: One button does everything
- Advanced: Mode buttons (T+Φ, Φ, T) for fine-tuning
- Expert: REF button for multi-track scenarios

### 3. Always Provide Feedback
- User should never wonder "is it working?"
- Every state has clear visual indication
- Progress is always visible

### 4. Fail Gracefully
- Not enough audio? Tell them, don't fail silently
- No reference track? Guide them, don't show error
- Analysis failed? Explain why in human terms

---

## The User Journey

### First-Time User (Knows Nothing)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. User has kick drum recorded with 2 mics (common scenario)               │
│     - Track 1: Kick In (close mic)                                          │
│     - Track 2: Kick Out (resonant head)                                     │
│     - They sound weird together (phase issues)                              │
│                                                                             │
│  2. User heard about Magic Phase, drops it on both tracks                   │
│     - First instance auto-becomes reference (Track 1)                       │
│     - Second instance shows Track 1 in the list with "REF" badge            │
│                                                                             │
│  3. User sees big gold button "MAGIC ALIGN" on Track 2                      │
│     - They click it (obvious action)                                        │
│     - Button changes: "▶ PLAY TRACKS"                                       │
│     - User understands: "oh, I need to play"                                │
│                                                                             │
│  4. User hits spacebar (play in DAW)                                        │
│     - Button shows progress: "▶ 2.1s / 7.5s"                                │
│     - They see it counting up, understand it's working                      │
│                                                                             │
│  5. After 7.5 seconds                                                       │
│     - Button changes: "ANALYZING..."                                        │
│     - Brief moment (100ms)                                                  │
│     - Button changes: "✓ ALIGNED" (green)                                   │
│     - Results appear: "-1.0ms  φ -8°"                                       │
│                                                                             │
│  6. User hears the kick drum                                                │
│     - Low end is HUGE now                                                   │
│     - "Holy shit, that's what it's supposed to sound like"                  │
│     - User is now a customer for life                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Power User (Multi-Track)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Scenario: Full drum kit, 8 mics                                            │
│     - Track 1: Kick In (want this as master reference)                      │
│     - Track 2: Kick Out                                                     │
│     - Track 3: Snare Top                                                    │
│     - Track 4: Snare Bottom                                                 │
│     - Track 5-8: Toms, OH, Room                                             │
│                                                                             │
│  1. User loads Magic Phase on all 8 tracks                                  │
│     - Track 1 auto-becomes reference                                        │
│     - All other tracks see Track 1 as REF                                   │
│                                                                             │
│  2. User plays audio, clicks MAGIC ALIGN on tracks 2-8                      │
│     - Each track aligns to Track 1                                          │
│     - Could click them all quickly, they queue up                           │
│                                                                             │
│  3. User decides snare should align to snare top, not kick                  │
│     - Opens Track 3 plugin                                                  │
│     - Clicks REF button (explicit override)                                 │
│     - Track 3 now reference for snare mics                                  │
│     - Re-aligns Track 4 to Track 3                                          │
│                                                                             │
│  4. Result: Kick mics aligned together, snare mics aligned together         │
│     - Professional workflow, minimal clicks                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Button States

### MAGIC ALIGN Button

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │    MAGIC ALIGN      │  IDLE                                              │
│  └─────────────────────┘  - Gold background                                 │
│                           - Ready to click                                  │
│                           - This is the default state                       │
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │  ▶ PLAY TRACKS      │  WAITING FOR AUDIO                                 │
│  │     0.0s / 7.5s     │  - Pulsing/breathing animation                     │
│  └─────────────────────┘  - Progress text below                             │
│                           - Click again to cancel                           │
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │    ANALYZING...     │  ANALYZING                                         │
│  │        ◠           │  - Spinner animation                               │
│  └─────────────────────┘  - Brief state (~100ms)                            │
│                           - Cannot cancel                                   │
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │    ✓ ALIGNED        │  ALIGNED                                           │
│  │   -1.0ms   φ -8°    │  - Green background                                │
│  └─────────────────────┘  - Shows key results                               │
│                           - Click to re-align                               │
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │  ⚠ NO REFERENCE     │  ERROR: No reference                               │
│  │   Click REF on      │  - Orange/warning color                            │
│  │   another track     │  - Helpful guidance                                │
│  └─────────────────────┘  - Auto-dismisses when fixed                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### REF Button

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌───────┐                                                                  │
│  │  REF  │  OFF (not reference)                                             │
│  └───────┘  - Gray/subtle background                                        │
│             - Most users ignore this                                        │
│                                                                             │
│  ┌───────┐                                                                  │
│  │  REF  │  ON (this track is reference)                                    │
│  └───────┘  - Gold background                                               │
│             - Only one track can be REF at a time                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mode Buttons

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌───────┐ ┌───────┐ ┌───────┐                                              │
│  │  T+Φ  │ │   Φ   │ │   T   │                                              │
│  └───────┘ └───────┘ └───────┘                                              │
│                                                                             │
│  T+Φ : Time + Phase correction (DEFAULT - full magic)                       │
│  Φ   : Phase only (when time is already aligned)                            │
│  T   : Time only (classic delay compensation)                               │
│                                                                             │
│  - Only one can be active                                                   │
│  - Active = gold, inactive = gray                                           │
│  - For users who want to A/B the correction types                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### A/B Button

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌───────┐                                                                  │
│  │  A/B  │  Bypass toggle                                                   │
│  └───────┘  - OFF: Correction active (A)                                    │
│             - ON: Bypassed, hear original (B)                               │
│             - Essential for "before/after" comparison                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## State Machine

```
                                    ┌─────────────────────┐
                                    │                     │
                                    ▼                     │
┌────────┐  click ALIGN   ┌─────────────────┐            │
│  IDLE  │ ─────────────▶ │ WAITING_FOR_    │            │
│        │                │ AUDIO           │            │
└────────┘                └────────┬────────┘            │
    ▲                              │                     │
    │                              │ 7.5s of audio       │
    │ click (cancel)               │ received            │
    │                              ▼                     │
    │                     ┌─────────────────┐            │
    │                     │   ANALYZING     │            │
    │                     │  (background)   │            │
    │                     └────────┬────────┘            │
    │                              │                     │
    │                              │ analysis complete   │
    │                              ▼                     │
    │                     ┌─────────────────┐            │
    │                     │    ALIGNED      │ ───────────┘
    │                     │                 │  click (re-align)
    │                     └────────┬────────┘
    │                              │
    │                              │ error / ref removed
    └──────────────────────────────┘
```

---

## Auto-Reference Logic

```cpp
// When a new instance registers:

if (totalInstances == 1) {
    // First plugin loaded - auto-become reference
    setIsReference(true);
}
else if (totalInstances == 2 && noExplicitReferenceSet) {
    // Second plugin - first one is already reference
    // This instance is ready to align
}
else {
    // 3+ plugins - user should explicitly manage
    // But first one is still default reference
}
```

**Rules:**
1. First instance loaded → automatic reference
2. User can always override with explicit REF click
3. Only one reference at a time (clicking REF elsewhere clears previous)
4. Reference track doesn't need to align (it's the source of truth)

---

## Progress & Timing

### Accumulation Phase

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Minimum audio required: 7.5 seconds                                        │
│                                                                             │
│  Why 7.5s?                                                                  │
│  - ~350 STFT frames at 48kHz/1024 hop                                       │
│  - Enough for reliable cross-correlation                                    │
│  - Enough for coherent phase averaging                                      │
│  - Short enough that users don't get impatient                              │
│                                                                             │
│  Progress display:                                                          │
│  - Update every 0.1s (smooth counting)                                      │
│  - Format: "X.Xs / 7.5s"                                                    │
│  - Could add progress bar in future                                         │
│                                                                             │
│  Edge cases:                                                                │
│  - User pauses playback: Progress pauses, resumes when play resumes         │
│  - User stops playback: Progress stays, continues on next play              │
│  - User seeks: We detect discontinuity, may reset (TBD)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Analysis Phase

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Analysis runs in background thread                                         │
│                                                                             │
│  Duration: ~50-150ms depending on CPU                                       │
│  - Fast enough to feel instant                                              │
│  - User sees "ANALYZING..." briefly                                         │
│                                                                             │
│  During analysis:                                                           │
│  - Audio continues playing (no glitch)                                      │
│  - Old correction still active (if re-aligning)                             │
│  - GUI shows spinner                                                        │
│                                                                             │
│  On completion:                                                             │
│  - Results transferred to processor (thread-safe)                           │
│  - Correction activates immediately                                         │
│  - GUI updates to ALIGNED state                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

### User-Friendly Error Messages

| Internal State | User Sees | Guidance |
|---------------|-----------|----------|
| `refSlot == -1` | "No reference track" | "Load Magic Phase on another track first" |
| `refSlot == mySlot` | "This is the reference" | "Click ALIGN on a different track" |
| `frames < minimum` | "Need more audio" | "Keep playing..." (auto-resolves) |
| `correlation < 0.3` | "Tracks may be unrelated" | "Check that both tracks have the same source" |
| `analysis exception` | "Analysis failed" | "Try again, or contact support" |

### Never Show:
- Stack traces
- Error codes
- Technical jargon
- Blame ("you did something wrong")

---

## Results Display

### After Successful Alignment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Primary info (always visible):                                             │
│  ┌─────────────────────────────────────────────┐                            │
│  │  ✓ ALIGNED                                  │                            │
│  │  -1.0ms   φ -8°                             │                            │
│  └─────────────────────────────────────────────┘                            │
│                                                                             │
│  -1.0ms = time offset applied                                               │
│  φ -8°  = average phase rotation                                            │
│                                                                             │
│  Track row shows more detail:                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ ● Kick Out      OFFSET -1.0ms   PHASE -8°    0.89                   │    │
│  │   [spectral bands visualization]                           [T] [Φ] │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  0.89 = correlation (color coded: green >0.8, yellow >0.6, red <0.6)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Color Coding

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Correlation | > 0.8 | 0.5 - 0.8 | < 0.5 |
| Coherence | > 0.7 | 0.4 - 0.7 | < 0.4 |

---

## Sound Expectations

### What Users Should Hear

| Before | After |
|--------|-------|
| Thin, phasey low end | Full, coherent low end |
| Comb filtering artifacts | Clean combined signal |
| "Something's wrong" feeling | "That's a drum!" feeling |

### The "Holy Shit" Moment

The goal is that moment when the user:
1. Clicks A/B to compare
2. Hears the "before" (thin, weird)
3. Hears the "after" (full, powerful)
4. Says "holy shit"
5. Tells 10 friends about Magic Phase

**This moment is why we exist. Every UX decision serves this moment.**

---

## Non-Goals (Things We Don't Do)

1. **Manual delay entry** - User shouldn't need to type numbers
2. **Correlation meters** - User shouldn't need to interpret graphs
3. **Multiple algorithm choices** - One algorithm that works
4. **Preset systems** - Nothing to save, nothing to recall
5. **Sample-accurate display** - "-1.0ms" is enough, not "-1.0208ms"
6. **Phase rotation graphs** - Cool but unnecessary for UX
7. **Undo/redo** - Just click align again

---

## Implementation Checklist

Before shipping, every item must be TRUE:

### Core Flow
- [ ] First plugin loaded auto-becomes reference
- [ ] MAGIC ALIGN click starts waiting state
- [ ] Progress shows "X.Xs / 7.5s" during accumulation
- [ ] Analysis auto-triggers at 7.5s
- [ ] Analysis runs in background (no audio glitch)
- [ ] Results apply immediately on completion
- [ ] ALIGNED state shows results

### Edge Cases
- [ ] No reference → helpful message, not error
- [ ] Click on reference track → "This is the reference" message
- [ ] Cancel during waiting → returns to IDLE
- [ ] Re-align after aligned → starts fresh
- [ ] Playback pause → progress pauses
- [ ] Low correlation → warning but still works

### Polish
- [ ] Button animations are smooth
- [ ] Progress updates feel responsive (10Hz minimum)
- [ ] State transitions have no flicker
- [ ] A/B bypass is instantaneous
- [ ] Mode switching is instantaneous

---

## Metrics of Success

1. **Time to first alignment: < 15 seconds**
   - Load plugin (2s) + Load plugin (2s) + Click (1s) + Play (8s) = 13s

2. **Clicks to align: 3**
   - Load, Load, Align

3. **Things user must learn: 0**
   - No manual, no tutorial, no "how do I..."

4. **Support tickets about "how to use": 0**
   - If users ask how to use it, we failed

---

## The Golden Rule

> When in doubt, remove features, not add them.
>
> Every button, every option, every setting is a chance to confuse.
> The plugin that does one thing perfectly beats the plugin that does ten things adequately.

---

*This document is the source of truth. Code implements this spec, not the other way around.*

*Last updated: 2024 - The "blow minds" edition*
