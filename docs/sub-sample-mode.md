# Sub-Sample Correction Mode

## Overview

Add sub-sample time correction and a user toggle that controls whether the fractional
sample component is applied. This gives audio purists a bit-perfect integer-only signal
path while allowing pragmatists to get sub-sample precision (~0.1 sample accuracy).

## Motivation

The current `detectDelayTimeDomain` cross-correlation finds the peak at an integer
sample position (e.g., 49 samples). The true delay between two microphones is rarely
an exact integer number of samples. The fractional part (e.g., 0.3 samples) represents
a timing error of up to `1 / (2 * sampleRate)` seconds. At 48kHz that's up to ~10.4
microseconds, which matters for:

- Phase accuracy below ~5kHz where wavelengths are long
- Summing coherence on close-mic'd sources (snare top/bottom, inside/outside kick, etc.)
- Precision-critical mastering workflows

## Parabolic Interpolation

After `detectDelayTimeDomain` finds the integer peak at index `p` with correlation
values at `[p-1, p, p+1]` = `[c_minus, c_peak, c_plus]`:

```
fractional_offset = 0.5 * (c_minus - c_plus) / (c_minus - 2 * c_peak + c_plus)
sub_sample_delay = p + fractional_offset
```

This is standard parabolic (quadratic) interpolation around the correlation peak.
Accuracy is typically 0.05-0.1 samples for well-correlated signals.

### Edge cases

- If `c_minus - 2 * c_peak + c_plus == 0` (flat peak): fractional_offset = 0, use integer
- If `|fractional_offset| > 0.5`: clamp to [-0.5, 0.5] (parabola fit is only valid near peak)
- If peak is at the boundary of the search range: skip sub-sample refinement, use integer

## Correction Matrix

The sub-sample toggle is independent of T/Phi mode, creating four combinations:

```
                    Sub-sample OFF              Sub-sample ON
                    ~~~~~~~~~~~~~~              ~~~~~~~~~~~~~
T mode              Integer sample delay        Integer delay +
                    via delay buffer            fractional STFT phase ramp
                    (bypass STFT entirely)      (through STFT)

Phi mode            Integer sample delay +      Integer delay +
                    residual spectral phase     fractional phase ramp +
                    (through STFT)              residual spectral phase
                                                (through STFT)
```

### T mode, Sub-sample OFF (purist path)

- Signal does NOT pass through the STFT overlap-add at all
- A simple delay buffer shifts the audio by exactly N integer samples
- Bit-perfect: output[n] = input[n - N] (with polarity flip if detected)
- No windowing, no spectral processing, no overlap-add artifacts
- This is the "I don't trust your DSP" mode for skeptical mix engineers

### T mode, Sub-sample ON

- Integer shift via delay buffer (same as above)
- Fractional component applied as STFT phase ramp: `exp(-j * 2pi * k * frac / N)` per bin
- Signal passes through STFT overlap-add for the fractional correction only
- Alternatively: use sinc interpolation in time domain (avoids STFT, preserves purist path)
  - 16-point windowed sinc is standard, latency-free variant
  - Trade-off: sinc is slightly less accurate than STFT phase ramp but avoids overlap-add

### Phi mode, Sub-sample OFF

- Integer sample shift via delay buffer
- Residual spectral phase correction through STFT (per-bin rotation)
- Current behavior, just with a proper delay buffer for the integer part

### Phi mode, Sub-sample ON

- Integer sample shift via delay buffer
- Fractional delay + residual spectral phase combined in one STFT pass
- The fractional phase ramp is added to the per-bin spectral correction
- Most accurate mode: sub-sample time alignment + full spectral phase correction

## Implementation Plan

### Phase 1: Sub-sample delay detection

**File: `src/DSP/PhaseAnalyzer.h`**

Add member:
```cpp
float delaySubSample = 0.0f;  // Full sub-sample delay (e.g., 49.3)
```

Add getter:
```cpp
float getDelaySubSample() const { return delaySubSample; }
```

**File: `src/DSP/PhaseAnalyzer.cpp` — `detectDelayTimeDomain()`**

After finding the integer peak position and correlation values, add:

```cpp
// Parabolic interpolation for sub-sample precision
float c_minus = correlationAtPeakMinus1;
float c_peak  = correlationAtPeak;
float c_plus  = correlationAtPeakPlus1;

float denom = c_minus - 2.0f * c_peak + c_plus;
float fractional = 0.0f;
if (std::abs(denom) > 1e-10f)
{
    fractional = 0.5f * (c_minus - c_plus) / denom;
    fractional = std::clamp(fractional, -0.5f, 0.5f);
}

delaySubSample = static_cast<float>(peakIndex) + fractional;
```

Note: the existing `delaySamples` (integer) is preserved for backward compatibility.
`delaySubSample` is the new high-precision value.

### Phase 2: Pure integer delay buffer

**New file: `src/DSP/DelayBuffer.h`**

```cpp
class DelayBuffer
{
public:
    void prepare(int maxDelaySamples);
    void setDelay(int samples);
    void processBlock(float* data, int numSamples);
    // Processes in-place: output[n] = input[n - delay]

private:
    std::vector<float> buffer;
    int writePos = 0;
    int delaySamples = 0;
};
```

This is a simple circular buffer delay line. No interpolation, no spectral processing.
The output is a bit-perfect copy of the input shifted by N samples.

### Phase 3: Correction mode refactor in PluginProcessor

**File: `src/PluginProcessor.h`**

Add:
```cpp
std::atomic<bool> subSampleOn { false };
DelayBuffer delayBuffer;
float pendingDelaySubSample = 0.0f;  // From analysis (fractional)
int pendingDelayInteger = 0;          // From analysis (integer part)
```

**File: `src/PluginProcessor.cpp` — processBlock ALIGNED state**

```cpp
int mode = correctionMode.load();
bool subSample = subSampleOn.load();

if (mode == 0 && !subSample)
{
    // T mode, sub-sample OFF: pure delay buffer, bypass STFT
    delayBuffer.processBlock(channelData, numSamples);
    if (polarityInvert)
        for (int i = 0; i < numSamples; ++i)
            channelData[i] = -channelData[i];
    // Still need STFT for passthrough (maintain latency consistency)
    stftProcessor.processBlock(channelData, numSamples, nullptr);
}
else
{
    // All other modes go through STFT
    float timeDelay = subSample ? pendingDelaySubSample : (float)pendingDelayInteger;
    phaseCorrector.setDelaySamples(timeDelay);

    stftProcessor.processBlock(channelData, numSamples,
        [this, mode, subSample](std::complex<float>* frame, int numBins)
    {
        phaseCorrector.applyTimeCorrection(frame, numBins);  // Always (int or sub-sample)
        if (mode == 1)
            phaseCorrector.applyPhaseCorrection(frame, numBins);
    });
}
```

**Important latency consideration**: In T-purist mode, the delay buffer adds N samples of
latency. The STFT adds kFFTSize (4096) samples. For consistent PDC reporting, both paths
must report the same latency. Options:
1. Always report kFFTSize latency, add compensating delay in the purist path
2. Report different latency per mode (causes DAW PDC recalculation on mode switch — bad UX)

Option 1 is correct: purist path uses delay buffer for the correction delay PLUS the STFT
runs in passthrough mode to maintain the same total latency. This way the DAW's PDC doesn't
change when switching modes.

Wait — this means the signal still goes through STFT even in purist mode (for latency
matching). But we want bit-perfect... Revised approach:

- Purist path: delay buffer (N samples correction) + additional delay buffer
  (kFFTSize - N samples padding) = total kFFTSize samples latency
- No STFT processing at all
- Bit-perfect maintained, latency consistent

Actually even simpler: just run the STFT with a null callback (no frequency-domain
modification). The STFT overlap-add with Hann window and COLA should reconstruct the
signal perfectly (within floating-point precision). For true purists who want zero
STFT artifacts, the delay-buffer-only path with padding is needed. But in practice,
STFT with null callback introduces ~-140dB artifacts (floating point noise floor),
which is inaudible. Document this trade-off and let the user decide.

### Phase 4: GUI toggle

**File: `src/GUI/MainComponent.h`**

Add:
```cpp
juce::TextButton subSampleButton { "SS" };  // Sub-Sample toggle
```

**File: `src/GUI/MainComponent.cpp`**

- Add as a toggle button next to T and Phi
- Gold when ON, surface when OFF
- `onClick`: `processor.setSubSample(subSampleButton.getToggleState())`

**Bottom bar layout** (left to right):
```
[REF]  [MAGIC ALIGN / status]  ...  [T] [Phi] [SS] [A/B]
```

SS button is small (same size as T and Phi), positioned between Phi and A/B.

### Phase 5: Analysis pipeline update

**File: `src/PluginProcessor.cpp` — `runAnalysisInBackground()`**

After `detectDelayTimeDomain`:
```cpp
float rawDelaySubSample = rawDelayAnalyzer.getDelaySubSample();
float trueDelaySubSample = rawDelaySubSample - static_cast<float>(syncOffset);

// Integer part for delay buffer
int trueDelayInteger = static_cast<int>(std::round(trueDelaySubSample));

// Store both
pendingDelayInteger = trueDelayInteger;
pendingDelaySubSample = trueDelaySubSample;
```

For the time-alignment before spectral analysis (stage 2b), continue using integer shift.
The sub-sample fraction is tiny (~0.3 samples) and doesn't meaningfully affect spectral
phase estimation.

### Phase 6: Display

The GUI align button currently shows delay in ms. With sub-sample precision:
- Before: `+1.0ms` (integer 49 samples at 48kHz)
- After:  `+1.03ms` (sub-sample 49.3 samples at 48kHz)

Show one more decimal place when sub-sample is active.

## Testing

### Unit test for parabolic interpolation

```cpp
// Known delay of 49.3 samples: generate ref and shifted target
// Verify detectDelayTimeDomain returns integer 49
// Verify getDelaySubSample returns ~49.3 (within 0.15 tolerance)
```

### MagicPhaseTest integration

Add `--sub-sample` flag to MagicPhaseTest that prints the sub-sample delay:
```
Delay: 49 samples (49.31 sub-sample, 1.027 ms)
```

### A/B listening test

1. Load two identical tracks, offset one by 0.5 samples (using a DAW's sample-accurate nudge)
2. Run alignment with sub-sample ON and OFF
3. Sub-sample ON should null almost completely when summed (deep cancellation)
4. Sub-sample OFF leaves a 0.5-sample residual (audible as slight HF loss when summed)

### Purist path validation

1. Load a test signal (impulse, sine sweep)
2. Run alignment in T mode, sub-sample OFF
3. Compare output to input shifted by N samples
4. Should be bit-identical (within float precision: < -140dB difference)

## Alternative: Sinc interpolation instead of STFT for fractional delay

For T mode with sub-sample ON, an alternative to STFT phase ramp is windowed sinc
interpolation in the time domain:

```cpp
// 16-point windowed sinc interpolation
float fractional = delay - std::floor(delay);
for (int i = 0; i < numSamples; ++i)
{
    float sum = 0.0f;
    for (int tap = -7; tap <= 8; ++tap)
    {
        int idx = i - integerDelay + tap;
        if (idx >= 0 && idx < numSamples)
        {
            float x = static_cast<float>(tap) - fractional;
            float sinc = (std::abs(x) < 1e-6f) ? 1.0f : std::sin(M_PI * x) / (M_PI * x);
            float window = 0.5f + 0.5f * std::cos(M_PI * x / 8.0f);  // Hann window
            sum += input[idx] * sinc * window;
        }
    }
    output[i] = sum;
}
```

Pros:
- No STFT required for T+sub-sample mode (stays closer to purist path)
- Lower latency than STFT
- Well-understood, standard technique

Cons:
- Slightly less accurate than STFT phase ramp for very small fractional delays
- CPU cost per sample is higher (16 multiplies per sample vs one FFT per hop)
- Introduces its own windowing artifacts (though extremely small with 16+ taps)

Recommendation: Start with STFT phase ramp (already implemented in PhaseCorrector).
Consider sinc as a future option if users request a non-STFT sub-sample path.

## Files to modify (summary)

| File | Change |
|------|--------|
| `src/DSP/PhaseAnalyzer.h` | Add `delaySubSample` member and getter |
| `src/DSP/PhaseAnalyzer.cpp` | Parabolic interpolation after peak detection |
| `src/DSP/DelayBuffer.h` (new) | Simple integer delay line |
| `src/DSP/DelayBuffer.cpp` (new) | Delay line implementation |
| `src/PluginProcessor.h` | Add `subSampleOn`, delay buffer, pending results |
| `src/PluginProcessor.cpp` | Mode-dependent processing, analysis update |
| `src/GUI/MainComponent.h` | Add SS button |
| `src/GUI/MainComponent.cpp` | Button setup, layout, toggle handler |
| `src/Tools/MagicPhaseTest.cpp` | Print sub-sample delay |

## Estimated complexity

- Parabolic interpolation: ~10 lines
- DelayBuffer: ~40 lines
- ProcessBlock refactor: ~30 lines
- GUI button: ~15 lines
- Testing: ~50 lines

Total: ~150 lines of new code. Small, well-contained feature.
