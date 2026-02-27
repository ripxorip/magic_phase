# VST3 Integration Test Harness

## Problem

Our current test tools (MagicPhaseTest, FakeDAW) instantiate `MagicPhaseProcessor` directly
as a C++ class. They share the same source code as the VST but skip the VST3 wrapper, don't
provide an `AudioPlayHead`, and use sequential single-threaded processing. Bugs that only
manifest in a real DAW environment (sync timing, playhead-dependent logic, ring buffer races,
VST3 state serialization) are invisible to these tests.

Today's session proved this: MagicPhaseTest reported correct results (49 samples, 0.86
correlation) while the actual VST in a DAW had sync offsets, ring buffer read races, and
displayed wrong values. We spent hours on bugs that would have been caught instantly by
testing the actual binary.

## Solution

A command-line tool that loads the actual compiled `Magic Phase.vst3` binary, creates
multiple instances (like a DAW would), feeds them audio files, triggers alignment, and
outputs structured results for automated comparison.

## Architecture

```
                                 +------------------+
  test_definition.json -------->|                  |-------> result.json
  test_audio/*.wav ------------>|  VST3TestHarness |-------> output_audio/*.wav
  Magic Phase.vst3 ------------>|                  |-------> magic_phase_log.txt
                                 +------------------+
                                        |
                                        v
                                 SharedMemoryLayout
                                 (read for results)

  result.json + output_audio/ -------> compare.py -------> PASS / FAIL
```

### Components

1. **VST3TestHarness** (C++ JUCE GUI app, headless — no window)
   - Loads `.vst3` via `AudioPluginFormatManager`
   - Creates N instances, one per track
   - Provides `MockPlayHead` with advancing transport
   - Processes audio buffers in configurable order
   - Reads `SharedMemoryLayout` for result extraction
   - Writes output WAVs and result JSON

2. **Test definitions** (JSON files under `tests/`)
   - Declare input audio files, roles, modes, expected values
   - Parameterize buffer size, sample rate, processing order

3. **Python comparison scripts** (under `support/`)
   - Parse result JSON, compare against golden references
   - Independent audio analysis on output WAVs (cross-correlation, coherence)
   - Generate pass/fail report

## Test Definition Format

```json
{
  "name": "lfwh_sm57_front_vs_u87",
  "description": "Close mic SM57 vs room mic U87, known ~1ms delay",
  "plugin_path": "build/lib/VST3/Magic Phase.vst3/Contents/x86_64-win/Magic Phase.vst3",
  "sample_rate": 48000,
  "buffer_size": 64,
  "tracks": [
    {
      "file": "test_audio/lfwh_sm57_front.wav",
      "role": "reference"
    },
    {
      "file": "test_audio/lfwh_u87.wav",
      "role": "target",
      "mode": "phi"
    }
  ],
  "expected": {
    "delay_samples": { "value": 49, "tolerance": 2 },
    "delay_ms": { "value": 1.02, "tolerance": 0.1 },
    "correlation": { "min": 0.8 },
    "coherence": { "min": 0.7 },
    "polarity": "normal",
    "alignment_state": "ALIGNED"
  }
}
```

### Multi-track test (3+ mics)

```json
{
  "name": "drum_kit_3_mics",
  "description": "Kick inside, kick outside, overhead — tests multi-instance IPC",
  "plugin_path": "build/lib/VST3/Magic Phase.vst3/Contents/x86_64-win/Magic Phase.vst3",
  "sample_rate": 48000,
  "buffer_size": 128,
  "tracks": [
    { "file": "test_audio/kick_inside.wav", "role": "reference" },
    { "file": "test_audio/kick_outside.wav", "role": "target", "mode": "phi" },
    { "file": "test_audio/overhead_L.wav", "role": "target", "mode": "phi" }
  ],
  "expected_per_track": {
    "kick_outside": {
      "delay_ms": { "value": 0.5, "tolerance": 0.2 },
      "correlation": { "min": 0.7 }
    },
    "overhead_L": {
      "delay_ms": { "value": 3.2, "tolerance": 0.5 },
      "correlation": { "min": 0.5 }
    }
  }
}
```

### Parameterized tests (buffer size sweep)

```json
{
  "name": "buffer_size_sweep",
  "description": "Same audio, different buffer sizes — verifies sync robustness",
  "plugin_path": "build/lib/VST3/Magic Phase.vst3/Contents/x86_64-win/Magic Phase.vst3",
  "sample_rate": 48000,
  "buffer_sizes": [32, 64, 128, 256, 512, 1024, 2048, 4096],
  "tracks": [
    { "file": "test_audio/lfwh_sm57_front.wav", "role": "reference" },
    { "file": "test_audio/lfwh_u87.wav", "role": "target", "mode": "phi" }
  ],
  "expected": {
    "delay_samples": { "value": 49, "tolerance": 2 },
    "correlation": { "min": 0.8 },
    "coherence": { "min": 0.7 }
  }
}
```

This is the killer test — it would have caught today's sync bugs instantly. If the result
varies with buffer size, the sync mechanism is broken.

## Result Output Format

```json
{
  "test": "lfwh_sm57_front_vs_u87",
  "timestamp": "2026-02-27T14:32:05Z",
  "config": {
    "plugin_path": "build/lib/VST3/Magic Phase.vst3/Contents/x86_64-win/Magic Phase.vst3",
    "sample_rate": 48000,
    "buffer_size": 64,
    "plugin_loaded": true,
    "num_instances": 2
  },
  "tracks": [
    {
      "name": "lfwh_sm57_front",
      "role": "reference",
      "input_file": "test_audio/lfwh_sm57_front.wav",
      "output_file": "results/lfwh_sm57_front_vs_u87/lfwh_sm57_front_out.wav",
      "slot": 0
    },
    {
      "name": "lfwh_u87",
      "role": "target",
      "mode": "phi",
      "input_file": "test_audio/lfwh_u87.wav",
      "output_file": "results/lfwh_sm57_front_vs_u87/lfwh_u87_out.wav",
      "slot": 1,
      "results": {
        "alignment_state": "ALIGNED",
        "delay_samples": 49.0,
        "delay_ms": 1.02,
        "correlation": 0.864,
        "coherence": 0.776,
        "phase_degrees": -17.5,
        "polarity_inverted": false,
        "time_correction_on": true,
        "phase_correction_on": true,
        "spectral_bands": [0.92, 0.88, 0.85, 0.91, "...48 values..."]
      }
    }
  ],
  "timing": {
    "plugin_load_ms": 120,
    "prepare_ms": 5,
    "playback_ms": 7800,
    "analysis_wait_ms": 450,
    "total_ms": 8375
  },
  "diagnostics": {
    "sync_offset_samples": 0,
    "ref_raw_start_sample": 24000,
    "target_raw_start_sample": 24000,
    "raw_delay_before_offset": 49,
    "log_file": "results/lfwh_sm57_front_vs_u87/magic_phase_log.txt"
  }
}
```

## VST3TestHarness Implementation

### CMake target

```cmake
juce_add_gui_app(VST3TestHarness
    PRODUCT_NAME "VST3 Test Harness"
    NEEDS_CURL FALSE
    NEEDS_WEB_BROWSER FALSE)

target_sources(VST3TestHarness PRIVATE
    src/Tools/VST3TestHarness/Main.cpp
    src/Tools/VST3TestHarness/MockPlayHead.h
    src/Tools/VST3TestHarness/TestRunner.h
    src/Tools/VST3TestHarness/TestRunner.cpp
    src/Tools/VST3TestHarness/ResultWriter.h
    src/Tools/VST3TestHarness/ResultWriter.cpp
    src/IPC/SharedState.h
    src/IPC/SharedState.cpp
    src/IPC/PlatformSharedMemory.h
    src/IPC/PlatformSharedMemory_win32.cpp)

target_link_libraries(VST3TestHarness PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_audio_formats
    juce::juce_gui_basics)
```

Note: The harness links `SharedState` directly (same struct definitions) so it can read
the shared memory layout. It does NOT link the plugin code — the plugin is loaded as a
binary `.vst3` file. SharedState.h is shared because it defines the memory layout that
both the plugin and the harness need to agree on.

### MockPlayHead

```cpp
class MockPlayHead : public juce::AudioPlayHead
{
public:
    juce::Optional<PositionInfo> getPosition() const override
    {
        PositionInfo info;
        info.setTimeInSamples (currentSample);
        info.setTimeInSeconds (static_cast<double> (currentSample) / sampleRate);
        info.setIsPlaying (playing);
        info.setBpm (120.0);
        return info;
    }

    void advance (int numSamples) { currentSample += numSamples; }
    void setPlaying (bool p) { playing = p; }
    void setSampleRate (double sr) { sampleRate = sr; }
    void reset() { currentSample = 0; }

private:
    int64_t currentSample = 0;
    double sampleRate = 48000.0;
    bool playing = false;
};
```

Both plugin instances share the SAME MockPlayHead pointer. This means they see the same
transport position in the same "callback" — exactly like a real DAW. The harness advances
it once per buffer, after processing all instances.

### Processing Loop (core logic)

```cpp
void TestRunner::run (const TestDefinition& test)
{
    // 1. Load plugin
    juce::AudioPluginFormatManager formatManager;
    formatManager.addDefaultFormats();

    juce::String error;
    std::vector<std::unique_ptr<juce::AudioPluginInstance>> instances;

    for (auto& track : test.tracks)
    {
        juce::PluginDescription desc;
        desc.fileOrIdentifier = test.pluginPath;
        desc.pluginFormatName = "VST3";

        auto instance = formatManager.createPluginInstance (
            desc, test.sampleRate, test.bufferSize, error);

        if (instance == nullptr)
            return reportError ("Failed to load plugin: " + error);

        instance->setPlayHead (&playHead);
        instance->prepareToPlay (test.sampleRate, test.bufferSize);
        instances.push_back (std::move (instance));
    }

    // 2. Load audio files
    std::vector<juce::AudioBuffer<float>> inputBuffers;
    for (auto& track : test.tracks)
        inputBuffers.push_back (loadWavFile (track.file, test.sampleRate));

    // 3. Set reference track
    //    Access plugin parameters to toggle REF button on instance 0
    setPluginParameter (instances[0], "isReference", true);

    // 4. Playback phase — process audio WITHOUT alignment
    //    Let the plugins see each other in shared memory, settle heartbeats
    playHead.setPlaying (true);
    int settleBlocks = static_cast<int> (test.sampleRate / test.bufferSize);  // ~1 second

    for (int block = 0; block < settleBlocks; ++block)
    {
        for (size_t i = 0; i < instances.size(); ++i)
            processOneBlock (instances[i], inputBuffers[i], block);
        playHead.advance (test.bufferSize);
    }

    // 5. Trigger alignment on target instances
    for (size_t i = 0; i < instances.size(); ++i)
    {
        if (test.tracks[i].role == "target")
            triggerAlignment (instances[i]);
    }

    // 6. Accumulation phase — process 7.5+ seconds of audio
    int accumBlocks = static_cast<int> (
        (MagicPhaseProcessor::kRequiredSeconds + 1.0f)
        * test.sampleRate / test.bufferSize);

    std::vector<juce::AudioBuffer<float>> outputBuffers (instances.size());
    for (auto& buf : outputBuffers)
        buf.setSize (1, accumBlocks * test.bufferSize);

    for (int block = 0; block < accumBlocks; ++block)
    {
        for (size_t i = 0; i < instances.size(); ++i)
        {
            auto chunk = getAudioChunk (inputBuffers[i], block, test.bufferSize);
            juce::MidiBuffer midi;
            instances[i]->processBlock (chunk, midi);
            copyToOutput (outputBuffers[i], chunk, block, test.bufferSize);
        }
        playHead.advance (test.bufferSize);
    }

    // 7. Wait for analysis to complete
    //    Poll shared memory for alignment state change
    int maxWaitBlocks = static_cast<int> (5.0 * test.sampleRate / test.bufferSize);
    bool aligned = false;

    for (int block = 0; block < maxWaitBlocks && !aligned; ++block)
    {
        for (size_t i = 0; i < instances.size(); ++i)
        {
            auto chunk = getAudioChunk (inputBuffers[i],
                accumBlocks + block, test.bufferSize);
            juce::MidiBuffer midi;
            instances[i]->processBlock (chunk, midi);
            copyToOutput (outputBuffers[i], chunk, accumBlocks + block, test.bufferSize);
        }
        playHead.advance (test.bufferSize);

        // Check shared memory for ALIGNED state
        aligned = checkAllTargetsAligned (test);
    }

    // 8. Post-alignment processing — capture corrected output
    int postBlocks = static_cast<int> (2.0 * test.sampleRate / test.bufferSize);
    for (int block = 0; block < postBlocks; ++block)
    {
        for (size_t i = 0; i < instances.size(); ++i)
        {
            auto chunk = getAudioChunk (inputBuffers[i],
                accumBlocks + maxWaitBlocks + block, test.bufferSize);
            juce::MidiBuffer midi;
            instances[i]->processBlock (chunk, midi);
            // This output is the corrected audio — the important part
            copyToOutput (outputBuffers[i], chunk,
                accumBlocks + maxWaitBlocks + block, test.bufferSize);
        }
        playHead.advance (test.bufferSize);
    }

    // 9. Read results from shared memory
    SharedState sharedState;
    sharedState.initialize();
    auto results = readResultsFromSharedMemory (sharedState, test);

    // 10. Write output WAVs
    for (size_t i = 0; i < instances.size(); ++i)
        writeWavFile (outputBuffers[i], test.outputDir + "/" + track.name + "_out.wav");

    // 11. Write result JSON
    writeResultJson (test, results, timingInfo);

    // 12. Cleanup
    for (auto& instance : instances)
    {
        instance->releaseResources();
        instance.reset();
    }
    sharedState.shutdown();
}
```

### Triggering Alignment

Since the plugin is loaded as a black box, we can't call `startAlign()` directly. Options:

**Option A: Expose alignment trigger as a VST3 parameter**
Add a hidden parameter `"triggerAlign"` to the plugin's APVTS. When set to 1.0, the
plugin calls `startAlign()`. The test harness sets this parameter on the loaded instance.

```cpp
// In PluginProcessor — add to parameter layout:
params.push_back (std::make_unique<juce::AudioParameterBool> (
    juce::ParameterID { "triggerAlign", 1 }, "Trigger Align", false));

// In processBlock — check for trigger:
if (apvts.getRawParameterValue ("triggerAlign")->load() > 0.5f)
{
    apvts.getParameter ("triggerAlign")->setValueNotifyingHost (0.0f);
    startAlign();
}
```

Similarly for `"isReference"`, `"correctionMode"`, `"bypass"`.

**Option B: Use plugin state (setStateInformation)**
Serialize a state blob that includes alignment commands. More complex, less clean.

**Recommendation: Option A.** Add automatable parameters for all actions the test harness
needs to trigger. These parameters are also useful for DAW automation (user could automate
alignment trigger via a DAW automation lane).

### Parameters to Expose

| Parameter | Type | Purpose |
|-----------|------|---------|
| `triggerAlign` | Bool | Starts alignment (auto-resets to false) |
| `isReference` | Bool | Set this instance as the reference track |
| `correctionMode` | Int (0-1) | 0=T, 1=Phi |
| `bypass` | Bool | A/B bypass |

The existing `coherenceThreshold` and `maxCorrection` are already parameters.

### Reading Results from Shared Memory

The harness links `SharedState.h/cpp` and `PlatformSharedMemory` directly:

```cpp
TestResults readResultsFromSharedMemory (SharedState& state, const TestDefinition& test)
{
    TestResults results;
    auto* header = state.getHeader();

    for (int i = 0; i < kMaxInstances; ++i)
    {
        auto* slot = state.getSlot (i);
        if (slot == nullptr || slot->active.load() == 0)
            continue;

        TrackResult tr;
        tr.name = juce::String (slot->trackName);
        tr.slotIndex = i;
        tr.isReference = (i == header->referenceSlot.load());
        tr.isAligned = (slot->active.load() == 2);
        tr.delaySamples = slot->delaySamples;
        tr.delayMs = slot->delayMs;
        tr.correlation = slot->correlation;
        tr.coherence = slot->overallCoherence;
        tr.phaseDegrees = slot->phaseDegrees;
        tr.polarityInverted = slot->polarityInverted;
        tr.timeCorrectionOn = slot->timeCorrectionOn;
        tr.phaseCorrectionOn = slot->phaseCorrectionOn;
        std::memcpy (tr.spectralBands, slot->spectralBands, sizeof (tr.spectralBands));

        results.tracks.push_back (tr);
    }

    // Sync diagnostics
    results.syncAcknowledged = header->syncAcknowledged.load();
    results.refRawStartSample = header->refRawStartSample.load();

    return results;
}
```

## Python Comparison Scripts

### compare_results.py

```python
#!/usr/bin/env python3
"""Compare VST3 test harness results against golden references."""

import json
import sys
import numpy as np
import soundfile as sf
from pathlib import Path


def compare_values(actual, expected):
    """Compare a result value against an expected spec."""
    errors = []

    if "value" in expected:
        tolerance = expected.get("tolerance", 0)
        if abs(actual - expected["value"]) > tolerance:
            errors.append(
                f"  Expected {expected['value']} +/- {tolerance}, got {actual}"
            )

    if "min" in expected and actual < expected["min"]:
        errors.append(f"  Expected >= {expected['min']}, got {actual}")

    if "max" in expected and actual > expected["max"]:
        errors.append(f"  Expected <= {expected['max']}, got {actual}")

    return errors


def compare_result_json(result_path, test_def_path):
    """Compare result JSON against test definition expected values."""
    result = json.loads(Path(result_path).read_text())
    test_def = json.loads(Path(test_def_path).read_text())

    expected = test_def.get("expected", {})
    errors = []

    for track in result["tracks"]:
        if track["role"] == "reference":
            continue

        r = track["results"]

        for key, spec in expected.items():
            if key == "alignment_state":
                if r.get("alignment_state") != spec:
                    errors.append(f"  alignment_state: expected {spec}, "
                                  f"got {r.get('alignment_state')}")
            elif key == "polarity":
                actual_pol = "inverted" if r.get("polarity_inverted") else "normal"
                if actual_pol != spec:
                    errors.append(f"  polarity: expected {spec}, got {actual_pol}")
            elif key in r:
                errors.extend(compare_values(r[key], spec))

    return errors


def analyze_output_audio(ref_wav, target_wav):
    """Independent audio analysis — don't trust the plugin's numbers."""
    ref, sr = sf.read(ref_wav)
    tar, sr2 = sf.read(target_wav)
    assert sr == sr2, "Sample rate mismatch"

    # Trim to same length
    min_len = min(len(ref), len(tar))
    ref = ref[:min_len]
    tar = tar[:min_len]

    # Cross-correlation (verify delay)
    correlation = np.correlate(ref, tar, mode="full")
    peak_idx = np.argmax(np.abs(correlation))
    measured_delay = peak_idx - (min_len - 1)

    # Coherence (frequency domain)
    from scipy.signal import coherence as scipy_coherence
    freqs, coh = scipy_coherence(ref, tar, fs=sr, nperseg=4096)
    avg_coherence = np.mean(coh[(freqs > 100) & (freqs < 8000)])

    # Sum check — if phase-aligned, sum should be louder than either alone
    sum_signal = ref + tar
    ref_rms = np.sqrt(np.mean(ref ** 2))
    tar_rms = np.sqrt(np.mean(tar ** 2))
    sum_rms = np.sqrt(np.mean(sum_signal ** 2))
    sum_gain_db = 20 * np.log10(sum_rms / max(ref_rms, tar_rms))

    return {
        "measured_delay_samples": int(measured_delay),
        "avg_coherence_100_8k": float(avg_coherence),
        "sum_gain_db": float(sum_gain_db),
        "ref_rms": float(ref_rms),
        "tar_rms": float(tar_rms),
        "sum_rms": float(sum_rms),
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_results.py <result.json> <test_definition.json>")
        sys.exit(1)

    result_path = sys.argv[1]
    test_def_path = sys.argv[2]

    print(f"Comparing {result_path} against {test_def_path}")
    print("=" * 60)

    # Value comparison
    errors = compare_result_json(result_path, test_def_path)
    if errors:
        print("FAIL — Value comparison errors:")
        for e in errors:
            print(e)
    else:
        print("PASS — All values within tolerance")

    # Audio analysis
    result = json.loads(Path(result_path).read_text())
    ref_track = next(t for t in result["tracks"] if t["role"] == "reference")
    tar_tracks = [t for t in result["tracks"] if t["role"] == "target"]

    for tar_track in tar_tracks:
        print(f"\nAudio analysis: {tar_track['name']}")
        analysis = analyze_output_audio(
            ref_track["output_file"], tar_track["output_file"]
        )
        for k, v in analysis.items():
            print(f"  {k}: {v}")

        # Sum gain > 3dB means constructive summing (good phase alignment)
        if analysis["sum_gain_db"] > 3.0:
            print("  PASS — Constructive summing (phase-aligned)")
        else:
            print(f"  WARN — Sum gain only {analysis['sum_gain_db']:.1f} dB")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
```

### run_all_tests.py

```python
#!/usr/bin/env python3
"""Run all VST3 integration tests."""

import json
import subprocess
import sys
from pathlib import Path

TEST_DIR = Path("tests/integration")
HARNESS_EXE = Path("build/bin/Release/VST3TestHarness.exe")
RESULTS_DIR = Path("results")


def run_test(test_file):
    """Run a single test definition through the harness."""
    test_def = json.loads(test_file.read_text())
    test_name = test_def["name"]
    result_dir = RESULTS_DIR / test_name
    result_dir.mkdir(parents=True, exist_ok=True)

    result_file = result_dir / "result.json"

    # Run the C++ harness
    cmd = [
        str(HARNESS_EXE),
        "--test", str(test_file),
        "--output-dir", str(result_dir),
        "--result", str(result_file),
    ]

    # Handle buffer size sweep
    if "buffer_sizes" in test_def:
        all_passed = True
        for bs in test_def["buffer_sizes"]:
            sweep_result = result_dir / f"result_bs{bs}.json"
            sweep_cmd = cmd + ["--buffer-size", str(bs),
                               "--result", str(sweep_result)]
            print(f"  Buffer size {bs}...", end=" ", flush=True)
            proc = subprocess.run(sweep_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"HARNESS FAILED: {proc.stderr}")
                all_passed = False
                continue

            # Compare
            cmp = subprocess.run(
                [sys.executable, "support/compare_results.py",
                 str(sweep_result), str(test_file)],
                capture_output=True, text=True
            )
            if cmp.returncode != 0:
                print(f"FAIL")
                print(cmp.stdout)
                all_passed = False
            else:
                print("PASS")
        return all_passed

    # Single run
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  HARNESS FAILED: {proc.stderr}")
        return False

    # Compare
    cmp = subprocess.run(
        [sys.executable, "support/compare_results.py",
         str(result_file), str(test_file)],
        capture_output=True, text=True
    )
    print(cmp.stdout)
    return cmp.returncode == 0


def main():
    test_files = sorted(TEST_DIR.glob("*.json"))
    if not test_files:
        print(f"No test files found in {TEST_DIR}")
        sys.exit(1)

    print(f"Running {len(test_files)} integration tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_file in test_files:
        test_def = json.loads(test_file.read_text())
        print(f"\n[TEST] {test_def['name']}")

        if run_test(test_file):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
```

## Usage

### Single test

```bash
# Build
cmake --build build --config Release

# Run one test
./build/bin/Release/VST3TestHarness.exe \
    --test tests/integration/lfwh_sm57_vs_u87.json \
    --output-dir results/lfwh_sm57_vs_u87

# Compare
python support/compare_results.py \
    results/lfwh_sm57_vs_u87/result.json \
    tests/integration/lfwh_sm57_vs_u87.json
```

### Full suite

```bash
python support/run_all_tests.py
```

### Buffer size sweep (the sync robustness test)

```bash
./build/bin/Release/VST3TestHarness.exe \
    --test tests/integration/buffer_size_sweep.json \
    --output-dir results/buffer_sweep
```

This runs the same audio through buffer sizes 32, 64, 128, 256, 512, 1024, 2048, 4096
and verifies that delay, correlation, and coherence are consistent across all sizes.
If any buffer size produces different results, the sync mechanism has a bug.

## Directory Structure

```
magic_phase/
  src/Tools/VST3TestHarness/
    Main.cpp                    # JUCE app entry point
    MockPlayHead.h              # Transport simulation
    TestRunner.h/.cpp           # Core processing loop
    ResultWriter.h/.cpp         # JSON + WAV output
  tests/
    integration/
      lfwh_sm57_vs_u87.json    # Test definition
      drum_kit_3_mics.json
      buffer_size_sweep.json
      polarity_invert.json
  test_audio/
    lfwh_sm57_front.wav
    lfwh_u87.wav
    kick_inside.wav
    ...
  support/
    compare_results.py          # Golden reference comparison
    run_all_tests.py            # Test runner
  results/                      # Git-ignored output directory
    lfwh_sm57_vs_u87/
      result.json
      lfwh_sm57_front_out.wav
      lfwh_u87_out.wav
      magic_phase_log.txt
```

## Plugin Changes Required

Minimal changes to the plugin itself:

1. **Add automatable parameters** for test harness control:
   - `triggerAlign` (bool, auto-resets)
   - `isReference` (bool)
   - `correctionMode` (int 0-1)
   - `bypass` (bool)

2. These parameters are already implemented as internal methods (`startAlign()`,
   `setIsReference()`, etc.). The change is just exposing them through APVTS so the
   VST3 host (our test harness) can set them via the standard plugin API.

3. No "test mode" flags, no conditional compilation, no logging toggles.
   The plugin runs exactly as it would in a DAW. Results come from shared memory
   and output audio.

## What This Catches That Current Tests Don't

| Bug type | MagicPhaseTest | FakeDAW | VST3 Harness |
|----------|---------------|---------|--------------|
| DSP algorithm correctness | Yes | Yes | Yes |
| Shared memory IPC | No | Partial | Yes |
| Sync handshake timing | No | No | Yes |
| Playhead-dependent logic | No | No | Yes |
| Buffer size sensitivity | No | Partial | Yes (sweep) |
| VST3 wrapper bugs | No | No | Yes |
| Ring buffer race conditions | No | No | Yes |
| Parameter serialization | No | No | Yes |
| Multi-instance (3+ tracks) | No | No | Yes |
| Plugin load/unload stability | No | No | Yes |

## Future Extensions

- **CI/CD integration**: Run on every PR. Build VST3 + harness, run test suite, fail if
  regression detected.
- **Golden reference generation**: `--generate-golden` flag that writes result JSON as the
  new golden reference instead of comparing against one.
- **Audio diff tool**: Python script that computes and visualizes the spectral difference
  between output WAVs from two different builds.
- **Stress testing**: Rapid alignment/cancel cycles, plugin load/unload during processing,
  sample rate changes mid-session.
- **Processing order variation**: Run tests with ref-first and target-first processing
  order to verify sync works in both directions.
