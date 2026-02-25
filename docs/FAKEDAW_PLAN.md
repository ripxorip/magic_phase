# FakeDAW Test Harness - Complete Design Document

## Overview

A lightweight test harness that simulates DAW behavior for testing Magic Phase plugin
without the overhead and debugging difficulty of a real DAW. This gives us:

- Direct access to plugin internals (no VST3 wrapping)
- Full logging at every step
- Scripted test scenarios
- Automated verification
- Debugger-friendly (set breakpoints anywhere)
- CI/CD compatible

---

## Why Not JUCE AudioPluginHost?

| Aspect | JUCE AudioPluginHost | Custom FakeDAW |
|--------|---------------------|----------------|
| Plugin loading | VST3/AU format wrapping | Direct C++ instantiation |
| GUI | Full node graph editor | None (headless) |
| Logging | Limited, external | Full control, inline |
| Scripting | Not possible | Timeline-based events |
| Debugging | Hard (plugin as binary) | Easy (same codebase) |
| Lines of code | ~10,000+ | ~500-800 |
| CI/CD | Difficult | Easy |

**Key insight:** A DAW just calls `prepareToPlay()` and `processBlock()`. We can do that directly.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FakeDAW                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    TrackManager                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐           │   │
│  │  │  Track 0  │  │  Track 1  │  │  Track 2  │   ...     │   │
│  │  │           │  │           │  │           │           │   │
│  │  │ processor │  │ processor │  │ processor │           │   │
│  │  │ audioFile │  │ audioFile │  │ audioFile │           │   │
│  │  │ outBuffer │  │ outBuffer │  │ outBuffer │           │   │
│  │  └───────────┘  └───────────┘  └───────────┘           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   PlaybackEngine                         │   │
│  │                                                          │   │
│  │  - Maintains sample position                             │   │
│  │  - Calls processBlock() on ALL tracks each iteration    │   │
│  │  - Tracks are processed in sync (same sample position)  │   │
│  │  - Simulates real DAW playback behavior                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   EventTimeline                          │   │
│  │                                                          │   │
│  │  Events sorted by time:                                  │   │
│  │    t=0.0s:  SetReference(track=0)                       │   │
│  │    t=7.5s:  TriggerAlign(track=1)                       │   │
│  │    t=8.0s:  Checkpoint("after_align")                   │   │
│  │    t=15.0s: Stop()                                      │   │
│  │                                                          │   │
│  │  Processes events when playback reaches their time      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Logger                              │   │
│  │                                                          │   │
│  │  Categories:                                             │   │
│  │    [INIT]  - Initialization, registration               │   │
│  │    [IPC]   - Shared memory operations                   │   │
│  │    [DSP]   - STFT frames, analysis                      │   │
│  │    [GUI]   - Simulated user actions                     │   │
│  │    [STATE] - Plugin state changes                       │   │
│  │    [AUDIO] - RMS, energy measurements                   │   │
│  │                                                          │   │
│  │  Output: console + log file                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Verifier                              │   │
│  │                                                          │   │
│  │  - Compares actual results to expected                  │   │
│  │  - Checks delay, correlation, energy gain               │   │
│  │  - Generates pass/fail report                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Classes

### 1. FakeDAW (main orchestrator)

```cpp
class FakeDAW
{
public:
    FakeDAW (double sampleRate = 48000.0, int blockSize = 128);

    // Track management
    int addTrack (const std::string& name, const std::string& audioFilePath);
    Track& getTrack (int index);
    int getNumTracks() const;

    // Timeline
    void addEvent (double timeSeconds, std::unique_ptr<Action> action);

    // Execution
    void run();
    void runScenario (const std::string& scenarioName);

    // Results
    const Results& getResults() const;
    void saveOutputs (const std::string& outputDir);
    bool verify (const ExpectedResults& expected);

private:
    double sampleRate;
    int blockSize;
    std::vector<std::unique_ptr<Track>> tracks;
    EventTimeline timeline;
    Logger logger;
    Results results;
};
```

### 2. Track

```cpp
struct Track
{
    std::string name;
    MagicPhaseProcessor processor;
    std::vector<float> inputAudio;      // Loaded from file
    std::vector<float> outputAudio;     // Recorded during playback
    juce::AudioBuffer<float> buffer;    // Working buffer for processBlock

    void loadAudio (const std::string& path);
    void prepare (double sampleRate, int blockSize);
    void process (int startSample);     // Fills buffer, calls processBlock
    void release();
};
```

### 3. EventTimeline

```cpp
class EventTimeline
{
public:
    void addEvent (double timeSeconds, std::unique_ptr<Action> action);
    void processEventsUpTo (double timeSeconds, FakeDAW& daw);
    bool hasMoreEvents() const;

private:
    struct TimedEvent {
        double time;
        std::unique_ptr<Action> action;
    };
    std::vector<TimedEvent> events;  // Sorted by time
    size_t nextEventIndex = 0;
};
```

### 4. Action (base class + derived)

```cpp
struct Action
{
    virtual ~Action() = default;
    virtual void execute (FakeDAW& daw, Logger& log) = 0;
    virtual std::string describe() const = 0;
};

struct SetReference : Action
{
    int trackIndex;
    void execute (FakeDAW& daw, Logger& log) override {
        log.gui ("Track {}: setIsReference(true)", trackIndex);
        daw.getTrack (trackIndex).processor.setIsReference (true);
    }
};

struct ClearReference : Action
{
    int trackIndex;
    void execute (FakeDAW& daw, Logger& log) override {
        log.gui ("Track {}: setIsReference(false)", trackIndex);
        daw.getTrack (trackIndex).processor.setIsReference (false);
    }
};

struct TriggerAlign : Action
{
    int trackIndex;
    void execute (FakeDAW& daw, Logger& log) override {
        log.gui ("Track {}: triggerAlign()", trackIndex);
        auto& proc = daw.getTrack (trackIndex).processor;
        proc.triggerAlign();

        // Log results
        auto& analyzer = proc.getPhaseAnalyzer();
        log.dsp ("Track {}: Analysis complete:", trackIndex);
        log.dsp ("  Delay: {} samples ({} ms)",
                 analyzer.getDelaySamples(), analyzer.getDelayMs());
        log.dsp ("  Correlation: {}", analyzer.getCorrelation());
        log.dsp ("  Polarity: {}", analyzer.getPolarityInverted() ? "INVERTED" : "Normal");
        log.dsp ("  Coherence: {}", analyzer.getOverallCoherence());
        log.dsp ("  Phase avg: {}°", analyzer.getPhaseDegrees());
    }
};

struct SetCorrectionMode : Action
{
    int trackIndex;
    int mode;  // 0=T+Phi, 1=Phi, 2=T
    void execute (FakeDAW& daw, Logger& log) override {
        const char* modeNames[] = {"T+Phi", "Phi", "T"};
        log.gui ("Track {}: setCorrectionMode({})", trackIndex, modeNames[mode]);
        daw.getTrack (trackIndex).processor.setCorrectionMode (mode);
    }
};

struct SetBypass : Action
{
    int trackIndex;
    bool bypassed;
    void execute (FakeDAW& daw, Logger& log) override {
        log.gui ("Track {}: setBypass({})", trackIndex, bypassed);
        daw.getTrack (trackIndex).processor.setBypass (bypassed);
    }
};

struct Checkpoint : Action
{
    std::string name;
    void execute (FakeDAW& daw, Logger& log) override {
        log.state ("=== CHECKPOINT: {} ===", name);
        // Capture current state of all tracks
        for (int i = 0; i < daw.getNumTracks(); i++) {
            auto& track = daw.getTrack (i);
            float rms = calculateRMS (track.outputAudio);
            log.audio ("Track {}: RMS = {} dB", i, 20.0f * std::log10 (rms));
        }
        // Save intermediate outputs if needed
    }
};

struct Stop : Action
{
    void execute (FakeDAW& daw, Logger& log) override {
        log.state ("Playback stopped");
        daw.stop();
    }
};
```

### 5. Logger

```cpp
class Logger
{
public:
    enum Category { INIT, IPC, DSP, GUI, STATE, AUDIO };

    void setOutputFile (const std::string& path);
    void setVerbose (bool verbose);

    template<typename... Args>
    void init (const char* fmt, Args&&... args);

    template<typename... Args>
    void ipc (const char* fmt, Args&&... args);

    template<typename... Args>
    void dsp (const char* fmt, Args&&... args);

    template<typename... Args>
    void gui (const char* fmt, Args&&... args);

    template<typename... Args>
    void state (const char* fmt, Args&&... args);

    template<typename... Args>
    void audio (const char* fmt, Args&&... args);

private:
    void log (Category cat, double timeSeconds, const std::string& message);

    std::ofstream file;
    bool verbose = false;
    double currentTime = 0.0;
};
```

---

## Playback Loop (Core Logic)

```cpp
void FakeDAW::run()
{
    logger.init ("FakeDAW starting: {} tracks, {}Hz, {} block size",
                 tracks.size(), sampleRate, blockSize);

    // 1. Initialize all tracks
    for (int i = 0; i < tracks.size(); i++) {
        tracks[i]->prepare (sampleRate, blockSize);
        logger.init ("Track {} '{}' loaded: {} samples",
                     i, tracks[i]->name, tracks[i]->inputAudio.size());
        logger.ipc ("Track {} registered as slot {}",
                    i, tracks[i]->processor.getMySlot());
    }

    // 2. Find the longest track (determines max playback time)
    size_t maxSamples = 0;
    for (auto& track : tracks)
        maxSamples = std::max (maxSamples, track->inputAudio.size());

    // 3. Playback loop
    running = true;
    int samplePos = 0;
    juce::MidiBuffer emptyMidi;

    while (running && samplePos < maxSamples)
    {
        double timeSeconds = samplePos / sampleRate;
        logger.setCurrentTime (timeSeconds);

        // Process any events at this time
        timeline.processEventsUpTo (timeSeconds, *this);

        // Process all tracks (synchronized, like a DAW)
        for (auto& track : tracks)
        {
            // Fill buffer from input audio
            int samplesToProcess = std::min (blockSize,
                                              (int)(track->inputAudio.size() - samplePos));
            if (samplesToProcess <= 0) {
                // Track ended, fill with silence
                track->buffer.clear();
                samplesToProcess = blockSize;
            } else {
                // Copy from input audio to buffer
                std::memcpy (track->buffer.getWritePointer (0),
                            track->inputAudio.data() + samplePos,
                            samplesToProcess * sizeof (float));
            }

            // Call processBlock (this is what the DAW does!)
            track->processor.processBlock (track->buffer, emptyMidi);

            // Record output
            auto* outputData = track->buffer.getReadPointer (0);
            track->outputAudio.insert (track->outputAudio.end(),
                                       outputData, outputData + samplesToProcess);
        }

        samplePos += blockSize;

        // Progress indication (every second)
        if (samplePos % (int)sampleRate < blockSize) {
            logger.state ("Playback: {:.1f}s / {:.1f}s",
                         timeSeconds, maxSamples / sampleRate);
        }
    }

    // 4. Cleanup
    for (auto& track : tracks) {
        track->release();
    }

    // 5. Calculate final results
    calculateResults();

    logger.state ("FakeDAW finished");
}
```

---

## Adding Logging to SharedState

To see IPC operations, we add logging hooks to SharedState:

```cpp
// In SharedState.cpp

int SharedState::registerInstance (const juce::String& trackName)
{
    // ... existing code ...

    #ifdef FAKEDAW_LOGGING
    FakeDAWLogger::ipc ("registerInstance('{}') -> slot {}", trackName, slot);
    #endif

    return slot;
}

void SharedState::writeReferenceFrame (const std::complex<float>* frame, int numBins)
{
    // ... existing code ...

    #ifdef FAKEDAW_LOGGING
    FakeDAWLogger::dsp ("writeReferenceFrame: pos={}, bins={}", writePos, numBins);
    #endif
}

std::vector<std::vector<std::complex<float>>> SharedState::readReferenceFrames() const
{
    // ... existing code ...

    #ifdef FAKEDAW_LOGGING
    FakeDAWLogger::ipc ("readReferenceFrames: {} frames", result.size());
    #endif

    return result;
}
```

Or use a callback system to avoid preprocessor directives:

```cpp
// In SharedState.h
class SharedState
{
public:
    // Logging callback (set by FakeDAW)
    std::function<void(const std::string&)> onLogMessage;

    // ... rest of class ...
};

// In SharedState.cpp
void SharedState::writeReferenceFrame (...)
{
    // ... existing code ...

    if (onLogMessage)
        onLogMessage (fmt::format ("writeReferenceFrame: pos={}", writePos));
}
```

---

## Test Scenarios

### Scenario 1: Basic Two-Track Alignment

**File: scenarios/basic_align.cpp**

```cpp
void runBasicAlignScenario (FakeDAW& daw)
{
    daw.addTrack ("Kick In", "input/lfwh_u87.wav");
    daw.addTrack ("Kick Out", "input/lfwh_sm57_front.wav");

    daw.addEvent (0.0,  std::make_unique<SetReference> (0));
    daw.addEvent (7.5,  std::make_unique<TriggerAlign> (1));
    daw.addEvent (7.5,  std::make_unique<Checkpoint> ("after_align"));
    daw.addEvent (15.0, std::make_unique<Stop>());

    daw.run();

    ExpectedResults expected;
    expected.delayMsRange = {-2.0, 0.0};      // -1.02ms expected
    expected.minCorrelation = 0.7;
    expected.minEnergyGainDb = 3.0;

    bool passed = daw.verify (expected);
    daw.saveOutputs ("output/basic_align/");
}
```

### Scenario 2: Delayed Reference Selection

```cpp
void runDelayedRefScenario (FakeDAW& daw)
{
    daw.addTrack ("Kick In", "input/lfwh_u87.wav");
    daw.addTrack ("Kick Out", "input/lfwh_sm57_front.wav");

    // User waits 3 seconds before clicking REF
    daw.addEvent (3.0,  std::make_unique<SetReference> (0));
    daw.addEvent (10.5, std::make_unique<TriggerAlign> (1));
    daw.addEvent (15.0, std::make_unique<Stop>());

    daw.run();

    // Should still work - enough frames after ref was set
    ExpectedResults expected;
    expected.minCorrelation = 0.7;
    expected.minEnergyGainDb = 2.0;  // Might be slightly worse

    bool passed = daw.verify (expected);
}
```

### Scenario 3: Three Tracks

```cpp
void runThreeTrackScenario (FakeDAW& daw)
{
    daw.addTrack ("Kick In", "input/lfwh_u87.wav");
    daw.addTrack ("Kick Out", "input/lfwh_sm57_front.wav");
    daw.addTrack ("Kick Room", "input/lfwh_room.wav");  // If you have this file

    daw.addEvent (0.0,  std::make_unique<SetReference> (0));
    daw.addEvent (7.5,  std::make_unique<TriggerAlign> (1));
    daw.addEvent (7.5,  std::make_unique<TriggerAlign> (2));
    daw.addEvent (7.5,  std::make_unique<Checkpoint> ("both_aligned"));
    daw.addEvent (15.0, std::make_unique<Stop>());

    daw.run();
}
```

### Scenario 4: Mode Switching

```cpp
void runModeSwitchScenario (FakeDAW& daw)
{
    daw.addTrack ("Kick In", "input/lfwh_u87.wav");
    daw.addTrack ("Kick Out", "input/lfwh_sm57_front.wav");

    daw.addEvent (0.0,  std::make_unique<SetReference> (0));
    daw.addEvent (7.5,  std::make_unique<TriggerAlign> (1));

    // Test all modes
    daw.addEvent (8.0,  std::make_unique<SetCorrectionMode> (1, 0));  // T+Phi
    daw.addEvent (9.0,  std::make_unique<Checkpoint> ("mode_t_phi"));

    daw.addEvent (9.0,  std::make_unique<SetCorrectionMode> (1, 2));  // T only
    daw.addEvent (10.0, std::make_unique<Checkpoint> ("mode_t_only"));

    daw.addEvent (10.0, std::make_unique<SetCorrectionMode> (1, 1));  // Phi only
    daw.addEvent (11.0, std::make_unique<Checkpoint> ("mode_phi_only"));

    daw.addEvent (11.0, std::make_unique<SetBypass> (1, true));
    daw.addEvent (12.0, std::make_unique<Checkpoint> ("bypassed"));

    daw.addEvent (12.0, std::make_unique<Stop>());

    daw.run();

    // Verify each checkpoint captured different audio
    // (manual inspection or automated RMS comparison)
}
```

### Scenario 5: Reference Switch

```cpp
void runRefSwitchScenario (FakeDAW& daw)
{
    daw.addTrack ("Kick In", "input/lfwh_u87.wav");
    daw.addTrack ("Kick Out", "input/lfwh_sm57_front.wav");

    // Start with track 0 as reference
    daw.addEvent (0.0,  std::make_unique<SetReference> (0));
    daw.addEvent (7.5,  std::make_unique<TriggerAlign> (1));
    daw.addEvent (7.5,  std::make_unique<Checkpoint> ("aligned_to_track0"));

    // Switch reference to track 1
    daw.addEvent (8.0,  std::make_unique<ClearReference> (0));
    daw.addEvent (8.0,  std::make_unique<SetReference> (1));

    // Now track 0 aligns to track 1 (role reversal)
    daw.addEvent (15.5, std::make_unique<TriggerAlign> (0));
    daw.addEvent (15.5, std::make_unique<Checkpoint> ("track0_aligned_to_track1"));

    daw.addEvent (20.0, std::make_unique<Stop>());

    daw.run();
}
```

### Scenario 6: Edge Case - Insufficient Data

```cpp
void runInsufficientDataScenario (FakeDAW& daw)
{
    // Very short audio files (if you have them, or create synthetically)
    daw.addTrack ("Short 1", "input/short_hit.wav");
    daw.addTrack ("Short 2", "input/short_hit_delayed.wav");

    daw.addEvent (0.0, std::make_unique<SetReference> (0));
    daw.addEvent (0.5, std::make_unique<TriggerAlign> (1));  // Only 0.5s of data!
    daw.addEvent (1.0, std::make_unique<Stop>());

    daw.run();

    // Expect graceful handling (low correlation, warning in log, no crash)
}
```

---

## CLI Interface

```cpp
int main (int argc, char* argv[])
{
    ArgumentParser args (argc, argv);

    FakeDAW daw (args.getSampleRate(), args.getBlockSize());
    daw.getLogger().setVerbose (args.hasFlag ("--verbose"));

    if (args.hasOption ("--scenario"))
    {
        std::string scenario = args.getOption ("--scenario");

        if (scenario == "basic_align")
            runBasicAlignScenario (daw);
        else if (scenario == "delayed_ref")
            runDelayedRefScenario (daw);
        else if (scenario == "three_track")
            runThreeTrackScenario (daw);
        else if (scenario == "mode_switch")
            runModeSwitchScenario (daw);
        else if (scenario == "ref_switch")
            runRefSwitchScenario (daw);
        else if (scenario == "insufficient_data")
            runInsufficientDataScenario (daw);
        else
            std::cerr << "Unknown scenario: " << scenario << "\n";
    }
    else if (args.hasOption ("--run-all"))
    {
        // Run all scenarios as regression suite
        runAllScenarios();
    }
    else
    {
        printUsage();
    }

    return 0;
}
```

**Usage:**

```bash
# Run single scenario
./bin/FakeDAW --scenario basic_align --verbose

# Run all scenarios
./bin/FakeDAW --run-all --output output/regression/

# Custom sample rate / block size
./bin/FakeDAW --scenario basic_align --sample-rate 44100 --block-size 256
```

---

## Output Files

Each scenario run produces:

```
output/basic_align/
├── log.txt                      # Full timestamped log
├── track_0_input.wav            # Original input (for reference)
├── track_0_output.wav           # Output (should be unchanged for ref track)
├── track_1_input.wav            # Original target
├── track_1_output.wav           # Corrected target
├── sum_before.wav               # track_0 + track_1 (before correction)
├── sum_after.wav                # track_0 + track_1 (after correction)
├── analysis.json                # Results in machine-readable format
│   {
│     "track_1": {
│       "delay_samples": -49,
│       "delay_ms": -1.02,
│       "correlation": 0.89,
│       "polarity_inverted": false,
│       "coherence": 0.78,
│       "phase_deg": -8.1
│     }
│   }
└── report.txt                   # Human-readable summary
    =====================================
    FakeDAW Test Report: basic_align
    =====================================

    Tracks:
      0: Kick In (reference)
      1: Kick Out (aligned)

    Analysis Results (Track 1):
      Delay:       -49 samples (-1.02 ms)
      Correlation: 0.89
      Polarity:    Normal
      Coherence:   0.78
      Phase avg:   -8.1°

    Energy Comparison:
      Sum before:  -12.3 dB RMS
      Sum after:   -6.1 dB RMS
      Gain:        +6.2 dB  ✓

    Verification:
      [PASS] Delay within expected range
      [PASS] Correlation above threshold
      [PASS] Energy gain above threshold

    OVERALL: PASS
```

---

## Implementation Phases

### Phase 1: Basic Harness (Day 1)

**Goal:** Get the playback loop working with two tracks

- [ ] FakeDAW class skeleton
- [ ] Track class with audio loading
- [ ] Basic playback loop (processBlock calls)
- [ ] Console logging
- [ ] Save output WAVs

**Verify:** Can run two tracks through plugin processors, get output files

### Phase 2: Event System (Day 1-2)

**Goal:** Add timeline-based events

- [ ] EventTimeline class
- [ ] Action base class + all derived actions
- [ ] SetReference, TriggerAlign, SetCorrectionMode, SetBypass, Stop
- [ ] Checkpoint action

**Verify:** Can script the basic_align scenario

### Phase 3: Logging & Results (Day 2)

**Goal:** Comprehensive logging and result capture

- [ ] Logger class with categories
- [ ] Log file output
- [ ] Analysis results capture
- [ ] JSON output
- [ ] Human-readable report

**Verify:** Log shows exactly what's happening at each step

### Phase 4: Verification (Day 2-3)

**Goal:** Automated pass/fail checking

- [ ] ExpectedResults struct
- [ ] verify() method with tolerance checking
- [ ] Sum file generation (before/after)
- [ ] Energy comparison

**Verify:** Can automatically detect if alignment worked

### Phase 5: All Scenarios (Day 3)

**Goal:** Complete test coverage

- [ ] basic_align
- [ ] delayed_ref
- [ ] three_track
- [ ] mode_switch
- [ ] ref_switch
- [ ] insufficient_data
- [ ] Run-all mode

**Verify:** Full regression suite passing

### Phase 6: Debugging Integration (Optional)

**Goal:** Make debugging even easier

- [ ] Add logging callbacks to SharedState
- [ ] Add logging to STFTProcessor
- [ ] Add logging to PhaseAnalyzer
- [ ] Configurable verbosity levels

---

## CMake Integration

```cmake
# In CMakeLists.txt

if(MP_BUILD_CLI_TEST)
    # ... existing MagicPhaseTest and IPCTest ...

    # FakeDAW
    juce_add_console_app(FakeDAW
        PRODUCT_NAME "FakeDAW"
        COMPANY_NAME "MiGiC Music"
    )

    target_sources(FakeDAW PRIVATE
        src/Tools/FakeDAW/main.cpp
        src/Tools/FakeDAW/FakeDAW.cpp
        src/Tools/FakeDAW/Track.cpp
        src/Tools/FakeDAW/EventTimeline.cpp
        src/Tools/FakeDAW/Actions.cpp
        src/Tools/FakeDAW/Logger.cpp
        src/Tools/FakeDAW/Scenarios.cpp
        src/Tools/AudioFile.cpp
        # Include all plugin sources (same as MagicPhase target)
        src/PluginProcessor.cpp
        src/DSP/STFTProcessor.cpp
        src/DSP/PhaseAnalyzer.cpp
        src/DSP/PhaseCorrector.cpp
        src/IPC/SharedState.cpp
    )

    # Platform-specific
    if(WIN32)
        target_sources(FakeDAW PRIVATE src/IPC/PlatformSharedMemory_win32.cpp)
    else()
        target_sources(FakeDAW PRIVATE src/IPC/PlatformSharedMemory_posix.cpp)
    endif()

    target_include_directories(FakeDAW PRIVATE
        src
        src/Tools
        src/Tools/FakeDAW
    )

    target_link_libraries(FakeDAW PRIVATE
        juce::juce_core
        juce::juce_dsp
        juce::juce_audio_basics
        juce::juce_audio_processors  # Needed for AudioProcessor base class
        juce::juce_data_structures
        juce::juce_events
    )

    # Linux needs rt for shared memory
    if(MP_LINUX)
        target_link_libraries(FakeDAW PRIVATE Threads::Threads dl rt)
    endif()

    set_target_properties(FakeDAW PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
```

---

## Success Criteria

Before moving to Reaper, ALL of these must pass:

| Test | Criteria |
|------|----------|
| basic_align | Delay detected within 10 samples, correlation >0.7, energy gain >3dB |
| delayed_ref | Same as basic, proves late ref selection works |
| three_track | Both targets align correctly to same reference |
| mode_switch | Output differs between T+Phi, T-only, Phi-only, bypassed |
| ref_switch | Track can switch from target to reference role |
| insufficient_data | No crash, graceful handling, warning logged |

**If all scenarios pass, we have ~95% confidence the DAW will work.**

The remaining 5% is:
- GUI rendering (cosmetic, not functional)
- Real-time audio thread priority (shouldn't affect logic)
- DAW-specific quirks (edge cases we can't predict)

---

## Notes

### Why This Beats Real DAW Testing

1. **Instant feedback** - Run test, see results in <1 second
2. **Full logging** - See exactly what happened at each sample
3. **Reproducible** - Same test, same results, every time
4. **Debuggable** - Set breakpoints, step through code
5. **Scriptable** - Automate edge cases that are tedious to reproduce manually
6. **CI/CD ready** - Run on every commit, catch regressions early

### Common Issues This Will Catch

- Registration fails silently → Log shows "registered as slot -1"
- Frames not being written → Log shows "writeReferenceFrame" never called
- Frames overwritten too fast → Log shows frame count mismatch
- triggerAlign() returns early → Log shows no analysis results
- Correction not applied → Output RMS matches input RMS
- Mode switching broken → Checkpoints show identical output
- Memory corruption → Crash during run (with stack trace!)

---

## Next Steps

1. Review this plan
2. Start with Phase 1 (basic harness)
3. Iterate through phases
4. Run all scenarios
5. Fix any issues found
6. Only THEN test in Reaper

When ready to implement, we start with `src/Tools/FakeDAW/FakeDAW.cpp`.
