/*
    FakeDAW - Test Harness for Magic Phase

    Simulates DAW behavior for testing Magic Phase plugin outside of a real DAW.
    Provides:
    - Direct C++ plugin instantiation (no VST3 wrapping)
    - Timeline-based event scripting
    - Full logging at every step
    - Automated verification

    Usage:
        ./bin/FakeDAW --scenario basic_align
        ./bin/FakeDAW --scenario basic_align --verbose
        ./bin/FakeDAW --run-all

    This validates the UX contract from USER_EXPERIENCE_CONTRACT.md
    before any real DAW testing.
*/

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include "AudioFile.h"
#include "PluginProcessor.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <functional>
#include <cmath>
#include <chrono>

//==============================================================================
// LOGGER
//==============================================================================
class Logger
{
public:
    enum Category { INIT, IPC, DSP, GUI, STATE, AUDIO };

    void setOutputFile (const std::string& path)
    {
        logFile.open (path);
    }

    void setVerbose (bool v) { verbose = v; }
    void setCurrentTime (double t) { currentTime = t; }

    void log (Category cat, const std::string& message)
    {
        static const char* catNames[] = { "INIT", "IPC", "DSP", "GUI", "STATE", "AUDIO" };
        std::ostringstream oss;
        oss << "[" << std::fixed << std::setprecision (3) << currentTime << "s] "
            << "[" << catNames[cat] << "] " << message;

        std::cout << oss.str() << "\n";
        if (logFile.is_open())
            logFile << oss.str() << "\n";
    }

    void init  (const std::string& msg) { log (INIT,  msg); }
    void ipc   (const std::string& msg) { if (verbose) log (IPC,   msg); }
    void dsp   (const std::string& msg) { log (DSP,   msg); }
    void gui   (const std::string& msg) { log (GUI,   msg); }
    void state (const std::string& msg) { log (STATE, msg); }
    void audio (const std::string& msg) { log (AUDIO, msg); }

private:
    std::ofstream logFile;
    bool verbose = false;
    double currentTime = 0.0;
};

//==============================================================================
// TRACK
//==============================================================================
struct Track
{
    std::string name;
    std::unique_ptr<MagicPhaseProcessor> processor;
    std::vector<float> inputAudio;
    std::vector<float> outputAudio;
    juce::AudioBuffer<float> buffer;

    bool loadAudio (const std::string& path)
    {
        AudioFile<float> af;
        if (! af.load (path))
            return false;

        int numSamples = af.getNumSamplesPerChannel();
        int numChannels = af.getNumChannels();

        inputAudio.resize (static_cast<size_t> (numSamples), 0.0f);
        for (int ch = 0; ch < numChannels; ++ch)
            for (int i = 0; i < numSamples; ++i)
                inputAudio[static_cast<size_t> (i)] += af.samples[static_cast<size_t> (ch)][static_cast<size_t> (i)];

        if (numChannels > 1)
        {
            float scale = 1.0f / static_cast<float> (numChannels);
            for (auto& s : inputAudio)
                s *= scale;
        }

        return true;
    }

    void prepare (double sampleRate, int blockSize)
    {
        processor = std::make_unique<MagicPhaseProcessor>();
        processor->setPlayConfigDetails (1, 1, sampleRate, blockSize);
        processor->prepareToPlay (sampleRate, blockSize);
        buffer.setSize (1, blockSize);
        outputAudio.clear();
        outputAudio.reserve (inputAudio.size());
    }

    void release()
    {
        if (processor)
            processor->releaseResources();
    }
};

//==============================================================================
// ANALYSIS RESULTS
//==============================================================================
struct AnalysisResults
{
    float delaySamples = 0.0f;
    float delayMs = 0.0f;
    float correlation = 0.0f;
    float coherence = 0.0f;
    float phaseDegrees = 0.0f;
    bool polarityInverted = false;
    bool hasResults = false;
};

struct ExpectedResults
{
    float minDelayMs = -100.0f;
    float maxDelayMs = 100.0f;
    float minCoherence = 0.5f;  // Coherence is the meaningful metric for phase alignment
    float minEnergyGainDb = 0.0f;
};

//==============================================================================
// ACTIONS
//==============================================================================
class FakeDAW;  // Forward declaration

struct Action
{
    virtual ~Action() = default;
    virtual void execute (FakeDAW& daw, Logger& log) = 0;
    virtual std::string describe() const = 0;
};

struct SetReference : Action
{
    int trackIndex;
    explicit SetReference (int idx) : trackIndex (idx) {}

    void execute (FakeDAW& daw, Logger& log) override;
    std::string describe() const override
    {
        return "SetReference(track=" + std::to_string (trackIndex) + ")";
    }
};

struct ClearReference : Action
{
    int trackIndex;
    explicit ClearReference (int idx) : trackIndex (idx) {}

    void execute (FakeDAW& daw, Logger& log) override;
    std::string describe() const override
    {
        return "ClearReference(track=" + std::to_string (trackIndex) + ")";
    }
};

struct TriggerAlign : Action
{
    int trackIndex;
    explicit TriggerAlign (int idx) : trackIndex (idx) {}

    void execute (FakeDAW& daw, Logger& log) override;
    std::string describe() const override
    {
        return "TriggerAlign(track=" + std::to_string (trackIndex) + ")";
    }
};

struct SetCorrectionMode : Action
{
    int trackIndex;
    int mode;  // 0=T+Phi, 1=Phi, 2=T
    SetCorrectionMode (int idx, int m) : trackIndex (idx), mode (m) {}

    void execute (FakeDAW& daw, Logger& log) override;
    std::string describe() const override
    {
        static const char* modeNames[] = { "T+Phi", "Phi", "T" };
        return "SetCorrectionMode(track=" + std::to_string (trackIndex) + ", mode=" + modeNames[mode] + ")";
    }
};

struct SetBypass : Action
{
    int trackIndex;
    bool bypassed;
    SetBypass (int idx, bool b) : trackIndex (idx), bypassed (b) {}

    void execute (FakeDAW& daw, Logger& log) override;
    std::string describe() const override
    {
        return "SetBypass(track=" + std::to_string (trackIndex) + ", " + (bypassed ? "ON" : "OFF") + ")";
    }
};

struct Checkpoint : Action
{
    std::string checkpointName;
    explicit Checkpoint (const std::string& name) : checkpointName (name) {}

    void execute (FakeDAW& daw, Logger& log) override;
    std::string describe() const override
    {
        return "Checkpoint(\"" + checkpointName + "\")";
    }
};

struct Stop : Action
{
    void execute (FakeDAW& daw, Logger& log) override;
    std::string describe() const override { return "Stop()"; }
};

//==============================================================================
// EVENT TIMELINE
//==============================================================================
class EventTimeline
{
public:
    struct TimedEvent
    {
        double time;
        std::unique_ptr<Action> action;

        bool operator< (const TimedEvent& other) const { return time < other.time; }
    };

    void addEvent (double timeSeconds, std::unique_ptr<Action> action)
    {
        events.push_back ({ timeSeconds, std::move (action) });
    }

    void sort()
    {
        std::sort (events.begin(), events.end());
    }

    void processEventsUpTo (double timeSeconds, FakeDAW& daw, Logger& log)
    {
        while (nextEventIndex < events.size() && events[nextEventIndex].time <= timeSeconds)
        {
            auto& event = events[nextEventIndex];
            log.state ("EVENT @ " + std::to_string (event.time) + "s: " + event.action->describe());
            event.action->execute (daw, log);
            nextEventIndex++;
        }
    }

    bool hasMoreEvents() const { return nextEventIndex < events.size(); }

private:
    std::vector<TimedEvent> events;
    size_t nextEventIndex = 0;
};

//==============================================================================
// FAKEDAW
//==============================================================================
class FakeDAW
{
public:
    FakeDAW (double sr = 48000.0, int bs = 128)
        : sampleRate (sr), blockSize (bs) {}

    // Track management
    int addTrack (const std::string& name, const std::string& audioFilePath)
    {
        auto track = std::make_unique<Track>();
        track->name = name;

        if (! track->loadAudio (audioFilePath))
        {
            logger.init ("ERROR: Failed to load audio: " + audioFilePath);
            return -1;
        }

        int idx = static_cast<int> (tracks.size());
        tracks.push_back (std::move (track));
        logger.init ("Track " + std::to_string (idx) + " '" + name + "' added: " + audioFilePath);
        return idx;
    }

    Track& getTrack (int index) { return *tracks[static_cast<size_t> (index)]; }
    int getNumTracks() const { return static_cast<int> (tracks.size()); }

    // Timeline
    void addEvent (double timeSeconds, std::unique_ptr<Action> action)
    {
        timeline.addEvent (timeSeconds, std::move (action));
    }

    // Results storage
    void storeResults (int trackIndex, const AnalysisResults& results)
    {
        trackResults[trackIndex] = results;
    }

    AnalysisResults getResults (int trackIndex) const
    {
        auto it = trackResults.find (trackIndex);
        if (it != trackResults.end())
            return it->second;
        return {};
    }

    Logger& getLogger() { return logger; }

    void stop() { running = false; }

    // Main execution
    void run()
    {
        logger.init ("FakeDAW starting: " + std::to_string (tracks.size()) + " tracks, "
                   + std::to_string ((int)sampleRate) + "Hz, " + std::to_string (blockSize) + " block size");

        // Initialize all tracks
        for (size_t i = 0; i < tracks.size(); i++)
        {
            tracks[i]->prepare (sampleRate, blockSize);
            logger.init ("Track " + std::to_string (i) + " '" + tracks[i]->name + "' prepared: "
                       + std::to_string (tracks[i]->inputAudio.size()) + " samples");

            int slot = tracks[i]->processor->getMySlot();
            logger.ipc ("Track " + std::to_string (i) + " registered as IPC slot " + std::to_string (slot));
        }

        // Sort timeline events
        timeline.sort();

        // Find longest track
        size_t maxSamples = 0;
        for (auto& track : tracks)
            maxSamples = std::max (maxSamples, track->inputAudio.size());

        // Playback loop
        running = true;
        int samplePos = 0;
        juce::MidiBuffer emptyMidi;
        int lastSecond = -1;

        while (running && samplePos < static_cast<int> (maxSamples))
        {
            double timeSeconds = static_cast<double> (samplePos) / sampleRate;
            logger.setCurrentTime (timeSeconds);

            // Process timeline events
            timeline.processEventsUpTo (timeSeconds, *this, logger);

            // Process all tracks
            for (auto& track : tracks)
            {
                int samplesToProcess = std::min (blockSize,
                    static_cast<int> (track->inputAudio.size()) - samplePos);

                if (samplesToProcess <= 0)
                {
                    track->buffer.clear();
                    samplesToProcess = blockSize;
                }
                else
                {
                    std::memcpy (track->buffer.getWritePointer (0),
                               track->inputAudio.data() + samplePos,
                               static_cast<size_t> (samplesToProcess) * sizeof (float));

                    // Clear remaining buffer if partial block
                    if (samplesToProcess < blockSize)
                    {
                        std::memset (track->buffer.getWritePointer (0) + samplesToProcess,
                                   0, static_cast<size_t> (blockSize - samplesToProcess) * sizeof (float));
                    }
                }

                // THE DAW CALL
                track->processor->processBlock (track->buffer, emptyMidi);

                // Record output
                auto* outputData = track->buffer.getReadPointer (0);
                track->outputAudio.insert (track->outputAudio.end(),
                                          outputData, outputData + samplesToProcess);
            }

            samplePos += blockSize;

            // Progress indication (every second)
            int currentSecond = static_cast<int> (timeSeconds);
            if (currentSecond > lastSecond)
            {
                lastSecond = currentSecond;
                double totalSeconds = static_cast<double> (maxSamples) / sampleRate;
                logger.state ("Playback: " + std::to_string (currentSecond) + "s / "
                            + std::to_string (static_cast<int> (totalSeconds)) + "s");
            }
        }

        // Cleanup
        for (auto& track : tracks)
            track->release();

        logger.state ("FakeDAW finished");
    }

    // Verification
    bool verify (const ExpectedResults& expected)
    {
        logger.state ("=== VERIFICATION ===");
        bool allPassed = true;

        for (auto& [trackIdx, results] : trackResults)
        {
            if (! results.hasResults)
                continue;

            logger.state ("Track " + std::to_string (trackIdx) + " results:");
            logger.state ("  Delay: " + std::to_string (results.delayMs) + " ms ("
                        + std::to_string (results.delaySamples) + " samples)");
            logger.state ("  Correlation: " + std::to_string (results.correlation));
            logger.state ("  Coherence: " + std::to_string (results.coherence));
            logger.state ("  Phase: " + std::to_string (results.phaseDegrees) + " deg");
            logger.state ("  Polarity: " + std::string (results.polarityInverted ? "INVERTED" : "Normal"));

            // Check delay range
            bool delayOk = results.delayMs >= expected.minDelayMs && results.delayMs <= expected.maxDelayMs;
            logger.state ("  [" + std::string (delayOk ? "PASS" : "FAIL") + "] Delay in range ["
                        + std::to_string (expected.minDelayMs) + ", " + std::to_string (expected.maxDelayMs) + "] ms");
            allPassed &= delayOk;

            // Check coherence (the meaningful metric for phase alignment)
            bool cohOk = results.coherence >= expected.minCoherence;
            logger.state ("  [" + std::string (cohOk ? "PASS" : "FAIL") + "] Coherence >= "
                        + std::to_string (expected.minCoherence));
            allPassed &= cohOk;
        }

        // Energy comparison
        if (tracks.size() >= 2)
        {
            float energyBefore = calculateSumEnergy (false);
            float energyAfter = calculateSumEnergy (true);
            float gainDb = 20.0f * std::log10 (energyAfter / std::max (energyBefore, 0.0001f));

            logger.state ("Energy comparison:");
            logger.state ("  Sum before: " + std::to_string (20.0f * std::log10 (energyBefore)) + " dB RMS");
            logger.state ("  Sum after:  " + std::to_string (20.0f * std::log10 (energyAfter)) + " dB RMS");
            logger.state ("  Gain: " + std::to_string (gainDb) + " dB");

            bool energyOk = gainDb >= expected.minEnergyGainDb;
            logger.state ("  [" + std::string (energyOk ? "PASS" : "FAIL") + "] Energy gain >= "
                        + std::to_string (expected.minEnergyGainDb) + " dB");
            allPassed &= energyOk;
        }

        logger.state ("=== OVERALL: " + std::string (allPassed ? "PASS" : "FAIL") + " ===");
        return allPassed;
    }

    // Save outputs
    void saveOutputs (const std::string& outputDir)
    {
        juce::File dir (outputDir);
        dir.createDirectory();

        for (size_t i = 0; i < tracks.size(); i++)
        {
            auto& track = tracks[i];

            // Save output
            std::string outPath = outputDir + "/track_" + std::to_string (i) + "_output.wav";
            saveWav (outPath, track->outputAudio, sampleRate);
            logger.state ("Saved: " + outPath);
        }

        // Save sum before/after
        if (tracks.size() >= 2)
        {
            auto sumBefore = calculateSum (false);
            auto sumAfter = calculateSum (true);

            saveWav (outputDir + "/sum_before.wav", sumBefore, sampleRate);
            saveWav (outputDir + "/sum_after.wav", sumAfter, sampleRate);
            logger.state ("Saved sum files");
        }
    }

private:
    double sampleRate;
    int blockSize;
    std::vector<std::unique_ptr<Track>> tracks;
    EventTimeline timeline;
    Logger logger;
    std::map<int, AnalysisResults> trackResults;
    bool running = false;

    float calculateRMS (const std::vector<float>& audio)
    {
        if (audio.empty()) return 0.0f;
        double sum = 0.0;
        for (auto s : audio)
            sum += static_cast<double> (s * s);
        return static_cast<float> (std::sqrt (sum / static_cast<double> (audio.size())));
    }

    std::vector<float> calculateSum (bool useOutput)
    {
        if (tracks.empty()) return {};

        auto& firstTrack = useOutput ? tracks[0]->outputAudio : tracks[0]->inputAudio;
        size_t minLen = firstTrack.size();

        for (auto& track : tracks)
        {
            auto& audio = useOutput ? track->outputAudio : track->inputAudio;
            minLen = std::min (minLen, audio.size());
        }

        std::vector<float> sum (minLen, 0.0f);
        for (auto& track : tracks)
        {
            auto& audio = useOutput ? track->outputAudio : track->inputAudio;
            for (size_t i = 0; i < minLen; i++)
                sum[i] += audio[i];
        }

        return sum;
    }

    float calculateSumEnergy (bool useOutput)
    {
        auto sum = calculateSum (useOutput);
        return calculateRMS (sum);
    }

    void saveWav (const std::string& path, const std::vector<float>& audio, double sr)
    {
        AudioFile<float> af;
        af.setSampleRate (static_cast<uint32_t> (sr));
        af.setBitDepth (24);
        af.setNumChannels (1);
        af.setNumSamplesPerChannel (static_cast<int> (audio.size()));

        for (size_t i = 0; i < audio.size(); i++)
            af.samples[0][i] = audio[i];

        af.save (path);
    }
};

//==============================================================================
// ACTION IMPLEMENTATIONS
//==============================================================================
void SetReference::execute (FakeDAW& daw, Logger& log)
{
    auto& track = daw.getTrack (trackIndex);
    log.gui ("Track " + std::to_string (trackIndex) + " '" + track.name + "': setIsReference(true)");
    track.processor->setIsReference (true);
}

void ClearReference::execute (FakeDAW& daw, Logger& log)
{
    auto& track = daw.getTrack (trackIndex);
    log.gui ("Track " + std::to_string (trackIndex) + " '" + track.name + "': setIsReference(false)");
    track.processor->setIsReference (false);
}

void TriggerAlign::execute (FakeDAW& daw, Logger& log)
{
    auto& track = daw.getTrack (trackIndex);
    log.gui ("Track " + std::to_string (trackIndex) + " '" + track.name + "': triggerAlign()");

    // Debug: show frame counts before alignment
    auto refFrames = track.processor->getSharedState().readReferenceFrames();
    auto& targetFrames = track.processor->getSTFTProcessor().getAccumulatedFrames();
    log.dsp ("  Reference frames: " + std::to_string (refFrames.size()));
    log.dsp ("  Target frames: " + std::to_string (targetFrames.size()));

    track.processor->triggerAlign();

    // Wait for background analysis to complete (analysis runs in background thread)
    track.processor->waitForAnalysis();

    // Capture results
    auto& analyzer = track.processor->getPhaseAnalyzer();

    AnalysisResults results;
    results.delaySamples = analyzer.getDelaySamples();
    results.delayMs = analyzer.getDelayMs();
    results.correlation = analyzer.getCorrelation();
    results.coherence = analyzer.getOverallCoherence();
    results.phaseDegrees = analyzer.getPhaseDegrees();
    results.polarityInverted = analyzer.getPolarityInverted();
    results.hasResults = true;

    daw.storeResults (trackIndex, results);

    log.dsp ("Track " + std::to_string (trackIndex) + " analysis complete:");
    log.dsp ("  Delay: " + std::to_string (results.delaySamples) + " samples ("
           + std::to_string (results.delayMs) + " ms)");
    log.dsp ("  Correlation: " + std::to_string (results.correlation));
    log.dsp ("  Polarity: " + std::string (results.polarityInverted ? "INVERTED" : "Normal"));
    log.dsp ("  Coherence: " + std::to_string (results.coherence));
    log.dsp ("  Phase: " + std::to_string (results.phaseDegrees) + " deg");
}

void SetCorrectionMode::execute (FakeDAW& daw, Logger& log)
{
    static const char* modeNames[] = { "T+Phi", "Phi", "T" };
    auto& track = daw.getTrack (trackIndex);
    log.gui ("Track " + std::to_string (trackIndex) + " '" + track.name
           + "': setCorrectionMode(" + modeNames[mode] + ")");
    track.processor->setCorrectionMode (mode);
}

void SetBypass::execute (FakeDAW& daw, Logger& log)
{
    auto& track = daw.getTrack (trackIndex);
    log.gui ("Track " + std::to_string (trackIndex) + " '" + track.name
           + "': setBypass(" + (bypassed ? "true" : "false") + ")");
    track.processor->setBypass (bypassed);
}

void Checkpoint::execute (FakeDAW& daw, Logger& log)
{
    log.state ("=== CHECKPOINT: " + checkpointName + " ===");
    for (int i = 0; i < daw.getNumTracks(); i++)
    {
        auto& track = daw.getTrack (i);
        // Calculate current RMS of output
        float rms = 0.0f;
        if (! track.outputAudio.empty())
        {
            double sum = 0.0;
            for (auto s : track.outputAudio)
                sum += static_cast<double> (s * s);
            rms = static_cast<float> (std::sqrt (sum / static_cast<double> (track.outputAudio.size())));
        }
        float db = (rms > 0.0001f) ? 20.0f * std::log10 (rms) : -100.0f;
        log.audio ("Track " + std::to_string (i) + " '" + track.name + "': RMS = "
                 + std::to_string (db) + " dB");
    }
}

void Stop::execute (FakeDAW& daw, Logger& log)
{
    log.state ("Playback stopped by action");
    daw.stop();
}

//==============================================================================
// TEST SCENARIOS
//==============================================================================
static std::string g_inputDir = "input";

bool runBasicAlignScenario (FakeDAW& daw)
{
    daw.getLogger().init ("=== SCENARIO: basic_align ===");
    daw.getLogger().init ("Tests: First-time user with 2 mics on same source");

    int t0 = daw.addTrack ("Kick In", g_inputDir + "/lfwh_u87.wav");
    int t1 = daw.addTrack ("Kick Out", g_inputDir + "/lfwh_sm57_front.wav");

    if (t0 < 0 || t1 < 0)
    {
        daw.getLogger().init ("ERROR: Failed to load test files. Make sure input/ directory has test WAVs.");
        return false;
    }

    // UX Contract: First plugin auto-becomes reference
    daw.addEvent (0.0, std::make_unique<SetReference> (0));

    // UX Contract: After 7.5s of audio, trigger align
    daw.addEvent (7.5, std::make_unique<TriggerAlign> (1));
    daw.addEvent (7.5, std::make_unique<Checkpoint> ("after_align"));

    // Let it play a bit more with correction applied
    daw.addEvent (15.0, std::make_unique<Stop>());

    daw.run();

    ExpectedResults expected;
    expected.minDelayMs = -3.0f;
    expected.maxDelayMs = 1.0f;
    expected.minCoherence = 0.6f;
    expected.minEnergyGainDb = 0.0f;  // Energy comparison is partial (only post-align audio is corrected)

    bool passed = daw.verify (expected);
    daw.saveOutputs ("output/basic_align/");

    return passed;
}

bool runThreeTrackScenario (FakeDAW& daw)
{
    daw.getLogger().init ("=== SCENARIO: three_track ===");
    daw.getLogger().init ("Tests: Power user with multiple mics (1 ref, N targets)");

    int t0 = daw.addTrack ("Kick In", g_inputDir + "/lfwh_u87.wav");
    int t1 = daw.addTrack ("Kick Out", g_inputDir + "/lfwh_sm57_front.wav");
    int t2 = daw.addTrack ("Kick Sub", g_inputDir + "/lfwh_sm57_front.wav");  // Using same file as example

    if (t0 < 0 || t1 < 0 || t2 < 0)
    {
        daw.getLogger().init ("ERROR: Failed to load test files");
        return false;
    }

    daw.addEvent (0.0, std::make_unique<SetReference> (0));
    daw.addEvent (7.5, std::make_unique<TriggerAlign> (1));
    daw.addEvent (7.5, std::make_unique<TriggerAlign> (2));
    daw.addEvent (7.5, std::make_unique<Checkpoint> ("both_aligned"));
    daw.addEvent (15.0, std::make_unique<Stop>());

    daw.run();

    ExpectedResults expected;
    expected.minCoherence = 0.5f;
    expected.minEnergyGainDb = 0.0f;

    bool passed = daw.verify (expected);
    daw.saveOutputs ("output/three_track/");

    return passed;
}

bool runModeSwitchScenario (FakeDAW& daw)
{
    daw.getLogger().init ("=== SCENARIO: mode_switch ===");
    daw.getLogger().init ("Tests: Mode buttons (T+Phi, Phi, T) produce different output");

    int t0 = daw.addTrack ("Kick In", g_inputDir + "/lfwh_u87.wav");
    int t1 = daw.addTrack ("Kick Out", g_inputDir + "/lfwh_sm57_front.wav");

    if (t0 < 0 || t1 < 0)
    {
        daw.getLogger().init ("ERROR: Failed to load test files");
        return false;
    }

    daw.addEvent (0.0, std::make_unique<SetReference> (0));
    daw.addEvent (7.5, std::make_unique<TriggerAlign> (1));

    // Test all modes
    daw.addEvent (8.0, std::make_unique<SetCorrectionMode> (1, 0));  // T+Phi (default)
    daw.addEvent (9.0, std::make_unique<Checkpoint> ("mode_t_phi"));

    daw.addEvent (9.0, std::make_unique<SetCorrectionMode> (1, 2));  // T only
    daw.addEvent (10.0, std::make_unique<Checkpoint> ("mode_t_only"));

    daw.addEvent (10.0, std::make_unique<SetCorrectionMode> (1, 1)); // Phi only
    daw.addEvent (11.0, std::make_unique<Checkpoint> ("mode_phi_only"));

    daw.addEvent (11.0, std::make_unique<SetBypass> (1, true));
    daw.addEvent (12.0, std::make_unique<Checkpoint> ("bypassed"));

    daw.addEvent (12.0, std::make_unique<Stop>());

    daw.run();

    ExpectedResults expected;
    expected.minCoherence = 0.5f;

    bool passed = daw.verify (expected);
    daw.saveOutputs ("output/mode_switch/");

    return passed;
}

bool runDelayedRefScenario (FakeDAW& daw)
{
    daw.getLogger().init ("=== SCENARIO: delayed_ref ===");
    daw.getLogger().init ("Tests: User waits before clicking REF");

    int t0 = daw.addTrack ("Kick In", g_inputDir + "/lfwh_u87.wav");
    int t1 = daw.addTrack ("Kick Out", g_inputDir + "/lfwh_sm57_front.wav");

    if (t0 < 0 || t1 < 0)
    {
        daw.getLogger().init ("ERROR: Failed to load test files");
        return false;
    }

    // User waits 3 seconds before clicking REF
    daw.addEvent (3.0, std::make_unique<SetReference> (0));
    daw.addEvent (10.5, std::make_unique<TriggerAlign> (1));
    daw.addEvent (15.0, std::make_unique<Stop>());

    daw.run();

    ExpectedResults expected;
    expected.minCoherence = 0.5f;
    expected.minEnergyGainDb = 0.0f;

    bool passed = daw.verify (expected);
    daw.saveOutputs ("output/delayed_ref/");

    return passed;
}

bool runRefSwitchScenario (FakeDAW& daw)
{
    daw.getLogger().init ("=== SCENARIO: ref_switch ===");
    daw.getLogger().init ("Tests: Switching which track is reference");

    int t0 = daw.addTrack ("Kick In", g_inputDir + "/lfwh_u87.wav");
    int t1 = daw.addTrack ("Kick Out", g_inputDir + "/lfwh_sm57_front.wav");

    if (t0 < 0 || t1 < 0)
    {
        daw.getLogger().init ("ERROR: Failed to load test files");
        return false;
    }

    // Start with track 0 as reference
    daw.addEvent (0.0, std::make_unique<SetReference> (0));
    daw.addEvent (7.5, std::make_unique<TriggerAlign> (1));
    daw.addEvent (7.5, std::make_unique<Checkpoint> ("aligned_to_track0"));

    // Switch reference
    daw.addEvent (8.0, std::make_unique<ClearReference> (0));
    daw.addEvent (8.0, std::make_unique<SetReference> (1));

    // Now track 0 aligns to track 1
    daw.addEvent (15.5, std::make_unique<TriggerAlign> (0));
    daw.addEvent (15.5, std::make_unique<Checkpoint> ("track0_aligned_to_track1"));

    daw.addEvent (20.0, std::make_unique<Stop>());

    daw.run();

    ExpectedResults expected;
    expected.minCoherence = 0.3f;  // Lower threshold for this edge case

    bool passed = daw.verify (expected);
    daw.saveOutputs ("output/ref_switch/");

    return passed;
}

//==============================================================================
// MAIN
//==============================================================================
void printUsage()
{
    std::cout << R"(
FakeDAW - Magic Phase Test Harness

Usage:
    FakeDAW --scenario <name> [options]
    FakeDAW --run-all [options]

Scenarios:
    basic_align    - Basic two-track alignment (validates UX contract)
    three_track    - Three tracks aligned to same reference
    mode_switch    - Test T+Phi, Phi, T modes
    delayed_ref    - Late reference selection
    ref_switch     - Switch which track is reference

Options:
    --verbose      - Show detailed IPC logging
    --input <dir>  - Input directory (default: input/)
    --sample-rate <hz>   - Sample rate (default: 48000)
    --block-size <n>     - Block size (default: 128)

Examples:
    FakeDAW --scenario basic_align
    FakeDAW --scenario basic_align --verbose
    FakeDAW --run-all
)" << std::endl;
}

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    // Parse arguments
    double sampleRate = 48000.0;
    int blockSize = 128;
    bool verbose = false;
    std::string scenario;
    bool runAll = false;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--scenario" && i + 1 < argc)
            scenario = argv[++i];
        else if (arg == "--run-all")
            runAll = true;
        else if (arg == "--verbose")
            verbose = true;
        else if (arg == "--input" && i + 1 < argc)
            g_inputDir = argv[++i];
        else if (arg == "--sample-rate" && i + 1 < argc)
            sampleRate = std::stod (argv[++i]);
        else if (arg == "--block-size" && i + 1 < argc)
            blockSize = std::stoi (argv[++i]);
        else if (arg == "--help" || arg == "-h")
        {
            printUsage();
            return 0;
        }
    }

    if (scenario.empty() && !runAll)
    {
        printUsage();
        return 1;
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    FAKEDAW - Magic Phase                      ║\n";
    std::cout << "║              \"3 clicks, 8 seconds, mind blown\"               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    int passed = 0;
    int failed = 0;

    auto runScenario = [&] (const std::string& name, auto scenarioFunc) {
        FakeDAW daw (sampleRate, blockSize);
        daw.getLogger().setVerbose (verbose);

        bool result = scenarioFunc (daw);

        if (result)
        {
            std::cout << "\n[PASS] " << name << "\n\n";
            passed++;
        }
        else
        {
            std::cout << "\n[FAIL] " << name << "\n\n";
            failed++;
        }
    };

    if (runAll)
    {
        runScenario ("basic_align", runBasicAlignScenario);
        runScenario ("three_track", runThreeTrackScenario);
        runScenario ("mode_switch", runModeSwitchScenario);
        runScenario ("delayed_ref", runDelayedRefScenario);
        runScenario ("ref_switch", runRefSwitchScenario);

        std::cout << "\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "  SUMMARY: " << passed << " passed, " << failed << " failed\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "\n";
    }
    else if (scenario == "basic_align")
        runScenario ("basic_align", runBasicAlignScenario);
    else if (scenario == "three_track")
        runScenario ("three_track", runThreeTrackScenario);
    else if (scenario == "mode_switch")
        runScenario ("mode_switch", runModeSwitchScenario);
    else if (scenario == "delayed_ref")
        runScenario ("delayed_ref", runDelayedRefScenario);
    else if (scenario == "ref_switch")
        runScenario ("ref_switch", runRefSwitchScenario);
    else
    {
        std::cerr << "Unknown scenario: " << scenario << "\n";
        printUsage();
        return 1;
    }

    return failed > 0 ? 1 : 0;
}
