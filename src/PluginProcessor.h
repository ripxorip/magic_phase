#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <thread>
#include <mutex>
#include "STFTProcessor.h"
#include "PhaseAnalyzer.h"
#include "PhaseCorrector.h"
#include "SharedState.h"

// Alignment state machine (per UX contract)
enum class AlignmentState
{
    IDLE,       // Ready to start - shows "MAGIC ALIGN"
    WAITING,    // Accumulating audio - shows "▶ PLAY TRACKS X.Xs / 7.5s"
    ANALYZING,  // Running analysis - shows "ANALYZING..."
    ALIGNED,    // Correction active - shows "✓ ALIGNED"
    NO_REF      // No reference track set - shows "SET REF FIRST"
};

class MagicPhaseProcessor : public juce::AudioProcessor
{
public:
    MagicPhaseProcessor();
    ~MagicPhaseProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts; }

    SharedState& getSharedState() { return sharedState; }
    STFTProcessor& getSTFTProcessor() { return stftProcessor; }
    PhaseAnalyzer& getPhaseAnalyzer() { return phaseAnalyzer; }
    PhaseCorrector& getPhaseCorrector() { return phaseCorrector; }

    // Alignment state machine
    void startAlign();      // Called by GUI - transitions to WAITING
    void cancelAlign();     // Called by GUI - transitions back to IDLE
    void triggerAlign();    // Internal - runs the actual analysis

    AlignmentState getAlignmentState() const { return alignmentState.load(); }
    float getAccumulatedSeconds() const { return accumulatedSeconds.load(); }
    static constexpr float kRequiredSeconds = 7.5f;

    // Wait for background analysis to complete (for testing)
    void waitForAnalysis();

    void setIsReference (bool isRef);
    bool getIsReference() const { return isReference.load(); }
    void setCorrectionMode (int mode); // 0=T+Phi, 1=Phi, 2=T
    int getCorrectionMode() const { return correctionMode.load(); }
    void setBypass (bool bypassed);
    bool getBypassed() const { return isBypassed.load(); }

    int getMySlot() const { return mySlot; }
    juce::String getTrackName() const;

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    juce::AudioProcessorValueTreeState apvts;

    STFTProcessor stftProcessor;
    PhaseAnalyzer phaseAnalyzer;
    PhaseCorrector phaseCorrector;
    SharedState sharedState;

    std::atomic<bool> isReference { false };
    std::atomic<int> correctionMode { 0 }; // 0=T+Phi, 1=Phi, 2=T
    std::atomic<bool> isBypassed { false };

    // Alignment state machine
    std::atomic<AlignmentState> alignmentState { AlignmentState::IDLE };
    std::atomic<float> accumulatedSeconds { 0.0f };
    int accumulatedSamples = 0;
    std::atomic<bool> needsClear { false };  // Signal audio thread to clear frames
    AlignmentState lastSeenState { AlignmentState::IDLE };  // For detecting state changes on audio thread

    int mySlot = -1;
    double currentSampleRate = 44100.0;
    uint32_t lastSyncCounter = 0;  // Track sync requests from targets

    // Background analysis thread
    std::thread analysisThread;
    std::mutex analysisMutex;
    std::atomic<bool> analysisComplete { false };
    std::atomic<bool> shouldStopThread { false };

    // Pending results from background analysis (protected by analysisMutex)
    float pendingDelaySamples = 0.0f;
    bool pendingPolarityInvert = false;
    std::array<float, 2049> pendingPhaseCorrection {};

    void runAnalysisInBackground();
    void applyPendingResults();
    bool isDawPlaying() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MagicPhaseProcessor)
};
