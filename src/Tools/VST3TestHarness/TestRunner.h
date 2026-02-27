#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include "MockPlayHead.h"
#include "SharedState.h"
#include <vector>
#include <string>

struct TrackDefinition
{
    juce::String file;
    juce::String role;       // "reference" or "target"
    juce::String mode;       // "phi" or "t" (target tracks only)
};

struct TestDefinition
{
    juce::String name;
    juce::String description;
    juce::String pluginPath;
    double sampleRate = 48000.0;
    int bufferSize = 64;
    std::vector<int> bufferSizes;  // For sweep mode
    std::vector<TrackDefinition> tracks;
    juce::String outputDir;
    juce::String resultPath;
};

struct TrackResult
{
    juce::String name;
    juce::String role;
    juce::String mode;
    juce::String inputFile;
    juce::String outputFile;
    int slotIndex = -1;
    bool isAligned = false;
    juce::String alignmentState;
    float delaySamples = 0.0f;
    float delayMs = 0.0f;
    float correlation = 0.0f;
    float coherence = 0.0f;
    float phaseDegrees = 0.0f;
    bool polarityInverted = false;
    bool timeCorrectionOn = false;
    bool phaseCorrectionOn = false;
    float spectralBands[48] {};
};

struct TimingInfo
{
    double pluginLoadMs = 0.0;
    double prepareMs = 0.0;
    double playbackMs = 0.0;
    double analysisWaitMs = 0.0;
    double totalMs = 0.0;
};

struct TestResults
{
    std::vector<TrackResult> tracks;
    TimingInfo timing;
    int64_t refRawStartSample = 0;
    bool success = false;
    juce::String errorMessage;
};

class TestRunner
{
public:
    TestRunner();
    ~TestRunner();

    static TestDefinition loadTestDefinition (const juce::File& jsonFile);
    TestResults run (const TestDefinition& test);

private:
    MockPlayHead playHead;
    juce::AudioFormatManager audioFormatManager;

    juce::AudioBuffer<float> loadWavFile (const juce::String& path, double targetSampleRate);
    juce::AudioBuffer<float> getAudioChunk (const juce::AudioBuffer<float>& source,
                                             int blockIndex, int blockSize);

    void setPluginParameter (juce::AudioPluginInstance* instance,
                             const juce::String& paramId, float normalizedValue);
    void setPluginParameterByValue (juce::AudioPluginInstance* instance,
                                    const juce::String& paramId, float value);

    bool checkAllTargetsAligned (SharedState& sharedState, const TestDefinition& test);
    TestResults readResultsFromSharedMemory (SharedState& sharedState, const TestDefinition& test);
};
