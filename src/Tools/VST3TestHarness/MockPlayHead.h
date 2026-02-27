#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

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
    int64_t getCurrentSample() const { return currentSample; }

private:
    int64_t currentSample = 0;
    double sampleRate = 48000.0;
    bool playing = false;
};
