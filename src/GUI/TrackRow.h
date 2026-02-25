#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "LookAndFeel.h"

struct TrackRowData
{
    juce::String name;
    bool isReference = false;
    bool isThisInstance = false;  // True if this row represents THIS plugin instance
    bool isActive = false;
    bool isAligned = false;
    float offsetMs = 0.0f;
    float phaseDeg = 0.0f;
    float correlation = 0.0f;
    bool timeCorrOn = true;
    bool phaseCorrOn = true;
    float spectralBands[48] {};
};

class TrackRow : public juce::Component
{
public:
    TrackRow();

    void paint (juce::Graphics&) override;
    void resized() override;

    void updateData (const TrackRowData& data);
    const TrackRowData& getData() const { return data; }

    std::function<void (bool)> onTimeCorrToggled;
    std::function<void (bool)> onPhaseCorrToggled;
    std::function<void()> onSetAsReference;

private:
    void drawSpectralBands (juce::Graphics& g, juce::Rectangle<float> area);

    TrackRowData data;

    juce::TextButton timeToggle { "T" };
    juce::TextButton phaseToggle { juce::String::fromUTF8 ("\xce\xa6") }; // Phi

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TrackRow)
};
