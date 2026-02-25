#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"
#include "MainComponent.h"

class MagicPhaseEditor : public juce::AudioProcessorEditor
{
public:
    explicit MagicPhaseEditor (MagicPhaseProcessor&);
    ~MagicPhaseEditor() override = default;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    MagicPhaseProcessor& processorRef;
    MainComponent mainComponent;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MagicPhaseEditor)
};
