#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include "LookAndFeel.h"
#include "TrackRow.h"

class MagicPhaseProcessor;

class MainComponent : public juce::Component,
                      public juce::Timer
{
public:
    explicit MainComponent (MagicPhaseProcessor& processor);
    ~MainComponent() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    void refreshTrackList();
    void onAlignClicked();
    void onModeClicked (int mode);

    MagicPhaseProcessor& processor;
    MagicPhaseLookAndFeel lookAndFeel;

    // Header
    juce::Label coherenceLabel;

    // Track list
    juce::Viewport trackViewport;
    juce::Component trackListContainer;
    juce::OwnedArray<TrackRow> trackRows;

    // Bottom bar
    juce::TextButton refButton { "REF" };
    juce::TextButton alignButton { "MAGIC ALIGN" };
    juce::TextButton modeTPhi { juce::String::fromUTF8 ("T+\xce\xa6") };
    juce::TextButton modePhi { juce::String::fromUTF8 ("\xce\xa6") };
    juce::TextButton modeT { "T" };
    juce::TextButton abButton { "A/B" };

    int activeMode = 0; // 0=T+Phi, 1=Phi, 2=T
    uint32_t lastVersion = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};
