#include "PluginEditor.h"

MagicPhaseEditor::MagicPhaseEditor (MagicPhaseProcessor& p)
    : AudioProcessorEditor (&p),
      processorRef (p),
      mainComponent (p)
{
    setSize (740, 520);
    addAndMakeVisible (mainComponent);
}

void MagicPhaseEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff2a2a2e));
}

void MagicPhaseEditor::resized()
{
    mainComponent.setBounds (getLocalBounds());
}
