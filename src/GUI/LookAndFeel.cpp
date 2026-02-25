#include "LookAndFeel.h"

MagicPhaseLookAndFeel::MagicPhaseLookAndFeel()
{
    setColour (juce::ResizableWindow::backgroundColourId, MagicColors::bg);
    setColour (juce::TextButton::buttonColourId, MagicColors::surface);
    setColour (juce::TextButton::textColourOffId, MagicColors::text2);
    setColour (juce::TextButton::textColourOnId, MagicColors::bg);
}

void MagicPhaseLookAndFeel::drawButtonBackground (juce::Graphics& g, juce::Button& button,
                                                   const juce::Colour& backgroundColour,
                                                   bool isHighlighted, bool isDown)
{
    auto bounds = button.getLocalBounds().toFloat().reduced (0.5f);
    auto cornerSize = 3.0f;

    auto baseColour = backgroundColour;
    if (isDown)
        baseColour = baseColour.darker (0.1f);
    else if (isHighlighted)
        baseColour = baseColour.brighter (0.05f);

    g.setColour (baseColour);
    g.fillRoundedRectangle (bounds, cornerSize);

    g.setColour (juce::Colour (255, 255, 255).withAlpha (0.12f));
    g.drawRoundedRectangle (bounds, cornerSize, 1.5f);
}

void MagicPhaseLookAndFeel::drawButtonText (juce::Graphics& g, juce::TextButton& button,
                                            bool /*isHighlighted*/, bool /*isDown*/)
{
    auto font = getTextButtonFont (button, button.getHeight());
    g.setFont (font);
    g.setColour (button.findColour (button.getToggleState()
                    ? juce::TextButton::textColourOnId
                    : juce::TextButton::textColourOffId));

    auto textArea = button.getLocalBounds();
    g.drawFittedText (button.getButtonText(), textArea, juce::Justification::centred, 1);
}

juce::Font MagicPhaseLookAndFeel::getTextButtonFont (juce::TextButton&, int buttonHeight)
{
    return juce::Font (juce::FontOptions (std::min (11.0f, static_cast<float> (buttonHeight) * 0.6f))
                           .withStyle ("Bold"));
}
