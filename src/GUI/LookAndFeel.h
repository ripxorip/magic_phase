#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

namespace MagicColors
{
    static const juce::Colour bg        { 0xff2a2a2e };
    static const juce::Colour surface   { 0xff323237 };
    static const juce::Colour surface2  { 0xff3a3a40 };
    static const juce::Colour surface3  { 0xff44444b };
    static const juce::Colour border    { juce::Colour (255, 255, 255).withAlpha (0.06f) };
    static const juce::Colour text      { 0xffeaeaec };
    static const juce::Colour text2     { 0xff8e8e96 };
    static const juce::Colour text3     { 0xff5a5a63 };
    static const juce::Colour gold      { 0xffe8e8ec };
    static const juce::Colour goldBright { 0xffffffff };
    static const juce::Colour green     { 0xff3ecf8e };
    static const juce::Colour blue      { 0xff5b8def };
    static const juce::Colour pink      { 0xffe8618c };
    static const juce::Colour yellow    { 0xffedc039 };
    static const juce::Colour red       { 0xffef4444 };
    static const juce::Colour trackBg   { 0xff28282c };
}

class MagicPhaseLookAndFeel : public juce::LookAndFeel_V4
{
public:
    MagicPhaseLookAndFeel();

    void drawButtonBackground (juce::Graphics&, juce::Button&, const juce::Colour&,
                               bool isHighlighted, bool isDown) override;

    void drawButtonText (juce::Graphics&, juce::TextButton&,
                         bool isHighlighted, bool isDown) override;

    juce::Font getTextButtonFont (juce::TextButton&, int buttonHeight) override;
};
