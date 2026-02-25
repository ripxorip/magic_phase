#include "TrackRow.h"

TrackRow::TrackRow()
{
    timeToggle.setClickingTogglesState (true);
    timeToggle.setToggleState (true, juce::dontSendNotification);
    timeToggle.onClick = [this] {
        if (onTimeCorrToggled)
            onTimeCorrToggled (timeToggle.getToggleState());
    };
    addAndMakeVisible (timeToggle);

    phaseToggle.setClickingTogglesState (true);
    phaseToggle.setToggleState (true, juce::dontSendNotification);
    phaseToggle.onClick = [this] {
        if (onPhaseCorrToggled)
            onPhaseCorrToggled (phaseToggle.getToggleState());
    };
    addAndMakeVisible (phaseToggle);
}

void TrackRow::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds().reduced (0, 3).toFloat();

    // Background bar
    g.setColour (MagicColors::trackBg);
    g.fillRoundedRectangle (bounds, 4.0f);

    // Border
    auto borderColour = data.isReference
        ? juce::Colour (255, 255, 255).withAlpha (0.25f)
        : (data.isAligned ? MagicColors::green.withAlpha (0.3f)
                          : juce::Colour (255, 255, 255).withAlpha (0.12f));
    g.setColour (borderColour);
    g.drawRoundedRectangle (bounds, 4.0f, 1.5f);

    // Spectral bands in background
    drawSpectralBands (g, bounds);

    auto contentBounds = bounds.reduced (14.0f, 0.0f);
    float y = contentBounds.getY();
    float h = contentBounds.getHeight();

    // Color pip (gold if this is OUR instance)
    float pipX = contentBounds.getX();
    float pipY = y + (h - 10.0f) / 2.0f;
    g.setColour (data.isThisInstance ? MagicColors::gold : MagicColors::text2);
    g.fillRoundedRectangle (pipX, pipY, 10.0f, 10.0f, 2.0f);

    // Track name
    float nameX = pipX + 20.0f;
    g.setColour (MagicColors::text);
    g.setFont (juce::Font (juce::FontOptions (13.0f).withStyle ("Bold")));
    g.drawText (data.name, juce::Rectangle<float> (nameX, y, 120.0f, h),
                juce::Justification::centredLeft);

    // REF badge
    if (data.isReference)
    {
        float badgeX = nameX + g.getCurrentFont().getStringWidthFloat (data.name) + 6.0f;
        auto badgeBounds = juce::Rectangle<float> (badgeX, y + (h - 14.0f) / 2.0f, 28.0f, 14.0f);
        g.setColour (MagicColors::gold);
        g.drawRoundedRectangle (badgeBounds, 2.0f, 1.5f);
        g.setFont (juce::Font (juce::FontOptions (7.0f).withStyle ("Bold")));
        g.drawText ("REF", badgeBounds, juce::Justification::centred);
    }

    // Stats (offset, phase)
    float statsX = contentBounds.getRight() - 310.0f;

    // Offset
    g.setColour (MagicColors::text3);
    g.setFont (juce::Font (juce::FontOptions (7.0f).withStyle ("Bold")));
    g.drawText ("OFFSET", juce::Rectangle<float> (statsX, y + 8.0f, 50.0f, 10.0f),
                juce::Justification::centredRight);
    g.setColour (MagicColors::text);
    g.setFont (juce::Font (juce::FontOptions (12.0f).withStyle ("Bold")));
    auto offsetStr = juce::String (data.offsetMs >= 0.0f ? "+" : "") + juce::String (data.offsetMs, 1) + "ms";
    g.drawText (offsetStr, juce::Rectangle<float> (statsX, y + 20.0f, 50.0f, 14.0f),
                juce::Justification::centredRight);

    // Phase
    float phaseX = statsX + 66.0f;
    g.setColour (MagicColors::text3);
    g.setFont (juce::Font (juce::FontOptions (7.0f).withStyle ("Bold")));
    g.drawText ("PHASE", juce::Rectangle<float> (phaseX, y + 8.0f, 50.0f, 10.0f),
                juce::Justification::centredRight);
    g.setColour (MagicColors::text);
    g.setFont (juce::Font (juce::FontOptions (12.0f).withStyle ("Bold")));
    auto phaseStr = juce::String (data.phaseDeg >= 0.0f ? "+" : "") + juce::String (static_cast<int> (data.phaseDeg)) + juce::String::fromUTF8 ("\xc2\xb0");
    g.drawText (phaseStr, juce::Rectangle<float> (phaseX, y + 20.0f, 50.0f, 14.0f),
                juce::Justification::centredRight);

    // Correlation value
    float corrX = phaseX + 66.0f;
    auto corrColour = data.correlation > 0.9f ? MagicColors::green
                    : data.correlation > 0.7f ? MagicColors::yellow
                    : MagicColors::red;
    g.setColour (corrColour);
    g.setFont (juce::Font (juce::FontOptions (16.0f).withStyle ("Bold")));
    g.drawText (juce::String (data.correlation, 2),
                juce::Rectangle<float> (corrX, y, 44.0f, h),
                juce::Justification::centredRight);
}

void TrackRow::resized()
{
    auto bounds = getLocalBounds().reduced (0, 3);
    auto area = bounds.reduced (14, 0);

    // Toggles at far right
    int toggleW = 28;
    int toggleH = 16;
    int toggleY = area.getCentreY() - toggleH / 2;

    int rightEdge = area.getRight() - 4;

    phaseToggle.setBounds (rightEdge - toggleW, toggleY, toggleW, toggleH);
    timeToggle.setBounds (rightEdge - toggleW * 2 - 20, toggleY, toggleW, toggleH);
}

void TrackRow::updateData (const TrackRowData& newData)
{
    data = newData;
    timeToggle.setToggleState (data.timeCorrOn, juce::dontSendNotification);
    phaseToggle.setToggleState (data.phaseCorrOn, juce::dontSendNotification);
    repaint();
}

void TrackRow::drawSpectralBands (juce::Graphics& g, juce::Rectangle<float> area)
{
    const int numBands = 48;
    float bandW = area.getWidth() / numBands;

    for (int i = 0; i < numBands; ++i)
    {
        float val = data.spectralBands[i];
        float barH = val * area.getHeight() * 0.95f;
        float x = area.getX() + i * bandW;
        float y = area.getBottom() - barH;

        // Gradient bar
        float alpha = 0.15f + val * 0.2f;
        g.setColour (MagicColors::text2.withAlpha (alpha));
        g.fillRect (x + 1.0f, y, bandW - 2.0f, barH);

        // Top cap
        if (val > 0.08f)
        {
            g.setColour (MagicColors::text2.withAlpha (0.3f + val * 0.2f));
            g.fillRect (x + 1.0f, y, bandW - 2.0f, 2.0f);
        }
    }
}
