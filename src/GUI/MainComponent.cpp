#include "MainComponent.h"
#include "PluginProcessor.h"

MainComponent::MainComponent (MagicPhaseProcessor& p)
    : processor (p)
{
    setLookAndFeel (&lookAndFeel);

    // Coherence display
    coherenceLabel.setFont (juce::Font (juce::FontOptions (16.0f).withStyle ("Bold")));
    coherenceLabel.setColour (juce::Label::textColourId, MagicColors::green);
    coherenceLabel.setJustificationType (juce::Justification::centredRight);
    coherenceLabel.setText ("0.00", juce::dontSendNotification);
    addAndMakeVisible (coherenceLabel);

    // Track viewport
    trackViewport.setViewedComponent (&trackListContainer, false);
    trackViewport.setScrollBarsShown (true, false);
    trackViewport.getVerticalScrollBar().setColour (juce::ScrollBar::thumbColourId,
                                                     juce::Colour (255, 255, 255).withAlpha (0.12f));
    addAndMakeVisible (trackViewport);

    // REF button (toggle)
    refButton.setClickingTogglesState (true);
    refButton.setColour (juce::TextButton::buttonColourId, MagicColors::surface);
    refButton.setColour (juce::TextButton::buttonOnColourId, MagicColors::gold);
    refButton.setColour (juce::TextButton::textColourOffId, MagicColors::text2);
    refButton.setColour (juce::TextButton::textColourOnId, MagicColors::bg);
    refButton.onClick = [this] {
        processor.setIsReference (refButton.getToggleState());
    };
    addAndMakeVisible (refButton);

    // Align button
    alignButton.setColour (juce::TextButton::buttonColourId, MagicColors::gold);
    alignButton.setColour (juce::TextButton::textColourOffId, MagicColors::bg);
    alignButton.onClick = [this] { onAlignClicked(); };
    addAndMakeVisible (alignButton);

    // Mode buttons
    auto setupModeBtn = [this] (juce::TextButton& btn, int mode) {
        btn.setClickingTogglesState (false);
        btn.onClick = [this, mode] { onModeClicked (mode); };
        addAndMakeVisible (btn);
    };
    setupModeBtn (modeTPhi, 0);
    setupModeBtn (modePhi, 1);
    setupModeBtn (modeT, 2);

    // A/B button
    abButton.setClickingTogglesState (true);
    abButton.onClick = [this] { processor.setBypass (abButton.getToggleState()); };
    addAndMakeVisible (abButton);

    // Initial mode state
    onModeClicked (0);

    // Timer for GUI refresh (30Hz)
    startTimerHz (30);
}

MainComponent::~MainComponent()
{
    setLookAndFeel (nullptr);
}

void MainComponent::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds();

    // Background
    g.fillAll (MagicColors::bg);

    // === Header ===
    auto headerArea = bounds.removeFromTop (48);
    g.setColour (MagicColors::bg);
    g.fillRect (headerArea);

    // Gold line under header
    g.setColour (MagicColors::gold);
    g.fillRect (headerArea.getX(), headerArea.getBottom() - 2, headerArea.getWidth(), 2);

    // Brand block "PKG"
    auto brandBlock = headerArea.reduced (20, 10);
    g.setColour (MagicColors::gold);
    g.fillRoundedRectangle (brandBlock.getX(), brandBlock.getY(), 36.0f, 22.0f, 2.0f);
    g.setColour (MagicColors::bg);
    g.setFont (juce::Font (juce::FontOptions (10.0f).withStyle ("Bold")));
    g.drawText ("PKG", juce::Rectangle<float> (brandBlock.getX(), brandBlock.getY(), 36.0f, 22.0f),
                juce::Justification::centred);

    // "MAGIC PHASE" title
    g.setColour (MagicColors::text);
    g.setFont (juce::Font (juce::FontOptions (18.0f).withStyle ("Bold")));
    g.drawText ("MAGIC PHASE",
                juce::Rectangle<float> (brandBlock.getX() + 46.0f, brandBlock.getY(), 200.0f, 22.0f),
                juce::Justification::centredLeft);

    // Coherence chip background
    auto cohArea = headerArea.reduced (20, 8).removeFromRight (180);
    g.setColour (MagicColors::surface);
    g.fillRoundedRectangle (cohArea.toFloat(), 3.0f);
    g.setColour (juce::Colour (255, 255, 255).withAlpha (0.12f));
    g.drawRoundedRectangle (cohArea.toFloat(), 3.0f, 1.5f);

    g.setColour (MagicColors::text3);
    g.setFont (juce::Font (juce::FontOptions (8.0f).withStyle ("Bold")));
    g.drawText ("COHERENCE", cohArea.reduced (12, 0), juce::Justification::centredLeft);

    // === Bottom border ===
    auto bottomArea = bounds.removeFromBottom (68);
    g.setColour (juce::Colour (255, 255, 255).withAlpha (0.1f));
    g.fillRect (bounds.getX(), bottomArea.getY(), bounds.getWidth(), 1);
}

void MainComponent::resized()
{
    auto bounds = getLocalBounds();

    // Header
    auto headerArea = bounds.removeFromTop (48);

    // Coherence label position (right side of header)
    auto cohLabelArea = headerArea.reduced (20, 8).removeFromRight (80);
    coherenceLabel.setBounds (cohLabelArea);

    // Bottom bar
    auto bottomArea = bounds.removeFromBottom (68);
    auto bottomInner = bottomArea.reduced (20, 12);

    refButton.setBounds (bottomInner.removeFromLeft (52).reduced (0, 6));
    bottomInner.removeFromLeft (12);  // spacing
    alignButton.setBounds (bottomInner.removeFromLeft (160).reduced (0, 0));

    auto rightGroup = bottomInner.removeFromRight (200);
    int btnW = 42;
    int btnH = 34;
    int y = rightGroup.getCentreY() - btnH / 2;

    abButton.setBounds (rightGroup.getRight() - btnW, y, btnW, btnH);
    modeT.setBounds (rightGroup.getRight() - btnW - 48, y, btnW, btnH);
    modePhi.setBounds (rightGroup.getRight() - btnW - 96, y, btnW, btnH);
    modeTPhi.setBounds (rightGroup.getRight() - btnW - 148, y, 52, btnH);

    // Track viewport
    auto trackArea = bounds.reduced (0, 4);
    trackViewport.setBounds (trackArea);

    // Layout track rows
    int rowH = 54;
    int totalH = static_cast<int> (trackRows.size()) * (rowH + 6);
    trackListContainer.setBounds (0, 0, trackArea.getWidth(), std::max (totalH, trackArea.getHeight()));

    for (int i = 0; i < trackRows.size(); ++i)
    {
        trackRows[i]->setBounds (20, i * (rowH + 6), trackArea.getWidth() - 40, rowH);
    }
}

void MainComponent::timerCallback()
{
    refreshTrackList();
}

void MainComponent::refreshTrackList()
{
    auto& sharedState = processor.getSharedState();
    if (! sharedState.isInitialized())
        return;

    auto* header = sharedState.getHeader();
    if (header == nullptr)
        return;

    // Check for changes
    uint32_t currentVersion = header->version.load();
    int refSlot = header->referenceSlot.load();
    int mySlot = processor.getMySlot();

    // Sync REF button state
    bool amIReference = (mySlot >= 0 && mySlot == refSlot);
    if (refButton.getToggleState() != amIReference)
        refButton.setToggleState (amIReference, juce::dontSendNotification);

    // Count active slots
    int numActive = 0;
    for (int i = 0; i < kMaxInstances; ++i)
    {
        auto* slot = sharedState.getSlot (i);
        if (slot != nullptr && slot->active.load() > 0)
            numActive++;
    }

    // Rebuild track rows if count changed
    if (numActive != trackRows.size() || currentVersion != lastVersion)
    {
        lastVersion = currentVersion;

        // Ensure we have the right number of rows
        while (trackRows.size() > numActive)
            trackRows.removeLast();

        while (trackRows.size() < numActive)
        {
            auto* row = trackRows.add (new TrackRow());
            trackListContainer.addAndMakeVisible (row);
        }

        // Update row data
        int rowIdx = 0;
        for (int i = 0; i < kMaxInstances && rowIdx < numActive; ++i)
        {
            auto* slot = sharedState.getSlot (i);
            if (slot == nullptr || slot->active.load() == 0)
                continue;

            TrackRowData data;
            data.name = juce::String (slot->trackName);
            data.isReference = (i == refSlot);
            data.isThisInstance = (i == mySlot);
            data.isActive = true;
            data.isAligned = (slot->active.load() == 2);
            data.offsetMs = slot->delayMs;
            data.phaseDeg = slot->phaseDegrees;
            data.correlation = slot->correlation;
            data.timeCorrOn = slot->timeCorrectionOn;
            data.phaseCorrOn = slot->phaseCorrectionOn;
            std::memcpy (data.spectralBands, slot->spectralBands, sizeof (data.spectralBands));

            trackRows[rowIdx]->updateData (data);
            rowIdx++;
        }

        resized();
    }

    // Update coherence display
    float overallCoh = 0.0f;
    int cohCount = 0;
    for (int i = 0; i < kMaxInstances; ++i)
    {
        auto* slot = sharedState.getSlot (i);
        if (slot != nullptr && slot->active.load() > 0 && i != refSlot)
        {
            overallCoh += slot->overallCoherence;
            cohCount++;
        }
    }
    if (cohCount > 0)
        overallCoh /= cohCount;

    coherenceLabel.setText (juce::String (overallCoh, 2), juce::dontSendNotification);
}

void MainComponent::onAlignClicked()
{
    processor.triggerAlign();

    // Visual feedback
    alignButton.setButtonText ("ALIGNING...");
    alignButton.setColour (juce::TextButton::buttonColourId, MagicColors::surface);
    alignButton.setColour (juce::TextButton::textColourOffId, MagicColors::gold);

    // Reset button text after a delay
    juce::Timer::callAfterDelay (2000, [this] {
        alignButton.setButtonText ("MAGIC ALIGN");
        alignButton.setColour (juce::TextButton::buttonColourId, MagicColors::gold);
        alignButton.setColour (juce::TextButton::textColourOffId, MagicColors::bg);
    });
}

void MainComponent::onModeClicked (int mode)
{
    activeMode = mode;
    processor.setCorrectionMode (mode);

    auto setActive = [this] (juce::TextButton& btn, bool active) {
        btn.setColour (juce::TextButton::buttonColourId,
                       active ? MagicColors::gold : MagicColors::surface);
        btn.setColour (juce::TextButton::textColourOffId,
                       active ? MagicColors::bg : MagicColors::text2);
    };

    setActive (modeTPhi, mode == 0);
    setActive (modePhi, mode == 1);
    setActive (modeT, mode == 2);
}
