#include "PluginProcessor.h"
#include "PluginEditor.h"

MagicPhaseProcessor::MagicPhaseProcessor()
    : AudioProcessor (BusesProperties()
                      .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                      .withOutput ("Output", juce::AudioChannelSet::mono(), true)),
      apvts (*this, nullptr, "Parameters", createParameterLayout())
{
}

MagicPhaseProcessor::~MagicPhaseProcessor()
{
    // Stop analysis thread if running
    shouldStopThread.store (true);
    if (analysisThread.joinable())
        analysisThread.join();

    if (mySlot >= 0)
        sharedState.deregisterInstance (mySlot);
}

juce::AudioProcessorValueTreeState::ParameterLayout MagicPhaseProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { "coherenceThreshold", 1 },
        "Coherence Threshold",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f),
        0.4f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { "maxCorrection", 1 },
        "Max Correction",
        juce::NormalisableRange<float> (0.0f, 180.0f, 1.0f),
        120.0f));

    return { params.begin(), params.end() };
}

void MagicPhaseProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;

    stftProcessor.prepare (sampleRate, samplesPerBlock);
    phaseAnalyzer.prepare (sampleRate);
    phaseCorrector.prepare (sampleRate);

    sharedState.initialize();
    mySlot = sharedState.registerInstance (getTrackName());
    sharedState.setSampleRate (static_cast<uint32_t> (sampleRate));
}

void MagicPhaseProcessor::releaseResources()
{
    if (mySlot >= 0)
    {
        sharedState.deregisterInstance (mySlot);
        mySlot = -1;
    }
}

bool MagicPhaseProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono())
        return false;

    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::mono())
        return false;

    return true;
}

void MagicPhaseProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    if (mySlot < 0)
        return;

    sharedState.updateHeartbeat (mySlot);

    auto* channelData = buffer.getWritePointer (0);
    const int numSamples = buffer.getNumSamples();

    // Check if background analysis completed and apply results
    if (analysisComplete.load())
    {
        applyPendingResults();
        analysisComplete.store (false);
        alignmentState.store (AlignmentState::ALIGNED);
    }

    // Detect state transition to WAITING and clear frames
    AlignmentState currentState = alignmentState.load();
    if (currentState == AlignmentState::WAITING && lastSeenState != AlignmentState::WAITING)
    {
        stftProcessor.clearAccumulatedFrames();
        needsClear.store (false);
    }
    lastSeenState = currentState;

    const bool playing = isDawPlaying();

    if (isReference.load())
    {
        // Check if a target requested sync (fresh capture)
        uint32_t currentSync = sharedState.getSyncCounter();
        if (currentSync != lastSyncCounter)
        {
            lastSyncCounter = currentSync;
            sharedState.clearReferenceBuffer();
        }

        // Reference: write STFT frames to shared memory (only when playing)
        if (playing)
        {
            stftProcessor.processBlock (channelData, numSamples, [this] (std::complex<float>* frame, int numBins) {
                sharedState.writeReferenceFrame (frame, numBins);
            });
        }
        else
        {
            stftProcessor.processBlock (channelData, numSamples, nullptr);
        }
        sharedState.setReferenceSlot (mySlot);
    }
    else if (isBypassed.load())
    {
        // Bypassed: just pass through, but still process STFT for frame accumulation
        stftProcessor.processBlock (channelData, numSamples, nullptr);
    }
    else
    {
        // Target track processing
        AlignmentState state = alignmentState.load();

        if (state == AlignmentState::WAITING)
        {
            // Only accumulate frames when DAW is playing
            stftProcessor.setAccumulateFrames (playing);
            stftProcessor.processBlock (channelData, numSamples, nullptr);

            if (playing)
            {
                accumulatedSamples += numSamples;
                float seconds = static_cast<float> (accumulatedSamples) / static_cast<float> (currentSampleRate);
                accumulatedSeconds.store (seconds);

                // Auto-trigger analysis when we have enough audio
                if (seconds >= kRequiredSeconds)
                {
                    alignmentState.store (AlignmentState::ANALYZING);
                    triggerAlign();
                }
            }
        }
        else if (state == AlignmentState::ALIGNED)
        {
            // Apply correction
            int mode = correctionMode.load();
            bool applyTime = (mode == 0 || mode == 2);
            bool applyPhase = (mode == 0 || mode == 1);

            stftProcessor.processBlock (channelData, numSamples, [this, applyTime, applyPhase] (std::complex<float>* frame, int numBins) {
                if (applyTime)
                    phaseCorrector.applyTimeCorrection (frame, numBins);
                if (applyPhase)
                    phaseCorrector.applyPhaseCorrection (frame, numBins);
            });
        }
        else
        {
            // IDLE or ANALYZING: pass through but still process STFT for frame accumulation
            stftProcessor.processBlock (channelData, numSamples, nullptr);
        }
    }

    // Update shared state with analysis data
    if (mySlot >= 0)
    {
        sharedState.updateInstanceData (mySlot,
            phaseAnalyzer.getDelaySamples(),
            phaseAnalyzer.getDelayMs(),
            phaseAnalyzer.getCorrelation(),
            phaseAnalyzer.getOverallCoherence(),
            phaseAnalyzer.getPhaseDegrees(),
            phaseAnalyzer.getPolarityInverted(),
            phaseCorrector.getTimeCorrectionOn(),
            phaseCorrector.getPhaseCorrectionOn(),
            phaseAnalyzer.getSpectralBands());
    }
}

juce::AudioProcessorEditor* MagicPhaseProcessor::createEditor()
{
    return new MagicPhaseEditor (*this);
}

void MagicPhaseProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void MagicPhaseProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml (getXmlFromBinary (data, sizeInBytes));
    if (xml != nullptr && xml->hasTagName (apvts.state.getType()))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

void MagicPhaseProcessor::startAlign()
{
    // Signal audio thread to clear frames (thread-safe)
    needsClear.store (true);

    // Signal reference track to also start fresh capture
    sharedState.requestSync();

    // Transition to WAITING state - start accumulating audio
    alignmentState.store (AlignmentState::WAITING);
    accumulatedSamples = 0;
    accumulatedSeconds.store (0.0f);
}

void MagicPhaseProcessor::cancelAlign()
{
    // Cancel alignment and go back to IDLE
    alignmentState.store (AlignmentState::IDLE);
    accumulatedSamples = 0;
    accumulatedSeconds.store (0.0f);
}

void MagicPhaseProcessor::triggerAlign()
{
    // Check for valid reference
    int refSlot = sharedState.getReferenceSlot();
    if (refSlot < 0)
    {
        // No reference track set
        alignmentState.store (AlignmentState::NO_REF);
        return;
    }
    if (refSlot == mySlot)
    {
        // This track IS the reference - can't align to self
        alignmentState.store (AlignmentState::IDLE);
        return;
    }

    // Stop any existing analysis thread
    shouldStopThread.store (true);
    if (analysisThread.joinable())
        analysisThread.join();

    // Start new background analysis
    shouldStopThread.store (false);
    analysisThread = std::thread (&MagicPhaseProcessor::runAnalysisInBackground, this);
}

void MagicPhaseProcessor::runAnalysisInBackground()
{
    // Get coherence threshold and max correction from parameters
    float cohThreshold = apvts.getRawParameterValue ("coherenceThreshold")->load();
    float maxCorr = apvts.getRawParameterValue ("maxCorrection")->load();

    phaseAnalyzer.setCoherenceThreshold (cohThreshold);
    phaseAnalyzer.setMaxCorrectionDeg (maxCorr);

    // Read reference frames and analyze
    auto refFrames = sharedState.readReferenceFrames();
    auto targetFrames = stftProcessor.getAccumulatedFrames();

    phaseAnalyzer.analyze (refFrames, targetFrames);

    if (shouldStopThread.load())
        return;

    // Store results in pending variables (protected by mutex)
    {
        std::lock_guard<std::mutex> lock (analysisMutex);
        pendingDelaySamples = phaseAnalyzer.getDelaySamples();
        pendingPolarityInvert = phaseAnalyzer.getPolarityInverted();

        const auto& phaseCorr = phaseAnalyzer.getPhaseCorrection();
        size_t copySize = std::min (phaseCorr.size(), pendingPhaseCorrection.size());
        std::copy (phaseCorr.begin(), phaseCorr.begin() + copySize, pendingPhaseCorrection.begin());
    }

    // Update shared state
    if (mySlot >= 0)
        sharedState.setInstanceAligned (mySlot);

    // Signal completion (audio thread will pick this up)
    analysisComplete.store (true);
}

void MagicPhaseProcessor::applyPendingResults()
{
    std::lock_guard<std::mutex> lock (analysisMutex);

    phaseCorrector.setDelaySamples (pendingDelaySamples);
    phaseCorrector.setPolarityInvert (pendingPolarityInvert);

    std::vector<float> phaseCorr (pendingPhaseCorrection.begin(), pendingPhaseCorrection.end());
    phaseCorrector.setPhaseCorrection (phaseCorr);
    phaseCorrector.setTimeCorrectionOn (true);
    phaseCorrector.setPhaseCorrectionOn (true);
}

bool MagicPhaseProcessor::isDawPlaying() const
{
    auto* playHead = getPlayHead();
    if (playHead == nullptr)
        return true; // Assume playing if no playhead (standalone, tests, etc.)

    auto position = playHead->getPosition();
    if (! position.hasValue())
        return true; // Assume playing if position unavailable

    return position->getIsPlaying();
}

void MagicPhaseProcessor::waitForAnalysis()
{
    // Wait for analysis thread to complete (for testing/FakeDAW)
    if (analysisThread.joinable())
        analysisThread.join();

    // Apply results immediately if available
    if (analysisComplete.load())
    {
        applyPendingResults();
        analysisComplete.store (false);
        alignmentState.store (AlignmentState::ALIGNED);
    }
}

void MagicPhaseProcessor::setIsReference (bool isRef)
{
    isReference.store (isRef);
    if (isRef && mySlot >= 0)
    {
        sharedState.setReferenceSlot (mySlot);
        lastSyncCounter = sharedState.getSyncCounter();
    }
}

void MagicPhaseProcessor::setCorrectionMode (int mode)
{
    correctionMode.store (mode);
}

void MagicPhaseProcessor::setBypass (bool bypassed)
{
    isBypassed.store (bypassed);
}

juce::String MagicPhaseProcessor::getTrackName() const
{
    return "Track " + juce::String (mySlot + 1);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MagicPhaseProcessor();
}
