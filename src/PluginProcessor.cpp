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

    if (isReference.load())
    {
        // Reference: write STFT frames to shared memory
        stftProcessor.processBlock (channelData, numSamples, [this] (std::complex<float>* frame, int numBins) {
            sharedState.writeReferenceFrame (frame, numBins);
        });
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
            // Accumulating audio - just process STFT without correction
            stftProcessor.processBlock (channelData, numSamples, nullptr);

            // Track accumulated time
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
    // Transition to WAITING state - start accumulating audio
    alignmentState.store (AlignmentState::WAITING);
    accumulatedSamples = 0;
    accumulatedSeconds.store (0.0f);

    // Clear accumulated frames for fresh analysis
    stftProcessor.clearAccumulatedFrames();
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
    // Read reference STFT frames from shared memory and run analysis
    int refSlot = sharedState.getReferenceSlot();
    if (refSlot < 0 || refSlot == mySlot)
    {
        // No valid reference - go back to IDLE
        alignmentState.store (AlignmentState::IDLE);
        return;
    }

    // Get coherence threshold and max correction from parameters
    float cohThreshold = apvts.getRawParameterValue ("coherenceThreshold")->load();
    float maxCorr = apvts.getRawParameterValue ("maxCorrection")->load();

    phaseAnalyzer.setCoherenceThreshold (cohThreshold);
    phaseAnalyzer.setMaxCorrectionDeg (maxCorr);

    // Analyze: accumulate phase differences from reference STFT frames
    auto refFrames = sharedState.readReferenceFrames();
    phaseAnalyzer.analyze (refFrames, stftProcessor.getAccumulatedFrames());

    // Transfer analysis results to corrector
    phaseCorrector.setDelaySamples (phaseAnalyzer.getDelaySamples());
    phaseCorrector.setPolarityInvert (phaseAnalyzer.getPolarityInverted());
    phaseCorrector.setPhaseCorrection (phaseAnalyzer.getPhaseCorrection());
    phaseCorrector.setTimeCorrectionOn (true);
    phaseCorrector.setPhaseCorrectionOn (true);

    // Update shared state
    if (mySlot >= 0)
        sharedState.setInstanceAligned (mySlot);

    // Transition to ALIGNED state - correction is now active
    alignmentState.store (AlignmentState::ALIGNED);
}

void MagicPhaseProcessor::setIsReference (bool isRef)
{
    isReference.store (isRef);
    if (isRef && mySlot >= 0)
        sharedState.setReferenceSlot (mySlot);
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
