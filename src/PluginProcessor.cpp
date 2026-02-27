#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <fstream>
#include <ctime>
#include <cmath>

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

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { "triggerAlign", 1 },
        "Trigger Align",
        false));

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { "isReference", 1 },
        "Is Reference",
        false));

    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID { "correctionMode", 1 },
        "Correction Mode",
        0, 1, 1));

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { "bypass", 1 },
        "Bypass",
        false));

    return { params.begin(), params.end() };
}

void MagicPhaseProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    totalSamplesProcessed = 0;

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

    // Sync APVTS parameters → internal state
    if (apvts.getRawParameterValue ("triggerAlign")->load() > 0.5f)
    {
        apvts.getParameter ("triggerAlign")->setValueNotifyingHost (0.0f);
        startAlign();
    }

    bool paramIsRef = apvts.getRawParameterValue ("isReference")->load() > 0.5f;
    if (paramIsRef != isReference.load())
        setIsReference (paramIsRef);

    int paramMode = static_cast<int> (apvts.getRawParameterValue ("correctionMode")->load());
    if (paramMode != correctionMode.load())
        setCorrectionMode (paramMode);

    bool paramBypass = apvts.getRawParameterValue ("bypass")->load() > 0.5f;
    if (paramBypass != isBypassed.load())
        setBypass (paramBypass);

    auto* channelData = buffer.getWritePointer (0);
    const int numSamples = buffer.getNumSamples();
    totalSamplesProcessed += numSamples;

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
            sharedState.clearRawSampleBuffer();

            // Record playhead position and acknowledge sync
            sharedState.acknowledgeSyncRequest (getPlayheadSamplePos());
        }

        // Reference: write STFT frames + raw samples to shared memory (only when playing)
        if (playing)
        {
            sharedState.writeRawSamples (channelData, numSamples);
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
            if (playing)
            {
                // Wait for reference to acknowledge sync before accumulating raw samples
                if (! rawAccumActive)
                {
                    if (sharedState.isSyncAcknowledged (pendingSyncCounter))
                    {
                        rawAccumActive = true;
                        targetRawStartSample = getPlayheadSamplePos();
                    }
                }

                // Accumulate raw samples BEFORE STFT (which modifies channelData in-place)
                if (rawAccumActive)
                {
                    size_t space = static_cast<size_t> (kMaxLocalRawSamples) - localRawSamples.size();
                    int toAdd = std::min (numSamples, static_cast<int> (space));
                    if (toAdd > 0)
                        localRawSamples.insert (localRawSamples.end(), channelData, channelData + toAdd);
                }
            }

            // STFT processBlock modifies channelData via overlap-add
            stftProcessor.setAccumulateFrames (playing);
            stftProcessor.processBlock (channelData, numSamples, nullptr);

            if (playing && rawAccumActive)
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
            // Apply correction: T always on, phase only in Φ mode
            int mode = correctionMode.load();
            bool applyTime = true;           // T is ALWAYS applied
            bool applyPhase = (mode == 1);   // Phase only in Φ mode

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

    // Update shared state with stored analysis results
    if (mySlot >= 0)
    {
        sharedState.updateInstanceData (mySlot,
            resultDelaySamples.load(),
            resultDelaySubSample.load(),
            resultDelayMs.load(),
            resultCorrelation.load(),
            resultCoherence.load(),
            resultPhaseDeg.load(),
            resultPolarityInv.load(),
            phaseCorrector.getTimeCorrectionOn(),
            phaseCorrector.getPhaseCorrectionOn(),
            resultSpectralBands);
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

    // Clear local raw sample buffer
    localRawSamples.clear();
    rawAccumActive = false;
    targetRawStartSample = 0;

    // Signal reference track to also start fresh capture
    sharedState.requestSync();
    pendingSyncCounter = sharedState.getSyncCounter();

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

    // =========================================================================
    // Stage 1: Read raw samples
    // =========================================================================
    int analyzeWindowSamples = static_cast<int> (kRequiredSeconds * currentSampleRate);
    auto refRaw = sharedState.readRawSamples (analyzeWindowSamples);
    auto tarRaw = localRawSamples;  // copy from local accumulation

    if (refRaw.empty() || tarRaw.empty())
    {
        alignmentState.store (AlignmentState::IDLE);
        return;
    }

    // Stop accumulating STFT frames
    stftProcessor.setAccumulateFrames (false);

    if (shouldStopThread.load())
        return;

    // =========================================================================
    // Stage 2a: Time-domain delay detection on raw samples
    //
    // rawDelay = trueDelay + syncOffset
    // We know syncOffset from playhead timestamps recorded during handshake.
    // trueDelay = rawDelay - syncOffset
    // =========================================================================
    PhaseAnalyzer rawDelayAnalyzer;
    rawDelayAnalyzer.prepare (currentSampleRate);
    rawDelayAnalyzer.detectDelayTimeDomain (refRaw, tarRaw);

    float rawDelay = rawDelayAnalyzer.getDelaySamples();
    float rawDelaySubSample = rawDelayAnalyzer.getDelaySubSample();
    float rawCorr = rawDelayAnalyzer.getCorrelation();
    bool rawPolarityInv = rawDelayAnalyzer.getPolarityInverted();

    // Compute sync offset from playhead timestamps
    int64_t refStart = sharedState.getRefRawStartSample();
    int64_t syncOffset = targetRawStartSample - refStart;

    // True delay = raw delay minus the sync offset between when the two plugins
    // started accumulating. This is the actual audio delay we need to correct.
    float trueDelay = rawDelay - static_cast<float> (syncOffset);
    float trueDelaySubSample = rawDelaySubSample - static_cast<float> (syncOffset);
    float trueDelayMs = (trueDelaySubSample / static_cast<float> (currentSampleRate)) * 1000.0f;

    if (shouldStopThread.load())
        return;

    // =========================================================================
    // Stage 2b: Integer-shift + polarity-flip target to create time-aligned version
    // Uses rawDelay (includes sync offset) because refRaw and tarRaw have that offset
    // =========================================================================
    int delayInt = static_cast<int> (std::round (rawDelay));
    float polaritySign = rawPolarityInv ? -1.0f : 1.0f;
    int tarLen = static_cast<int> (tarRaw.size());

    std::vector<float> tarAligned (static_cast<size_t> (tarLen), 0.0f);
    for (int i = 0; i < tarLen; ++i)
    {
        int srcIdx = i + delayInt;
        if (srcIdx >= 0 && srcIdx < tarLen)
            tarAligned[static_cast<size_t> (i)] = tarRaw[static_cast<size_t> (srcIdx)] * polaritySign;
    }

    if (shouldStopThread.load())
        return;

    // =========================================================================
    // Stage 2c: Re-STFT ref + aligned-target, then analyzeSpectralPhase
    // =========================================================================
    constexpr int kBlockSize = 128;

    STFTProcessor refSTFT2;
    STFTProcessor corrSTFT2;
    refSTFT2.prepare (currentSampleRate, kBlockSize);
    corrSTFT2.prepare (currentSampleRate, kBlockSize);

    std::vector<std::vector<std::complex<float>>> refFrames2;
    std::vector<std::vector<std::complex<float>>> corrFrames2;

    auto collectRef = [&refFrames2] (std::complex<float>* frame, int numBins) {
        refFrames2.emplace_back (frame, frame + numBins);
    };
    auto collectCorr = [&corrFrames2] (std::complex<float>* frame, int numBins) {
        corrFrames2.emplace_back (frame, frame + numBins);
    };

    int analyzeLen = static_cast<int> (std::min (refRaw.size(), tarAligned.size()));
    for (int pos = 0; pos < analyzeLen; pos += kBlockSize)
    {
        if (shouldStopThread.load())
            return;

        int thisBlock = std::min (kBlockSize, analyzeLen - pos);
        refSTFT2.processBlock (refRaw.data() + pos, thisBlock, collectRef);
        corrSTFT2.processBlock (tarAligned.data() + pos, thisBlock, collectCorr);
    }

    PhaseAnalyzer spectralAnalyzer;
    spectralAnalyzer.prepare (currentSampleRate);
    spectralAnalyzer.setCoherenceThreshold (cohThreshold);
    spectralAnalyzer.setMaxCorrectionDeg (maxCorr);
    spectralAnalyzer.analyzeSpectralPhase (refFrames2, corrFrames2);

    if (shouldStopThread.load())
        return;

    float coherence = spectralAnalyzer.getOverallCoherence();
    float phaseDeg = spectralAnalyzer.getPhaseDegrees();
    const auto& phaseCorr = spectralAnalyzer.getPhaseCorrection();

    // Store display values for GUI (atomic, safe to read from any thread)
    resultDelaySamples.store (trueDelay);
    resultDelaySubSample.store (trueDelaySubSample);
    resultDelayMs.store (trueDelayMs);
    resultCorrelation.store (rawCorr);
    resultCoherence.store (coherence);
    resultPhaseDeg.store (phaseDeg);
    resultPolarityInv.store (rawPolarityInv);
    std::memcpy (resultSpectralBands, spectralAnalyzer.getSpectralBands(), sizeof (resultSpectralBands));

    // =========================================================================
    // Diagnostic logging
    // =========================================================================
    {
        auto desktopPath = juce::File::getSpecialLocation (juce::File::userDesktopDirectory)
                               .getChildFile ("magic_phase_log.txt");

        std::ofstream logFile (desktopPath.getFullPathName().toStdString(), std::ios::app);
        if (logFile.is_open())
        {
            std::time_t now = std::time (nullptr);
            char timeBuf[64];
            std::strftime (timeBuf, sizeof (timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime (&now));

            float avgPhaseDeg = 0.0f;
            if (! phaseCorr.empty())
            {
                float sum = 0.0f;
                for (auto v : phaseCorr)
                    sum += v;
                avgPhaseDeg = (sum / static_cast<float> (phaseCorr.size())) * 180.0f / 3.14159265f;
            }

            auto computeRms = [] (const std::vector<float>& buf) -> float {
                if (buf.empty()) return 0.0f;
                double sum = 0.0;
                for (auto s : buf) sum += static_cast<double> (s) * s;
                return static_cast<float> (std::sqrt (sum / buf.size()));
            };

            logFile << "[" << timeBuf << "] === Magic Phase Analysis ===\n"
                    << "  Ref raw samples: " << refRaw.size()
                    << " (" << (static_cast<float> (refRaw.size()) / static_cast<float> (currentSampleRate)) << "s)"
                    << " RMS=" << computeRms (refRaw) << "\n"
                    << "  Target raw samples: " << tarRaw.size()
                    << " (" << (static_cast<float> (tarRaw.size()) / static_cast<float> (currentSampleRate)) << "s)"
                    << " RMS=" << computeRms (tarRaw) << "\n"
                    << "  Raw delay: " << rawDelay << " samples corr=" << rawCorr
                    << " pol=" << (rawPolarityInv ? "INV" : "N") << "\n"
                    << "  Sync offset: " << syncOffset << " samples"
                    << " (refStart=" << refStart << " targetStart=" << targetRawStartSample << ")\n"
                    << "  True delay: " << trueDelay << " samples (" << trueDelayMs << " ms)\n"
                    << "  Re-STFT frames: " << refFrames2.size() << " ref, " << corrFrames2.size() << " target\n"
                    << "  Overall coherence: " << coherence << "\n"
                    << "  Avg phase correction: " << avgPhaseDeg << " deg\n"
                    << "  --- Compare with MagicPhaseTest: delay~49, corr~0.86, coh~0.78 ---\n\n";
        }
    }

    // =========================================================================
    // Store results — use trueDelay for real-time correction
    // =========================================================================
    {
        std::lock_guard<std::mutex> lock (analysisMutex);
        pendingDelaySamples = trueDelay;
        pendingDelaySubSample = trueDelaySubSample;
        pendingPolarityInvert = rawPolarityInv;

        size_t copySize = std::min (phaseCorr.size(), pendingPhaseCorrection.size());
        std::copy (phaseCorr.begin(), phaseCorr.begin() + copySize, pendingPhaseCorrection.begin());
        if (copySize < pendingPhaseCorrection.size())
            std::fill (pendingPhaseCorrection.begin() + copySize, pendingPhaseCorrection.end(), 0.0f);
    }

    // Update shared state
    if (mySlot >= 0)
    {
        sharedState.updateInstanceData (mySlot,
            trueDelay, trueDelaySubSample, trueDelayMs, rawCorr, coherence, phaseDeg,
            rawPolarityInv, true, true, spectralAnalyzer.getSpectralBands());
        sharedState.setInstanceAligned (mySlot);
    }

    // Signal completion (audio thread will pick this up)
    analysisComplete.store (true);
}

void MagicPhaseProcessor::applyPendingResults()
{
    std::lock_guard<std::mutex> lock (analysisMutex);

    // Use sub-sample delay when enabled, integer otherwise
    float delayToApply = subSampleOn.load() ? pendingDelaySubSample : pendingDelaySamples;
    phaseCorrector.setDelaySamples (delayToApply);
    phaseCorrector.setPolarityInvert (pendingPolarityInvert);

    std::vector<float> phaseCorr (pendingPhaseCorrection.begin(), pendingPhaseCorrection.end());
    phaseCorrector.setPhaseCorrection (phaseCorr);
    phaseCorrector.setTimeCorrectionOn (true);
    phaseCorrector.setPhaseCorrectionOn (true);
}

bool MagicPhaseProcessor::isDawPlaying() const
{
    auto* ph = getPlayHead();
    if (ph == nullptr)
        return true; // Assume playing if no playhead (standalone, tests, etc.)

    auto position = ph->getPosition();
    if (! position.hasValue())
        return true; // Assume playing if position unavailable

    return position->getIsPlaying();
}

int64_t MagicPhaseProcessor::getPlayheadSamplePos() const
{
    auto* ph = getPlayHead();
    if (ph != nullptr)
    {
        auto position = ph->getPosition();
        if (position.hasValue() && position->getTimeInSamples().hasValue())
            return *position->getTimeInSamples();
    }
    // Fallback: use total samples processed counter (works for FakeDAW/tests)
    return totalSamplesProcessed;
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
