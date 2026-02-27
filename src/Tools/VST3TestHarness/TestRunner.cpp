#include "TestRunner.h"
#include <iostream>
#include <chrono>
#include <map>

using Clock = std::chrono::high_resolution_clock;

static double msElapsed (Clock::time_point start)
{
    return std::chrono::duration<double, std::milli> (Clock::now() - start).count();
}

TestRunner::TestRunner()
{
    audioFormatManager.registerBasicFormats();
}

TestRunner::~TestRunner() = default;

//==============================================================================
// JSON Parsing
//==============================================================================
TestDefinition TestRunner::loadTestDefinition (const juce::File& jsonFile)
{
    TestDefinition def;

    auto jsonText = jsonFile.loadFileAsString();
    auto parsed = juce::JSON::parse (jsonText);

    if (! parsed.isObject())
    {
        std::cerr << "Failed to parse test definition: " << jsonFile.getFullPathName() << "\n";
        return def;
    }

    auto* obj = parsed.getDynamicObject();
    def.name = obj->getProperty ("name").toString();
    def.description = obj->getProperty ("description").toString();
    def.pluginPath = obj->getProperty ("plugin_path").toString();
    def.sampleRate = static_cast<double> (obj->getProperty ("sample_rate"));
    def.bufferSize = static_cast<int> (obj->getProperty ("buffer_size"));

    // Buffer size sweep
    if (obj->hasProperty ("buffer_sizes"))
    {
        auto* bsArray = obj->getProperty ("buffer_sizes").getArray();
        if (bsArray != nullptr)
        {
            for (auto& bs : *bsArray)
                def.bufferSizes.push_back (static_cast<int> (bs));
        }
    }

    // Tracks
    auto* tracksArray = obj->getProperty ("tracks").getArray();
    if (tracksArray != nullptr)
    {
        for (auto& trackVar : *tracksArray)
        {
            auto* trackObj = trackVar.getDynamicObject();
            if (trackObj == nullptr) continue;

            TrackDefinition track;
            track.file = trackObj->getProperty ("file").toString();
            track.role = trackObj->getProperty ("role").toString();
            track.mode = trackObj->getProperty ("mode").toString();
            if (trackObj->hasProperty ("gain"))
                track.gain = static_cast<float> (trackObj->getProperty ("gain"));
            def.tracks.push_back (track);
        }
    }

    return def;
}

//==============================================================================
// Audio Loading
//==============================================================================
juce::AudioBuffer<float> TestRunner::loadWavFile (const juce::String& path, double /*targetSampleRate*/)
{
    juce::File file (path);
    std::unique_ptr<juce::AudioFormatReader> reader (audioFormatManager.createReaderFor (file));

    if (reader == nullptr)
    {
        std::cerr << "Failed to load audio file: " << path << "\n";
        return {};
    }

    int numSamples = static_cast<int> (reader->lengthInSamples);
    int numChannels = static_cast<int> (reader->numChannels);

    juce::AudioBuffer<float> buffer (numChannels, numSamples);
    reader->read (&buffer, 0, numSamples, 0, true, true);

    // Mix to mono if stereo
    if (numChannels > 1)
    {
        juce::AudioBuffer<float> mono (1, numSamples);
        mono.clear();
        for (int ch = 0; ch < numChannels; ++ch)
            mono.addFrom (0, 0, buffer, ch, 0, numSamples, 1.0f / numChannels);
        return mono;
    }

    return buffer;
}

juce::AudioBuffer<float> TestRunner::getAudioChunk (const juce::AudioBuffer<float>& source,
                                                     int blockIndex, int blockSize)
{
    juce::AudioBuffer<float> chunk (1, blockSize);
    int sourceLen = source.getNumSamples();

    if (sourceLen == 0)
    {
        chunk.clear();
        return chunk;
    }

    auto* dest = chunk.getWritePointer (0);
    auto* src = source.getReadPointer (0);
    int startSample = blockIndex * blockSize;

    for (int i = 0; i < blockSize; ++i)
    {
        int srcIdx = (startSample + i) % sourceLen;
        dest[i] = src[srcIdx];
    }

    return chunk;
}

//==============================================================================
// Parameter Access
//==============================================================================
static juce::AudioProcessorParameter* findParameter (juce::AudioPluginInstance* instance,
                                                      const juce::String& paramId)
{
    // Try 1: match by AudioProcessorParameterWithID (works for direct instantiation)
    for (auto* param : instance->getParameters())
    {
        if (auto* paramWithID = dynamic_cast<juce::AudioProcessorParameterWithID*> (param))
        {
            if (paramWithID->getParameterID() == paramId)
                return param;
        }
    }

    // Try 2: match by getName() (works for hosted VST3 — JUCE preserves parameter names)
    // Map our paramIDs to the display names we set in createParameterLayout()
    static const std::map<juce::String, juce::String> idToName = {
        { "triggerAlign",        "Trigger Align" },
        { "isReference",         "Is Reference" },
        { "correctionMode",      "Correction Mode" },
        { "bypass",              "Bypass" },
        { "coherenceThreshold",  "Coherence Threshold" },
        { "maxCorrection",       "Max Correction" },
    };

    juce::String targetName;
    auto it = idToName.find (paramId);
    if (it != idToName.end())
        targetName = it->second;

    if (targetName.isNotEmpty())
    {
        for (auto* param : instance->getParameters())
        {
            if (param->getName (256) == targetName)
                return param;
        }
    }

    std::cerr << "Warning: parameter '" << paramId << "' not found. Available:\n";
    for (auto* param : instance->getParameters())
        std::cerr << "  [" << param->getParameterIndex() << "] " << param->getName (256) << "\n";

    return nullptr;
}

void TestRunner::setPluginParameter (juce::AudioPluginInstance* instance,
                                     const juce::String& paramId,
                                     float normalizedValue)
{
    if (auto* param = findParameter (instance, paramId))
        param->setValueNotifyingHost (normalizedValue);
}

void TestRunner::setPluginParameterByValue (juce::AudioPluginInstance* instance,
                                             const juce::String& paramId,
                                             float value)
{
    if (auto* param = findParameter (instance, paramId))
    {
        if (auto* ranged = dynamic_cast<juce::RangedAudioParameter*> (param))
            param->setValueNotifyingHost (ranged->convertTo0to1 (value));
        else
            param->setValueNotifyingHost (value);
    }
}

//==============================================================================
// Shared Memory Polling
//==============================================================================
bool TestRunner::checkAllTargetsAligned (SharedState& sharedState, const TestDefinition& test)
{
    auto* header = sharedState.getHeader();
    if (header == nullptr) return false;

    for (int i = 0; i < kMaxInstances; ++i)
    {
        auto* slot = sharedState.getSlot (i);
        if (slot == nullptr || slot->active.load() == 0)
            continue;

        // Skip reference slot
        if (i == header->referenceSlot.load())
            continue;

        // Check if aligned (active == 2)
        if (slot->active.load() != 2)
            return false;
    }

    // Must have found at least one non-reference slot
    int refSlot = header->referenceSlot.load();
    for (int i = 0; i < kMaxInstances; ++i)
    {
        auto* slot = sharedState.getSlot (i);
        if (slot != nullptr && slot->active.load() > 0 && i != refSlot)
            return true;
    }

    return false;
}

TestResults TestRunner::readResultsFromSharedMemory (SharedState& sharedState,
                                                      const TestDefinition& test)
{
    TestResults results;
    auto* header = sharedState.getHeader();
    if (header == nullptr) return results;

    int refSlot = header->referenceSlot.load();
    int trackIdx = 0;

    for (int i = 0; i < kMaxInstances; ++i)
    {
        auto* slot = sharedState.getSlot (i);
        if (slot == nullptr || slot->active.load() == 0)
            continue;

        TrackResult tr;
        tr.name = juce::String (slot->trackName);
        tr.slotIndex = i;

        bool isRef = (i == refSlot);
        tr.role = isRef ? "reference" : "target";

        if (! isRef && trackIdx < static_cast<int> (test.tracks.size()))
        {
            // Find matching target track
            for (auto& td : test.tracks)
            {
                if (td.role == "target")
                {
                    tr.mode = td.mode;
                    tr.inputFile = td.file;
                    break;
                }
            }
        }
        else if (isRef && ! test.tracks.empty())
        {
            tr.inputFile = test.tracks[0].file;
        }

        tr.isAligned = (slot->active.load() == 2);
        tr.alignmentState = tr.isAligned ? "ALIGNED" : "NOT_ALIGNED";
        tr.delaySamples = slot->delaySamples;
        tr.delaySubSample = slot->delaySubSample;
        tr.delayMs = slot->delayMs;
        tr.correlation = slot->correlation;
        tr.coherence = slot->overallCoherence;
        tr.phaseDegrees = slot->phaseDegrees;
        tr.polarityInverted = slot->polarityInverted;
        tr.timeCorrectionOn = slot->timeCorrectionOn;
        tr.phaseCorrectionOn = slot->phaseCorrectionOn;
        std::memcpy (tr.spectralBands, slot->spectralBands, sizeof (tr.spectralBands));

        results.tracks.push_back (tr);
        trackIdx++;
    }

    results.refRawStartSample = header->refRawStartSample.load();
    return results;
}

//==============================================================================
// Main Processing Loop
//==============================================================================
TestResults TestRunner::run (const TestDefinition& test)
{
    TestResults results;
    auto totalStart = Clock::now();

    std::cout << "=== Running test: " << test.name << " ===\n";
    std::cout << "  Plugin: " << test.pluginPath << "\n";
    std::cout << "  Sample rate: " << test.sampleRate << " Hz\n";
    std::cout << "  Buffer size: " << test.bufferSize << "\n";
    std::cout << "  Tracks: " << test.tracks.size() << "\n";

    // =========================================================================
    // Step 1: Load plugin instances
    // =========================================================================
    auto loadStart = Clock::now();

    juce::AudioPluginFormatManager formatManager;
    formatManager.addDefaultFormats();

    // Scan for plugin
    juce::OwnedArray<juce::PluginDescription> descriptions;
    juce::KnownPluginList pluginList;

    for (auto* format : formatManager.getFormats())
    {
        pluginList.scanAndAddFile (test.pluginPath, false, descriptions, *format);
    }

    if (descriptions.isEmpty())
    {
        results.errorMessage = "No plugin found at: " + test.pluginPath;
        std::cerr << results.errorMessage << "\n";
        return results;
    }

    std::cout << "  Found plugin: " << descriptions[0]->name << "\n";

    std::vector<std::unique_ptr<juce::AudioPluginInstance>> instances;
    juce::String error;

    for (size_t i = 0; i < test.tracks.size(); ++i)
    {
        auto instance = formatManager.createPluginInstance (
            *descriptions[0], test.sampleRate, test.bufferSize, error);

        if (instance == nullptr)
        {
            results.errorMessage = "Failed to load plugin instance " + juce::String (static_cast<int> (i))
                                    + ": " + error;
            std::cerr << results.errorMessage << "\n";
            return results;
        }

        instances.push_back (std::move (instance));
    }

    results.timing.pluginLoadMs = msElapsed (loadStart);
    std::cout << "  Plugin load: " << results.timing.pluginLoadMs << " ms\n";

    // =========================================================================
    // Step 2: Load audio files
    // =========================================================================
    std::vector<juce::AudioBuffer<float>> inputBuffers;
    for (auto& track : test.tracks)
    {
        auto buf = loadWavFile (track.file, test.sampleRate);
        if (buf.getNumSamples() == 0)
        {
            results.errorMessage = "Failed to load audio: " + track.file;
            std::cerr << results.errorMessage << "\n";
            return results;
        }
        std::cout << "  Loaded: " << track.file << " (" << buf.getNumSamples() << " samples)\n";
        inputBuffers.push_back (std::move (buf));
    }

    // =========================================================================
    // Step 3: Prepare plugin instances
    // =========================================================================
    auto prepStart = Clock::now();

    playHead.setSampleRate (test.sampleRate);
    playHead.reset();
    playHead.setPlaying (false);

    for (auto& instance : instances)
    {
        instance->setPlayHead (&playHead);
        instance->prepareToPlay (test.sampleRate, test.bufferSize);
    }

    results.timing.prepareMs = msElapsed (prepStart);

    // =========================================================================
    // Step 4: Initialize shared memory (AFTER plugin prepareToPlay)
    // =========================================================================
    SharedState sharedState;
    sharedState.initialize();

    // Set reference track parameter on instance[0]
    setPluginParameter (instances[0].get(), "isReference", 1.0f);
    std::cout << "  Set instance 0 as reference\n";

    // =========================================================================
    // Step 5: Settle phase (~1 second of playback to stabilize heartbeats)
    // =========================================================================
    auto playStart = Clock::now();
    playHead.setPlaying (true);

    int settleBlocks = static_cast<int> (test.sampleRate / test.bufferSize);
    std::cout << "  Settling (" << settleBlocks << " blocks)...\n";

    for (int block = 0; block < settleBlocks; ++block)
    {
        for (size_t i = 0; i < instances.size(); ++i)
        {
            auto chunk = getAudioChunk (inputBuffers[i], block, test.bufferSize);
            juce::MidiBuffer midi;
            instances[i]->processBlock (chunk, midi);
        }
        playHead.advance (test.bufferSize);
    }

    // =========================================================================
    // Step 6: Trigger alignment on target instances
    // =========================================================================
    for (size_t i = 0; i < instances.size(); ++i)
    {
        if (test.tracks[i].role == "target")
        {
            // Set correction mode
            int mode = (test.tracks[i].mode == "t") ? 0 : 1;  // default to phi
            setPluginParameterByValue (instances[i].get(), "correctionMode", static_cast<float> (mode));

            // Trigger alignment
            setPluginParameter (instances[i].get(), "triggerAlign", 1.0f);
            std::cout << "  Triggered alignment on track " << i
                      << " (mode=" << test.tracks[i].mode << ")\n";
        }
    }

    // =========================================================================
    // Step 7: Accumulation phase (~8.5 seconds of audio)
    // =========================================================================
    constexpr float kRequiredSeconds = 7.5f;
    int accumBlocks = static_cast<int> ((kRequiredSeconds + 1.0f) * test.sampleRate / test.bufferSize);
    int totalBlocksSoFar = settleBlocks;

    std::cout << "  Accumulating (" << accumBlocks << " blocks, ~"
              << static_cast<float> (accumBlocks * test.bufferSize) / test.sampleRate << "s)...\n";

    for (int block = 0; block < accumBlocks; ++block)
    {
        int globalBlock = totalBlocksSoFar + block;
        for (size_t i = 0; i < instances.size(); ++i)
        {
            auto chunk = getAudioChunk (inputBuffers[i], globalBlock, test.bufferSize);
            juce::MidiBuffer midi;
            instances[i]->processBlock (chunk, midi);
        }
        playHead.advance (test.bufferSize);
    }

    totalBlocksSoFar += accumBlocks;
    results.timing.playbackMs = msElapsed (playStart);

    // =========================================================================
    // Step 8: Wait for analysis to complete (poll shared memory, max 30s wall-clock)
    //
    // The analysis thread runs cross-correlation + STFT re-analysis on ~360k samples,
    // which can take 10-30s. We need to keep calling processBlock (so the plugin can
    // pick up analysisComplete) while giving the analysis thread real wall-clock time.
    // =========================================================================
    auto waitStart = Clock::now();
    constexpr double kMaxWaitSeconds = 60.0;
    int waitBlockCount = 0;
    bool aligned = false;

    std::cout << "  Waiting for alignment (max " << kMaxWaitSeconds << "s wall-clock)...\n";

    while (! aligned && msElapsed (waitStart) < kMaxWaitSeconds * 1000.0)
    {
        // Process a small batch of blocks
        for (int b = 0; b < 16 && ! aligned; ++b)
        {
            int globalBlock = totalBlocksSoFar + waitBlockCount;
            for (size_t i = 0; i < instances.size(); ++i)
            {
                auto chunk = getAudioChunk (inputBuffers[i], globalBlock, test.bufferSize);
                juce::MidiBuffer midi;
                instances[i]->processBlock (chunk, midi);
            }
            playHead.advance (test.bufferSize);
            waitBlockCount++;

            aligned = checkAllTargetsAligned (sharedState, test);
        }

        if (! aligned)
            juce::Thread::sleep (100);  // Give analysis thread CPU time
    }

    totalBlocksSoFar += waitBlockCount;
    results.timing.analysisWaitMs = msElapsed (waitStart);

    if (aligned)
        std::cout << "  Alignment complete!\n";
    else
        std::cerr << "  WARNING: Alignment did not complete within timeout\n";

    // =========================================================================
    // Step 9a: Flush STFT internal state — process silence through all
    //          instances to clear overlap-add buffers from the alignment cycle.
    //          Then reset playhead for a clean replay.
    // =========================================================================
    constexpr int kFlushSamples = 8192;  // 2x FFT window (4096) to fully flush overlap-add
    int flushBlocks = (kFlushSamples + test.bufferSize - 1) / test.bufferSize;

    playHead.reset();
    playHead.setPlaying (true);

    std::cout << "  Flushing STFT state (" << flushBlocks << " blocks of silence)...\n";

    for (int block = 0; block < flushBlocks; ++block)
    {
        for (auto& instance : instances)
        {
            juce::AudioBuffer<float> silence (1, test.bufferSize);
            silence.clear();
            juce::MidiBuffer midi;
            instance->processBlock (silence, midi);
        }
        playHead.advance (test.bufferSize);
    }

    // =========================================================================
    // Step 9b: Full replay — reset playhead and process entire input with
    //          correction active, so output WAVs contain the full song aligned
    // =========================================================================
    int maxInputSamples = 0;
    for (auto& buf : inputBuffers)
        maxInputSamples = std::max (maxInputSamples, buf.getNumSamples());

    int replayBlocks = (maxInputSamples + test.bufferSize - 1) / test.bufferSize;

    // Fresh output buffers sized to the input length
    std::vector<juce::AudioBuffer<float>> replayBuffers;
    for (size_t i = 0; i < instances.size(); ++i)
    {
        juce::AudioBuffer<float> buf (1, replayBlocks * test.bufferSize);
        buf.clear();
        replayBuffers.push_back (std::move (buf));
    }

    playHead.reset();
    playHead.setPlaying (true);

    std::cout << "  Full replay (" << replayBlocks << " blocks, "
              << (static_cast<float> (maxInputSamples) / test.sampleRate) << "s)...\n";

    for (int block = 0; block < replayBlocks; ++block)
    {
        for (size_t i = 0; i < instances.size(); ++i)
        {
            auto chunk = getAudioChunk (inputBuffers[i], block, test.bufferSize);
            juce::MidiBuffer midi;
            instances[i]->processBlock (chunk, midi);

            int destStart = block * test.bufferSize;
            replayBuffers[i].copyFrom (0, destStart, chunk, 0, 0, test.bufferSize);
        }
        playHead.advance (test.bufferSize);
    }

    // Trim to actual input length
    int writeLen = maxInputSamples;

    // =========================================================================
    // Step 10: Read results from shared memory
    // =========================================================================
    TimingInfo savedTiming = results.timing;
    results = readResultsFromSharedMemory (sharedState, test);
    results.timing = savedTiming;

    for (size_t i = 0; i < results.tracks.size() && i < test.tracks.size(); ++i)
    {
        results.tracks[i].inputFile = test.tracks[i].file;

        juce::File outFile (test.outputDir);
        juce::String trackName = juce::File (test.tracks[i].file).getFileNameWithoutExtension();
        results.tracks[i].outputFile = outFile.getChildFile (trackName + "_out.wav").getFullPathName();
        results.tracks[i].mode = test.tracks[i].mode;
    }

    results.timing.totalMs = msElapsed (totalStart);
    results.success = aligned;

    // =========================================================================
    // Step 11: Write output WAVs (full corrected audio)
    // =========================================================================
    juce::File outputDir (test.outputDir);
    outputDir.createDirectory();

    for (size_t i = 0; i < instances.size() && i < test.tracks.size(); ++i)
    {
        juce::String trackName = juce::File (test.tracks[i].file).getFileNameWithoutExtension();
        juce::File outFile = outputDir.getChildFile (trackName + "_out.wav");
        outFile.deleteFile();  // Ensure clean overwrite on Windows

        juce::WavAudioFormat wavFormat;
        std::unique_ptr<juce::AudioFormatWriter> writer (
            wavFormat.createWriterFor (
                new juce::FileOutputStream (outFile),
                test.sampleRate, 1, 16, {}, 0));

        if (writer != nullptr)
        {
            writer->writeFromAudioSampleBuffer (replayBuffers[i], 0, writeLen);
            std::cout << "  Wrote: " << outFile.getFullPathName() << "\n";
        }
        else
        {
            std::cerr << "  Failed to write: " << outFile.getFullPathName() << "\n";
        }
    }

    // Write summed output (all tracks mixed)
    if (replayBuffers.size() >= 2)
    {
        float baseGain = 1.0f / static_cast<float> (replayBuffers.size());
        juce::AudioBuffer<float> sumBuffer (1, writeLen);
        sumBuffer.clear();
        for (size_t i = 0; i < replayBuffers.size(); ++i)
        {
            float trackGain = baseGain * test.tracks[i].gain;
            sumBuffer.addFrom (0, 0, replayBuffers[i], 0, 0, writeLen, trackGain);
        }

        juce::File sumFile = outputDir.getChildFile ("sum.wav");
        sumFile.deleteFile();  // Ensure clean overwrite on Windows
        juce::WavAudioFormat wavFmt;
        std::unique_ptr<juce::AudioFormatWriter> sumWriter (
            wavFmt.createWriterFor (
                new juce::FileOutputStream (sumFile),
                test.sampleRate, 1, 16, {}, 0));

        if (sumWriter != nullptr)
        {
            sumWriter->writeFromAudioSampleBuffer (sumBuffer, 0, writeLen);
            std::cout << "  Wrote: " << sumFile.getFullPathName() << " (summed)\n";
        }
    }

    // Write raw (uncorrected) sum for comparison
    {
        int rawLen = 0;
        for (auto& buf : inputBuffers)
            rawLen = std::max (rawLen, buf.getNumSamples());

        float baseGain = 1.0f / static_cast<float> (inputBuffers.size());
        juce::AudioBuffer<float> rawSum (1, rawLen);
        rawSum.clear();
        for (size_t i = 0; i < inputBuffers.size(); ++i)
        {
            float trackGain = baseGain * test.tracks[i].gain;
            rawSum.addFrom (0, 0, inputBuffers[i], 0, 0, std::min (rawLen, inputBuffers[i].getNumSamples()), trackGain);
        }

        juce::File rawSumFile = outputDir.getChildFile ("raw_sum.wav");
        rawSumFile.deleteFile();  // Ensure clean overwrite on Windows
        juce::WavAudioFormat rawFmt;
        std::unique_ptr<juce::AudioFormatWriter> rawWriter (
            rawFmt.createWriterFor (
                new juce::FileOutputStream (rawSumFile),
                test.sampleRate, 1, 16, {}, 0));

        if (rawWriter != nullptr)
        {
            rawWriter->writeFromAudioSampleBuffer (rawSum, 0, rawLen);
            std::cout << "  Wrote: " << rawSumFile.getFullPathName() << " (raw uncorrected sum)\n";
        }
    }

    // Print diagnostic: compare corrected vs uncorrected RMS
    if (replayBuffers.size() >= 2)
    {
        float baseGain = 1.0f / static_cast<float> (replayBuffers.size());

        // Compute RMS of corrected sum
        float corrRms = 0.0f;
        {
            juce::AudioBuffer<float> corrSum (1, writeLen);
            corrSum.clear();
            for (size_t i = 0; i < replayBuffers.size(); ++i)
            {
                float trackGain = baseGain * test.tracks[i].gain;
                corrSum.addFrom (0, 0, replayBuffers[i], 0, 0, writeLen, trackGain);
            }

            auto* data = corrSum.getReadPointer (0);
            double sum = 0.0;
            for (int s = 0; s < writeLen; ++s)
                sum += static_cast<double> (data[s]) * data[s];
            corrRms = static_cast<float> (std::sqrt (sum / writeLen));
        }

        // Compute RMS of raw sum
        float rawRms = 0.0f;
        {
            juce::AudioBuffer<float> rawSum (1, writeLen);
            rawSum.clear();
            for (size_t i = 0; i < inputBuffers.size(); ++i)
            {
                float trackGain = baseGain * test.tracks[i].gain;
                rawSum.addFrom (0, 0, inputBuffers[i], 0, 0, std::min (writeLen, inputBuffers[i].getNumSamples()), trackGain);
            }

            auto* data = rawSum.getReadPointer (0);
            double sum = 0.0;
            for (int s = 0; s < writeLen; ++s)
                sum += static_cast<double> (data[s]) * data[s];
            rawRms = static_cast<float> (std::sqrt (sum / writeLen));
        }

        std::cout << "  Diagnostic: corrected_sum_rms=" << corrRms
                  << " raw_sum_rms=" << rawRms
                  << " (should differ if corrections applied)\n";
    }

    // =========================================================================
    // Step 12: Cleanup
    // =========================================================================
    for (auto& instance : instances)
    {
        instance->releaseResources();
        instance.reset();
    }

    sharedState.shutdown();

    std::cout << "=== Test complete: " << test.name << " ===\n";
    std::cout << "  Timing: load=" << results.timing.pluginLoadMs
              << "ms prepare=" << results.timing.prepareMs
              << "ms playback=" << results.timing.playbackMs
              << "ms wait=" << results.timing.analysisWaitMs
              << "ms total=" << results.timing.totalMs << "ms\n";

    return results;
}
