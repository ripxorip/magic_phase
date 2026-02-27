#include "ResultWriter.h"

void ResultWriter::writeResultJson (const TestDefinition& test,
                                     const TestResults& results,
                                     const juce::File& outputFile)
{
    auto* root = new juce::DynamicObject();

    root->setProperty ("test", test.name);
    root->setProperty ("timestamp", juce::Time::getCurrentTime().toISO8601 (true));

    // Config
    auto* config = new juce::DynamicObject();
    config->setProperty ("plugin_path", test.pluginPath);
    config->setProperty ("sample_rate", test.sampleRate);
    config->setProperty ("buffer_size", test.bufferSize);
    config->setProperty ("plugin_loaded", true);
    config->setProperty ("num_instances", static_cast<int> (results.tracks.size()));
    root->setProperty ("config", juce::var (config));

    // Tracks
    juce::Array<juce::var> tracksArray;
    for (auto& track : results.tracks)
    {
        auto* trackObj = new juce::DynamicObject();
        trackObj->setProperty ("name", track.name);
        trackObj->setProperty ("role", track.role);
        trackObj->setProperty ("input_file", track.inputFile);
        trackObj->setProperty ("output_file", track.outputFile);
        trackObj->setProperty ("slot", track.slotIndex);

        if (track.role == "target")
        {
            trackObj->setProperty ("mode", track.mode);

            auto* resultsObj = new juce::DynamicObject();
            resultsObj->setProperty ("alignment_state", track.alignmentState);
            resultsObj->setProperty ("delay_samples", static_cast<double> (track.delaySamples));
            resultsObj->setProperty ("delay_ms", static_cast<double> (track.delayMs));
            resultsObj->setProperty ("correlation", static_cast<double> (track.correlation));
            resultsObj->setProperty ("coherence", static_cast<double> (track.coherence));
            resultsObj->setProperty ("phase_degrees", static_cast<double> (track.phaseDegrees));
            resultsObj->setProperty ("polarity_inverted", track.polarityInverted);
            resultsObj->setProperty ("time_correction_on", track.timeCorrectionOn);
            resultsObj->setProperty ("phase_correction_on", track.phaseCorrectionOn);

            juce::Array<juce::var> bandsArray;
            for (int b = 0; b < 48; ++b)
                bandsArray.add (static_cast<double> (track.spectralBands[b]));
            resultsObj->setProperty ("spectral_bands", bandsArray);

            trackObj->setProperty ("results", juce::var (resultsObj));
        }

        tracksArray.add (juce::var (trackObj));
    }
    root->setProperty ("tracks", tracksArray);

    // Timing
    auto* timingObj = new juce::DynamicObject();
    timingObj->setProperty ("plugin_load_ms", results.timing.pluginLoadMs);
    timingObj->setProperty ("prepare_ms", results.timing.prepareMs);
    timingObj->setProperty ("playback_ms", results.timing.playbackMs);
    timingObj->setProperty ("analysis_wait_ms", results.timing.analysisWaitMs);
    timingObj->setProperty ("total_ms", results.timing.totalMs);
    root->setProperty ("timing", juce::var (timingObj));

    // Diagnostics
    auto* diagObj = new juce::DynamicObject();
    diagObj->setProperty ("ref_raw_start_sample", static_cast<juce::int64> (results.refRawStartSample));
    root->setProperty ("diagnostics", juce::var (diagObj));

    // Write
    auto jsonString = juce::JSON::toString (juce::var (root), true);
    outputFile.getParentDirectory().createDirectory();
    outputFile.replaceWithText (jsonString);

    std::cout << "  Result written to: " << outputFile.getFullPathName() << "\n";
}
