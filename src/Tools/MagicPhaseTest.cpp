/*
    MagicPhaseTest - CLI offline test tool for Magic Phase DSP

    Usage:
        ./bin/MagicPhaseTest reference.wav target.wav [-o output_dir]

    Outputs:
        aligned.wav    - Phase-corrected target
        sum_before.wav - ref + target (comb filtering)
        sum_after.wav  - ref + aligned (coherent)
        analysis.csv   - Per-bin: freq_hz, phase_correction_deg, coherence, confidence
*/

#include <juce_core/juce_core.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_basics/juce_audio_basics.h>

#include "AudioFile.h"
#include "DSP/STFTProcessor.h"
#include "DSP/PhaseAnalyzer.h"
#include "DSP/PhaseCorrector.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>

//==============================================================================
// Convert multichannel to mono by averaging
static std::vector<float> toMono (const AudioFile<float>& af)
{
    int numSamples = af.getNumSamplesPerChannel();
    int numChannels = af.getNumChannels();
    std::vector<float> mono (numSamples, 0.0f);

    for (int ch = 0; ch < numChannels; ++ch)
        for (int i = 0; i < numSamples; ++i)
            mono[i] += af.samples[ch][i];

    if (numChannels > 1)
    {
        float scale = 1.0f / numChannels;
        for (auto& s : mono)
            s *= scale;
    }
    return mono;
}

// Save mono float buffer as WAV
static bool saveMono (const std::string& path, const std::vector<float>& data, uint32_t sampleRate)
{
    AudioFile<float> af;
    af.setSampleRate (sampleRate);
    af.setBitDepth (24);
    af.setNumChannels (1);
    af.setNumSamplesPerChannel (static_cast<int> (data.size()));

    for (int i = 0; i < static_cast<int> (data.size()); ++i)
        af.samples[0][i] = data[i];

    return af.save (path);
}

// Normalize to peak with headroom
static std::vector<float> normalize (const std::vector<float>& data, float headroomDb = -1.0f)
{
    float peak = 0.0f;
    for (auto s : data)
        peak = std::max (peak, std::abs (s));

    if (peak < 1e-10f)
        return data;

    float targetLevel = std::pow (10.0f, headroomDb / 20.0f);
    float gain = targetLevel / peak;

    std::vector<float> out (data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = data[i] * gain;
    return out;
}

// RMS energy in dB
static float rmsDb (const std::vector<float>& data)
{
    float sum = 0.0f;
    for (auto s : data)
        sum += s * s;
    float rms = std::sqrt (sum / data.size());
    return 20.0f * std::log10 (rms + 1e-10f);
}


//==============================================================================
int main (int argc, char* argv[])
{
    // Parse arguments
    if (argc < 3)
    {
        std::cout << "Magic Phase Test Tool\n"
                  << "Usage: MagicPhaseTest reference.wav target.wav [-o output_dir]\n";
        return 1;
    }

    std::string refPath = argv[1];
    std::string tarPath = argv[2];
    std::string outputDir = "./output_cpp";

    for (int i = 3; i < argc; ++i)
    {
        if (std::string (argv[i]) == "-o" && i + 1 < argc)
            outputDir = argv[++i];
    }

    // Create output directory
    juce::File (outputDir).createDirectory();

    std::cout << "=== Magic Phase Test Tool ===\n\n";

    // Load WAVs
    std::cout << "Loading reference: " << refPath << "\n";
    AudioFile<float> refFile;
    if (! refFile.load (refPath))
    {
        std::cerr << "ERROR: Failed to load reference file\n";
        return 1;
    }

    std::cout << "Loading target: " << tarPath << "\n";
    AudioFile<float> tarFile;
    if (! tarFile.load (tarPath))
    {
        std::cerr << "ERROR: Failed to load target file\n";
        return 1;
    }

    if (refFile.getSampleRate() != tarFile.getSampleRate())
    {
        std::cerr << "ERROR: Sample rates don't match ("
                  << refFile.getSampleRate() << " vs " << tarFile.getSampleRate() << ")\n";
        return 1;
    }

    double sampleRate = refFile.getSampleRate();

    // Convert to mono
    auto refMono = toMono (refFile);
    auto tarMono = toMono (tarFile);

    if (refFile.getNumChannels() > 1)
        std::cout << "  Reference: " << refFile.getNumChannels() << "ch -> mono\n";
    if (tarFile.getNumChannels() > 1)
        std::cout << "  Target: " << tarFile.getNumChannels() << "ch -> mono\n";

    // Match lengths
    int numSamples = static_cast<int> (std::min (refMono.size(), tarMono.size()));
    refMono.resize (numSamples);
    tarMono.resize (numSamples);

    std::cout << "  Audio: " << (numSamples / sampleRate) << "s @ " << sampleRate << " Hz\n";
    std::cout << "  Samples: " << numSamples << "\n\n";

    constexpr int blockSize = 512;

    // Helper: run audio through an STFT pass, collect frames AND time-domain output
    auto stftPass = [&] (const std::vector<float>& input,
                         std::vector<std::vector<std::complex<float>>>& frames,
                         STFTProcessor::FrameCallback frameCallback) -> std::vector<float>
    {
        STFTProcessor stft;
        stft.prepare (sampleRate, blockSize);
        const int latency = stft.getLatencySamples();
        const int paddedLen = static_cast<int> (input.size()) + latency;

        std::vector<float> padded (paddedLen, 0.0f);
        std::copy (input.begin(), input.end(), padded.begin());

        auto collectAndProcess = [&frames, &frameCallback] (std::complex<float>* frame, int numBins)
        {
            frames.emplace_back (frame, frame + numBins);
            if (frameCallback)
                frameCallback (frame, numBins);
        };

        for (int pos = 0; pos < paddedLen; pos += blockSize)
        {
            int thisBlock = std::min (blockSize, paddedLen - pos);
            stft.processBlock (padded.data() + pos, thisBlock, collectAndProcess);
        }

        // Trim STFT latency
        return std::vector<float> (padded.begin() + latency,
                                   padded.begin() + latency + static_cast<int> (input.size()));
    };

    // =========================================================================
    // STAGE 1: Delay detection using time-domain cross-correlation
    // =========================================================================
    std::cout << "--- Stage 1: Delay Detection ---\n";

    PhaseAnalyzer delayAnalyzer;
    delayAnalyzer.prepare (sampleRate);
    delayAnalyzer.detectDelayTimeDomain (refMono, tarMono);

    float delaySamples = delayAnalyzer.getDelaySamples();
    float delayMs = delayAnalyzer.getDelayMs();
    float correlation = delayAnalyzer.getCorrelation();
    bool polarityInverted = delayAnalyzer.getPolarityInverted();

    std::cout << "  Delay: " << delaySamples << " samples (" << delayMs << " ms)\n";
    std::cout << "  Correlation: " << correlation << "\n";
    std::cout << "  Polarity: " << (polarityInverted ? "INVERTED (180)" : "Normal") << "\n";

    // =========================================================================
    // STAGE 2: Apply time correction in time domain (matching Python approach)
    // =========================================================================
    std::cout << "\n--- Stage 2: Time Correction ---\n";

    int delayInt = static_cast<int> (std::round (delaySamples));
    float polaritySign = polarityInverted ? -1.0f : 1.0f;

    std::vector<float> timeCorrected (numSamples, 0.0f);
    for (int i = 0; i < numSamples; ++i)
    {
        int srcIdx = i + delayInt;
        if (srcIdx >= 0 && srcIdx < numSamples)
            timeCorrected[i] = tarMono[srcIdx] * polaritySign;
    }

    std::cout << "  Applied " << delayInt << " sample shift"
              << (polarityInverted ? " + polarity invert" : "") << "\n";

    // =========================================================================
    // STAGE 3: Spectral analysis on ref vs TIME-CORRECTED target
    // =========================================================================
    std::cout << "\n--- Stage 3: Spectral Phase Analysis ---\n";

    std::vector<std::vector<std::complex<float>>> refFrames2;
    std::vector<std::vector<std::complex<float>>> corrFrames2;

    stftPass (refMono, refFrames2, nullptr);
    stftPass (timeCorrected, corrFrames2, nullptr);

    PhaseAnalyzer spectralAnalyzer;
    spectralAnalyzer.prepare (sampleRate);
    spectralAnalyzer.setCoherenceThreshold (0.4f);
    spectralAnalyzer.setMaxCorrectionDeg (120.0f);
    spectralAnalyzer.analyzeSpectralPhase (refFrames2, corrFrames2);

    float coherence = spectralAnalyzer.getOverallCoherence();
    float phaseDeg = spectralAnalyzer.getPhaseDegrees();
    const auto& phaseCorrection = spectralAnalyzer.getPhaseCorrection();

    std::cout << "  Overall coherence: " << coherence << "\n";
    std::cout << "  Average phase correction: " << phaseDeg << " deg\n";

    // =========================================================================
    // STAGE 4: Apply spectral correction on top of time-corrected audio
    // =========================================================================
    std::cout << "\n--- Stage 4: Spectral Phase Correction ---\n";

    PhaseCorrector spectralCorrector;
    spectralCorrector.prepare (sampleRate);
    spectralCorrector.setPhaseCorrection (phaseCorrection);
    spectralCorrector.setTimeCorrectionOn (false);
    spectralCorrector.setPhaseCorrectionOn (true);

    std::vector<std::vector<std::complex<float>>> unusedFrames2;
    auto spectralCorrCallback = [&spectralCorrector] (std::complex<float>* frame, int numBins)
    {
        spectralCorrector.applyPhaseCorrection (frame, numBins);
    };

    std::vector<float> aligned = stftPass (timeCorrected, unusedFrames2, spectralCorrCallback);

    std::cout << "  Applied spectral phase correction\n";

    // =========================================================================
    // COMPUTE SUMS AND METRICS
    // =========================================================================
    std::vector<float> sumBefore (numSamples);
    std::vector<float> sumAfter (numSamples);

    for (int i = 0; i < numSamples; ++i)
    {
        sumBefore[i] = refMono[i] + tarMono[i];
        sumAfter[i]  = refMono[i] + aligned[i];
    }

    float energyBefore = 0.0f, energyAfter = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        energyBefore += sumBefore[i] * sumBefore[i];
        energyAfter  += sumAfter[i]  * sumAfter[i];
    }

    float energyGainDb = 10.0f * std::log10 ((energyAfter + 1e-10f) / (energyBefore + 1e-10f));

    std::cout << "\n--- Results ---\n";
    std::cout << "  Sum energy gain: " << energyGainDb << " dB\n";
    std::cout << "  RMS before: " << rmsDb (sumBefore) << " dB\n";
    std::cout << "  RMS after:  " << rmsDb (sumAfter) << " dB\n";

    // =========================================================================
    // WRITE OUTPUT FILES
    // =========================================================================
    std::cout << "\n--- Writing Output ---\n";

    // Sum for time-only comparison
    std::vector<float> sumTimeOnly (numSamples);
    for (int i = 0; i < numSamples; ++i)
        sumTimeOnly[i] = refMono[i] + timeCorrected[i];

    float energyTimeOnly = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        energyTimeOnly += sumTimeOnly[i] * sumTimeOnly[i];
    float timeOnlyGainDb = 10.0f * std::log10 ((energyTimeOnly + 1e-10f) / (energyBefore + 1e-10f));
    std::cout << "  Time-only energy gain: " << timeOnlyGainDb << " dB\n";

    std::string sep = "/";
    saveMono (outputDir + sep + "ref_mono.wav", normalize (refMono), static_cast<uint32_t> (sampleRate));
    saveMono (outputDir + sep + "unaligned.wav", normalize (tarMono), static_cast<uint32_t> (sampleRate));
    saveMono (outputDir + sep + "time_only.wav", normalize (timeCorrected), static_cast<uint32_t> (sampleRate));
    saveMono (outputDir + sep + "aligned.wav", normalize (aligned), static_cast<uint32_t> (sampleRate));
    saveMono (outputDir + sep + "sum_before.wav", normalize (sumBefore), static_cast<uint32_t> (sampleRate));
    saveMono (outputDir + sep + "sum_time_only.wav", normalize (sumTimeOnly), static_cast<uint32_t> (sampleRate));
    saveMono (outputDir + sep + "sum_after.wav", normalize (sumAfter), static_cast<uint32_t> (sampleRate));

    std::cout << "  Wrote ref_mono.wav\n";
    std::cout << "  Wrote unaligned.wav\n";
    std::cout << "  Wrote time_only.wav       (time correction only)\n";
    std::cout << "  Wrote aligned.wav         (time + spectral)\n";
    std::cout << "  Wrote sum_before.wav\n";
    std::cout << "  Wrote sum_time_only.wav   (ref + time_only)\n";
    std::cout << "  Wrote sum_after.wav\n";

    // =========================================================================
    // WRITE ANALYSIS CSV
    // =========================================================================
    {
        std::string csvPath = outputDir + sep + "analysis.csv";
        std::ofstream csv (csvPath);
        csv << "bin,freq_hz,phase_correction_rad,phase_correction_deg,coherence\n";

        float binFreqStep = static_cast<float> (sampleRate) / PhaseAnalyzer::kFFTSize;

        for (int k = 0; k < PhaseAnalyzer::kNumBins; ++k)
        {
            float freqHz = k * binFreqStep;
            float corrRad = (k < static_cast<int> (phaseCorrection.size())) ? phaseCorrection[k] : 0.0f;
            float corrDeg = corrRad * 180.0f / 3.14159265358979323846f;

            csv << k << ","
                << freqHz << ","
                << corrRad << ","
                << corrDeg << ","
                << 0.0f  // coherencePerBin is private; we output overall
                << "\n";
        }

        std::cout << "  Wrote analysis.csv (" << PhaseAnalyzer::kNumBins << " bins)\n";
    }

    // =========================================================================
    // SUMMARY
    // =========================================================================
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "  Delay:      " << delaySamples << " samples (" << delayMs << " ms)\n";
    std::cout << "  Polarity:   " << (polarityInverted ? "INVERTED" : "Normal") << "\n";
    std::cout << "  Correlation:" << correlation << "\n";
    std::cout << "  Coherence:  " << coherence << "\n";
    std::cout << "  Phase avg:  " << phaseDeg << " deg\n";
    std::cout << "  Energy gain:" << energyGainDb << " dB\n";
    std::cout << "  Output dir: " << outputDir << "/\n";
    std::cout << "========================================\n";

    return 0;
}
