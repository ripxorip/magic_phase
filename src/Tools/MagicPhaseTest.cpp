/*
    MagicPhaseTest - CLI offline test tool for Magic Phase DSP

    This tool emulates VST behavior:
    1. Process audio in small blocks (128 samples) like a DAW
    2. Accumulate STFT frames during analyze window (default 7.5s)
    3. After analyze window: compute delay + spectral phase correction
    4. Apply corrections to full file

    Usage:
        ./bin/MagicPhaseTest reference.wav target.wav [-o output_dir] [-w analyze_window_sec]

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
// VST emulation parameters
constexpr int kBlockSize = 128;              // DAW-like block size
constexpr float kDefaultAnalyzeWindow = 7.5f; // seconds

//==============================================================================
// Convert multichannel to mono by averaging
static std::vector<float> toMono (const AudioFile<float>& af)
{
    int numSamples = af.getNumSamplesPerChannel();
    int numChannels = af.getNumChannels();
    std::vector<float> mono (static_cast<size_t> (numSamples), 0.0f);

    for (int ch = 0; ch < numChannels; ++ch)
        for (int i = 0; i < numSamples; ++i)
            mono[static_cast<size_t> (i)] += af.samples[static_cast<size_t> (ch)][static_cast<size_t> (i)];

    if (numChannels > 1)
    {
        float scale = 1.0f / static_cast<float> (numChannels);
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

    for (size_t i = 0; i < data.size(); ++i)
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
    float rms = std::sqrt (sum / static_cast<float> (data.size()));
    return 20.0f * std::log10 (rms + 1e-10f);
}


//==============================================================================
int main (int argc, char* argv[])
{
    // Parse arguments
    if (argc < 3)
    {
        std::cout << "Magic Phase Test Tool (VST Emulation Mode)\n"
                  << "Usage: MagicPhaseTest reference.wav target.wav [-o output_dir] [-w analyze_window_sec]\n"
                  << "\n"
                  << "Options:\n"
                  << "  -o DIR    Output directory (default: ./output_cpp)\n"
                  << "  -w SEC    Analyze window in seconds (default: 7.5)\n"
                  << "\n"
                  << "This emulates VST behavior:\n"
                  << "  1. Process audio in " << kBlockSize << "-sample blocks (like a DAW)\n"
                  << "  2. Accumulate STFT frames during analyze window\n"
                  << "  3. After window: compute delay + phase correction\n"
                  << "  4. Apply corrections to full file\n";
        return 1;
    }

    std::string refPath = argv[1];
    std::string tarPath = argv[2];
    std::string outputDir = "./output_cpp";
    float analyzeWindowSec = kDefaultAnalyzeWindow;

    for (int i = 3; i < argc; ++i)
    {
        if (std::string (argv[i]) == "-o" && i + 1 < argc)
            outputDir = argv[++i];
        else if (std::string (argv[i]) == "-w" && i + 1 < argc)
            analyzeWindowSec = std::stof (argv[++i]);
    }

    // Create output directory
    juce::File (outputDir).createDirectory();

    std::cout << "=== Magic Phase Test Tool (VST Emulation) ===\n\n";
    std::cout << "  Block size:      " << kBlockSize << " samples\n";
    std::cout << "  Analyze window:  " << analyzeWindowSec << " seconds\n\n";

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
    refMono.resize (static_cast<size_t> (numSamples));
    tarMono.resize (static_cast<size_t> (numSamples));

    int analyzeWindowSamples = static_cast<int> (analyzeWindowSec * sampleRate);
    analyzeWindowSamples = std::min (analyzeWindowSamples, numSamples);

    std::cout << "  Audio: " << (numSamples / sampleRate) << "s @ " << sampleRate << " Hz\n";
    std::cout << "  Samples: " << numSamples << "\n";
    std::cout << "  Analyze window: " << analyzeWindowSamples << " samples ("
              << (analyzeWindowSamples / sampleRate) << "s)\n\n";

    // =========================================================================
    // PHASE 1: VST EMULATION - ACCUMULATE FRAMES DURING ANALYZE WINDOW
    // =========================================================================
    std::cout << "=== Phase 1: VST Emulation - Accumulating Frames ===\n";
    std::cout << "  Processing " << kBlockSize << "-sample blocks like a DAW...\n";

    STFTProcessor refSTFT;
    STFTProcessor tarSTFT;
    refSTFT.prepare (sampleRate, kBlockSize);
    tarSTFT.prepare (sampleRate, kBlockSize);

    std::vector<std::vector<std::complex<float>>> refFrames;
    std::vector<std::vector<std::complex<float>>> tarFrames;

    // Collect frames callback
    auto collectRefFrames = [&refFrames] (std::complex<float>* frame, int numBins)
    {
        refFrames.emplace_back (frame, frame + numBins);
    };

    auto collectTarFrames = [&tarFrames] (std::complex<float>* frame, int numBins)
    {
        tarFrames.emplace_back (frame, frame + numBins);
    };

    // Process analyze window in blocks (emulating DAW playback)
    int blocksProcessed = 0;
    for (int pos = 0; pos < analyzeWindowSamples; pos += kBlockSize)
    {
        int thisBlock = std::min (kBlockSize, analyzeWindowSamples - pos);

        // Feed reference STFT
        refSTFT.processBlock (refMono.data() + pos, thisBlock, collectRefFrames);

        // Feed target STFT
        tarSTFT.processBlock (tarMono.data() + pos, thisBlock, collectTarFrames);

        blocksProcessed++;
    }

    std::cout << "  Processed " << blocksProcessed << " blocks\n";
    std::cout << "  Accumulated " << refFrames.size() << " ref frames, "
              << tarFrames.size() << " target frames\n";

    // =========================================================================
    // PHASE 2: ANALYSIS (triggered after accumulation, like clicking "Analyze")
    // =========================================================================
    std::cout << "\n=== Phase 2: Analysis (User clicked 'Align') ===\n";

    // Extract analyze window for delay detection
    std::vector<float> refAnalyze (refMono.begin(), refMono.begin() + analyzeWindowSamples);
    std::vector<float> tarAnalyze (tarMono.begin(), tarMono.begin() + analyzeWindowSamples);

    // Stage 2a: Delay detection using time-domain cross-correlation
    std::cout << "\n--- Stage 2a: Delay Detection ---\n";

    PhaseAnalyzer delayAnalyzer;
    delayAnalyzer.prepare (sampleRate);
    delayAnalyzer.detectDelayTimeDomain (refAnalyze, tarAnalyze);

    float delaySamples = delayAnalyzer.getDelaySamples();
    float delayMs = delayAnalyzer.getDelayMs();
    float correlation = delayAnalyzer.getCorrelation();
    bool polarityInverted = delayAnalyzer.getPolarityInverted();

    std::cout << "  Delay: " << delaySamples << " samples (" << delayMs << " ms)\n";
    std::cout << "  Correlation: " << correlation << "\n";
    std::cout << "  Polarity: " << (polarityInverted ? "INVERTED (180)" : "Normal") << "\n";

    // Stage 2b: Apply time correction to analyze window for spectral analysis
    std::cout << "\n--- Stage 2b: Time Correction (Analyze Window) ---\n";

    int delayInt = static_cast<int> (std::round (delaySamples));
    float polaritySign = polarityInverted ? -1.0f : 1.0f;

    std::vector<float> tarAnalyzeCorrected (static_cast<size_t> (analyzeWindowSamples), 0.0f);
    for (int i = 0; i < analyzeWindowSamples; ++i)
    {
        int srcIdx = i + delayInt;
        if (srcIdx >= 0 && srcIdx < analyzeWindowSamples)
            tarAnalyzeCorrected[static_cast<size_t> (i)] = tarAnalyze[static_cast<size_t> (srcIdx)] * polaritySign;
    }

    std::cout << "  Applied " << delayInt << " sample shift"
              << (polarityInverted ? " + polarity invert" : "") << " to analyze window\n";

    // Stage 2c: Re-run STFT on time-corrected analyze window for spectral analysis
    std::cout << "\n--- Stage 2c: Spectral Phase Analysis ---\n";

    STFTProcessor refSTFT2;
    STFTProcessor corrSTFT2;
    refSTFT2.prepare (sampleRate, kBlockSize);
    corrSTFT2.prepare (sampleRate, kBlockSize);

    std::vector<std::vector<std::complex<float>>> refFrames2;
    std::vector<std::vector<std::complex<float>>> corrFrames2;

    auto collectRef2 = [&refFrames2] (std::complex<float>* frame, int numBins)
    {
        refFrames2.emplace_back (frame, frame + numBins);
    };

    auto collectCorr2 = [&corrFrames2] (std::complex<float>* frame, int numBins)
    {
        corrFrames2.emplace_back (frame, frame + numBins);
    };

    // Process analyze window again with time-corrected target
    for (int pos = 0; pos < analyzeWindowSamples; pos += kBlockSize)
    {
        int thisBlock = std::min (kBlockSize, analyzeWindowSamples - pos);
        refSTFT2.processBlock (refAnalyze.data() + pos, thisBlock, collectRef2);
        corrSTFT2.processBlock (tarAnalyzeCorrected.data() + pos, thisBlock, collectCorr2);
    }

    PhaseAnalyzer spectralAnalyzer;
    spectralAnalyzer.prepare (sampleRate);
    spectralAnalyzer.setCoherenceThreshold (0.4f);
    spectralAnalyzer.setMaxCorrectionDeg (120.0f);
    spectralAnalyzer.analyzeSpectralPhase (refFrames2, corrFrames2);

    float coherence = spectralAnalyzer.getOverallCoherence();
    float phaseDeg = spectralAnalyzer.getPhaseDegrees();
    const auto& phaseCorrection = spectralAnalyzer.getPhaseCorrection();

    std::cout << "  Analyzed " << refFrames2.size() << " frames\n";
    std::cout << "  Overall coherence: " << coherence << "\n";
    std::cout << "  Average phase correction: " << phaseDeg << " deg\n";

    // =========================================================================
    // PHASE 3: APPLY CORRECTIONS TO FULL FILE
    // =========================================================================
    std::cout << "\n=== Phase 3: Apply Corrections to Full File ===\n";

    // Stage 3a: Apply time correction to full target
    std::cout << "\n--- Stage 3a: Time Correction (Full File) ---\n";

    std::vector<float> timeCorrected (static_cast<size_t> (numSamples), 0.0f);
    for (int i = 0; i < numSamples; ++i)
    {
        int srcIdx = i + delayInt;
        if (srcIdx >= 0 && srcIdx < numSamples)
            timeCorrected[static_cast<size_t> (i)] = tarMono[static_cast<size_t> (srcIdx)] * polaritySign;
    }

    std::cout << "  Applied time correction to full " << numSamples << " samples\n";

    // Stage 3b: Apply spectral phase correction to full file
    std::cout << "\n--- Stage 3b: Spectral Phase Correction (Full File) ---\n";

    PhaseCorrector spectralCorrector;
    spectralCorrector.prepare (sampleRate);
    spectralCorrector.setPhaseCorrection (phaseCorrection);
    spectralCorrector.setTimeCorrectionOn (false);
    spectralCorrector.setPhaseCorrectionOn (true);

    // Process full file through STFT with correction
    STFTProcessor outputSTFT;
    outputSTFT.prepare (sampleRate, kBlockSize);
    const int latency = outputSTFT.getLatencySamples();
    const int paddedLen = numSamples + latency;

    std::vector<float> padded (static_cast<size_t> (paddedLen), 0.0f);
    std::copy (timeCorrected.begin(), timeCorrected.end(), padded.begin());

    auto correctionCallback = [&spectralCorrector] (std::complex<float>* frame, int numBins)
    {
        spectralCorrector.applyPhaseCorrection (frame, numBins);
    };

    int outputBlocksProcessed = 0;
    for (int pos = 0; pos < paddedLen; pos += kBlockSize)
    {
        int thisBlock = std::min (kBlockSize, paddedLen - pos);
        outputSTFT.processBlock (padded.data() + pos, thisBlock, correctionCallback);
        outputBlocksProcessed++;
    }

    // Trim STFT latency
    std::vector<float> aligned (padded.begin() + latency, padded.begin() + latency + numSamples);

    std::cout << "  Processed " << outputBlocksProcessed << " blocks through correction STFT\n";
    std::cout << "  Applied spectral phase correction to full file\n";

    // =========================================================================
    // COMPUTE SUMS AND METRICS
    // =========================================================================
    std::vector<float> sumBefore (static_cast<size_t> (numSamples));
    std::vector<float> sumAfter (static_cast<size_t> (numSamples));

    for (int i = 0; i < numSamples; ++i)
    {
        sumBefore[static_cast<size_t> (i)] = refMono[static_cast<size_t> (i)] + tarMono[static_cast<size_t> (i)];
        sumAfter[static_cast<size_t> (i)]  = refMono[static_cast<size_t> (i)] + aligned[static_cast<size_t> (i)];
    }

    float energyBefore = 0.0f, energyAfter = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        energyBefore += sumBefore[static_cast<size_t> (i)] * sumBefore[static_cast<size_t> (i)];
        energyAfter  += sumAfter[static_cast<size_t> (i)]  * sumAfter[static_cast<size_t> (i)];
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
    std::vector<float> sumTimeOnly (static_cast<size_t> (numSamples));
    for (int i = 0; i < numSamples; ++i)
        sumTimeOnly[static_cast<size_t> (i)] = refMono[static_cast<size_t> (i)] + timeCorrected[static_cast<size_t> (i)];

    float energyTimeOnly = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        energyTimeOnly += sumTimeOnly[static_cast<size_t> (i)] * sumTimeOnly[static_cast<size_t> (i)];
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
            float freqHz = static_cast<float> (k) * binFreqStep;
            float corrRad = (k < static_cast<int> (phaseCorrection.size())) ? phaseCorrection[static_cast<size_t> (k)] : 0.0f;
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
    std::cout << "SUMMARY (VST Emulation)\n";
    std::cout << "========================================\n";
    std::cout << "  Block size:    " << kBlockSize << " samples\n";
    std::cout << "  Analyze window:" << analyzeWindowSec << "s (" << analyzeWindowSamples << " samples)\n";
    std::cout << "  Frames analyzed: " << refFrames2.size() << "\n";
    std::cout << "  ────────────────────────────\n";
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
