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

// Time-domain cross-correlation for delay detection (matches Python approach)
struct DelayResult
{
    int delaySamples;
    float correlation;
    bool polarityInverted;
};

static DelayResult detectDelayTimeDomain (const std::vector<float>& ref,
                                           const std::vector<float>& target,
                                           double sampleRate,
                                           float maxDelayMs = 50.0f)
{
    const int maxDelaySamples = static_cast<int> (maxDelayMs * sampleRate / 1000.0f);
    const int n = static_cast<int> (std::min (ref.size(), target.size()));

    // Compute energy for normalization
    double refEnergy = 0.0, tarEnergy = 0.0;
    for (int i = 0; i < n; ++i)
    {
        refEnergy += ref[i] * ref[i];
        tarEnergy += target[i] * target[i];
    }
    double normFactor = std::sqrt (refEnergy * tarEnergy);

    // Search for best correlation within ±maxDelaySamples
    double bestCorr = 0.0;
    int bestLag = 0;

    for (int lag = -maxDelaySamples; lag <= maxDelaySamples; ++lag)
    {
        double sum = 0.0;
        int count = 0;

        for (int i = 0; i < n; ++i)
        {
            int j = i + lag;
            if (j >= 0 && j < n)
            {
                sum += static_cast<double> (target[i]) * ref[j];
                count++;
            }
        }

        if (std::abs (sum) > std::abs (bestCorr))
        {
            bestCorr = sum;
            bestLag = lag;
        }
    }

    DelayResult result;
    // Negate to match Python's scipy.correlate convention:
    // positive delay means target is delayed relative to reference
    result.delaySamples = -bestLag;
    result.polarityInverted = (bestCorr < 0.0);
    result.correlation = static_cast<float> (std::abs (bestCorr) / (normFactor + 1e-10));

    return result;
}

//==============================================================================
// Test FFT round-trip to verify JUCE scaling
static void testFFTRoundTrip()
{
    std::cout << "--- FFT Round-Trip Test ---\n";

    constexpr int N = 4096;
    juce::dsp::FFT fft (12);  // 2^12 = 4096

    // Test 1: Simple sine wave
    std::array<float, N * 2> data {};
    std::array<float, N> original {};

    for (int i = 0; i < N; ++i)
    {
        original[i] = std::sin (2.0f * juce::MathConstants<float>::pi * 440.0f * i / 48000.0f);
        data[i] = original[i];
    }

    fft.performRealOnlyForwardTransform (data.data(), false);
    fft.performRealOnlyInverseTransform (data.data());

    float maxErr = 0.0f, sumErr2 = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        float err = std::abs (data[i] - original[i]);
        maxErr = std::max (maxErr, err);
        sumErr2 += err * err;
    }
    float rmsErr = std::sqrt (sumErr2 / N);

    std::cout << "  Sine FFT round-trip:\n";
    std::cout << "    max error: " << maxErr << "\n";
    std::cout << "    rms error: " << rmsErr << " (" << (20.0f * std::log10 (rmsErr + 1e-10f)) << " dB)\n";

    // Check scale factor (compare first non-zero sample)
    std::cout << "    scale factor: " << data[100] / original[100] << "\n";
}

//==============================================================================
// Verify COLA sum for Hann window with 75% overlap
static void testCOLASum()
{
    std::cout << "\n--- COLA Verification ---\n";

    constexpr int N = 4096;
    constexpr int hop = N / 4;  // 75% overlap

    // Periodic Hann window (same as STFTProcessor)
    std::array<float, N> window {};
    for (int i = 0; i < N; ++i)
        window[i] = 0.5f * (1.0f - std::cos (2.0f * juce::MathConstants<float>::pi * i / N));

    // Compute COLA sum at several positions
    // For 75% overlap, 4 frames overlap at any point in steady state
    // COLA sum = sum of w[position] from 4 overlapping frames
    std::cout << "  Window type: Periodic Hann (" << N << " samples, hop=" << hop << ")\n";

    // COLA sum of w (analysis-only)
    float colaSum_w = 0.0f;
    for (int f = 0; f < 4; ++f)
        colaSum_w += window[f * hop];  // Sample at position 0 of output
    std::cout << "  COLA sum of w: " << colaSum_w << " (expected: 2.0 for OLA)\n";

    // COLA sum of w^2 (analysis + synthesis windowing = WOLA)
    float colaSum_w2 = 0.0f;
    for (int f = 0; f < 4; ++f)
        colaSum_w2 += window[f * hop] * window[f * hop];
    std::cout << "  COLA sum of w^2: " << colaSum_w2 << " (expected: 1.5 for WOLA)\n";

    // Check COLA sum at different positions within the hop
    float minCola = 1e10f, maxCola = 0.0f;
    for (int pos = 0; pos < hop; ++pos)
    {
        float cola = 0.0f;
        for (int f = 0; f < 4; ++f)
        {
            int winIdx = pos + f * hop;
            if (winIdx < N)
                cola += window[winIdx] * window[winIdx];
        }
        minCola = std::min (minCola, cola);
        maxCola = std::max (maxCola, cola);
    }
    std::cout << "  COLA range: min=" << minCola << " max=" << maxCola << "\n";
    std::cout << "  Ideal scale factor: " << (1.0f / colaSum_w2) << "\n";
}

//==============================================================================
int main (int argc, char* argv[])
{
    // Test FFT first
    testFFTRoundTrip();
    testCOLASum();

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
    // SANITY CHECK: STFT passthrough should be transparent
    // =========================================================================
    std::cout << "--- Passthrough Check ---\n";
    {
        std::vector<std::vector<std::complex<float>>> dummyFrames;
        std::vector<float> passthrough = stftPass (tarMono, dummyFrames, nullptr);

        // Measure reconstruction error
        float maxErr = 0.0f;
        float sumErr2 = 0.0f;
        int maxErrIdx = 0;
        for (int i = 0; i < numSamples; ++i)
        {
            float err = std::abs (passthrough[i] - tarMono[i]);
            if (err > maxErr)
            {
                maxErr = err;
                maxErrIdx = i;
            }
            sumErr2 += err * err;
        }
        float rmsErr = std::sqrt (sumErr2 / numSamples);
        std::cout << "  STFT passthrough error: max=" << maxErr
                  << " rms=" << rmsErr
                  << " (" << (20.0f * std::log10 (rmsErr + 1e-10f)) << " dB)\n";
        std::cout << "  Max error at sample " << maxErrIdx << " (t=" << (maxErrIdx / sampleRate) << "s)\n";

        // Error distribution: first 10%, middle 80%, last 10%
        float sumErr2_first = 0.0f, sumErr2_mid = 0.0f, sumErr2_last = 0.0f;
        int first10 = numSamples / 10;
        int last10 = numSamples - first10;
        for (int i = 0; i < numSamples; ++i)
        {
            float err = passthrough[i] - tarMono[i];
            float err2 = err * err;
            if (i < first10)
                sumErr2_first += err2;
            else if (i >= last10)
                sumErr2_last += err2;
            else
                sumErr2_mid += err2;
        }
        std::cout << "  Error distribution:\n";
        std::cout << "    First 10%: " << (20.0f * std::log10 (std::sqrt (sumErr2_first / first10) + 1e-10f)) << " dB\n";
        std::cout << "    Middle 80%: " << (20.0f * std::log10 (std::sqrt (sumErr2_mid / (numSamples - 2 * first10)) + 1e-10f)) << " dB\n";
        std::cout << "    Last 10%: " << (20.0f * std::log10 (std::sqrt (sumErr2_last / first10) + 1e-10f)) << " dB\n";
    }

    // =========================================================================
    // STAGE 1: Delay detection using time-domain cross-correlation
    // =========================================================================
    std::cout << "\n--- Stage 1: Delay Detection (Time-Domain) ---\n";

    // Use time-domain cross-correlation (matches Python approach)
    auto delayResult = detectDelayTimeDomain (refMono, tarMono, sampleRate);

    float delaySamples = static_cast<float> (delayResult.delaySamples);
    float delayMs = delaySamples / static_cast<float> (sampleRate) * 1000.0f;
    float correlation = delayResult.correlation;
    bool polarityInverted = delayResult.polarityInverted;

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
    spectralAnalyzer.analyze (refFrames2, corrFrames2);

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
