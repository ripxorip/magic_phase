#pragma once
#include <complex>
#include <vector>
#include <array>
#include <cstdint>

class PhaseAnalyzer
{
public:
    static constexpr int kFFTSize = 4096;
    static constexpr int kNumBins = kFFTSize / 2 + 1;
    static constexpr int kNumSpectralBands = 48;

    PhaseAnalyzer();

    void prepare (double sampleRate);

    void setCoherenceThreshold (float threshold) { coherenceThreshold = threshold; }
    void setMaxCorrectionDeg (float maxDeg) { maxCorrectionDeg = maxDeg; }
    void setMaxDelayMs (float maxMs) { maxDelayMs = maxMs; }

    // Detect delay using time-domain cross-correlation (more accurate than spectral method)
    void detectDelayTimeDomain (const std::vector<float>& refSamples,
                                const std::vector<float>& targetSamples);

    // Compute spectral phase correction from STFT frames
    // Call this AFTER detectDelayTimeDomain, using time-aligned frames
    void analyzeSpectralPhase (const std::vector<std::vector<std::complex<float>>>& refFrames,
                               const std::vector<std::vector<std::complex<float>>>& targetFrames);

    // Legacy: Run full analysis using spectral-only method (less accurate delay detection)
    void analyze (const std::vector<std::vector<std::complex<float>>>& refFrames,
                  const std::vector<std::vector<std::complex<float>>>& targetFrames);

    // Results
    float getDelaySamples() const { return delaySamples; }
    float getDelayMs() const { return delayMs; }
    float getCorrelation() const { return correlation; }
    float getOverallCoherence() const { return overallCoherence; }
    float getPhaseDegrees() const { return phaseDegrees; }
    bool getPolarityInverted() const { return polarityInverted; }
    const float* getSpectralBands() const { return spectralBands.data(); }
    const std::vector<float>& getPhaseCorrection() const { return phaseCorrection; }

private:
    void detectDelay (const std::vector<std::vector<std::complex<float>>>& refFrames,
                      const std::vector<std::vector<std::complex<float>>>& targetFrames);

    void computeSpectralPhase (const std::vector<std::vector<std::complex<float>>>& refFrames,
                               const std::vector<std::vector<std::complex<float>>>& targetFrames);

    void gaussianSmooth (std::vector<float>& data, float sigma);

    double sampleRate = 44100.0;
    float coherenceThreshold = 0.4f;
    float maxCorrectionDeg = 120.0f;
    float maxDelayMs = 50.0f;

    // Results
    float delaySamples = 0.0f;
    float delayMs = 0.0f;
    float correlation = 0.0f;
    float overallCoherence = 0.0f;
    float phaseDegrees = 0.0f;
    bool polarityInverted = false;

    std::vector<float> phaseCorrection;      // Per-bin phase correction (radians)
    std::vector<float> coherencePerBin;      // Per-bin coherence
    std::array<float, kNumSpectralBands> spectralBands {}; // For GUI visualization
};
