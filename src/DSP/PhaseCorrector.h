#pragma once
#include <complex>
#include <vector>

class PhaseCorrector
{
public:
    static constexpr int kFFTSize = 4096;
    static constexpr int kNumBins = kFFTSize / 2 + 1;

    PhaseCorrector();

    void prepare (double sampleRate);

    // Set correction parameters (called from analysis results)
    void setDelaySamples (float delay);
    void setPolarityInvert (bool invert);
    void setPhaseCorrection (const std::vector<float>& correction);

    void setTimeCorrectionOn (bool on) { timeCorrectionOn = on; }
    void setPhaseCorrectionOn (bool on) { phaseCorrectionOn = on; }
    bool getTimeCorrectionOn() const { return timeCorrectionOn; }
    bool getPhaseCorrectionOn() const { return phaseCorrectionOn; }

    // Apply corrections to an STFT frame (called in frequency domain)
    void applyTimeCorrection (std::complex<float>* frame, int numBins);
    void applyPhaseCorrection (std::complex<float>* frame, int numBins);

private:
    void recomputeTimeCorrectionFactors();

    double sampleRate = 44100.0;
    float delaySamples = 0.0f;
    bool polarityInvert = false;
    bool timeCorrectionOn = false;
    bool phaseCorrectionOn = false;

    // Pre-computed per-bin correction factors
    std::vector<std::complex<float>> timeCorrectionFactors;
    std::vector<std::complex<float>> phaseCorrectionFactors;
};
