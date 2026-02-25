#include "PhaseCorrector.h"
#include <cmath>

static constexpr float kTwoPi = 2.0f * 3.14159265358979323846f;

PhaseCorrector::PhaseCorrector()
{
    timeCorrectionFactors.resize (kNumBins, {1.0f, 0.0f});
    phaseCorrectionFactors.resize (kNumBins, {1.0f, 0.0f});
}

void PhaseCorrector::prepare (double sr)
{
    sampleRate = sr;
    delaySamples = 0.0f;
    polarityInvert = false;
    timeCorrectionOn = false;
    phaseCorrectionOn = false;
    timeCorrectionFactors.assign (kNumBins, {1.0f, 0.0f});
    phaseCorrectionFactors.assign (kNumBins, {1.0f, 0.0f});
}

void PhaseCorrector::setDelaySamples (float delay)
{
    delaySamples = delay;
    recomputeTimeCorrectionFactors();
}

void PhaseCorrector::setPolarityInvert (bool invert)
{
    polarityInvert = invert;
    recomputeTimeCorrectionFactors();
}

void PhaseCorrector::setPhaseCorrection (const std::vector<float>& correction)
{
    phaseCorrectionFactors.resize (kNumBins);
    for (int k = 0; k < kNumBins && k < static_cast<int> (correction.size()); ++k)
    {
        // exp(-j * correction[k]) to rotate phase
        float angle = -correction[k];
        phaseCorrectionFactors[k] = std::polar (1.0f, angle);
    }
}

void PhaseCorrector::recomputeTimeCorrectionFactors()
{
    timeCorrectionFactors.resize (kNumBins);

    float polaritySign = polarityInvert ? -1.0f : 1.0f;

    for (int k = 0; k < kNumBins; ++k)
    {
        // exp(2j*pi*k*delay/N) for sub-sample time correction
        float angle = kTwoPi * k * delaySamples / kFFTSize;
        timeCorrectionFactors[k] = polaritySign * std::polar (1.0f, angle);
    }
}

void PhaseCorrector::applyTimeCorrection (std::complex<float>* frame, int numBins)
{
    if (! timeCorrectionOn)
        return;

    int n = std::min (numBins, static_cast<int> (timeCorrectionFactors.size()));
    for (int k = 0; k < n; ++k)
        frame[k] *= timeCorrectionFactors[k];
}

void PhaseCorrector::applyPhaseCorrection (std::complex<float>* frame, int numBins)
{
    if (! phaseCorrectionOn)
        return;

    int n = std::min (numBins, static_cast<int> (phaseCorrectionFactors.size()));
    for (int k = 0; k < n; ++k)
        frame[k] *= phaseCorrectionFactors[k];
}
