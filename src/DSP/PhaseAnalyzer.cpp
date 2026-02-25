#include "PhaseAnalyzer.h"
#include <juce_dsp/juce_dsp.h>
#include <cmath>
#include <algorithm>
#include <numeric>

static constexpr float kPi = 3.14159265358979323846f;
static constexpr float kTwoPi = 2.0f * kPi;

PhaseAnalyzer::PhaseAnalyzer()
{
    phaseCorrection.resize (kNumBins, 0.0f);
    coherencePerBin.resize (kNumBins, 0.0f);
    spectralBands.fill (0.0f);
}

void PhaseAnalyzer::prepare (double sr)
{
    sampleRate = sr;
    phaseCorrection.assign (kNumBins, 0.0f);
    coherencePerBin.assign (kNumBins, 0.0f);
    spectralBands.fill (0.0f);
    delaySamples = 0.0f;
    delayMs = 0.0f;
    correlation = 0.0f;
    overallCoherence = 0.0f;
    phaseDegrees = 0.0f;
    polarityInverted = false;
}

void PhaseAnalyzer::detectDelayTimeDomain (const std::vector<float>& refSamples,
                                            const std::vector<float>& targetSamples)
{
    const int maxDelaySamples = static_cast<int> (maxDelayMs * sampleRate / 1000.0f);
    const int n = static_cast<int> (std::min (refSamples.size(), targetSamples.size()));

    if (n < maxDelaySamples * 2)
        return;  // Not enough samples

    // Compute energy for normalization
    double refEnergy = 0.0, tarEnergy = 0.0;
    for (int i = 0; i < n; ++i)
    {
        refEnergy += static_cast<double> (refSamples[i]) * refSamples[i];
        tarEnergy += static_cast<double> (targetSamples[i]) * targetSamples[i];
    }
    double normFactor = std::sqrt (refEnergy * tarEnergy);

    // Search for best correlation within ±maxDelaySamples
    double bestCorr = 0.0;
    int bestLag = 0;

    for (int lag = -maxDelaySamples; lag <= maxDelaySamples; ++lag)
    {
        double sum = 0.0;

        for (int i = 0; i < n; ++i)
        {
            int j = i + lag;
            if (j >= 0 && j < n)
                sum += static_cast<double> (targetSamples[i]) * refSamples[j];
        }

        if (std::abs (sum) > std::abs (bestCorr))
        {
            bestCorr = sum;
            bestLag = lag;
        }
    }

    // Negate lag to match convention: positive delay means target is delayed
    delaySamples = static_cast<float> (-bestLag);
    delayMs = delaySamples / static_cast<float> (sampleRate) * 1000.0f;
    polarityInverted = (bestCorr < 0.0);
    correlation = static_cast<float> (std::abs (bestCorr) / (normFactor + 1e-10));
}

void PhaseAnalyzer::analyzeSpectralPhase (const std::vector<std::vector<std::complex<float>>>& refFrames,
                                           const std::vector<std::vector<std::complex<float>>>& targetFrames)
{
    if (refFrames.empty() || targetFrames.empty())
        return;

    computeSpectralPhase (refFrames, targetFrames);
}

void PhaseAnalyzer::analyze (const std::vector<std::vector<std::complex<float>>>& refFrames,
                             const std::vector<std::vector<std::complex<float>>>& targetFrames)
{
    if (refFrames.empty() || targetFrames.empty())
        return;

    // Legacy: Use spectral method for delay detection (less accurate)
    detectDelay (refFrames, targetFrames);

    // Compute per-frequency spectral phase correction
    computeSpectralPhase (refFrames, targetFrames);
}

void PhaseAnalyzer::detectDelay (const std::vector<std::vector<std::complex<float>>>& refFrames,
                                 const std::vector<std::vector<std::complex<float>>>& targetFrames)
{
    // Average cross-spectrum across frames
    const int numFrames = static_cast<int> (std::min (refFrames.size(), targetFrames.size()));
    const int numBins = kNumBins;

    std::vector<std::complex<float>> crossSpectrum (numBins, {0.0f, 0.0f});
    float refEnergy = 0.0f;
    float tarEnergy = 0.0f;

    for (int f = 0; f < numFrames; ++f)
    {
        for (int k = 0; k < numBins; ++k)
        {
            crossSpectrum[k] += targetFrames[f][k] * std::conj (refFrames[f][k]);
            refEnergy += std::norm (refFrames[f][k]);
            tarEnergy += std::norm (targetFrames[f][k]);
        }
    }

    // IFFT the averaged cross-spectrum to get cross-correlation
    // Pack into interleaved real/imag for JUCE FFT
    juce::dsp::FFT fft (12); // 2^12 = 4096
    std::array<float, kFFTSize * 2> fftData {};

    for (int k = 0; k < numBins; ++k)
    {
        fftData[k * 2]     = crossSpectrum[k].real();
        fftData[k * 2 + 1] = crossSpectrum[k].imag();
    }

    fft.performRealOnlyInverseTransform (fftData.data());

    // Search for the peak within ±maxDelay (50ms)
    const int maxDelaySamples = static_cast<int> (50.0f * static_cast<float> (sampleRate) / 1000.0f);
    const int searchRange = std::min (maxDelaySamples, kFFTSize / 2);

    float bestVal = 0.0f;
    int bestLag = 0;

    // Search positive lags (target is delayed relative to ref): 0..searchRange
    for (int lag = 0; lag <= searchRange; ++lag)
    {
        float val = fftData[lag];
        if (std::abs (val) > std::abs (bestVal))
        {
            bestVal = val;
            bestLag = lag;
        }
    }

    // Search negative lags (target is early relative to ref): wrap-around at end of IFFT
    for (int lag = 1; lag <= searchRange; ++lag)
    {
        float val = fftData[kFFTSize - lag];
        if (std::abs (val) > std::abs (bestVal))
        {
            bestVal = val;
            bestLag = -lag;
        }
    }

    delaySamples = static_cast<float> (bestLag);
    delayMs = delaySamples / static_cast<float> (sampleRate) * 1000.0f;

    // Polarity: negative peak means inverted
    polarityInverted = (bestVal < 0.0f);

    // Correlation coefficient: |peak| / sqrt(refEnergy * tarEnergy)
    float denom = std::sqrt (refEnergy * tarEnergy);
    if (denom > 1e-10f)
        correlation = std::abs (bestVal) / denom;
}

void PhaseAnalyzer::computeSpectralPhase (const std::vector<std::vector<std::complex<float>>>& refFrames,
                                          const std::vector<std::vector<std::complex<float>>>& targetFrames)
{
    const int numFrames = static_cast<int> (std::min (refFrames.size(), targetFrames.size()));
    const int numBins = kNumBins;

    // Compute phase difference per bin using complex averaging (circular statistics)
    // phase_diff = arg(Ztar * conj(Zref))
    // weighted by magnitude product
    std::vector<std::complex<float>> weightedComplex (numBins, {0.0f, 0.0f});
    std::vector<float> magProductSum (numBins, 0.0f);

    for (int f = 0; f < numFrames; ++f)
    {
        for (int k = 0; k < numBins; ++k)
        {
            std::complex<float> crossSpec = targetFrames[f][k] * std::conj (refFrames[f][k]);
            float phaseDiff = std::arg (crossSpec);
            float magProduct = std::abs (refFrames[f][k]) * std::abs (targetFrames[f][k]);

            weightedComplex[k] += magProduct * std::exp (std::complex<float> (0.0f, phaseDiff));
            magProductSum[k] += magProduct;
        }
    }

    // Extract average phase and coherence per bin
    std::vector<float> avgPhase (numBins, 0.0f);
    coherencePerBin.assign (numBins, 0.0f);

    for (int k = 0; k < numBins; ++k)
    {
        avgPhase[k] = std::arg (weightedComplex[k]);
        if (magProductSum[k] > 1e-10f)
            coherencePerBin[k] = std::abs (weightedComplex[k]) / magProductSum[k];
    }

    // Smooth into frequency bands (log-spaced, like the Python prototype)
    constexpr int nBands = 64;
    constexpr float lowFreq = 20.0f;
    float highFreq = static_cast<float> (sampleRate) / 2.0f;
    float binFreqStep = static_cast<float> (sampleRate) / kFFTSize;

    // Log-spaced band edges
    std::vector<float> bandEdges (nBands + 1);
    float logLow = std::log10 (std::max (lowFreq, 1.0f));
    float logHigh = std::log10 (std::min (highFreq, static_cast<float> (sampleRate) / 2.0f));
    for (int i = 0; i <= nBands; ++i)
        bandEdges[i] = std::pow (10.0f, logLow + (logHigh - logLow) * i / nBands);

    std::vector<float> smoothedPhase (numBins, 0.0f);
    std::vector<float> smoothedCoherence (numBins, 0.0f);

    for (int b = 0; b < nBands; ++b)
    {
        float bandLow = bandEdges[b];
        float bandHigh = bandEdges[b + 1];

        std::complex<float> bandWeightedComplex (0.0f, 0.0f);
        float bandCohSum = 0.0f;
        float bandWeightSum = 0.0f;
        int bandCount = 0;

        for (int k = 0; k < numBins; ++k)
        {
            float freq = k * binFreqStep;
            if (freq >= bandLow && freq < bandHigh)
            {
                float w = coherencePerBin[k] * coherencePerBin[k]; // Square to emphasize high-coherence
                bandWeightedComplex += w * std::exp (std::complex<float> (0.0f, avgPhase[k]));
                bandWeightSum += w;
                bandCohSum += coherencePerBin[k];
                bandCount++;
            }
        }

        float bandAvgPhase = 0.0f;
        float bandAvgCoherence = 0.0f;
        if (bandWeightSum > 1e-10f)
            bandAvgPhase = std::arg (bandWeightedComplex);
        if (bandCount > 0)
            bandAvgCoherence = bandCohSum / bandCount;

        // Apply to all bins in this band
        for (int k = 0; k < numBins; ++k)
        {
            float freq = k * binFreqStep;
            if (freq >= bandLow && freq < bandHigh)
            {
                smoothedPhase[k] = bandAvgPhase;
                smoothedCoherence[k] = bandAvgCoherence;
            }
        }
    }

    // Apply coherence-weighted confidence
    std::vector<float> confidence (numBins);
    for (int k = 0; k < numBins; ++k)
    {
        float c = (smoothedCoherence[k] - coherenceThreshold) / (1.0f - coherenceThreshold);
        confidence[k] = std::clamp (c, 0.0f, 1.0f);
    }

    // Gaussian smooth the confidence
    gaussianSmooth (confidence, 3.0f);

    // Apply confidence weighting
    phaseCorrection.resize (numBins);
    float maxCorrRad = maxCorrectionDeg * kPi / 180.0f;

    for (int k = 0; k < numBins; ++k)
    {
        phaseCorrection[k] = smoothedPhase[k] * confidence[k];
        phaseCorrection[k] = std::clamp (phaseCorrection[k], -maxCorrRad, maxCorrRad);
    }

    // Gaussian smooth the final correction
    gaussianSmooth (phaseCorrection, 2.0f);

    // Compute overall coherence
    float cohSum = 0.0f;
    for (int k = 0; k < numBins; ++k)
        cohSum += coherencePerBin[k];
    overallCoherence = cohSum / numBins;

    // Compute average phase correction in degrees
    float phaseSum = 0.0f;
    for (int k = 0; k < numBins; ++k)
        phaseSum += phaseCorrection[k];
    phaseDegrees = (phaseSum / numBins) * 180.0f / kPi;

    // Compute spectral bands for GUI visualization (48 bands)
    for (int b = 0; b < kNumSpectralBands; ++b)
    {
        float bandLow = lowFreq * std::pow (highFreq / lowFreq, static_cast<float> (b) / kNumSpectralBands);
        float bandHigh = lowFreq * std::pow (highFreq / lowFreq, static_cast<float> (b + 1) / kNumSpectralBands);

        float bandCoh = 0.0f;
        int count = 0;
        for (int k = 0; k < numBins; ++k)
        {
            float freq = k * binFreqStep;
            if (freq >= bandLow && freq < bandHigh)
            {
                bandCoh += coherencePerBin[k];
                count++;
            }
        }
        spectralBands[b] = (count > 0) ? (bandCoh / count) : 0.0f;
    }
}

void PhaseAnalyzer::gaussianSmooth (std::vector<float>& data, float sigma)
{
    if (data.empty() || sigma <= 0.0f)
        return;

    int radius = static_cast<int> (std::ceil (sigma * 3.0f));
    std::vector<float> kernel (2 * radius + 1);
    float kernelSum = 0.0f;

    for (int i = -radius; i <= radius; ++i)
    {
        kernel[i + radius] = std::exp (-0.5f * (i * i) / (sigma * sigma));
        kernelSum += kernel[i + radius];
    }
    for (auto& k : kernel)
        k /= kernelSum;

    std::vector<float> smoothed (data.size());
    for (int i = 0; i < static_cast<int> (data.size()); ++i)
    {
        float val = 0.0f;
        for (int j = -radius; j <= radius; ++j)
        {
            int idx = std::clamp (i + j, 0, static_cast<int> (data.size()) - 1);
            val += data[idx] * kernel[j + radius];
        }
        smoothed[i] = val;
    }
    data = smoothed;
}
