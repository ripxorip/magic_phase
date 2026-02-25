#include "PhaseAnalyzer.h"
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

void PhaseAnalyzer::analyze (const std::vector<std::vector<std::complex<float>>>& refFrames,
                             const std::vector<std::vector<std::complex<float>>>& targetFrames)
{
    if (refFrames.empty() || targetFrames.empty())
        return;

    // Step 1: Detect time delay via cross-correlation in frequency domain
    detectDelay (refFrames, targetFrames);

    // Step 2: Compute per-frequency spectral phase correction
    computeSpectralPhase (refFrames, targetFrames);
}

void PhaseAnalyzer::detectDelay (const std::vector<std::vector<std::complex<float>>>& refFrames,
                                 const std::vector<std::vector<std::complex<float>>>& targetFrames)
{
    // Average cross-spectrum across frames for delay detection
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

    // Inverse FFT of cross-spectrum to get cross-correlation
    // We'll use a simple approach: find peak of |cross-spectrum| to get GCC-PHAT-like delay
    // For simplicity, find the phase slope of the cross-spectrum

    // Actually, compute the weighted average phase slope for delay estimation
    // delay = -d(phase)/d(omega) * sr / (2*pi)
    // Use a least-squares fit of the phase vs frequency

    float sumW = 0.0f;
    float sumWF = 0.0f;
    float sumWP = 0.0f;
    float sumWFF = 0.0f;
    float sumWFP = 0.0f;

    float prevPhase = 0.0f;
    float unwrappedPhase = 0.0f;

    for (int k = 1; k < numBins; ++k)
    {
        float mag = std::abs (crossSpectrum[k]);
        float phase = std::arg (crossSpectrum[k]);

        // Unwrap phase
        if (k == 1)
        {
            unwrappedPhase = phase;
        }
        else
        {
            float diff = phase - prevPhase;
            while (diff > kPi) diff -= kTwoPi;
            while (diff < -kPi) diff += kTwoPi;
            unwrappedPhase += diff;
        }
        prevPhase = phase;

        float freq = static_cast<float> (k);
        float w = mag * mag; // Weight by magnitude squared

        sumW += w;
        sumWF += w * freq;
        sumWP += w * unwrappedPhase;
        sumWFF += w * freq * freq;
        sumWFP += w * freq * unwrappedPhase;
    }

    if (sumW > 1e-10f)
    {
        // Weighted least squares: phase = slope * freq + intercept
        float denom = sumW * sumWFF - sumWF * sumWF;
        if (std::abs (denom) > 1e-10f)
        {
            float slope = (sumW * sumWFP - sumWF * sumWP) / denom;
            // slope = 2*pi*delay/N, so delay = slope * N / (2*pi)
            delaySamples = slope * kFFTSize / kTwoPi;
        }
    }

    delayMs = delaySamples / static_cast<float> (sampleRate) * 1000.0f;

    // Correlation coefficient
    float denom = std::sqrt (refEnergy * tarEnergy);
    if (denom > 1e-10f)
    {
        float crossMag = 0.0f;
        for (int k = 0; k < numBins; ++k)
            crossMag += std::abs (crossSpectrum[k]);
        correlation = crossMag / denom;
    }

    // Check polarity: if the real part of the average cross-spectrum is negative, polarity is inverted
    float realSum = 0.0f;
    for (int k = 0; k < numBins; ++k)
        realSum += crossSpectrum[k].real();
    polarityInverted = (realSum < 0.0f);
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
