#include "STFTProcessor.h"
#include <cmath>
#include <cstring>

STFTProcessor::STFTProcessor()
    : fft (kFFTOrder)
{
    // Hann window
    for (int i = 0; i < kFFTSize; ++i)
        window[i] = 0.5f * (1.0f - std::cos (2.0f * juce::MathConstants<float>::pi * i / (kFFTSize - 1)));

    inputBuffer.fill (0.0f);
    outputBuffer.fill (0.0f);
    fftData.fill (0.0f);
    overlapBuffer.fill (0.0f);
}

void STFTProcessor::prepare (double sr, int /*maxBlockSize*/)
{
    sampleRate = sr;
    inputWritePos = 0;
    outputReadPos = 0;
    hopCounter = 0;
    overlapWritePos = 0;

    inputBuffer.fill (0.0f);
    outputBuffer.fill (0.0f);
    fftData.fill (0.0f);
    overlapBuffer.fill (0.0f);
    accumulatedFrames.clear();
}

void STFTProcessor::processBlock (float* channelData, int numSamples, FrameCallback callback)
{
    for (int i = 0; i < numSamples; ++i)
    {
        // Push sample into input ring buffer
        inputBuffer[inputWritePos] = channelData[i];
        inputWritePos = (inputWritePos + 1) % kFFTSize;

        ++hopCounter;
        if (hopCounter >= kHopSize)
        {
            hopCounter = 0;
            processFrame (callback);
        }

        // Read from overlap buffer (with latency)
        channelData[i] = overlapBuffer[outputReadPos];
        overlapBuffer[outputReadPos] = 0.0f;
        outputReadPos = (outputReadPos + 1) % (kFFTSize * 2);
    }
}

void STFTProcessor::processFrame (FrameCallback& callback)
{
    // Copy input buffer into FFT data with windowing
    // Read from inputWritePos backwards (the last kFFTSize samples)
    for (int i = 0; i < kFFTSize; ++i)
    {
        int readIdx = (inputWritePos - kFFTSize + i + kFFTSize) % kFFTSize;
        fftData[i] = inputBuffer[readIdx] * window[i];
    }

    // Zero the imaginary part (JUCE FFT uses interleaved real/imag for performRealOnlyForwardTransform)
    // Actually JUCE performRealOnlyForwardTransform takes real data and produces complex
    fft.performRealOnlyForwardTransform (fftData.data(), true);

    // Convert to std::complex for callback
    auto* complexData = reinterpret_cast<std::complex<float>*> (fftData.data());

    // Store frame for analysis
    if (accumulatedFrames.size() < kMaxAccumulatedFrames)
    {
        accumulatedFrames.emplace_back (complexData, complexData + kNumBins);
    }
    else
    {
        // Circular: overwrite oldest
        static int frameWritePos = 0;
        accumulatedFrames[frameWritePos % kMaxAccumulatedFrames].assign (complexData, complexData + kNumBins);
        frameWritePos++;
    }

    // User callback for frequency-domain processing
    if (callback)
        callback (complexData, kNumBins);

    // Inverse FFT
    fft.performRealOnlyInverseTransform (fftData.data());

    // Overlap-add with window
    for (int i = 0; i < kFFTSize; ++i)
    {
        int writeIdx = (overlapWritePos + i) % (kFFTSize * 2);
        overlapBuffer[writeIdx] += fftData[i] * window[i] * (2.0f / 3.0f); // Normalization for 75% overlap Hann
    }

    overlapWritePos = (overlapWritePos + kHopSize) % (kFFTSize * 2);
}
