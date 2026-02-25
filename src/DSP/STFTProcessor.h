#pragma once
#include <juce_dsp/juce_dsp.h>
#include <complex>
#include <functional>
#include <vector>

class STFTProcessor
{
public:
    static constexpr int kFFTOrder = 12;           // 2^12 = 4096
    static constexpr int kFFTSize = 1 << kFFTOrder; // 4096
    static constexpr int kHopSize = kFFTSize / 4;   // 1024 (75% overlap)
    static constexpr int kNumBins = kFFTSize / 2 + 1; // 2049

    using FrameCallback = std::function<void (std::complex<float>* frame, int numBins)>;

    STFTProcessor();

    void prepare (double sampleRate, int maxBlockSize);
    void processBlock (float* channelData, int numSamples, FrameCallback callback);

    // Get accumulated frames for analysis
    const std::vector<std::vector<std::complex<float>>>& getAccumulatedFrames() const { return accumulatedFrames; }
    void clearAccumulatedFrames() { accumulatedFrames.clear(); }

    // Latency = FFTSize for proper overlap-add timing
    // (we need all 4 overlapping frames to contribute before reading)
    int getLatencySamples() const { return kFFTSize; }

private:
    void processFrame (FrameCallback& callback);

    juce::dsp::FFT fft;
    std::array<float, kFFTSize> window {};
    std::array<float, kFFTSize> inputBuffer {};
    std::array<float, kFFTSize> outputBuffer {};
    std::array<float, kFFTSize * 2> fftData {};  // interleaved complex

    int inputWritePos = 0;
    int outputReadPos = 0;
    int hopCounter = 0;

    // Overlap-add accumulation buffer
    std::array<float, kFFTSize * 2> overlapBuffer {};
    int overlapWritePos = 0;

    // Store frames for analysis
    std::vector<std::vector<std::complex<float>>> accumulatedFrames;
    static constexpr int kMaxAccumulatedFrames = 128;

    double sampleRate = 44100.0;
};
