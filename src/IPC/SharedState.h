#pragma once
#include <juce_core/juce_core.h>
#include <complex>
#include <vector>
#include <atomic>
#include <cstdint>

// Shared memory layout for inter-instance communication
struct SharedHeader
{
    std::atomic<uint32_t> version { 0 };
    std::atomic<uint32_t> numInstances { 0 };
    std::atomic<int32_t> referenceSlot { -1 };
    uint32_t sampleRate { 44100 };
};

struct InstanceSlot
{
    std::atomic<uint32_t> active { 0 };      // 0=free, 1=active, 2=aligned
    std::atomic<uint32_t> heartbeat { 0 };
    char trackName[64] {};
    uint32_t instanceId { 0 };
    float delaySamples { 0.0f };
    float delayMs { 0.0f };
    float correlation { 0.0f };
    float overallCoherence { 0.0f };
    float phaseDegrees { 0.0f };
    bool polarityInverted { false };
    bool timeCorrectionOn { false };
    bool phaseCorrectionOn { false };
    float spectralBands[48] {};
};

static constexpr int kMaxInstances = 16;
static constexpr int kMaxRefFrames = 128;
static constexpr int kNumBins = 2049;

struct ReferenceSTFTBuffer
{
    std::atomic<uint32_t> writePos { 0 };
    // Stored as interleaved float pairs [real, imag]
    float frames[kMaxRefFrames][kNumBins * 2] {};
};

struct SharedMemoryLayout
{
    SharedHeader header;
    InstanceSlot slots[kMaxInstances];
    ReferenceSTFTBuffer refBuffer;
};

class SharedState
{
public:
    SharedState();
    ~SharedState();

    bool initialize();
    void shutdown();

    // Instance management
    int registerInstance (const juce::String& trackName);
    void deregisterInstance (int slot);
    void updateHeartbeat (int slot);
    void cleanupStaleInstances();

    // Reference
    void setReferenceSlot (int slot);
    int getReferenceSlot() const;

    void setSampleRate (uint32_t sr);

    // Reference STFT frame sharing
    void writeReferenceFrame (const std::complex<float>* frame, int numBins);
    std::vector<std::vector<std::complex<float>>> readReferenceFrames() const;

    // Instance data
    void updateInstanceData (int slot, float delaySamples, float delayMs,
                             float correlation, float coherence, float phaseDeg,
                             bool polarityInv, bool timeOn, bool phaseOn,
                             const float* bands);
    void setInstanceAligned (int slot);

    // Read all slots (for GUI)
    const SharedHeader* getHeader() const;
    const InstanceSlot* getSlot (int index) const;
    int getNumActiveInstances() const;

    bool isInitialized() const { return shmPtr != nullptr; }

private:
    static constexpr const char* kShmName = "/magic_phase";
    static constexpr size_t kShmSize = sizeof (SharedMemoryLayout);

    int shmFd = -1;
    void* shmPtr = nullptr;
    SharedMemoryLayout* layout = nullptr;

    static std::atomic<uint32_t> nextInstanceId;
};
