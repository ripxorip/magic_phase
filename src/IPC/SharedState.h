#pragma once
#include "PlatformSharedMemory.h"
#include <juce_core/juce_core.h>
#include <complex>
#include <vector>
#include <atomic>
#include <cstdint>

// Magic cookie to detect corrupted/incompatible shared memory
static constexpr uint32_t kMagicCookie = 0x4D475048;  // "MGPH"
static constexpr uint32_t kLayoutVersion = 2;

// Shared memory layout for inter-instance communication
struct SharedHeader
{
    uint32_t magic { 0 };                              // Must be kMagicCookie
    uint32_t layoutVersion { 0 };                      // Must match kLayoutVersion
    std::atomic<uint32_t> version { 0 };               // Change counter for GUI polling
    std::atomic<uint32_t> numInstances { 0 };
    std::atomic<int32_t> referenceSlot { -1 };
    uint32_t sampleRate { 44100 };
    std::atomic<uint32_t> syncCounter { 0 };           // Incremented when target wants fresh capture
    std::atomic<uint32_t> syncAcknowledged { 0 };      // Set by reference when it processes sync
    std::atomic<int64_t> refRawStartSample { 0 };      // Playhead sample when ref started raw accumulation
};

struct InstanceSlot
{
    std::atomic<uint32_t> active { 0 };           // 0=free, 1=active, 2=aligned
    std::atomic<uint64_t> heartbeatMs { 0 };      // Timestamp in milliseconds
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
static constexpr int kMaxRefFrames = 512;  // ~10s @ 48kHz with 1024 hop
static constexpr int kNumBins = 2049;

struct ReferenceSTFTBuffer
{
    std::atomic<uint32_t> writePos { 0 };
    // Stored as interleaved float pairs [real, imag]
    float frames[kMaxRefFrames][kNumBins * 2] {};
};

static constexpr int kMaxRawSamples = 720000;  // 15s @ 48kHz

struct RawSampleBuffer
{
    std::atomic<uint32_t> writePos { 0 };
    float samples[kMaxRawSamples] {};  // mono ring buffer
};

struct SharedMemoryLayout
{
    SharedHeader header;
    InstanceSlot slots[kMaxInstances];
    ReferenceSTFTBuffer refBuffer;
    RawSampleBuffer rawSampleBuffer;
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
    void clearReferenceBuffer();

    // Raw sample sharing (for time-domain delay detection)
    void writeRawSamples (const float* data, int numSamples);
    std::vector<float> readRawSamples (int maxSamples) const;
    void clearRawSampleBuffer();

    // Sync mechanism - target signals reference to start fresh capture
    void requestSync();
    uint32_t getSyncCounter() const;
    void acknowledgeSyncRequest (int64_t refStartSample);
    bool isSyncAcknowledged (uint32_t syncValue) const;
    int64_t getRefRawStartSample() const;

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

    bool isInitialized() const { return shm_.isOpen(); }

private:
    static constexpr const char* kShmName = "/magic_phase";
    static constexpr size_t kShmSize = sizeof (SharedMemoryLayout);
    static constexpr uint64_t kHeartbeatTimeoutMs = 2000;  // 2 seconds

    PlatformSharedMemory shm_;
    SharedMemoryLayout* layout = nullptr;

    void initializeLayout();
    bool validateLayout() const;
    uint64_t getCurrentTimeMs() const;

    static std::atomic<uint32_t> nextInstanceId;
};
