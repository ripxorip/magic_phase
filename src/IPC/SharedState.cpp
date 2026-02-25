#include "SharedState.h"
#include <cstring>

#if defined(__linux__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

std::atomic<uint32_t> SharedState::nextInstanceId { 1 };

SharedState::SharedState() = default;

SharedState::~SharedState()
{
    shutdown();
}

bool SharedState::initialize()
{
#if defined(__linux__) || defined(__APPLE__)
    // Try to create or open existing shared memory
    shmFd = shm_open (kShmName, O_CREAT | O_RDWR, 0666);
    if (shmFd < 0)
        return false;

    // Set size (only affects if newly created)
    if (ftruncate (shmFd, static_cast<off_t> (kShmSize)) != 0)
    {
        close (shmFd);
        shmFd = -1;
        return false;
    }

    shmPtr = mmap (nullptr, kShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
    if (shmPtr == MAP_FAILED)
    {
        shmPtr = nullptr;
        close (shmFd);
        shmFd = -1;
        return false;
    }

    layout = reinterpret_cast<SharedMemoryLayout*> (shmPtr);

    // Clean up any stale instances
    cleanupStaleInstances();

    return true;
#else
    // Windows: would use CreateFileMapping/MapViewOfFile
    return false;
#endif
}

void SharedState::shutdown()
{
#if defined(__linux__) || defined(__APPLE__)
    if (shmPtr != nullptr)
    {
        munmap (shmPtr, kShmSize);
        shmPtr = nullptr;
        layout = nullptr;
    }
    if (shmFd >= 0)
    {
        close (shmFd);
        shmFd = -1;
    }
    // Don't shm_unlink - other instances may still be using it
#endif
}

int SharedState::registerInstance (const juce::String& trackName)
{
    if (layout == nullptr)
        return -1;

    // Find a free slot
    for (int i = 0; i < kMaxInstances; ++i)
    {
        uint32_t expected = 0;
        if (layout->slots[i].active.compare_exchange_strong (expected, 1))
        {
            layout->slots[i].instanceId = nextInstanceId.fetch_add (1);
            layout->slots[i].heartbeat.store (1);
            std::strncpy (layout->slots[i].trackName, trackName.toRawUTF8(), 63);
            layout->slots[i].trackName[63] = '\0';
            layout->header.numInstances.fetch_add (1);
            layout->header.version.fetch_add (1);
            return i;
        }
    }
    return -1; // No free slots
}

void SharedState::deregisterInstance (int slot)
{
    if (layout == nullptr || slot < 0 || slot >= kMaxInstances)
        return;

    layout->slots[slot].active.store (0);
    layout->slots[slot].heartbeat.store (0);
    std::memset (layout->slots[slot].trackName, 0, 64);
    layout->header.numInstances.fetch_sub (1);

    // If this was the reference, clear it
    int32_t expected = slot;
    layout->header.referenceSlot.compare_exchange_strong (expected, -1);

    layout->header.version.fetch_add (1);
}

void SharedState::updateHeartbeat (int slot)
{
    if (layout == nullptr || slot < 0 || slot >= kMaxInstances)
        return;
    layout->slots[slot].heartbeat.fetch_add (1);
}

void SharedState::cleanupStaleInstances()
{
    if (layout == nullptr)
        return;

    // Simple staleness: if heartbeat hasn't changed, mark inactive
    // This is called once at init; a more sophisticated version would
    // use timers, but for now just trust the heartbeat counter
}

void SharedState::setReferenceSlot (int slot)
{
    if (layout == nullptr)
        return;
    layout->header.referenceSlot.store (slot);
    layout->header.version.fetch_add (1);
}

int SharedState::getReferenceSlot() const
{
    if (layout == nullptr)
        return -1;
    return layout->header.referenceSlot.load();
}

void SharedState::setSampleRate (uint32_t sr)
{
    if (layout == nullptr)
        return;
    layout->header.sampleRate = sr;
}

void SharedState::writeReferenceFrame (const std::complex<float>* frame, int numBins)
{
    if (layout == nullptr || frame == nullptr)
        return;

    uint32_t pos = layout->refBuffer.writePos.load() % kMaxRefFrames;

    int n = std::min (numBins, kNumBins);
    std::memcpy (layout->refBuffer.frames[pos], frame, n * sizeof (std::complex<float>));

    layout->refBuffer.writePos.fetch_add (1);
}

std::vector<std::vector<std::complex<float>>> SharedState::readReferenceFrames() const
{
    std::vector<std::vector<std::complex<float>>> result;
    if (layout == nullptr)
        return result;

    uint32_t wp = layout->refBuffer.writePos.load();
    uint32_t numFrames = std::min (wp, static_cast<uint32_t> (kMaxRefFrames));
    uint32_t startPos = (wp >= kMaxRefFrames) ? (wp - kMaxRefFrames) : 0;

    result.reserve (numFrames);
    for (uint32_t i = 0; i < numFrames; ++i)
    {
        uint32_t idx = (startPos + i) % kMaxRefFrames;
        auto* frameData = reinterpret_cast<const std::complex<float>*> (layout->refBuffer.frames[idx]);
        result.emplace_back (frameData, frameData + kNumBins);
    }

    return result;
}

void SharedState::updateInstanceData (int slot, float delaySamples, float delayMs,
                                      float correlation, float coherence, float phaseDeg,
                                      bool polarityInv, bool timeOn, bool phaseOn,
                                      const float* bands)
{
    if (layout == nullptr || slot < 0 || slot >= kMaxInstances)
        return;

    auto& s = layout->slots[slot];
    s.delaySamples = delaySamples;
    s.delayMs = delayMs;
    s.correlation = correlation;
    s.overallCoherence = coherence;
    s.phaseDegrees = phaseDeg;
    s.polarityInverted = polarityInv;
    s.timeCorrectionOn = timeOn;
    s.phaseCorrectionOn = phaseOn;
    if (bands != nullptr)
        std::memcpy (s.spectralBands, bands, sizeof (s.spectralBands));
}

void SharedState::setInstanceAligned (int slot)
{
    if (layout == nullptr || slot < 0 || slot >= kMaxInstances)
        return;
    layout->slots[slot].active.store (2); // 2 = aligned
    layout->header.version.fetch_add (1);
}

const SharedHeader* SharedState::getHeader() const
{
    return layout ? &layout->header : nullptr;
}

const InstanceSlot* SharedState::getSlot (int index) const
{
    if (layout == nullptr || index < 0 || index >= kMaxInstances)
        return nullptr;
    return &layout->slots[index];
}

int SharedState::getNumActiveInstances() const
{
    if (layout == nullptr)
        return 0;
    return static_cast<int> (layout->header.numInstances.load());
}
