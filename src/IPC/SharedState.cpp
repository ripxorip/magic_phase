#include "SharedState.h"
#include <cstring>
#include <chrono>

std::atomic<uint32_t> SharedState::nextInstanceId { 1 };

SharedState::SharedState() = default;

SharedState::~SharedState()
{
    shutdown();
}

bool SharedState::initialize()
{
    bool createdNew = false;
    if (! shm_.open (kShmName, kShmSize, createdNew))
        return false;

    layout = reinterpret_cast<SharedMemoryLayout*> (shm_.data());

    if (createdNew)
    {
        // We created fresh shared memory - initialize it
        initializeLayout();
    }
    else
    {
        // Existing shared memory - validate it
        if (! validateLayout())
        {
            // Incompatible layout - reinitialize
            // This handles version upgrades and corruption
            initializeLayout();
        }
    }

    // Clean up any stale instances from crashed plugins
    cleanupStaleInstances();

    return true;
}

void SharedState::initializeLayout()
{
    // Zero everything first
    std::memset (layout, 0, kShmSize);

    // Set up header
    layout->header.magic = kMagicCookie;
    layout->header.layoutVersion = kLayoutVersion;
    layout->header.version.store (0);
    layout->header.numInstances.store (0);
    layout->header.referenceSlot.store (-1);
    layout->header.sampleRate = 44100;
}

bool SharedState::validateLayout() const
{
    if (layout == nullptr)
        return false;

    // Check magic cookie
    if (layout->header.magic != kMagicCookie)
        return false;

    // Check layout version
    if (layout->header.layoutVersion != kLayoutVersion)
        return false;

    return true;
}

void SharedState::shutdown()
{
    shm_.close();
    layout = nullptr;
}

uint64_t SharedState::getCurrentTimeMs() const
{
    using namespace std::chrono;
    return static_cast<uint64_t> (
        duration_cast<milliseconds> (
            steady_clock::now().time_since_epoch()
        ).count()
    );
}

int SharedState::registerInstance (const juce::String& trackName)
{
    if (layout == nullptr)
        return -1;

    uint64_t now = getCurrentTimeMs();

    // Find a free slot
    for (int i = 0; i < kMaxInstances; ++i)
    {
        uint32_t expected = 0;
        if (layout->slots[i].active.compare_exchange_strong (expected, 1))
        {
            layout->slots[i].instanceId = nextInstanceId.fetch_add (1);
            layout->slots[i].heartbeatMs.store (now);
            std::strncpy (layout->slots[i].trackName, trackName.toRawUTF8(), 63);
            layout->slots[i].trackName[63] = '\0';

            // Reset analysis data
            layout->slots[i].delaySamples = 0.0f;
            layout->slots[i].delayMs = 0.0f;
            layout->slots[i].correlation = 0.0f;
            layout->slots[i].overallCoherence = 0.0f;
            layout->slots[i].phaseDegrees = 0.0f;
            layout->slots[i].polarityInverted = false;
            layout->slots[i].timeCorrectionOn = false;
            layout->slots[i].phaseCorrectionOn = false;
            std::memset (layout->slots[i].spectralBands, 0, sizeof (layout->slots[i].spectralBands));

            layout->header.numInstances.fetch_add (1);
            layout->header.version.fetch_add (1);
            return i;
        }
    }
    return -1;  // No free slots
}

void SharedState::deregisterInstance (int slot)
{
    if (layout == nullptr || slot < 0 || slot >= kMaxInstances)
        return;

    layout->slots[slot].active.store (0);
    layout->slots[slot].heartbeatMs.store (0);
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

    layout->slots[slot].heartbeatMs.store (getCurrentTimeMs());
}

void SharedState::cleanupStaleInstances()
{
    if (layout == nullptr)
        return;

    uint64_t now = getCurrentTimeMs();
    bool changed = false;

    for (int i = 0; i < kMaxInstances; ++i)
    {
        uint32_t status = layout->slots[i].active.load();
        if (status == 0)
            continue;  // Already free

        uint64_t lastHeartbeat = layout->slots[i].heartbeatMs.load();

        // Check for timeout (with wraparound protection)
        if (lastHeartbeat > 0 && now > lastHeartbeat &&
            (now - lastHeartbeat) > kHeartbeatTimeoutMs)
        {
            // Stale instance - clean it up
            layout->slots[i].active.store (0);
            layout->slots[i].heartbeatMs.store (0);
            std::memset (layout->slots[i].trackName, 0, 64);
            layout->header.numInstances.fetch_sub (1);

            // Clear reference if it was this slot
            int32_t expected = i;
            layout->header.referenceSlot.compare_exchange_strong (expected, -1);

            changed = true;
        }
    }

    if (changed)
        layout->header.version.fetch_add (1);
}

void SharedState::setReferenceSlot (int slot)
{
    if (layout == nullptr)
        return;

    int oldRef = layout->header.referenceSlot.load();
    layout->header.referenceSlot.store (slot);

    // If reference changed, clear the reference buffer to avoid stale data
    if (oldRef != slot)
        layout->refBuffer.writePos.store (0);

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

void SharedState::clearReferenceBuffer()
{
    if (layout == nullptr)
        return;
    layout->refBuffer.writePos.store (0);
}

void SharedState::writeRawSamples (const float* data, int numSamples)
{
    if (layout == nullptr || data == nullptr || numSamples <= 0)
        return;

    uint32_t wp = layout->rawSampleBuffer.writePos.load();

    for (int i = 0; i < numSamples; ++i)
    {
        uint32_t pos = (wp + static_cast<uint32_t> (i)) % kMaxRawSamples;
        layout->rawSampleBuffer.samples[pos] = data[i];
    }

    layout->rawSampleBuffer.writePos.store (wp + static_cast<uint32_t> (numSamples));
}

std::vector<float> SharedState::readRawSamples (int maxSamples) const
{
    if (layout == nullptr)
        return {};

    uint32_t wp = layout->rawSampleBuffer.writePos.load();
    uint32_t available = std::min (wp, static_cast<uint32_t> (kMaxRawSamples));
    uint32_t toRead = std::min (available, static_cast<uint32_t> (maxSamples));

    if (toRead == 0)
        return {};

    // Read from position 0 (start of buffer since last clear), not from the end.
    // The buffer is cleared at sync, so position 0 is the first sample after sync.
    // Reading from the end would shift if the reference keeps writing during analysis.
    std::vector<float> result (toRead);
    for (uint32_t i = 0; i < toRead; ++i)
        result[i] = layout->rawSampleBuffer.samples[i % kMaxRawSamples];

    return result;
}

void SharedState::clearRawSampleBuffer()
{
    if (layout == nullptr)
        return;
    layout->rawSampleBuffer.writePos.store (0);
}

void SharedState::requestSync()
{
    if (layout == nullptr)
        return;
    layout->rawSampleBuffer.writePos.store (0);
    layout->header.syncCounter.fetch_add (1);
}

uint32_t SharedState::getSyncCounter() const
{
    if (layout == nullptr)
        return 0;
    return layout->header.syncCounter.load();
}

void SharedState::acknowledgeSyncRequest (int64_t refStartSample)
{
    if (layout == nullptr)
        return;
    layout->header.refRawStartSample.store (refStartSample);
    layout->header.syncAcknowledged.store (layout->header.syncCounter.load());
}

bool SharedState::isSyncAcknowledged (uint32_t syncValue) const
{
    if (layout == nullptr)
        return false;
    return layout->header.syncAcknowledged.load() >= syncValue;
}

int64_t SharedState::getRefRawStartSample() const
{
    if (layout == nullptr)
        return 0;
    return layout->header.refRawStartSample.load();
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
    layout->slots[slot].active.store (2);  // 2 = aligned
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
