/*
    IPCTest - IPC/Shared Memory Test Tool for Magic Phase

    Tests the shared memory IPC layer outside of a DAW environment.
    Simulates two VST plugin instances communicating via shared memory.

    Usage:
        ./bin/IPCTest ref.wav target.wav              # Orchestrator mode
        ./bin/IPCTest --ref ref.wav                   # Reference process only
        ./bin/IPCTest --target target.wav             # Target process only

    For debugging:
        Terminal 1: ./bin/IPCTest --ref kick_in.wav
        Terminal 2: ./bin/IPCTest --target kick_out.wav

    The orchestrator mode spawns both processes automatically.
*/

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>

#include "AudioFile.h"
#include "IPC/SharedState.h"
#include "DSP/STFTProcessor.h"
#include "DSP/PhaseAnalyzer.h"
#include "DSP/PhaseCorrector.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <csignal>

//==============================================================================
constexpr int kBlockSize = 128;
constexpr float kAnalyzeWindowSec = 7.5f;
constexpr int kMaxWaitMs = 30000;  // 30 second timeout

static std::atomic<bool> g_shouldExit { false };

void signalHandler (int)
{
    g_shouldExit.store (true);
}

//==============================================================================
static std::vector<float> loadMonoAudio (const std::string& path, double& outSampleRate)
{
    AudioFile<float> af;
    if (! af.load (path))
    {
        std::cerr << "[ERROR] Failed to load: " << path << "\n";
        return {};
    }

    outSampleRate = af.getSampleRate();
    int numSamples = af.getNumSamplesPerChannel();
    int numChannels = af.getNumChannels();

    std::vector<float> mono (static_cast<size_t> (numSamples), 0.0f);
    for (int ch = 0; ch < numChannels; ++ch)
        for (int i = 0; i < numSamples; ++i)
            mono[static_cast<size_t> (i)] += af.samples[static_cast<size_t> (ch)][static_cast<size_t> (i)];

    if (numChannels > 1)
    {
        float scale = 1.0f / static_cast<float> (numChannels);
        for (auto& s : mono)
            s *= scale;
    }

    return mono;
}

//==============================================================================
// REFERENCE PROCESS
//==============================================================================
int runReference (const std::string& audioPath)
{
    std::cout << "[REF] Starting reference process\n";
    std::cout << "[REF] Audio: " << audioPath << "\n";

    // Load audio
    double sampleRate = 0;
    auto audio = loadMonoAudio (audioPath, sampleRate);
    if (audio.empty())
        return 1;

    std::cout << "[REF] Loaded " << audio.size() << " samples @ " << sampleRate << " Hz\n";

    // Initialize IPC
    SharedState sharedState;
    sharedState.initialize();

    int mySlot = sharedState.registerInstance ("Reference");
    if (mySlot < 0)
    {
        std::cerr << "[REF] ERROR: Failed to register instance\n";
        return 1;
    }
    std::cout << "[REF] Registered as slot " << mySlot << "\n";

    sharedState.setSampleRate (static_cast<uint32_t> (sampleRate));
    sharedState.setReferenceSlot (mySlot);
    std::cout << "[REF] Set as reference slot\n";

    // Prepare STFT
    STFTProcessor stft;
    stft.prepare (sampleRate, kBlockSize);

    int analyzeWindowSamples = static_cast<int> (kAnalyzeWindowSec * sampleRate);
    analyzeWindowSamples = std::min (analyzeWindowSamples, static_cast<int> (audio.size()));

    std::cout << "[REF] Processing " << analyzeWindowSamples << " samples in " << kBlockSize << "-sample blocks\n";

    // Process audio and write frames to shared memory
    int framesWritten = 0;
    for (int pos = 0; pos < analyzeWindowSamples && !g_shouldExit.load(); pos += kBlockSize)
    {
        int thisBlock = std::min (kBlockSize, analyzeWindowSamples - pos);

        stft.processBlock (audio.data() + pos, thisBlock, [&sharedState, &framesWritten] (std::complex<float>* frame, int numBins) {
            sharedState.writeReferenceFrame (frame, numBins);
            framesWritten++;
        });

        sharedState.updateHeartbeat (mySlot);

        // Simulate real-time: small delay between blocks
        std::this_thread::sleep_for (std::chrono::microseconds (100));
    }

    std::cout << "[REF] Wrote " << framesWritten << " STFT frames to shared memory\n";
    std::cout << "[REF] Waiting for target to finish (Ctrl+C to exit)...\n";

    // Keep heartbeat alive until target is done or timeout
    auto startWait = std::chrono::steady_clock::now();
    while (!g_shouldExit.load())
    {
        sharedState.updateHeartbeat (mySlot);
        std::this_thread::sleep_for (std::chrono::milliseconds (100));

        auto elapsed = std::chrono::steady_clock::now() - startWait;
        if (std::chrono::duration_cast<std::chrono::milliseconds> (elapsed).count() > kMaxWaitMs)
        {
            std::cout << "[REF] Timeout waiting for target\n";
            break;
        }
    }

    sharedState.deregisterInstance (mySlot);
    std::cout << "[REF] Done.\n";
    return 0;
}

//==============================================================================
// TARGET PROCESS
//==============================================================================
int runTarget (const std::string& audioPath, const std::string& outputDir)
{
    std::cout << "[TAR] Starting target process\n";
    std::cout << "[TAR] Audio: " << audioPath << "\n";

    // Load audio
    double sampleRate = 0;
    auto audio = loadMonoAudio (audioPath, sampleRate);
    if (audio.empty())
        return 1;

    std::cout << "[TAR] Loaded " << audio.size() << " samples @ " << sampleRate << " Hz\n";

    // Initialize IPC
    SharedState sharedState;
    sharedState.initialize();

    int mySlot = sharedState.registerInstance ("Target");
    if (mySlot < 0)
    {
        std::cerr << "[TAR] ERROR: Failed to register instance\n";
        return 1;
    }
    std::cout << "[TAR] Registered as slot " << mySlot << "\n";

    // Wait for reference to be set
    std::cout << "[TAR] Waiting for reference...\n";
    auto startWait = std::chrono::steady_clock::now();
    int refSlot = -1;

    while (!g_shouldExit.load())
    {
        refSlot = sharedState.getReferenceSlot();
        if (refSlot >= 0 && refSlot != mySlot)
        {
            std::cout << "[TAR] Found reference at slot " << refSlot << "\n";
            break;
        }

        std::this_thread::sleep_for (std::chrono::milliseconds (50));

        auto elapsed = std::chrono::steady_clock::now() - startWait;
        if (std::chrono::duration_cast<std::chrono::milliseconds> (elapsed).count() > kMaxWaitMs)
        {
            std::cerr << "[TAR] ERROR: Timeout waiting for reference\n";
            sharedState.deregisterInstance (mySlot);
            return 1;
        }
    }

    if (g_shouldExit.load())
    {
        sharedState.deregisterInstance (mySlot);
        return 1;
    }

    // Prepare STFT
    STFTProcessor stft;
    stft.prepare (sampleRate, kBlockSize);

    PhaseAnalyzer analyzer;
    analyzer.prepare (sampleRate);
    analyzer.setCoherenceThreshold (0.4f);
    analyzer.setMaxCorrectionDeg (120.0f);

    int analyzeWindowSamples = static_cast<int> (kAnalyzeWindowSec * sampleRate);
    analyzeWindowSamples = std::min (analyzeWindowSamples, static_cast<int> (audio.size()));

    std::cout << "[TAR] Processing " << analyzeWindowSamples << " samples...\n";

    // Process audio and accumulate frames locally
    std::vector<std::vector<std::complex<float>>> myFrames;

    for (int pos = 0; pos < analyzeWindowSamples && !g_shouldExit.load(); pos += kBlockSize)
    {
        int thisBlock = std::min (kBlockSize, analyzeWindowSamples - pos);

        stft.processBlock (audio.data() + pos, thisBlock, [&myFrames] (std::complex<float>* frame, int numBins) {
            myFrames.emplace_back (frame, frame + numBins);
        });

        sharedState.updateHeartbeat (mySlot);
        std::this_thread::sleep_for (std::chrono::microseconds (100));
    }

    std::cout << "[TAR] Accumulated " << myFrames.size() << " local frames\n";

    // Wait for reference to have enough frames
    std::cout << "[TAR] Waiting for reference frames...\n";
    startWait = std::chrono::steady_clock::now();

    while (!g_shouldExit.load())
    {
        auto refFrames = sharedState.readReferenceFrames();
        if (refFrames.size() >= myFrames.size() / 2)  // At least half as many frames
        {
            std::cout << "[TAR] Got " << refFrames.size() << " reference frames\n";
            break;
        }

        std::this_thread::sleep_for (std::chrono::milliseconds (50));
        sharedState.updateHeartbeat (mySlot);

        auto elapsed = std::chrono::steady_clock::now() - startWait;
        if (std::chrono::duration_cast<std::chrono::milliseconds> (elapsed).count() > kMaxWaitMs)
        {
            std::cerr << "[TAR] ERROR: Timeout waiting for reference frames (got "
                      << sharedState.readReferenceFrames().size() << ")\n";
            sharedState.deregisterInstance (mySlot);
            return 1;
        }
    }

    // =========================================================================
    // TRIGGER ALIGN (emulates user clicking "MAGIC ALIGN")
    // =========================================================================
    std::cout << "\n[TAR] === TRIGGERING ALIGNMENT ===\n";

    auto refFrames = sharedState.readReferenceFrames();
    std::cout << "[TAR] Read " << refFrames.size() << " reference frames from shared memory\n";

    // Run analysis
    analyzer.analyze (refFrames, myFrames);

    float delaySamples = analyzer.getDelaySamples();
    float delayMs = analyzer.getDelayMs();
    float correlation = analyzer.getCorrelation();
    bool polarityInverted = analyzer.getPolarityInverted();
    float coherence = analyzer.getOverallCoherence();
    float phaseDeg = analyzer.getPhaseDegrees();

    std::cout << "\n[TAR] === ANALYSIS RESULTS ===\n";
    std::cout << "[TAR]   Delay:       " << delaySamples << " samples (" << delayMs << " ms)\n";
    std::cout << "[TAR]   Correlation: " << correlation << "\n";
    std::cout << "[TAR]   Polarity:    " << (polarityInverted ? "INVERTED" : "Normal") << "\n";
    std::cout << "[TAR]   Coherence:   " << coherence << "\n";
    std::cout << "[TAR]   Phase avg:   " << phaseDeg << " deg\n";

    // Update shared state
    sharedState.updateInstanceData (mySlot,
        delaySamples, delayMs, correlation, coherence, phaseDeg,
        polarityInverted, true, true, analyzer.getSpectralBands());

    sharedState.setInstanceAligned (mySlot);

    // =========================================================================
    // Apply correction and save output (optional)
    // =========================================================================
    if (!outputDir.empty())
    {
        juce::File (outputDir).createDirectory();

        PhaseCorrector corrector;
        corrector.prepare (sampleRate);
        corrector.setDelaySamples (delaySamples);
        corrector.setPolarityInvert (polarityInverted);
        corrector.setPhaseCorrection (analyzer.getPhaseCorrection());
        corrector.setTimeCorrectionOn (true);
        corrector.setPhaseCorrectionOn (true);

        // Apply corrections to full file
        STFTProcessor outputSTFT;
        outputSTFT.prepare (sampleRate, kBlockSize);

        int numSamples = static_cast<int> (audio.size());
        int latency = outputSTFT.getLatencySamples();
        int paddedLen = numSamples + latency;

        std::vector<float> padded (static_cast<size_t> (paddedLen), 0.0f);

        // First apply time correction
        int delayInt = static_cast<int> (std::round (delaySamples));
        float polaritySign = polarityInverted ? -1.0f : 1.0f;
        for (int i = 0; i < numSamples; ++i)
        {
            int srcIdx = i + delayInt;
            if (srcIdx >= 0 && srcIdx < numSamples)
                padded[static_cast<size_t> (i)] = audio[static_cast<size_t> (srcIdx)] * polaritySign;
        }

        // Then apply spectral phase correction
        for (int pos = 0; pos < paddedLen; pos += kBlockSize)
        {
            int thisBlock = std::min (kBlockSize, paddedLen - pos);
            outputSTFT.processBlock (padded.data() + pos, thisBlock, [&corrector] (std::complex<float>* frame, int numBins) {
                corrector.applyPhaseCorrection (frame, numBins);
            });
        }

        // Save aligned output
        std::vector<float> aligned (padded.begin() + latency, padded.begin() + latency + numSamples);

        AudioFile<float> af;
        af.setSampleRate (static_cast<uint32_t> (sampleRate));
        af.setBitDepth (24);
        af.setNumChannels (1);
        af.setNumSamplesPerChannel (numSamples);
        for (int i = 0; i < numSamples; ++i)
            af.samples[0][static_cast<size_t> (i)] = aligned[static_cast<size_t> (i)];

        std::string outPath = outputDir + "/aligned_ipc.wav";
        af.save (outPath);
        std::cout << "[TAR] Saved: " << outPath << "\n";
    }

    sharedState.deregisterInstance (mySlot);
    std::cout << "[TAR] Done.\n";

    // Signal to reference process that we're done
    g_shouldExit.store (true);

    return 0;
}

//==============================================================================
// ORCHESTRATOR
//==============================================================================
int runOrchestrator (const std::string& refPath, const std::string& targetPath, const std::string& outputDir)
{
    std::cout << "=== Magic Phase IPC Test (Orchestrator Mode) ===\n\n";
    std::cout << "Reference: " << refPath << "\n";
    std::cout << "Target:    " << targetPath << "\n";
    std::cout << "Output:    " << outputDir << "\n\n";

    // Get our own executable path
    juce::File exe = juce::File::getSpecialLocation (juce::File::currentExecutableFile);
    std::string exePath = exe.getFullPathName().toStdString();

    std::cout << "Spawning child processes...\n\n";

    // Spawn reference process
    juce::ChildProcess refProcess;
    juce::StringArray refArgs;
    refArgs.add (exePath);
    refArgs.add ("--ref");
    refArgs.add (refPath);

    if (! refProcess.start (refArgs))
    {
        std::cerr << "ERROR: Failed to start reference process\n";
        return 1;
    }
    std::cout << "[ORCH] Started reference process\n";

    // Small delay to let reference initialize
    std::this_thread::sleep_for (std::chrono::milliseconds (500));

    // Spawn target process
    juce::ChildProcess targetProcess;
    juce::StringArray targetArgs;
    targetArgs.add (exePath);
    targetArgs.add ("--target");
    targetArgs.add (targetPath);
    if (! outputDir.empty())
    {
        targetArgs.add ("-o");
        targetArgs.add (outputDir);
    }

    if (! targetProcess.start (targetArgs))
    {
        std::cerr << "ERROR: Failed to start target process\n";
        refProcess.kill();
        return 1;
    }
    std::cout << "[ORCH] Started target process\n";

    std::cout << "\n[ORCH] Waiting for processes to complete...\n";
    std::cout << "[ORCH] (You can attach debuggers to PIDs above)\n\n";

    // Read and print output from both processes
    while (refProcess.isRunning() || targetProcess.isRunning())
    {
        // Read from reference
        char buffer[1024];
        int bytesRead = refProcess.readProcessOutput (buffer, sizeof (buffer) - 1);
        if (bytesRead > 0)
        {
            buffer[bytesRead] = '\0';
            std::cout << buffer;
        }

        // Read from target
        bytesRead = targetProcess.readProcessOutput (buffer, sizeof (buffer) - 1);
        if (bytesRead > 0)
        {
            buffer[bytesRead] = '\0';
            std::cout << buffer;
        }

        std::this_thread::sleep_for (std::chrono::milliseconds (10));
    }

    // Final output
    char buffer[4096];
    int bytesRead = refProcess.readProcessOutput (buffer, sizeof (buffer) - 1);
    if (bytesRead > 0) { buffer[bytesRead] = '\0'; std::cout << buffer; }

    bytesRead = targetProcess.readProcessOutput (buffer, sizeof (buffer) - 1);
    if (bytesRead > 0) { buffer[bytesRead] = '\0'; std::cout << buffer; }

    uint32_t refExit = refProcess.getExitCode();
    uint32_t targetExit = targetProcess.getExitCode();

    std::cout << "\n[ORCH] Reference exit code: " << refExit << "\n";
    std::cout << "[ORCH] Target exit code:    " << targetExit << "\n";

    return (refExit == 0 && targetExit == 0) ? 0 : 1;
}

//==============================================================================
// MAIN
//==============================================================================
int main (int argc, char* argv[])
{
    std::signal (SIGINT, signalHandler);
    std::signal (SIGTERM, signalHandler);

    if (argc < 2)
    {
        std::cout << "Magic Phase IPC Test Tool\n\n"
                  << "Usage:\n"
                  << "  IPCTest ref.wav target.wav [-o output_dir]    Orchestrator mode\n"
                  << "  IPCTest --ref ref.wav                         Reference process\n"
                  << "  IPCTest --target target.wav [-o output_dir]   Target process\n"
                  << "\n"
                  << "For debugging, run --ref and --target in separate terminals.\n"
                  << "This lets you attach debuggers to each process independently.\n";
        return 1;
    }

    std::string mode = argv[1];

    // Reference mode
    if (mode == "--ref")
    {
        if (argc < 3)
        {
            std::cerr << "Usage: IPCTest --ref <audio.wav>\n";
            return 1;
        }
        return runReference (argv[2]);
    }

    // Target mode
    if (mode == "--target")
    {
        if (argc < 3)
        {
            std::cerr << "Usage: IPCTest --target <audio.wav> [-o output_dir]\n";
            return 1;
        }
        std::string outputDir = "./output_ipc";
        for (int i = 3; i < argc; ++i)
        {
            if (std::string (argv[i]) == "-o" && i + 1 < argc)
                outputDir = argv[++i];
        }
        return runTarget (argv[2], outputDir);
    }

    // Orchestrator mode (default)
    if (argc < 3)
    {
        std::cerr << "Usage: IPCTest ref.wav target.wav [-o output_dir]\n";
        return 1;
    }

    std::string refPath = argv[1];
    std::string targetPath = argv[2];
    std::string outputDir = "./output_ipc";

    for (int i = 3; i < argc; ++i)
    {
        if (std::string (argv[i]) == "-o" && i + 1 < argc)
            outputDir = argv[++i];
    }

    return runOrchestrator (refPath, targetPath, outputDir);
}
