/*
    VST3TestHarness - Integration Test Harness for Magic Phase

    Loads the compiled .vst3 binary as a black box (exactly like a DAW),
    creates multiple instances, feeds them audio, triggers alignment,
    and outputs structured results for automated comparison.

    Usage:
        VST3TestHarness --test tests/integration/test.json --output-dir results/test1 --result results/test1/result.json
        VST3TestHarness --test tests/integration/test.json --output-dir results/test1 --buffer-size 256
*/

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "TestRunner.h"
#include "ResultWriter.h"

#include <iostream>

int main (int argc, char* argv[])
{
    // ScopedJuceInitialiser_GUI provides the message pump needed for VST3 plugin loading
    juce::ScopedJuceInitialiser_GUI juceInit;

    // Parse command line arguments
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (argv[i]);

    juce::String testPath;
    juce::String outputDir;
    juce::String resultPath;
    int overrideBufferSize = 0;

    for (int i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--test" && i + 1 < args.size())
            testPath = args[++i];
        else if (args[i] == "--output-dir" && i + 1 < args.size())
            outputDir = args[++i];
        else if (args[i] == "--result" && i + 1 < args.size())
            resultPath = args[++i];
        else if (args[i] == "--buffer-size" && i + 1 < args.size())
            overrideBufferSize = args[++i].getIntValue();
    }

    if (testPath.isEmpty())
    {
        std::cerr << "Usage: VST3TestHarness --test <test.json> [--output-dir <dir>] "
                     "[--result <result.json>] [--buffer-size <size>]\n";
        return 1;
    }

    // Load test definition
    juce::File testFile (testPath);
    if (! testFile.existsAsFile())
    {
        std::cerr << "Test file not found: " << testPath << "\n";
        return 1;
    }

    auto testDef = TestRunner::loadTestDefinition (testFile);
    if (testDef.tracks.empty())
    {
        std::cerr << "Failed to parse test definition or no tracks defined\n";
        return 1;
    }

    // Set output paths
    if (outputDir.isNotEmpty())
        testDef.outputDir = outputDir;
    else
        testDef.outputDir = "results/" + testDef.name;

    if (resultPath.isEmpty())
        resultPath = testDef.outputDir + "/result.json";

    // Handle buffer_sizes sweep
    if (! testDef.bufferSizes.empty() && overrideBufferSize == 0)
    {
        std::cout << "Buffer size sweep: " << testDef.bufferSizes.size() << " sizes\n";

        bool allPassed = true;
        for (int bs : testDef.bufferSizes)
        {
            std::cout << "\n--- Buffer size: " << bs << " ---\n";

            TestDefinition sweepDef = testDef;
            sweepDef.bufferSize = bs;
            sweepDef.outputDir = testDef.outputDir + "/bs" + juce::String (bs);

            TestRunner runner;
            auto results = runner.run (sweepDef);

            juce::File sweepResultFile (testDef.outputDir + "/result_bs" + juce::String (bs) + ".json");
            ResultWriter::writeResultJson (sweepDef, results, sweepResultFile);

            if (! results.success)
                allPassed = false;
        }

        return allPassed ? 0 : 1;
    }

    // Override buffer size if specified
    if (overrideBufferSize > 0)
        testDef.bufferSize = overrideBufferSize;

    // Single run
    TestRunner runner;
    auto results = runner.run (testDef);

    // Write result JSON
    ResultWriter::writeResultJson (testDef, results, juce::File (resultPath));

    return results.success ? 0 : 1;
}
