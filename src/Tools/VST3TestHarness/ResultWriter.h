#pragma once
#include "TestRunner.h"
#include <juce_core/juce_core.h>

namespace ResultWriter
{
    void writeResultJson (const TestDefinition& test,
                          const TestResults& results,
                          const juce::File& outputFile);
}
