cmake_minimum_required(VERSION 3.22)
project(MagicPhase VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(contrib/JUCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Platform detection
if(WIN32)
    set(MP_WINDOWS TRUE)
elseif(APPLE)
    set(MP_APPLE TRUE)
elseif(UNIX)
    set(MP_LINUX TRUE)
endif()

# Build options
option(MP_BUILD_VST3 "Build VST3 plugin" ON)
option(MP_BUILD_STANDALONE "Build Standalone application" ON)
option(MP_BUILD_CLI_TEST "Build CLI test tool" OFF)
option(MP_COPY_AFTER_BUILD "Copy plugin to system folder after build" OFF)

# LTO off on Linux
if(NOT DEFINED CACHE{MP_ENABLE_LTO})
    set(_MP_LTO_DEFAULT ON)
    if(MP_LINUX)
        set(_MP_LTO_DEFAULT OFF)
    endif()
    set(MP_ENABLE_LTO ${_MP_LTO_DEFAULT} CACHE BOOL "Enable Link-Time Optimisation")
endif()

# Plugin formats
set(MP_FORMATS "")
if(MP_BUILD_VST3)
    list(APPEND MP_FORMATS "VST3")
endif()
if(MP_BUILD_STANDALONE)
    list(APPEND MP_FORMATS "Standalone")
endif()

# Source files
set(MP_SOURCES
    src/PluginProcessor.cpp
    src/PluginEditor.cpp
    src/DSP/STFTProcessor.cpp
    src/DSP/PhaseAnalyzer.cpp
    src/DSP/PhaseCorrector.cpp
    src/IPC/SharedState.cpp
    src/GUI/MainComponent.cpp
    src/GUI/TrackRow.cpp
    src/GUI/LookAndFeel.cpp
)

# Platform-specific shared memory implementation
if(WIN32)
    list(APPEND MP_SOURCES src/IPC/PlatformSharedMemory_win32.cpp)
else()
    list(APPEND MP_SOURCES src/IPC/PlatformSharedMemory_posix.cpp)
endif()

# Plugin configuration
juce_add_plugin(MagicPhase
    VERSION ${PROJECT_VERSION}
    COMPANY_NAME "MiGiC Music"
    COMPANY_WEBSITE "www.migic.com"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD ${MP_COPY_AFTER_BUILD}
    PLUGIN_MANUFACTURER_CODE Manu
    PLUGIN_CODE MgPh
    FORMATS ${MP_FORMATS}
    PRODUCT_NAME "Magic Phase"
    DESCRIPTION "Spectral Phase Alignment"
    BUNDLE_ID "com.MiGiCMusic.MagicPhase"
    PLUGIN_NAME "Magic Phase"
)

target_sources(MagicPhase PRIVATE ${MP_SOURCES})

target_compile_definitions(MagicPhase
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_STRICT_REFCOUNTEDPTR=1
        JUCE_DISABLE_JUCE_VERSION_PRINTING=1
)

target_link_libraries(MagicPhase
    PRIVATE
        juce::juce_dsp
    PUBLIC
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_plugin_client
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)

# LTO
if(MP_ENABLE_LTO)
    target_link_libraries(MagicPhase PUBLIC juce::juce_recommended_lto_flags)
else()
    if(MP_LINUX)
        set_property(TARGET MagicPhase PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(MagicPhase PRIVATE -fno-lto)
            target_link_options(MagicPhase PRIVATE -fno-lto)
        endif()
    endif()
endif()

target_include_directories(MagicPhase PRIVATE
    src
    src/DSP
    src/IPC
    src/GUI
)

# Platform-specific
if(MP_LINUX)
    target_compile_definitions(MagicPhase PRIVATE MP_LINUX=1)
    find_package(Threads REQUIRED)
    target_link_libraries(MagicPhase PRIVATE Threads::Threads dl rt)
elseif(MP_APPLE)
    target_compile_definitions(MagicPhase PRIVATE MP_APPLE=1)
elseif(MP_WINDOWS)
    target_compile_definitions(MagicPhase PRIVATE MP_WINDOWS=1)
endif()

set_target_properties(MagicPhase PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# CLI Test Tool
if(MP_BUILD_CLI_TEST)
    juce_add_console_app(MagicPhaseTest
        PRODUCT_NAME "MagicPhaseTest"
        COMPANY_NAME "MiGiC Music"
    )

    target_sources(MagicPhaseTest PRIVATE
        src/Tools/MagicPhaseTest.cpp
        src/Tools/AudioFile.cpp
        src/DSP/STFTProcessor.cpp
        src/DSP/PhaseAnalyzer.cpp
        src/DSP/PhaseCorrector.cpp
    )

    target_compile_definitions(MagicPhaseTest PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_STRICT_REFCOUNTEDPTR=1
        JUCE_DISABLE_JUCE_VERSION_PRINTING=1
    )

    target_include_directories(MagicPhaseTest PRIVATE
        src
        src/Tools
    )

    target_link_libraries(MagicPhaseTest PRIVATE
        juce::juce_core
        juce::juce_dsp
        juce::juce_audio_basics
        juce::juce_data_structures
        juce::juce_events
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
    )

    if(MP_LINUX)
        target_compile_definitions(MagicPhaseTest PRIVATE MP_LINUX=1)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(MagicPhaseTest PRIVATE -fno-lto)
            target_link_options(MagicPhaseTest PRIVATE -fno-lto)
        endif()
        find_package(Threads REQUIRED)
        target_link_libraries(MagicPhaseTest PRIVATE Threads::Threads dl)
    endif()

    set_target_properties(MagicPhaseTest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "  CLI Test Tool: ON")
endif()

# IPC Test Tool
if(MP_BUILD_CLI_TEST)
    juce_add_console_app(IPCTest
        PRODUCT_NAME "IPCTest"
        COMPANY_NAME "MiGiC Music"
    )

    set(IPC_TEST_SOURCES
        src/Tools/IPCTest.cpp
        src/Tools/AudioFile.cpp
        src/DSP/STFTProcessor.cpp
        src/DSP/PhaseAnalyzer.cpp
        src/DSP/PhaseCorrector.cpp
        src/IPC/SharedState.cpp
    )

    # Platform-specific shared memory
    if(WIN32)
        list(APPEND IPC_TEST_SOURCES src/IPC/PlatformSharedMemory_win32.cpp)
    else()
        list(APPEND IPC_TEST_SOURCES src/IPC/PlatformSharedMemory_posix.cpp)
    endif()

    target_sources(IPCTest PRIVATE ${IPC_TEST_SOURCES})

    target_compile_definitions(IPCTest PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_STRICT_REFCOUNTEDPTR=1
        JUCE_DISABLE_JUCE_VERSION_PRINTING=1
    )

    target_include_directories(IPCTest PRIVATE
        src
        src/Tools
    )

    target_link_libraries(IPCTest PRIVATE
        juce::juce_core
        juce::juce_dsp
        juce::juce_audio_basics
        juce::juce_data_structures
        juce::juce_events
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
    )

    if(MP_LINUX)
        target_compile_definitions(IPCTest PRIVATE MP_LINUX=1)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(IPCTest PRIVATE -fno-lto)
            target_link_options(IPCTest PRIVATE -fno-lto)
        endif()
        find_package(Threads REQUIRED)
        target_link_libraries(IPCTest PRIVATE Threads::Threads dl rt)
    endif()

    set_target_properties(IPCTest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "  IPC Test Tool: ON")
endif()

# FakeDAW Test Harness
if(MP_BUILD_CLI_TEST)
    juce_add_console_app(FakeDAW
        PRODUCT_NAME "FakeDAW"
        COMPANY_NAME "MiGiC Music"
    )

    set(FAKEDAW_SOURCES
        src/Tools/FakeDAW/FakeDAW.cpp
        src/Tools/AudioFile.cpp
        src/PluginProcessor.cpp
        src/PluginEditor.cpp
        src/DSP/STFTProcessor.cpp
        src/DSP/PhaseAnalyzer.cpp
        src/DSP/PhaseCorrector.cpp
        src/IPC/SharedState.cpp
        src/GUI/MainComponent.cpp
        src/GUI/TrackRow.cpp
        src/GUI/LookAndFeel.cpp
    )

    # Platform-specific shared memory
    if(WIN32)
        list(APPEND FAKEDAW_SOURCES src/IPC/PlatformSharedMemory_win32.cpp)
    else()
        list(APPEND FAKEDAW_SOURCES src/IPC/PlatformSharedMemory_posix.cpp)
    endif()

    target_sources(FakeDAW PRIVATE ${FAKEDAW_SOURCES})

    target_compile_definitions(FakeDAW PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_STRICT_REFCOUNTEDPTR=1
        JUCE_DISABLE_JUCE_VERSION_PRINTING=1
        # Provide plugin name for headless use
        JucePlugin_Name="Magic Phase"
    )

    target_include_directories(FakeDAW PRIVATE
        src
        src/Tools
        src/DSP
        src/IPC
        src/GUI
    )

    target_link_libraries(FakeDAW PRIVATE
        juce::juce_core
        juce::juce_dsp
        juce::juce_audio_basics
        juce::juce_audio_processors
        juce::juce_data_structures
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
    )

    if(MP_LINUX)
        target_compile_definitions(FakeDAW PRIVATE MP_LINUX=1)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(FakeDAW PRIVATE -fno-lto)
            target_link_options(FakeDAW PRIVATE -fno-lto)
        endif()
        find_package(Threads REQUIRED)
        target_link_libraries(FakeDAW PRIVATE Threads::Threads dl rt)
    endif()

    set_target_properties(FakeDAW PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    message(STATUS "  FakeDAW Test Harness: ON")
endif()

message(STATUS "Magic Phase Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Plugin Formats: ${MP_FORMATS}")
