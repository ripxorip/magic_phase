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
    COPY_PLUGIN_AFTER_BUILD FALSE
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

message(STATUS "Magic Phase Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Plugin Formats: ${MP_FORMATS}")
