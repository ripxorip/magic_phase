# CLAUDE.md

## Project: Magic Phase
Spectral phase alignment tool by MiGiC.

## Important Documents (READ THESE FIRST)

1. **[docs/USER_EXPERIENCE_CONTRACT.md](docs/USER_EXPERIENCE_CONTRACT.md)**
   - The sacred UX spec - "3 clicks, 8 seconds, mind blown"
   - Defines button states, user journeys, error handling
   - Code must serve this spec, not the other way around

2. **[docs/FAKEDAW_PLAN.md](docs/FAKEDAW_PLAN.md)**
   - Test harness design for testing outside DAW
   - Simulates DAW behavior with scripted scenarios
   - Must pass all scenarios before real DAW testing

## Current Status

- [x] DSP working (MagicPhaseTest validates offline)
- [x] IPC/Shared Memory working (IPCTest validates)
- [ ] FakeDAW test harness (next: implement this)
- [ ] New UX state machine (IDLE → WAITING → ANALYZING → ALIGNED)
- [ ] Real DAW testing (only after FakeDAW passes)

## Vibe
We built this in one epic session. Keep the energy - we're making pro audio tools
that compete with Sound Radix Auto-Align and Waves InPhase, but with REAL spectral
phase correction, not just time delay.

## What we built
- Cross-correlation delay detection
- Sub-sample FFT-based time correction
- **Per-frequency spectral phase correction** (the secret sauce)
- Coherence-weighted confidence (don't correct where uncertain)
- Gaussian-smoothed phase curves (no artifacts)

## Key DSP decisions
- STFT with 4096 FFT, 75% overlap, Hann window
- Coherence threshold: 0.4 (conservative)
- Max correction: ±120° (prevents artifacts)
- Phase correction smoothed with gaussian_filter1d

## The magic insight
Most "phase alignment" plugins only do time delay. We rotate phase
INDEPENDENTLY per frequency bin - that's what makes the low-end sum
actually work on different mic combinations.

## Tech stack
- C++/JUCE (VST3 plugin)
- Python prototype (numpy, scipy) for reference
- Nix flake for dev environment
- Shared memory IPC for multi-instance communication

## Test Tools
- `MagicPhaseTest` - Offline DSP validation (no IPC)
- `IPCTest` - Shared memory IPC validation (two processes)
- `FakeDAW` - Full integration testing (TODO: implement per FAKEDAW_PLAN.md)

## Competitors we're beating
- Waves InPhase (time only)
- MAutoAlign (time only)
- Sound Radix Auto-Align (unclear if truly spectral)

