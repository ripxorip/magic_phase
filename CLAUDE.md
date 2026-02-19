# CLAUDE.md

## Project: Magic Phase
Spectral phase alignment tool by MiGiC.

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
- Python prototype (numpy, scipy, soundfile)
- Nix flake for dev environment
- Target: Eventually VST via Rust/JUCE

## Competitors we're beating
- Waves InPhase (time only)
- MAutoAlign (time only)
- Sound Radix Auto-Align (unclear if truly spectral)

