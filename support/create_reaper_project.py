#!/usr/bin/env python3
"""Generate a Reaper project (.rpp) from Magic Phase test results.

Creates a project with aligned output followed by original input on each track,
so you just hit play and hear: aligned → original for easy A/B comparison.

Usage:
    python create_reaper_project.py results/tbm_overheads_kick/
"""

import json
import sys
from pathlib import Path

import soundfile as sf


def get_wav_duration(wav_path: Path) -> float:
    """Get duration of a WAV file in seconds."""
    info = sf.info(str(wav_path))
    return info.duration


def relative_path(from_dir: Path, to_file: Path) -> str:
    """Get a relative path from from_dir to to_file, using forward slashes."""
    try:
        return to_file.resolve().relative_to(from_dir.resolve()).as_posix()
    except ValueError:
        # Files on different drives or can't compute relative - use os.path.relpath
        import os
        return os.path.relpath(to_file.resolve(), from_dir.resolve()).replace("\\", "/")


def make_track(name: str, items: list, color: int = 0) -> str:
    """Generate RPP track block.

    items: list of (position, length, item_name, file_path) tuples
    color: Reaper track color (0 = default)
    """
    lines = []
    lines.append("  <TRACK")
    lines.append(f'    NAME "{name}"')
    if color:
        lines.append(f"    PEAKCOL {color}")
    lines.append("    VOLPAN 1 0 -1 -1 1")
    lines.append("    MUTESOLO 0 0 0")
    lines.append("    IPHASE 0")
    lines.append("    ISBUS 0 0")
    lines.append("    FX 1")
    lines.append("    TRACKID {00000000-0000-0000-0000-000000000000}")

    for pos, length, item_name, file_path in items:
        lines.append("    <ITEM")
        lines.append(f"      POSITION {pos:.6f}")
        lines.append(f"      LENGTH {length:.6f}")
        lines.append(f'      NAME "{item_name}"')
        lines.append("      VOLPAN 1 0 1 -1")
        lines.append("      SOFFS 0")
        lines.append("      PLAYRATE 1 1 0 -1 0 0.0025")
        lines.append("      CHANMODE 0")
        lines.append("      FADEIN 1 0 0 1 0 0 0")
        lines.append("      FADEOUT 1 0 0 1 0 0 0")
        lines.append("      <SOURCE WAVE")
        lines.append(f'        FILE "{file_path}"')
        lines.append("      >")
        lines.append("    >")

    lines.append("  >")
    return "\n".join(lines)


def create_rpp(results_dir: Path) -> Path:
    """Create a Reaper project from test results.

    Args:
        results_dir: Path to a results directory containing result.json

    Returns:
        Path to the created .rpp file
    """
    result_file = results_dir / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"No result.json found in {results_dir}")

    result = json.loads(result_file.read_text())
    sample_rate = int(result["config"]["sample_rate"])
    test_name = result["test"]

    # Create reaper subfolder
    reaper_dir = results_dir / "reaper"
    reaper_dir.mkdir(parents=True, exist_ok=True)
    rpp_path = reaper_dir / f"{test_name}.rpp"

    # Project root for resolving relative input paths
    project_root = Path(__file__).parent.parent

    tracks_rpp = []
    gap = 1.0  # 1 second gap between aligned and original

    # Reference track color: green-ish (Reaper uses native color int)
    # Reaper color format: 0x01BBGGRR (the 0x01 prefix means "custom color")
    ref_color = 0x0100BB00   # green
    target_color = 0          # default

    for track in result["tracks"]:
        role = track["role"]
        input_rel = track["input_file"]   # relative to project root
        output_abs = track["output_file"]  # absolute path

        input_path = project_root / input_rel
        output_path = Path(output_abs)

        # Derive a nice track name from the input file stem
        track_name = Path(input_rel).stem

        # Get durations
        if not output_path.exists():
            print(f"  Warning: output file not found, skipping: {output_path}")
            continue
        out_duration = get_wav_duration(output_path)

        if not input_path.exists():
            print(f"  Warning: input file not found, skipping: {input_path}")
            continue
        in_duration = get_wav_duration(input_path)

        # Compute relative paths from the .rpp file location
        out_rel = relative_path(reaper_dir, output_path)
        in_rel = relative_path(reaper_dir, input_path)

        items = [
            (0.0, out_duration, f"{track_name} (aligned)", out_rel),
            (out_duration + gap, in_duration, f"{track_name} (original)", in_rel),
        ]

        color = ref_color if role == "reference" else target_color
        tracks_rpp.append(make_track(track_name, items, color))

    # Assemble RPP
    rpp = f"""<REAPER_PROJECT 0.1 "7.0" 1728000000
  RIPPLE 0
  GROUPOVERRIDE 0 0 0
  AUTOXFADE 129
  ENVATTACH 3
  MIXERUIFLAGS 11 48
  PEAKGAIN 1
  FEEDBACK 0
  PANLAW 1
  PROJOFFS 0 0 0
  MAXPROJLEN 0 600
  GRID 3199 8 1 8 1 0 0 0
  TIMEMODE 1 5 -1 30 0 0 -1
  VIDEO_CONFIG 0 0 256
  PANMODE 6
  CURSOR 0
  ZOOM 100 0 0
  VZOOMEX 6 0
  USE_REC_CFG 0
  RECMODE 1
  SMPTESYNC 0 30 100 40 1000 300 0 0 1 0 0
  LOOP 0
  LOOPGRAN 0 4
  RECORD_PATH "" ""
  <RECORD_CFG
  >
  <APPLYFX_CFG
  >
  RENDER_FILE ""
  RENDER_PATTERN ""
  RENDER_FMT 0 2 0
  RENDER_1X 0
  RENDER_RANGE 1 0 0 18 1000
  RENDER_RESAMPLE 3 0 1
  RENDER_ADDTOPROJ 0
  RENDER_STEMS 0
  RENDER_DITHER 0
  TIMELOCKMODE 1
  TEMPOENVLOCKMODE 1
  ITEMMIX 1
  DEFPITCHMODE 589824 0
  TAKELANE 1
  SAMPLERATE {sample_rate} 0 0
  <RENDER_CFG
  >
  LOCK 0
  <METRONOME 6 2
    VOL 0.25 0.125
    FREQ 800 1600 1
    BEATLEN 4
    SAMPLES "" ""
    PATTERN 2863311530 2863311529
    MULT 1
  >
  GLOBAL_AUTO -1
  TEMPO 120 4 4
  PLAYRATE 1 0 0.25 4
  SELECTION 0 0
  SELECTION2 0 0
  MASTERAUTOMODE 0
  MASTERTRACKHEIGHT 0 0
  MASTERPEAKCOL 16576
  MASTERMUTESOLO 0
  MASTERTRACKVIEW 0 0.6667 0.5 0.5 -1 -1 -1 0 0 0 -1 -1 0
  MASTERHWOUT 0 0 1 0 0 0 0 -1
  MASTER_NCH 2 2
  MASTER_VOLUME 1 0 -1 -1 1
  MASTER_PANMODE 6
  MASTER_FX 1
  MASTER_SEL 0
  <MASTERPLAYSPEEDENV
    EGUID {{00000000-0000-0000-0000-000000000000}}
    ACT 0 -1
    VIS 0 1 1
    LANEHEIGHT 0 0
    ARM 0
    DEFSHAPE 0 -1 -1
  >
  <TEMPOENVEX
    EGUID {{00000000-0000-0000-0000-000000000000}}
    ACT 0 -1
    VIS 1 0 1
    LANEHEIGHT 0 0
    ARM 0
    DEFSHAPE 1 -1 -1
  >
{chr(10).join(tracks_rpp)}
>
"""

    rpp_path.write_text(rpp)
    return rpp_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_reaper_project.py <results_dir>")
        print("Example: python create_reaper_project.py results/tbm_overheads_kick/")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    rpp_path = create_rpp(results_dir)
    print(f"Created Reaper project: {rpp_path}")


if __name__ == "__main__":
    main()
