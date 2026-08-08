#!/usr/bin/env python3
"""Does the training excitation actually reach the device's own saturation ceiling?

THE QUESTION. A knob grid can be perfectly dense (grid_adequacy.py says so) and the dataset can
still be missing the device's real nonlinear/breakup character, for a completely different reason:
the excitation itself may never get loud enough to drive the circuit into it. A model can only
learn what's in the data -- if every rendered permutation stays in the device's linear region, the
trained model has no idea what happens past it, no matter how dense the knob grid is.

This is not hypothetical. On TAD Blackface 85 Reverb (2026-07-31), the default preflight probe
level came back well under this device's own measured saturation onset -- caught only because
someone asked "was this checked?" after the config was otherwise fully measured (oversample, grid).
Nothing else in the pipeline would have caught it: grid_adequacy.py measures interpolation density,
not whether the sampled region includes saturation at all.

WHAT THIS CHECKS. Runs tools/preflight.py --find-peak (a clean-tone amplitude sweep at the config's
knobs) to find the device's saturation onset -- the input voltage where output RMS stops rising --
then compares it against the actual peak of the config's own training excitation file. If the
excitation's peak sits well under the onset, the trained dataset likely never explores that region.

WHAT THIS DOES NOT DO. --find-peak sweeps at DEFAULT (0.5) knob settings, not the grid's own hottest
corner. A device whose Volume/Gain knob changes how hard the excitation drives later stages could
have a MUCH lower onset at the grid's own maximum setting than at the default -- this check can
false-positive (warn on a device that's actually fine at its loud knob settings) exactly the way it
did on TAD, where the amp turned out to have real, dense nonlinear behavior concentrated at LOW
Volume rather than at high input level. Treat a WARN here as "go investigate at the grid's hot
corner directly" (see docs/adding-a-device.md), not as an automatic verdict either way.

Usage:
    ./tools/check_input_headroom.py --config ~/work/parametric-nam-models/pedals/my-device/config.toml
    ./tools/check_input_headroom.py --config ~/work/parametric-nam-models/pedals/my-device/config.toml --margin 0.9
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_config(p: Path) -> dict:
    with open(p, "rb") as f:
        return tomllib.load(f)


def read_peak_v(wav_path: str) -> float:
    """Peak sample value of a WAV file. Under this fleet's V0dBFS=1V convention (see
    internal engineering notes), a full-scale sample of 1.0 IS 1 volt at the schx's Input --
    so the raw peak (no separate conversion) is directly comparable to preflight's onset_99pct_input_v.
    Handles 16-bit int, 24-bit int, and 32-bit float PCM (this fleet's three actual formats)."""
    with open(wav_path, "rb") as f:
        raw = f.read()
    fmt_idx = raw.find(b"fmt ")
    fmt_size = struct.unpack("<I", raw[fmt_idx + 4:fmt_idx + 8])[0]
    fmt = raw[fmt_idx + 8:fmt_idx + 8 + fmt_size]
    audio_format, channels, sr, byte_rate, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    data_idx = raw.find(b"data")
    data_size = struct.unpack("<I", raw[data_idx + 4:data_idx + 8])[0]
    data = raw[data_idx + 8:data_idx + 8 + data_size]

    if audio_format == 3 and bits == 32:
        import array
        a = array.array("f")
        a.frombytes(data)
        return max(abs(min(a)), abs(max(a)))
    if audio_format == 1 and bits == 16:
        import array
        a = array.array("h")
        a.frombytes(data)
        return max(abs(min(a)), abs(max(a))) / 32768.0
    if audio_format == 1 and bits == 24:
        vals = []
        for i in range(0, len(data), 3):
            b = data[i:i + 3]
            v = b[0] | (b[1] << 8) | (b[2] << 16)
            if v & 0x800000:
                v -= 0x1000000
            vals.append(v)
        return max(abs(v) for v in vals) / (2 ** 23)
    raise ValueError(f"unhandled WAV format: audio_format={audio_format} bits={bits}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the same TOML the pipeline uses")
    ap.add_argument("--oversample", type=int, default=None,
                    help="override the config's oversample for the --find-peak probe (default: "
                         "the config's own value, or 8 if it's 'auto')")
    ap.add_argument("--margin", type=float, default=0.8,
                    help="WARN if excitation peak < margin * onset (default 0.8 -- the excitation "
                         "should reach at least 80%% of the way to the saturation onset)")
    ap.add_argument("--peak-max-v", type=float, default=40.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    schx = os.path.expanduser(cfg["schx"])
    inp = os.path.expanduser(cfg["input"])
    knobs = ",".join(cfg["knobs"].keys())
    fixed = ",".join(f"{k}={v}" for k, v in (cfg.get("fixed") or {}).items())
    oversample = args.oversample
    if oversample is None:
        ov = cfg.get("oversample", 8)
        oversample = 8 if str(ov).strip().lower() == "auto" else int(ov)

    print(f"Checking whether {Path(inp).name} reaches {Path(schx).name}'s own saturation ceiling "
          f"(knobs at default 0.5 -- see this tool's docstring for what that does and doesn't tell you)\n")

    with tempfile.TemporaryDirectory() as scratch:
        json_path = Path(scratch) / "preflight.json"
        cmd = [sys.executable, str(HERE / "preflight.py"),
               "--schx", schx, "--knobs", knobs, "--input", inp,
               "--oversample", str(oversample), "--find-peak",
               "--peak-max-v", str(args.peak_max_v), "--json", str(json_path)]
        if fixed:
            cmd += ["--fixed-params", fixed]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout)
        if r.returncode not in (0, 1):  # preflight exits 1 on hard-fail knobs, which is fine here
            print(r.stderr, file=sys.stderr)
            print("ERROR: preflight --find-peak itself failed to run.", file=sys.stderr)
            return 2
        if not json_path.exists():
            print("ERROR: preflight did not write a JSON report.", file=sys.stderr)
            return 2
        report = json.loads(json_path.read_text())

    sat = report.get("saturation")
    onset = sat.get("onset_99pct_input_v") if sat else None
    if onset is None:
        print("WARN: no saturation onset found within the sweep range "
              f"(0-{args.peak_max_v}V) -- either this device never saturates (plausible for a "
              "very clean/high-headroom design) or --peak-max-v needs to go higher. Not scored "
              "against the excitation peak; investigate directly.")
        return 1

    excitation_peak = read_peak_v(inp)
    ratio = excitation_peak / onset if onset > 0 else float("inf")
    print(f"Excitation ({Path(inp).name}) peak:      {excitation_peak:.4f} V")
    print(f"Device saturation onset (99% of ceiling): {onset:.4f} V")
    print(f"Ratio: {ratio * 100:.1f}%")

    if excitation_peak < args.margin * onset:
        print(f"\nWARN: the excitation peak is only {ratio * 100:.0f}% of the way to this device's "
              f"saturation onset at DEFAULT knob settings. The trained dataset may never explore "
              f"this device's nonlinear/breakup region. This is NOT an automatic verdict -- see "
              f"this tool's docstring: a Volume/Gain knob at its grid's own hottest setting can "
              f"reach a much lower onset than the default-knobs probe above. Investigate at the "
              f"grid's own extreme before concluding the excitation needs to run hotter, or that "
              f"this is simply a genuine high-headroom device.")
        return 1

    print(f"\nOK: excitation reaches {ratio * 100:.0f}% of the device's saturation onset at default "
          f"knob settings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
