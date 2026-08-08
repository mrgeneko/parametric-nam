#!/usr/bin/env python3
"""Build a device training excitation that actually covers the full input range.

A high-crest real-playing clip (e.g. sweep-v3.wav, ~22 dB crest) samples its own loud region
essentially never (<0.1% of time within 6 dB of peak), so a model trained on it alone
never learns the device's saturation/blocking behavior and goes out-of-distribution when a
hot input (upstream boost) arrives. This concatenates:

  [ real-playing clip @ --realistic-peak ]  (dynamics/perceptual realism, mostly low level)
  [ amplitude-stepped log sine sweeps ]      (dense level x frequency coverage of the loud region)
  [ short fade-out to zero ]

so the whole 0 -> --sweep-peaks[-1] transfer, including the memory-dependent blocking region,
is genuinely learned. Under the V0dBFS=1V convention a sample value == drive volts, so pass
--sweep-peaks in volts and set the max to the device's saturation/max-output point + headroom.
Written float32 so values >1.0 survive (they represent >1 V drive, which is legitimate).

See docs/input_calibration.md.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000


def _audio_provenance(path, x=None):
    """Same identity fields as batch_harness.py's input_provenance() (name/path/sha1 of the
    MONO SAMPLE BYTES/samplerate/frames/duration) -- deliberately duplicated, not imported,
    so this tool has no dependency on the harness and the two hashes stay independently
    verifiable against each other (same audio -> same hash, computed two different ways)."""
    if x is None:
        x, _ = sf.read(str(path), dtype="float32")
    mono = x if x.ndim == 1 else x.mean(axis=1)
    mono = np.asarray(mono, dtype=np.float32)
    return {
        "name": Path(path).name,
        "path": str(path),
        "audio_sha1": hashlib.sha1(mono.tobytes()).hexdigest(),
        "samplerate": SR,
        "frames": int(len(mono)),
        "duration_s": round(len(mono) / SR, 3),
    }


def _tool_git_rev():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _fade(y, ms_in, ms_out):
    y = y.copy()
    ni, no = int(SR * ms_in / 1000), int(SR * ms_out / 1000)
    if ni > 0: y[:ni] *= np.sin(np.linspace(0, np.pi / 2, ni)) ** 2
    if no > 0: y[-no:] *= np.sin(np.linspace(np.pi / 2, 0, no)) ** 2
    return y


def _log_sweep(f0, f1, dur, amp):
    t = np.arange(int(SR * dur)) / SR; T = t[-1]
    K = T * 2 * np.pi * f0 / np.log(f1 / f0)
    L = np.log(f1 / f0) / T
    return (amp * np.sin(K * (np.exp(t * L) - 1))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="real-playing clip (e.g. sweep-v3.wav)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--realistic-peak", type=float, default=1.0,
                    help="peak (V, at V0dBFS=1) to scale the real-playing clip to")
    ap.add_argument("--realistic-dur", type=float, default=None,
                    help="seconds of --input to keep (prefix), default: the whole file. The "
                         "realistic clip only needs to sample varied dynamics/perceptual content "
                         "-- it is not what provides saturation coverage (the sweeps do) -- so a "
                         "short prefix is normally enough, and every second of it costs a "
                         "render-time multiplier across the whole permutation grid.")
    ap.add_argument("--sweep-peaks", default="0.5,1.0,1.5,2.0",
                    help="comma list of sine-sweep peak amplitudes (V); last = training max drive")
    ap.add_argument("--sweep-f0", type=float, default=40.0)
    ap.add_argument("--sweep-f1", type=float, default=12000.0)
    ap.add_argument("--sweep-dur", type=float, default=3.0, help="seconds per amplitude step")
    ap.add_argument("--fade-out-ms", type=float, default=150.0)
    args = ap.parse_args()

    x, sr = sf.read(args.input, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    if sr != SR: raise SystemExit(f"input sr {sr} != {SR}")
    if args.realistic_dur is not None:
        x = x[:int(SR * args.realistic_dur)]

    real = _fade((x / max(np.abs(x).max(), 1e-9) * args.realistic_peak).astype(np.float32), 10, 10)
    peaks = [float(p) for p in args.sweep_peaks.split(",") if p.strip()]
    pad = np.zeros(int(SR * 0.1), dtype=np.float32)
    parts = [real, pad] + [_fade(_log_sweep(args.sweep_f0, args.sweep_f1, args.sweep_dur, a), 8, 8)
                           for a in peaks]
    comp = np.concatenate(parts)
    nf = int(SR * args.fade_out_ms / 1000)
    if nf > 0: comp[-nf:] *= np.sin(np.linspace(np.pi / 2, 0, nf)) ** 2

    sf.write(args.output, comp, SR, subtype="FLOAT")
    a = np.abs(comp)
    print(f"wrote {args.output}  dur {len(comp)/SR:.1f}s  peak {a.max():.3f}  rms {np.sqrt((comp**2).mean()):.4f}")
    for thr in sorted(set([0.5] + peaks)):
        print(f"  time >= {thr:.2f} V : {100*np.mean(a >= thr):6.3f}%")

    # Recipe sidecar: HOW this excitation was built, not just which bytes it is. Without this,
    # a derived excitation's provenance chain stops at "some file named *_src95.wav" -- the exact
    # window/args used to cut it from the source are lost the moment the config-file comment that
    # (informally, inconsistently) recorded them is out of date or missing. batch_harness.py's
    # input_provenance() picks this up automatically (same directory, <stem>.recipe.json) and
    # embeds it in every dataset's config.json -> parametric-nam-models' dataset_config.json.
    recipe = {
        "tool": "build_excitation.py",
        "tool_git_rev": _tool_git_rev(),
        "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "args": {
            "realistic_peak": args.realistic_peak,
            "realistic_dur": args.realistic_dur,
            "sweep_peaks": peaks,
            "sweep_f0": args.sweep_f0,
            "sweep_f1": args.sweep_f1,
            "sweep_dur": args.sweep_dur,
            "fade_out_ms": args.fade_out_ms,
        },
        "source": _audio_provenance(args.input),
        "output": {**_audio_provenance(args.output, x=comp),
                   "peak": round(float(a.max()), 6),
                   "rms": round(float(np.sqrt((comp ** 2).mean())), 6)},
    }
    recipe_path = Path(args.output).with_suffix(".recipe.json")
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"wrote {recipe_path}  (build recipe -- picked up automatically by batch_harness.py)")


if __name__ == "__main__":
    main()
