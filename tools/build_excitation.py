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

See internal engineering notes.
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
    """Same identity fields as gen_dataset_from_schx.py's input_provenance() (name/path/sha1 of the
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


def _transient_burst(sec, amp, decay_tau):
    """Broadband (20 Hz - 16 kHz, Schroeder-phase -- same deterministic construction
    make_sweep.py's flat_burst() uses) burst with an INSTANT attack (no fade-in) and an
    exponential decay, mimicking a real hard pick-attack's amplitude envelope (high crest
    factor: sharp onset, fast decay to near-silence) rather than a sustained tone. No RNG --
    same input always writes the same bytes.

    Exists as a deterministic, license-free alternative to extracting a real-playing noise
    burst (--noise-burst-*): a real hard attack's rise time is close to the 1-sample physical
    limit already, so a synthesized instant attack loses nothing there, while avoiding any
    dependency on restricted third-party source audio. decay_tau=0.03s (the default) gives
    crest factor ~8.5, matching a real hard-attack transient -- see internal engineering
    notes on the corner-instability investigation this was built to fix.
    """
    n = int(SR * sec)
    f = np.fft.rfftfreq(n, 1 / SR)
    mag = ((f >= 20) & (f <= 16000)).astype(float)
    k = np.arange(len(f))
    spec = mag * np.exp(1j * (np.pi * k ** 2 / max(mag.sum(), 1)))
    x = np.fft.irfft(spec, n)
    t = np.arange(n) / SR
    env = np.exp(-t / decay_tau)
    y = x * env
    # Two-stage normalize: x's own peak (from the Schroeder-phase construction) doesn't
    # necessarily land at t=0, where the decay envelope is strongest -- normalizing x alone
    # (as flat_burst does) then multiplying by env systematically undershoots `amp` by
    # however much the envelope has already decayed by the time x reaches ITS peak.
    # Renormalizing the final enveloped signal instead makes "peak level == amp" exact,
    # matching --sweep-peaks/--realistic-peak's own contract elsewhere in this file.
    y /= (np.abs(y).max() + 1e-12)
    return (amp * y).astype(np.float32)


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
    ap.add_argument("--noise-burst-src", default=None,
                    help="source WAV to pull a broadband white-noise burst staircase from -- "
                         "default: same file as --input. A sine sweep only tests ONE frequency "
                         "at a time at each level; a broadband noise ATTACK (0 -> level, sharp "
                         "rise) stress-tests saturation onset the way a real pick attack does, "
                         "across the whole spectrum at once, at a level the sweep-tail alone "
                         "doesn't guarantee (see check_transient_coverage.py's docstring -- this "
                         "is the same gap that under-covered Tweed 5F6-A originally). T3K-sweep-v3 "
                         "has exactly this built in already (a 3-step discrete noise staircase "
                         "plus a continuous swell, found by scanning for high spectral-flatness "
                         "windows) -- reusing it beats synthesizing a new one from scratch.")
    ap.add_argument("--noise-burst-window", default=None, metavar="START,END",
                    help="seconds into --noise-burst-src to extract (e.g. '12.0,17.0' for "
                         "T3K-sweep-v3's own noise staircase). Required if --noise-burst-src "
                         "or a --noise-burst-peak is given.")
    ap.add_argument("--noise-burst-peak", type=float, default=None,
                    help="peak (V) to rescale the extracted noise-burst window to -- scales the "
                         "WHOLE window by one factor, preserving its internal staircase shape "
                         "(so the low steps land at the same fraction of this peak the source "
                         "window has). Typically the same as --sweep-peaks' last value. Omit to "
                         "skip the noise-burst segment entirely (default: off, matches prior "
                         "behavior).")
    ap.add_argument("--synth-burst-peaks", default=None,
                    help="comma list of synthesized transient-burst peak amplitudes (V) -- a "
                         "deterministic, license-free alternative to --noise-burst-* (which "
                         "depends on a restricted real-playing source and only ever tests ONE "
                         "level). Inserts one burst per level: broadband (20Hz-16kHz), instant "
                         "attack, exponential decay -- crest factor ~8.5 (see "
                         "--synth-burst-decay-tau), matching a real hard pick-attack -- so "
                         "saturation-onset behavior under a sharp transient gets tested at "
                         "EVERY level, not just the loudest. Typically the same list as "
                         "--sweep-peaks.")
    ap.add_argument("--synth-burst-decay-tau", type=float, default=0.03,
                    help="exponential decay time constant (s) for --synth-burst-peaks. "
                         "Default 0.03s gives crest factor ~8.5.")
    ap.add_argument("--synth-burst-dur", type=float, default=0.5,
                    help="duration (s) of each synthesized transient burst")
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

    noise_burst = None
    if args.noise_burst_peak is not None:
        if not args.noise_burst_window:
            raise SystemExit("--noise-burst-peak needs --noise-burst-window START,END")
        burst_src = args.noise_burst_src or args.input
        # Read fresh, UNTRUNCATED -- the burst window can fall past --realistic-dur's prefix
        # cutoff (e.g. T3K's own noise staircase runs to ~17s, past a 15s realistic-dur).
        bx, bsr = sf.read(burst_src, dtype="float32")
        if bx.ndim > 1: bx = bx[:, 0]
        if bsr != SR: raise SystemExit(f"--noise-burst-src sr {bsr} != {SR}")
        w0, w1 = (float(v) for v in args.noise_burst_window.split(","))
        window = bx[int(SR * w0):int(SR * w1)]
        if len(window) == 0:
            raise SystemExit(f"--noise-burst-window {args.noise_burst_window} is empty against "
                             f"{burst_src} ({len(bx)/SR:.1f}s)")
        noise_burst = _fade((window / max(np.abs(window).max(), 1e-9)
                             * args.noise_burst_peak).astype(np.float32), 10, 10)
        parts += [pad, noise_burst]

    synth_bursts = []
    synth_peaks = []
    if args.synth_burst_peaks:
        synth_peaks = [float(p) for p in args.synth_burst_peaks.split(",") if p.strip()]
        for a in synth_peaks:
            burst = _transient_burst(args.synth_burst_dur, a, args.synth_burst_decay_tau)
            synth_bursts.append(burst)
            parts += [pad, burst]

    comp = np.concatenate(parts)
    nf = int(SR * args.fade_out_ms / 1000)
    if nf > 0: comp[-nf:] *= np.sin(np.linspace(np.pi / 2, 0, nf)) ** 2

    sf.write(args.output, comp, SR, subtype="FLOAT")
    a = np.abs(comp)
    print(f"wrote {args.output}  dur {len(comp)/SR:.1f}s  peak {a.max():.3f}  rms {np.sqrt((comp**2).mean()):.4f}")
    for thr in sorted(set([0.5] + peaks)):
        print(f"  time >= {thr:.2f} V : {100*np.mean(a >= thr):6.3f}%")
    if noise_burst is not None:
        print(f"  noise burst: {len(noise_burst)/SR:.2f}s from {args.noise_burst_src or args.input} "
              f"[{args.noise_burst_window}]  rescaled peak {np.abs(noise_burst).max():.3f}")
    for peak, burst in zip(synth_peaks, synth_bursts):
        b_peak = np.abs(burst).max()
        b_rms = np.sqrt(np.mean(burst**2))
        print(f"  synth burst: {len(burst)/SR:.2f}s  peak={b_peak:.3f}  "
              f"crest={b_peak/(b_rms+1e-12):.2f}  (target level {peak:.3f})")

    # Recipe sidecar: HOW this excitation was built, not just which bytes it is. Without this,
    # a derived excitation's provenance chain stops at "some file named *_src95.wav" -- the exact
    # window/args used to cut it from the source are lost the moment the config-file comment that
    # (informally, inconsistently) recorded them is out of date or missing. gen_dataset_from_schx.py's
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
            "noise_burst_src": args.noise_burst_src or (args.input if noise_burst is not None else None),
            "noise_burst_window": args.noise_burst_window if noise_burst is not None else None,
            "noise_burst_peak": args.noise_burst_peak,
            "synth_burst_peaks": synth_peaks or None,
            "synth_burst_decay_tau": args.synth_burst_decay_tau if synth_peaks else None,
            "synth_burst_dur": args.synth_burst_dur if synth_peaks else None,
            "fade_out_ms": args.fade_out_ms,
        },
        "source": _audio_provenance(args.input),
        "output": {**_audio_provenance(args.output, x=comp),
                   "peak": round(float(a.max()), 6),
                   "rms": round(float(np.sqrt((comp ** 2).mean())), 6)},
    }
    recipe_path = Path(args.output).with_suffix(".recipe.json")
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"wrote {recipe_path}  (build recipe -- picked up automatically by gen_dataset_from_schx.py)")


if __name__ == "__main__":
    main()
