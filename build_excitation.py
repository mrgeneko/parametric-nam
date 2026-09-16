#!/usr/bin/env python3
"""Build a device training excitation that actually covers the full input range.

NAMING (2026-09-10): TONE3000 calls this style of file a "sweep signal" (see
tone3000.com/create/capture) -- the term this tool now uses too, for what used to be called
the "--input"/"real" clip. --sweep-file is NOT necessarily a real-playing recording: e.g.
sweep-v3.wav (NAM's/TONE3000's standard capture sweep, ~22 dB crest) is itself a synthesized
capture sweep (frequency sweep + noise-staircase + calibration blips), not musical dynamics --
its high crest factor comes from that structure. A genuine real-playing recording works
equally well here and is not required to be this specific file. Separately, this tool ALSO
generates its own internal amplitude-stepped log sine tones -- those are called "chirps"
(--chirp-levels/--chirp-f0/--chirp-f1/--chirp-dur) specifically so the name does not collide
with --sweep-file, which is an unrelated, externally-supplied concept.

A high-crest --sweep-file clip samples its own loud region essentially never (<0.1% of time
within 6 dB of peak), so a model trained on it alone never learns the device's
saturation/blocking behavior and goes out-of-distribution when a hot input (upstream boost)
arrives. This concatenates:

  [ --sweep-file clip @ --sweep-peak ]       (dynamics/perceptual realism, mostly low level)
  [ amplitude-stepped log sine chirps ]      (dense level x frequency coverage of the loud region)
  [ short fade-out to zero ]

so the whole 0 -> --chirp-levels[-1] transfer, including the memory-dependent blocking region,
is genuinely learned. Under the V0dBFS=1V convention a sample value == drive volts, so pass
--chirp-levels in volts and set the max to the device's saturation/max-output point + headroom.
Written float32 so values >1.0 survive (they represent >1 V drive, which is legitimate).

LEADING SILENCE (--lead-silence-s, default 3.0): every render starts a `.tran` from a cold,
all-capacitors-at-0V initial condition, not the already-biased-up state a real (already
powered-on) device is always in. For a circuit with a slow-charging DC-blocking network (e.g.
a large output-coupling cap into a high-value pot -- the MOSFET-clipping pedal's C10/Volume-pot
leg has a ~5s RC time constant), starting real content at t=0 captures a genuine but
non-representative multi-second "circuit powering on" transient: measured directly on that
pedal (ngspice backend, see its ngspice-generator script in the private devices repo), a
sustained tone with no lead-in showed RMS slowly drifting for ~15s and then an ABRUPT jump to a
different steady value at ~16s -- neither of which a real, already-running pedal ever does.
Prepending 3s of silence before any real content let the circuit reach its true,
cold-start-independent operating bias first; every render taken after that showed the tone
snapping to a single stable, unchanging level within about a second of starting. Cheap fix,
not device-specific -- applied by default to every excitation this tool builds, silent segment
included in the file (not stripped after generation), so it also gives the DC bias network real
settling time before the `real` segment's own dynamics are what's being sampled. See internal
engineering notes and that pedal's own investigation for the concrete before/after traces this
was based on.
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
    """Broadband (20 Hz - 16 kHz, Schroeder-phase deterministic construction) burst with an
    INSTANT attack (no fade-in) and an exponential decay, mimicking a real hard
    pick-attack's amplitude envelope (high crest
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
    # matching --chirp-levels/--sweep-peak's own contract elsewhere in this file.
    y /= (np.abs(y).max() + 1e-12)
    return (amp * y).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-file", required=True,
                    help="high-crest source clip, TONE3000's term for this style of file "
                         "(e.g. sweep-v3.wav, itself a synthesized capture sweep, not a "
                         "real-playing recording -- see module docstring)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sweep-peak", type=float, default=1.0,
                    help="peak (V, at V0dBFS=1) to scale the --sweep-file clip to")
    ap.add_argument("--sweep-dur", type=float, default=None,
                    help="seconds of --sweep-file to keep (prefix), default: the whole file. "
                         "This segment only needs to sample varied dynamics/perceptual content "
                         "-- it is not what provides saturation coverage (the chirps do) -- so a "
                         "short prefix is normally enough, and every second of it costs a "
                         "render-time multiplier across the whole combination grid.")
    ap.add_argument("--chirp-levels", default="0.5,1.0,1.5,2.0",
                    help="comma list of sine-chirp peak amplitudes (V); last = training max drive")
    ap.add_argument("--chirp-f0", type=float, default=15.0,
                    help="sweep floor (Hz). Was 40 -- raised to 15 (2026-09-16) after a "
                         "tweed-style amp's excitation never chirping below 40 Hz left its "
                         "trained model with zero supervision for sustained near-DC (<20 Hz) "
                         "input, which blew up 8x on a real capture sweep's own infrasonic "
                         "segment at a corner no amount of amplitude-only sizing would have "
                         "caught (see scan_film_runaway.py). Sizing is unaffected by this knob "
                         "-- it comes from a separate onset probe, not the chirp itself.")
    ap.add_argument("--chirp-f1", type=float, default=12000.0)
    ap.add_argument("--chirp-dur", type=float, default=6.0,
                    help="seconds per amplitude step. Was 3 -- raised to 6 (2026-09-16) "
                         "alongside the --chirp-f0 floor drop so time-per-octave doesn't "
                         "shrink: a log sweep spends time per OCTAVE, not per Hz, so widening "
                         "the sweep (40-12000 Hz, 8.23 octaves) to (15-12000 Hz, 9.64 octaves) "
                         "at the old 3s would have cut density from 0.365 to 0.311 s/octave "
                         "fleet-wide, not just at the new low end. 6s instead raises it to "
                         "0.622 s/octave -- denser than before, not just not-worse.")
    ap.add_argument("--noise-burst-src", default=None,
                    help="source WAV to pull a broadband white-noise burst staircase from -- "
                         "default: same file as --sweep-file. A sine chirp only tests ONE "
                         "frequency at a time at each level; a broadband noise ATTACK (0 -> "
                         "level, sharp rise) stress-tests saturation onset the way a real pick "
                         "attack does, across the whole spectrum at once, at a level the "
                         "chirp-tail alone doesn't guarantee (see check_transient_coverage.py's "
                         "docstring -- this is the same gap that under-covered the tweed-style "
                         "amp originally). T3K-sweep-v3 has exactly this built in already (a "
                         "3-step discrete noise staircase plus a continuous swell, found by "
                         "scanning for high spectral-flatness windows) -- reusing it beats "
                         "synthesizing a new one from scratch.")
    ap.add_argument("--noise-burst-window", default=None, metavar="START,END",
                    help="seconds into --noise-burst-src to extract (e.g. '12.0,17.0' for "
                         "T3K-sweep-v3's own noise staircase). Required if --noise-burst-src "
                         "or a --noise-burst-peak is given.")
    ap.add_argument("--noise-burst-peak", type=float, default=None,
                    help="peak (V) to rescale the extracted noise-burst window to -- scales the "
                         "WHOLE window by one factor, preserving its internal staircase shape "
                         "(so the low steps land at the same fraction of this peak the source "
                         "window has). Typically the same as --chirp-levels' last value. Omit to "
                         "skip the noise-burst segment entirely (default: off, matches prior "
                         "behavior).")
    ap.add_argument("--synth-burst-peaks", default=None,
                    help="comma list of synthesized transient-burst peak amplitudes (V) -- a "
                         "deterministic, license-free alternative to --noise-burst-* (which "
                         "depends on a restricted third-party source, e.g. T3K-sweep-v3's own "
                         "noise staircase, and only ever tests ONE level). Inserts one burst "
                         "per level: broadband (20Hz-16kHz), instant "
                         "attack, exponential decay -- crest factor ~8.5 (see "
                         "--synth-burst-decay-tau), matching a real hard pick-attack -- so "
                         "saturation-onset behavior under a sharp transient gets tested at "
                         "EVERY level, not just the loudest. Typically the same list as "
                         "--chirp-levels.")
    ap.add_argument("--synth-burst-decay-tau", type=float, default=0.03,
                    help="exponential decay time constant (s) for --synth-burst-peaks. "
                         "Default 0.03s gives crest factor ~8.5.")
    ap.add_argument("--synth-burst-dur", type=float, default=0.5,
                    help="duration (s) of each synthesized transient burst")
    ap.add_argument("--fade-out-ms", type=float, default=150.0)
    ap.add_argument("--lead-silence-s", type=float, default=3.0,
                    help="silence prepended before any real content, so a slow DC-blocking "
                         "network (e.g. a large output-coupling cap into a high-value pot) "
                         "reaches its true bias before the excitation's own dynamics start -- "
                         "see module docstring's LEADING SILENCE note. 0 to disable.")
    args = ap.parse_args()

    x, sr = sf.read(args.sweep_file, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    if sr != SR: raise SystemExit(f"sweep-file sr {sr} != {SR}")
    if args.sweep_dur is not None:
        x = x[:int(SR * args.sweep_dur)]

    sweep_seg = _fade((x / max(np.abs(x).max(), 1e-9) * args.sweep_peak).astype(np.float32), 10, 10)
    chirp_levels = [float(p) for p in args.chirp_levels.split(",") if p.strip()]
    pad = np.zeros(int(SR * 0.1), dtype=np.float32)
    lead_silence = np.zeros(int(SR * args.lead_silence_s), dtype=np.float32)
    parts = ([lead_silence] if args.lead_silence_s > 0 else []) + [sweep_seg, pad] + \
            [_fade(_log_sweep(args.chirp_f0, args.chirp_f1, args.chirp_dur, a), 8, 8)
             for a in chirp_levels]

    noise_burst = None
    if args.noise_burst_peak is not None:
        if not args.noise_burst_window:
            raise SystemExit("--noise-burst-peak needs --noise-burst-window START,END")
        burst_src = args.noise_burst_src or args.sweep_file
        # Read fresh, UNTRUNCATED -- the burst window can fall past --sweep-dur's prefix
        # cutoff (e.g. T3K's own noise staircase runs to ~17s, past a 15s sweep-dur).
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
    for thr in sorted(set([0.5] + chirp_levels)):
        print(f"  time >= {thr:.2f} V : {100*np.mean(a >= thr):6.3f}%")
    if noise_burst is not None:
        print(f"  noise burst: {len(noise_burst)/SR:.2f}s from {args.noise_burst_src or args.sweep_file} "
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
            "lead_silence_s": args.lead_silence_s,
            "sweep_peak": args.sweep_peak,
            "sweep_dur": args.sweep_dur,
            "chirp_levels": chirp_levels,
            "chirp_f0": args.chirp_f0,
            "chirp_f1": args.chirp_f1,
            "chirp_dur": args.chirp_dur,
            "noise_burst_src": args.noise_burst_src or (args.sweep_file if noise_burst is not None else None),
            "noise_burst_window": args.noise_burst_window if noise_burst is not None else None,
            "noise_burst_peak": args.noise_burst_peak,
            "synth_burst_peaks": synth_peaks or None,
            "synth_burst_decay_tau": args.synth_burst_decay_tau if synth_peaks else None,
            "synth_burst_dur": args.synth_burst_dur if synth_peaks else None,
            "fade_out_ms": args.fade_out_ms,
        },
        "source": _audio_provenance(args.sweep_file),
        "output": {**_audio_provenance(args.output, x=comp),
                   "peak": round(float(a.max()), 6),
                   "rms": round(float(np.sqrt((comp ** 2).mean())), 6)},
    }
    recipe_path = Path(args.output).with_suffix(".recipe.json")
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"wrote {recipe_path}  (build recipe -- picked up automatically by gen_dataset_from_schx.py)")


if __name__ == "__main__":
    main()
