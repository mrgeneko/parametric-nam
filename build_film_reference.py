#!/usr/bin/env python3
"""Build the bundled, license-free reference clip for scan_film_runaway.py's FiLM/LeakyReLU
runaway check.

WHY THIS EXISTS. scan_film_runaway.py needs a reference clip DISTINCT from whatever a device's
own training excitation was built from -- testing a model against the exact signal it was
fit to can't reveal a gap that signal itself under-covered (see that module's docstring: two
published models blew up 80-260x at a knob-corner x transient combination their own training
excitation never exercised). A synthesized reference works exactly as well for this as a real
recording -- what matters is varied, hard transient dynamics across a wide frequency/level
range, not authenticity (see feedback memory on this exact point). Reusing a licensed capture
sweep (e.g. T3K-sweep-v3.wav) means every machine needs its own separately-downloaded copy at
a different path; this file is fully synthesized so it can be committed directly and used
identically everywhere.

USED ONLY as scan_film_runaway.py's --reference. Not a training excitation, not consumed by
any other tool.

DOES NOT ERODE as training excitations get more comprehensive. Extending a device's own chirp
floor (e.g. to 15 Hz, as done for the Tweed 5F6-A Full sag-ac fix) closes THAT device's gap on
THAT frequency axis -- it does not make this file's content a training example: different
sweep rate, different duration, different exact transient shape. The failure space this check
guards against is (knob corner) x (frequency) x (level) x (transient shape), which no finite
training excitation exhausts. This is the reactive half of a deliberately two-layered design:
grid_adequacy.py / check_transient_coverage.py are the proactive half (make training as
comprehensive as possible up front); this stays a useful, independent check regardless.

STRUCTURE: at each of several amplitude levels, four different stimulus shapes back to back --
a log-frequency sweep (sustained, one-frequency-at-a-time), a broadband Schroeder-phase
transient burst (the same deterministic construction build_excitation.py's --synth-burst-peaks
uses -- instant attack, exponential decay), a sustained white-noise segment (broadband, ALL
frequencies at once, unlike the sweep), and a single-sample pop (the sharpest attack physically
representable -- one non-zero sample, maximally broadband, decays instantly). Levels span quiet
to hot. Short silences are interspersed throughout, both between segments and as a few longer
dedicated gaps, so the reference also exercises silence-to-content transitions and quiescent
behavior, not just continuously "loud" content.

Deterministic: the sweep and burst constructions have no RNG at all. The white-noise segments
use a FIXED-SEED PRNG specifically so they are genuine statistical white noise (flat expected
power spectral density) rather than the burst's deterministic Schroeder-phase construction --
same seed always produces the same bytes.

Usage:
  python build_film_reference.py --output reference/film_runaway_reference.wav
"""
import argparse
import sys

import numpy as np
import soundfile as sf

from build_excitation import _log_sweep, _transient_burst, SR
from param_train import RECEPTIVE_FIELD_SAMPLES as RF_SAMPLES   # derive the floor, never hardcode it

DEFAULT_LEVELS = [0.1, 0.3, 0.6, 1.0, 1.5]
NOISE_SEED = 20260916


def _fade(x, in_n, out_n):
    y = x.copy()
    if in_n:
        y[:in_n] *= np.linspace(0, 1, in_n, dtype=np.float32)
    if out_n:
        y[-out_n:] *= np.linspace(1, 0, out_n, dtype=np.float32)
    return y


def _white_noise(sec, amp, rng):
    """Genuine statistical white noise (flat expected PSD), not the burst's deterministic
    Schroeder-phase construction -- fixed-seed rng makes it reproducible byte-for-byte."""
    x = rng.standard_normal(int(SR * sec)).astype(np.float32)
    x /= (np.abs(x).max() + 1e-12)
    return _fade(amp * x, 8, 8)


def _single_sample_pop(amp, silence_pad_s=0.05):
    """The sharpest attack physically representable: one non-zero sample, maximally
    broadband (a true impulse has flat spectral content across all frequencies), padded
    with silence on both sides so the scan's windowing has quiet context around it."""
    pad = int(SR * silence_pad_s)
    y = np.zeros(2 * pad + 1, dtype=np.float32)
    y[pad] = amp
    return y


def build(levels, sweep_dur, burst_dur, noise_dur, burst_decay_tau, f0, f1,
          lead_silence_s, gap_s, long_silence_s):
    rng = np.random.default_rng(NOISE_SEED)
    gap = np.zeros(int(SR * gap_s), dtype=np.float32)
    long_gap = np.zeros(int(SR * long_silence_s), dtype=np.float32)

    segs = []
    if lead_silence_s > 0:
        segs.append(np.zeros(int(SR * lead_silence_s), dtype=np.float32))
    for i, amp in enumerate(levels):
        sweep = _fade(_log_sweep(f0, f1, sweep_dur, amp), 8, 8)
        burst = _fade(_transient_burst(burst_dur, amp, burst_decay_tau), 8, 8)
        noise = _white_noise(noise_dur, amp, rng)
        pop = _single_sample_pop(amp)
        segs += [sweep, gap, burst, gap, noise, gap, pop, gap]
        if i % 2 == 1:
            # a dedicated longer silence every other level, so the reference also tests
            # silence-to-content transitions, not just back-to-back stimuli
            segs.append(long_gap)
    return np.concatenate(segs).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS),
                    help=f"comma list of amplitude levels, V (default: {DEFAULT_LEVELS})")
    ap.add_argument("--sweep-dur", type=float, default=4.0, help="seconds per level's sweep")
    ap.add_argument("--burst-dur", type=float, default=1.0, help="seconds per level's burst")
    ap.add_argument("--noise-dur", type=float, default=1.5,
                    help="seconds per level's white-noise segment")
    ap.add_argument("--burst-decay-tau", type=float, default=0.03)
    ap.add_argument("--f0", type=float, default=20.0,
                    help=f"sweep floor, Hz (default: %(default)s). RAISED FROM 3.0 ON "
                         f"2026-09-17. The old value was chosen to sit below any device's own "
                         f"chirp floor so this stayed a genuine stress test -- but a stress "
                         f"test has to probe something the model can REPRESENT. The A2 stack's "
                         f"receptive field is {RF_SAMPLES} samples / {1000.0*RF_SAMPLES/SR:.1f} ms, "
                         f"so the slowest periodicity resolvable inside one window is "
                         f"SR/RF = {SR/RF_SAMPLES:.2f} Hz; below that the input completes less "
                         f"than one cycle and is structurally indistinguishable from a slow DC "
                         f"drift. Probing there measures undefined behaviour, not instability. "
                         f"Measured cost of the old default: 12.5%% of the clip's energy sat "
                         f"below that floor and produced a 47x excursion in a published Mesa "
                         f"Orange that is CLEAN on real playing (0/576 across its trained grid) "
                         f"-- a phantom defect that cost a fleet-wide false alarm. 20 Hz keeps "
                         f"the clip a stress test (every device's own chirp floor is 40 Hz, and "
                         f"the capture chain high-passes at 18 Hz) while staying inside what the "
                         f"architecture can actually model.")
    ap.add_argument("--f1", type=float, default=20000.0)
    ap.add_argument("--lead-silence-s", type=float, default=1.0)
    ap.add_argument("--gap-s", type=float, default=0.2, help="short silence between segments")
    ap.add_argument("--long-silence-s", type=float, default=1.0,
                    help="dedicated longer silence inserted every other level")
    args = ap.parse_args()

    rf_floor = SR / RF_SAMPLES
    if args.f0 < rf_floor:
        print(f"WARNING: --f0 {args.f0} Hz is below the A2 receptive-field floor "
              f"({rf_floor:.2f} Hz = SR/{RF_SAMPLES}).\n"
              f"         Content there completes <1 cycle in the model's window and is "
              f"indistinguishable from a\n         slow DC drift -- anything it provokes is "
              f"undefined behaviour, not instability.", file=sys.stderr)
    levels = [float(v) for v in args.levels.split(",") if v.strip()]
    y = build(levels, args.sweep_dur, args.burst_dur, args.noise_dur, args.burst_decay_tau,
              args.f0, args.f1, args.lead_silence_s, args.gap_s, args.long_silence_s)
    sf.write(args.output, y, SR, subtype="FLOAT")
    print(f"wrote {args.output}  dur {len(y)/SR:.1f}s  levels={levels}  "
          f"sweep {args.f0}-{args.f1} Hz x {args.sweep_dur}s, burst {args.burst_dur}s "
          f"(tau={args.burst_decay_tau}), white noise {args.noise_dur}s, single-sample pop "
          f"per level  peak {np.abs(y).max():.4f}")


if __name__ == "__main__":
    main()
