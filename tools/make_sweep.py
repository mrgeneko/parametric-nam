#!/usr/bin/env python3
"""Build a training sweep: a generated excitation section + a real playing section.

WHY A GENERATED SECTION AT ALL.
`sweepv4.wav` is real guitar, and real guitar barely excites the top of the band. Measured on it:

    80 Hz – 1.28 kHz    71% of the energy
    2.5 – 5 kHz          2.6%
    5 – 10 kHz           1.9%
    10 – 20 kHz          0.2%

A model can only learn the circuit's response where the INPUT has energy. With 0.2% above 10 kHz the
tone stack's top end, the presence control and the treble taper are essentially unconstrained by the
data -- the model is free to invent them, and it will. A log chirp costs a few seconds and pins the
whole band.

WHY MORE THAN ONE CHIRP.
The circuit is NONLINEAR, so its response is not one curve, it is a curve PER LEVEL. A single
full-scale chirp teaches the band only where the amp is clipping; a single quiet chirp teaches it
only where the amp is clean. Both are needed, and so is the continuum between them -- hence the
amplitude ramp, which walks a fixed tone from silence to full scale and back so the model sees the
transfer curve swept, not sampled.

WHY NO HARD STEPS OR SPLICES.
Every segment is fade-joined through silence. An abrupt amplitude discontinuity in the INPUT is not
just unrealistic, it breaks the simulator: an abrupt volume-jump splice in an earlier sweep was what
caused ngspice convergence failures at that exact timestamp (see the sweep-files README, sweep45s).
The emitter's own test signal DOES use a full-scale single-sample step -- deliberately, to stress
Newton -- but that is a STABILITY probe, not training data.

DETERMINISTIC. Same inputs -> same bytes. No RNG anywhere: the "noise" burst is a fixed
frequency-domain construction, not random. A training input that cannot be regenerated is a training
input whose datasets cannot be reproduced, which is the mistake that lost sweep120s.

    ./tools/make_sweep.py --playing ../sweep-files/sweepv4.wav -o ../sweep-files/sweepv5.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000


def _fade(x: np.ndarray, sr: int, ms: float = 15.0) -> np.ndarray:
    """Raised-cosine fade in/out. Every segment gets one, so no join is a discontinuity."""
    n = max(1, int(sr * ms / 1000))
    n = min(n, len(x) // 2)
    w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    x = x.copy()
    x[:n] *= w
    x[-n:] *= w[::-1]
    return x


def log_chirp(sec: float, f0: float, f1: float, amp: float, sr: int = SR) -> np.ndarray:
    """Logarithmic sine sweep. Log, not linear: it spends equal time per OCTAVE, which is how the
    circuit's filters are spaced and how we hear. A linear sweep dawdles at the top, where a guitar
    amp has almost nothing to say, and races through the bass, where it has everything."""
    t = np.arange(int(sec * sr)) / sr
    k = (f1 / f0) ** (1.0 / sec)
    phase = 2 * np.pi * f0 * (k ** t - 1) / np.log(k)
    return _fade(amp * np.sin(phase), sr)


def amp_ramp(sec: float, freq: float, peak: float, sr: int = SR) -> np.ndarray:
    """A fixed tone walked from silence to `peak` and back. This is the one segment that teaches the
    TRANSFER CURVE rather than the frequency response: the amp's gain, its knee, and where it folds
    over are all functions of level, and nothing else in the sweep traverses level continuously."""
    t = np.arange(int(sec * sr)) / sr
    env = np.sin(np.pi * t / sec) ** 2          # 0 -> 1 -> 0, smooth at both ends
    return _fade(peak * env * np.sin(2 * np.pi * freq * t), sr)


def flat_burst(sec: float, amp: float, sr: int = SR) -> np.ndarray:
    """Broadband, dense-spectrum excitation -- every bin from 20 Hz to 16 kHz at equal magnitude with
    a deterministic quadratic phase (a Schroeder-flat signal). Serves the same purpose as noise but
    is REPRODUCIBLE: no RNG, so the same command always writes the same bytes."""
    n = int(sec * sr)
    f = np.fft.rfftfreq(n, 1 / sr)
    mag = ((f >= 20) & (f <= 16000)).astype(float)
    k = np.arange(len(f))
    spec = mag * np.exp(1j * (np.pi * k ** 2 / max(mag.sum(), 1)))   # Schroeder phase
    x = np.fft.irfft(spec, n)
    x /= (np.abs(x).max() + 1e-12)
    return _fade(amp * x, sr)


def silence(sec: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sec * sr))


def build_excitation(sr: int = SR) -> np.ndarray:
    """~27 s. Ordered quiet -> loud so the file eases the solver in rather than slamming it.

    Kept SHORT on purpose. Render cost is linear in length and every second here is paid once per
    knob permutation per device, on top of the 4x that oversample=8 already costs. 6 s of log chirp
    is ~0.6 s per octave over 20 Hz - 20 kHz, which is plenty; 9 s bought no more coverage, only a
    55% longer file.
    """
    gap = silence(0.30, sr)
    return np.concatenate([
        # 1 s of LEADING SILENCE, and both reasons matter:
        #
        #   THE SOLVER needs it. A file that starts mid-signal asks the simulator to jump straight to
        #   a large-signal solution from an uninitialised state at t=0, and ngspice simply DIVERGES
        #   ("diverged at t~0", every rung, no output). The Boss DS-1 fails on an 8 s slice of this
        #   very sweep and succeeds on the whole file, for exactly this reason.
        #
        #   THE WARMUP CONVENTION is 1.0 s. batch_harness discards the first warmup_s=1.0 when it
        #   computes stats and ESR, to keep a startup transient out of the number. sweepv4 opens with
        #   1.048 s of silence and satisfies both. My first cut used 0.25 s -- which meant the 1 s
        #   warmup skip was EATING 0.75 s OF REAL SIGNAL: the opening of the clean 20 Hz chirp, i.e.
        #   the low end of the quiet sweep, was never scored by anything.
        silence(1.0, sr),
        log_chirp(6.0, 20, 20000, 0.10, sr),   # clean:  the LINEAR response. Tone stack, filters.
        gap,
        log_chirp(6.0, 20, 20000, 0.45, sr),   # driven: the band as it breaks up.
        gap,
        log_chirp(6.0, 20, 20000, 0.90, sr),   # hot:    the band under heavy clipping.
        gap,
        amp_ramp(6.0, 220.0, 0.90, sr),        # the transfer curve, swept continuously.
        gap,
        flat_burst(1.5, 0.50, sr),             # dense broadband, deterministic.
        silence(0.40, sr),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playing", type=Path, required=True,
                    help="the real-playing sweep to append (e.g. sweep-files/sweepv4.wav)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--peak", type=float, default=0.90,
                    help="hard ceiling; the file is scaled down if it would exceed this (default 0.9). "
                         "Never let a training input clip: the model would learn the clipper.")
    args = ap.parse_args()

    play, sr = sf.read(str(args.playing), dtype="float64")
    if play.ndim > 1:
        play = play.mean(axis=1)
    if sr != SR:
        ap.error(f"{args.playing} is {sr} Hz; this generator assumes {SR}")

    exc = build_excitation(sr)
    out = np.concatenate([exc, silence(0.5, sr), play])

    pk = float(np.abs(out).max())
    if pk > args.peak:
        out *= args.peak / pk
        print(f"  scaled by {args.peak / pk:.4f} to hold peak <= {args.peak}")

    sf.write(str(args.output), out.astype(np.float32), sr, subtype="FLOAT")
    print(f"  wrote {args.output}")
    print(f"    excitation {len(exc)/sr:6.1f}s + playing {len(play)/sr:6.1f}s "
          f"= {len(out)/sr:.1f}s   peak={np.abs(out).max():.3f} "
          f"rms={np.sqrt((out**2).mean()):.4f}")


if __name__ == "__main__":
    main()
