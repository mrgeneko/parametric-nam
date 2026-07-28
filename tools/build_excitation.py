#!/usr/bin/env python3
"""Build a device training excitation that actually covers the full input range.

A high-crest real-playing clip (e.g. [redacted], ~22 dB crest) samples its own loud region
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
import numpy as np
import soundfile as sf

SR = 48000


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
    ap.add_argument("--input", required=True, help="real-playing clip (e.g. [redacted] sweep)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--realistic-peak", type=float, default=1.0,
                    help="peak (V, at V0dBFS=1) to scale the real-playing clip to")
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


if __name__ == "__main__":
    main()
