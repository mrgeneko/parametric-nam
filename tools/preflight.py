#!/usr/bin/env python3
"""Pre-generation sanity gate for a device dataset.

Run this BEFORE rendering a (slow, large) training dataset. It renders a handful
of short probe points through the oracle and refuses generation when a knob is
dead, moves the WRONG WAY, or the input-level calibration is implausible -- the
three failure modes that have each cost us a full render + train cycle:

  * dead knob      -- Klon Tone, Deluxe Reverb Middle (0.000 effect)
  * reversed knob  -- Fulltone OCD Tone (treble FELL as the knob rose)
  * input mis-cal  -- input_level_dbu ~25 dB hot (V0dBFS is a circuit-drive
                      voltage, not an interface reference)

Exit status is nonzero if any HARD check fails, so a generation script can gate
on it:  `python tools/preflight.py ... && python batch_harness.py ...`

Usage:
  python tools/preflight.py --schx DEVICE.schx --knobs Gain,Tone --input sweep.wav \
     [--fixed-params "Volume=1.0"] [--oversample 8] [--iterations 256] \
     [--seconds 10] [--esr-dead 1e-3] [--json report.json]

Direction convention: a knob turned UP (value 0->1) should INCREASE the quantity
it is named for -- Gain/Drive -> more distortion+level, Tone/Treble -> more
treble, Bass -> more lows, Volume/Level -> louder. Names we don't recognize are
checked for responsiveness only (direction reported as 'unknown').
"""
import argparse, json, math, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from batch_harness import LIVESPICE_CLI, parse_schx_controls, resolve_knobs  # noqa: E402
from param_train import _schx_input_v0dbfs, _input_level_dbu, INPUT_LEVEL_DBU_REF  # noqa: E402

SR = 48000
# (name keywords, metric, human label). First match wins; order specific->general.
# metric returns a scalar that should RISE as the knob rises.
DIRECTION_RULES = [
    (("treble", "tone", "bright", "presence", "high", "top", "tref"), "hi",  "treble/high-freq"),
    (("bass", "low", "depth", "body", "bottom", "sub"),               "lo",  "bass/low-freq"),
    (("mid", "middle"),                                               "mid", "midrange"),
    (("gain", "drive", "dist", "overdrive", "fuzz", "sustain", "sat", "od", "pre"), "drive", "gain/distortion"),
    (("volume", "level", "master", "output", "vol", "loud", "post"),  "rms", "output level"),
]


def _band(y, f_lo, f_hi):
    Y = np.abs(np.fft.rfft(y.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(Y[(f >= f_lo) & (f < f_hi)].sum())


def metric(y, kind):
    y = y[int(0.1 * SR):]  # drop attack
    if kind == "rms" or kind == "drive":
        return float(np.sqrt((y ** 2).mean()) + 1e-12)
    if kind == "hi":
        return _band(y, 2000, 8000) / (_band(y, 40, 800) + 1e-12)
    if kind == "lo":
        return _band(y, 40, 200) / (_band(y, 400, 4000) + 1e-12)
    if kind == "mid":
        return _band(y, 300, 1500) / (_band(y, 40, 200) + _band(y, 3000, 8000) + 1e-12)
    raise ValueError(kind)


def classify(knob):
    n = knob.lower()
    for keys, kind, label in DIRECTION_RULES:
        if any(k in n for k in keys):
            return kind, label
    return None, "unknown"


def esr(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return float(((a - b) ** 2).sum() / ((b ** 2).sum() + 1e-12))


def render(schx, params, oversample, iterations, in_wav, scratch, tag):
    out = f"{scratch}/pf_{tag}.wav"
    kv = ",".join(f"{k}={v}" for k, v in params.items())
    r = subprocess.run([str(LIVESPICE_CLI), "--input", in_wav, "--output", out,
                        "--circuit", schx, "--params", kv,
                        "--oversample", str(oversample), "--iterations", str(iterations)],
                       capture_output=True, text=True)
    try:
        y, _ = sf.read(out, dtype="float32")
        return y[:, 0] if y.ndim > 1 else y
    except Exception:
        sys.stderr.write(r.stderr[-400:] + "\n")
        return None


def spikes(y):
    a = np.abs(y); p99 = np.percentile(a, 99)
    ln = np.maximum(np.abs(np.roll(y, 1)), np.abs(np.roll(y, -1)))
    return int(((a > 3 * ln) & (a > 2 * p99)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schx", required=True)
    ap.add_argument("--knobs", required=True, help="comma-separated knobs to sweep")
    ap.add_argument("--input", required=True, help="the sweep that generation will use")
    ap.add_argument("--fixed-params", default="", help="e.g. 'Volume=1.0' — held during probing")
    ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=10.0, help="probe-input length")
    ap.add_argument("--esr-dead", type=float, default=1e-3)
    ap.add_argument("--dir-margin", type=float, default=0.05,
                    help="min relative metric change to call a direction (else inconclusive)")
    ap.add_argument("--input-level-dbu", type=float, default=None,
                    help="override the exported interface reference for the check "
                         "(default: param_train.INPUT_LEVEL_DBU_REF)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    knobs = [k.strip() for k in args.knobs.split(",") if k.strip()]
    control_map = parse_schx_controls(args.schx)
    resolve_knobs(knobs, control_map)  # hard-fails on a typo'd knob name
    fixed = {}
    for kv in filter(None, (s.strip() for s in args.fixed_params.split(","))):
        k, v = kv.split("="); fixed[k.strip()] = float(v)

    hard_fail, warn = [], []
    report = {"schx": args.schx, "knobs": {}, "input_calibration": {}}

    with tempfile.TemporaryDirectory() as scratch:
        # short probe input at native level (no normalization)
        x, sr = sf.read(args.input, dtype="float32")
        if x.ndim > 1: x = x[:, 0]
        if sr != SR:
            sys.exit(f"input sample rate {sr} != {SR}")
        probe = f"{scratch}/probe.wav"
        sf.write(probe, x[: int(args.seconds * SR)], SR)

        print(f"Preflight: {Path(args.schx).name}  knobs={knobs}  fixed={args.fixed_params or '-'}")
        print(f"  probe {args.seconds:.0f}s @ os={args.oversample} it={args.iterations}\n")

        for knob in knobs:
            kind, label = classify(knob)
            base = {k: 0.5 for k in knobs}; base.update(fixed)
            ys = {}
            for tag, val in (("lo", 0.2), ("mid", 0.5), ("hi", 0.8)):
                p = dict(base); p[knob] = val
                ys[tag] = render(args.schx, p, args.oversample, args.iterations, probe, scratch, f"{knob}_{tag}")
            if any(v is None for v in ys.values()):
                hard_fail.append(f"{knob}: RENDER FAILED"); report["knobs"][knob] = {"status": "render_fail"}
                print(f"  {knob:10} RENDER FAILED"); continue

            e = esr(ys["hi"], ys["lo"])
            sp = max(spikes(ys[t]) for t in ys)
            peak = max(float(np.abs(ys[t]).max()) for t in ys)
            rec = {"esr_hi_lo": e, "direction_metric": kind, "label": label,
                   "spikes": sp, "peak": peak}

            if e < args.esr_dead:
                hard_fail.append(f"{knob}: DEAD (ESR={e:.4g})"); rec["status"] = "dead"
                print(f"  {knob:10} DEAD           ESR={e:.4g}  (expected to affect {label})")
            elif kind is None:
                warn.append(f"{knob}: responds but name unrecognized — direction NOT checked")
                rec["status"] = "responds_dir_unknown"
                print(f"  {knob:10} responds       ESR={e:.4g}  direction UNCHECKED (unknown knob '{knob}')")
            else:
                m_lo, m_hi, m_mid = (metric(ys["lo"], kind), metric(ys["hi"], kind), metric(ys["mid"], kind))
                rel = (m_hi - m_lo) / (abs(m_lo) + 1e-12)
                rec.update({"metric_lo": m_lo, "metric_mid": m_mid, "metric_hi": m_hi, "metric_rel_change": rel})
                mono = (m_lo <= m_mid <= m_hi) or (m_lo >= m_mid >= m_hi)
                if abs(rel) < args.dir_margin:
                    warn.append(f"{knob}: responds (ESR={e:.4g}) but {label} metric barely moved ({rel:+.1%}) — direction inconclusive")
                    rec["status"] = "dir_inconclusive"
                    print(f"  {knob:10} responds       ESR={e:.4g}  {label} {rel:+.1%} INCONCLUSIVE")
                elif rel > 0:
                    rec["status"] = "ok"
                    tail = "" if mono else "  (non-monotonic!)"
                    print(f"  {knob:10} OK             ESR={e:.4g}  {label} {rel:+.1%} rising{tail}")
                    if not mono: warn.append(f"{knob}: direction OK but non-monotonic across 0.2/0.5/0.8")
                else:
                    hard_fail.append(f"{knob}: REVERSED ({label} {rel:+.1%} — falls as knob rises)")
                    rec["status"] = "reversed"
                    print(f"  {knob:10} REVERSED       ESR={e:.4g}  {label} {rel:+.1%} (should RISE)")
            if sp > 0:
                warn.append(f"{knob}: {sp} sample-spike(s) in probe renders")
            report["knobs"][knob] = rec

        # ---- input-level calibration ----
        # What export_nam will actually WRITE (a device may override the standard ref).
        exported_ild = float(args.input_level_dbu) if args.input_level_dbu is not None else INPUT_LEVEL_DBU_REF
        v0 = _schx_input_v0dbfs(args.schx)
        in_peak = float(np.abs(x).max())
        ic = {"v0dbfs_volts": v0, "input_peak_dbfs": 20 * math.log10(in_peak + 1e-12),
              "exported_input_level_dbu": exported_ild}
        print("\n  input calibration:")
        print(f"    training input peak = {in_peak:.3f} ({ic['input_peak_dbfs']:.1f} dBFS)")
        if v0 is not None:
            # V0dBFS is the circuit-drive voltage; report it as FYI only (NOT exported as dbu).
            print(f"    schx V0dBFS = {v0} V (circuit drive; informational)")
        print(f"    input_level_dbu to be EXPORTED = {exported_ild:.2f} dBu")
        # A real interface's 0dBFS reference is ~+4..+24 dBu; outside that a calibrating host
        # mis-gains (the old V0dBFS-derived -20.8 ran ~25-35 dB hot).
        if not (-10.0 <= exported_ild <= 30.0):
            warn.append(f"exported input_level_dbu={exported_ild:.1f} dBu is outside the plausible "
                        f"interface range (-10..+30 dBu) — a calibrating host will mis-gain. "
                        f"Check INPUT_LEVEL_DBU_REF / config['input_level_dbu'].")
            print(f"    ^ IMPLAUSIBLE interface reference")
        report["input_calibration"] = ic

    print()
    for w in warn:  print(f"  WARN: {w}")
    for f in hard_fail: print(f"  FAIL: {f}")
    report["warnings"] = warn; report["failures"] = hard_fail
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    if hard_fail:
        print(f"\nPREFLIGHT FAILED ({len(hard_fail)} hard issue(s)) — do NOT generate.")
        sys.exit(1)
    print(f"\nPREFLIGHT PASSED{' with warnings' if warn else ''}.")


if __name__ == "__main__":
    main()
