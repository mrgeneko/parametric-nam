#!/usr/bin/env python3
"""Pre-generation sanity gate for a device dataset -- backend-agnostic (--backend
{livespice,ngspice-deck,ltspice-deck}, see render_backends.py). Run this BEFORE rendering a (slow, large)
training dataset. It renders a handful of short probe points through the oracle and refuses
generation when a knob is dead, moves the WRONG WAY, or the input-level calibration is
implausible -- the three failure modes that have each cost a full render + train cycle:

  * dead knob      -- the transparent-overdrive pedal's Tone, the low-wattage clean combo amp's Middle (0.000 effect)
  * reversed knob  -- the MOSFET-clipping pedal's Tone (treble FELL as the knob rose)
  * input mis-cal  -- input_level_dbu ~25 dB hot (V0dBFS is a circuit-drive
                      voltage, not an interface reference)

Exit status is nonzero if any HARD check fails, so a generation script can gate on it:
  python preflight.py --backend livespice ... && python gen_dataset_from_schx.py ...

Direction convention: a knob turned UP (value 0->1) should INCREASE the quantity it is named
for -- Gain/Drive -> more distortion+level, Tone/Treble -> more treble, Bass -> more lows,
Volume/Level -> louder. Names we don't recognize are checked for responsiveness only
(direction reported as 'unknown').

LEVEL-AWARE EQ CHECKS: a passive tone stack's direction is level-independent, but hard
clipping SWAMPS it -- on a circuit driven near its saturation ceiling, turning an EQ knob up
can read as a FALSE REVERSAL because the extra content just becomes more clipping mush, not
more clean signal at that band. Two independent, complementary mitigations, both on by
default:

  1. ROLE-AWARE BASELINE (--eq-check-drive-level, cheap, no extra renders): while checking an
     EQ/tone knob, every drive/gain-classified OTHER knob (classify()'s 'drive' kind) is held
     LOW instead of the usual 0.5 center, so the circuit isn't already self-saturating from
     its own gain control regardless of input level. Confirmed directly on the MOSFET-clipping pedal's
     ngspice deck: this alone fixed a false Tone-REVERSED reading that --find-peak's input-
     level scaling (below) barely moved.
  2. INPUT-LEVEL SCALING (--find-peak / --clean-probe-peak): when a saturation ceiling is
     available, EQ/tone knobs are ALSO probed at a LINEAR-region input level (half the top of
     the flat-gain region, which catches early preamp clipping AND gain-expansion, not just
     the RMS ceiling) alongside the native/driven level: PASS if EITHER shows a clear correct
     direction, hard-FAIL only if BOTH reverse, WARN when they disagree (level-dependent, not
     necessarily wrong). Without a ceiling, EQ knobs use the native level only (legacy).

Drive/level knobs are never touched by either mitigation -- they must see the full range
including saturation.

Usage:
  livespice: python preflight.py --backend livespice --schx DEVICE.schx --knobs Gain,Tone \\
      --input sweep.wav [--fixed-params "Volume=1.0"] [--oversample 8] [--iterations 256] \\
      [--find-peak] [--json report.json]

  ngspice:   python preflight.py --backend ngspice-deck --pedal-dir ~/work/parametric-devices/pedals \\
      --module gen_ocd_ngspice --probe-node OUT --input sweep.wav [--vin 1.0] \\
      [--exclude-knob Volume] [--find-peak]

  ltspice:   python preflight.py --backend ltspice-deck --pedal-dir ~/work/parametric-devices/pedals \\
      --module gen_ocd_ltspice --probe-node spk --input sweep.wav [--vin 1.0] \\
      [--exclude-knob Volume] [--find-peak]
"""
import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_dataset_from_schx import parse_schx_controls, resolve_knobs  # noqa: E402
from param_train import _schx_input_v0dbfs, _input_level_dbu  # noqa: E402

from find_saturation_point import (find_saturation_point, _linear_region_top,  # noqa: E402
                                   findpeak_cache_key, scratch_dir)
from render_backends import LiveSpiceBackend, NgspiceBackend, LtspiceBackend  # noqa: E402
from knob_classify import classify as _classify_by_name  # noqa: E402

SR = 48000
_DBU_0_RMS_VOLTS = 0.7746  # 0dBu reference, matches param_train.py


_WINDOW_CACHE = {}


def _band(y, f_lo, f_hi):
    """Energy in [f_lo, f_hi). Hann-windowed -- NOT optional.

    An unwindowed (rectangular) FFT has -13dB first sidelobes falling only 6dB/octave, so a
    loud band leaks into a quiet one and swamps it. That is not hypothetical: on the budget
    clone pedal the 40-200Hz band sits ~51dB above 400-4000Hz, and turning Bass up 11x
    raised the *measured* 400-4000 energy 114x (2.23 -> 255.1) -- pure leakage. Both bands
    then scale together, the lo ratio moves a few percent, its sign is noise, and preflight
    called a correctly-wired Bass control REVERSED. Hann-windowed, the same band reads
    0.0593 -> 0.0544 (flat, as a direct per-frequency sweep confirms) and the ratio rises
    +8118% -- unambiguous. Ratios are unaffected by the window's energy scaling, since every
    band in a given metric is windowed identically."""
    y = y.astype(np.float64)
    w = _WINDOW_CACHE.get(len(y))
    if w is None:
        w = np.hanning(len(y))
        _WINDOW_CACHE[len(y)] = w
    Y = np.abs(np.fft.rfft(y * w)) ** 2
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


def esr(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return float(((a - b) ** 2).sum() / ((b ** 2).sum() + 1e-12))


def spikes(y):
    a = np.abs(y); p99 = np.percentile(a, 99)
    ln = np.maximum(np.abs(np.roll(y, 1)), np.abs(np.roll(y, -1)))
    return int(((a > 3 * ln) & (a > 2 * p99)).sum())


def _build_backend(args):
    if args.backend == "livespice":
        if not args.schx:
            sys.exit("--backend livespice needs --schx")
        knobs = [k.strip() for k in args.knobs.split(",") if k.strip()]
        control_map = parse_schx_controls(args.schx)
        resolve_knobs(knobs, control_map)  # hard-fails on a typo'd knob name
        backend = LiveSpiceBackend(args.schx, oversample=args.oversample, iterations=args.iterations)
        identity = Path(args.schx).read_bytes()
        cache_extra = f"os={args.oversample}|it={args.iterations}|maxv={args.peak_max_v}"
        return backend, knobs, identity, cache_extra
    if args.backend == "ngspice-deck":
        if not (args.pedal_dir and args.module):
            sys.exit("--backend ngspice-deck needs --pedal-dir and --module")
        sys.path.insert(0, os.path.abspath(args.pedal_dir))
        mod = importlib.import_module(args.module)
        knobs = [k for k in mod.KNOB_NAMES if k not in args.exclude_knob]
        backend = NgspiceBackend(mod.build_deck, probe_node=args.probe_node,
                                  maxstep=args.maxstep, parallel_sims=args.parallel_sims)
        identity = Path(mod.__file__).read_bytes()
        cache_extra = f"maxv={args.peak_max_v}"
        return backend, knobs, identity, cache_extra
    if args.backend == "ltspice-deck":
        if not (args.pedal_dir and args.module):
            sys.exit("--backend ltspice-deck needs --pedal-dir and --module")
        sys.path.insert(0, os.path.abspath(args.pedal_dir))
        mod = importlib.import_module(args.module)
        knobs = [k for k in mod.KNOB_NAMES if k not in args.exclude_knob]
        backend = LtspiceBackend(mod.build_deck, tap=args.probe_node,
                                 maxstep=args.maxstep, parallel_sims=args.parallel_sims,
                                 out_scale=args.out_scale, timeout=args.render_timeout)
        identity = Path(mod.__file__).read_bytes()
        cache_extra = f"maxv={args.peak_max_v}"
        return backend, knobs, identity, cache_extra
    sys.exit(f"unknown --backend {args.backend!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["livespice", "ngspice-deck", "ltspice-deck"])

    # livespice-only
    ap.add_argument("--schx", help="[livespice] path to .schx file")
    ap.add_argument("--knobs", default="", help="[livespice] comma-separated knobs to sweep")
    ap.add_argument("--fixed-params", default="", help="[livespice] e.g. 'Volume=1.0' — held during probing")
    ap.add_argument("--oversample", type=int, default=8, help="[livespice]")
    ap.add_argument("--iterations", type=int, default=256, help="[livespice]")
    ap.add_argument("--input-level-dbu", type=float, default=None,
                     help="[livespice] override the exported input_level_dbu (default: derived from V0dBFS)")

    # ngspice/ltspice-deck (shared -- both drive a hand-written gen_*_ngspice.py/gen_*_ltspice.py
    # module exposing build_deck/KNOB_NAMES)
    ap.add_argument("--pedal-dir", help="[ngspice-deck, ltspice-deck] directory containing --module, added to sys.path")
    ap.add_argument("--module", help="[ngspice-deck, ltspice-deck] module exposing build_deck/KNOB_NAMES, e.g. gen_ocd_ngspice")
    ap.add_argument("--probe-node", default="OUT", help="[ngspice-deck, ltspice-deck] node/tap to render and measure")
    ap.add_argument("--maxstep", type=float, default=3e-6, help="[ngspice-deck, ltspice-deck]")
    ap.add_argument("--parallel-sims", type=int, default=8, help="[ngspice-deck, ltspice-deck]")
    ap.add_argument("--render-timeout", type=float, default=None,
                    help="[ltspice-deck] per-render wall ceiling in seconds. Default: scales with clip duration AND --parallel-sims (see ltspice_spicelib.default_timeout), deliberately generous because a too-short ceiling does not error -- it reports every render as a convergence failure. On hardware slower than this was tuned on (older CPU, spinning disk, throttled or busy machine) set LTSPICE_TIMEOUT_SCALE=<multiplier> rather than passing a number here per run.")
    ap.add_argument("--exclude-knob", action="append", default=[],
                     help="[ngspice-deck, ltspice-deck] knob to exclude from the swept check (repeatable) -- "
                          "e.g. a 0/1 toggle, not a continuous sweep; build_deck's own default applies")
    ap.add_argument("--lead-silence-s", type=float, default=3.0,
                     help="[ngspice-deck] silence prepended before probe content -- see this repo's "
                          "README ('Known issue: excitation needs a silent lead-in'). 0 to disable. "
                          "Not used for ltspice-deck -- LTspice's own .ic/uic bias hints replace the "
                          "need for a cold-start settling lead-in, see ltspice_spicelib.py.")
    ap.add_argument("--out-scale", type=float, default=0.05,
                     help="[ltspice-deck] LTspice .wave output is +/-1V-PCM-bounded -- see "
                          "ltspice_spicelib.py's docstring")

    # shared
    ap.add_argument("--input", required=True, help="the sweep/clip generation will use")
    ap.add_argument("--vin", type=float, default=None,
                     help="probe drive level, volts -- rescales --input's peak to this. Omit to "
                          "use --input's own native peak unscaled (livespice's historical default; "
                          "ngspice decks conventionally pass an explicit value, e.g. 1.0)")
    ap.add_argument("--seconds", type=float, default=10.0, help="probe-input length")
    ap.add_argument("--esr-dead", type=float, default=1e-3)
    ap.add_argument("--keep-scratch", action="store_true",
                    help="keep this run's intermediate renders instead of deleting them on "
                         "exit (they go to ~/.cache/parametric-nam/preflight_scratch). For debugging "
                         "a bad render; off by default because these are write-only files "
                         "nothing reads back.")
    ap.add_argument("--dir-margin", type=float, default=0.05,
                     help="min relative metric change to call a direction (else inconclusive)")
    ap.add_argument("--eq-check-drive-level", type=float, default=0.1,
                     help="while checking an EQ/tone knob, hold drive/gain-classified OTHER "
                          "knobs at this value instead of 0.5 -- see module docstring's "
                          "LEVEL-AWARE EQ CHECKS note. Same value used for both backends.")
    ap.add_argument("--find-peak", action="store_true",
                     help="sweep a clean sine tone's input amplitude and report where output "
                          "RMS stops rising (the device's own saturation ceiling). Uses "
                          "--fixed-params (livespice) for any knob not swept; other knobs "
                          "default to 0.5 (or --eq-check-drive-level while testing an EQ knob).")
    ap.add_argument("--peak-max-v", type=float, default=40.0,
                     help="upper bound of the --find-peak amplitude sweep -- the 40V default "
                          "suits an amp; lower it (e.g. 3-5) for a small pedal circuit")
    ap.add_argument("--clean-probe-peak", type=float, default=None,
                     help="probe EQ/tone knobs with the input scaled to this peak voltage "
                          "instead of deriving one from --find-peak")
    ap.add_argument("--knob-kind", default=None,
                     help="NAME=kind,... override for classify()'s name-based guess (kind: "
                          "hi/lo/mid/drive/rms -- same as a config's [knob-kind] table, see "
                          "run_pipeline.py). Use this for a knob whose real-world name doesn't "
                          "match the keyword heuristic (an unrecognized name gets NEITHER a "
                          "direction check NOR --eq-check-drive-level's EQ-swamp protection, "
                          "silently -- the least scrutiny, not the most).")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-cache", action="store_true",
                     help="ignore (and overwrite) any cached --find-peak saturation sweep")
    args = ap.parse_args()

    knob_kind_override = {}
    for kv in filter(None, (s.strip() for s in (args.knob_kind or "").split(","))):
        k, v = kv.split("="); knob_kind_override[k.strip()] = v.strip()

    def classify(knob):
        if knob in knob_kind_override:
            kind = knob_kind_override[knob]
            return kind, f"{kind} (explicit --knob-kind)"
        return _classify_by_name(knob)

    backend, knobs, identity, cache_extra = _build_backend(args)
    fixed = {}
    for kv in filter(None, (s.strip() for s in args.fixed_params.split(","))):
        k, v = kv.split("="); fixed[k.strip()] = float(v)

    hard_fail, warn = [], []
    report = {"backend": args.backend, "knobs": {}, "input_calibration": {}}
    scratch = str(scratch_dir("preflight", args.keep_scratch))

    x, sr = sf.read(args.input, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    if sr != SR:
        sys.exit(f"input sample rate {sr} != {SR}")
    xprobe = x[: int(args.seconds * SR)]
    # TWO DIFFERENT LEVELS, and conflating them silently mis-exports the calibration.
    #   probe_peak  -- peak of the TRUNCATED probe slice; what the probe renders are driven at.
    #   native_peak -- peak of the WHOLE input; what the dataset will actually be rendered at,
    #                  and therefore the only correct reference for input_level_dbu.
    # --seconds takes a PREFIX, and an excitation built by build_excitation.py puts its
    # loud content LAST (real playing, then amplitude-stepped sweeps, then transient bursts).
    # Measured on the budget clone pedal's 38 s excitation: the 10 s prefix peaks at 3.940 V against the
    # file's 14.833 V -- 11.5 dB down -- so deriving the exported dBu from the probe slice
    # understated it by exactly that, while labelling it "training input peak".
    probe_peak = float(np.abs(xprobe).max()) + 1e-12
    native_peak = float(np.abs(x).max()) + 1e-12
    vin = args.vin if args.vin is not None else probe_peak
    calib_v = args.vin if args.vin is not None else native_peak

    print(f"Preflight ({args.backend}): {Path(args.schx).name if args.schx else args.module}  "
          f"knobs={knobs}  fixed={fixed or '-'}")
    print(f"  probe {args.seconds:.0f}s @ {vin:.3f}V"
          f"{f'  (+{args.lead_silence_s:.0f}s lead-silence, unscored)' if args.backend == 'ngspice' else ''}\n")

    lead_silence_s = args.lead_silence_s if args.backend == "ngspice-deck" else 0.0
    lead_n = int(lead_silence_s * SR)
    probe_raw = np.concatenate([np.zeros(lead_n, dtype=np.float32), xprobe]) if lead_n else xprobe
    base_handle = backend.prepare_input(probe_raw, SR, vin, scratch, "native")

    # ---- saturation sweep first: its curve sets a clean (linear-region) probe level ----
    sat = None
    if args.find_peak:
        base = {k: 0.5 for k in knobs}; base.update(fixed)
        cpath = findpeak_cache_key(identity, base, cache_extra)
        if cpath.exists() and not args.no_cache:
            sat = json.loads(cpath.read_text())
            print(f"  saturation sweep: CACHED (params={base})  [{cpath.name}]")
        else:
            print(f"  saturation sweep (params={base}):")
            def _sat_progress(done, total, elapsed):
                print(f"    {done}/{total} amplitude probes rendered ({elapsed:.0f}s)", flush=True)
            sat = find_saturation_point(backend, base, scratch, lead_silence_s=lead_silence_s,
                                         max_v=args.peak_max_v, progress=_sat_progress)
            if sat is not None:
                cpath.write_text(json.dumps(sat))
        if sat is None:
            warn.append("--find-peak: all sweep renders failed")
            print("    FAILED (all renders failed)")
        else:
            onset = sat["onset_99pct_input_v"]
            print(f"    ceiling = {sat['ceiling_rms']:.3f} V_rms (reached by input={sat['ceiling_at_input_v']:.3f} V)")
            print(f"    output reaches 99% of ceiling at input ~ {onset:.4g} V"
                  if onset is not None else "    99% onset not bracketed by the sweep")
            report["saturation"] = sat

    # ---- clean probe level for EQ/tone knobs ----
    clean_peak = args.clean_probe_peak
    if clean_peak is None and sat is not None and sat.get("curve"):
        lin_top = _linear_region_top(sat["curve"])
        if lin_top is not None:
            clean_peak = 0.5 * lin_top
    clean_handle = None
    if clean_peak is not None:
        clean_peak = min(clean_peak, vin)  # never AMPLIFY the probe
        if clean_peak < vin * 0.999:
            clean_handle = backend.prepare_input(probe_raw, SR, clean_peak, scratch, "clean")

    if clean_handle is not None:
        print(f"\n  EQ/tone knobs probed at CLEAN {clean_peak:.3f} V (linear) AND driven "
              f"{vin:.3f} V; drive/level knobs at {vin:.3f} V. Drive-classified knobs held at "
              f"{args.eq_check_drive_level} during any EQ knob's check.\n")
    else:
        print(f"\n  all knobs probed at native input level ({vin:.3f} V), drive-classified "
              f"knobs held at {args.eq_check_drive_level} during any EQ knob's check"
              f"{' — pass --find-peak or --clean-probe-peak for input-level-scaled EQ checks' if not sat else ''}.\n")

    def _base_for(target_knob, target_kind):
        b = {}
        for k in knobs:
            if k == target_knob:
                continue
            kk, _ = classify(k)
            b[k] = (args.eq_check_drive_level if target_kind in ("hi", "lo", "mid") and kk == "drive"
                    else 0.5)
        b.update(fixed)
        return b

    def _probe(knob, kind, handle, tag_prefix):
        base = _base_for(knob, kind)
        jobs = [{"params": {**base, knob: val}, "tag": f"{tag_prefix}_{knob}_{tg}"}
                for tg, val in (("lo", 0.2), ("mid", 0.5), ("hi", 0.8))]
        ys = backend.render_many(jobs, handle, scratch)
        out = {}
        for tg, j in zip(("lo", "mid", "hi"), jobs):
            y = ys.get(j["tag"])
            out[tg] = None if y is None else y[lead_n:]
        return out

    def _dir_stats(ys, kind):
        if any(v is None for v in ys.values()):
            return None
        d = {"esr": esr(ys["hi"], ys["lo"]), "spikes": max(spikes(ys[tg]) for tg in ys)}
        if kind is not None:
            m_lo, m_mid, m_hi = metric(ys["lo"], kind), metric(ys["mid"], kind), metric(ys["hi"], kind)
            d.update({"m_lo": m_lo, "m_mid": m_mid, "m_hi": m_hi,
                      "rel": (m_hi - m_lo) / (abs(m_lo) + 1e-12),
                      "mono": (m_lo <= m_mid <= m_hi) or (m_lo >= m_mid >= m_hi)})
        return d

    MARG = args.dir_margin
    n_knobs = len(knobs)
    for i, knob in enumerate(knobs, 1):
        # Each knob's own render(s) run concurrently (render_many's thread pool), so there's
        # nothing to report WITHIN one knob's probe -- but a stiff circuit's renders can each
        # take tens of seconds, and with no marker between knobs the previous knob's result
        # line (printed once its probe finishes) is the last thing on screen for the entire
        # duration of the NEXT knob's probe, indistinguishable from a hang partway through a
        # multi-knob device.
        print(f"  [{i}/{n_knobs}] probing {knob} ...", flush=True)
        kind, label = classify(knob)
        is_eq = kind in ("hi", "lo", "mid")
        rec = {"direction_metric": kind, "label": label}

        # === EQ/tone knob WITH a clean level: probe clean (primary) + driven (cross-check) ===
        if is_eq and clean_handle is not None:
            cl = _dir_stats(_probe(knob, kind, clean_handle, "clean"), kind)
            dr = _dir_stats(_probe(knob, kind, base_handle, "drv"), kind)
            if cl is None or dr is None:
                hard_fail.append(f"{knob}: RENDER FAILED"); rec["status"] = "render_fail"
                print(f"  {knob:10} RENDER FAILED"); report["knobs"][knob] = rec; continue
            sp = max(cl["spikes"], dr["spikes"])
            rc, rd = cl["rel"], dr["rel"]
            clean_ok, driven_ok = rc > MARG, rd > MARG
            clean_rev, driven_rev = rc < -MARG, rd < -MARG
            rec.update({"esr_hi_lo": cl["esr"], "spikes": sp, "probe_peak_clean_v": clean_peak,
                        "probe_peak_driven_v": vin, "metric_rel_change": rc, "metric_rel_change_driven": rd})
            if cl["esr"] < args.esr_dead and dr["esr"] < args.esr_dead:
                hard_fail.append(f"{knob}: DEAD (ESR clean {cl['esr']:.4g}, driven {dr['esr']:.4g})")
                rec["status"] = "dead"
                print(f"  {knob:10} DEAD           ESR clean={cl['esr']:.4g} driven={dr['esr']:.4g}  (expected {label})")
            elif clean_ok or driven_ok:
                rec["status"] = "ok"
                print(f"  {knob:10} OK             {label} clean {rc:+.1%} / driven {rd:+.1%}")
                if not (clean_ok and driven_ok):
                    warn.append(f"{knob}: LEVEL-DEPENDENT — {label} clean {rc:+.1%} @ {clean_peak:.3f}V but "
                                f"{rd:+.1%} driven @ {vin:.3f}V (EQ swamped by clipping at the hot level; "
                                f"direction is correct where the stage is linear)")
            elif clean_rev and driven_rev:
                hard_fail.append(f"{knob}: REVERSED at BOTH levels ({label} clean {rc:+.1%}, driven {rd:+.1%})")
                rec["status"] = "reversed"
                print(f"  {knob:10} REVERSED       {label} clean {rc:+.1%} / driven {rd:+.1%} (should RISE)")
            else:
                warn.append(f"{knob}: direction inconclusive ({label} clean {rc:+.1%}, driven {rd:+.1%})")
                rec["status"] = "dir_inconclusive"
                print(f"  {knob:10} INCONCLUSIVE   {label} clean {rc:+.1%} / driven {rd:+.1%}")
            if sp > 0: warn.append(f"{knob}: {sp} sample-spike(s) in probe renders")
            report["knobs"][knob] = rec; continue

        # === non-EQ knob, or EQ without a clean level: single native-level check ===
        d = _dir_stats(_probe(knob, kind, base_handle, "native"), kind)
        if d is None:
            hard_fail.append(f"{knob}: RENDER FAILED"); rec["status"] = "render_fail"
            print(f"  {knob:10} RENDER FAILED"); report["knobs"][knob] = rec; continue
        e, sp = d["esr"], d["spikes"]
        rec.update({"esr_hi_lo": e, "spikes": sp, "probe_peak_v": vin})
        if e < args.esr_dead:
            hard_fail.append(f"{knob}: DEAD (ESR={e:.4g})"); rec["status"] = "dead"
            print(f"  {knob:10} DEAD           ESR={e:.4g}  (expected to affect {label})")
        elif kind is None:
            warn.append(f"{knob}: responds but name unrecognized — direction NOT checked")
            rec["status"] = "responds_dir_unknown"
            print(f"  {knob:10} responds       ESR={e:.4g}  direction UNCHECKED (unknown knob '{knob}')")
        else:
            rel, mono = d["rel"], d["mono"]
            rec.update({"metric_lo": d["m_lo"], "metric_mid": d["m_mid"], "metric_hi": d["m_hi"],
                        "metric_rel_change": rel})
            if abs(rel) < MARG:
                warn.append(f"{knob}: responds (ESR={e:.4g}) but {label} metric barely moved ({rel:+.1%}) — direction inconclusive")
                rec["status"] = "dir_inconclusive"
                print(f"  {knob:10} responds       ESR={e:.4g}  {label} {rel:+.1%} INCONCLUSIVE")
            elif rel > 0:
                rec["status"] = "ok"
                tail = "" if mono else "  (non-monotonic!)"
                print(f"  {knob:10} OK             ESR={e:.4g}  {label} {rel:+.1%} rising{tail}")
                if not mono: warn.append(f"{knob}: direction OK but non-monotonic across 0.2/0.5/0.8")
            else:
                hint = " (try --find-peak/--clean-probe-peak — may be clipping-swamp at this level)" if is_eq else ""
                hard_fail.append(f"{knob}: REVERSED ({label} {rel:+.1%} — falls as knob rises){hint}")
                rec["status"] = "reversed"
                print(f"  {knob:10} REVERSED       ESR={e:.4g}  {label} {rel:+.1%} (should RISE)")
        if sp > 0:
            warn.append(f"{knob}: {sp} sample-spike(s) in probe renders")
        report["knobs"][knob] = rec

    # ---- input-level calibration ----
    print("\n  input calibration:")
    if args.backend == "livespice":
        v0 = _schx_input_v0dbfs(args.schx)
        if args.input_level_dbu is not None:
            exported_ild = float(args.input_level_dbu)
        elif v0 is not None:
            exported_ild = _input_level_dbu(v0)
        else:
            exported_ild = None
        ic = {"v0dbfs_volts": v0, "input_peak_dbfs": 20 * math.log10(native_peak + 1e-12),
              "exported_input_level_dbu": exported_ild}
        print(f"    training input peak = {native_peak:.3f} ({ic['input_peak_dbfs']:.1f} dBFS)")
        if v0 is not None:
            print(f"    schx V0dBFS = {v0} V")
        if exported_ild is None:
            warn.append("schx has no readable Input V0dBFS — input_level_dbu will be omitted from the .nam")
            print("    input_level_dbu to be EXPORTED = (omitted — no V0dBFS)")
        else:
            print(f"    input_level_dbu to be EXPORTED = {exported_ild:.2f} dBu")
            if not (-5.0 <= exported_ild <= 24.0):
                warn.append(f"input_level_dbu={exported_ild:.1f} dBu is off the plausible interface "
                            f"range — V0dBFS is likely off the 1V ecosystem convention (1V => ~-0.8 dBu). "
                            f"A calibrating host will mis-gain. Fix V0dBFS in the schx.")
                print("    ^ OFF-CONVENTION V0dBFS (expected ~1V => ~-0.8 dBu)")
        report["input_calibration"] = ic
    else:
        exported_ild = 20.0 * math.log10((calib_v / math.sqrt(2)) / _DBU_0_RMS_VOLTS)
        ic = {"vin_volts": vin, "calibration_volts": calib_v,
              "input_peak_dbfs": 20 * math.log10(native_peak + 1e-12),
              "probe_peak_volts": probe_peak,
              "exported_input_level_dbu": exported_ild}
        print(f"    training input peak = {native_peak:.3f} ({ic['input_peak_dbfs']:.1f} dBFS)"
              f"  <- whole file, what the dataset renders at")
        print(f"    probe vin (V0dBFS-equivalent) = {vin} V"
              f"{'' if abs(probe_peak - native_peak) < 1e-9 else f'  <- {args.seconds:.0f}s prefix only'}")
        print(f"    input_level_dbu to be EXPORTED = {exported_ild:.2f} dBu")
        if not (-5.0 <= exported_ild <= 24.0):
            warn.append(f"input_level_dbu={exported_ild:.1f} dBu is off the plausible interface "
                        f"range — vin is likely off the 1V ecosystem convention (1V => ~-0.8 dBu).")
            print(f"    ^ OFF-CONVENTION vin (expected ~1V => ~-0.8 dBu; this render used {vin}V)")
        report["input_calibration"] = ic

    # A truncated probe can be far quieter than the file it gates: --seconds takes a PREFIX,
    # and build_excitation.py puts its loud content LAST. Measured on the budget clone pedal's 38 s
    # excitation, the 10 s prefix peaks 11.5 dB below the file. Worth saying out loud, because
    # it cuts both ways -- cleaner direction readings, untested saturation.
    if probe_peak + 1e-9 < native_peak:
        _down_db = 20 * math.log10(native_peak / probe_peak)
        if _down_db >= 3.0:
            warn.append(f"probe slice is {_down_db:.1f} dB quieter than the full input "
                        f"({probe_peak:.3f} V vs {native_peak:.3f} V) -- --seconds takes a "
                        f"PREFIX and build_excitation.py puts its loud content last, so these "
                        f"knob checks never saw the excitation's loud region. Direction "
                        f"readings are CLEANER this way (clipping swamps EQ direction), but "
                        f"saturation behaviour there is untested -- that is "
                        f"check_transient_coverage.py's job.")

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
