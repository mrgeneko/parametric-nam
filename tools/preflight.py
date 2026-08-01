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

Level-aware EQ checks: a passive tone stack's direction is level-independent, but hard
clipping SWAMPS it -- on a clean amp driven near its ceiling, turning Treble up reads as
a FALSE reversal (the extra HF becomes clipping mush, not clean treble). So when a
saturation ceiling is available (--find-peak, or an explicit --clean-probe-peak), EQ/tone
knobs (treble/bass/mid) are probed at BOTH a LINEAR level (half the top of the flat-gain
region, which catches early preamp clipping AND gain-expansion, not just power-amp ceiling)
and the native/driven level: PASS if EITHER shows a clear correct direction, hard-FAIL
only if BOTH reverse, and WARN (not fail) when the two disagree (level-dependent). Drive/
level knobs are always probed at the native level -- they must see the full range including
saturation. Without a ceiling, all knobs use the native level (legacy behavior).
"""
import argparse, hashlib, json, math, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from batch_harness import LIVESPICE_CLI, parse_schx_controls, resolve_knobs  # noqa: E402
from param_train import _schx_input_v0dbfs, _input_level_dbu  # noqa: E402

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
    # An explicit volume/level name is a LEVEL control even when it also contains a tone word:
    # "BrightVol"/"Bright Volume" is a VOLUME with a bright cap, not a treble knob. Its relative
    # treble legitimately FALLS as it rises (the bright cap boosts most at low settings), which the
    # treble rule would misread as REVERSED. Check the unambiguous volume tokens first.
    if any(k in n for k in ("volume", "vol", "master")):
        return "rms", "output level"
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


def render_many(jobs, workers=None):
    """Run a list of independent render jobs concurrently. Each job is a dict with keys
    schx, params, oversample, iterations, in_wav, scratch, tag. Every render is a
    self-contained subprocess writing a unique-by-tag output file, so they parallelize
    cleanly across a thread pool. Returns {tag: y_or_None}. This turns preflight's ~30-50
    serial probe renders (~20-30 min on a full amp) into a handful of pool batches."""
    if not jobs:
        return {}
    workers = max(1, min(workers or os.cpu_count() or 4, len(jobs)))
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(render, j["schx"], j["params"], j["oversample"], j["iterations"],
                          j["in_wav"], j["scratch"], j["tag"]): j["tag"] for j in jobs}
        for f in futs:
            out[futs[f]] = f.result()
    return out


def spikes(y):
    a = np.abs(y); p99 = np.percentile(a, 99)
    ln = np.maximum(np.abs(np.roll(y, 1)), np.abs(np.roll(y, -1)))
    return int(((a > 3 * ln) & (a > 2 * p99)).sum())


def _linear_region_top(curve, tol=0.05):
    """Top of the constant-gain (linear) input region, from a find-peak curve
    [(in_v, out_rms), ...] ascending. Returns the largest input whose small-signal
    gain (out/in) still matches the lowest-level gain within `tol`. Unlike the
    saturation onset (power-amp RMS ceiling), this catches EARLY preamp nonlinearity
    AND gain-expansion regions -- the level below which the tone stack sees an
    undistorted signal, so its EQ direction is measured cleanly. Returns None if the
    curve is too short."""
    pts = [(a, r) for a, r in curve if a > 0 and r > 0]
    if len(pts) < 2:
        return None
    g0 = pts[0][1] / pts[0][0]          # small-signal gain (lowest input)
    top = pts[0][0]
    for a, r in pts:
        if abs((r / a) / g0 - 1.0) > tol:
            break
        top = a
    return top


def _loglog_interp(x1, y1, x2, y2, ytarget):
    lx1, ly1 = math.log(x1), math.log(y1)
    lx2, ly2 = math.log(x2), math.log(y2)
    frac = (math.log(ytarget) - ly1) / (ly2 - ly1)
    return math.exp(lx1 + frac * (lx2 - lx1))


def find_saturation_point(schx, params, oversample, iterations, scratch,
                           freq=200.0, dur=1.0, start_v=0.005, max_v=40.0, npoints=20,
                           plateau_frac=0.005, plateau_run=3, workers=None):
    """Sweep a clean sine tone's input amplitude and find where output RMS stops
    rising with more input -- the device's own saturation ceiling, matching the
    methodology in docs/input_calibration.md (the OCD investigation). Not the same
    as clipping-onset (THD-based); this is "where does volume stop increasing."

    All `npoints` amplitudes are rendered CONCURRENTLY (they're independent 1s tones),
    which is faster across a thread pool than the old sequential-with-early-stop scan
    and makes the plateau_frac/plateau_run params vestigial. Returns None if every
    render fails.
    """
    t = np.arange(int(SR * dur)) / SR
    log_amps = np.geomspace(start_v, max_v, npoints)
    jobs = []
    for i, a in enumerate(log_amps):
        x = (a * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        in_wav = f"{scratch}/peak_in_{i}.wav"
        sf.write(in_wav, x, SR)
        jobs.append({"schx": schx, "params": params, "oversample": oversample,
                     "iterations": iterations, "in_wav": in_wav, "scratch": scratch,
                     "tag": f"peak_{i}"})
    ys = render_many(jobs, workers=workers)
    curve = []
    for i, a in enumerate(log_amps):
        y = ys.get(f"peak_{i}")
        if y is None:
            continue
        steady = y[int(SR * 0.5):]
        if len(steady) == 0:
            continue
        curve.append((float(a), float(np.sqrt((steady ** 2).mean()))))
    if not curve:
        return None
    ceiling_a, ceiling = max(curve, key=lambda p: p[1])
    target = 0.99 * ceiling
    onset = None
    for i in range(1, len(curve)):
        a0, r0 = curve[i - 1]; a1, r1 = curve[i]
        if r0 < target <= r1:
            onset = _loglog_interp(a0, r0, a1, r1, target)
            break
    return {"ceiling_rms": ceiling, "ceiling_at_input_v": ceiling_a,
            "onset_99pct_input_v": onset, "curve": curve}


def _findpeak_cache_path(schx, base, oversample, iterations, max_v):
    """Stable cache location for a find-peak result, keyed on the schx CONTENT (so an
    edited circuit re-sweeps automatically) plus the sweep-defining parameters."""
    h = hashlib.sha256()
    h.update(Path(schx).read_bytes())
    h.update(repr(sorted((str(k), str(v)) for k, v in base.items())).encode())
    h.update(f"os={oversample}|it={iterations}|maxv={max_v}".encode())
    d = Path.home() / ".cache" / "parametric-nam" / "findpeak"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h.hexdigest()[:16]}.json"


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
                    help="override the exported input_level_dbu for the check "
                         "(default: derived from the schx V0dBFS)")
    ap.add_argument("--find-peak", action="store_true",
                    help="sweep a clean sine tone's input amplitude and report where output "
                         "RMS stops rising (the device's own saturation ceiling) -- see "
                         "docs/input_calibration.md. Uses --fixed-params for any knob not "
                         "swept; other knobs default to 0.5.")
    ap.add_argument("--peak-max-v", type=float, default=40.0,
                    help="upper bound of the --find-peak amplitude sweep")
    ap.add_argument("--clean-probe-peak", type=float, default=None,
                    help="probe EQ/tone knobs (treble/bass/mid) with the input scaled to this peak "
                         "voltage — a level in the device's LINEAR region, so the tone-stack shaping "
                         "isn't swamped by clipping harmonics (which can read as a false reversal on "
                         "clean amps driven hot). Drive/level knobs stay at the native input level. "
                         "If omitted, this is auto-derived from --find-peak's saturation onset; if "
                         "neither is given, all knobs are probed at the native level (legacy behavior).")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore (and overwrite) any cached --find-peak saturation sweep for this schx")
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
        xprobe = x[: int(args.seconds * SR)]
        native_probe = f"{scratch}/probe_native.wav"
        sf.write(native_probe, xprobe, SR)
        native_peak = float(np.abs(xprobe).max()) + 1e-12

        print(f"Preflight: {Path(args.schx).name}  knobs={knobs}  fixed={args.fixed_params or '-'}")
        print(f"  probe {args.seconds:.0f}s @ os={args.oversample} it={args.iterations}\n")

        # ---- saturation sweep FIRST: its ceiling sets a clean (linear-region) probe level ----
        sat = None
        if args.find_peak:
            base = {k: 0.5 for k in knobs}; base.update(fixed)
            cpath = _findpeak_cache_path(args.schx, base, args.oversample, args.iterations, args.peak_max_v)
            if cpath.exists() and not args.no_cache:
                sat = json.loads(cpath.read_text())
                print(f"  saturation sweep: CACHED (params={base})  [{cpath.name}]")
            else:
                print(f"  saturation sweep (params={base}):")
                sat = find_saturation_point(args.schx, base, args.oversample, args.iterations,
                                             scratch, max_v=args.peak_max_v)
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

        # ---- clean probe level for EQ/tone knobs (treble/bass/mid) ----
        # A passive tone stack's DIRECTION is level-independent, but hard clipping SWAMPS it: on a
        # clean amp driven near its ceiling, turning Treble up can read as a FALSE reversal because
        # the extra HF just makes more clipping mush, not more clean treble (seen on the Deluxe
        # Reverb: reversed at 0.9V, correct at 0.3V). So probe EQ knobs in the LINEAR region. Drive/
        # level knobs are NOT scaled -- they must see the full range including saturation.
        # Conservative: only deviate from native when we have a level to scale to (explicit
        # --clean-probe-peak, or 0.3x the --find-peak onset); else legacy native-level behavior.
        clean_peak = args.clean_probe_peak
        if clean_peak is None and sat is not None and sat.get("curve"):
            lin_top = _linear_region_top(sat["curve"])        # top of the flat-gain (linear) region
            if lin_top is not None:
                clean_peak = 0.5 * lin_top                     # comfortably inside the linear region
        clean_probe = None
        if clean_peak is not None:
            clean_peak = min(clean_peak, native_peak)         # never AMPLIFY the sweep
            if clean_peak < native_peak * 0.999:
                sf.write(f"{scratch}/probe_clean.wav",
                         (xprobe * (clean_peak / native_peak)).astype(np.float32), SR)
                clean_probe = f"{scratch}/probe_clean.wav"
        if clean_probe is not None:
            print(f"\n  EQ/tone knobs probed at CLEAN {clean_peak:.3f} V (linear) AND driven "
                  f"{native_peak:.3f} V; drive/level knobs at {native_peak:.3f} V.\n")
        else:
            print(f"\n  all knobs probed at native input level ({native_peak:.3f} V)"
                  f"{' — pass --find-peak or --clean-probe-peak for level-aware EQ checks' if not sat else ''}.\n")

        LEVELS = (("lo", 0.2), ("mid", 0.5), ("hi", 0.8))

        def _knob_jobs(knob, probe_wav, tagbase):
            jobs = []
            for suf, val in LEVELS:
                p = {k: 0.5 for k in knobs}; p.update(fixed); p[knob] = val
                jobs.append({"schx": args.schx, "params": p, "oversample": args.oversample,
                             "iterations": args.iterations, "in_wav": probe_wav,
                             "scratch": scratch, "tag": f"{tagbase}_{suf}"})
            return jobs

        def probe_dir_from(ys, tagbase, kind):
            """3-point (0.2/0.5/0.8) probe read from pre-rendered results {tag: y}."""
            lo, mid, hi = (ys.get(f"{tagbase}_{s}") for s, _ in LEVELS)
            if lo is None or mid is None or hi is None:
                return None
            d = {"esr": esr(hi, lo),
                 "spikes": max(spikes(lo), spikes(mid), spikes(hi)),
                 "peak": max(float(np.abs(v).max()) for v in (lo, mid, hi))}
            if kind is not None:
                ml, mm, mh = metric(lo, kind), metric(mid, kind), metric(hi, kind)
                d.update({"m_lo": ml, "m_mid": mm, "m_hi": mh,
                          "rel": (mh - ml) / (abs(ml) + 1e-12),
                          "mono": (ml <= mm <= mh) or (ml >= mm >= mh)})
            return d

        # Build EVERY probe render up front and run them across a thread pool -- the knob
        # probes are all independent, so this is the big serial->parallel win (#1).
        all_jobs = []
        for knob in knobs:
            kind, _ = classify(knob)
            if kind in ("hi", "lo", "mid") and clean_probe is not None:
                all_jobs += _knob_jobs(knob, clean_probe, f"{knob}_clean")
                all_jobs += _knob_jobs(knob, native_probe, f"{knob}_drv")
            else:
                all_jobs += _knob_jobs(knob, native_probe, knob)
        nworkers = max(1, min(os.cpu_count() or 4, len(all_jobs)))
        print(f"  probing {len(knobs)} knob(s) via {len(all_jobs)} renders across {nworkers} workers...\n")
        ys_all = render_many(all_jobs, workers=nworkers)

        MARG = args.dir_margin
        for knob in knobs:
            kind, label = classify(knob)
            is_eq = kind in ("hi", "lo", "mid")
            rec = {"direction_metric": kind, "label": label}

            # === EQ/tone knob WITH a clean level: probe clean (primary) + driven (cross-check) ===
            if is_eq and clean_probe is not None:
                cl = probe_dir_from(ys_all, f"{knob}_clean", kind)
                dr = probe_dir_from(ys_all, f"{knob}_drv", kind)
                if cl is None or dr is None:
                    hard_fail.append(f"{knob}: RENDER FAILED"); rec["status"] = "render_fail"
                    print(f"  {knob:10} RENDER FAILED"); report["knobs"][knob] = rec; continue
                sp = max(cl["spikes"], dr["spikes"])
                rec.update({"esr_hi_lo": cl["esr"], "spikes": sp, "peak": max(cl["peak"], dr["peak"]),
                            "probe_peak_clean_v": clean_peak, "probe_peak_driven_v": native_peak,
                            "metric_lo": cl["m_lo"], "metric_mid": cl["m_mid"], "metric_hi": cl["m_hi"],
                            "metric_rel_change": cl["rel"], "metric_rel_change_driven": dr["rel"]})
                rc, rd = cl["rel"], dr["rel"]
                clean_ok, driven_ok = rc > MARG, rd > MARG
                clean_rev, driven_rev = rc < -MARG, rd < -MARG
                if cl["esr"] < args.esr_dead and dr["esr"] < args.esr_dead:
                    hard_fail.append(f"{knob}: DEAD (ESR clean {cl['esr']:.4g}, driven {dr['esr']:.4g})")
                    rec["status"] = "dead"
                    print(f"  {knob:10} DEAD           ESR clean={cl['esr']:.4g} driven={dr['esr']:.4g}  (expected {label})")
                elif clean_ok or driven_ok:
                    rec["status"] = "ok"
                    print(f"  {knob:10} OK             {label} clean {rc:+.1%} / driven {rd:+.1%}")
                    if not (clean_ok and driven_ok):
                        warn.append(f"{knob}: LEVEL-DEPENDENT — {label} clean {rc:+.1%} @ {clean_peak:.3f}V but "
                                    f"{rd:+.1%} driven @ {native_peak:.3f}V (EQ swamped by clipping at the hot level; "
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

            # === non-EQ knob, or EQ without a clean level: single native-level check (legacy) ===
            d = probe_dir_from(ys_all, knob, kind)
            if d is None:
                hard_fail.append(f"{knob}: RENDER FAILED"); rec["status"] = "render_fail"
                print(f"  {knob:10} RENDER FAILED"); report["knobs"][knob] = rec; continue
            e, sp = d["esr"], d["spikes"]
            rec.update({"esr_hi_lo": e, "spikes": sp, "peak": d["peak"], "probe_peak_v": native_peak})
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
                    # EQ knob reversed but we had no clean level to disambiguate -> could be
                    # clipping-swamp. Hard-fail (legacy/conservative) but point at the level fix.
                    hint = " (try --find-peak/--clean-probe-peak — may be clipping-swamp at this level)" if is_eq else ""
                    hard_fail.append(f"{knob}: REVERSED ({label} {rel:+.1%} — falls as knob rises){hint}")
                    rec["status"] = "reversed"
                    print(f"  {knob:10} REVERSED       ESR={e:.4g}  {label} {rel:+.1%} (should RISE)")
            if sp > 0:
                warn.append(f"{knob}: {sp} sample-spike(s) in probe renders")
            report["knobs"][knob] = rec

        # ---- input-level calibration ----
        # What export_nam will actually WRITE: config override, else derived from V0dBFS.
        v0 = _schx_input_v0dbfs(args.schx)
        if args.input_level_dbu is not None:
            exported_ild = float(args.input_level_dbu)
        elif v0 is not None:
            exported_ild = _input_level_dbu(v0)
        else:
            exported_ild = None
        in_peak = float(np.abs(x).max())
        ic = {"v0dbfs_volts": v0, "input_peak_dbfs": 20 * math.log10(in_peak + 1e-12),
              "exported_input_level_dbu": exported_ild}
        print("\n  input calibration:")
        print(f"    training input peak = {in_peak:.3f} ({ic['input_peak_dbfs']:.1f} dBFS)")
        if v0 is not None:
            print(f"    schx V0dBFS = {v0} V")
        if exported_ild is None:
            warn.append("schx has no readable Input V0dBFS — input_level_dbu will be omitted from the .nam")
            print("    input_level_dbu to be EXPORTED = (omitted — no V0dBFS)")
        else:
            print(f"    input_level_dbu to be EXPORTED = {exported_ild:.2f} dBu")
            # Ecosystem convention is V0dBFS=1V => ~-0.8 dBu. A value far outside a plausible
            # interface reference (~-5..+24 dBu) means V0dBFS is off-convention (e.g. the old
            # 0.1V pedals => -20.8, ~13-33 dB hot in a calibrating host).
            if not (-5.0 <= exported_ild <= 24.0):
                warn.append(f"input_level_dbu={exported_ild:.1f} dBu is off the plausible interface "
                            f"range — V0dBFS is likely off the 1V ecosystem convention (1V => ~-0.8 dBu). "
                            f"A calibrating host will mis-gain. Fix V0dBFS in the schx.")
                print(f"    ^ OFF-CONVENTION V0dBFS (expected ~1V => ~-0.8 dBu)")
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
