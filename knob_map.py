#!/usr/bin/env python3
"""Multi-baseline knob direction map (used by preflight.py; backend-agnostic).

WHY. A single-baseline direction check (every other knob at 0.5) assumes a knob's effect does not depend on the others. It
does, on any circuit with two signal paths or a loaded tone stack, and the single check then mis-reads in both directions:
  * a correct knob fails (a dual-channel amp whose Bass cuts low end when the channel that bypasses the tone stack is the
    louder one: reversed at 0.5/0.5, correct everywhere the stack's own channel dominates), and
  * a broken knob passes (a Bass whose whole effect sits inside knob 0..0.1 reads "OK" between any two knob positions
    that straddle the first tenth of the knob, and "dead" between 0.2 and 0.8).
It also mis-reads when the probe clips: raising a gain knob at a fixed input pushes a later stage into clipping and the
fundamentals the check measures fall.

WHAT IT DOES. For every knob with a recognised role (hi/lo/mid/drive/rms):
  1. picks several BASELINES for the other knobs: all 0.5, plus N fixed pseudo-random corners (each knob 0.2 or 0.8);
  2. at each baseline, sets the input so the output sits near 50 mV with the tested knob at 0.8, and checks the circuit is
     still linear there (output halves when the input halves; the input did not hit its 1 V cap). A baseline that fails
     is SKIPPED, not scored: the check measures direction, and clipping swamps direction;
  3. renders a multi-tone probe with the knob at 0.2 and 0.8 and reads the level of the band the knob should move
     (lo: 60-250 Hz, mid: 400-1600 Hz, hi: 3-8 kHz, rms/drive: all of them), in dB;
  4. verdict over the usable baselines (a change over 1 dB counts as moving):
       OK           moves the right way somewhere, never the wrong way
       INTERACTION  right way at some baselines, wrong way at others -- a property of the circuit; reported as a WARNING
                    with the per-baseline map
       REVERSED     wrong way at every baseline that moves   (hard failure)
       DEAD         does not move at any usable baseline     (hard failure; for drive knobs only a warning, since a
                    clipper's level is flat by design)
       UNTESTED     no usable baseline
  5. TRAVEL (tone knobs only): at the first usable baseline, renders the knob at 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1 and
     reports it when more than 70 percent of the total change (at least 3 dB) falls inside the first or last tenth of the
     knob. That is a mirrored or wrong-shaped taper: the Bass of two different amps' tone stacks both had their whole
     range inside 0..0.1, which no direction check sees.
"""
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SR = 48000
BANDS = {"lo": [60, 100, 150, 250], "mid": [400, 700, 1000, 1600], "hi": [3000, 5000, 8000]}
ALL_TONES = sum(BANDS.values(), [])
DUR, DROP = 3.0, 1.0           # probe length, seconds ignored at the start
TARGET_RMS = 0.05              # output level the probe is set to (volts rms)
MOVE_DB = 1.0
TRAVEL_X = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
TRAVEL_MIN_DB, TRAVEL_FRAC = 3.0, 0.7
EQ_KINDS = ("hi", "lo", "mid")


def multitone():
    t = np.arange(int(SR * DUR)) / SR
    x = sum(np.sin(2 * np.pi * f * t + 1.3 * i) for i, f in enumerate(ALL_TONES))
    return (x / len(ALL_TONES) ** 0.5).astype(np.float32)


def baselines(names, n_random=6):
    """All-0.5, plus up to `n_random` DISTINCT corners (each knob 0.2 or 0.8), drawn with a fixed seed."""
    import itertools
    corners = [dict(zip(names, v)) for v in itertools.product((0.2, 0.8), repeat=len(names))]
    random.Random(7).shuffle(corners)
    return [{n: 0.5 for n in names}] + corners[:n_random]


def _tone_levels(y):
    y = np.asarray(y, dtype=np.float64)[int(SR * DROP):]
    k = np.arange(len(y)) / SR
    return {f: abs(np.sum(y * np.exp(-2j * np.pi * f * k))) / len(y) * 2 for f in ALL_TONES}


def _band_db(levels, kind):
    names = [kind] if kind in BANDS else list(BANDS)
    fs = sum((BANDS[b] for b in names), [])
    return 10 * np.log10(sum(levels[f] ** 2 for f in fs) + 1e-30)


def _rms(y):
    return float(np.std(np.asarray(y, dtype=np.float64)[int(SR * DROP):]))   # AC only: a preamp's output sits on its plate voltage


def direction_map(backend, knobs, classify, fixed, scratch, lead_n=0, workers=4, n_random=6, progress=None):
    """Returns {knob: record}. `backend` is a render_backends backend (prepare_input / render_many); `classify(knob)` gives
    (kind, label); `fixed` are parameters held constant (not tested, not varied); `lead_n` is the lead silence in samples
    (prepended to the probe and dropped from the output, as preflight does)."""
    sig = np.concatenate([np.zeros(lead_n, dtype=np.float32), multitone()]) if lead_n else multitone()
    varied = [k for k in knobs if k not in fixed]
    bases = baselines(varied, n_random)
    counter = [0]

    def render(params, amp, tag):
        counter[0] += 1
        t = f"km{counter[0]}_{tag}"
        h = backend.prepare_input(sig, SR, amp, scratch, t)
        y = backend.render_many([{"params": {**fixed, **params}, "tag": t}], h, scratch).get(t)
        return None if y is None else np.asarray(y)[lead_n:]

    def calibrate(base, knob, tag):
        amp = 1e-3
        p = dict(base); p[knob] = 0.8
        for _ in range(2):
            y = render(p, amp, tag + "c")
            if y is None:
                return None
            amp = min(max(amp * TARGET_RMS / max(_rms(y), 1e-12), 1e-6), 1.0)
        y1, y2 = render(p, amp, tag + "l1"), render(p, amp / 2, tag + "l2")
        if y1 is None or y2 is None:
            return None
        if amp >= 0.99 or not (1.8 < _rms(y1) / max(_rms(y2), 1e-12) < 2.2):
            return "NL"
        return amp

    def one(job):
        knob, kind, bi, base = job
        tag = f"{abs(hash((knob, bi))) % 10**8}"
        amp = calibrate(base, knob, tag)
        if amp is None:
            return job, None, None
        if amp == "NL":
            return job, "NL", None
        lo, hi = dict(base), dict(base)
        lo[knob], hi[knob] = 0.2, 0.8
        a, b = render(lo, amp, tag + "a"), render(hi, amp, tag + "b")
        if a is None or b is None:
            return job, None, None
        return job, _band_db(_tone_levels(b), kind) - _band_db(_tone_levels(a), kind), amp

    jobs = [(k, classify(k)[0], bi, b) for k in varied if classify(k)[0] for bi, b in enumerate(bases)]
    with ThreadPoolExecutor(max(1, workers)) as ex:
        results = list(ex.map(one, jobs))

    out = {}
    for knob in varied:
        kind, label = classify(knob)
        if not kind:
            continue
        rows = [(bi, r, amp) for (k, kd, bi, base), r, amp in results if k == knob]
        usable = [(bi, r, amp) for bi, r, amp in rows if isinstance(r, float)]
        skipped = sum(1 for _, r, _ in rows if r == "NL")
        failed = sum(1 for _, r, _ in rows if r is None)
        ok = sum(1 for _, r, _ in usable if r > MOVE_DB)
        rev = sum(1 for _, r, _ in usable if r < -MOVE_DB)
        if not usable:
            verdict = "UNTESTED"
        elif ok == 0 and rev == 0:
            verdict = "DEAD"
        elif ok == 0:
            verdict = "REVERSED"
        elif rev == 0:
            verdict = "OK"
        else:
            verdict = "INTERACTION"
        rec = {"kind": kind, "label": label, "verdict": verdict, "ok": ok, "reversed": rev,
               "flat": len(usable) - ok - rev, "usable": len(usable), "skipped_nonlinear": skipped, "render_failed": failed,
               "per_baseline": [{"baseline": bases[bi], "delta_db": round(r, 2)} for bi, r, _ in usable]}
        if kind in EQ_KINDS and usable and verdict in ("OK", "INTERACTION"):
            bi, _, amp = next(((bi, r, a) for bi, r, a in usable if bi == 0), usable[0])
            xs = list(TRAVEL_X)
            ys = [render({**bases[bi], knob: x}, amp, f"{abs(hash((knob, 't', x))) % 10**8}") for x in xs]
            if all(y is not None for y in ys):
                m = [_band_db(_tone_levels(y), kind) for y in ys]
                total = m[-1] - m[0]
                rec["travel_db"] = [round(v, 1) for v in m]
                if abs(total) >= TRAVEL_MIN_DB:
                    lo_frac, hi_frac = (m[1] - m[0]) / total, (m[-1] - m[-2]) / total
                    rec["travel_first_tenth"], rec["travel_last_tenth"] = round(lo_frac, 2), round(hi_frac, 2)
                    rec["travel_concentrated"] = bool(lo_frac > TRAVEL_FRAC or hi_frac > TRAVEL_FRAC)
        out[knob] = rec
        if progress:
            progress(knob, rec)
    return out
