#!/usr/bin/env python3
"""Backend-agnostic saturation-onset finder, shared by preflight.py (--find-peak) and
prepare_excitation.py, for ANY backend implementing render_backends.py's two-method contract
(LiveSpiceBackend, NgspiceBackend, or a future LTspiceBackend).

Sweeps a clean sine tone's amplitude (log-spaced, default 0.005V-40V, 20 points) through a
circuit at fixed knob params, and finds where output RMS stops rising -- the device's own
saturation ceiling. Ceiling = max steady-state RMS (dropping the attack transient and, for
backends that need it, a leading-silence settle region); onset = the input level where RMS
first reaches 99% of that ceiling, log-log interpolated between the two curve points
straddling the target (not just the nearest sample). Deliberately NOT a clipping-onset/THD-
based measure -- it answers "where does output stop getting louder with more input."

LEADING SILENCE (`lead_silence_s`, default 0.0): a circuit with a slow-charging DC-blocking
network can show a multi-second "powering on" transient before settling to its true steady
response, since a render always starts from a cold (all-capacitors-at-0V) state -- see this
repo's README ("Known issue: excitation needs a silent lead-in"). Confirmed directly on
the MOSFET-clipping pedal's ngspice deck: a sustained tone with no lead-in showed RMS drifting for ~15s
then jumping abruptly at ~16s -- a false, unstable reading entirely from measuring
mid-transient. LiveSPICE circuits haven't shown this so far (default stays 0.0 there); set it
explicitly for a backend/circuit combination where it matters.

Usage:
    from render_backends import NgspiceBackend  # or LiveSpiceBackend
    from find_saturation_point import find_saturation_point
    backend = NgspiceBackend(build_deck, probe_node='OUT')
    result = find_saturation_point(backend, {'Gain': 0.1, 'Tone': 0.5}, tmp='/tmp/fp',
                                    lead_silence_s=3.0, max_v=5.0)
    # {'ceiling_rms':..., 'ceiling_at_input_v':..., 'onset_99pct_input_v':..., 'curve':[(v,rms),...]}
"""
import atexit
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import itertools
import numpy as np

SR = 48000


ONSET_METHOD = "knee+sat95-v1"   # in the findpeak cache key: bump to invalidate every cached onset


def _loglog_interp(x1, y1, x2, y2, ytarget):
    lx1, ly1 = math.log(x1), math.log(y1)
    lx2, ly2 = math.log(x2), math.log(y2)
    frac = (math.log(ytarget) - ly1) / (ly2 - ly1)
    return math.exp(lx1 + frac * (lx2 - lx1))


def find_saturation_point(backend, params, tmp, freq=200.0, dur=2.0, lead_silence_s=0.0,
                           start_v=0.005, max_v=40.0, npoints=20, sr=SR, workers=8,
                           progress=None, max_extend_decades=4, min_start_v=1e-9,
                           capture=None):
    """Sweep a clean `freq` Hz tone's amplitude (log-spaced, `start_v`..`max_v`, `npoints`
    points) through `backend` at fixed `params`, and find where output RMS stops rising.

    Each amplitude needs its own input (a different level), so this calls
    `backend.prepare_input` once per amplitude and `backend.render_many` once per amplitude
    too (a single-job "batch") -- parallelized across amplitudes ourselves via a thread pool,
    since the backend's own render_many parallelism is designed for many knob settings against
    ONE shared input, the opposite axis from what a saturation sweep needs.

    `progress`, if given, is called as `progress(done, total, elapsed_s)` as each amplitude's
    render completes (submission order, not completion order, doesn't matter here -- there's
    no per-amplitude output to preserve order for). Default None (silent) since some callers
    (prepare_excitation.py, check_transient_coverage.py) call this once per grid corner and a
    per-amplitude print at every corner would be far noisier than useful; a one-shot caller
    like preflight.py --find-peak should pass one, since this sweep (up to `npoints` renders,
    each a real backend render) used to run with NO output at all -- indistinguishable from a
    hang for a stiff circuit's renders taking tens of seconds each.

    `capture`, if given, is the virtual capture chain's kwargs (capture_chain.py) -- the
    measurement is then taken through the same audio-interface input stage the dataset is
    rendered through. Whoever passes it MUST also put capture_chain.cache_tag(capture) in
    their findpeak cache_extra, or a raw-measured onset gets served to a chained caller.

    Returns None if every amplitude fails to converge.
    """
    Path(tmp).mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(sr * lead_silence_s), dtype=np.float32)
    t_tone = np.arange(int(sr * dur)) / sr
    tone = np.sin(2 * np.pi * freq * t_tone).astype(np.float32)
    raw = np.concatenate([silence, tone])
    tone_start = len(silence)

    t0 = time.monotonic()
    seq = itertools.count()

    def _sweep(lo, hi, n):
        """Render `n` log-spaced amplitudes in [lo, hi] and return [(in_v, out_rms), ...]."""
        def _one(i_amp):
            i, amp = i_amp
            tag = f"fp_{i}"
            handle = backend.prepare_input(raw, sr, float(amp), tmp, tag)
            ys = backend.render_many([{"params": params, "tag": tag}], handle, tmp)
            return float(amp), ys.get(tag)

        amps = list(np.geomspace(lo, hi, n))
        with ThreadPoolExecutor(max_workers=min(workers, len(amps))) as ex:
            futures = [ex.submit(_one, (next(seq), a)) for a in amps]
            raw_results = []
            for fut in as_completed(futures):
                raw_results.append(fut.result())
                if progress is not None:
                    progress(len(raw_results), len(futures), time.monotonic() - t0)

        out = []
        for amp, y in raw_results:
            if y is None:
                continue
            if capture:
                # Measure what the MODEL will be trained on, not the raw node. Saturation is
                # found by watching output RMS stop rising; if a large share of that RMS is
                # sub-audio bias wander this measures the wander, not the saturation. Duke of
                # Tone (Distortion) 2026-09-10: 64% of its output energy was below 19 Hz.
                # Filter the WHOLE render before slicing, so the filter's own startup transient
                # lands in the discarded region rather than inside the steady-state window.
                from capture_chain import capture_chain as _cc
                y = _cc(np.asarray(y, dtype=np.float64), sr, **capture)
            steady = y[tone_start + int(sr * dur * 0.5):]
            if len(steady) == 0:
                continue
            out.append((amp, float(np.sqrt((steady ** 2).mean()))))
        return out

    curve = _sweep(start_v, max_v, npoints)
    curve.sort()
    if not curve:
        return None

    # The sweep can START ABOVE the onset. For a high-gain circuit the whole
    # `start_v`..`max_v` range sits on the saturated plateau, so output RMS is flat, the
    # `r0 < target <= r1` crossing below never fires, and this returns onset=None -- which
    # reads as "never saturates, raise --peak-max-v" when the truth is the exact opposite
    # and raising the ceiling only adds more plateau. Measured on Mesa Dual Rectifier Ch1
    # (2026-09-10): 0.005-400 V was FLAT at 11.2 V out (80000x input range, 0.8% output
    # change) because its real onset is 2.08 mV, below start_v. So when the lowest point is
    # already at the plateau, extend DOWNWARD a decade at a time until it isn't.
    for _ in range(max_extend_decades):
        # TRIGGER IS THE SAME TEST THAT DEFINES ONSET (_linear_region_top), deliberately.
        # It used to be "lowest point is within 1% of the ceiling", while onset was DEFINED as
        # the 99%-of-ceiling crossing -- two different questions sharing one threshold, and a
        # soft-compressing circuit defeats both at once. Mesa Orange (sag v30), 2026-09-12: at
        # OR Gain=0.95 the 5 mV floor already sat at 85% of ceiling with gain falling
        # monotonically from the first point (585 -> 70 out/in over six points), so it was
        # plainly past its knee -- but 0.85 < 0.99 read as "off the plateau" and the sweep
        # never extended down. Asking instead "is the knee still AT the floor?" makes the
        # trigger and the definition agree by construction.
        if _linear_region_top(curve) != curve[0][0]:
            break  # a linear region is resolved inside the sweep -- the knee is in range
        lo = curve[0][0]
        if lo <= min_start_v:
            break
        new_lo = max(lo / 100.0, min_start_v)
        # Printed UNCONDITIONALLY, not gated on `progress`: prepare_excitation.py (the main
        # caller, one sweep per grid corner) passes no progress callback, so gating this on it
        # made the extension silent in exactly the pipeline that depends on it -- Ch1's 2026-09-10
        # run extended down on real corners and reported nothing. It fires rarely (only when the
        # floor is on the plateau), so it cannot flood even a 200-corner run.
        g0 = curve[0][1] / curve[0][0] if curve[0][0] > 0 else float("nan")
        g1 = curve[1][1] / curve[1][0] if len(curve) > 1 and curve[1][0] > 0 else float("nan")
        print(f"    sweep started above the saturation onset (gain already falling at "
              f"{lo:.4g} V: {g0:.4g} -> {g1:.4g} out/in) -- extending down to {new_lo:.3g} V",
              flush=True)
        extra = _sweep(new_lo, lo, max(4, npoints // 2))
        if not extra:
            break
        seen = {a for a, _ in curve}
        curve.extend((a, r) for a, r in extra if a not in seen)
        curve.sort()

    ceiling_a, ceiling = max(curve, key=lambda p: p[1])

    # ONSET IS A GAIN MEASUREMENT, NOT A LEVEL ONE. The old rule -- first input reaching 99%
    # of max(out_rms) -- anchors on a single point of a curve that, on a compressing amp, is
    # flat to within a few percent over four decades. Which point happens to BE the max is
    # then decided by ripple, and the answer swings wildly for no physical reason. Measured on
    # Mesa Orange (sag v30) 2026-09-12, two adjacent cells, freshly rendered:
    #
    #   OR Gain=0.95 Or Master=0.15   out_rms 2.93..3.70   max at 0.053 V  ->  onset  0.043 V
    #   OR Gain=0.95 Or Master=0.20   out_rms 3.94..4.83   max at 40.0  V  ->  onset 21.239 V
    #
    # Same circuit, same shape, both ~97% of final by 0.053 V; a 0.05 change in a master volume
    # "moved" the onset 490x. Across the 72-cell grid the rule produced a required drive that
    # rose MONOTONICALLY with gain (13.7 V at Gain=0.1 up to 24.6 V at Gain=1.0) -- backwards,
    # since more gain must saturate on less input -- and sized an excitation at 49.26 V peak,
    # a level no guitar produces.
    #
    # Reading the small-signal slope and finding where it departs is immune to all of that:
    # whatever sag does to the level at 10 V is simply not part of the question. Taking the
    # FIRST point past the knee (rather than the last one still linear) keeps the figure on
    # the conservative side for excitation sizing -- it is a level known to be saturating.
    lin_top = _linear_region_top(curve)
    knee = None
    if lin_top is not None and lin_top < curve[-1][0]:
        for a, _r in curve:
            if a > lin_top:
                knee = a
                break

    # KNEE AND SIZING ARE DIFFERENT QUESTIONS, and conflating them is how the old rule went
    # wrong in BOTH directions. Measured on Mesa Orange (sag v30) at OR Gain=0.1/Or Master=0.15:
    # small-signal gain is 52.3 out/in and dead flat to 0.033 V, the knee is at 0.053 V, and the
    # cell does not reach its ceiling until ~1.46 V -- knee and full saturation are 27x apart.
    # An excitation sized at 2x the KNEE would leave that cell at ~80% of ceiling, never actually
    # clipping, so callers sizing against this need the saturated level, not the departure point.
    #
    # The ceiling here is the MEDIAN over the top half-decade of inputs, not max(): a single
    # ripple point decided max(), which is what let a 0.05 change in a master volume move the
    # old answer 490x. 95% (not 99%) of it, because the last 1% of an asymptotic approach costs
    # decades of input on a circuit with sag and carries no audible meaning.
    # Window the ceiling to levels ABOVE the knee. A top-decade-of-input window looks right on
    # a 20-point sweep but silently includes still-linear points on a coarse one, dragging the
    # median down until a level where the circuit is provably linear gets called saturated.
    hi = [r for a, r in curve if knee is not None and a > knee] or [curve[-1][1]]
    robust_ceiling = float(np.median(hi))
    # Saturation REQUIRES a departure from linear. Without this guard a perfectly linear
    # circuit gets a bogus onset: the median of a rising straight line is just a mid-sweep
    # value, and "95% of it" is crossed halfway up. That is the same defect as the old rule --
    # a number manufactured out of the swept range rather than measured -- so gate on the knee.
    sat = None
    if knee is not None:
        for a, r in curve:
            if r >= 0.95 * robust_ceiling:
                sat = a
                break
    onset = sat
    # onset=None still means "never departs from linear across the swept range", the signal
    # callers already treat as a hard stop -- unchanged, only now it is honest about why.
    return {"ceiling_rms": ceiling, "ceiling_at_input_v": ceiling_a,
            "robust_ceiling_rms": robust_ceiling,
            "knee_v": knee,                 # where gain departs -- drives the extension logic
            "onset_v": onset,               # where the cell is SATURATED -- what sizing wants
            "onset_method": ONSET_METHOD,
            # Name retained because five call sites and the findpeak cache read it; it is no
            # longer a 99%-of-ceiling figure. ONSET_METHOD is in the cache key, so an entry
            # written by the old rule can never be served to this code.
            "onset_99pct_input_v": onset,
            "curve": curve}


def _linear_region_top(curve, tol=0.05):
    """Top of the constant-gain (linear) input region, from a find-peak curve
    [(in_v, out_rms), ...] ascending. Returns the largest input whose small-signal gain
    (out/in) still matches the lowest-level gain within `tol`. Unlike the saturation onset
    (power-amp RMS ceiling), this catches EARLY preamp nonlinearity AND gain-expansion
    regions -- the level below which a tone stack sees an undistorted signal. Returns None if
    the curve is too short."""
    pts = [(a, r) for a, r in curve if a > 0 and r > 0]
    if len(pts) < 2:
        return None
    g0 = pts[0][1] / pts[0][0]
    top = pts[0][0]
    for a, r in pts:
        if abs((r / a) / g0 - 1.0) > tol:
            break
        top = a
    return top


def scratch_dir(tool: str, keep: bool = False) -> Path:
    """A scratch directory for one run's intermediate renders, deleted when the process exits.

    SCRATCH IS NOT CACHE, and the distinction is the whole point of this helper. A cache
    (findpeak_cache_key above, grid_adequacy's gridadq) is content-keyed, read back on a later
    run, and worth keeping. Scratch is write-only intermediate renders under fixed names --
    nothing ever reads them again. preflight.py and prepare_excitation.py used to drop theirs
    in ~/.cache/parametric-nam/<tool>_scratch, next to the real caches and never cleaned, where
    they quietly reached 123 MB of files for devices nobody was working on any more.

    Cleanup is registered with atexit rather than a `with` block because both callers are long
    single-function CLIs with many sys.exit() paths; atexit covers every one of them, including
    an unhandled exception. It does NOT cover SIGKILL, which is the accepted gap -- the previous
    behaviour did not clean up on ANY exit.

    `keep` (wired to --keep-scratch) skips both the cleanup and the tempdir, using a stable
    predictable path instead, because the reason to keep scratch is to go and look at it.
    """
    if keep:
        d = Path.home() / ".cache" / "parametric-nam" / f"{tool}_scratch"
        d.mkdir(parents=True, exist_ok=True)
        print(f"--keep-scratch: intermediate renders kept in {d} (not cleaned up)",
              file=sys.stderr)
        return d
    d = Path(tempfile.mkdtemp(prefix=f"parametric-nam-{tool}-"))
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


def findpeak_cache_key(identity_bytes, params, extra):
    """Stable cache location for a find-peak result, keyed on caller-supplied `identity_bytes`
    (e.g. a .schx file's own bytes, or a gen_*_ngspice.py module's source bytes) plus the
    sweep-defining parameters -- so an edited circuit re-sweeps automatically."""
    h = hashlib.sha256()
    h.update(ONSET_METHOD.encode())
    h.update(identity_bytes)
    h.update(repr(sorted((str(k), str(v)) for k, v in params.items())).encode())
    h.update(extra.encode())
    d = Path.home() / ".cache" / "parametric-nam" / "findpeak"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h.hexdigest()[:16]}.json"


def cache_findpeak(cpath, sat):
    """Write `sat` to the findpeak cache -- but ONLY if it actually found an onset.

    A result whose `onset_99pct_input_v` is None is a FAILURE, not an answer, and caching
    it is permanent: every later run reads the null back and re-fails identically, so a fix
    to the sweep itself has no effect until someone manually clears ~/.cache. That is not
    hypothetical -- Mesa Dual Rectifier Ch1 (2026-09-10) kept failing on the exact corners a
    verified fix had already solved, because the three call sites all guarded on
    `if sat is not None` and find_saturation_point returns a DICT CONTAINING a null onset,
    never a bare None, so the guard never fired. Returns True if it cached.
    """
    if not sat or sat.get("onset_99pct_input_v") is None:
        return False
    cpath.write_text(json.dumps(sat))
    return True
