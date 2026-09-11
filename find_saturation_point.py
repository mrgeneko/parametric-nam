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
        ceiling_now = max(r for _, r in curve)
        if curve[0][1] < 0.99 * ceiling_now:
            break  # lowest point is off the plateau -- a crossing is in range
        lo = curve[0][0]
        if lo <= min_start_v:
            break
        new_lo = max(lo / 100.0, min_start_v)
        # Printed UNCONDITIONALLY, not gated on `progress`: prepare_excitation.py (the main
        # caller, one sweep per grid corner) passes no progress callback, so gating this on it
        # made the extension silent in exactly the pipeline that depends on it -- Ch1's 2026-09-10
        # run extended down on real corners and reported nothing. It fires rarely (only when the
        # floor is on the plateau), so it cannot flood even a 200-corner run.
        print(f"    sweep started above the saturation onset (out flat at "
              f"{ceiling_now:.4g} V from {lo:.4g} V up) -- extending down to {new_lo:.3g} V",
              flush=True)
        extra = _sweep(new_lo, lo, max(4, npoints // 2))
        if not extra:
            break
        seen = {a for a, _ in curve}
        curve.extend((a, r) for a, r in extra if a not in seen)
        curve.sort()

    ceiling_a, ceiling = max(curve, key=lambda p: p[1])
    target = 0.99 * ceiling
    onset = None
    for j in range(1, len(curve)):
        a0, r0 = curve[j - 1]
        a1, r1 = curve[j]
        if r0 < target <= r1:
            onset = _loglog_interp(a0, r0, a1, r1, target)
            break
    return {"ceiling_rms": ceiling, "ceiling_at_input_v": ceiling_a,
            "onset_99pct_input_v": onset, "curve": curve}


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
