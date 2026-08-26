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
Fulltone OCD's ngspice deck: a sustained tone with no lead-in showed RMS drifting for ~15s
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
import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SR = 48000


def _loglog_interp(x1, y1, x2, y2, ytarget):
    lx1, ly1 = math.log(x1), math.log(y1)
    lx2, ly2 = math.log(x2), math.log(y2)
    frac = (math.log(ytarget) - ly1) / (ly2 - ly1)
    return math.exp(lx1 + frac * (lx2 - lx1))


def find_saturation_point(backend, params, tmp, freq=200.0, dur=2.0, lead_silence_s=0.0,
                           start_v=0.005, max_v=40.0, npoints=20, sr=SR, workers=8,
                           progress=None):
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

    Returns None if every amplitude fails to converge.
    """
    Path(tmp).mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(sr * lead_silence_s), dtype=np.float32)
    t_tone = np.arange(int(sr * dur)) / sr
    tone = np.sin(2 * np.pi * freq * t_tone).astype(np.float32)
    raw = np.concatenate([silence, tone])
    tone_start = len(silence)

    log_amps = np.geomspace(start_v, max_v, npoints)

    def _one(i_amp):
        i, amp = i_amp
        tag = f"fp_{i}"
        handle = backend.prepare_input(raw, sr, float(amp), tmp, tag)
        ys = backend.render_many([{"params": params, "tag": tag}], handle, tmp)
        return float(amp), ys.get(tag)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers, len(log_amps))) as ex:
        futures = [ex.submit(_one, ia) for ia in enumerate(log_amps)]
        raw_results = []
        for i, fut in enumerate(as_completed(futures), 1):
            raw_results.append(fut.result())
            if progress is not None:
                progress(i, len(futures), time.monotonic() - t0)

    curve = []
    for amp, y in raw_results:
        if y is None:
            continue
        steady = y[tone_start + int(sr * dur * 0.5):]
        if len(steady) == 0:
            continue
        curve.append((amp, float(np.sqrt((steady ** 2).mean()))))
    curve.sort()
    if not curve:
        return None

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
