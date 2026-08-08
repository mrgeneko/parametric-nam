#!/usr/bin/env python3
"""Measure BDF2 truncation error for one device.

WHY THIS FILE EXISTS AS A FILE.
The first fleet-wide truncation table (internal engineering notes, commit 9d986c5) was produced by an
ad-hoc script that was never committed. When its Big Muff number (2.25e-02) later failed to
reproduce -- a careful re-measurement on the same input, same circuit, same worst corner gives
9.33e-03 -- there was nothing to audit and no way to find out which of the two was wrong. A
measurement that decides whether we retrain the entire model fleet has to be re-runnable by
someone who doubts it. So it lives here, in the repo, and the docs quote it rather than the
other way round.

WHAT IT MEASURES.
oversample is a DISCRETISATION choice. LiveSPICE integrates with BDF2, which is O(h^2), so the
simulated circuit is not quite the real circuit; the gap is TRUNCATION ERROR. It is not a bug and
it is never zero. But its SIZE decides whether it matters: if the truncation in the training target
is the same order as the ESR of the model you fit to that target, the model is spending capacity
learning the integrator's mistakes. A model cannot be more right than its target.

HOW (and the trap).
Against a REFERENCE at high oversample -- NOT against the next rung up. ESR(os, 2*os) is the
tempting cheap estimate and it UNDERSTATES the error ~3x, which is the dangerous direction: you
pick too low an oversample and ship a contaminated target. The finer render is not the truth, it
is merely less wrong, and the two errors are correlated and largely cancel. Measured on the Big
Muff, worst corner, whole file:

    os      ESR vs 2*os      ESR vs os=32     understated by
     2       2.83e-03         9.33e-03            3.3x
     4       6.45e-04         1.98e-03            3.1x
     8       1.63e-04         3.70e-04            2.3x

All renders at 256 Newton iterations, so under-convergence is not a confound (that is a different
failure and internal engineering notes cover it).

KNOB SETTINGS. Both ends of every knob, plus both corners, plus the mid point -- and we report
WHERE the worst was. "All knobs at max" is not reliably the stiff setting: the Boss DS-1's Dist pot
is ReverseLinear, so all-max is MINIMUM drive.

Usage:
    ./measure_truncation.py --input ../sweep-files/sweep60_composite.wav \
        --config ../parametric-nam-models/pedals/big-muff-pi-v1-66-5/config.toml

`--config` is the same TOML `run_pipeline.py`/`grid_adequacy.py` take (schx + knobs come straight
from it, like every other analysis tool here) -- no registry, no `parametric-devices` dependency.
One device per invocation, on purpose: a fleet-wide table is rare (it matters only when
re-deriving the shared oversample floor, or auditing every device after a measurement-methodology
change; see internal engineering notes) and is just as easy from the outside:

    for f in ../parametric-nam-models/*/*/config.toml*; do
        ./measure_truncation.py --input ../sweep-files/sweep60_composite.wav --config "$f"
    done
"""
from __future__ import annotations

import argparse
import os as _os
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf

from batch_harness import _livespice_batch, write_probe_clip


def esr_terms(a: np.ndarray, b: np.ndarray, lead_n: int, sr: int) -> tuple[float, float]:
    """(numerator, denominator) of the ESR of `a` against reference `b`, dropping the
    leading transient. Returned as terms, not a ratio, so window measurements can be
    POOLED: sum(err)/sum(sig) across windows is exactly the whole-file ESR restricted
    to the sampled windows (see choose_oversample). On a whole-file measurement
    (lead_n=0) the skip reduces to the historical 10%."""
    m = min(len(a), len(b))
    sk = max(m // 10, lead_n + int(0.02 * sr))
    if m - sk < 4800:
        return 0.0, 0.0
    return (float(np.sum((a[sk:m] - b[sk:m]) ** 2)),
            float(np.sum(b[sk:m] ** 2)))


def probe_clips(input_wav: Path, probe_s: float, n_windows: int, td: Path
                ) -> tuple[list[Path], int, int]:
    """Cut stratified probe windows from the input; returns (clips, lead_n, sr).

    probe_s <= 0 means measure the WHOLE file (the historical behavior): one 'window'
    that is the input itself, no lead-in.

    Truncation is a property of the solver at a knob setting, not of sweep length --
    but WHERE you probe decides the answer (our sweeps open quiet and end in a decay
    tail), so this mirrors choose_oversample exactly: evenly-spaced window CENTRES
    across the file, each window rendered separately with write_probe_clip's silent
    lead-in + ramp (never spliced -- a splice manufactures step discontinuities and
    the probe then measures its own splice), numerator/denominator pooled across
    windows. At the default 10 s of a 60 s sweep this cuts the render bill ~6x, and
    the reference renders (16x the cost of an os=2 render each) with it.
    """
    if probe_s <= 0:
        sr = sf.info(str(input_wav)).samplerate
        return [input_wav], 0, sr
    sig, sr = sf.read(str(input_wav))
    total = len(sig)
    win = min(total, int(probe_s * sr) // n_windows or total)
    if total <= win * n_windows:
        starts = [0]
        win = total
    else:
        span = total - win
        starts = [int((i + 0.5) * span / n_windows) for i in range(n_windows)]
    clips, lead_n = [], 0
    for wi, st in enumerate(starts):
        c = td / f"probe_w{wi}.wav"
        lead_n = write_probe_clip(sig[st:st + win], sr, c)
        clips.append(c)
    return clips, lead_n, sr


def probe_settings(knobs: list[str]) -> list[dict]:
    """Both ends of each knob, both corners, and the mid point."""
    picks: list[dict] = []
    for k in knobs:
        for v in (0.0, 1.0):
            p = {j: 0.5 for j in knobs}
            p[k] = v
            picks.append(p)
    picks.append({k: 0.0 for k in knobs})
    picks.append({k: 1.0 for k in knobs})
    picks.append({k: 0.5 for k in knobs})
    seen, uniq = set(), []
    for p in picks:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def load_device(config_path: Path) -> tuple[str, Path, list[str]]:
    """(name, schx, knobs) from a config.toml -- the same file run_pipeline.py trains from.

    knobs come from the config's own [knobs] table, not every real control on the device: a
    recipe that pins a control in [fixed] (e.g. the Big Muff's Volume) isn't swept in training,
    so it shouldn't be swept here either -- this measures the truncation error THIS RECIPE will
    actually see, not a hypothetical full-control sweep.
    """
    cfg = tomllib.loads(config_path.read_text())
    if "schx" not in cfg:
        raise ValueError(f"{config_path}: no [schx] -- not a SPICE-backed config "
                          "(e.g. a real-hardware-capture recipe); truncation doesn't apply to it")
    knobs = list((cfg.get("knobs") or {}).keys())
    if not knobs:
        raise ValueError(f"{config_path}: no [knobs] -- nothing to sweep")
    schx = Path(_os.path.expanduser(cfg["schx"]))
    return (config_path.parent.name, schx, knobs)


def measure(schx: Path, knobs: list[str], clips: list[Path], lead_n: int, sr: int,
            ref_os: int, candidates: tuple[int, ...], iterations: int,
            speaker: str | None, td: Path, workers: int) -> dict:
    """Worst-over-knob-settings truncation ESR at each candidate oversample.

    Renders run CONCURRENTLY. Serially this was leaving 13 of 14 cores idle while a single
    oversample-32 render -- 16x the work of the os=2 one it is the reference for -- ground through a
    60 s file, and a 7-device fleet would have taken hours. Threads are the right tool: every unit of
    work is a subprocess, so the GIL is irrelevant.

    `clips` are the probe windows from probe_clips() (possibly just the whole input); each
    (setting, oversample) is rendered per window and the ESR numerator/denominator POOLED
    across windows, so the reported number estimates the same whole-file quantity either way.
    """
    settings = probe_settings(knobs)

    # All (setting, oversample, window) renders through the oracle's --jobs batch mode:
    # parallel across workers, per-invocation fixed cost paid per worker, not per render.
    def run_batch(triples) -> dict[tuple, np.ndarray | None]:
        jobs, keymap = [], {}
        for p, os_, wi in triples:
            key = (tuple(sorted(p.items())), os_, wi)
            w = td / f"{abs(hash((str(schx), key)))}.wav"
            jobs.append({"input": str(clips[wi]), "output": str(w),
                         "params": ",".join(f"{k}={v}" for k, v in p.items()),
                         "oversample": os_, "iterations": iterations})
            keymap[str(w)] = key
        errs = _livespice_batch(str(schx), jobs, workers, speaker)
        out: dict[tuple, np.ndarray | None] = {}
        for wpath, key in keymap.items():
            if errs.get(wpath) is None and Path(wpath).exists():
                d, _ = sf.read(wpath)
                out[key] = np.asarray(d, dtype=np.float64)
            else:
                print(f"      render failed (os={key[1]}): {errs.get(wpath)}", file=sys.stderr)
                out[key] = None
        return out

    cache = run_batch([(p, os_, wi) for p in settings for os_ in (*candidates, ref_os)
                       for wi in range(len(clips))])

    def pooled_esr(k: tuple, os_a: int, os_b: int) -> float:
        num = den = 0.0
        for wi in range(len(clips)):
            a, b = cache.get((k, os_a, wi)), cache.get((k, os_b, wi))
            if a is None or b is None:
                continue
            n, d = esr_terms(a, b, lead_n, sr)
            num += n
            den += d
        return num / den if den > 0 else float("nan")

    res: dict = {"n_settings": len(settings)}
    for os_ in candidates:
        worst, at = 0.0, None
        for p in settings:
            e = pooled_esr(tuple(sorted(p.items())), os_, ref_os)
            if np.isfinite(e) and e >= worst:
                worst, at = e, p
        res[os_] = (worst, at)

    # IS THE REFERENCE ITSELF CONVERGED?
    #
    # Every number above is measured against os=ref_os, on the assumption that the reference is
    # close enough to the truth to stand in for it. That assumption is exactly the kind we have been
    # burned by -- a reference that has not converged is not a reference, the same way an oracle
    # built from the thing under test is not an oracle -- and it is cheap to check: render the worst
    # setting at 2*ref_os and see how far the reference still moves.
    #
    # If ESR(ref, 2*ref) is not far BELOW the smallest number in the table, the table is measuring
    # the reference's own error, the ratios go flat, and every entry is an UNDERSTATEMENT.
    _, at_worst = res[candidates[0]]
    if at_worst is not None:
        cache.update(run_batch([(at_worst, ref_os * 2, wi) for wi in range(len(clips))]))
        res["ref_error"] = pooled_esr(tuple(sorted(at_worst.items())), ref_os, ref_os * 2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="the sweep to measure on. Truncation ESR is INPUT-DEPENDENT (a quiet tail "
                         "shrinks the denominator), so the table is only meaningful with the input "
                         "named. Use the sweep the datasets were actually rendered with.")
    ap.add_argument("--config", type=Path, required=True, help="the same TOML the pipeline uses")
    ap.add_argument("--ref-os", type=int, default=32, help="reference oversample (default 32)")
    ap.add_argument("--candidates", default="2,4,8",
                    help="oversamples to score against the reference (default 2,4,8)")
    ap.add_argument("--iterations", type=int, default=256,
                    help="Newton iterations; high, so convergence is not a confound (default 256)")
    ap.add_argument("--speaker", help="speaker to capture on multi-speaker circuits")
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 4) - 2),
                    help="concurrent renders (default: cores-2)")
    ap.add_argument("--probe-s", type=float, default=10.0,
                    help="total seconds of the input to actually render, as stratified windows "
                         "(default 10). Truncation is a property of the solver, not of sweep "
                         "length; windowed+pooled estimates the same whole-file quantity at a "
                         "fraction of the cost. Validated on the Big Muff vs the documented "
                         "whole-file table: ~1.3x HIGH at every rung (the safe direction -- "
                         "borderline circuits get more oversample, not less), same worst "
                         "setting, identical fall ratios. 0 = whole file (slow; the historical "
                         "behavior).")
    ap.add_argument("--n-windows", type=int, default=4,
                    help="number of stratified windows --probe-s is split into (default 4)")
    args = ap.parse_args()

    cands = tuple(int(c) for c in args.candidates.split(","))
    if max(cands) >= args.ref_os:
        ap.error(f"--candidates must all be below --ref-os ({args.ref_os}): you cannot measure the "
                 f"reference against itself")

    try:
        name, schx, knobs = load_device(args.config)
    except ValueError as e:
        ap.error(str(e))
    if not schx.exists():
        ap.error(f"{name}: MISSING {schx}")

    print(f"input:      {args.input.name}")
    print(f"reference:  oversample={args.ref_os}, {args.iterations} Newton iterations")
    print(f"device:     {name} ({len(knobs)} knobs)\n")

    rows = []
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        clips, lead_n, sr = probe_clips(args.input, args.probe_s, args.n_windows, td)
        if len(clips) > 1 or clips[0] != args.input:
            print(f"probing:    {len(clips)} x {sf.info(str(clips[0])).frames / sr:.1f}s windows "
                  f"(--probe-s {args.probe_s:g}; 0 = whole file)\n")
        print(f"  measuring {name} ({len(knobs)} knobs) ...", flush=True, file=sys.stderr)
        r = measure(schx, knobs, clips, lead_n, sr, args.ref_os, cands,
                    args.iterations, args.speaker, td, args.workers)
        rows.append((name, r))

    hdr = " | ".join(f"@ os={c}" for c in cands)
    print(f"\n| device | {hdr} | worst setting @ os={cands[0]} | ref err | verdict |")
    print("|---|" + "---:|" * len(cands) + "---|---:|---|")
    for name, r in rows:
        cells = []
        for c in cands:
            v, _ = r.get(c, (float('nan'), None))
            cells.append(f"{v:.2e}")
        v0, at0 = r.get(cands[0], (float('nan'), None))
        vlast, _ = r.get(cands[-1], (float('nan'), None))
        atstr = ", ".join(f"{k}={v:g}" for k, v in sorted(at0.items())) if at0 else "-"

        # EVERY consecutive ratio, not just the first. Checking only cands[0]/cands[1] would have
        # passed the JCM800 hot-rod as healthy -- it falls a respectable 4.4x from os=2 to os=4 and
        # then STALLS at 1.6x from 4 to 8. The stall is the finding; a verdict that reads only the
        # first rung reports the circuit as smooth and hides it.
        ratios = []
        for a_, b_ in zip(cands, cands[1:]):
            va, _ = r.get(a_, (float('nan'), None))
            vb, _ = r.get(b_, (float('nan'), None))
            ratios.append(va / vb if (np.isfinite(va) and np.isfinite(vb) and vb > 0) else float('nan'))
        fall = " → ".join(f"{x:.1f}x" for x in ratios) if ratios else "-"

        # The reference must be much cleaner than the finest number it is used to score. If it is
        # not, the table bottoms out on the REFERENCE's error, not the circuit's.
        ref_err = r.get("ref_error", float("nan"))
        refstr = f"{ref_err:.1e}" if np.isfinite(ref_err) else "?"

        stalled = [(a_, b_, x) for (a_, b_), x in zip(zip(cands, cands[1:]), ratios)
                   if np.isfinite(x) and x < 2.0]
        if np.isfinite(ref_err) and np.isfinite(vlast) and ref_err > 0.25 * vlast:
            verdict = f"**REFERENCE NOT CONVERGED** — os={args.ref_os} still moves by {ref_err:.1e}, " \
                      f"not far below the @os={cands[-1]} figure; that column is a floor, not a measurement"
        elif stalled:
            a_, b_, x = stalled[0]
            verdict = f"**STALLS** {a_}→{b_}: falls only {x:.1f}x, not ~4x ({fall})"
        else:
            verdict = f"falls {fall}"
        print(f"| {name} | " + " | ".join(cells) + f" | {atstr} | {refstr} | {verdict} |")

    print("\n`ref err` = ESR(reference, 2x reference) at the worst setting: how far the reference "
          f"itself\nstill moves. It must sit well below the @os={cands[-1]} column, or that column is "
          "measuring the\nreference's own error rather than the circuit's.\n")
    print("O(h^2) demands the error fall ~4x per doubling. A device that falls <2x does not "
          "converge in the timestep:\nsomething in it is not smooth, and oversampling will not fix "
          "its dataset.")


if __name__ == "__main__":
    main()
