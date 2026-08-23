#!/usr/bin/env python3
"""Measure ngspice timestep (`maxstep`) truncation error for a hand-written ngspice-deck device.

WHY THIS FILE EXISTS.
render_ngspice_deck.py / render_backends.py's NgspiceBackend / ngspice_spicelib.py's
render_grid() all default `maxstep=3e-6` -- picked so a render usually CONVERGES (produces
output at all), never measured for ACCURACY. Found directly on the Fulltone OCD
(gen_ocd_ngspice.py, real 2N7000 MOSFET clipping): re-rendering the IDENTICAL Gain=0.3 probe
window at maxstep 3e-6 / 1e-6 / 3e-7 gave RMS 1.210 / 1.614 / 1.593 -- a ~33% swing between the
default and a 10x finer step, and still not settled between the two finer steps.
tools/grid_adequacy.py --apply was exploding (40 -> 1512+ permutations, ~10-100x over target at
nearly every cell, barely improving as it densified) instead of converging -- exactly the
signature of an interpolation-error measurement dominated by non-smooth INTEGRATION noise, not
real circuit curvature.

This is the maxstep analogue of measure_truncation.py's BDF2/oversample story for LiveSPICE --
same "MEASURE the information, don't guess" principle (see docs/adding-a-device.md), previously
missing for the ngspice-deck backend entirely. probe_clips/probe_settings/esr_terms are reused
directly from measure_truncation.py -- they're already backend-agnostic (pure knob-list /
wav-window / ESR math, no LiveSPICE dependency), so there is no reason to re-derive them.

HOW (same trap measure_truncation.py already documents for oversample, direction inverted since
SMALLER maxstep is finer, not larger): score each candidate maxstep against a REFERENCE at a
much smaller maxstep, not the next rung down -- the next rung understates the error the same way
ESR(os, 2*os) does on the LiveSPICE side. Report where the worst knob setting is (both ends of
every knob, both corners, the midpoint -- same set measure_truncation.py uses), and whether the
reference itself has actually converged (ESR(ref, ref/2) must sit well below the worst
candidate's own number, or every column in the table is a floor, not a measurement).

Usage:
    ./tools/measure_ngspice_timestep.py --input ~/work/parametric-devices/pedals/ocd_excitation_v4.wav \\
        --config ~/work/parametric-nam-models/pedals/fulltone-ocd-lp/config.toml
"""
from __future__ import annotations

import argparse
import os as _os
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measure_truncation import esr_terms, probe_clips, probe_settings  # noqa: E402

from tools.ngspice_spicelib import load_input  # noqa: E402
from tools.render_backends import NgspiceBackend  # noqa: E402


def load_config(p: Path) -> dict:
    with open(p, "rb") as f:
        return tomllib.load(f)


def load_device(config_path: Path) -> tuple:
    """(name, build_deck, probe_node, knobs, fixed) from a config.toml -- same [knobs]-comes-
    from-the-recipe reasoning as measure_truncation.py's load_device: a control pinned in
    [fixed] isn't swept in training, so it shouldn't be swept here either."""
    cfg = load_config(config_path)
    if cfg.get("backend") != "ngspice-deck":
        raise ValueError(f"{config_path}: backend != \"ngspice-deck\" -- this tool measures the "
                         "hand-written-deck timestep; use measure_truncation.py for a .schx config")
    knobs = list((cfg.get("knobs") or {}).keys())
    if not knobs:
        raise ValueError(f"{config_path}: no [knobs] -- nothing to sweep")
    pedal_dir = _os.path.expanduser(cfg["pedal-dir"])
    module = cfg["module"]
    sys.path.insert(0, _os.path.abspath(pedal_dir))
    import importlib
    mod = importlib.import_module(module)
    fixed = {k: float(v) for k, v in (cfg.get("fixed") or {}).items()}
    return (config_path.parent.name, mod.build_deck, cfg.get("probe-node", "OUT"), knobs, fixed)


def measure(build_deck, probe_node: str, knobs: list, fixed: dict, clips: list, lead_n: int,
           sr: int, ref_maxstep: float, candidates: tuple, parallel_sims: int, td: Path) -> dict:
    """Worst-over-knob-settings truncation ESR at each candidate maxstep, against a reference
    at ref_maxstep (must be smaller/finer than every candidate)."""
    settings = probe_settings(knobs)
    handles = [load_input(str(c), None, str(td), src_name=f"mst_{i}.src")
              for i, c in enumerate(clips)]

    cache: dict = {}
    fail_counts: dict = {}

    def run_batch(maxstep: float):
        backend = NgspiceBackend(build_deck, probe_node=probe_node, parallel_sims=parallel_sims,
                                 maxstep=maxstep)
        for wi, handle in enumerate(handles):
            jobs, keymap = [], {}
            for p in settings:
                full_params = {**fixed, **p}
                key = (tuple(sorted(p.items())), maxstep, wi)
                tag = str(abs(hash(key)))
                jobs.append({"params": full_params, "tag": tag})
                keymap[tag] = key
            ys = backend.render_many(jobs, handle, td)
            for tag, key in keymap.items():
                y = ys.get(tag)
                cache[key] = y
                if y is None:
                    fail_counts["did not converge"] = fail_counts.get("did not converge", 0) + 1

    for ms in (*candidates, ref_maxstep):
        run_batch(ms)

    for msg, n in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
        print(f"      {n} render(s) failed: {msg}", file=sys.stderr)

    if cache and all(v is None for v in cache.values()):
        raise RuntimeError(
            f"EVERY render failed ({len(cache)}/{len(cache)}). Nothing was measured, so there is "
            "no table to print. This usually means a config problem (wrong knob/fixed-param "
            "name, wrong --probe-node), not a convergence issue; fix that and re-run.")

    def pooled_esr(key_base: tuple, ms_a: float, ms_b: float) -> float:
        num = den = 0.0
        for wi in range(len(clips)):
            a, b = cache.get((key_base, ms_a, wi)), cache.get((key_base, ms_b, wi))
            if a is None or b is None:
                continue
            n, d = esr_terms(a, b, lead_n, sr)
            num += n
            den += d
        return num / den if den > 0 else float("nan")

    res: dict = {"n_settings": len(settings)}
    for ms in candidates:
        worst, at = 0.0, None
        for p in settings:
            e = pooled_esr(tuple(sorted(p.items())), ms, ref_maxstep)
            if np.isfinite(e) and e >= worst:
                worst, at = e, p
        res[ms] = (worst, at)

    # Is the reference itself converged? Render the worst setting at ref_maxstep/2 and see how
    # far the reference still moves -- same check measure_truncation.py runs on `ref_os*2`.
    _, at_worst = res[candidates[0]]
    if at_worst is not None:
        run_batch(ref_maxstep / 2)
        res["ref_error"] = pooled_esr(tuple(sorted(at_worst.items())), ref_maxstep, ref_maxstep / 2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="the excitation to measure on. Truncation ESR is INPUT-DEPENDENT, so "
                         "the table is only meaningful with the input named. Use the excitation "
                         "the dataset was actually rendered with.")
    ap.add_argument("--config", type=Path, required=True,
                    help="the same ngspice-deck config.toml grid_adequacy.py/run_pipeline.py use")
    ap.add_argument("--ref-maxstep", type=float, default=3e-8,
                    help="reference (finest) timestep -- must be smaller than every candidate "
                         "(default 3e-8)")
    ap.add_argument("--candidates", default="3e-6,1e-6,3e-7",
                    help="maxsteps to score against the reference, coarsest first (default "
                         "3e-6,1e-6,3e-7 -- 3e-6 is NgspiceBackend's own hardcoded default)")
    ap.add_argument("--parallel-sims", type=int, default=max(1, (_os.cpu_count() or 4) - 2))
    ap.add_argument("--probe-s", type=float, default=8.0,
                    help="total seconds of the input to render, as stratified windows (default "
                         "8 -- ngspice SEGFAULTS on shorter clips, same convention as "
                         "grid_adequacy.py's Renderer)")
    ap.add_argument("--n-windows", type=int, default=2)
    args = ap.parse_args()

    cands = tuple(float(c) for c in args.candidates.split(","))
    if min(cands) <= args.ref_maxstep:
        ap.error(f"--candidates must all be LARGER (coarser) than --ref-maxstep "
                 f"({args.ref_maxstep:g}): you cannot measure the reference against itself")

    try:
        name, build_deck, probe_node, knobs, fixed = load_device(args.config)
    except (ValueError, KeyError) as e:
        ap.error(str(e))

    probe_s = max(args.probe_s, 8.0)
    print(f"input:      {args.input.name}")
    print(f"reference:  maxstep={args.ref_maxstep:g}")
    print(f"device:     {name} ({len(knobs)} knobs, fixed={fixed or '-'})\n")

    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        clips, lead_n, sr = probe_clips(args.input, probe_s, args.n_windows, td)
        print(f"probing:    {len(clips)} x {probe_s / args.n_windows:.1f}s windows\n")
        print(f"  measuring {name} ({len(knobs)} knobs) ...", flush=True, file=sys.stderr)
        try:
            r = measure(build_deck, probe_node, knobs, fixed, clips, lead_n, sr,
                       args.ref_maxstep, cands, args.parallel_sims, td)
        except RuntimeError as e:
            sys.exit(f"\nERROR: {e}")

    hdr = " | ".join(f"@ ms={c:g}" for c in cands)
    print(f"\n| device | {hdr} | worst setting @ ms={cands[0]:g} | ref err | verdict |")
    print("|---|" + "---:|" * len(cands) + "---|---:|---|")
    cells = []
    for c in cands:
        v, _ = r.get(c, (float("nan"), None))
        cells.append(f"{v:.2e}")
    v0, at0 = r.get(cands[0], (float("nan"), None))
    vlast, _ = r.get(cands[-1], (float("nan"), None))
    atstr = ", ".join(f"{k}={v:g}" for k, v in sorted(at0.items())) if at0 else "-"

    # EVERY consecutive ratio, not just the first -- a stall further down the table (candidates
    # getting finer) is the finding, same trap measure_truncation.py's own verdict logic guards.
    ratios = []
    for a_, b_ in zip(cands, cands[1:]):
        va, _ = r.get(a_, (float("nan"), None))
        vb, _ = r.get(b_, (float("nan"), None))
        ratios.append(va / vb if (np.isfinite(va) and np.isfinite(vb) and vb > 0) else float("nan"))
    fall = " -> ".join(f"{x:.1f}x" for x in ratios) if ratios else "-"

    ref_err = r.get("ref_error", float("nan"))
    refstr = f"{ref_err:.1e}" if np.isfinite(ref_err) else "?"

    stalled = [(a_, b_, x) for (a_, b_), x in zip(zip(cands, cands[1:]), ratios)
              if np.isfinite(x) and x < 2.0]
    if np.isfinite(ref_err) and np.isfinite(vlast) and ref_err > 0.25 * vlast:
        verdict = (f"**REFERENCE NOT CONVERGED** -- maxstep={args.ref_maxstep:g} still moves by "
                  f"{ref_err:.1e}, not far below the @ms={cands[-1]:g} figure; that column is a "
                  f"floor, not a measurement")
    elif stalled:
        a_, b_, x = stalled[0]
        verdict = f"**STALLS** {a_:g}->{b_:g}: falls only {x:.1f}x ({fall})"
    else:
        verdict = f"falls {fall}"
    print(f"| {name} | " + " | ".join(cells) + f" | {atstr} | {refstr} | {verdict} |")

    print("\n`ref err` = ESR(reference, reference/2) at the worst setting: how far the reference "
          "itself\nstill moves. It must sit well below the finest candidate column, or that "
          "column is measuring the\nreference's own error rather than the circuit's.\n")
    print("A device whose error does not fall as maxstep shrinks does not converge in the "
          "timestep:\nsomething in it is not smooth, and a smaller maxstep alone will not fix "
          "its dataset.")


if __name__ == "__main__":
    main()
