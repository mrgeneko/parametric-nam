#!/usr/bin/env python3
"""Pre-generation gate: does the excitation's TRANSIENT content actually reach
saturation at every knob-grid corner, not just the excitation's overall peak?

Background (internal engineering notes): a device can have a peak level
(from its synthetic chirp tail) that comfortably clears its saturation onset, while
the TRANSIENT-bearing segment -- build_excitation.py's `--sweep-file` clip, placed at its own
`--sweep-peak` -- never does, at some corners. (That segment is NOT necessarily real
playing: the standard capture sweep normally passed as `--sweep-file` is itself synthesized --
frequency sweep + noise-staircase + calibration blips, TONE3000's own term for this style of
file -- and its high crest factor comes from that structure, not from musical dynamics. See
build_excitation.py's docstring. What matters here is only that it is the crest-bearing part,
whatever its source.) The tweed-style amp is the exact case
this happened on: `find-peak @ knobs=0.5` measured onset ~0.51 V, but
`--sweep-peak` was set to 0.2 V regardless (chosen for input-signal realism,
not cross-checked against the measured onset) -- so that `--sweep-file` content
stayed in the LINEAR region at every corner tested, while only the chirp (a smooth
tone, no attack shape) crossed into saturation there. The network never saw a
transient AND saturation together at that corner, and ran open-loop when a real
one eventually arrived. This tool automates the cross-check that was missing:
for every corner (reduced hypercube set, matching scan_film_runaway.py's
convention), find that corner's OWN saturation onset (reusing
find_saturation_point.py's shared, backend-agnostic algorithm) and compare it
against the excitation's transient peak.

Exit status is nonzero if any corner fails, so a generation script can gate on it
(same convention as preflight.py):
  python check_transient_coverage.py ... && python gen_dataset_from_schx.py ...

Usage:
  livespice:   python check_transient_coverage.py --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml \
      [--transient-peak 0.2] [--margin 1.0] [--oversample 8] [--iterations 256] \
      [--json report.json] [--no-cache]

  ngspice-deck: same --config, with [knobs]/[fixed]/backend="ngspice-deck"/pedal-dir/module/
      probe-node in the TOML (same convention as preflight.py/prepare_excitation.py --backend
      ngspice-deck) -- for a device whose clipping needs a real component .schx has no model
      for (a MOSFET, a real BJT), so there's no .schx at all to check against.

--transient-peak: the excitation's transient-bearing `--sweep-file` segment peak, in volts
  at V0dBFS=1. Auto-read from the excitation's <stem>.recipe.json sidecar
  (build_excitation.py's `args.sweep_peak`, or its pre-2026-09-10 name `args.realistic_peak`
  for an older recipe) if present; otherwise REQUIRED -- this tool refuses to guess it from
  the raw audio (silently mis-slicing the file's sweep/chirp boundary would be worse than
  refusing to run).
"""
import argparse
import importlib
import itertools
import random
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_pipeline import load_config  # noqa: E402
from capture_chain import (add_cli_args as _cc_add_cli_args, resolve as _cc_resolve,  # noqa: E402
                           cache_tag)
from find_saturation_point import (find_saturation_point, findpeak_cache_key,  # noqa: E402
                                    cache_findpeak)
from render_backends import (LiveSpiceBackend, NgspiceBackend, LtspiceBackend,  # noqa: E402
                             NgspiceSchxBackend, parse_conv, conv_cache_tag)
# NOT imported at module level: prepare_excitation.py imports FROM this module
# (resolve_sample_grid/_corners/_sample_interior), so a top-level import here would be
# circular. Imported lazily, at first use, inside _check_corners() instead -- by then this
# module is already fully loaded, so the cycle never actually forms.

SR = 48000


DEFAULT_MAX_CORNERS = 512


def _corners(knob_ranges: dict, full_hypercube: "bool | None" = None,
             max_full_corners: int = DEFAULT_MAX_CORNERS,
             max_corners: "int | None" = None,
             sample_grid: int = 0) -> list:
    """Corner set: all-min, all-max, center, each knob solo-extreme (rest at their own
    center), PLUS (if full_hypercube) the full binary hypercube -- every knob independently
    at its own min or max, 2**n corners total (all-min/all-max are 2 of them; deduped below).

    The binary hypercube was added after a real miss: the tweed-style amp's shipped blowup
    corner was NormalVol/BrightVol held at their grid-min SIMULTANEOUSLY with
    Treble/Bass/Middle at their grid-max. The solo set can't represent that -- solo holds
    every OTHER knob at center, never at another extreme -- so a MIXED
    some-knobs-low-others-high corner went completely untested, and the excitation was
    never checked (or built) to saturate there. 2**n is still exponential, just far cheaper
    than the full trained grid this was originally reduced from (972 renders x 20 amplitude
    points for Tweed alone) -- capped at max_full_corners (default 512, i.e. up to 9 knobs);
    pass full_hypercube=False for a config with more knobs than that rather than let this
    silently balloon.

    Uses each knob's OWN grid min/max/center (not a blanket 0/1/0.5), since
    swept ranges are frequently narrowed (e.g. Tweed's tone stack is 0.2..0.8).
    """
    names = list(knob_ranges.keys())
    lo = {n: min(vs) for n, vs in knob_ranges.items()}
    hi = {n: max(vs) for n, vs in knob_ranges.items()}
    mid = {n: vs[len(vs) // 2] for n, vs in knob_ranges.items()}   # nearest-to-center grid point

    # Structural corners are DEDUPED, not just the hypercube ones below. `mid` is the
    # nearest-to-centre GRID POINT, so on an even-cardinality axis it is not central: for a
    # 2-value axis [0.2, 0.8] it picks index 1, i.e. the MAX -- which makes that axis's
    # "hi-solo" bit-identical to "center". Measured on Mesa Dual Rectifier (2026-09-04):
    # center, Bass=hi-solo, Mid=hi-solo and Treble=hi-solo all returned onset 14.041 V,
    # because they were the same knob setting sweep-probed four times. Three wasted onset
    # sweeps out of 43, on a run where each costs ~50s. Deduping is the conservative fix;
    # redefining `mid` would change which point "centre" means for every device.
    corners = []
    seen_struct = set()
    def _push(label, vals):
        key = tuple(sorted(vals.items()))
        if key in seen_struct:
            return
        seen_struct.add(key)
        corners.append((label, vals))

    _push("all-min", dict(lo))
    _push("all-max", dict(hi))
    _push("center", dict(mid))
    for n in names:
        solo_lo = dict(mid); solo_lo[n] = lo[n]
        solo_hi = dict(mid); solo_hi[n] = hi[n]
        _push(f"{n}=lo-solo", solo_lo)
        _push(f"{n}=hi-solo", solo_hi)

    if full_hypercube is False:
        # DEPRECATED. Kept so existing callers keep working, but it is the WORST available
        # reduction and it warns: the structural set above holds every OTHER knob at CENTER
        # while moving one, so a mixed some-low-others-high corner is unreachable BY
        # CONSTRUCTION, no matter how long you run. That is the exact blind spot that shipped
        # the tweed blowup (see this function's docstring) and, on 2026-09-04, sized Duke of
        # Tone's excitation 0.3-1.4% short at three Gain=lo,Volume=lo corners. Use max_corners
        # instead -- it keeps the structural corners AND fills from the hypercube, so it
        # degrades gracefully instead of going blind.
        print("WARNING: full_hypercube=False selects the structural-corner set only, which "
              "cannot represent MIXED some-knobs-low-others-high corners at all. Prefer "
              "max_corners=N (a budget) -- same cost, not blind. See _corners.__doc__.",
              file=sys.stderr)
        return corners

    if not names:
        return corners

    # NOTE two different bounds, deliberately: max_full_corners bounds the HYPERCUBE portion
    # (legacy semantics -- its "default 512, i.e. up to 9 knobs" doc counts 2**n only), while
    # max_corners bounds the TOTAL corner count including the structural ones. Conflating them
    # would silently drop 9-knob configs that work today.
    n_full = 2 ** len(names)
    seen = {tuple(sorted(v.items())) for _, v in corners}
    fits = (n_full <= max_full_corners) if max_corners is None else (n_full + len(corners) <= max_corners)

    def _add(bits):
        vals = {n: (hi[n] if b else lo[n]) for n, b in zip(names, bits)}
        key = tuple(sorted(vals.items()))
        if key in seen:
            return False
        seen.add(key)
        corners.append((",".join(f"{n}={'hi' if b else 'lo'}" for n, b in zip(names, bits)), vals))
        return True

    if fits:
        for bits in itertools.product((0, 1), repeat=len(names)):
            _add(bits)
        return corners
    budget = max_corners if max_corners is not None else max_full_corners
    room = max(0, budget - len(corners))

    # Over budget: SAMPLE the hypercube rather than abandon it. Deterministic (fixed seed, so
    # two runs of two different tools agree on the same corner set), and drawn as raw integers
    # so this stays cheap for a 16-knob device where materialising 2**16 patterns to shuffle
    # would be silly. Previously this raised and the caller's only out was full_hypercube=False
    # -- i.e. the guard against a too-big hypercube pushed you onto the structurally-blind set.
    if max_corners is None:
        raise ValueError(
            f"full hypercube would be {n_full} corners (> max_full_corners={max_full_corners}) "
            f"for {len(names)} knobs -- pass max_corners=N to sample it instead (recommended), "
            f"or raise max_full_corners explicitly. full_hypercube=False also works but is "
            f"structurally blind to mixed corners; see _corners.__doc__.")
    rng = random.Random(0xC0FFEE)
    tries = 0
    while len(corners) < budget and tries < room * 64:
        tries += 1
        _add([(rng.randrange(n_full) >> i) & 1 for i in range(len(names))])
    return corners


def interior_sample_budget(n_knobs: int) -> int:
    """How many FULL-GRID points to probe on top of the corner set, from the knob count.

    Corners are a heuristic and Mesa Dual Rectifier ORANGE showed they are not sufficient in
    principle: its highest saturation onset (23.177 V, at Bass=min with every OTHER knob at
    its CENTRE grid value) is 1.27x the highest of all 32 hypercube vertices. Onset is not
    monotonic in the knobs, so its maximum over the grid need not sit at a vertex.
    --sweep-peak-frac's 1.3 buys headroom against that; only probing the interior
    actually MEASURES it.

    Scaled at ~1.5x the corner count (2*n+3 structural + 2**n hypercube), because that is
    what the corner set already costs -- so sizing gets meaningfully better coverage for
    roughly 1.5x the probe time it was already paying, not an open-ended bill. Capped at 64:
    beyond that the marginal point buys little and a full amp's onset sweep is ~50s, so the
    cap is what keeps a 6-knob device's sizing from quietly turning into an hour.
    Probing every grid point WOULD guarantee it and is not worth it -- 648 points at a
    measured ~1.2/min is ~9h per channel, before every render, to choose one scalar.
    """
    if n_knobs <= 0:
        return 0          # no grid to sample; also the default when a caller omits n_knobs
    corners = 2 * n_knobs + 3 + 2 ** n_knobs
    return min(64, int(corners * 1.5))

def _sample_interior(knob_ranges: dict, corners: list, n_points: int) -> list:
    """Add n_points drawn from the FULL grid product, not just its min/max hypercube.

    Corners -- vertices, solos, all-min/all-max, center -- are a heuristic, and the Mesa
    Dual Rectifier proved on 2026-09-04 that they are not sufficient in principle: the
    ORANGE channel's highest saturation onset (23.177 V, at Bass=min with every other knob
    at its CENTRE grid value) is 27% above the highest of all 32 hypercube vertices
    (18.232 V). Onset is not monotonic in the knobs, so its maximum over the box need not
    sit at a vertex, and vertex enumeration cannot find it. The solo corners happened to
    catch that one; nothing guarantees the next circuit's maximum lies on a solo axis
    either.

    Probing the whole grid would guarantee it and is not worth it: 648 points at the
    measured ~1.2 points/min is ~9 hours per channel, before every render, to choose one
    scalar. A bounded deterministic sample gets most of the protection at a cost the caller
    picks. Deterministic (fixed seed) so a sizing run and a later checking run agree on the
    same points.
    """
    names = list(knob_ranges.keys())
    if not names or n_points <= 0:
        return corners
    seen = {tuple(sorted(v.items())) for _, v in corners}
    rng = random.Random(0x5EED)
    added = 0
    for _ in range(n_points * 64):
        if added >= n_points:
            break
        vals = {k: rng.choice(list(vs)) for k, vs in knob_ranges.items()}
        key = tuple(sorted(vals.items()))
        if key in seen:
            continue
        seen.add(key)
        corners.append((",".join(f"{k}={vals[k]:g}" for k in names), vals))
        added += 1
    return corners


def resolve_sample_grid(requested, knob_ranges):
    """--sample-grid, with None meaning AUTO (interior_sample_budget from the knob count).

    Defaulting this to 0 was a live footgun: scaffold_config.py passed a budget explicitly,
    so a scaffolded device got interior probing, while calling this tool BY HAND silently got
    corners only. That is how the Mesa Orange 2-knob grid came to be sized from 9 corners
    (worst onset 1.76 V) when an interior cell needed 11.18 V -- 6.3x -- and the transient
    gate then correctly refused the render. Corners are a heuristic; onset is not monotonic
    in the knobs, so the grid maximum need not sit at a vertex.

    Explicit 0 still disables it, for reproducing an older sizing exactly.
    """
    if requested is not None:
        return requested
    n = len(knob_ranges or {})
    return interior_sample_budget(n)


def _transient_peak_from_recipe(input_wav: Path) -> "float | None":
    recipe_path = input_wav.with_suffix(".recipe.json")
    if not recipe_path.exists():
        return None
    try:
        args = json.loads(recipe_path.read_text())["args"]
        # The TRANSIENT peak is not just the `--sweep-file` segment. build_excitation.py can
        # append transient BURSTS -- broadband, instant attack, exponential decay, crest ~8.5,
        # "matching a real hard pick-attack" in its own words -- at levels well above
        # --sweep-peak, precisely so sharp-attack behaviour is exercised at EVERY level
        # rather than only the loudest. Reading sweep_peak alone therefore UNDERSTATES what
        # the file actually contains, and fails corners the excitation genuinely covers.
        #
        # Found on the budget clone pedal: sweep_peak 7.4166 V against an all-min onset of 7.417 V failed
        # by a hair, while the file held 11.125 V and 14.833 V bursts (measured crest 13.6) that
        # clear it outright.
        #
        # "sweep_peak" is the arg's name from 2026-09-10 on (--real-clip/--realistic-peak were
        # renamed to --sweep-file/--sweep-peak to match TONE3000's own "sweep signal" term --
        # see build_excitation.py's docstring); "realistic_peak" is read as a fallback so a
        # recipe.json written before that rename still works.
        peaks = [float(args["sweep_peak"] if "sweep_peak" in args else args["realistic_peak"])]
        for key in ("synth_burst_peaks", "noise_burst_peak"):
            val = args.get(key)
            if isinstance(val, (list, tuple)):
                peaks += [float(v) for v in val]
            elif val is not None:
                peaks.append(float(val))
        return max(peaks)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _check_corners(backend, identity: bytes, cache_extra: str, knob_ranges: dict, fixed: dict,
                    transient_peak: float, label: str, margin: float = 1.0,
                    peak_max_v: float = 40.0, no_cache: bool = False, quiet: bool = False,
                    full_hypercube: "bool | None" = None, lead_silence_s: float = 0.0,
                    max_corners: "int | None" = None, sample_grid: int = 0,
                    capture: dict = None, workers: int = 8,
                    min_start_v: float = 1e-9, start_v: float = 0.005,
                    corner_workers: int = 1, shard: str = None, emit_onsets: str = None,
                    backend_name: str = "livespice") -> "dict | None":
    """Backend-agnostic core: every corner's own saturation onset (find_saturation_point.py)
    vs. the excitation's transient peak. Shared by check_coverage() (.schx/LiveSPICE) and
    check_coverage_ngspice_deck() (a hand-written ngspice deck with no .schx at all) -- the
    ONLY thing that differs between them is how `backend`/`identity`/`cache_extra` get built.

    Returns {"ok": bool, "rows": [...]}, or None when `emit_onsets` is set (this shard's rows
    are written to disk instead -- see prepare_excitation.py's --emit-onsets/--merge-onsets,
    the same convention, reused here via shard_corners()/merge_onset_shards()/solver_identity()
    imported from there). Unlike a worst-case-onset SIZING pass, a coverage CHECK needs no
    cross-shard reduction -- each corner's pass/fail depends only on its own onset, so a shard
    computes real, final "ok" values for its own corners; merge only verifies completeness/
    solver-agreement and concatenates, mirroring prepare_excitation.py's division of labor
    exactly except for that one simplification.

    CORNER-LEVEL PARALLELISM (2026-09-21, same pattern as prepare_excitation.py's
    worst_case_onset(), same reason it is structured this way): a single shared amplitude-level
    ThreadPoolExecutor is built ONCE here, in the main thread, before the outer corner-level
    ThreadPoolExecutor is created -- so no ThreadPoolExecutor is ever constructed from a worker
    thread, avoiding the reproduced concurrent.futures deadlock prepare_excitation.py's own
    --corner-workers hit before that fix (2026-09-19/20, see its commit history). Corners are
    independent (each corner's own adaptive range-extension never reads another corner's
    result), so this parallelises cleanly the same way.

    "ok" is False if ANY corner fails OR its onset couldn't be determined (a render failure is
    not a pass -- see main()'s same convention).
    """
    from prepare_excitation import shard_corners, solver_identity
    corners_all = _corners(knob_ranges, full_hypercube=full_hypercube, max_corners=max_corners, sample_grid=sample_grid)
    corners_all = _sample_interior(knob_ranges, corners_all, sample_grid)
    corner_total = len(corners_all)
    corners = list(enumerate(corners_all))   # [(global_index, (clabel, vals)), ...]
    if shard:
        corners = shard_corners(corners_all, shard)   # already [(global_index, corner), ...]
        if not quiet:
            print(f"  shard {shard}: {len(corners)} of {corner_total} corners")
    if not quiet:
        print(f"Transient saturation coverage: {label}")
        print(f"  transient peak = {transient_peak:.3f} V   {len(corners)} corners "
              f"({'structural-only (DEPRECATED)' if full_hypercube is False else                 f'budgeted, max {max_corners}' if max_corners is not None else                 f'full binary hypercube'} set)  "
              f"margin={margin}x\n")

    # Concurrency is a PRODUCT, not a sum -- same reasoning/formula as
    # prepare_excitation.py's worst_case_onset(): find_saturation_point already fans out its
    # own amplitude points (`workers`), so sweep workers divide down as corner_workers rises,
    # keeping the total near the core count instead of multiplying it.
    sweep_workers = max(1, workers // max(1, corner_workers))

    def _measure(idx_clabel_vals, tmp, amp_executor=None):
        idx, (clabel, vals) = idx_clabel_vals
        params = dict(vals); params.update(fixed)
        cpath = findpeak_cache_key(identity, params, cache_extra)
        if cpath.exists() and not no_cache:
            sat = json.loads(cpath.read_text())
        else:
            # Each corner's own saturation sweep is up to 20 real backend renders (see
            # find_saturation_point.py) -- on a stiff circuit, or with a full 2**n-corner
            # hypercube (up to 512 corners), that's real minutes-to-hours of work with
            # nothing printed between corners otherwise. One line per corner as it STARTS
            # (not per-amplitude within it -- that's for a one-shot caller like preflight.py
            # --find-peak; here it would mean thousands of lines across a full hypercube).
            if not quiet:
                print(f"  [{clabel}] rendering ...", flush=True)
            sat = find_saturation_point(backend, params, str(tmp), max_v=peak_max_v,
                                         lead_silence_s=lead_silence_s, capture=capture,
                                         workers=sweep_workers, min_start_v=min_start_v,
                                         start_v=start_v, executor=amp_executor)
            cache_findpeak(cpath, sat)
        # FIXED (2026-09-21): this block used to sit inside the `else:` above, so a CACHE HIT
        # loaded `sat` and then silently dropped the corner -- no row, no print, not counted
        # as failed/skipped/passed, just missing. Only bit --no-cache runs (this session's own
        # AC30 checks) since caching was never exercised on a mid-restructure run, but it is a
        # real latent bug independent of everything else changed here.
        onset = sat.get("onset_99pct_input_v") if sat else None
        if onset is None:
            # Two very different situations were previously collapsed into one vague
            # message. `sat is None` means EVERY amplitude in the sweep failed to render
            # -- see stderr (each failure now names its own cause: an application
            # exception, or "KILLED BY SIGNAL ..." for an OS-level kill, almost always
            # memory pressure under parallel load rather than a circuit problem -- check
            # `vm_stat` and whether something else is training/rendering concurrently
            # before assuming the circuit itself is broken). `sat` present but `onset`
            # None means renders WORKED but the swept amplitude range (start_v..max_v)
            # never bracketed 99% of the ceiling -- a sweep-range tuning issue, not a
            # render failure at all.
            if sat is None:
                status = "SKIP (every render in the sweep failed -- see stderr for why)"
            else:
                status = "SKIP (renders OK, but sweep range never bracketed the onset)"
            ok = None
        else:
            ok = transient_peak >= onset * margin
            status = "OK" if ok else "FAIL -- transient never reaches saturation here"
        if not quiet:
            onset_str = "NONE" if onset is None else f"{onset:.3f} V"
            print(f"  {clabel:16} onset={onset_str:>10}  {status}")
        return {"corner": clabel, "params": params, "onset_v": onset, "ok": ok, "index": idx}

    results = [None] * len(corners)
    if emit_onsets:
        # SHARDED MODE: measure only this slice, compute each corner's OWN final "ok" (a
        # pass/fail check needs no cross-corner reduction, unlike a worst-case-onset SIZING
        # pass -- each corner's verdict depends only on its own onset), and write it out.
        # merge_onsets combines shards, verifying completeness/solver-agreement.
        with tempfile.TemporaryDirectory() as scratch:
            if corner_workers > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                amp_pool_size = max(1, corner_workers * sweep_workers)
                with ThreadPoolExecutor(max_workers=amp_pool_size) as amp_ex, \
                     ThreadPoolExecutor(max_workers=corner_workers) as ex:
                    futs = {ex.submit(_measure, ic, Path(scratch) / f"corner_{n:04d}", amp_ex): n
                            for n, ic in enumerate(corners)}
                    for fut in as_completed(futs):
                        results[futs[fut]] = fut.result()
            else:
                for n, ic in enumerate(corners):
                    results[n] = _measure(ic, scratch)
        Path(emit_onsets).write_text(json.dumps({
            "corner_total": corner_total, "solver": solver_identity(backend_name),
            "shard": shard, "rows": results,
        }))
        print(f"wrote {len(results)} row(s) to {emit_onsets} (solver "
              f"{solver_identity(backend_name)}) -- NOT summarized; merge with --merge-onsets "
              f"to report once across every shard")
        return None

    with tempfile.TemporaryDirectory() as scratch:
        if corner_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            amp_pool_size = max(1, corner_workers * sweep_workers)
            with ThreadPoolExecutor(max_workers=amp_pool_size) as amp_ex, \
                 ThreadPoolExecutor(max_workers=corner_workers) as ex:
                futs = {ex.submit(_measure, ic, Path(scratch) / f"corner_{n:04d}", amp_ex): n
                        for n, ic in enumerate(corners)}
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
        else:
            for n, ic in enumerate(corners):
                results[n] = _measure(ic, scratch)
    rows = results

    failed = [r for r in rows if r["ok"] is False]
    skipped = [r for r in rows if r["ok"] is None]
    if not quiet:
        print()
        if failed:
            print(f"FAILED: {len(failed)}/{len(rows)} corners never see a transient past their own "
                  f"saturation onset -- the model can go out-of-distribution there on real playing. "
                  f"Raise --sweep-peak (build_excitation.py) past the highest FAILED onset, or "
                  f"lengthen --sweep-dur for more varied transient shapes, and rebuild.")
        if skipped:
            print(f"WARNING: {len(skipped)}/{len(rows)} corners' onset could not be determined "
                  f"(render failures or onset above --peak-max-v) -- treat as unverified, not passing.")
        if not failed and not skipped:
            print(f"PASSED: transient content reaches saturation at every checked corner.")

    return {"ok": not (failed or skipped), "rows": rows}


def check_coverage(schx: str, knob_ranges: dict, fixed: dict, oversample: int,
                   transient_peak: float, margin: float = 1.0, iterations: int = 256,
                   peak_max_v: float = 40.0, no_cache: bool = False, quiet: bool = False,
                   full_hypercube: "bool | None" = None, max_corners: "int | None" = None,
                   sample_grid: int = 0, capture: dict = None, workers: int = 8,
                   min_start_v: float = 1e-9, start_v: float = 0.005,
                   corner_workers: int = 1, shard: str = None, emit_onsets: str = None) -> "dict | None":
    """[.schx / LiveSPICE path] Importable directly (gen_dataset_from_schx.py's hard gate uses
    this in-process -- no subprocess, no re-parsing a config, and it can't be silently skipped
    by someone calling gen_dataset_from_schx.py without going through run_pipeline.py / this
    tool's own CLI first). See _check_corners() for the actual check."""
    backend = LiveSpiceBackend(schx, oversample=oversample, iterations=iterations)
    identity = Path(schx).read_bytes()
    cache_extra = (f"os={oversample}|it={iterations}|maxv={peak_max_v}|minv={min_start_v}"
                   f"|startv={start_v}") + cache_tag(capture)
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           capture=capture, min_start_v=min_start_v, start_v=start_v,
                           label=Path(schx).name, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube,
                           max_corners=max_corners, sample_grid=sample_grid, workers=workers,
                           corner_workers=corner_workers, shard=shard, emit_onsets=emit_onsets,
                           backend_name="livespice")


def check_coverage_ngspice(schx: str, knob_ranges: dict, fixed: dict, oversample: int,
                          transient_peak: float, margin: float = 1.0,
                          peak_max_v: float = 40.0, no_cache: bool = False, quiet: bool = False,
                          full_hypercube: "bool | None" = None, max_corners: "int | None" = None,
                          sample_grid: int = 0, capture: dict = None, conv: dict = None,
                          min_start_v: float = 1e-9, start_v: float = 0.005,
                          corner_workers: int = 1, shard: str = None,
                          emit_onsets: str = None) -> "dict | None":
    """[.schx / GENERIC ngspice path, i.e. --backend "ngspice"] For a circuit whose .schx
    exists but whose LiveSPICE render diverges under real signal (e.g. Arbiter Fuzz Face's
    tight DC-coupled feedback loop) yet needs no hand-written deck at all -- unlike
    check_coverage_ngspice_deck below, which is for a device with NO .schx counterpart.

    Before this, main()'s backend_name dispatch had NO branch for "ngspice" at all: it fell
    through to check_coverage() below, which hardcodes LiveSpiceBackend -- silently checking
    transient coverage against the WRONG solver's render for exactly the circuits that need
    this tool most (LiveSPICE diverges on them). Caught 2026-09-11 while wiring generic-
    ngspice support into prepare_excitation.py for Arbiter Fuzz Face.
    """
    conv = conv or {}
    backend = NgspiceSchxBackend(schx, oversample=oversample, conv=conv)
    identity = Path(schx).read_bytes()
    cache_extra = (f"backend=ngspice|os={oversample}|maxv={peak_max_v}|minv={min_start_v}"
                  f"|startv={start_v}" + cache_tag(capture) + conv_cache_tag(conv))
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           capture=capture, min_start_v=min_start_v, start_v=start_v,
                           label=Path(schx).name, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube,
                           max_corners=max_corners, sample_grid=sample_grid,
                           corner_workers=corner_workers, shard=shard, emit_onsets=emit_onsets,
                           backend_name="ngspice")


def check_coverage_ngspice_deck(build_deck, module_file: str, probe_node: str, knob_ranges: dict,
                                fixed: dict, transient_peak: float, margin: float = 1.0,
                                maxstep: float = 3e-6, parallel_sims: int = 8,
                                peak_max_v: float = 40.0, no_cache: bool = False,
                                quiet: bool = False, full_hypercube: "bool | None" = None,
                                max_corners: "int | None" = None, sample_grid: int = 0,
                                lead_silence_s: float = 3.0, capture: dict = None,
                                min_start_v: float = 1e-9, start_v: float = 0.005,
                                corner_workers: int = 1, shard: str = None,
                                emit_onsets: str = None) -> "dict | None":
    """[hand-written ngspice-deck path] For a device whose real component (a MOSFET, a real
    BJT) has no .schx model at all -- see render_backends.py's NgspiceBackend and
    preflight.py/prepare_excitation.py's identical --backend ngspice-deck split. `module_file`
    is the gen_*_ngspice.py module's own `__file__` (its source bytes are the cache identity,
    same convention as preflight.py's _build_backend -- an edited deck re-checks automatically).
    See _check_corners() for the actual check."""
    backend = NgspiceBackend(build_deck, probe_node=probe_node, maxstep=maxstep,
                             parallel_sims=parallel_sims)
    identity = Path(module_file).read_bytes()
    # The BACKEND NAME and maxstep must be in the key. Without the name, ngspice-deck and
    # ltspice-deck produced an IDENTICAL key -- same identity (the generator module's bytes)
    # and the same "maxv=..." extra -- so one simulator's onset was served to the other. That
    # is not a corner case: docs/backends.md says ltspice-deck exists for a device whose
    # ngspice deck cannot converge, i.e. THE SAME MODULE through both. Without maxstep, a
    # sweep from 3e-6 down to 3e-8 -- which that same doc describes doing -- gets the first
    # value's answers back for every step after it.
    #
    # The livespice extra is deliberately left alone: it carries "os=..|it=.." which no deck
    # backend emits, so it cannot collide with either, and changing it would invalidate every
    # cached entry in the fleet to fix a bug it does not have.
    cache_extra = (f"backend=ngspice-deck|maxstep={maxstep}|maxv={peak_max_v}"
                   f"|minv={min_start_v}|startv={start_v}") + cache_tag(capture)
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           capture=capture, min_start_v=min_start_v, start_v=start_v,
                           label=Path(module_file).stem, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube,
                           max_corners=max_corners, sample_grid=sample_grid, lead_silence_s=lead_silence_s,
                           corner_workers=corner_workers, shard=shard, emit_onsets=emit_onsets,
                           backend_name="ngspice-deck")


def check_coverage_ltspice_deck(build_deck, module_file: str, tap: str, knob_ranges: dict,
                                fixed: dict, transient_peak: float, margin: float = 1.0,
                                maxstep: float = 3e-6, parallel_sims: int = 8,
                                out_scale: float = 0.05, peak_max_v: float = 40.0,
                                no_cache: bool = False, quiet: bool = False,
                                full_hypercube: "bool | None" = None, max_corners: "int | None" = None,
                                sample_grid: int = 0, capture: dict = None,
                                min_start_v: float = 1e-9, start_v: float = 0.005,
                                corner_workers: int = 1, shard: str = None,
                                emit_onsets: str = None) -> "dict | None":
    """[hand-written LTspice-deck path] For a device whose ngspice-deck counterpart can't
    converge on real playing content at all -- see ltspice_spicelib.py's own docstring.
    `module_file` is the gen_*_ltspice.py module's own `__file__` (its source bytes are the
    cache identity, same convention as check_coverage_ngspice_deck). See _check_corners() for
    the actual check. No lead_silence_s here (unlike ngspice-deck): LTspice's `.ic`/`uic`
    initial-condition hints replace the need for a cold-start settling lead-in -- see
    ltspice_spicelib.py's docstring."""
    backend = LtspiceBackend(build_deck, tap=tap, maxstep=maxstep, parallel_sims=parallel_sims,
                             out_scale=out_scale)
    identity = Path(module_file).read_bytes()
    cache_extra = (f"backend=ltspice-deck|maxstep={maxstep}|maxv={peak_max_v}"
                   f"|minv={min_start_v}|startv={start_v}") + cache_tag(capture)
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           capture=capture, min_start_v=min_start_v, start_v=start_v,
                           label=Path(module_file).stem, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube,
                           max_corners=max_corners, sample_grid=sample_grid,
                           corner_workers=corner_workers, shard=shard, emit_onsets=emit_onsets,
                           backend_name="ltspice-deck")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="per-circuit TOML (same as run_pipeline.py --config)")
    ap.add_argument("--transient-peak", type=float, default=None,
                    help="excitation's transient-segment peak, volts (auto from "
                         "<input>.recipe.json if omitted)")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="require transient-peak >= onset * margin (default 1.0 = must just "
                         "reach onset; >1.0 demands headroom past it)")
    ap.add_argument("--oversample", type=int, default=None, help="default: config's own")
    ap.add_argument("--iterations", type=int, default=256)
    ap.add_argument("--peak-max-v", type=float, default=40.0)
    ap.add_argument("--sweep-start-v", type=float, default=0.005,
                     help="find_saturation_point's initial sweep floor (default: %(default)s). "
                          "Raise for a circuit with an ACTIVE internal supply (an AC-driven "
                          "sag/rectifier network) whose own ripple floor sits above the "
                          "default -- otherwise every corner's onset re-check here silently "
                          "re-derives the same near-0V floor artifact prepare_excitation.py's "
                          "own --sweep-start-v/--min-start-v exist to avoid, making this tool's "
                          "'OK' verdicts on those corners a false pass (any nonzero transient "
                          "trivially clears a spurious near-0V onset). MUST match whatever "
                          "values were used to size the excitation being checked, or this is "
                          "checking coverage against a different definition of onset than the "
                          "one that built it. See prepare_excitation.py's own --min-start-v "
                          "help for the full story (the Vox AC30 Top Boost sag-ac case).")
    ap.add_argument("--min-start-v", type=float, default=1e-9,
                     help="find_saturation_point's downward-extension floor (default: "
                          "%(default)s). See --sweep-start-v above -- both are needed together "
                          "and must match the values used to size the excitation being checked.")
    # EXPOSED BECAUSE THE FAILURE MESSAGE ALREADY TELLS YOU TO USE IT. A render killed by the OS
    # reports "KILLED BY SIGNAL SIGABRT ... Lower --workers before assuming the circuit is
    # broken" -- advice this tool could not take, having no such flag. Hit on Mesa RED with tube
    # capacitance enabled (2026-09-14): capacitance adds state per triode, 8 concurrent sweeps
    # exhausted memory, and all 25 corners died with the message recommending a flag that did
    # not exist. The default matches find_saturation_point's own.
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent renders within one corner's amplitude sweep. Lower it if "
                         "renders are killed by the OS (SIGABRT/SIGKILL) -- a memory-pressure "
                         "symptom, not a circuit fault.")
    ap.add_argument("--conv", default=None,
                    help="[ngspice] device-model convergence/fidelity overrides key=val,... "
                         "(same format gen_dataset_from_schx.py --conv uses; e.g. "
                         "bjt_vaf=102.207,bjt_rb=173.312 for a real datasheet-fitted "
                         "transistor). Default: --config's own `conv` field.")
    _cc_add_cli_args(ap)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--sample-grid", type=int, default=None, metavar="N",
                    help="AUTO by default (interior_sample_budget: ~1.5x the corner "
                          "count, capped 64) -- 0 disables. Corners are a HEURISTIC and have "
                          "been shown insufficient twice: Mesa Orange's 5-knob grid had its "
                          "true worst onset 1.27x above every one of 32 vertices, and its "
                          "2-knob Gain x Master grid had an interior cell at 11.18 V against "
                          "a worst CORNER of 1.76 V -- 6.3x. Sizing from corners alone there "
                          "produced an excitation that could not drive 2 of 25 probed cells "
                          "into saturation at all, and the transient gate correctly refused "
                          "the render (2026-09-12). Onset is not monotonic in the knobs, so "
                          "its maximum over the grid need not sit at a vertex; only probing "
                          "the interior MEASURES it. Deterministic, so a sizing run and a "
                          "later coverage check agree.")
    ap.add_argument("--max-corners", type=int, default=None,
                     help="cap the TOTAL corner count, deterministically sampling the hypercube "
                          "when it does not all fit, instead of abandoning it. Prefer this to "
                          "--no-full-hypercube on a many-knob device: same budget, still reaches "
                          "mixed low/high corners (16 knobs at --max-corners 48 gets 13; "
                          "--no-full-hypercube gets 0).")
    ap.add_argument("--no-full-hypercube", action="store_true",
                    help="skip the full 2**n min/max hypercube (only the solo + all-min/"
                         "all-max/center corners) -- use for a config with many knobs where "
                         "2**n would be impractically large")
    ap.add_argument("--lead-silence-s", type=float, default=3.0,
                    help="[ngspice-deck] silence prepended before each saturation-sweep probe "
                         "tone -- see this repo's README ('Known issue: excitation needs a "
                         "silent lead-in')")
    ap.add_argument("--maxstep", type=float, default=3e-6,
                    help="[ngspice-deck, ltspice-deck] timestep ceiling")
    ap.add_argument("--out-scale", type=float, default=0.05,
                    help="[ltspice-deck] LTspice .wave output is +/-1V-PCM-bounded -- see "
                         "ltspice_spicelib.py's docstring")
    ap.add_argument("--corner-workers", type=int, default=1,
                    help="measure this many knob corners concurrently (default: 1, serial -- "
                         "this tool had NO corner-level parallelism at all until 2026-09-21). "
                         "Same safe shared-executor pattern as prepare_excitation.py's own "
                         "--corner-workers (see _check_corners()'s docstring): concurrency is "
                         "a PRODUCT with --workers, which divides down as this rises to keep "
                         "the total near the core count.")
    ap.add_argument("--shard", metavar="LOW-HIGH/TOTAL", default=None,
                    help="measure only the corners whose index modulo TOTAL falls in "
                         "[LOW, HIGH] -- shard.py's shared contract, same as "
                         "prepare_excitation.py/gen_dataset_from_schx.py. Requires "
                         "--emit-onsets. Unlike a worst-case-onset SIZING pass, a coverage "
                         "CHECK needs no cross-shard reduction (each corner's pass/fail "
                         "depends only on its own onset) -- --merge-onsets just verifies "
                         "completeness/solver-agreement and reports the union.")
    ap.add_argument("--emit-onsets", metavar="PATH", default=None,
                    help="write this shard's measured rows (each already carrying its own "
                         "final pass/fail) as JSON instead of printing a summary. Carries the "
                         "corner total and a fingerprint of the renderer binary so "
                         "--merge-onsets can refuse mismatched work.")
    ap.add_argument("--merge-onsets", metavar="PATH", nargs="+", default=None,
                    help="combine --emit-onsets files from every shard, verify completeness "
                         "and solver agreement, then print the combined coverage report once.")
    args = ap.parse_args()

    if args.merge_onsets:
        from prepare_excitation import merge_onset_shards
        rows = merge_onset_shards(args.merge_onsets)
        failed = [r for r in rows if r["ok"] is False]
        skipped = [r for r in rows if r["ok"] is None]
        for r in rows:
            onset_str = "NONE" if r["onset_v"] is None else f"{r['onset_v']:.3f} V"
            status = ("OK" if r["ok"] else "FAIL -- transient never reaches saturation here"
                      if r["ok"] is False else "SKIP (see the shard's own stderr for why)")
            print(f"  {r['corner']:16} onset={onset_str:>10}  {status}")
        print()
        if failed:
            print(f"FAILED: {len(failed)}/{len(rows)} corners never see a transient past their own "
                  f"saturation onset -- the model can go out-of-distribution there on real playing. "
                  f"Raise --sweep-peak (build_excitation.py) past the highest FAILED onset, or "
                  f"lengthen --sweep-dur for more varied transient shapes, and rebuild.")
        if skipped:
            print(f"WARNING: {len(skipped)}/{len(rows)} corners' onset could not be determined "
                  f"(render failures or onset above --peak-max-v) -- treat as unverified, not passing.")
        if not failed and not skipped:
            print("PASSED: transient content reaches saturation at every checked corner.")
        sys.exit(0 if not (failed or skipped) else 1)

    cfg = load_config(Path(args.config))
    _capture = _cc_resolve(args, cfg)
    backend_name = cfg.get("backend", "livespice")
    input_wav = Path(cfg["input"]).expanduser()

    transient_peak = args.transient_peak
    if transient_peak is None:
        transient_peak = _transient_peak_from_recipe(input_wav)
    if transient_peak is None:
        ap.error(f"--transient-peak not given and no {input_wav.with_suffix('.recipe.json').name} "
                 f"sidecar found -- refusing to guess it from the raw audio. Pass it explicitly "
                 f"(the value used for build_excitation.py's --sweep-peak).")

    knob_ranges = {}
    for entry in cfg.get("ranges", []):
        name, vals = entry.split("=", 1)
        knob_ranges[name.strip()] = [float(v) for v in vals.split(",")]
    if not knob_ranges:
        ap.error("config has no [knobs] table -- nothing to check corners over")

    fixed = {}
    for kv in filter(None, (s.strip() for s in (cfg.get("fixed_params") or "").split(","))):
        k, v = kv.split("="); fixed[k.strip()] = float(v)

    print(f"  excitation: {input_wav.name}")
    if backend_name == "ngspice-deck":
        pedal_dir = os.path.expanduser(cfg["pedal_dir"])
        module = cfg["module"]
        probe_node = cfg.get("probe_node", "OUT")
        sys.path.insert(0, os.path.abspath(pedal_dir))
        mod = importlib.import_module(module)
        result = check_coverage_ngspice_deck(mod.build_deck, mod.__file__, probe_node,
                                             knob_ranges, fixed, transient_peak,
                                             margin=args.margin, maxstep=args.maxstep,
                                             peak_max_v=args.peak_max_v,
                                             min_start_v=args.min_start_v,
                                             start_v=args.sweep_start_v,
                                             no_cache=args.no_cache,
                                             full_hypercube=(False if args.no_full_hypercube else None),
                                             max_corners=args.max_corners,
                                             sample_grid=resolve_sample_grid(args.sample_grid, knob_ranges),
                                             capture=_capture,
                                             lead_silence_s=args.lead_silence_s,
                                             corner_workers=args.corner_workers,
                                             shard=args.shard, emit_onsets=args.emit_onsets)
        schx_or_module = module
    elif backend_name == "ltspice-deck":
        pedal_dir = os.path.expanduser(cfg["pedal_dir"])
        module = cfg["module"]
        tap = cfg.get("probe_node", "OUT")
        sys.path.insert(0, os.path.abspath(pedal_dir))
        mod = importlib.import_module(module)
        result = check_coverage_ltspice_deck(mod.build_deck, mod.__file__, tap,
                                             knob_ranges, fixed, transient_peak,
                                             margin=args.margin, maxstep=args.maxstep,
                                             out_scale=args.out_scale,
                                             peak_max_v=args.peak_max_v,
                                             min_start_v=args.min_start_v,
                                             start_v=args.sweep_start_v,
                                             no_cache=args.no_cache,
                                             full_hypercube=(False if args.no_full_hypercube else None),
                                             max_corners=args.max_corners,
                                             sample_grid=resolve_sample_grid(args.sample_grid, knob_ranges),
                                             capture=_capture,
                                             corner_workers=args.corner_workers,
                                             shard=args.shard, emit_onsets=args.emit_onsets)
        schx_or_module = module
    elif backend_name == "ngspice":
        # Previously fell into the `else` branch below, which hardcodes LiveSpiceBackend --
        # silently checking coverage against the wrong solver for exactly the circuits that
        # need "ngspice" (LiveSPICE diverges on them). See check_coverage_ngspice()'s docstring.
        schx = str(cfg["schx"])
        oversample = args.oversample or cfg.get("oversample", 2)
        _conv = parse_conv(args.conv if args.conv is not None else cfg.get("conv"))
        result = check_coverage_ngspice(schx, knob_ranges, fixed, oversample, transient_peak,
                                        margin=args.margin,
                                        peak_max_v=args.peak_max_v, no_cache=args.no_cache,
                                        min_start_v=args.min_start_v, start_v=args.sweep_start_v,
                                        full_hypercube=(False if args.no_full_hypercube else None),
                                        max_corners=args.max_corners,
                                        sample_grid=resolve_sample_grid(args.sample_grid, knob_ranges),
                                        capture=_capture, conv=_conv,
                                        corner_workers=args.corner_workers,
                                        shard=args.shard, emit_onsets=args.emit_onsets)
        schx_or_module = schx
    else:
        schx = str(cfg["schx"])
        oversample = args.oversample or cfg.get("oversample", 8)
        result = check_coverage(schx, knob_ranges, fixed, oversample, transient_peak,
                                margin=args.margin, iterations=args.iterations,
                                peak_max_v=args.peak_max_v, no_cache=args.no_cache,
                                min_start_v=args.min_start_v, start_v=args.sweep_start_v,
                                full_hypercube=(False if args.no_full_hypercube else None),
                                max_corners=args.max_corners,
                                sample_grid=resolve_sample_grid(args.sample_grid, knob_ranges),
                                capture=_capture, workers=args.workers,
                                corner_workers=args.corner_workers,
                                shard=args.shard, emit_onsets=args.emit_onsets)
        schx_or_module = schx

    if result is None:
        return 0

    if args.json:
        Path(args.json).write_text(json.dumps({
            "schx": schx_or_module, "input": str(input_wav), "transient_peak_v": transient_peak,
            "margin": args.margin, "corners": result["rows"],
        }, indent=2))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
