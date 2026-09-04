#!/usr/bin/env python3
"""Pre-generation gate: does the excitation's TRANSIENT content actually reach
saturation at every knob-grid corner, not just the excitation's overall peak?

Background (internal engineering notes): a device can have a peak level
(from its synthetic sweep tail) that comfortably clears its saturation onset, while
the TRANSIENT-bearing (real-playing) segment -- built independently, at its own
`--realistic-peak` -- never does, at some corners. The tweed-style amp is the exact case
this happened on: `find-peak @ knobs=0.5` measured onset ~0.51 V, but
`--realistic-peak` was set to 0.2 V regardless (chosen for input-signal realism,
not cross-checked against the measured onset) -- so the real-playing content
stayed in the LINEAR region at every corner tested, while only the sweep (a smooth
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

--transient-peak: the excitation's transient/real-playing segment peak, in volts
  at V0dBFS=1. Auto-read from the excitation's <stem>.recipe.json sidecar
  (build_excitation.py's `args.realistic_peak`) if present; otherwise REQUIRED --
  this tool refuses to guess it from the raw audio (silently mis-slicing the
  file's realistic/sweep boundary would be worse than refusing to run).
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
from find_saturation_point import find_saturation_point, findpeak_cache_key  # noqa: E402
from render_backends import LiveSpiceBackend, NgspiceBackend, LtspiceBackend  # noqa: E402

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
    --realistic-peak-frac's 1.3 buys headroom against that; only probing the interior
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


def _transient_peak_from_recipe(input_wav: Path) -> "float | None":
    recipe_path = input_wav.with_suffix(".recipe.json")
    if not recipe_path.exists():
        return None
    try:
        args = json.loads(recipe_path.read_text())["args"]
        # The TRANSIENT peak is not just the real-playing segment. build_excitation.py can
        # append transient BURSTS -- broadband, instant attack, exponential decay, crest ~8.5,
        # "matching a real hard pick-attack" in its own words -- at levels well above
        # --realistic-peak, precisely so sharp-attack behaviour is exercised at EVERY level
        # rather than only the loudest. Reading realistic_peak alone therefore UNDERSTATES what
        # the file actually contains, and fails corners the excitation genuinely covers.
        #
        # Found on the budget clone pedal: realistic_peak 7.4166 V against an all-min onset of 7.417 V failed
        # by a hair, while the file held 11.125 V and 14.833 V bursts (measured crest 13.6) that
        # clear it outright.
        peaks = [float(args["realistic_peak"])]
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
                    max_corners: "int | None" = None, sample_grid: int = 0) -> dict:
    """Backend-agnostic core: every corner's own saturation onset (find_saturation_point.py)
    vs. the excitation's transient peak. Shared by check_coverage() (.schx/LiveSPICE) and
    check_coverage_ngspice_deck() (a hand-written ngspice deck with no .schx at all) -- the
    ONLY thing that differs between them is how `backend`/`identity`/`cache_extra` get built.

    Returns {"ok": bool, "rows": [...]}. "ok" is False if ANY corner fails OR its onset
    couldn't be determined (a render failure is not a pass -- see main()'s same convention).
    """
    corners = _corners(knob_ranges, full_hypercube=full_hypercube, max_corners=max_corners, sample_grid=sample_grid)
    corners = _sample_interior(knob_ranges, corners, sample_grid)
    if not quiet:
        print(f"Transient saturation coverage: {label}")
        print(f"  transient peak = {transient_peak:.3f} V   {len(corners)} corners "
              f"({'structural-only (DEPRECATED)' if full_hypercube is False else                 f'budgeted, max {max_corners}' if max_corners is not None else                 f'full binary hypercube'} set)  "
              f"margin={margin}x\n")

    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for i, (clabel, vals) in enumerate(corners, 1):
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
                    print(f"  [{i}/{len(corners)}] {clabel} — rendering ...", flush=True)
                sat = find_saturation_point(backend, params, scratch, max_v=peak_max_v,
                                             lead_silence_s=lead_silence_s)
                if sat is not None:
                    cpath.write_text(json.dumps(sat))
            onset = sat.get("onset_99pct_input_v") if sat else None
            if onset is None:
                status = "SKIP (onset not bracketed / render failed)"
                ok = None
            else:
                ok = transient_peak >= onset * margin
                status = "OK" if ok else "FAIL -- transient never reaches saturation here"
            if not quiet:
                onset_str = "NONE" if onset is None else f"{onset:.3f} V"
                print(f"  {clabel:16} onset={onset_str:>10}  {status}")
            rows.append({"corner": clabel, "params": params, "onset_v": onset, "ok": ok})

    failed = [r for r in rows if r["ok"] is False]
    skipped = [r for r in rows if r["ok"] is None]
    if not quiet:
        print()
        if failed:
            print(f"FAILED: {len(failed)}/{len(rows)} corners never see a transient past their own "
                  f"saturation onset -- the model can go out-of-distribution there on real playing. "
                  f"Raise --realistic-peak (build_excitation.py) past the highest FAILED onset, or "
                  f"lengthen --realistic-dur for more varied transient shapes, and rebuild.")
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
                   sample_grid: int = 0) -> dict:
    """[.schx / LiveSPICE path] Importable directly (gen_dataset_from_schx.py's hard gate uses
    this in-process -- no subprocess, no re-parsing a config, and it can't be silently skipped
    by someone calling gen_dataset_from_schx.py without going through run_pipeline.py / this
    tool's own CLI first). See _check_corners() for the actual check."""
    backend = LiveSpiceBackend(schx, oversample=oversample, iterations=iterations)
    identity = Path(schx).read_bytes()
    cache_extra = f"os={oversample}|it={iterations}|maxv={peak_max_v}"
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           label=Path(schx).name, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube,
                           max_corners=max_corners, sample_grid=sample_grid)


def check_coverage_ngspice_deck(build_deck, module_file: str, probe_node: str, knob_ranges: dict,
                                fixed: dict, transient_peak: float, margin: float = 1.0,
                                maxstep: float = 3e-6, parallel_sims: int = 8,
                                peak_max_v: float = 40.0, no_cache: bool = False,
                                quiet: bool = False, full_hypercube: "bool | None" = None,
                                max_corners: "int | None" = None, sample_grid: int = 0,
                                lead_silence_s: float = 3.0) -> dict:
    """[hand-written ngspice-deck path] For a device whose real component (a MOSFET, a real
    BJT) has no .schx model at all -- see render_backends.py's NgspiceBackend and
    preflight.py/prepare_excitation.py's identical --backend ngspice-deck split. `module_file`
    is the gen_*_ngspice.py module's own `__file__` (its source bytes are the cache identity,
    same convention as preflight.py's _build_backend -- an edited deck re-checks automatically).
    See _check_corners() for the actual check."""
    backend = NgspiceBackend(build_deck, probe_node=probe_node, maxstep=maxstep,
                             parallel_sims=parallel_sims)
    identity = Path(module_file).read_bytes()
    cache_extra = f"maxv={peak_max_v}"
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           label=Path(module_file).stem, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube,
                           max_corners=max_corners, sample_grid=sample_grid, lead_silence_s=lead_silence_s)


def check_coverage_ltspice_deck(build_deck, module_file: str, tap: str, knob_ranges: dict,
                                fixed: dict, transient_peak: float, margin: float = 1.0,
                                maxstep: float = 3e-6, parallel_sims: int = 8,
                                out_scale: float = 0.05, peak_max_v: float = 40.0,
                                no_cache: bool = False, quiet: bool = False,
                                full_hypercube: "bool | None" = None, max_corners: "int | None" = None,
                   sample_grid: int = 0) -> dict:
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
    cache_extra = f"maxv={peak_max_v}"
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           label=Path(module_file).stem, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube,
                           max_corners=max_corners, sample_grid=sample_grid)


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
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--sample-grid", type=int, default=0, metavar="N",
                    help="ALSO probe N points drawn from the full grid product, not just the "
                         "min/max hypercube. Corners are a heuristic: Mesa Orange's highest "
                         "onset sits at Bass=min with every other knob CENTRED, 27%% above the "
                         "best of all 32 vertices, because onset is not monotonic in the knobs. "
                         "Deterministic, so a sizing run and a later check agree. Costs about "
                         "one render-sweep per point -- budget it, do not probe the whole grid "
                         "(648 points is ~9h on a full amp, for one scalar).")
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
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    backend_name = cfg.get("backend", "livespice")
    input_wav = Path(cfg["input"]).expanduser()

    transient_peak = args.transient_peak
    if transient_peak is None:
        transient_peak = _transient_peak_from_recipe(input_wav)
    if transient_peak is None:
        ap.error(f"--transient-peak not given and no {input_wav.with_suffix('.recipe.json').name} "
                 f"sidecar found -- refusing to guess it from the raw audio. Pass it explicitly "
                 f"(the value used for build_excitation.py's --realistic-peak).")

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
                                             no_cache=args.no_cache,
                                             full_hypercube=(False if args.no_full_hypercube else None),
                                             max_corners=args.max_corners,
                                             sample_grid=args.sample_grid,
                                             lead_silence_s=args.lead_silence_s)
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
                                             no_cache=args.no_cache,
                                             full_hypercube=(False if args.no_full_hypercube else None),
                                             max_corners=args.max_corners,
                                             sample_grid=args.sample_grid)
        schx_or_module = module
    else:
        schx = str(cfg["schx"])
        oversample = args.oversample or cfg.get("oversample", 8)
        result = check_coverage(schx, knob_ranges, fixed, oversample, transient_peak,
                                margin=args.margin, iterations=args.iterations,
                                peak_max_v=args.peak_max_v, no_cache=args.no_cache,
                                full_hypercube=(False if args.no_full_hypercube else None),
                                max_corners=args.max_corners,
                                sample_grid=args.sample_grid)
        schx_or_module = schx

    if args.json:
        Path(args.json).write_text(json.dumps({
            "schx": schx_or_module, "input": str(input_wav), "transient_peak_v": transient_peak,
            "margin": args.margin, "corners": result["rows"],
        }, indent=2))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
