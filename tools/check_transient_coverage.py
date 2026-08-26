#!/usr/bin/env python3
"""Pre-generation gate: does the excitation's TRANSIENT content actually reach
saturation at every knob-grid corner, not just the excitation's overall peak?

Background (internal engineering notes): a device can have a peak level
(from its synthetic sweep tail) that comfortably clears its saturation onset, while
the TRANSIENT-bearing (real-playing) segment -- built independently, at its own
`--realistic-peak` -- never does, at some corners. Tweed 5F6-A is the exact case
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
  python tools/check_transient_coverage.py ... && python gen_dataset_from_schx.py ...

Usage:
  livespice:   python tools/check_transient_coverage.py --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml \
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
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_pipeline import load_config  # noqa: E402
# Package-qualified (not `from find_saturation_point import ...`): direct script execution
# auto-adds HERE (tools/) to sys.path so either form works, but importing this module as
# tools.check_transient_coverage (e.g. from gen_dataset_from_schx.py) does NOT add tools/
# itself -- only `from run_pipeline import ...` above (repo root) would resolve; a bare
# `find_saturation_point` import would 404. This form works both ways.
from tools.find_saturation_point import find_saturation_point, findpeak_cache_key  # noqa: E402
from tools.render_backends import LiveSpiceBackend, NgspiceBackend, LtspiceBackend  # noqa: E402

SR = 48000


def _corners(knob_ranges: dict, full_hypercube: bool = True, max_full_corners: int = 512) -> list:
    """Corner set: all-min, all-max, center, each knob solo-extreme (rest at their own
    center), PLUS (if full_hypercube) the full binary hypercube -- every knob independently
    at its own min or max, 2**n corners total (all-min/all-max are 2 of them; deduped below).

    The binary hypercube was added after a real miss: Tweed 5F6-A Full's shipped blowup
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

    corners = [("all-min", dict(lo)), ("all-max", dict(hi)), ("center", dict(mid))]
    for n in names:
        solo_lo = dict(mid); solo_lo[n] = lo[n]
        solo_hi = dict(mid); solo_hi[n] = hi[n]
        corners.append((f"{n}=lo-solo", solo_lo))
        corners.append((f"{n}=hi-solo", solo_hi))

    if full_hypercube and names:
        n_full = 2 ** len(names)
        if n_full > max_full_corners:
            raise ValueError(f"full hypercube would be {n_full} corners (> max_full_corners="
                             f"{max_full_corners}) for {len(names)} knobs -- pass "
                             f"full_hypercube=False, or raise max_full_corners, explicitly")
        seen = {tuple(sorted(v.items())) for _, v in corners}
        for bits in itertools.product((0, 1), repeat=len(names)):
            vals = {n: (hi[n] if b else lo[n]) for n, b in zip(names, bits)}
            key = tuple(sorted(vals.items()))
            if key in seen:
                continue
            seen.add(key)
            corners.append((",".join(f"{n}={'hi' if b else 'lo'}" for n, b in zip(names, bits)),
                            vals))
    return corners


def _transient_peak_from_recipe(input_wav: Path) -> "float | None":
    recipe_path = input_wav.with_suffix(".recipe.json")
    if not recipe_path.exists():
        return None
    try:
        recipe = json.loads(recipe_path.read_text())
        return float(recipe["args"]["realistic_peak"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _check_corners(backend, identity: bytes, cache_extra: str, knob_ranges: dict, fixed: dict,
                    transient_peak: float, label: str, margin: float = 1.0,
                    peak_max_v: float = 40.0, no_cache: bool = False, quiet: bool = False,
                    full_hypercube: bool = True, lead_silence_s: float = 0.0) -> dict:
    """Backend-agnostic core: every corner's own saturation onset (find_saturation_point.py)
    vs. the excitation's transient peak. Shared by check_coverage() (.schx/LiveSPICE) and
    check_coverage_ngspice_deck() (a hand-written ngspice deck with no .schx at all) -- the
    ONLY thing that differs between them is how `backend`/`identity`/`cache_extra` get built.

    Returns {"ok": bool, "rows": [...]}. "ok" is False if ANY corner fails OR its onset
    couldn't be determined (a render failure is not a pass -- see main()'s same convention).
    """
    corners = _corners(knob_ranges, full_hypercube=full_hypercube)
    if not quiet:
        print(f"Transient saturation coverage: {label}")
        print(f"  transient peak = {transient_peak:.3f} V   {len(corners)} corners "
              f"({'full binary hypercube' if full_hypercube else 'reduced hypercube'} set)  "
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
                print(f"  {clabel:16} onset={onset if onset is None else f'{onset:.3f} V':>10}  {status}")
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
                   full_hypercube: bool = True) -> dict:
    """[.schx / LiveSPICE path] Importable directly (gen_dataset_from_schx.py's hard gate uses
    this in-process -- no subprocess, no re-parsing a config, and it can't be silently skipped
    by someone calling gen_dataset_from_schx.py without going through run_pipeline.py / this
    tool's own CLI first). See _check_corners() for the actual check."""
    backend = LiveSpiceBackend(schx, oversample=oversample, iterations=iterations)
    identity = Path(schx).read_bytes()
    cache_extra = f"os={oversample}|it={iterations}|maxv={peak_max_v}"
    return _check_corners(backend, identity, cache_extra, knob_ranges, fixed, transient_peak,
                           label=Path(schx).name, margin=margin, peak_max_v=peak_max_v,
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube)


def check_coverage_ngspice_deck(build_deck, module_file: str, probe_node: str, knob_ranges: dict,
                                fixed: dict, transient_peak: float, margin: float = 1.0,
                                maxstep: float = 3e-6, parallel_sims: int = 8,
                                peak_max_v: float = 40.0, no_cache: bool = False,
                                quiet: bool = False, full_hypercube: bool = True,
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
                           lead_silence_s=lead_silence_s)


def check_coverage_ltspice_deck(build_deck, module_file: str, tap: str, knob_ranges: dict,
                                fixed: dict, transient_peak: float, margin: float = 1.0,
                                maxstep: float = 3e-6, parallel_sims: int = 8,
                                out_scale: float = 0.05, peak_max_v: float = 40.0,
                                no_cache: bool = False, quiet: bool = False,
                                full_hypercube: bool = True) -> dict:
    """[hand-written LTspice-deck path] For a device whose ngspice-deck counterpart can't
    converge on real playing content at all -- see tools/ltspice_spicelib.py's own docstring.
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
                           no_cache=no_cache, quiet=quiet, full_hypercube=full_hypercube)


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
                         "tools/ltspice_spicelib.py's docstring")
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
                                             full_hypercube=not args.no_full_hypercube,
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
                                             full_hypercube=not args.no_full_hypercube)
        schx_or_module = module
    else:
        schx = str(cfg["schx"])
        oversample = args.oversample or cfg.get("oversample", 8)
        result = check_coverage(schx, knob_ranges, fixed, oversample, transient_peak,
                                margin=args.margin, iterations=args.iterations,
                                peak_max_v=args.peak_max_v, no_cache=args.no_cache,
                                full_hypercube=not args.no_full_hypercube)
        schx_or_module = schx

    if args.json:
        Path(args.json).write_text(json.dumps({
            "schx": schx_or_module, "input": str(input_wav), "transient_peak_v": transient_peak,
            "margin": args.margin, "corners": result["rows"],
        }, indent=2))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
