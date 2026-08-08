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
convention), find that corner's OWN saturation onset (reusing preflight.py's
find_saturation_point) and compare it against the excitation's transient peak.

Exit status is nonzero if any corner fails, so a generation script can gate on it
(same convention as preflight.py):
  python tools/check_transient_coverage.py ... && python gen_dataset_from_schx.py ...

Usage:
  python tools/check_transient_coverage.py --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml \
      [--transient-peak 0.2] [--margin 1.0] [--oversample 8] [--iterations 256] \
      [--json report.json] [--no-cache]

--transient-peak: the excitation's transient/real-playing segment peak, in volts
  at V0dBFS=1. Auto-read from the excitation's <stem>.recipe.json sidecar
  (build_excitation.py's `args.realistic_peak`) if present; otherwise REQUIRED --
  this tool refuses to guess it from the raw audio (silently mis-slicing the
  file's realistic/sweep boundary would be worse than refusing to run).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_pipeline import load_config  # noqa: E402
# Package-qualified (not `from preflight import ...`): direct script execution auto-adds HERE
# (tools/) to sys.path so either form works, but importing this module as tools.check_transient_
# coverage (e.g. from gen_dataset_from_schx.py) does NOT add tools/ itself -- only `from run_pipeline
# import ...` above (repo root) would resolve; `preflight` bare would 404. This form works both ways.
from tools.preflight import find_saturation_point, _findpeak_cache_path  # noqa: E402

SR = 48000


def _corners(knob_ranges: dict) -> list:
    """Reduced hypercube corner set: all-min, all-max, center, each knob solo-
    extreme (rest at their own center). Same convention as
    scan_film_runaway.py's hypercube_corners() and this fleet's existing
    knob-grid-design practice (internal engineering notes) -- NOT the full
    grid, which would be 972 renders x 20 amplitude points for Tweed alone.
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


def check_coverage(schx: str, knob_ranges: dict, fixed: dict, oversample: int,
                   transient_peak: float, margin: float = 1.0, iterations: int = 256,
                   peak_max_v: float = 40.0, no_cache: bool = False, quiet: bool = False) -> dict:
    """Core check, importable directly (gen_dataset_from_schx.py's hard gate uses this in-process --
    no subprocess, no re-parsing a config, and it can't be silently skipped by someone calling
    gen_dataset_from_schx.py without going through run_pipeline.py / this tool's own CLI first).

    Returns {"ok": bool, "rows": [...]}. "ok" is False if ANY corner fails OR its onset
    couldn't be determined (a render failure is not a pass -- see main()'s same convention).
    """
    corners = _corners(knob_ranges)
    if not quiet:
        print(f"Transient saturation coverage: {Path(schx).name}")
        print(f"  transient peak = {transient_peak:.3f} V   {len(corners)} corners (reduced "
              f"hypercube set)  margin={margin}x\n")

    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for label, vals in corners:
            params = dict(vals); params.update(fixed)
            cpath = _findpeak_cache_path(schx, params, oversample, iterations, peak_max_v)
            if cpath.exists() and not no_cache:
                sat = json.loads(cpath.read_text())
            else:
                sat = find_saturation_point(schx, params, oversample, iterations,
                                            scratch, max_v=peak_max_v)
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
                print(f"  {label:16} onset={onset if onset is None else f'{onset:.3f} V':>10}  {status}")
            rows.append({"corner": label, "params": params, "onset_v": onset, "ok": ok})

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
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    schx = str(cfg["schx"])
    input_wav = Path(cfg["input"]).expanduser()
    oversample = args.oversample or cfg.get("oversample", 8)

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
    result = check_coverage(schx, knob_ranges, fixed, oversample, transient_peak,
                            margin=args.margin, iterations=args.iterations,
                            peak_max_v=args.peak_max_v, no_cache=args.no_cache)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "schx": schx, "input": str(input_wav), "transient_peak_v": transient_peak,
            "margin": args.margin, "corners": result["rows"],
        }, indent=2))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
