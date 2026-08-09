#!/usr/bin/env python3
"""
scaffold_config.py -- generate a starting config.toml for a new circuit.

Cuts the two most mechanical steps out of writing a new device's config.toml by hand:

  1. DISCOVER CONTROLS. Every real control (pot/switch, ganged pots collapsed to one) is
     read straight from the .schx via parse_schx_controls() -- the same discovery
     `gen_dataset_from_schx.py --schx X` (no --knobs) prints, reused here instead of
     hand-typing names into [knobs].
  2. MEASURE OVERSAMPLE. Runs measure_truncation.py's own measure() against a small set
     of candidate oversamples and prints the same kind of table it does, so you start
     from a real number instead of a guess -- but it does NOT pick one for you; see WHY
     NOT below.

WHAT IT DOES NOT DO.
The knob GRID stays a placeholder (--grid-points evenly-spaced values per knob, default
3) -- a good grid needs render probes at values the circuit's actual response curve
implies, which is exactly tools/grid_adequacy.py --apply's job. Run that next.

WHY IT DOESN'T PICK AN OVERSAMPLE FOR YOU.
measure_truncation.py's own docs describe a "stall" -- a candidate's error falling much
less than the trend elsewhere (~1.6x instead of ~4-5x) -- as a FINDING to investigate, not
a converged signal to act on: it can mean a render-side problem unrelated to true BDF2
convergence (see its docstring, the JCM800 hot-rod example). Automating past that would
risk silently shipping a bad oversample for exactly the circuits that most need scrutiny.
So this writes the largest tested candidate (a documented "fleet floor" starting point,
not a derived one) and prints the full table so you can apply the same judgment call the
docs walk through, by hand.

Usage:
    python tools/scaffold_config.py --schx path/to/circuit.schx
    python tools/scaffold_config.py --schx path/to/circuit.schx --input my_sweep.wav \\
        --output my_device/config.toml --grid-points 4

Then:
    python tools/grid_adequacy.py --config <output> --apply
"""
from __future__ import annotations

import argparse
import os as _os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gen_dataset_from_schx import parse_schx_controls
from measure_truncation import measure, probe_clips

TEMPLATE = Path(__file__).resolve().parent.parent / "examples" / "template.config.toml"


def _replace_line(text: str, key: str, new_line: str) -> str:
    """Replace a top-level `key = ...` line -- and any indented comment CONTINUATION
    lines immediately after it (e.g. the template's multi-line `input` comment) -- with
    `new_line`."""
    pattern = re.compile(rf"^{re.escape(key)}[ \t]*=.*\n(?:[ \t]+#.*\n?)*", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"template has no top-level '{key} = ...' line")
    return pattern.sub(lambda _m: new_line + "\n", text, count=1)


def _replace_table(text: str, header: str, body_lines: list) -> str:
    """Same consecutive-assignment-lines splice as grid_adequacy.write_knobs, so a
    trailing comment ahead of the next section (like the template's own `[fixed]`
    note) survives untouched."""
    m = re.search(rf"^\[{re.escape(header)}\][ \t]*(?:#.*)?\n", text, re.MULTILINE)
    if not m:
        raise ValueError(f"template has no [{header}] table")
    start = m.end()
    assign_re = re.compile(r"^[A-Za-z0-9_.-]+[ \t]*=.*\n", re.MULTILINE)
    end = start
    while (am := assign_re.match(text, end)):
        end = am.end()
    return text[:start] + "".join(l + "\n" for l in body_lines) + text[end:]


def _format_knobs(names: list, n_points: int) -> list:
    width = max(len(n_) for n_ in names)
    pts = [round(float(v), 4) for v in np.linspace(0.0, 1.0, max(2, n_points))]
    return [f"{name:<{width}} = {pts}" for name in names]


def _measure_oversample(schx: Path, knobs: list, input_wav: Path, candidates: tuple,
                        ref_os: int, probe_s: float, workers: int):
    """Run measure_truncation's own measure(), print its table, and return (chosen_os,
    comment) -- or (None, None) if the oracle isn't built or the render fails outright
    (best-effort: a failed measurement degrades to a placeholder, it doesn't abort the
    scaffold)."""
    if not input_wav.exists():
        print(f"  WARNING: --input {input_wav} not found -- skipping oversample "
              f"measurement, writing a placeholder instead.")
        return None, None
    try:
        with tempfile.TemporaryDirectory() as tds:
            td = Path(tds)
            clips, lead_n, sr = probe_clips(input_wav, probe_s, 4, td)
            print(f"  measuring truncation error at oversample {candidates} "
                  f"(vs os={ref_os} reference) ...", flush=True)
            r = measure(schx, knobs, clips, lead_n, sr, ref_os, candidates, 256, None, td,
                       workers)
    except Exception as e:
        print(f"  WARNING: oversample measurement failed ({e}) -- is the oracle built? "
              f"(see setup.sh --no-cli / $LIVESPICE_CLI). Writing a placeholder instead.")
        return None, None

    vals = {c: r.get(c, (float("nan"), None))[0] for c in candidates}
    print(f"\n  {'os':>4}  {'truncation ESR':>15}  worst setting")
    for c in candidates:
        at = r.get(c, (float("nan"), None))[1]
        loc = ", ".join(f"{k}={v:g}" for k, v in sorted(at.items())) if at else "-"
        print(f"  {c:>4}  {vals[c]:>15.2e}  {loc}")
    falls = [f"{vals[a]/vals[b]:.1f}x" for a, b in zip(candidates, candidates[1:])
             if np.isfinite(vals[a]) and np.isfinite(vals[b]) and vals[b] > 0]
    if falls:
        print(f"  fall per step: {' -> '.join(falls)}  (O(h^2) wants ~4x per doubling; a "
              f"MUCH smaller fall than that (e.g. ~1.5-2x) usually means a render-side "
              f"issue, not convergence -- worth checking by hand against "
              f"measure_truncation.py's docstring before trusting the number below)")

    chosen = candidates[-1]
    comment = f"measured by scaffold_config.py -- {vals[chosen]:.2e} at os={chosen} (see table above)"
    return chosen, comment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schx", type=Path, required=True)
    ap.add_argument("--input", default="examples/sweepv5.wav",
                    help="sweep/DI to measure truncation against and record as `input` "
                         "(default: the bundled examples/sweepv5.wav -- assumes you run "
                         "this from the repo root)")
    ap.add_argument("--output", type=Path, default=None,
                    help="config.toml path to write (default: <schx stem>.config.toml "
                         "in the current directory)")
    ap.add_argument("--backend", choices=["livespice", "ngspice"], default="livespice",
                    help="oversample auto-measurement only runs for livespice -- ngspice's "
                         "adaptive timestepping isn't tuned the same way (see "
                         "ngspice/README.md)")
    ap.add_argument("--grid-points", type=int, default=3,
                    help="placeholder points per knob, evenly spaced 0..1 (default 3) -- "
                         "refine with grid_adequacy.py --apply afterward")
    ap.add_argument("--candidates", default="2,4,8",
                    help="oversamples to measure (default 2,4,8, same as measure_truncation.py)")
    ap.add_argument("--ref-os", type=int, default=32)
    ap.add_argument("--probe-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if not args.schx.exists():
        sys.exit(f"schx not found: {args.schx}")

    controls = parse_schx_controls(str(args.schx))
    if not controls:
        sys.exit(f"no pots/switches found in {args.schx} -- nothing to scaffold a grid for")
    names = sorted(set(controls.values()))
    print(f"  {len(names)} control(s) discovered in {args.schx.name}: {', '.join(names)}")

    output = args.output or Path(f"{args.schx.stem}.config.toml")

    text = TEMPLATE.read_text()
    text = _replace_line(text, "schx",
        f'schx       = "{args.schx}"   # {len(names)} control(s) discovered by scaffold_config.py')
    text = _replace_line(text, "input", f'input      = "{args.input}"')
    text = _replace_line(text, "backend", f'backend    = "{args.backend}"')

    if args.backend == "livespice":
        cands = tuple(int(c) for c in args.candidates.split(","))
        oversample, comment = _measure_oversample(
            args.schx, names, Path(args.input), cands, args.ref_os, args.probe_s, args.workers)
        if oversample is None:
            oversample, comment = 8, "COULD NOT MEASURE (see warning above) -- placeholder"
        text = _replace_line(text, "oversample", f"oversample = {oversample}   # {comment}")
    # else: leave the template's placeholder oversample + comment untouched -- ngspice
    # tuning is a different question (see ngspice/README.md), not this tool's job.

    text = _replace_table(text, "knobs", _format_knobs(names, args.grid_points))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    tot = max(2, args.grid_points) ** len(names)
    print(f"\n  wrote {output}  ([knobs] is a {args.grid_points}-point-per-axis placeholder, "
          f"{tot} permutations)")
    print(f"  next: python tools/grid_adequacy.py --config {output} --apply   "
          f"# refine the placeholder grid")


if __name__ == "__main__":
    main()
