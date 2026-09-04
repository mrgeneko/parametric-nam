#!/usr/bin/env python3
"""
scaffold_config.py -- generate a starting config.toml for a new circuit.

Cuts the three most mechanical steps out of writing a new device's config.toml by hand:

  1. DISCOVER CONTROLS. Every real control (pot/switch, ganged pots collapsed to one) is
     read straight from the .schx via parse_schx_controls() -- the same discovery
     `gen_dataset_from_schx.py --schx X` (no --knobs) prints, reused here instead of
     hand-typing names into [knobs].
  2. MEASURE OVERSAMPLE. Runs measure_truncation.py's own measure() against a small set
     of candidate oversamples and prints the same kind of table it does, so you start
     from a real number instead of a guess -- but it does NOT pick one for you; see WHY
     NOT below.
  3. GUESS KNOB ROLES. Each discovered name is run through knob_classify.classify() --
     the same hi/lo/mid/drive/rms heuristic preflight.py uses for its own direction
     and EQ-swamp checks -- and written into a [knob-kind] table. This is explicitly a
     GUESS, not a determination: a name that matches no known keyword is written
     commented-out as UNCONFIRMED rather than silently defaulted, and every guessed line
     carries the matched label as a trailing comment so a wrong guess is easy to spot on
     read-through. Always review this table by hand -- an unclassified or misclassified
     knob doesn't just get the wrong sensitivity metric in gen_dataset_from_schx.py, it
     also silently skips preflight.py's role-aware EQ-check protection for every OTHER
     knob's direction check.

WHAT IT DOES NOT DO.
The knob GRID stays a placeholder -- role-aware (see GUESS KNOB ROLES / _grid_for_kind
below), not a naive linspace(0,1) for every knob regardless of role, but still a starting
point, not a measured one. A good grid needs render probes at values the circuit's actual
response curve implies, which is exactly grid_adequacy.py --apply's job. Run that next.

WHY IT DOESN'T PICK AN OVERSAMPLE FOR YOU.
measure_truncation.py's own docs describe a "stall" -- a candidate's error falling much
less than the trend elsewhere (~1.6x instead of ~4-5x) -- as a FINDING to investigate, not
a converged signal to act on: it can mean a render-side problem unrelated to true BDF2
convergence (see its docstring, the British-stack amp's hot-rod variant example). Automating past that would
risk silently shipping a bad oversample for exactly the circuits that most need scrutiny.
So this writes the largest tested candidate (a documented "fleet floor" starting point,
not a derived one) and prints the full table so you can apply the same judgment call the
docs walk through, by hand.

Usage:
    python scaffold_config.py --schx path/to/circuit.schx
    python scaffold_config.py --schx path/to/circuit.schx --input my_sweep.wav \\
        --output my_device/config.toml --grid-points 4

Then:
    python grid_adequacy.py --config <output> --apply
"""
from __future__ import annotations

import argparse
import os as _os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_dataset_from_schx import parse_schx_controls
from measure_truncation import measure, probe_clips
from knob_classify import classify  # noqa: E402

DEFAULT_REALISTIC_DUR_CAP = 150.0

TEMPLATE = Path(__file__).resolve().parent / "examples" / "template.config.toml"


def _replace_line(text: str, key: str, new_line: str) -> str:
    """Replace a top-level `key = ...` line -- and any indented comment CONTINUATION
    lines immediately after it (e.g. the template's multi-line `input` comment) -- with
    `new_line`."""
    pattern = re.compile(rf"^{re.escape(key)}[ \t]*=.*\n(?:[ \t]+#.*\n?)*", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"template has no top-level '{key} = ...' line")
    return pattern.sub(lambda _m: new_line + "\n", text, count=1)


def _toml_key(name: str) -> str:
    """Quote a TOML key unless it's already a valid bare key (letters/digits/underscore/
    dot/hyphen only). Almost every real knob name in this fleet has a space (e.g. "Lead
    Pre", "OR Gain"), which is NOT a valid bare TOML key -- writing it unquoted produces a
    file Python's own tomllib (what run_pipeline.py and grid_adequacy.py both load configs
    with) refuses to parse. Found 2026-08-29 the hard way: every [knobs]/[knob-kind] line
    this module ever wrote for a multi-word knob was invalid TOML."""
    return name if re.fullmatch(r"[A-Za-z0-9_.-]+", name) else f'"{name}"'


def _replace_table(text: str, header: str, body_lines: list) -> str:
    """Same consecutive-assignment-lines splice as grid_adequacy.write_knobs, so a
    trailing comment ahead of the next section (like the template's own `[fixed]`
    note) survives untouched."""
    m = re.search(rf"^\[{re.escape(header)}\][ \t]*(?:#.*)?\n", text, re.MULTILINE)
    if not m:
        raise ValueError(f"template has no [{header}] table")
    start = m.end()
    # Bare key OR a quoted key ("..."/'...') -- see _toml_key(); without the quoted
    # alternative this loop matches zero lines on any already-quoted table, same bug
    # documented in grid_adequacy.write_knobs.
    assign_re = re.compile(r'^(?:[A-Za-z0-9_.-]+|"[^"]*"|\'[^\']*\')[ \t]*=.*\n', re.MULTILINE)
    end = start
    while (am := assign_re.match(text, end)):
        end = am.end()
    return text[:start] + "".join(l + "\n" for l in body_lines) + text[end:]


DEFAULT_GRID_POINTS = 3  # this module's own --grid-points default, named so _grid_for_kind
                          # can tell "caller didn't ask for more than the default" apart from
                          # an explicit override -- see the drive/rms branch below.


def _grid_for_kind(kind: "str | None", n_points: int) -> list:
    """Default knob-grid VALUES for a classify() kind -- still a placeholder meant to be
    refined by grid_adequacy.py --apply, but an informed one instead of a naive linspace(0,1)
    for every knob regardless of role. Empirical basis (this fleet's own hand-tuned grids,
    e.g. the non-midpoint-default pedal's / the MOSFET-clipping pedal's Gain=[0.1,0.15,0.25,...,0.9-0.95,1.0]):

      hi/lo/mid (tone/EQ knob): STILL evenly spaced (no observed need for endpoint density
        here), but narrowed to [0.2, 0.8] -- the fully-CCW/CW extremes rarely hold a tone
        stack's interesting behavior, so a naive [0,1] grid wastes render budget there.

      drive/rms (gain/volume knob): NOT evenly spaced, and denser than the 6-point default
        this used to be -- now [0.1, 0.15, 0.25, 0.50, 0.75, 0.95, 1.0], 7 fixed anchor
        points. A gain-type knob's audible character changes fastest near the bottom of its
        range, so 0.1 (not 0.0 -- a knob truly at zero is often a degenerate/dead corner
        anyway) gets two extra points just above it (+0.05, +0.15); the top gets one extra
        point just below max (-0.05) to "stabilize" the max grid point; 0.50 and 0.75 fill
        the middle instead of leaving it to a single linspace-derived point. Raised from 5
        anchors (+1 incidental baseline midpoint) to these 7 fixed points now that
        gen_dataset_from_schx.py's --shard makes a denser default grid affordable -- dataset
        generation no longer has to render the whole knob-grid product in one pass. At or
        below DEFAULT_GRID_POINTS this returns exactly these 7 anchors, nothing added or
        deduped away; --grid-points still spreads that many points evenly across the full
        range as a baseline ONLY when explicitly raised above the default (so a larger
        --grid-points adds real middle-of-range coverage on top of the anchors, not just
        these 7), deduped/sorted together with them.

      unclassified (no keyword match): legacy behavior, evenly spaced [0, 1] -- see
      knob_classify.classify()'s own docstring for why an unclassified knob gets the LEAST
      special treatment everywhere, not the most; this is no exception.
    """
    if kind in ("hi", "lo", "mid"):
        lo, hi = 0.2, 0.8
        return [round(float(v), 4) for v in np.linspace(lo, hi, max(2, n_points))]
    if kind in ("drive", "rms"):
        lo, hi = 0.1, 1.0
        anchors = [0.1, 0.15, 0.25, 0.50, 0.75, 0.95, 1.0]
        if n_points <= DEFAULT_GRID_POINTS:
            return anchors
        baseline = list(np.linspace(lo, hi, n_points))
        return sorted(set(round(float(v), 4) for v in baseline + anchors))
    return [round(float(v), 4) for v in np.linspace(0.0, 1.0, max(2, n_points))]


def _format_knobs(names: list, n_points: int) -> list:
    keys = {name: _toml_key(name) for name in names}
    width = max(len(k) for k in keys.values())
    lines = []
    for name in names:
        kind, _ = classify(name)
        pts = _grid_for_kind(kind, n_points)
        lines.append(f"{keys[name]:<{width}} = {pts}")
    return lines


def _format_knob_kind(names: list) -> list:
    """Auto-guess each knob's role from its NAME via knob_classify.classify() -- the same
    heuristic preflight.py uses for its own direction/EQ-swamp checks. A guess is
    ALWAYS a guess: an unconventional or non-English name can silently match nothing, so
    every line gets a trailing comment naming the label that drove the guess (or flagging
    UNCONFIRMED for a no-match) precisely so this doesn't read as more certain than it is --
    review it by hand before trusting it."""
    keys = {name: _toml_key(name) for name in names}
    width = max(len(k) for k in keys.values())
    lines = []
    for name in names:
        kind, label = classify(name)
        if kind is None:
            lines.append(f'# {keys[name]:<{width}} = "UNCONFIRMED"   # name matched no known keyword -- classify by hand')
        else:
            lines.append(f'{keys[name]:<{width}} = "{kind}"   # guessed from name ({label}) -- verify')
    return lines


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


def _interior_sample_budget(n_knobs: int) -> int:
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


def _prepare_excitation(config_path: Path, real_clip: Path, output_wav: Path,
                        realistic_dur_cap: float, n_knobs: int = 0) -> bool:
    """Build a properly-calibrated excitation via prepare_excitation.py (which measures
    this circuit's REAL saturation onset across the knob grid's corners, rather than
    pointing `input` at a raw downloaded sweep with no calibration behind it at all --
    the gap that made check_transient_coverage.py hard-refuse to run on this scaffold's
    own output the first time it was tried, 2026-08-29).

    Reads --config (this function's caller has already written it, with schx + the
    placeholder [knobs] grid), so prepare_excitation.py's own corner-finder has real
    knob ranges to work from -- best-effort: a failure here degrades to leaving `input`
    pointed at the raw clip (same "best-effort, don't abort the scaffold" philosophy as
    _measure_oversample), not an aborted scaffold.

    --realistic-dur is capped at `realistic_dur_cap` -- build_excitation.py's own default
    is "the whole --input file" (uncapped), which for a typical multi-minute downloaded
    sweep would make the realistic-content portion LONGER than the excitation needs to be
    for its stated purpose (sampling dynamics/perceptual variety, not saturation coverage
    -- the sweeps provide that). Only passed when the source clip actually exceeds the
    cap; a shorter source is left alone rather than padded or otherwise altered.
    """
    info = sf.info(str(real_clip))
    # NOTE: deliberately does NOT pass --no-full-hypercube. It used to be hardcoded here, which
    # opted EVERY scaffolded device out of the full-hypercube corner set -- the set that exists
    # precisely because the structural-only set cannot represent a mixed
    # some-knobs-low-others-high corner (see check_transient_coverage._corners.__doc__ and the
    # tweed blowup it records). For a 4-knob pedal that "saving" was 11 corners instead of 25,
    # and it cost Duke of Tone a failed coverage gate on 2026-09-04: the excitation was sized
    # from a reduced-set worst onset of 8.647 V, then the full set found 8.766 V.
    # prepare_excitation.py raises a clear, actionable error above 9 knobs; pass --max-corners
    # there instead of going back to the blind set.
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "prepare_excitation.py"),
           "--backend", "livespice", "--config", str(config_path),
           "--real-clip", str(real_clip), "--output", str(output_wav)]
    # Probe the grid INTERIOR too, not just corners -- see _interior_sample_budget.
    budget = _interior_sample_budget(n_knobs)
    if budget:
        cmd += ["--sample-grid", str(budget)]
        print(f"  probing {budget} interior grid point(s) on top of the corner set -- corners "
              f"alone cannot see a non-monotonic onset maximum (Mesa Orange's was 1.27x every "
              f"vertex)")
    if info.duration > realistic_dur_cap:
        print(f"  {real_clip.name} is {info.duration:.1f}s, longer than the "
              f"{realistic_dur_cap:.0f}s realistic-content cap -- truncating to "
              f"{realistic_dur_cap:.0f}s (--realistic-dur {realistic_dur_cap:.0f}).")
        cmd += ["--realistic-dur", str(realistic_dur_cap)]
    print(f"  building a calibrated excitation (measuring saturation onset across the "
          f"knob grid's corners, then build_excitation.py) -- this renders, budget "
          f"real time for it ...")
    r = subprocess.run(cmd)
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schx", type=Path, required=True)
    ap.add_argument("--input", default="examples/T3K-sweep-v3.wav",
                    help="sweep/DI to measure truncation against and record as `input` "
                         "(default: examples/T3K-sweep-v3.wav -- NOT bundled, download it "
                         "from https://www.tone3000.com/capture; assumes you run "
                         "this from the repo root)")
    ap.add_argument("--output", type=Path, default=None,
                    help="config.toml path to write (default: <schx stem>.config.toml "
                         "in the current directory)")
    ap.add_argument("--backend", choices=["livespice", "ngspice"], default="livespice",
                    help="oversample auto-measurement only runs for livespice -- ngspice's "
                         "adaptive timestepping isn't tuned the same way (see "
                         "ngspice/README.md)")
    ap.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS,
                    help=f"placeholder points per knob (default {DEFAULT_GRID_POINTS}), spread "
                         "evenly across that knob's role-aware range as a BASELINE (see "
                         "_grid_for_kind) -- a drive/rms knob uses its own fixed 7-point anchor "
                         "set instead at or below this default, only adding baseline points on "
                         "top when raised above it. Refine with grid_adequacy.py --apply afterward")
    ap.add_argument("--candidates", default="2,4,8",
                    help="oversamples to measure (default 2,4,8, same as measure_truncation.py)")
    ap.add_argument("--ref-os", type=int, default=32)
    ap.add_argument("--probe-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 4) - 2))
    ap.add_argument("--skip-prepare-excitation", action="store_true",
                    help="leave `input` pointed at the raw --input clip as-is, instead of "
                         "the default of building a properly-calibrated excitation from it "
                         "via prepare_excitation.py (measures this circuit's real "
                         "saturation onset across the knob grid, sizes the excitation from "
                         "that instead of a guess). The raw-clip default this flag restores "
                         "is exactly what made check_transient_coverage.py hard-refuse to "
                         "run against this scaffold's own output -- only skip this if you "
                         "specifically want that raw, uncalibrated behavior back (e.g. to "
                         "avoid the extra render time right now).")
    ap.add_argument("--realistic-dur-cap", type=float, default=DEFAULT_REALISTIC_DUR_CAP,
                    help=f"cap (seconds) on the excitation's realistic-content portion "
                         f"when --input exceeds it (default {DEFAULT_REALISTIC_DUR_CAP:.0f}s) "
                         f"-- build_excitation.py's own default is the WHOLE --input file, "
                         f"uncapped, which the realistic segment doesn't need (it only "
                         f"samples dynamics/perceptual variety; the sweeps provide "
                         f"saturation coverage). No effect if --skip-prepare-excitation.")
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

    knob_kind_lines = _format_knob_kind(names)
    text = _replace_table(text, "knob-kind", knob_kind_lines)
    unconfirmed = [n for n, l in zip(names, knob_kind_lines) if l.lstrip().startswith("#")]
    if unconfirmed:
        print(f"  [knob-kind]: {len(unconfirmed)}/{len(names)} name(s) matched no known "
              f"keyword, left UNCONFIRMED (commented out): {', '.join(unconfirmed)} -- "
              f"classify by hand (hi/lo/mid/drive/rms, see knob_classify.py)")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    counts = {name: len(_grid_for_kind(classify(name)[0], args.grid_points)) for name in names}
    tot = 1
    for c in counts.values(): tot *= c
    axes = ", ".join(f"{name}={n}" for name, n in counts.items())
    print(f"\n  wrote {output}  ([knobs] is a role-aware placeholder -- {axes} points/axis, "
          f"{tot} combinations)")

    if args.backend == "livespice" and not args.skip_prepare_excitation and not Path(args.input).exists():
        print(f"  WARNING: --input {args.input} not found -- skipping the calibrated-excitation "
              f"build, leaving `input` pointed at the raw path as given.")
    elif args.backend == "livespice" and not args.skip_prepare_excitation:
        excitation_wav = output.with_name(f"{output.stem}_excitation.wav")
        ok = _prepare_excitation(output, Path(args.input), excitation_wav,
                                 args.realistic_dur_cap, n_knobs=len(names))
        if ok and excitation_wav.exists():
            text = output.read_text()
            text = _replace_line(text, "input",
                f'input      = "{excitation_wav}"   # built by prepare_excitation.py from '
                f'{Path(args.input).name} -- see its own recipe.json sidecar for the '
                f'measured saturation onset this was sized from')
            output.write_text(text)
            print(f"  input -> {excitation_wav} (calibrated; run "
                  f"check_transient_coverage.py against {output} to verify independently)")
        else:
            print(f"  WARNING: prepare_excitation.py failed or produced no output -- "
                  f"leaving `input` pointed at the raw {args.input}. "
                  f"gen_dataset_from_schx.py will refuse to start generation against this "
                  f"config unless you pass --transient-peak explicitly or "
                  f"--skip-transient-check.")

    print(f"  next: python grid_adequacy.py --config {output} --apply   "
          f"# refine the placeholder grid")


if __name__ == "__main__":
    main()
