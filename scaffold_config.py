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
from gen_dataset_from_schx import parse_schx_controls, _is_convergence_failure
from measure_truncation import measure, merge_truncation_shards, probe_clips, ref_error_check, score_rows
from knob_classify import classify  # noqa: E402
from run_pipeline import set_input_line  # noqa: E402
# ONE definition, shared with gen_dataset_from_schx.py's transient gate. If the sizer and
# the gate computed this separately they could drift, and a gate weaker than the sizing
# that fed it is worse than no gate: it reads as confirmation while checking less.
from check_transient_coverage import interior_sample_budget as _interior_sample_budget  # noqa: E402

DEFAULT_SWEEP_DUR_CAP = 150.0

from capture_chain import (add_cli_args as _cc_add_cli_args, resolve as _cc_resolve,  # noqa: E402
                           describe as _cc_describe)

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


def _insert_line_after(text: str, after_key: str, new_lines: str) -> str:
    """Insert `new_lines` right after `after_key`'s line (and any indented comment
    continuation lines) -- for a field the shared template does not declare at all,
    because it only applies to one backend (ngspice-deck's --pedal-dir/--module/
    --probe-node/--maxstep). Same splice convention as run_pipeline.set_input_line,
    generalized to insert rather than replace."""
    pat = re.compile(rf"^{re.escape(after_key)}[ \t]*=.*\n(?:[ \t]+#.*\n?)*", re.MULTILINE)
    m = pat.search(text)
    if not m:
        raise ValueError(f"template has no top-level '{after_key} = ...' line to insert after")
    return text[:m.end()] + new_lines.rstrip("\n") + "\n" + text[m.end():]


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
        this used to be -- now [0.1, 0.15, 0.25, 0.50, 0.75, 0.90, 1.0], 7 fixed anchor
        points. A gain-type knob's audible character changes fastest near the bottom of its
        range, so 0.1 (not 0.0 -- a knob truly at zero is often a degenerate/dead corner
        anyway) gets two extra points just above it (+0.05, +0.15); the top gets one extra
        point just below max (-0.10) to "stabilize" the max grid point; 0.50 and 0.75 fill
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
        anchors = [0.1, 0.15, 0.25, 0.50, 0.75, 0.90, 1.0]
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


class _BackendDiverged(RuntimeError):
    """The oracle ran and the circuit did not converge -- a LOUD backend failure.

    Distinct from "the oracle is missing", which is a setup problem and says nothing about the
    circuit. See _measure_oversample's except block and _report_divergence below.
    """


def _sidecar_stub(schx: Path, backend: str, err: str) -> str:
    """The sidecar this measurement justifies, ready to drop beside the .schx.

    Generated rather than hand-written, because this IS a measurement: the scaffold just ran
    the circuit through the backend and watched it fail. Hand-writing should be reserved for
    the judgements a machine cannot make -- "renders stably, but the clipping stage is only an
    approximation because LiveSPICE has no MOSFET component" is an opinion about fidelity;
    "Circuit.SimulationDiverged near t=2.65s" is an observation.
    """
    one_line = " ".join(err.split())[:400]
    return (f'# Backend-validity sidecar for {schx.name}\n'
            f'# Generated by scaffold_config.py on a MEASURED divergence -- not hand-written.\n'
            f'# Read by gen_dataset_from_schx.py, which derives this path from the .schx.\n'
            f'#\n'
            f'# REVIEW BEFORE COMMITTING. A divergence is evidence the backend cannot hold this\n'
            f'# circuit, but it can also be a probe artefact. Worth ruling out first: too few\n'
            f'# Newton iterations reads as stiffness (a circuit that "needs more oversample" is\n'
            f'# often one that needs more iterations -- check that first, it is far cheaper), and\n'
            f'# a genuine resolution problem shrinks as oversample rises while a model\n'
            f'# instability does not.\n'
            f'{backend} = {{ valid = false, reason = """\n'
            f'DIVERGES during scaffold_config.py oversample measurement: {one_line}\n'
            f'Measured, not asserted. Confirm it is not a resolution artefact (see above), then\n'
            f'record which backend DOES work and why.""" }}\n')


def _report_divergence(schx: Path, backend: str, err: str, write: bool) -> None:
    sidecar = schx.with_suffix(".backends.toml")
    print(f"\n  {backend} DIVERGED on {schx.name} during oversample measurement:")
    print(f"      {' '.join(err.split())[:160]}")
    print(f"  That is a LOUD backend failure -- the circuit did not converge, and no oversample")
    print(f"  will fix a model instability. A config naming this backend would render a grid of")
    print(f"  crashes.")
    if sidecar.exists():
        print(f"  {sidecar.name} already exists -- left untouched.")
    elif write:
        sidecar.write_text(_sidecar_stub(schx, backend, err))
        print(f"  Wrote {sidecar.name} recording it. REVIEW IT before committing.")
    else:
        print(f"  Re-run with --write-backend-sidecar to record it in {sidecar.name}, or pass")
        print(f"  --backend ngspice if you already know which backend holds this circuit.")


def _measure_oversample(schx: Path, knobs: list, input_wav: Path, candidates: tuple,
                        ref_os: int, probe_s: float, workers: int,
                        oversample_results: list | None = None):
    """Run measure_truncation's own measure(), print its table, and return (chosen_os,
    comment) -- or (None, None) if the oracle isn't built or the render fails outright
    (best-effort: a failed measurement degrades to a placeholder, it doesn't abort the
    scaffold).

    `oversample_results`, if given, is a list of measure_truncation.py --emit files (e.g. from
    `distribute_pull.py --tool measure_truncation ... --collect`) -- this SKIPS the local
    sweep entirely and scores from that pre-sharded, already-collected set instead, the same
    way `measure_truncation.py --merge` does. Only the reference-convergence check still
    renders here (fresh, against the worst setting the merge finds -- see ref_error_check's own
    docstring on why a merge can't reuse a shard's cache). Added 2026-09-21: on a full amp
    (6 knobs, sag supply) this step alone measured 70+ minutes single-machine, and this file's
    own `--workers` was local-only -- see measure_truncation.py's docstring for the full story.
    """
    if oversample_results:
        print(f"  scoring {len(oversample_results)} pre-measured shard file(s) instead of "
              f"rendering locally ...", flush=True)
        rows, m_cands, m_ref_os = merge_truncation_shards([str(p) for p in oversample_results])
        if tuple(m_cands) != tuple(candidates) or m_ref_os != ref_os:
            print(f"  WARNING: the merged shards were measured with candidates={m_cands}/"
                  f"ref_os={m_ref_os}, not this run's {candidates}/{ref_os} -- using the "
                  f"SHARDS' own values instead of what was requested.")
            candidates, ref_os = tuple(m_cands), m_ref_os
        r = score_rows(rows, candidates)
        _, at_worst = r[candidates[0]]
        if at_worst is not None:
            try:
                with tempfile.TemporaryDirectory() as tds:
                    td = Path(tds)
                    clips, lead_n, sr = probe_clips(input_wav, probe_s, 4, td)
                    r["ref_error"] = ref_error_check(schx, at_worst, ref_os, clips, lead_n, sr,
                                                     256, None, td, workers)
            except Exception as e:
                print(f"  WARNING: post-merge reference-convergence check failed ({e}) -- "
                      f"table below has no ref_error column.")
        return _print_oversample_table(candidates, r)
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
        # TWO VERY DIFFERENT FAILURES, and lumping them together threw away the more valuable
        # one. "the oracle isn't built" is setup; "the oracle ran and the circuit DIVERGED" is
        # the single strongest evidence that this backend cannot render this circuit -- the same
        # thing a backends sidecar records as a LOUD failure. This function used to blame both
        # on a missing oracle and downgrade to a placeholder oversample, so the scaffold could
        # MEASURE a circuit diverging and still write backend = "livespice" without comment.
        if _is_convergence_failure(str(e)):
            raise _BackendDiverged(str(e)) from e
        print(f"  WARNING: oversample measurement failed ({e}) -- is the oracle built? "
              f"(see setup.sh --no-cli / $LIVESPICE_CLI). Writing a placeholder instead.")
        return None, None

    return _print_oversample_table(candidates, r)


def _print_oversample_table(candidates: tuple, r: dict):
    """Print the (os, truncation ESR, worst setting) table and return (chosen_os, comment) --
    shared by _measure_oversample()'s local-render path and its --oversample-results
    (pre-sharded/merged) path, so both produce an identical table from an identical `res` dict.
    """
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

    excluded = r.get("excluded_near_silent")
    if excluded:
        from measure_truncation import NEAR_SILENT_DEN_RATIO
        for c, params_list in excluded.items():
            locs = "; ".join(", ".join(f"{k}={v:g}" for k, v in sorted(p.items()))
                             for p in params_list)
            print(f"  NOTE: @ os={c}: {len(params_list)} setting(s) excluded from the "
                  f"worst-setting pick as near-silent (reference energy < "
                  f"{NEAR_SILENT_DEN_RATIO:.0e} of the loudest setting measured) -- ESR is not "
                  f"a meaningful number there (a Volume/Master-shaped knob muting the output "
                  f"is the usual cause), not evidence of a truncation problem: {locs}")

    chosen = candidates[-1]
    comment = f"measured by scaffold_config.py -- {vals[chosen]:.2e} at os={chosen} (see table above)"
    return chosen, comment




def _prepare_excitation(config_path: Path, sweep_file: Path, output_wav: Path,
                        sweep_dur_cap: float, n_knobs: int = 0, workers: int = 4,
                        backend: str = "livespice", deck_args: dict = None,
                        knob_ranges: dict = None, chain: dict = None) -> bool:
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

    `backend="ngspice-deck"` is a DIFFERENT contract: prepare_excitation.py's ngspice-deck
    branch ignores --config for corner resolution entirely (it only reads --config, if given,
    to update `input` afterward -- see its own main(), the `if args.backend == "ngspice-deck"`
    branch requires --pedal-dir/--module/--range explicitly and never touches args.config until
    the final update step). So for this backend `deck_args` (pedal_dir/module/probe_node/
    maxstep) and `knob_ranges` ({name: [v1, v2, ...]}, the SAME placeholder grid just written
    into [knobs]) must be supplied and are forwarded as explicit --range flags -- there is no
    way to hand prepare_excitation.py a config and have it figure out ngspice-deck knobs itself.
    For the SAME reason, `chain` (the capture-chain dict scaffold_config.py already resolved
    and wrote into this config, or None for --no-capture-chain) is also forwarded explicitly --
    prepare_excitation.py's ngspice-deck branch resolves capture from CLI flags only, never
    from --config (see its own `_capture = _cc_resolve(args)` at the top of that dispatch,
    only re-resolved WITH cfg inside the livespice branch specifically).

    --sweep-dur is capped at `sweep_dur_cap` -- build_excitation.py's own default is "the
    whole --sweep-file" (uncapped), which for a typical multi-minute downloaded sweep would
    make that segment LONGER than the excitation needs to be for its stated purpose
    (sampling dynamics/perceptual variety, not saturation coverage -- the chirps provide
    that). Only passed when the source clip actually exceeds the cap; a shorter source is
    left alone rather than padded or otherwise altered.
    """
    info = sf.info(str(sweep_file))
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
           "--backend", backend, "--config", str(config_path),
           "--sweep-file", str(sweep_file), "--output", str(output_wav)]
    if backend == "ngspice-deck":
        cmd += ["--pedal-dir", str(deck_args["pedal_dir"]), "--module", deck_args["module"],
                "--probe-node", deck_args["probe_node"], "--maxstep", str(deck_args["maxstep"])]
        for name, vals in knob_ranges.items():
            cmd += ["--range", f"{name}=" + ",".join(str(v) for v in vals)]
        if chain is None:
            cmd += ["--no-capture-chain"]
        else:
            cmd += ["--capture-hp-hz", str(chain["corner_hz"]), "--capture-order", str(chain["order"])]
    # Pass our own --workers through. This was missing: scaffold used --workers for its
    # truncation phase (12 concurrent renders) and then handed prepare_excitation.py nothing,
    # so onset measurement -- by far the longer phase on a many-knob circuit -- ran serially
    # on the same machine. Divided by 4 because find_saturation_point() fans out its own
    # amplitude sweep underneath; see --corner-workers' help for the product argument.
    cmd += ["--corner-workers", str(max(1, min(6, workers // 4)))]
    # Probe the grid INTERIOR too, not just corners -- see _interior_sample_budget.
    budget = _interior_sample_budget(n_knobs)
    if budget:
        cmd += ["--sample-grid", str(budget)]
        print(f"  probing {budget} interior grid point(s) on top of the corner set -- corners "
              f"alone cannot see a non-monotonic onset maximum (Mesa Orange's was 1.27x every "
              f"vertex)")
    if info.duration > sweep_dur_cap:
        print(f"  {sweep_file.name} is {info.duration:.1f}s, longer than the "
              f"{sweep_dur_cap:.0f}s sweep-content cap -- truncating to "
              f"{sweep_dur_cap:.0f}s (--sweep-dur {sweep_dur_cap:.0f}).")
        cmd += ["--sweep-dur", str(sweep_dur_cap)]
    print(f"  building a calibrated excitation (measuring saturation onset across the "
          f"knob grid's corners, then build_excitation.py) -- this renders, budget "
          f"real time for it ...")
    r = subprocess.run(cmd)
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schx", type=Path, default=None,
                    help="required for --backend livespice/ngspice. Optional for ngspice-deck "
                         "(--pedal-dir/--module are the ones that matter there) -- only used "
                         "for the supply-rail bound and config.json metadata if this device "
                         "happens to also have one.")
    ap.add_argument("--input", default="examples/T3K-sweep-v3.wav",
                    help="sweep/DI to measure truncation against and record as `input` "
                         "(default: examples/T3K-sweep-v3.wav -- NOT bundled, download it "
                         "from https://www.tone3000.com/create/capture; assumes you run "
                         "this from the repo root)")
    ap.add_argument("--output", type=Path, default=None,
                    help="config.toml path to write (default: <schx stem>.config.toml, or "
                         "<module>.config.toml for ngspice-deck, in the current directory)")
    ap.add_argument("--backend", choices=["livespice", "ngspice", "ngspice-deck"], default="livespice",
                    help="oversample auto-measurement only runs for livespice -- ngspice's "
                         "adaptive timestepping isn't tuned the same way (see "
                         "ngspice/README.md), and ngspice-deck has no supersample+decimate "
                         "step for it to measure at all (see --maxstep)")
    ap.add_argument("--pedal-dir", help="ngspice-deck: directory containing --module, added to "
                    "sys.path. Same convention as render_ngspice_deck.py/preflight.py/"
                    "gen_dataset_from_schx.py --backend ngspice-deck.")
    ap.add_argument("--module", help="ngspice-deck: module exposing build_deck/KNOB_NAMES, "
                    "e.g. gen_boss_od3_ngspice. KNOB_NAMES is used for control discovery "
                    "instead of parse_schx_controls().")
    ap.add_argument("--probe-node", default="OUT",
                    help="ngspice-deck: node/tap to render and measure (default: %(default)s)")
    ap.add_argument("--maxstep", type=float, default=3e-6,
                    help="ngspice-deck: initial max internal solver step, written into the "
                         "config (default: %(default)s) -- the equivalent of --oversample for "
                         "this backend.")
    _cc_add_cli_args(ap)
    ap.add_argument("--write-backend-sidecar", action="store_true",
                    help="if the chosen backend DIVERGES while measuring oversample, write the "
                         "<stem>.backends.toml recording it, instead of only reporting it. The "
                         "verdict is a measurement -- the scaffold just watched the circuit fail "
                         "-- so it does not need to be hand-written. Review before committing: a "
                         "divergence can be a probe artefact rather than a model instability.")
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
    ap.add_argument("--oversample-results", nargs="+", metavar="PATH",
                    help="skip the local oversample-measurement sweep and score from these "
                         "measure_truncation.py --emit files instead (e.g. collected by "
                         "distribute_pull.py --tool measure_truncation --collect). Use this "
                         "for a full amp, where the local sweep alone can run over an hour on "
                         "one box -- see measure_truncation.py's own docstring. --candidates/"
                         "--ref-os are still validated against what the shards were actually "
                         "measured with (a mismatch is a WARNING, not an error: the shards' "
                         "own values win).")
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
    ap.add_argument("--sweep-dur-cap", type=float, default=DEFAULT_SWEEP_DUR_CAP,
                    help=f"cap (seconds) on the excitation's sweep-file-derived portion "
                         f"when --input exceeds it (default {DEFAULT_SWEEP_DUR_CAP:.0f}s) "
                         f"-- build_excitation.py's own default is the WHOLE --sweep-file, "
                         f"uncapped, which that segment doesn't need (it only "
                         f"samples dynamics/perceptual variety; the chirps provide "
                         f"saturation coverage). No effect if --skip-prepare-excitation.")
    args = ap.parse_args()

    if args.backend == "ngspice-deck":
        if not (args.pedal_dir and args.module):
            ap.error("--pedal-dir and --module are required for --backend ngspice-deck")
        sys.path.insert(0, _os.path.abspath(args.pedal_dir))
        import importlib
        deck_mod = importlib.import_module(args.module)
        if not hasattr(deck_mod, "build_deck") or not hasattr(deck_mod, "KNOB_NAMES"):
            sys.exit(f"{args.module} must expose build_deck() and KNOB_NAMES")
        names = list(deck_mod.KNOB_NAMES)
        if not names:
            sys.exit(f"{args.module}.KNOB_NAMES is empty -- nothing to scaffold a grid for")
        print(f"  {len(names)} control(s) discovered in {args.module}.KNOB_NAMES: {', '.join(names)}")
        if args.schx is not None and not args.schx.exists():
            sys.exit(f"schx not found: {args.schx}")
        output = args.output or Path(f"{args.module}.config.toml")
    else:
        if args.schx is None:
            ap.error(f"--schx is required for --backend {args.backend}")
        if not args.schx.exists():
            sys.exit(f"schx not found: {args.schx}")
        controls = parse_schx_controls(str(args.schx))
        if not controls:
            sys.exit(f"no pots/switches found in {args.schx} -- nothing to scaffold a grid for")
        names = sorted(set(controls.values()))
        print(f"  {len(names)} control(s) discovered in {args.schx.name}: {', '.join(names)}")
        output = args.output or Path(f"{args.schx.stem}.config.toml")

    text = TEMPLATE.read_text()
    if args.backend == "ngspice-deck":
        if args.schx is not None:
            text = _replace_line(text, "schx",
                f'schx       = "{args.schx}"   # OPTIONAL for ngspice-deck -- not used for '
                f'rendering (--module below is), only the supply-rail bound + metadata')
        else:
            text = _replace_line(text, "schx",
                '# schx not used -- ngspice-deck renders via --module below, not a .schx')
        text = _replace_line(text, "backend", f'backend    = "{args.backend}"')
        text = _insert_line_after(text, "backend",
            f'pedal-dir  = "{args.pedal_dir}"\n'
            f'module     = "{args.module}"   # {len(names)} control(s) discovered by scaffold_config.py\n'
            f'probe-node = "{args.probe_node}"\n'
            f'maxstep    = {args.maxstep:g}   # equivalent of oversample for this backend -- see '
            f'gen_dataset_from_schx.py --maxstep help')
    else:
        text = _replace_line(text, "schx",
            f'schx       = "{args.schx}"   # {len(names)} control(s) discovered by scaffold_config.py')
        text = _replace_line(text, "backend", f'backend    = "{args.backend}"')
    text = _replace_line(text, "input", f'input      = "{args.input}"')

    if args.backend == "ngspice-deck":
        text = _replace_line(text, "oversample",
            "# oversample does not apply to --backend ngspice-deck (no supersample+decimate "
            "step in this path, unlike the schx-translated ngspice backend) -- maxstep above "
            "is the equivalent knob.")
    elif args.backend == "livespice":
        cands = tuple(int(c) for c in args.candidates.split(","))
        try:
            oversample, comment = _measure_oversample(
                args.schx, names, Path(args.input), cands, args.ref_os, args.probe_s, args.workers,
                oversample_results=args.oversample_results)
        except _BackendDiverged as e:
            # Do NOT write a config naming a backend we just watched fail. Everything after this
            # point -- prepare_excitation's corner sweep especially -- renders through the same
            # backend, so continuing would burn a lot of time producing a config for a grid of
            # crashes.
            _report_divergence(args.schx, args.backend, str(e), args.write_backend_sidecar)
            sys.exit(2)
        if oversample is None:
            oversample, comment = 8, "COULD NOT MEASURE (see warning above) -- placeholder"
        text = _replace_line(text, "oversample", f"oversample = {oversample}   # {comment}")
    # else: leave the template's placeholder oversample + comment untouched -- ngspice
    # tuning is a different question (see ngspice/README.md), not this tool's job.

    # Write the RESOLVED chain, not the template's literals: a --capture-* override passed
    # to this tool must end up in the config it generates, or the device is scaffolded with
    # one chain and rendered with another. prepare_excitation.py is invoked below with
    # --config pointing at this file, so writing it here is also how the onset measurement
    # gets the same chain -- no separate forwarding needed.
    _chain = _cc_resolve(args)
    if _chain is None:
        text = _replace_line(text, "capture-hp-hz", "no-capture-chain = true")
        text = _replace_line(text, "capture-order",
                             "# capture chain DISABLED for this device (--no-capture-chain)")
    else:
        text = _replace_line(text, "capture-hp-hz", f"capture-hp-hz = {_chain['corner_hz']:g}")
        text = _replace_line(text, "capture-order", f"capture-order = {_chain['order']}")
    print(f"  {_cc_describe(_chain)}")

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

    _excitation_backends = ("livespice", "ngspice-deck")
    if args.backend in _excitation_backends and not args.skip_prepare_excitation and not Path(args.input).exists():
        print(f"  WARNING: --input {args.input} not found -- skipping the calibrated-excitation "
              f"build, leaving `input` pointed at the raw path as given.")
    elif args.backend in _excitation_backends and not args.skip_prepare_excitation:
        excitation_wav = output.with_name(f"{output.stem}_excitation.wav")
        if args.backend == "ngspice-deck":
            # ngspice-deck's own corner-finder ignores --config for knob ranges (see
            # _prepare_excitation's docstring) -- hand it the SAME placeholder grid just
            # written into [knobs] explicitly, so the two cannot silently diverge.
            knob_ranges = {name: _grid_for_kind(classify(name)[0], args.grid_points) for name in names}
            deck_args = {"pedal_dir": args.pedal_dir, "module": args.module,
                         "probe_node": args.probe_node, "maxstep": args.maxstep}
            ok = _prepare_excitation(output, Path(args.input), excitation_wav,
                                     args.sweep_dur_cap, n_knobs=len(names), workers=args.workers,
                                     backend="ngspice-deck", deck_args=deck_args,
                                     knob_ranges=knob_ranges, chain=_chain)
        else:
            ok = _prepare_excitation(output, Path(args.input), excitation_wav,
                                     args.sweep_dur_cap, n_knobs=len(names), workers=args.workers)
        if ok and excitation_wav.exists():
            # prepare_excitation.py already pointed `input` here, and its line carries the
            # measured worst-case onset and corner count that this function does not have.
            # Rewriting it from out here just replaced those numbers with something vaguer.
            # Verify instead -- a config still naming the raw sweep is a silent trap, since
            # the raw sweep is a perfectly valid wav that simply never saturates the circuit.
            if str(excitation_wav) not in output.read_text():
                output.write_text(set_input_line(output.read_text(), excitation_wav))
                print(f"  NOTE: pointed `input` at {excitation_wav.name} from the scaffold "
                      f"(prepare_excitation.py did not)")
            print(f"  input -> {excitation_wav}\n"
                  f"  PROVISIONAL: sized against the PLACEHOLDER grid above. Re-run "
                  f"prepare_excitation.py after grid_adequacy.py --apply settles the real "
                  f"grid -- refining it changes the corner set, and so the worst-case onset "
                  f"the excitation has to reach.")
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
