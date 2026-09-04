#!/usr/bin/env python3
"""Wire measured saturation onset directly to excitation building -- backend-agnostic
(--backend {livespice,ngspice-deck}, see render_backends.py). Closes the manual human-in-the-loop
gap that's existed between find_saturation_point.py and build_excitation.py: until now,
someone had to read an onset number by hand and pick --realistic-peak/--sweep-peaks themselves
(this is literally how every existing config's excitation was sized, e.g. the non-midpoint-default pedal's "peak sized
from a direct Gain=0.5-vs-1.0 output-V-vs-input-V sweep" config comment).

Runs find_saturation_point() at EVERY corner of the knob grid (reusing check_transient_
coverage.py's own _corners() -- the same all-min/all-max/center/solo-extreme/full-hypercube
set that tool checks against, not just one hand-picked knob setting), takes the WORST-CASE
(highest) onset across them, derives --sweep-peaks (a staged ramp up to `--margin` x
worst-case onset) and --realistic-peak (a fraction of worst-case onset), and invokes
build_excitation.py with them. Refuses to build (raises) if any corner's onset can't be
determined, rather than silently building against a partial result -- same "refuse to guess"
convention as preflight.py/check_transient_coverage.py.

After building, running check_transient_coverage.py (livespice or --backend ngspice-deck) is
still worth doing as an independent gate before training -- and it is a REAL gate, not a
formality. This tool derives its levels from onset numbers measured at a SET OF PROBED POINTS,
and onset is not monotonic in the knobs, so the grid's true worst corner need not be one of
them: Mesa Dual Rectifier ORANGE's worst (23.177 V, Bass=min with the others centred) is 1.27x
the highest of all 32 hypercube vertices. --realistic-peak-frac (default 1.3) buys headroom
against that; --sample-grid N buys coverage of it. A clean check is expected, not guaranteed --
and both Mesa channels FAILED one on 2026-09-04 against an excitation whose peak was
hand-picked rather than measured at all.

Usage:
  livespice: python prepare_excitation.py --backend livespice \\
      --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml \\
      --real-clip examples/T3K-sweep-v3.wav --output ~/work/tmp/DEVICE_excitation.wav

  ngspice:   python prepare_excitation.py --backend ngspice-deck \\
      --pedal-dir ~/work/parametric-devices/pedals --module gen_ocd_ngspice \\
      --range "Gain=0.1,0.5,0.9" --range "Tone=0.2,0.5,0.8" --fixed-params "Volume=1.0" \\
      --real-clip ~/work/parametric-devices/pedals/ocd_realistic_clip.wav \\
      --output ~/work/tmp/ocd_excitation.wav
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_pipeline import load_config, set_input_line  # noqa: E402
from check_transient_coverage import _corners, _sample_interior  # noqa: E402
from find_saturation_point import find_saturation_point, findpeak_cache_key, scratch_dir  # noqa: E402
from render_backends import LiveSpiceBackend, NgspiceBackend, LtspiceBackend  # noqa: E402


def worst_case_onset(backend, identity, cache_extra, knob_ranges, fixed, tmp,
                      peak_max_v=40.0, no_cache=False, full_hypercube=None, quiet=False,
                      lead_silence_s=0.0, max_corners=None, sample_grid=0):
    """Find the worst-case (highest) saturation onset across every corner of knob_ranges.
    Reuses find_saturation_point.py directly (not check_transient_coverage.check_coverage --
    that function's pass/fail comparison against a transient_peak doesn't apply to this
    direction, only its per-corner onset detection would, so this calls the shared onset-
    finder and corner-generator directly instead of routing through a check meant for a
    different question). Raises if any corner's onset couldn't be determined -- see module
    docstring's "refuse to guess"."""
    corners = _corners(knob_ranges, full_hypercube=full_hypercube, max_corners=max_corners)
    corners = _sample_interior(knob_ranges, corners, sample_grid)
    if not quiet:
        # full_hypercube is now TRI-STATE (None = default/full, False = the deprecated
        # structural-only set, True = legacy callers), so a bare truthiness test mislabels the
        # default path: passing None printed "reduced hypercube set" for a run that had just
        # probed the full 32-vertex cube plus 64 interior points. Caught on the RED re-size,
        # 2026-09-04 -- harmless to the numbers, actively misleading to whoever reads the log.
        if full_hypercube is False:
            kind = "structural-only, DEPRECATED"
        elif sample_grid:
            kind = f"full binary hypercube + {sample_grid} interior grid point(s)"
        elif max_corners is not None:
            kind = f"budgeted, max {max_corners}"
        else:
            kind = "full binary hypercube"
        print(f"  {len(corners)} corners ({kind} set)")
    rows = []
    for label, vals in corners:
        params = dict(vals); params.update(fixed)
        cpath = findpeak_cache_key(identity, params, cache_extra)
        if cpath.exists() and not no_cache:
            sat = json.loads(cpath.read_text())
        else:
            sat = find_saturation_point(backend, params, tmp, max_v=peak_max_v,
                                         lead_silence_s=lead_silence_s)
            if sat is not None:
                cpath.write_text(json.dumps(sat))
        onset = sat.get("onset_99pct_input_v") if sat else None
        if not quiet:
            onset_str = "NONE (not reached)" if onset is None else f"{onset:.3f} V"
            print(f"  {label:16} onset={onset_str:>10}")
        rows.append({"corner": label, "params": params, "onset_v": onset})
    missing = [r for r in rows if r["onset_v"] is None]
    if missing:
        raise RuntimeError(f"no saturation onset found at corner(s) "
                            f"{[r['corner'] for r in missing]} -- refusing to build an "
                            f"excitation against a partial/failed result")
    worst = max(r["onset_v"] for r in rows)
    return worst, rows


def _parse_ranges(range_args):
    knob_ranges = {}
    for entry in range_args:
        name, vals = entry.split("=", 1)
        knob_ranges[name.strip()] = [float(v) for v in vals.split(",")]
    return knob_ranges


def _parse_fixed(fixed_str):
    fixed = {}
    for kv in filter(None, (s.strip() for s in (fixed_str or "").split(","))):
        k, v = kv.split("="); fixed[k.strip()] = float(v)
    return fixed


def _setup(args):
    """Returns (backend, identity, cache_extra, knob_ranges, fixed, lead_silence_s, label)."""
    if args.backend == "livespice":
        if args.config:
            cfg = load_config(Path(args.config))
            schx = str(cfg["schx"])
            oversample = args.oversample or cfg.get("oversample", 8)
            knob_ranges = _parse_ranges(cfg.get("ranges", []))
            fixed = _parse_fixed(cfg.get("fixed_params"))
        else:
            if not args.schx or not args.range:
                sys.exit("--backend livespice needs --config, or --schx + --range")
            schx = args.schx
            oversample = args.oversample or 8
            knob_ranges = _parse_ranges(args.range)
            fixed = _parse_fixed(args.fixed_params)
        if not knob_ranges:
            sys.exit("no [knobs]/--range entries -- nothing to check corners over")
        backend = LiveSpiceBackend(schx, oversample=oversample, iterations=args.iterations)
        identity = Path(schx).read_bytes()
        cache_extra = f"os={oversample}|it={args.iterations}|maxv={args.peak_max_v}"
        return backend, identity, cache_extra, knob_ranges, fixed, 0.0, Path(schx).name
    if args.backend == "ngspice-deck":
        if not (args.pedal_dir and args.module and args.range):
            sys.exit("--backend ngspice-deck needs --pedal-dir, --module, and --range")
        sys.path.insert(0, os.path.abspath(args.pedal_dir))
        mod = importlib.import_module(args.module)
        knob_ranges = _parse_ranges(args.range)
        fixed = _parse_fixed(args.fixed_params)
        backend = NgspiceBackend(mod.build_deck, probe_node=args.probe_node,
                                  maxstep=args.maxstep, parallel_sims=args.parallel_sims)
        identity = Path(mod.__file__).read_bytes()
        cache_extra = f"maxv={args.peak_max_v}"
        return backend, identity, cache_extra, knob_ranges, fixed, args.lead_silence_s, args.module
    if args.backend == "ltspice-deck":
        if not (args.pedal_dir and args.module and args.range):
            sys.exit("--backend ltspice-deck needs --pedal-dir, --module, and --range")
        sys.path.insert(0, os.path.abspath(args.pedal_dir))
        mod = importlib.import_module(args.module)
        knob_ranges = _parse_ranges(args.range)
        fixed = _parse_fixed(args.fixed_params)
        backend = LtspiceBackend(mod.build_deck, tap=args.probe_node,
                                 maxstep=args.maxstep, parallel_sims=args.parallel_sims,
                                 out_scale=args.out_scale, timeout=args.ltspice_timeout)
        identity = Path(mod.__file__).read_bytes()
        cache_extra = f"maxv={args.peak_max_v}"
        # No lead_silence_s: LTspice's .ic/uic hints replace the need for a cold-start
        # settling lead-in -- see ltspice_spicelib.py's docstring.
        return backend, identity, cache_extra, knob_ranges, fixed, 0.0, args.module
    sys.exit(f"unknown --backend {args.backend!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["livespice", "ngspice-deck", "ltspice-deck"])

    # livespice
    ap.add_argument("--config", help="[livespice] per-circuit TOML (same as run_pipeline.py --config)")
    ap.add_argument("--schx", help="[livespice] alternative to --config: circuit file directly")
    ap.add_argument("--oversample", type=int, default=None, help="[livespice] default: config's own, else 8")
    ap.add_argument("--iterations", type=int, default=256, help="[livespice]")

    # ngspice
    ap.add_argument("--pedal-dir", help="[ngspice-deck] directory containing --module, added to sys.path")
    ap.add_argument("--module", help="[ngspice-deck] module exposing build_deck, e.g. gen_ocd_ngspice")
    ap.add_argument("--probe-node", default="OUT", help="[ngspice-deck] node/tap to render and measure")
    ap.add_argument("--maxstep", type=float, default=3e-6, help="[ngspice-deck]")
    ap.add_argument("--parallel-sims", type=int, default=8, help="[ngspice-deck]")
    ap.add_argument("--ltspice-timeout", type=float, default=None,
                    help="[ltspice-deck] per-render wall ceiling in seconds. Default: scales with clip duration AND --parallel-sims (see ltspice_spicelib.default_timeout), deliberately generous because a too-short ceiling does not error -- it reports every render as a convergence failure. On hardware slower than this was tuned on (older CPU, spinning disk, throttled or busy machine) set LTSPICE_TIMEOUT_SCALE=<multiplier> rather than passing a number here per run.")
    ap.add_argument("--lead-silence-s", type=float, default=3.0,
                     help="[ngspice-deck] silence prepended before each saturation-sweep probe tone "
                          "-- see this repo's README ('Known issue: excitation needs a silent "
                          "lead-in')")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="keep this run's intermediate renders instead of deleting them on "
                         "exit (they go to ~/.cache/parametric-nam/prepare_excitation_scratch). For debugging "
                         "a bad render; off by default because these are write-only files "
                         "nothing reads back.")
    ap.add_argument("--out-scale", type=float, default=0.05,
                     help="[ltspice-deck] LTspice .wave output is +/-1V-PCM-bounded -- see "
                          "ltspice_spicelib.py's docstring")

    # shared corner/knob specification (both backends; --config covers this for livespice)
    ap.add_argument("--range", action="append", default=[],
                     help="NAME=v1,v2,... one knob's grid values, used to generate corners "
                          "(repeatable). Required for ngspice; alternative to --config for "
                          "livespice.")
    ap.add_argument("--fixed-params", default="", help="NAME=VAL,... held fixed at every corner")
    ap.add_argument("--no-full-hypercube", action="store_true",
                     help="DEPRECATED, prefer --max-corners. Drops to the structural corners only "
                          "(solo + all-min/all-max/center), which cannot represent a MIXED "
                          "some-knobs-low-others-high corner AT ALL -- the blind spot that shipped "
                          "the tweed blowup and, on 2026-09-04, sized Duke of Tone's excitation "
                          "short at three Gain=lo,Volume=lo corners. Warns when used.")
    ap.add_argument("--sample-grid", type=int, default=0, metavar="N",
                     help="ALSO probe N points from the full grid product, not just the min/max "
                          "hypercube -- see check_transient_coverage.py's _sample_interior for "
                          "the Mesa Orange case where the true worst onset sits 27%% above every "
                          "vertex, at Bass=min with the other knobs CENTRED. Deterministic, so a "
                          "later coverage check probes the same points.")
    ap.add_argument("--max-corners", type=int, default=None,
                     help="cap the TOTAL corner count, sampling the hypercube deterministically "
                          "when it does not all fit, instead of abandoning it. Use this rather "
                          "than --no-full-hypercube for a many-knob device: same budget, but it "
                          "still reaches mixed corners (a 16-knob config at --max-corners 48 gets "
                          "13 of them; --no-full-hypercube gets 0).")
    ap.add_argument("--peak-max-v", type=float, default=40.0,
                     help="find_saturation_point sweep ceiling -- the 40V default suits an "
                          "amp; lower it (e.g. 3-5) for a small pedal circuit")
    ap.add_argument("--no-cache", action="store_true")

    # excitation-building
    ap.add_argument("--real-clip", required=True,
                     help="real-playing clip, passed through to build_excitation.py --input")
    ap.add_argument("--output",
                    help="where to write the excitation wav (its .recipe.json sidecar goes "
                         "beside it -- every consumer finds the sidecar by deriving it from "
                         "the wav's own path, so they must stay together). Required unless "
                         "--workspace is given.")
    ap.add_argument("--no-update-config", action="store_true",
                    help="do not point --config's `input` at the excitation just built. Off "
                         "by default: leaving a config naming a superseded excitation is how "
                         "Mesa ORANGE and RED trained against transient content that never "
                         "reached saturation at 6/43 and 4/43 of their corners.")
    ap.add_argument("--workspace", type=Path,
                    help="write the excitation to <workspace>/excitation/excitation.wav, the "
                         "same layout run_pipeline.py --workspace uses. This argument exists "
                         "because --output had no default at all: the measured Mesa RED "
                         "excitation -- the only copy of its 104-corner sizing measurement -- "
                         "was written to /tmp on a worker and came within a cleanup of being "
                         "lost (2026-09-04). A run's excitation belongs with the run.")
    ap.add_argument("--margin", type=float, default=2.0,
                     help="sweep-peaks max = margin x worst-case onset (default 2.0x -- past "
                          "the onset, not just at it, matching this repo's own precedent, e.g. "
                          "the non-midpoint-default pedal's excitation peak sized with headroom past where Gain's own "
                          "effect saturates)")
    ap.add_argument("--sweep-peak-fracs", default="0.25,0.5,0.75,1.0",
                     help="comma list of fractions of the margined max, passed as --sweep-peaks")
    ap.add_argument("--realistic-peak-frac", type=float, default=1.3,
                     help="fraction of worst-case onset used for --realistic-peak. Never go BELOW "
                          "1.0: check_transient_coverage.py's own default "
                          "margin requires transient_peak >= onset AT THE WORST CORNER, and the "
                          "worst corner's own onset IS worst-case onset by definition -- any "
                          "fraction below 1.0 guarantees that check fails there, regardless of "
                          "margin or grid, contradicting this tool's own claim that a check run "
                          "afterward should pass cleanly. Lower it only if you deliberately want "
                          "the real-playing content to stay short of the worst corner (e.g. to "
                          "match a case where a real player realistically never drives that hard) "
                          "and are prepared for check_transient_coverage.py to FAIL there as a "
                          "correct, expected result, not a bug.\n"
                          "The default is 1.3, and the history is worth knowing. It was 1.0 -- "
                          "sizing EXACTLY at the measured worst onset -- which leaves zero slack "
                          "against a check comparing transient_peak >= onset, so any "
                          "re-measurement at a different oversample, or any corner the sizing run "
                          "did not probe, fails by a hair; Duke of Tone did exactly that on "
                          "2026-09-04 (8.647 V sized, 8.766 V found). 1.02 fixed the hair. It "
                          "does NOT fix the real hazard, which is the measured worst being below "
                          "the TRUE worst: onset is not monotonic in the knobs, so its maximum "
                          "over the grid need not sit at any probed point. Measured the same day "
                          "on Mesa Dual Rectifier ORANGE -- true worst 23.177 V at Bass=min with "
                          "every other knob CENTRED, 1.27x the highest of all 32 hypercube "
                          "vertices (18.232 V). 1.3 covers that observed ratio from vertex data "
                          "alone, at no extra probing cost. It is insurance, not a substitute for "
                          "--sample-grid: a hotter excitation drives every ALREADY-covered corner "
                          "further into saturation, so do not inflate it beyond what the "
                          "non-monotonicity actually demands.")
    ap.add_argument("--realistic-dur", type=float, default=None)
    ap.add_argument("--synth-burst-peaks", default=None,
                    help="passed through to build_excitation.py. 'auto' uses the derived "
                         "--sweep-peaks, so a broadband instant-attack burst is inserted at "
                         "EVERY level -- the reverse-linear-drive pedal shipped a model that spiked to 12.39 "
                         "peak on a real pick attack because its excitation never showed it a "
                         "stable response to one. Default off, preserving prior behaviour.")
    ap.add_argument("--synth-burst-dur", type=float, default=None,
                    help="passed through to build_excitation.py with --synth-burst-peaks")
    ap.add_argument("--excitation-lead-silence-s", type=float, default=3.0,
                     help="--lead-silence-s passed to build_excitation.py itself (distinct "
                          "from --lead-silence-s above, which is for the ngspice sweep probe)")
    args = ap.parse_args()
    if args.workspace and not args.output:
        d = args.workspace.expanduser() / "excitation"
        d.mkdir(parents=True, exist_ok=True)
        args.output = str(d / "excitation.wav")
        print(f"workspace {args.workspace}: --output {args.output}")
    elif not args.output:
        ap.error("--output is required (or pass --workspace to place it for you)")

    backend, identity, cache_extra, knob_ranges, fixed, sweep_lead_silence_s, label = _setup(args)

    print(f"finding saturation onset across the knob-grid corners of {label}...")
    tmp = str(scratch_dir("prepare_excitation", args.keep_scratch))
    worst, rows = worst_case_onset(backend, identity, cache_extra, knob_ranges, fixed, tmp,
                                    peak_max_v=args.peak_max_v, no_cache=args.no_cache,
                                    full_hypercube=(False if args.no_full_hypercube else None),
                                    max_corners=args.max_corners, sample_grid=args.sample_grid,
                                    quiet=False,
                                    lead_silence_s=sweep_lead_silence_s)
    print(f"worst-case onset: {worst:.4f} V (across {len(rows)} corners)")

    sweep_max = worst * args.margin
    fracs = [float(f) for f in args.sweep_peak_fracs.split(",") if f.strip()]
    sweep_peaks = [round(sweep_max * f, 4) for f in fracs]
    realistic_peak = round(worst * args.realistic_peak_frac, 4)
    # NOTE the default frac is 1.02, not 1.0 -- see --realistic-peak-frac's help for why sizing
    # EXACTLY at the measured worst is too tight to survive a re-measurement.
    print(f"derived: sweep_peaks={sweep_peaks}  realistic_peak={realistic_peak}  "
          f"(margin={args.margin}x onset)")

    build_script = HERE / "build_excitation.py"
    cmd = [sys.executable, str(build_script),
           "--input", args.real_clip, "--output", args.output,
           "--realistic-peak", str(realistic_peak),
           "--sweep-peaks", ",".join(str(p) for p in sweep_peaks),
           "--lead-silence-s", str(args.excitation_lead_silence_s)]
    if args.realistic_dur is not None:
        cmd += ["--realistic-dur", str(args.realistic_dur)]
    if args.synth_burst_peaks:
        # "auto" mirrors the derived sweep levels, which is what build_excitation.py's own
        # help recommends ("Typically the same list as --sweep-peaks") -- so saturation-onset
        # behaviour under a sharp transient is tested at every level rather than only the
        # loudest, which is the gap a single --noise-burst-* segment leaves.
        peaks = (",".join(str(p) for p in sweep_peaks)
                 if args.synth_burst_peaks == "auto" else args.synth_burst_peaks)
        cmd += ["--synth-burst-peaks", peaks]
        if args.synth_burst_dur is not None:
            cmd += ["--synth-burst-dur", str(args.synth_burst_dur)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # SIZING PROVENANCE. build_excitation.py records WHAT it built (args, source hash, output
    # hash); it cannot record WHY those numbers, because the measurement that justified them
    # happened here. Without this block a later check_transient_coverage.py failure is
    # unexplainable from the artifacts alone: you see realistic_peak=8.6473 and an onset of
    # 8.766 V and cannot tell whether the sizing run simply never probed that corner. That is
    # exactly what happened to Duke of Tone on 2026-09-04 -- sized against the reduced 11-corner
    # set, checked against the full 25, three mixed Gain=lo,Volume=lo corners missed by 0.3-1.4%
    # and nothing on disk said the two runs had used different corner sets.
    recipe_path = Path(args.output).with_suffix(".recipe.json")
    if recipe_path.exists():
        try:
            recipe = json.loads(recipe_path.read_text())
            recipe["sizing"] = {
                "tool": "prepare_excitation.py",
                "worst_case_onset_v": round(float(worst), 4),
                "corner_count": len(rows),
                "corner_set": ("structural-only (DEPRECATED --no-full-hypercube: cannot represent "
                               "mixed low/high corners)" if args.no_full_hypercube
                               else f"budgeted (--max-corners {args.max_corners})"
                               if args.max_corners is not None else "full binary hypercube"),
                "realistic_peak_frac": args.realistic_peak_frac,
                "sweep_margin": args.margin,
                "peak_max_v": args.peak_max_v,
                "onsets_v": {r["corner"]: (None if r["onset_v"] is None else round(float(r["onset_v"]), 4))
                             for r in rows},
            }
            recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
            print(f"recorded sizing provenance into {recipe_path.name} "
                  f"(worst {worst:.4f} V across {len(rows)} corners)")
        except Exception as e:
            print(f"WARNING: could not record sizing provenance into {recipe_path.name}: {e}",
                  file=sys.stderr)
    else:
        print(f"WARNING: {recipe_path.name} not found after build_excitation.py -- sizing "
              f"provenance NOT recorded", file=sys.stderr)

    # Point the config at what we just built. Only scaffold_config.py used to do this, so
    # re-sizing an excitation after grid_adequacy.py settled the real grid left `input` still
    # naming the superseded file -- silently, because both are valid wavs, and the whole
    # reason to re-size is that the old one no longer covers the grid. Mesa ORANGE and RED
    # trained against exactly that stale pointer and failed 6/43 and 4/43 corners.
    if args.config and not args.no_update_config:
        cfg = Path(args.config)
        try:
            cfg.write_text(set_input_line(
                cfg.read_text(), args.output,
                f"sized against this config's grid AS IT WAS -- worst-case onset "
                f"{worst:.4f} V across {len(rows)} corners (see the recipe.json sidecar). "
                f"CHANGING THE GRID INVALIDATES THIS: re-run prepare_excitation.py after "
                f"grid_adequacy.py --apply, or corners the new grid reaches will have no "
                f"transient content"))
            print(f"updated {cfg.name}: input -> {args.output}")
        except Exception as e:
            print(f"WARNING: could not update {cfg.name}'s `input` -- point it at "
                  f"{args.output} by hand: {e}", file=sys.stderr)

    print(f"wrote {args.output} -- built from a measured onset (worst-case {worst:.4f} V "
          f"across {len(rows)} corners), not a guess. For ngspice, render with your "
          f"render_*.py's --absolute flag (no --vin rescaling) to preserve this file's "
          f"intentional multi-level structure; for livespice, run "
          f"check_transient_coverage.py against the same config as an independent gate.")


if __name__ == "__main__":
    main()
