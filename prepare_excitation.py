#!/usr/bin/env python3
"""Wire measured saturation onset directly to excitation building -- backend-agnostic
(--backend {livespice,ngspice-deck}, see render_backends.py). Closes the manual human-in-the-loop
gap that's existed between find_saturation_point.py and build_excitation.py: until now,
someone had to read an onset number by hand and pick --realistic-peak/--sweep-peaks themselves
(this is literally how every existing config's excitation was sized, e.g. Timmy's "peak sized
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
still worth doing as an independent gate before training -- this tool derives its levels FROM
the same onset numbers such a check would verify against, so a clean result is expected, not a
coincidence, but it isn't a substitute for actually checking. This is only true at
--realistic-peak-frac's default of 1.0 (not less) -- see that flag's own help text.

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
from run_pipeline import load_config  # noqa: E402
from check_transient_coverage import _corners  # noqa: E402
from find_saturation_point import find_saturation_point, findpeak_cache_key  # noqa: E402
from render_backends import LiveSpiceBackend, NgspiceBackend, LtspiceBackend  # noqa: E402


def worst_case_onset(backend, identity, cache_extra, knob_ranges, fixed, tmp,
                      peak_max_v=40.0, no_cache=False, full_hypercube=True, quiet=False,
                      lead_silence_s=0.0):
    """Find the worst-case (highest) saturation onset across every corner of knob_ranges.
    Reuses find_saturation_point.py directly (not check_transient_coverage.check_coverage --
    that function's pass/fail comparison against a transient_peak doesn't apply to this
    direction, only its per-corner onset detection would, so this calls the shared onset-
    finder and corner-generator directly instead of routing through a check meant for a
    different question). Raises if any corner's onset couldn't be determined -- see module
    docstring's "refuse to guess"."""
    corners = _corners(knob_ranges, full_hypercube=full_hypercube)
    if not quiet:
        print(f"  {len(corners)} corners ({'full binary hypercube' if full_hypercube else 'reduced hypercube'} set)")
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
            print(f"  {label:16} onset={onset if onset is None else f'{onset:.3f} V':>10}")
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
                     help="skip the full 2**n min/max hypercube (only solo + all-min/all-max/"
                          "center corners) -- use for many knobs where 2**n would be impractical")
    ap.add_argument("--peak-max-v", type=float, default=40.0,
                     help="find_saturation_point sweep ceiling -- the 40V default suits an "
                          "amp; lower it (e.g. 3-5) for a small pedal circuit")
    ap.add_argument("--no-cache", action="store_true")

    # excitation-building
    ap.add_argument("--real-clip", required=True,
                     help="real-playing clip, passed through to build_excitation.py --input")
    ap.add_argument("--output", required=True)
    ap.add_argument("--margin", type=float, default=2.0,
                     help="sweep-peaks max = margin x worst-case onset (default 2.0x -- past "
                          "the onset, not just at it, matching this repo's own precedent, e.g. "
                          "Timmy's excitation peak sized with headroom past where Gain's own "
                          "effect saturates)")
    ap.add_argument("--sweep-peak-fracs", default="0.25,0.5,0.75,1.0",
                     help="comma list of fractions of the margined max, passed as --sweep-peaks")
    ap.add_argument("--realistic-peak-frac", type=float, default=1.0,
                     help="fraction of worst-case onset used for --realistic-peak. Default 1.0 "
                          "(not less) is load-bearing: check_transient_coverage.py's own default "
                          "margin requires transient_peak >= onset AT THE WORST CORNER, and the "
                          "worst corner's own onset IS worst-case onset by definition -- any "
                          "fraction below 1.0 guarantees that check fails there, regardless of "
                          "margin or grid, contradicting this tool's own claim that a check run "
                          "afterward should pass cleanly. Lower it only if you deliberately want "
                          "the real-playing content to stay short of the worst corner (e.g. to "
                          "match a case where a real player realistically never drives that hard) "
                          "and are prepared for check_transient_coverage.py to FAIL there as a "
                          "correct, expected result, not a bug.")
    ap.add_argument("--realistic-dur", type=float, default=None)
    ap.add_argument("--synth-burst-peaks", default=None,
                    help="passed through to build_excitation.py. 'auto' uses the derived "
                         "--sweep-peaks, so a broadband instant-attack burst is inserted at "
                         "EVERY level -- the Boss DS-1 shipped a model that spiked to 12.39 "
                         "peak on a real pick attack because its excitation never showed it a "
                         "stable response to one. Default off, preserving prior behaviour.")
    ap.add_argument("--synth-burst-dur", type=float, default=None,
                    help="passed through to build_excitation.py with --synth-burst-peaks")
    ap.add_argument("--excitation-lead-silence-s", type=float, default=3.0,
                     help="--lead-silence-s passed to build_excitation.py itself (distinct "
                          "from --lead-silence-s above, which is for the ngspice sweep probe)")
    args = ap.parse_args()

    backend, identity, cache_extra, knob_ranges, fixed, sweep_lead_silence_s, label = _setup(args)

    print(f"finding saturation onset across the knob-grid corners of {label}...")
    tmp = os.path.expanduser("~/.cache/parametric-nam/prepare_excitation_scratch")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    worst, rows = worst_case_onset(backend, identity, cache_extra, knob_ranges, fixed, tmp,
                                    peak_max_v=args.peak_max_v, no_cache=args.no_cache,
                                    full_hypercube=not args.no_full_hypercube, quiet=False,
                                    lead_silence_s=sweep_lead_silence_s)
    print(f"worst-case onset: {worst:.4f} V (across {len(rows)} corners)")

    sweep_max = worst * args.margin
    fracs = [float(f) for f in args.sweep_peak_fracs.split(",") if f.strip()]
    sweep_peaks = [round(sweep_max * f, 4) for f in fracs]
    realistic_peak = round(worst * args.realistic_peak_frac, 4)
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

    print(f"wrote {args.output} -- built from a measured onset (worst-case {worst:.4f} V "
          f"across {len(rows)} corners), not a guess. For ngspice, render with your "
          f"render_*.py's --absolute flag (no --vin rescaling) to preserve this file's "
          f"intentional multi-level structure; for livespice, run "
          f"check_transient_coverage.py against the same config as an independent gate.")


if __name__ == "__main__":
    main()
