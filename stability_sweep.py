#!/usr/bin/env python3
"""Which knob combinations does the SOLVER fail on? Answer it before the expensive render.

THE GAP THIS FILLS.
Four tools already gate a dataset, and none of them checks solver stability across the grid:

    preflight.py                 knob dead / reversed / input-level -- a handful of probes
    check_transient_coverage.py  does the excitation reach saturation -- 13-43 corners
    grid_adequacy.py             per-cell interpolation error -- ~50 probes
    measure_truncation.py        BDF2 truncation vs oversample -- a few settings

Newton stability is only discovered DURING the render, by the solver-spike detector and its
retry ladder -- which works, but you learn it hours in and pay 2-8x on every combination that
escalates.

MEASURED, on the Mesa Dual Rectifier RED (sag v30), 2026-09-01. 36 of 448 combinations tripped
the ladder mid-render; 12 escalated all the way to oversample=64, i.e. 8x the configured
timestep cost, AFTER a wasted attempt at 8. Every one of the 12 had the same signature:

    RD Gain = 0.1 or 0.15   AND   Red Master = 1.0

Minimum preamp gain into a fully-open master -- a tiny signal driving the power/sag stage wide
open. The device's own validation note said "converges at --oversample 8", which was true of
the 4 corners it tested (all-max, all-min, mixed-extreme, all-mid); this mixed extreme was not
among them. A 4-corner spot check cannot speak for a 448-cell grid.

WHY A SHORT PROBE IS ENOUGH, and this is the whole reason the sweep is affordable. The observed
spike sample indices clustered hard:

    0.93, 0.98, 0.98, 1.02, 1.05, 1.05, 1.05, 1.05 s     <- 8 of 10
    10.26, 10.56 s                                        <- 2 of 10

The excitation opens with 1 s of lead silence, so the dominant cluster is the FIRST TRANSIENT
AFTER SILENCE: the solver settles to quiescence, then gets hit. A few seconds spanning that
boundary provokes the same failure as the full clip. At 448 combinations that is the difference
between ~0.4 h and ~7 h.

    448 combos, 3 s probe, 4 machines / 48 cores    ~0.2 h
    448 combos, 8 s probe, 4 machines / 48 cores    ~0.4 h
    the real render, 120.5 s                        ~7 h  (before escalations)

USE --no-retry (the default here). The question is whether the CONFIGURED oversample is stable,
not whether the ladder can rescue it -- a rescued render still cost you the failed attempt.

Usage:
    # sweep the whole grid at the configured oversample
    ./stability_sweep.py --config device.toml --probe-s 3

    # compare candidate oversamples, and print the cheapest stable one
    ./stability_sweep.py --config device.toml --probe-s 3 --oversample 8,16,32

    # one worker's slice, for distribute_gen.sh-style sharding
    ./stability_sweep.py --config device.toml --shard 0-11/48
"""
import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shard import select as shard_select  # noqa: E402


def load_config(p: Path) -> dict:
    import tomllib
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def make_probe(src: Path, probe_s: float, out: Path) -> float:
    """Take the FIRST probe_s seconds -- deliberately, not a loud slice. The silence->signal
    onset is where the observed spikes clustered, and build_excitation.py always puts its lead
    silence first, so the opening seconds span exactly that boundary."""
    x, sr = sf.read(str(src))
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(probe_s * sr)
    sf.write(str(out), x[:n], sr)
    return float(np.abs(x[:n]).max())


def _recommend(oss, summary, cheapest, n_total, is_sharded):
    """Compare UNIFORM escalation against the RETRY LADDER and recommend the cheaper one.

    THE BUG THIS FIXES. This tool used to recommend raising the configured oversample the
    moment it saw ANY instability, with no arithmetic behind it. That is wrong most of the
    time, and it was wrong on the run that prompted this fix: it measured 17/48 unstable
    (35%) on a deliberately-chosen worst-corner subset and said "raise 8 -> 16". The grid
    was then rendered at oversample 8 with the ladder, and only 8 of 648 combinations
    (1.2%) ever escalated. Uniform 16 would have doubled 648 renders to avoid 8.

    THE COST MODEL, in units of one render at the base oversample. A combination that only
    succeeds at rung k has paid for every rung below it too, so it costs
    1 + 2 + 4 + ... + 2^k = 2^(k+1) - 1:

        uniform at the cheapest stable oversample U:  n_total * (U / base)
        retry ladder from base, failing fraction f:   n_total * (1 + 2f)

    Setting those equal gives f = 0.5: THE LADDER WINS WHENEVER FEWER THAN HALF THE
    COMBINATIONS FAIL. Escalation has to be the common case, not the exception, before
    paying for it up front is worth it.

    WHY f IS EASY TO OVERESTIMATE. f is measured on whatever grid the config describes.
    Probe the hard corners deliberately -- the sensible thing to do -- and f describes the
    corners, not the grid. Say so, rather than letting a corner rate be read as a grid rate.
    """
    base = oss[0]
    n_ok, n_tot, _ = summary[base]
    f = (n_tot - n_ok) / n_tot if n_tot else 0.0
    ladder_cost = n_total * (1 + 2 * f)
    uniform_cost = n_total * (cheapest / base)
    print(f"  failure rate at oversample {base}: {f*100:.1f}%  ({n_tot - n_ok}/{n_tot})")
    print(f"    uniform oversample {cheapest}:  {uniform_cost:6.0f} render-units")
    print(f"    retry ladder from {base}:       {ladder_cost:6.0f} render-units")
    if ladder_cost <= uniform_cost:
        print(f"  -> KEEP oversample {base}; let the retry ladder escalate the failures. "
              f"Cheaper here, and it stays cheaper until the failure rate exceeds 50%.")
    else:
        print(f"  -> RAISE the config from {base} to {cheapest}: at {f*100:.0f}% failures, "
              f"paying {cheapest//base}x up front beats the ladder's wasted attempts.")
    if is_sharded or n_total < 100:
        print(f"  CAVEAT: that rate is for the {n_total} combination(s) THIS config sweeps. "
              f"If it is a corner subset or a shard it is NOT the grid-wide rate, which is "
              f"usually far lower -- and the ladder correspondingly better. Sample the full "
              f"grid (e.g. --shard 0-0/8 against the real config) before trusting it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="the same TOML the pipeline uses")
    ap.add_argument("--probe-s", type=float, default=3.0,
                    help="seconds from the START of the excitation (default: %(default)s). The "
                         "opening seconds span the lead-silence->signal onset, which is where "
                         "spikes cluster; a loud mid-clip slice misses it.")
    ap.add_argument("--oversample", default=None,
                    help="comma-separated candidates to compare, e.g. 8,16,32. Default: the "
                         "config's own single value.")
    ap.add_argument("--shard", default=None, metavar="LOW-HIGH/TOTAL",
                    help="render only this slice of the grid -- same contract as the renderers "
                         "(shard.py), so one sweep can be split across a fleet")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write per-combination results (knobs, oversample, ok, spikes) to CSV")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("backend", "livespice") != "livespice":
        sys.exit("stability_sweep only supports the livespice backend "
                 "(the deck renderers have their own retry ladder and no spike detector)")
    knobs = cfg["knobs"]
    fixed = cfg.get("fixed", {})
    schx = os.path.expanduser(str(cfg["schx"]))
    src = Path(os.path.expanduser(str(cfg["input"])))
    oss = ([int(v) for v in args.oversample.split(",")] if args.oversample
           else [int(cfg.get("oversample", 8))])

    combos = list(itertools.product(*knobs.values()))
    indexed = list(enumerate(combos))
    if args.shard:
        before = len(indexed)
        indexed, *_ = shard_select(indexed, args.shard)
        print(f"shard {args.shard}: {len(indexed)}/{before} combinations")

    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe.wav"
        pk = make_probe(src, args.probe_s, probe)
        print(f"probe: first {args.probe_s:g} s of {src.name}, peak {pk:.2f} V")
        print(f"grid : {' x '.join(str(len(v)) for v in knobs.values())} = {len(combos)} "
              f"combination(s); sweeping {len(indexed)}")
        print(f"fixed: {fixed}\n")

        rows, summary = [], {}
        for os_ in oss:
            outdir = Path(td) / f"os{os_}"
            cmd = [sys.executable, str(HERE / "gen_dataset_from_schx.py"),
                   "--backend", "livespice", "--schx", schx,
                   "--knobs", ",".join(knobs.keys()),
                   "--oversample", str(os_), "--input", str(probe),
                   "--output", str(outdir), "--workers", str(args.workers),
                   "--skip-transient-check", "--no-retry"]
            for k, v in knobs.items():
                cmd += ["--range", f"{k}={','.join(str(x) for x in v)}"]
            if fixed:
                cmd += ["--fixed-params", ",".join(f"{k}={v}" for k, v in fixed.items())]
            if args.shard:
                cmd += ["--shard", args.shard]
            r = subprocess.run(cmd, capture_output=True, text=True)
            log = r.stdout + r.stderr
            n_spike = log.count("solver spike")
            n_ok = len(list((outdir / "sig").rglob("*.npy"))) if (outdir / "sig").exists() else 0
            summary[os_] = (n_ok, len(indexed), n_spike)
            print(f"  oversample={os_:<3d}  {n_ok}/{len(indexed)} converged  "
                  f"{n_spike} solver spike(s)"
                  + ("   STABLE" if n_ok == len(indexed) and n_spike == 0 else "   UNSTABLE"))
            # per-combination detail, so an unstable REGION is visible rather than just a count
            # The index is in the combo_NNNNNN token, NOT the leading [n/total] progress
            # counter -- parsing the latter silently yielded zero rows the first time.
            #   [   1/8]  12.5%  combo_000002  FAIL ... [solver spike: ...
            for line in log.splitlines():
                if "solver spike" not in line:
                    continue
                m = re.search(r"combo_(\d+)", line)
                if not m:
                    continue
                i = int(m.group(1))
                if i >= len(combos):
                    continue
                    rows.append({**dict(zip(knobs.keys(), combos[i])),
                                 "index": i, "oversample": os_, "ok": 0, "note": line.strip()[:120]})

    print()
    stable = [o for o in oss if summary[o][0] == summary[o][1] and summary[o][2] == 0]
    if stable:
        print(f"  cheapest STABLE oversample: {min(stable)}")
        if min(stable) != oss[0]:
            _recommend(oss, summary, min(stable), len(indexed), bool(args.shard))
    else:
        print("  NO candidate was fully stable. Either widen --oversample, or the instability "
              "is not a timestep problem -- check the unstable region below for a pattern.")
    if rows:
        ks = list(knobs.keys())
        seen = {}
        for r in rows:
            seen.setdefault(tuple(r[k] for k in ks), []).append(r["oversample"])
        print(f"\n  unstable combinations ({len(seen)} distinct):")
        for combo, at in sorted(seen.items())[:20]:
            print("    %s  at oversample %s" % (dict(zip(ks, combo)), sorted(set(at))))
        if len(seen) > 20:
            print(f"    ... and {len(seen)-20} more")
    if args.output and rows:
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n  wrote {args.output} ({len(rows)} row(s))")
    sys.exit(0 if stable else 1)


if __name__ == "__main__":
    main()
