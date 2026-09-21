#!/usr/bin/env python3
"""Measure BDF2 truncation error for one device.

WHY THIS FILE EXISTS AS A FILE.
The first fleet-wide truncation table (internal engineering notes, commit 9d986c5) was produced by an
ad-hoc script that was never committed. When its Large Muffin number (2.25e-02) later failed to
reproduce -- a careful re-measurement on the same input, same circuit, same worst corner gives
9.33e-03 -- there was nothing to audit and no way to find out which of the two was wrong. A
measurement that decides whether we retrain the entire model fleet has to be re-runnable by
someone who doubts it. So it lives here, in the repo, and the docs quote it rather than the
other way round.

WHAT IT MEASURES.
oversample is a DISCRETISATION choice. LiveSPICE integrates with BDF2, which is O(h^2), so the
simulated circuit is not quite the real circuit; the gap is TRUNCATION ERROR. It is not a bug and
it is never zero. But its SIZE decides whether it matters: if the truncation in the training target
is the same order as the ESR of the model you fit to that target, the model is spending capacity
learning the integrator's mistakes. A model cannot be more right than its target.

HOW (and the trap).
Against a REFERENCE at high oversample -- NOT against the next rung up. ESR(os, 2*os) is the
tempting cheap estimate and it UNDERSTATES the error ~3x, which is the dangerous direction: you
pick too low an oversample and ship a contaminated target. The finer render is not the truth, it
is merely less wrong, and the two errors are correlated and largely cancel. Measured on the Big
Muff, worst corner, whole file:

    os      ESR vs 2*os      ESR vs os=32     understated by
     2       2.83e-03         9.33e-03            3.3x
     4       6.45e-04         1.98e-03            3.1x
     8       1.63e-04         3.70e-04            2.3x

All renders at 256 Newton iterations, so under-convergence is not a confound (that is a different
failure and internal engineering notes cover it).

KNOB SETTINGS. Both ends of every knob, plus both corners, plus the mid point -- and we report
WHERE the worst was. "All knobs at max" is not reliably the stiff setting: the reverse-linear-drive pedal's Dist pot
is ReverseLinear, so all-max is MINIMUM drive.

Usage:
    ./measure_truncation.py --input ../sweep-files/sweep60_composite.wav \
        --config ../parametric-nam-models/pedals/mypedal/config.toml

`--config` is the same TOML `run_pipeline.py`/`grid_adequacy.py` take (schx + knobs come straight
from it, like every other analysis tool here) -- no registry, no `parametric-devices` dependency.
One device per invocation, on purpose: a fleet-wide table is rare (it matters only when
re-deriving the shared oversample floor, or auditing every device after a measurement-methodology
change; see internal engineering notes) and is just as easy from the outside:

    for f in ../parametric-nam-models/*/*/config.toml*; do
        ./measure_truncation.py --input ../sweep-files/sweep60_composite.wav --config "$f"
    done
"""
from __future__ import annotations

import argparse
import json
import os as _os
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf

from gen_dataset_from_schx import _livespice_batch, check_oracle, write_probe_clip


def esr_terms(a: np.ndarray, b: np.ndarray, lead_n: int, sr: int) -> tuple[float, float]:
    """(numerator, denominator) of the ESR of `a` against reference `b`, dropping the
    leading transient. Returned as terms, not a ratio, so window measurements can be
    POOLED: sum(err)/sum(sig) across windows is exactly the whole-file ESR restricted
    to the sampled windows (see choose_oversample). On a whole-file measurement
    (lead_n=0) the skip reduces to the historical 10%."""
    m = min(len(a), len(b))
    sk = max(m // 10, lead_n + int(0.02 * sr))
    if m - sk < 4800:
        return 0.0, 0.0
    return (float(np.sum((a[sk:m] - b[sk:m]) ** 2)),
            float(np.sum(b[sk:m] ** 2)))


def probe_clips(input_wav: Path, probe_s: float, n_windows: int, td: Path, lead_s: float = None
                ) -> tuple[list[Path], int, int]:
    """Cut stratified probe windows from the input; returns (clips, lead_n, sr).

    probe_s <= 0 means measure the WHOLE file (the historical behavior): one 'window'
    that is the input itself, no lead-in.

    Truncation is a property of the solver at a knob setting, not of sweep length --
    but WHERE you probe decides the answer (our sweeps open quiet and end in a decay
    tail), so this mirrors choose_oversample exactly: evenly-spaced window CENTRES
    across the file, each window rendered separately with write_probe_clip's silent
    lead-in + ramp (never spliced -- a splice manufactures step discontinuities and
    the probe then measures its own splice), numerator/denominator pooled across
    windows. At the default 10 s of a 60 s sweep this cuts the render bill ~6x, and
    the reference renders (16x the cost of an os=2 render each) with it.

    `lead_s`: forwarded to write_probe_clip -- override for a circuit whose own settling
    time exceeds write_probe_clip's 1.0 s default (see that function's docstring).
    """
    if probe_s <= 0:
        sr = sf.info(str(input_wav)).samplerate
        return [input_wav], 0, sr
    sig, sr = sf.read(str(input_wav))
    total = len(sig)
    win = min(total, int(probe_s * sr) // n_windows or total)
    if total <= win * n_windows:
        starts = [0]
        win = total
    else:
        span = total - win
        starts = [int((i + 0.5) * span / n_windows) for i in range(n_windows)]
    clips, lead_n = [], 0
    for wi, st in enumerate(starts):
        c = td / f"probe_w{wi}.wav"
        lead_n = write_probe_clip(sig[st:st + win], sr, c, lead_s=lead_s)
        clips.append(c)
    return clips, lead_n, sr


def probe_settings(knobs: list[str]) -> list[dict]:
    """Both ends of each knob, both corners, and the mid point."""
    picks: list[dict] = []
    for k in knobs:
        for v in (0.0, 1.0):
            p = {j: 0.5 for j in knobs}
            p[k] = v
            picks.append(p)
    picks.append({k: 0.0 for k in knobs})
    picks.append({k: 1.0 for k in knobs})
    picks.append({k: 0.5 for k in knobs})
    seen, uniq = set(), []
    for p in picks:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def load_device(config_path: Path) -> tuple[str, Path, list[str]]:
    """(name, schx, knobs) from a config.toml -- the same file run_pipeline.py trains from.

    knobs come from the config's own [knobs] table, not every real control on the device: a
    recipe that pins a control in [fixed] (e.g. Large Muffin's Volume) isn't swept in training,
    so it shouldn't be swept here either -- this measures the truncation error THIS RECIPE will
    actually see, not a hypothetical full-control sweep.
    """
    cfg = tomllib.loads(config_path.read_text())
    if "schx" not in cfg:
        raise ValueError(f"{config_path}: no [schx] -- not a SPICE-backed config "
                          "(e.g. a real-hardware-capture recipe); truncation doesn't apply to it")
    knobs = list((cfg.get("knobs") or {}).keys())
    if not knobs:
        raise ValueError(f"{config_path}: no [knobs] -- nothing to sweep")
    schx = Path(_os.path.expanduser(cfg["schx"]))
    return (config_path.parent.name, schx, knobs)


def _render_batch(schx: Path, clips: list[Path], iterations: int, speaker: str | None,
                   td: Path, workers: int, triples) -> dict[tuple, np.ndarray | None]:
    """Render every (setting, oversample, window) triple through the oracle's --jobs batch
    mode: parallel across workers, per-invocation fixed cost paid per worker, not per render.

    Extracted from measure() (2026-09-21) so the post-merge reference-convergence check in
    main()'s --merge path can render the same way without a second copy of this logic --
    exactly the drift shard.py's own docstring warns two copies of shard-selection would risk,
    applied here to render dispatch instead.
    """
    jobs, keymap = [], {}
    for p, os_, wi in triples:
        key = (tuple(sorted(p.items())), os_, wi)
        w = td / f"{abs(hash((str(schx), key)))}.wav"
        jobs.append({"input": str(clips[wi]), "output": str(w),
                     "params": ",".join(f"{k}={v}" for k, v in p.items()),
                     "oversample": os_, "iterations": iterations})
        keymap[str(w)] = key
    errs = _livespice_batch(str(schx), jobs, workers, speaker)
    out: dict[tuple, np.ndarray | None] = {}
    fails: dict[str, list[int]] = {}
    for wpath, key in keymap.items():
        if errs.get(wpath) is None and Path(wpath).exists():
            d, _ = sf.read(wpath)
            out[key] = np.asarray(d, dtype=np.float64)
        else:
            out[key] = None
            fails.setdefault(errs.get(wpath) or "no output", []).append(key[1])
    # GROUPED, not one line per render: a config-level failure (bad knob name, wrong
    # circuit) fails EVERY render with the identical message, and printing that message
    # 20+ times just buries the one thing worth reading. Group by message instead.
    for msg, oss in fails.items():
        oss_str = ",".join(str(o) for o in sorted(set(oss)))
        print(f"      {len(oss)} render(s) failed (os={oss_str}): {msg}", file=sys.stderr)
    return out


# A setting's reference render (den = sum(ref_signal**2)) below this fraction of the loudest
# reference measured for the same candidate is treated as NEAR-SILENT, not a real measurement
# -- see score_rows' own docstring. 1e-6 is -60dB relative to the loudest setting: comfortably
# below anything a genuinely quiet-but-audible setting would produce, but well above the
# floating-point/solver noise floor a truly muted knob (Volume=0, Master=0) collapses to.
NEAR_SILENT_DEN_RATIO = 1e-6


def score_rows(rows: list[dict], candidates: tuple[int, ...]) -> dict:
    """The worst-over-knob-settings reduction, shared by the live, --emit, and --merge paths.

    Operates on the row schema every path builds: {"index", "params", "esr": {os_str:
    {"num", "den"}}}. Sharing this one function is what makes a sharded run's table
    IDENTICAL to an unsharded one by construction rather than by argument -- the reduction
    runs the same code whether `rows` came from one process's own cache or from merging
    several workers' --emit files.

    NEAR-SILENT SETTINGS ARE EXCLUDED FROM THE WORST PICK, not just from divide-by-zero.
    probe_settings() always tests every knob at its own 0.0/1.0 extreme with the rest at 0.5 --
    for ANY device with a Volume/Master-shaped control mapped so that one extreme mutes the
    output (confirmed on this fleet: Master=0.0 on the Ceriatone Muchless Captain Reverb
    measures rms=0.000000), several probe settings render near-silent. `den` (the reference
    signal's own energy) is not exactly zero there -- floating-point/solver noise keeps it a
    tiny positive number -- so the old `den > 0` guard does not catch it, and `num/den` divides
    two noise-floor-sized quantities that do not scale down together, producing a confidently
    wrong "worst setting" that is actually a numerically unstable near-silence, not a real
    truncation problem. Excluded settings are still counted (see `res["excluded_near_silent"]`)
    so the caller can report them rather than have them just vanish from the table.
    """
    res: dict = {"n_settings": len(rows)}
    for os_ in candidates:
        c = str(os_)
        dens = [r["esr"][c]["den"] for r in rows if r["esr"][c]["den"] > 0]
        den_floor = max(dens) * NEAR_SILENT_DEN_RATIO if dens else 0.0
        worst, at = 0.0, None
        excluded = []
        for r in rows:
            num, den = r["esr"][c]["num"], r["esr"][c]["den"]
            if den <= den_floor:
                if den > 0:
                    excluded.append(r["params"])
                continue
            e = num / den
            if np.isfinite(e) and e >= worst:
                worst, at = e, r["params"]
        res[os_] = (worst, at)
        if excluded:
            res.setdefault("excluded_near_silent", {})[os_] = excluded
    return res


def merge_truncation_shards(paths: list[str]) -> tuple[list[dict], tuple[int, ...], int]:
    """Combine per-shard --emit files into one ordered row list, refusing anything incomplete.

    The same three guards as prepare_excitation.merge_onset_shards, applied to knob settings
    instead of corners -- each guards a way a distributed measurement goes wrong SILENTLY:
      * solver identity must agree across shards -- see solver_identity(). A truncation number
        measured by one livespice-cli build is not comparable to one measured by another (see
        that function's own docstring for the cross-architecture-binary-hash trap this avoids).
      * every setting index 0..N-1 must appear EXACTLY once. A missing index means a shard
        died and the worst-over-settings pick is computed with a hole; a duplicate means two
        shards overlapped and the run is not what it claims.
      * setting_total, candidates, and ref_os must all agree across shards, so shards from two
        different configs or --candidates/--ref-os invocations (a config edited mid-run, or a
        copy-pasted dispatch command with a typo) cannot be silently stitched together.

    Returns (rows, candidates, ref_os) -- the caller still needs candidates/ref_os to run the
    post-merge reference-convergence check and to sanity-check them against its own CLI args.
    """
    rows, seen, totals, solvers, cand_sets, ref_oss = {}, set(), set(), set(), set(), set()
    for p in paths:
        d = json.loads(Path(p).read_text())
        totals.add(d["setting_total"]); solvers.add(d["solver"])
        cand_sets.add(tuple(d["candidates"])); ref_oss.add(d["ref_os"])
        for r in d["rows"]:
            i = r["index"]
            if i in seen:
                raise SystemExit(f"setting index {i} appears in more than one shard -- shards "
                                 f"overlap; re-dispatch with disjoint --shard specs")
            seen.add(i); rows[i] = r
    if len(solvers) > 1:
        raise SystemExit(f"shards were measured by DIFFERENT renderer builds ({sorted(solvers)}) "
                         f"-- truncation numbers are not comparable. Rebuild every worker to the "
                         f"same revision and re-run.")
    if "livespice:UNKNOWN" in solvers:
        raise SystemExit("could not fingerprint the renderer binary on at least one worker -- "
                         "refusing to merge truncation numbers that cannot be proven comparable.")
    if len(totals) > 1:
        raise SystemExit(f"shards disagree on the setting count ({sorted(totals)}) -- they were "
                         f"measured against different knob grids; re-dispatch from one config.")
    if len(cand_sets) > 1 or len(ref_oss) > 1:
        raise SystemExit(f"shards disagree on candidates/ref_os ({sorted(cand_sets)} / "
                         f"{sorted(ref_oss)}) -- re-dispatch every shard with matching "
                         f"--candidates/--ref-os.")
    total = totals.pop()
    gaps = sorted(set(range(total)) - seen)
    if gaps:
        raise SystemExit(f"{len(gaps)} setting(s) missing from the merge (first: {gaps[:8]}) -- "
                         f"a shard did not finish. The worst-over-settings pick would be "
                         f"computed from an incomplete set; re-run the missing shard(s).")
    return [rows[i] for i in range(total)], cand_sets.pop(), ref_oss.pop()


def ref_error_check(schx: Path, at_worst: dict, ref_os: int, clips: list[Path], lead_n: int,
                    sr: int, iterations: int, speaker: str | None, td: Path,
                    workers: int) -> float:
    """ESR(reference, 2x reference) at the worst setting -- see measure()'s own docstring
    ("IS THE REFERENCE ITSELF CONVERGED?") for why this check exists. Extracted so main()'s
    --merge path can run it fresh (a merge has each shard's num/den terms, not the raw
    renders, so this always re-renders both sides rather than reusing a cache)."""
    k = tuple(sorted(at_worst.items()))
    got = _render_batch(schx, clips, iterations, speaker, td, workers,
                        [(at_worst, o, wi) for o in (ref_os, ref_os * 2) for wi in range(len(clips))])
    num = den = 0.0
    for wi in range(len(clips)):
        a, b = got.get((k, ref_os, wi)), got.get((k, ref_os * 2, wi))
        if a is None or b is None:
            continue
        n, d = esr_terms(a, b, lead_n, sr)
        num += n
        den += d
    return num / den if den > 0 else float("nan")


def measure(schx: Path, knobs: list[str], clips: list[Path], lead_n: int, sr: int,
            ref_os: int, candidates: tuple[int, ...], iterations: int,
            speaker: str | None, td: Path, workers: int,
            shard: str | None = None, emit: str | None = None) -> dict | None:
    """Worst-over-knob-settings truncation ESR at each candidate oversample.

    Renders run CONCURRENTLY. Serially this was leaving 13 of 14 cores idle while a single
    oversample-32 render -- 16x the work of the os=2 one it is the reference for -- ground through a
    60 s file, and a 7-device fleet would have taken hours. Threads are the right tool: every unit of
    work is a subprocess, so the GIL is irrelevant.

    `workers` is only local threads on one box. CROSS-MACHINE sharding is `shard`/`emit`
    instead (added 2026-09-21, same shape as prepare_excitation.py's --shard/--emit-onsets):
    pass `shard="LOW-HIGH/TOTAL"` to render only that slice of `probe_settings(knobs)` (striped
    by index modulo TOTAL, same shard.py contract every other shardable step uses), and `emit`
    to write that slice's (setting, candidate) ESR terms as JSON instead of scoring -- scoring
    needs EVERY setting (both the worst-over-settings table and the reference-convergence
    check), so a shard scoring from its own slice would be confidently wrong the same way a
    --shard'd prepare_excitation.py run does not size its own excitation. Merge with
    merge_truncation_shards() (main()'s --merge) once every shard has finished.

    `clips` are the probe windows from probe_clips() (possibly just the whole input); each
    (setting, oversample) is rendered per window and the ESR numerator/denominator POOLED
    across windows, so the reported number estimates the same whole-file quantity either way.
    """
    settings = probe_settings(knobs)
    setting_total = len(settings)
    shard_index = None
    if shard:
        from shard import select
        picked, *_ = select(list(enumerate(settings)), shard)
        shard_index = [i for i, _ in picked]
        settings = [s for _, s in picked]

    # Rendered and scored ONE SETTING AT A TIME, not one giant batch across every setting.
    # Two reasons: (1) it prints a per-item heartbeat ("N/M settings done") that
    # distribute_pull.py's Job.progress_re needs to tell a working shard from a stalled one --
    # without it `done` never advances and a slow-but-fine render gets killed as stuck, same
    # failure mode this module's own docstring on --workers used to just accept. (2) `workers`
    # still parallelizes WITHIN one setting's (candidate + ref_os) x window jobs -- plenty (at
    # least (len(candidates)+1) x len(clips)) to keep a many-core box busy -- so this costs
    # little concurrency for a real per-item progress signal.
    rows = []
    n_rendered = n_ok = 0
    for n, p in enumerate(settings):
        idx = shard_index[n] if shard_index is not None else n
        k = tuple(sorted(p.items()))
        got = _render_batch(schx, clips, iterations, speaker, td, workers,
                            [(p, os_, wi) for os_ in (*candidates, ref_os)
                             for wi in range(len(clips))])
        n_rendered += len(got)
        n_ok += sum(1 for v in got.values() if v is not None)
        esr_row = {}
        for os_ in candidates:
            num = den = 0.0
            for wi in range(len(clips)):
                a, b = got.get((k, os_, wi)), got.get((k, ref_os, wi))
                if a is None or b is None:
                    continue
                n_, d_ = esr_terms(a, b, lead_n, sr)
                num += n_
                den += d_
            esr_row[str(os_)] = {"num": num, "den": den}
        rows.append({"index": idx, "params": p, "esr": esr_row})
        print(f"  {n + 1}/{len(settings)} settings done", flush=True, file=sys.stderr)

    # EVERY render failed. Do NOT fall through to the table below: worst/at stay at their
    # initial (0.0, None), which prints as a confident-looking "0.00e+00 ... falls nanx" --
    # a precise-looking number produced by measuring NOTHING. That is exactly how this got
    # missed the first time: a wrong-case fixed-param name failed all 21 renders identically
    # and the tool still printed a verdict, just a garbled one, instead of stopping.
    if n_rendered and n_ok == 0:
        raise RuntimeError(
            f"{schx}: EVERY render failed ({n_rendered}/{n_rendered}). Nothing was measured, "
            "so there is no table to print. See the failure(s) above -- this usually means a "
            "config problem (wrong knob/fixed-param name, wrong --speaker, wrong backend), "
            "not a convergence issue; fix that and re-run.")

    if emit:
        from prepare_excitation import solver_identity
        ident = solver_identity("livespice")
        Path(emit).write_text(json.dumps({
            "setting_total": setting_total, "candidates": list(candidates), "ref_os": ref_os,
            "solver": ident, "shard": shard, "rows": rows,
        }, indent=2))
        print(f"wrote {len(rows)} setting row(s) to {emit} (solver {ident}) -- NOT scored; "
              f"merge with --merge to score once across every shard", file=sys.stderr)
        return None

    res = score_rows(rows, candidates)

    # IS THE REFERENCE ITSELF CONVERGED?
    #
    # Every number above is measured against os=ref_os, on the assumption that the reference is
    # close enough to the truth to stand in for it. That assumption is exactly the kind we have been
    # burned by -- a reference that has not converged is not a reference, the same way an oracle
    # built from the thing under test is not an oracle -- and it is cheap to check: render the worst
    # setting at 2*ref_os and see how far the reference still moves.
    #
    # If ESR(ref, 2*ref) is not far BELOW the smallest number in the table, the table is measuring
    # the reference's own error, the ratios go flat, and every entry is an UNDERSTATEMENT.
    _, at_worst = res[candidates[0]]
    if at_worst is not None:
        res["ref_error"] = ref_error_check(schx, at_worst, ref_os, clips, lead_n, sr,
                                           iterations, speaker, td, workers)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="the sweep to measure on. Truncation ESR is INPUT-DEPENDENT (a quiet tail "
                         "shrinks the denominator), so the table is only meaningful with the input "
                         "named. Use the sweep the datasets were actually rendered with.")
    ap.add_argument("--config", type=Path, required=True, help="the same TOML the pipeline uses")
    ap.add_argument("--ref-os", type=int, default=32, help="reference oversample (default 32)")
    ap.add_argument("--candidates", default="2,4,8",
                    help="oversamples to score against the reference (default 2,4,8)")
    ap.add_argument("--iterations", type=int, default=256,
                    help="Newton iterations; high, so convergence is not a confound (default 256)")
    ap.add_argument("--speaker", help="speaker to capture on multi-speaker circuits")
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 4) - 2),
                    help="concurrent renders (default: cores-2)")
    ap.add_argument("--probe-s", type=float, default=10.0,
                    help="total seconds of the input to actually render, as stratified windows "
                         "(default 10). Truncation is a property of the solver, not of sweep "
                         "length; windowed+pooled estimates the same whole-file quantity at a "
                         "fraction of the cost. Validated on Large Muffin vs the documented "
                         "whole-file table: ~1.3x HIGH at every rung (the safe direction -- "
                         "borderline circuits get more oversample, not less), same worst "
                         "setting, identical fall ratios. 0 = whole file (slow; the historical "
                         "behavior).")
    ap.add_argument("--n-windows", type=int, default=4,
                    help="number of stratified windows --probe-s is split into (default 4)")
    ap.add_argument("--lead-silence-s", type=float, default=None,
                    help="override write_probe_clip's 1.0s default lead-in for a circuit whose "
                         "own settling time is longer (e.g. a slow RC network) -- see that "
                         "function's docstring. Default: use its own 1.0s.")
    ap.add_argument("--shard", metavar="LOW-HIGH/TOTAL",
                    help="measure only the knob settings whose index modulo TOTAL falls in "
                         "[LOW, HIGH] -- shard.py's shared contract, same as "
                         "gen_dataset_from_schx.py/prepare_excitation.py. Requires --emit and "
                         "SKIPS scoring: both the worst-over-settings table and the reference-"
                         "convergence check need EVERY setting, so a shard scoring from its "
                         "own slice would be confidently wrong, the same way a --shard'd "
                         "prepare_excitation.py run does not size its own excitation. "
                         "Striping (modulo), not contiguous blocks, for the same reason as "
                         "every other shardable step here: settings are not equal cost.")
    ap.add_argument("--emit", metavar="PATH",
                    help="write this shard's measured (setting, candidate) ESR terms as JSON "
                         "instead of scoring. Carries the setting total, candidates/ref_os, "
                         "each row's GLOBAL setting index, and a fingerprint of the renderer "
                         "binary so --merge can refuse mismatched work.")
    ap.add_argument("--merge", nargs="+", metavar="PATH",
                    help="combine --emit files from every shard, verify completeness and "
                         "solver agreement, then score ONCE from the full set -- including a "
                         "fresh reference-convergence check at the worst setting the merge "
                         "finds (a merge has each shard's num/den terms, not the raw renders, "
                         "so that one check always re-renders). Refuses on a missing or "
                         "duplicated setting index, a solver-build mismatch, or a "
                         "candidates/ref_os disagreement -- each of which would otherwise "
                         "yield a silently wrong table.")
    args = ap.parse_args()

    cands = tuple(int(c) for c in args.candidates.split(","))
    if max(cands) >= args.ref_os:
        ap.error(f"--candidates must all be below --ref-os ({args.ref_os}): you cannot measure the "
                 f"reference against itself")
    if bool(args.shard) != bool(args.emit):
        ap.error("--shard and --emit must be used together")
    if args.merge and (args.shard or args.emit):
        ap.error("--merge is mutually exclusive with --shard/--emit")

    try:
        name, schx, knobs = load_device(args.config)
    except ValueError as e:
        ap.error(str(e))
    if not schx.exists():
        ap.error(f"{name}: MISSING {schx}")
    check_oracle("livespice")

    print(f"input:      {args.input.name}")
    print(f"reference:  oversample={args.ref_os}, {args.iterations} Newton iterations")
    print(f"device:     {name} ({len(knobs)} knobs)\n")

    if args.merge:
        # MERGE MODE: every setting was already measured elsewhere. Skip rendering the sweep
        # entirely and score from the verified union -- merge_truncation_shards() refuses
        # anything incomplete or measured by a divergent solver build, so reaching here means
        # the set is trustworthy. Only the reference-convergence check still renders (fresh --
        # see ref_error_check's own docstring for why a merge can't just reuse a shard's cache).
        merged_rows, m_cands, m_ref_os = merge_truncation_shards(args.merge)
        if m_cands != cands or m_ref_os != args.ref_os:
            ap.error(f"--candidates/--ref-os ({cands}/{args.ref_os}) don't match what the "
                     f"merged shards were dispatched with ({m_cands}/{m_ref_os}) -- pass the "
                     f"same values used to dispatch them.")
        print(f"merged {len(merged_rows)} setting(s) from {len(args.merge)} shard file(s)")
        r = score_rows(merged_rows, cands)
        _, at_worst = r[cands[0]]
        if at_worst is not None:
            with tempfile.TemporaryDirectory() as tds:
                td = Path(tds)
                clips, lead_n, sr = probe_clips(args.input, args.probe_s, args.n_windows, td,
                                                lead_s=args.lead_silence_s)
                r["ref_error"] = ref_error_check(schx, at_worst, args.ref_os, clips, lead_n, sr,
                                                 args.iterations, args.speaker, td, args.workers)
        rows = [(name, r)]
    elif args.shard:
        # SHARDED MODE: render only this slice and write it out; do NOT score. See measure()'s
        # own docstring for why scoring needs every setting.
        with tempfile.TemporaryDirectory() as tds:
            td = Path(tds)
            clips, lead_n, sr = probe_clips(args.input, args.probe_s, args.n_windows, td,
                                            lead_s=args.lead_silence_s)
            print(f"  measuring shard {args.shard} of {name} ({len(knobs)} knobs) ...",
                  flush=True, file=sys.stderr)
            try:
                measure(schx, knobs, clips, lead_n, sr, args.ref_os, cands, args.iterations,
                       args.speaker, td, args.workers, shard=args.shard, emit=args.emit)
            except RuntimeError as e:
                sys.exit(f"\nERROR: {e}")
        return
    else:
        rows = []
        with tempfile.TemporaryDirectory() as tds:
            td = Path(tds)
            clips, lead_n, sr = probe_clips(args.input, args.probe_s, args.n_windows, td,
                                            lead_s=args.lead_silence_s)
            if len(clips) > 1 or clips[0] != args.input:
                print(f"probing:    {len(clips)} x {sf.info(str(clips[0])).frames / sr:.1f}s windows "
                      f"(--probe-s {args.probe_s:g}; 0 = whole file)\n")
            print(f"  measuring {name} ({len(knobs)} knobs) ...", flush=True, file=sys.stderr)
            try:
                r = measure(schx, knobs, clips, lead_n, sr, args.ref_os, cands,
                            args.iterations, args.speaker, td, args.workers)
            except RuntimeError as e:
                sys.exit(f"\nERROR: {e}")
            rows.append((name, r))

    hdr = " | ".join(f"@ os={c}" for c in cands)
    print(f"\n| device | {hdr} | worst setting @ os={cands[0]} | ref err | verdict |")
    print("|---|" + "---:|" * len(cands) + "---|---:|---|")
    for name, r in rows:
        cells = []
        for c in cands:
            v, _ = r.get(c, (float('nan'), None))
            cells.append(f"{v:.2e}")
        v0, at0 = r.get(cands[0], (float('nan'), None))
        vlast, _ = r.get(cands[-1], (float('nan'), None))
        atstr = ", ".join(f"{k}={v:g}" for k, v in sorted(at0.items())) if at0 else "-"

        # EVERY consecutive ratio, not just the first. Checking only cands[0]/cands[1] would have
        # passed the British-stack amp's hot-rod variant as healthy -- it falls a respectable 4.4x from os=2 to os=4 and
        # then STALLS at 1.6x from 4 to 8. The stall is the finding; a verdict that reads only the
        # first rung reports the circuit as smooth and hides it.
        ratios = []
        for a_, b_ in zip(cands, cands[1:]):
            va, _ = r.get(a_, (float('nan'), None))
            vb, _ = r.get(b_, (float('nan'), None))
            ratios.append(va / vb if (np.isfinite(va) and np.isfinite(vb) and vb > 0) else float('nan'))
        fall = " → ".join(f"{x:.1f}x" for x in ratios) if ratios else "-"

        # The reference must be much cleaner than the finest number it is used to score. If it is
        # not, the table bottoms out on the REFERENCE's error, not the circuit's.
        ref_err = r.get("ref_error", float("nan"))
        refstr = f"{ref_err:.1e}" if np.isfinite(ref_err) else "?"

        stalled = [(a_, b_, x) for (a_, b_), x in zip(zip(cands, cands[1:]), ratios)
                   if np.isfinite(x) and x < 2.0]
        if np.isfinite(ref_err) and np.isfinite(vlast) and ref_err > 0.25 * vlast:
            verdict = f"**REFERENCE NOT CONVERGED** — os={args.ref_os} still moves by {ref_err:.1e}, " \
                      f"not far below the @os={cands[-1]} figure; that column is a floor, not a measurement"
        elif stalled:
            a_, b_, x = stalled[0]
            verdict = f"**STALLS** {a_}→{b_}: falls only {x:.1f}x, not ~4x ({fall})"
        else:
            verdict = f"falls {fall}"
        print(f"| {name} | " + " | ".join(cells) + f" | {atstr} | {refstr} | {verdict} |")

        excluded = r.get("excluded_near_silent")
        if excluded:
            for c, params_list in excluded.items():
                locs = "; ".join(", ".join(f"{k}={v:g}" for k, v in sorted(p.items()))
                                 for p in params_list)
                print(f"  NOTE: {name} @ os={c}: {len(params_list)} setting(s) excluded from "
                      f"the worst-setting pick as near-silent (reference energy < "
                      f"{NEAR_SILENT_DEN_RATIO:.0e} of the loudest setting measured) -- ESR is "
                      f"not a meaningful number there, not evidence of a truncation problem: "
                      f"{locs}")

    print("\n`ref err` = ESR(reference, 2x reference) at the worst setting: how far the reference "
          f"itself\nstill moves. It must sit well below the @os={cands[-1]} column, or that column is "
          "measuring the\nreference's own error rather than the circuit's.\n")
    print("O(h^2) demands the error fall ~4x per doubling. A device that falls <2x does not "
          "converge in the timestep:\nsomething in it is not smooth, and oversampling will not fix "
          "its dataset.")


if __name__ == "__main__":
    main()
