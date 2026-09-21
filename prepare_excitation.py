#!/usr/bin/env python3
"""Wire measured saturation onset directly to excitation building -- backend-agnostic
(--backend {livespice,ngspice,ngspice-deck,ltspice-deck}, see render_backends.py). Closes the
manual human-in-the-loop
gap that's existed between find_saturation_point.py and build_excitation.py: until now,
someone had to read an onset number by hand and pick --sweep-peak/--chirp-levels themselves
(this is literally how every existing config's excitation was sized, e.g. the non-midpoint-default pedal's "peak sized
from a direct Gain=0.5-vs-1.0 output-V-vs-input-V sweep" config comment).

NAMING (2026-09-10): --sweep-file (was --real-clip) follows TONE3000's own term for this style
of file (tone3000.com/create/capture calls it a "sweep signal") -- it is often itself a
synthesized capture sweep (e.g. T3K-sweep-v3.wav), not a real-playing recording. build_
excitation.py's OWN internally-generated tones are called "chirps" instead, specifically so
the two concepts (an externally-supplied file vs. an internally-synthesized tone) don't share
a name -- see that module's docstring.

Runs find_saturation_point() at EVERY corner of the knob grid (reusing check_transient_
coverage.py's own _corners() -- the same all-min/all-max/center/solo-extreme/full-hypercube
set that tool checks against, not just one hand-picked knob setting), takes the WORST-CASE
(highest) onset across them, derives --chirp-levels (a staged ramp up to `--margin` x
worst-case onset) and --sweep-peak (a fraction of worst-case onset), and invokes
build_excitation.py with them. Refuses to build (raises) if any corner's onset can't be
determined, rather than silently building against a partial result -- same "refuse to guess"
convention as preflight.py/check_transient_coverage.py.

After building, running check_transient_coverage.py (livespice or --backend ngspice-deck) is
still worth doing as an independent gate before training -- and it is a REAL gate, not a
formality. This tool derives its levels from onset numbers measured at a SET OF PROBED POINTS,
and onset is not monotonic in the knobs, so the grid's true worst corner need not be one of
them: Mesa Dual Rectifier ORANGE's worst (23.177 V, Bass=min with the others centred) is 1.27x
the highest of all 32 hypercube vertices. --sweep-peak-frac (default 1.3) buys headroom
against that; --sample-grid N buys coverage of it. A clean check is expected, not guaranteed --
and both Mesa channels FAILED one on 2026-09-04 against an excitation whose peak was
hand-picked rather than measured at all.

Usage:
  livespice: python prepare_excitation.py --backend livespice \\
      --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml \\
      --sweep-file examples/T3K-sweep-v3.wav --output ~/work/tmp/DEVICE_excitation.wav

  ngspice:   python prepare_excitation.py --backend ngspice-deck \\
      --pedal-dir ~/work/parametric-devices/pedals --module gen_ocd_ngspice \\
      --range "Gain=0.1,0.5,0.9" --range "Tone=0.2,0.5,0.8" --fixed-params "Volume=1.0" \\
      --sweep-file ~/Downloads/T3K-sweep-v3.wav \\
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
from check_transient_coverage import resolve_sample_grid, _corners, _sample_interior  # noqa: E402
from capture_chain import (add_cli_args as _cc_add_cli_args, resolve as _cc_resolve,  # noqa: E402
                           cache_tag)
from find_saturation_point import (find_saturation_point, findpeak_cache_key,  # noqa: E402
                                    cache_findpeak, scratch_dir)
from render_backends import (LiveSpiceBackend, NgspiceBackend, LtspiceBackend,  # noqa: E402
                             NgspiceSchxBackend, parse_conv, conv_cache_tag)


PRE_FIX_METHOD = "pre-2026-09-12/99pct-of-max"


def method_summary(rows) -> str:
    """How this run's onsets were derived, across every probed corner.

    A sizing pass can mix freshly measured corners with cache hits written by older code, so
    this reports a MIXTURE rather than collapsing to the first method found -- claiming an
    internal consistency the run does not have is worse than no field at all. An absent method
    is not "unknown": it identifies the pre-2026-09-12 99%-of-max rule, which is the thing a
    reader most needs to spot.
    """
    seen = sorted({(r.get("method") or PRE_FIX_METHOD) for r in rows}) or [PRE_FIX_METHOD]
    return seen[0] if len(seen) == 1 else "MIXED: " + ", ".join(seen)



def solver_identity(backend_name: str) -> str:
    """A fingerprint of the RENDERER SOURCE REVISION, so onsets measured on different machines
    can be proven comparable before they are merged.

    WHY SOURCE REVISION AND NOT A BINARY HASH. The first version of this hashed the
    livespice_cli executable, which is wrong on a heterogeneous fleet and wrong in a way that
    makes distributed sizing unusable: measured 2026-09-17 across four workers all built from
    the SAME source (livespice-cli 7234f0c, submodule 134d5c07), the binaries hashed to four
    different values -- b39e8050, b6fb0e60, 41bd0bb8, 0bda53ab -- because arm64 Macs and x86
    Linux boxes do not produce identical executables from identical code. A merge gated on
    binary equality would have refused every cross-architecture run, i.e. exactly the runs
    sharding exists for.

    What actually needs to agree is the SOLVER, and that is the git revision of livespice-cli
    plus its LiveSPICE submodule. The submodule is the load-bearing half: enabling
    SimulateCapacitances was unsolvable ("Failed to eliminate differentials from system of
    equations") until it moved to 134d5c0, which adds capacitor currents as system variables.
    Two of five machines in this fleet were on an older build as recently as this month, and a
    single corner measured by such a build silently mis-sizes the whole device -- the merge
    picks ONE number, the worst-case onset, out of every corner measured anywhere.

    Returns a marker rather than raising when the revision cannot be read, so the merge
    REFUSES rather than crashing mid-render after hours of work.
    """
    if backend_name != "livespice":
        return f"{backend_name}:unidentified"
    import subprocess
    for repo in (Path.home() / "work/livespice-cli", Path("/opt/livespice-cli")):
        if not (repo / ".git").exists():
            continue
        try:
            head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
            sub = subprocess.run(["git", "-C", str(repo), "submodule", "status"],
                                 capture_output=True, text=True, timeout=10)
            if head.returncode == 0:
                h = head.stdout.strip()[:12]
                subrev = ""
                for line in sub.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2 and "LiveSPICE" in parts[1]:
                        subrev = parts[0].lstrip("+-U")[:12]
                        break
                return f"livespice:{h}+{subrev or 'nosub'}"
        except (OSError, subprocess.SubprocessError):
            continue
    return "livespice:UNKNOWN"


def shard_corners(corners, spec):
    """The [LOW, HIGH] slice of `corners` by index modulo TOTAL -- shard.py's shared contract.

    Deliberately the same modulo striping gen_dataset_from_schx.py uses rather than
    contiguous blocks: corner cost here is wildly uneven (a corner whose sweep starts above
    the saturation onset re-sweeps over a 100x extended range), so contiguous blocks would
    hand one machine a run of slow corners. Striping mixes fast and slow across shards.
    """
    from shard import parse_shard
    low, high, total = parse_shard(spec)
    return [(i, c) for i, c in enumerate(corners) if low <= (i % total) <= high]


def merge_onset_shards(paths):
    """Combine per-shard onset files into one ordered row list, refusing anything incomplete.

    Three assertions, each guarding a way a distributed sizing run goes wrong SILENTLY:
      * solver identity must agree across shards -- see solver_identity().
      * every corner index 0..N-1 must appear EXACTLY once. A missing index means a shard
        died and the worst-case onset is computed from a hole; a duplicate means two shards
        overlapped and the run is not what it claims.
      * the corner TOTAL must agree across shards, so shards from two different grids (a
        config edited mid-run) cannot be stitched together.
    """
    rows, seen, totals, solvers = {}, set(), set(), set()
    for p in paths:
        d = json.loads(Path(p).read_text())
        totals.add(d["corner_total"]); solvers.add(d["solver"])
        for r in d["rows"]:
            i = r["index"]
            if i in seen:
                raise SystemExit(f"corner index {i} appears in more than one shard -- shards "
                                 f"overlap; re-dispatch with disjoint --shard specs")
            seen.add(i); rows[i] = r
    if len(solvers) > 1:
        raise SystemExit(f"shards were measured by DIFFERENT renderer builds ({sorted(solvers)}) "
                         f"-- onsets are not comparable and the worst-case pick would be "
                         f"meaningless. Rebuild every worker to the same revision and re-run.")
    if "livespice:UNKNOWN" in solvers:
        raise SystemExit("could not fingerprint the renderer binary on at least one worker -- "
                         "refusing to merge onsets that cannot be proven comparable.")
    if len(totals) > 1:
        raise SystemExit(f"shards disagree on the corner count ({sorted(totals)}) -- they were "
                         f"measured against different grids; re-dispatch from one config.")
    total = totals.pop()
    gaps = sorted(set(range(total)) - seen)
    if gaps:
        raise SystemExit(f"{len(gaps)} corner(s) missing from the merge (first: {gaps[:8]}) -- "
                         f"a shard did not finish. The worst-case onset would be computed from "
                         f"an incomplete set; re-run the missing shard(s).")
    return [rows[i] for i in range(total)]


def worst_case_onset(backend, identity, cache_extra, knob_ranges, fixed, tmp,
                      peak_max_v=40.0, no_cache=False, full_hypercube=None, quiet=False,
                      lead_silence_s=0.0, max_corners=None, sample_grid=0, capture=None,
                      corner_workers=1, shard=None, emit_onsets=None, backend_name="livespice",
                      min_start_v=1e-9, start_v=0.005):
    """Find the worst-case (highest) saturation onset across every corner of knob_ranges.
    Reuses find_saturation_point.py directly (not check_transient_coverage.check_coverage --
    that function's pass/fail comparison against a transient_peak doesn't apply to this
    direction, only its per-corner onset detection would, so this calls the shared onset-
    finder and corner-generator directly instead of routing through a check meant for a
    different question). Raises if any corner's onset couldn't be determined -- see module
    docstring's "refuse to guess"."""
    corners = _corners(knob_ranges, full_hypercube=full_hypercube, max_corners=max_corners)
    corners = _sample_interior(knob_ranges, corners, sample_grid)
    corner_total = len(corners)
    # SHARDED MODE: measure only this slice and write it out; do NOT size. The worst-case
    # onset is a max over EVERY corner, so a shard sizing from its own slice would produce a
    # confidently wrong peak. Sizing happens once, after --merge-onsets proves the set complete.
    shard_index = None
    if shard:
        picked = shard_corners(corners, shard)
        shard_index = [i for i, _ in picked]
        corners = [c for _, c in picked]
        if not quiet:
            print(f"  shard {shard}: {len(corners)} of {corner_total} corners")
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
    # CORNER-LEVEL PARALLELISM (2026-09-17). Corners are independent -- the adaptive
    # range-extension inside find_saturation_point() is per-corner and never reads another
    # corner's result -- so they parallelise cleanly. Previously this loop was strictly
    # serial: scaffold_config.py passes --workers to its truncation phase (12 renders at
    # once) and then nothing here, so onset measurement ran ~2 renders deep on a machine
    # with 12 cores. On a 141-corner device (6 knobs: 77 corner + 64 interior) whose every
    # corner needed the 100x range extension, that was ~2 hours.
    #
    # PER-CORNER TMP IS MANDATORY, not tidiness. find_saturation_point()'s sweep names its
    # render files `fp_{i}` from a counter created fresh per call, so two concurrent corners
    # both write fp_0, fp_1, ... If they shared `tmp` they would silently overwrite each
    # other's renders and return onsets measured from the wrong audio -- a wrong EXCITATION
    # PEAK for the device, with nothing failing. Each corner therefore gets its own subdir.
    #
    # Concurrency is a PRODUCT, not a sum: find_saturation_point already fans out its
    # amplitude points (workers=8 by default), so corner_workers x that must stay near the
    # core count or the machine thrashes -- the same workers x batch-size pathology
    # scan_film_runaway.py's --batch-size help documents. Hence sweep workers are divided
    # down as corner_workers rises, keeping the product roughly constant.
    sweep_workers = max(1, 8 // max(1, corner_workers))

    def _measure(idx_label_vals, amp_executor=None):
        idx, (label, vals) = idx_label_vals
        params = dict(vals); params.update(fixed)
        cpath = findpeak_cache_key(identity, params, cache_extra)
        if cpath.exists() and not no_cache:
            sat = json.loads(cpath.read_text())
        else:
            ctmp = Path(tmp) / f"corner_{idx:04d}" if corner_workers > 1 else tmp
            if corner_workers > 1:
                Path(ctmp).mkdir(parents=True, exist_ok=True)
            sat = find_saturation_point(backend, params, str(ctmp), max_v=peak_max_v,
                                         lead_silence_s=lead_silence_s, capture=capture,
                                         workers=sweep_workers, executor=amp_executor,
                                         min_start_v=min_start_v, start_v=start_v)
            cache_findpeak(cpath, sat)
        return label, params, sat

    def _report(label, sat, done, total):
        onset = sat.get("onset_99pct_input_v") if sat else None
        if not quiet:
            onset_str = "NONE (not reached)" if onset is None else f"{onset:.3f} V"
            prefix = f"  [{done}/{total}]" if corner_workers > 1 else "  "
            print(f"{prefix} {label:16} onset={onset_str:>10}", flush=True)
        return onset

    # Results are REPORTED as they land but STORED by index. Printing on completion keeps a
    # long run legible -- an earlier version collected everything before printing a line,
    # which on a 141-corner device meant no output at all for over an hour. Storing by index
    # keeps `rows` in corner order regardless of completion order, so the artifact and the
    # worst-case pick do not depend on scheduling.
    # FIXED (2026-09-19), was KNOWN OPEN BUG (2026-09-18): this nested-executor path (an outer
    # corner-level ThreadPoolExecutor whose workers each called find_saturation_point(), which
    # spun up its OWN inner ThreadPoolExecutor for the amplitude sweep) was observed to
    # DEADLOCK outright -- not slow, not thrashing, genuinely hung: `sample`'d a stuck process
    # and found the main thread and every worker thread parked in
    # `_PySemaphore_Wait`/`_pthread_cond_wait`, waiting on a semaphore nothing was going to
    # signal. Root-caused 2026-09-19: total subprocess concurrency is IDENTICAL between
    # --corner-workers 1 (safe) and --corner-workers >1 (buggy) -- both peak at 8 renders at
    # once, since sweep_workers = 8 // corner_workers. The one variable that actually differs
    # is whether more than one ThreadPoolExecutor is ever ALIVE, or being CONSTRUCTED, from a
    # non-main worker thread at the same time -- with corner_workers>1, every corner-worker
    # thread used to build its own separate inner pool concurrently, hitting
    # concurrent.futures.thread's process-global bookkeeping (the shared atexit/
    # _threads_queues registration every Executor.__init__/shutdown touches) from multiple
    # threads at once. Fix: build ONE shared amplitude-level pool here, in the main thread,
    # BEFORE any corner-worker thread starts, sized to the same total peak concurrency
    # (corner_workers x sweep_workers) as before, and pass it into every find_saturation_point()
    # call via `executor=` -- see that function's own `executor` docstring. No ThreadPoolExecutor
    # is now ever constructed from a worker thread. --corner-workers 1 is untouched (it never
    # had more than one pool alive to begin with) and remains a safe fallback if this
    # resurfaces in some other shape.
    results = [None] * len(corners)
    if corner_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        amp_pool_size = max(1, corner_workers * sweep_workers)
        with ThreadPoolExecutor(max_workers=amp_pool_size) as amp_ex, \
             ThreadPoolExecutor(max_workers=corner_workers) as ex:
            futs = {ex.submit(_measure, (i, c), amp_ex): i for i, c in enumerate(corners)}
            for n, fut in enumerate(as_completed(futs), 1):
                label, params, sat = fut.result()
                results[futs[fut]] = (label, params, sat, _report(label, sat, n, len(corners)))
    else:
        for i, c in enumerate(corners):
            label, params, sat = _measure((i, c))
            results[i] = (label, params, sat, _report(label, sat, i + 1, len(corners)))

    rows = []
    for label, params, sat, onset in results:
        rows.append({"corner": label, "params": params, "onset_v": onset,
                     # Recorded per corner, not once per run: a sizing pass can mix freshly
                     # measured corners with cache hits, and if those were written by different
                     # code the run is not internally consistent. Better to see the mixture in
                     # the artifact than to infer it later from build dates and plausibility.
                     "knee_v": (sat or {}).get("knee_v"),
                     "method": (sat or {}).get("onset_method")})
    if emit_onsets:
        for n, r in enumerate(rows):
            r["index"] = shard_index[n] if shard_index is not None else n
        Path(emit_onsets).write_text(json.dumps(
            {"corner_total": corner_total, "solver": solver_identity(backend_name),
             "shard": shard, "rows": rows}, indent=2))
        print(f"wrote {len(rows)} onset row(s) to {emit_onsets} "
              f"(solver {solver_identity(backend_name)}) -- NOT sized; merge with "
              f"--merge-onsets to size once across every shard")
        return None, rows
    missing = [r for r in rows if r["onset_v"] is None]
    if missing:
        raise RuntimeError(
            f"no saturation onset found at corner(s) {[r['corner'] for r in missing]} -- "
            f"refusing to build an excitation against a partial/failed result. "
            f"find_saturation_point already extends the sweep DOWNWARD when it starts above "
            f"the onset, so reaching here means either every render at that corner failed, or "
            f"the onset is below the extension floor. Run the corner directly and look at the "
            f"curve before assuming --peak-max-v (the sweep CEILING) is what needs raising -- "
            f"on a high-gain circuit it is the floor that matters.")
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
    """Returns (backend, identity, cache_extra, knob_ranges, fixed, lead_silence_s, label,
    capture) -- `capture` last, resolved from CLI+config (see capture_chain.resolve)."""
    # Resolve from CLI+defaults FIRST so every backend branch has a chain; the livespice
    # branch re-resolves once its config is loaded. Initialising to None instead would
    # silently DISABLE the chain on the deck backends, which never load a config.
    _capture = _cc_resolve(args)
    if args.backend == "livespice":
        if args.config:
            cfg = load_config(Path(args.config))
            _capture = _cc_resolve(args, cfg)
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
        cache_extra = f"os={oversample}|it={args.iterations}|maxv={args.peak_max_v}|minv={args.min_start_v}|startv={args.sweep_start_v}" + cache_tag(_capture)
        return backend, identity, cache_extra, knob_ranges, fixed, 0.0, Path(schx).name, _capture
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
        # The BACKEND NAME and maxstep must be in this key. Without the name, ngspice-deck and
        # ltspice-deck produced an IDENTICAL one -- same identity (the generator module's bytes)
        # and the same "maxv=..." extra -- so one simulator's onset was served to the other. Not a
        # corner case: docs/backends.md says ltspice-deck exists for a device whose ngspice deck
        # cannot converge, i.e. THE SAME MODULE through both. Without maxstep, the sweep from 3e-6
        # down to 3e-8 that the same doc describes gets the first value's answers back every time.
        #
        # The livespice extra is deliberately NOT changed: it carries "os=..|it=.." which no deck
        # backend emits, so it cannot collide with either, and touching it would invalidate every
        # cached entry in the fleet to fix a bug it does not have.
        cache_extra = f"backend=ngspice-deck|maxstep={args.maxstep}|maxv={args.peak_max_v}|minv={args.min_start_v}|startv={args.sweep_start_v}" + cache_tag(_capture)
        return backend, identity, cache_extra, knob_ranges, fixed, args.lead_silence_s, args.module, _capture
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
        cache_extra = f"backend=ltspice-deck|maxstep={args.maxstep}|maxv={args.peak_max_v}|minv={args.min_start_v}|startv={args.sweep_start_v}" + cache_tag(_capture)
        # No lead_silence_s: LTspice's .ic/uic hints replace the need for a cold-start
        # settling lead-in -- see ltspice_spicelib.py's docstring.
        return backend, identity, cache_extra, knob_ranges, fixed, 0.0, args.module, _capture
    if args.backend == "ngspice":
        # Same schx/--config convention as livespice -- this is the GENERIC schx-translated
        # path (ngspice/schx_to_ngspice.py via NgspiceSchxBackend), for a circuit whose .schx
        # exists but whose LiveSPICE render diverges under real signal (e.g. Arbiter Fuzz
        # Face's tight DC-coupled feedback loop -- see its own .md/.backends.toml). Not to be
        # confused with ngspice-deck, which is for a device with NO .schx counterpart at all.
        if args.config:
            cfg = load_config(Path(args.config))
            _capture = _cc_resolve(args, cfg)
            schx = str(cfg["schx"])
            oversample = args.oversample or cfg.get("oversample", 2)
            knob_ranges = _parse_ranges(cfg.get("ranges", []))
            fixed = _parse_fixed(cfg.get("fixed_params"))
            conv = parse_conv(args.conv if args.conv is not None else cfg.get("conv"))
        else:
            if not args.schx or not args.range:
                sys.exit("--backend ngspice needs --config, or --schx + --range")
            schx = args.schx
            oversample = args.oversample or 2
            knob_ranges = _parse_ranges(args.range)
            fixed = _parse_fixed(args.fixed_params)
            conv = parse_conv(args.conv)
        if not knob_ranges:
            sys.exit("no [knobs]/--range entries -- nothing to check corners over")
        backend = NgspiceSchxBackend(schx, oversample=oversample, conv=conv)
        identity = Path(schx).read_bytes()
        # backend=ngspice in the key: without it this would share livespice's "os=..|it=.."
        # extra on the SAME schx identity, serving a raw-node livespice onset to an ngspice
        # caller (or vice versa) -- the exact hazard --backend ngspice-deck's own comment
        # above documents for ngspice-deck vs ltspice-deck. conv_cache_tag guards the same
        # hazard for a device-model override (e.g. a corrected transistor fit): an onset
        # measured under one --conv must not be served to a caller expecting a different one.
        cache_extra = (f"backend=ngspice|os={oversample}|maxv={args.peak_max_v}|minv={args.min_start_v}|startv={args.sweep_start_v}"
                      + cache_tag(_capture) + conv_cache_tag(conv))
        # lead_silence_s IS needed here, same as ngspice-deck: this is ngspice under the hood
        # (schx_to_ngspice.py's generated .cir, not a hand-written deck, but the same solver),
        # so it has the same cold-start settling behaviour grid_adequacy.py's own --backend
        # ngspice already applies uniformly regardless of which ngspice path it is.
        return (backend, identity, cache_extra, knob_ranges, fixed, args.lead_silence_s,
               Path(schx).name, _capture)
    sys.exit(f"unknown --backend {args.backend!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True,
                    choices=["livespice", "ngspice", "ngspice-deck", "ltspice-deck"])

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
    ap.add_argument("--sample-grid", type=int, default=None, metavar="N",
                     help="AUTO by default (interior_sample_budget: ~1.5x the corner "
                          "count, capped 64) -- 0 disables. Corners are a HEURISTIC and have "
                          "been shown insufficient twice: Mesa Orange's 5-knob grid had its "
                          "true worst onset 1.27x above every one of 32 vertices, and its "
                          "2-knob Gain x Master grid had an interior cell at 11.18 V against "
                          "a worst CORNER of 1.76 V -- 6.3x. Sizing from corners alone there "
                          "produced an excitation that could not drive 2 of 25 probed cells "
                          "into saturation at all, and the transient gate correctly refused "
                          "the render (2026-09-12). Onset is not monotonic in the knobs, so "
                          "its maximum over the grid need not sit at a vertex; only probing "
                          "the interior MEASURES it. Deterministic, so a sizing run and a "
                          "later coverage check agree.")
    ap.add_argument("--max-corners", type=int, default=None,
                     help="cap the TOTAL corner count, sampling the hypercube deterministically "
                          "when it does not all fit, instead of abandoning it. Use this rather "
                          "than --no-full-hypercube for a many-knob device: same budget, but it "
                          "still reaches mixed corners (a 16-knob config at --max-corners 48 gets "
                          "13 of them; --no-full-hypercube gets 0).")
    ap.add_argument("--peak-max-v", type=float, default=40.0,
                     help="find_saturation_point sweep ceiling -- the 40V default suits an "
                          "amp; lower it (e.g. 3-5) for a small pedal circuit")
    ap.add_argument("--sweep-start-v", type=float, default=0.005,
                     help="find_saturation_point's initial sweep floor (default: %(default)s, "
                          "matching that function's own start_v default). Raise this for a "
                          "circuit with an ACTIVE internal supply whose own ripple floor sits "
                          "ABOVE the default -- --min-start-v alone cannot fix that case, since "
                          "it only bounds downward EXTENSION below this starting point, and "
                          "extension is never reached if the very first probe point is already "
                          "ripple-contaminated. Set above your circuit's own measured near-"
                          "silent output floor (with margin) -- see --min-start-v's help for "
                          "how to measure that floor directly and the Vox AC30 (sag ac) case "
                          "that motivated both flags.")
    ap.add_argument("--min-start-v", type=float, default=1e-9,
                     help="find_saturation_point's downward-extension floor (default: "
                          "%(default)s, matching that function's own default). Raise this for "
                          "a circuit with an ACTIVE internal supply (an AC-driven sag/rectifier "
                          "network, --backend livespice's FRONTEND=ac style) rather than ideal "
                          "DC rails -- such a circuit generates a small amount of its OWN "
                          "output (ripple) completely independent of the input signal, so as "
                          "the sweep extends toward ever-smaller inputs, measured gain "
                          "(output/input) diverges without bound and NEVER finds a genuine "
                          "linear region -- the sweep exhausts every extension decade and "
                          "reports the floor itself as 'the onset', sizing an excitation at "
                          "~0V (silent). Found 2026-09-20 on the Vox AC30 Top Boost (sag ac): "
                          "even after cutting the supply's own ripple ~17x (reservoir caps "
                          "4.7/22uF -> 47/470uF), residual ripple (~0.04V RMS) still triggered "
                          "this at the default 1e-9V floor -- no amount of realistic supply "
                          "filtering makes a truly-AC-coupled circuit's ripple exactly zero, so "
                          "the floor itself needs raising above it, not chased downward "
                          "forever. Set this above your circuit's own measured near-silent "
                          "output floor (with margin) -- a probe at a genuinely tiny input "
                          "(e.g. 1e-6 V) on the ACTUAL circuit tells you that floor directly.")
    _cc_add_cli_args(ap)
    ap.add_argument("--conv", default=None,
                    help="[ngspice] device-model convergence/fidelity overrides key=val,... "
                         "(same format gen_dataset_from_schx.py --conv uses; e.g. "
                         "bjt_vaf=102.207,bjt_rb=173.312 for a real datasheet-fitted "
                         "transistor). Default: --config's own `conv` field.")
    ap.add_argument("--shard", metavar="LOW-HIGH/TOTAL",
                    help="measure only the corners whose index modulo TOTAL falls in "
                         "[LOW, HIGH] -- shard.py's shared contract, same as "
                         "gen_dataset_from_schx.py. Requires --emit-onsets and SKIPS sizing: "
                         "the worst-case onset is a max over EVERY corner, so a shard that "
                         "sized from its own slice would produce a confidently wrong "
                         "excitation peak. Striping (modulo), not contiguous blocks, because "
                         "corner cost is wildly uneven -- a corner whose sweep starts above "
                         "the onset re-sweeps over a 100x extended range.")
    ap.add_argument("--emit-onsets", metavar="PATH",
                    help="write this run's measured onsets as JSON instead of sizing. Carries "
                         "the corner total, each row's GLOBAL corner index, and a fingerprint "
                         "of the renderer binary so --merge-onsets can refuse mismatched work.")
    ap.add_argument("--merge-onsets", nargs="+", metavar="PATH",
                    help="combine --emit-onsets files from every shard, verify completeness "
                         "and solver agreement, then size ONCE from the full set. Refuses on a "
                         "missing or duplicated corner index, a solver-build mismatch, or a "
                         "corner-count disagreement -- each of which would otherwise yield a "
                         "silently mis-sized excitation.")
    ap.add_argument("--corner-workers", type=int, default=None, metavar="N",
                    help="measure this many knob corners concurrently (default: auto, "
                         "cpu_count//4 capped at 6). Corners are independent, so this is the "
                         "main lever on sizing wall-clock: the loop was serial until "
                         "2026-09-17, which left onset measurement ~2 renders deep on a "
                         "12-core machine and cost ~2h on a 141-corner device. Concurrency is "
                         "a PRODUCT -- find_saturation_point() already fans out its amplitude "
                         "points -- so raising this divides the per-sweep workers down to keep "
                         "the total near the core count. 1 restores the old serial behaviour.")
    ap.add_argument("--no-cache", action="store_true")

    # excitation-building
    ap.add_argument("--sweep-file", required=True,
                     help="clip passed through to build_excitation.py --sweep-file -- it "
                          "becomes the crest-bearing segment placed at --sweep-peak. Commonly "
                          "TONE3000's standard capture sweep, which is SYNTHESIZED (sweep + "
                          "noise-staircase + blips), not a real-playing recording; a real "
                          "recording works equally well but is not required.")
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
                     help="chirp-levels max = margin x worst-case onset (default 2.0x -- past "
                          "the onset, not just at it, matching this repo's own precedent, e.g. "
                          "the non-midpoint-default pedal's excitation peak sized with headroom past where Gain's own "
                          "effect saturates)")
    ap.add_argument("--chirp-level-fracs", default="0.25,0.5,0.75,1.0",
                     help="comma list of fractions of the margined max, passed as --chirp-levels")
    ap.add_argument("--sweep-peak-frac", type=float, default=1.3,
                     help="fraction of worst-case onset used for --sweep-peak. Never go BELOW "
                          "1.0: check_transient_coverage.py's own default "
                          "margin requires transient_peak >= onset AT THE WORST CORNER, and the "
                          "worst corner's own onset IS worst-case onset by definition -- any "
                          "fraction below 1.0 guarantees that check fails there, regardless of "
                          "margin or grid, contradicting this tool's own claim that a check run "
                          "afterward should pass cleanly. Lower it only if you deliberately want "
                          "the --sweep-file content to stay short of the worst corner (e.g. to "
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
    ap.add_argument("--sweep-dur", type=float, default=None)
    ap.add_argument("--chirp-f0", type=float, default=None,
                    help="passed through to build_excitation.py (default there: 15 Hz, was "
                         "40 until 2026-09-16). Lower this to extend the chirp's frequency "
                         "floor further -- the tweed-style amp's excitation never chirped "
                         "below 40 Hz, so its trained models had zero supervision for "
                         "sustained near-DC (<20 Hz) input and blew up 8x on a real capture "
                         "sweep's own infrasonic segment at a corner no amount of "
                         "amplitude-only sizing would have caught (see scan_film_runaway.py). "
                         "Sizing (--chirp-levels/--sweep-peak, both amplitude-only) is "
                         "unaffected by this -- confirmed empirically: rebuilding at "
                         "chirp-f0=15 changed output duration/peak/rms by rounding error "
                         "only.")
    ap.add_argument("--chirp-f1", type=float, default=None,
                    help="passed through to build_excitation.py (default there: 12000 Hz)")
    ap.add_argument("--synth-burst-peaks", default=None,
                    help="passed through to build_excitation.py. 'auto' uses the derived "
                         "--chirp-levels, so a broadband instant-attack burst is inserted at "
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

    (backend, identity, cache_extra, knob_ranges, fixed, sweep_lead_silence_s, label,
     _capture) = _setup(args)

    if args.merge_onsets:
        # MERGE MODE: every corner was already measured elsewhere. Skip rendering entirely and
        # size from the verified union -- merge_onset_shards() refuses anything incomplete or
        # measured by a divergent solver build, so reaching here means the set is trustworthy.
        rows = merge_onset_shards(args.merge_onsets)
        missing = [r for r in rows if r.get("onset_v") is None]
        if missing:
            raise SystemExit(f"{len(missing)} merged corner(s) have no onset (first: "
                             f"{missing[0].get('corner')}) -- refusing to size around a corner "
                             f"whose saturation was never determined.")
        worst = max(r["onset_v"] for r in rows)
        print(f"merged {len(rows)} corner(s) from {len(args.merge_onsets)} shard file(s); "
              f"worst-case onset: {worst:.4f} V")
    else:
        print(f"finding saturation onset across the knob-grid corners of {label}...")
        tmp = str(scratch_dir("prepare_excitation", args.keep_scratch))
        worst, rows = worst_case_onset(backend, identity, cache_extra, knob_ranges, fixed, tmp,
                                        peak_max_v=args.peak_max_v, no_cache=args.no_cache,
                                        min_start_v=args.min_start_v, start_v=args.sweep_start_v,
                                        capture=_capture,
                                        corner_workers=(args.corner_workers if args.corner_workers
                                                        else max(1, min(6, (os.cpu_count() or 4) // 4))),
                                        shard=args.shard, emit_onsets=args.emit_onsets,
                                        backend_name=args.backend,
                                        full_hypercube=(False if args.no_full_hypercube else None),
                                        max_corners=args.max_corners,
                                        sample_grid=resolve_sample_grid(args.sample_grid, knob_ranges),
                                        quiet=False,
                                        lead_silence_s=sweep_lead_silence_s)
    if worst is None:      # --emit-onsets: this shard's work is written, sizing is not ours
        return 0
    print(f"worst-case onset: {worst:.4f} V (across {len(rows)} corners)")

    chirp_max = worst * args.margin
    fracs = [float(f) for f in args.chirp_level_fracs.split(",") if f.strip()]
    chirp_levels = [round(chirp_max * f, 4) for f in fracs]
    sweep_peak = round(worst * args.sweep_peak_frac, 4)
    # NOTE the default frac is 1.02, not 1.0 -- see --sweep-peak-frac's help for why sizing
    # EXACTLY at the measured worst is too tight to survive a re-measurement.
    print(f"derived: chirp_levels={chirp_levels}  sweep_peak={sweep_peak}  "
          f"(margin={args.margin}x onset)")

    build_script = HERE / "build_excitation.py"
    cmd = [sys.executable, str(build_script),
           "--sweep-file", args.sweep_file, "--output", args.output,
           "--sweep-peak", str(sweep_peak),
           "--chirp-levels", ",".join(str(p) for p in chirp_levels),
           "--lead-silence-s", str(args.excitation_lead_silence_s)]
    if args.sweep_dur is not None:
        cmd += ["--sweep-dur", str(args.sweep_dur)]
    if args.chirp_f0 is not None:
        cmd += ["--chirp-f0", str(args.chirp_f0)]
    if args.chirp_f1 is not None:
        cmd += ["--chirp-f1", str(args.chirp_f1)]
    if args.synth_burst_peaks:
        # "auto" mirrors the derived chirp levels, which is what build_excitation.py's own
        # help recommends ("Typically the same list as --chirp-levels") -- so saturation-onset
        # behaviour under a sharp transient is tested at every level rather than only the
        # loudest, which is the gap a single --noise-burst-* segment leaves.
        peaks = (",".join(str(p) for p in chirp_levels)
                 if args.synth_burst_peaks == "auto" else args.synth_burst_peaks)
        cmd += ["--synth-burst-peaks", peaks]
        if args.synth_burst_dur is not None:
            cmd += ["--synth-burst-dur", str(args.synth_burst_dur)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # SIZING PROVENANCE. build_excitation.py records WHAT it built (args, source hash, output
    # hash); it cannot record WHY those numbers, because the measurement that justified them
    # happened here. Without this block a later check_transient_coverage.py failure is
    # unexplainable from the artifacts alone: you see sweep_peak=8.6473 and an onset of
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
                # HOW the onsets were derived, not just what they were. Without this the only
                # way to tell a stale excitation from a current one is build date plus a guess
                # at whether the peak looks plausible -- which is exactly the forensics a scan
                # of nine devices needed on 2026-09-12, after find_saturation_point's onset rule
                # changed and every recipe on disk looked identical to a current one. A run that
                # mixes methods is reported as such rather than collapsed to the first.
                "onset_method": method_summary(rows),
                "worst_case_onset_v": round(float(worst), 4),
                "corner_count": len(rows),
                "corner_set": ("structural-only (DEPRECATED --no-full-hypercube: cannot represent "
                               "mixed low/high corners)" if args.no_full_hypercube
                               else f"budgeted (--max-corners {args.max_corners})"
                               if args.max_corners is not None else "full binary hypercube"),
                "sweep_peak_frac": args.sweep_peak_frac,
                "chirp_margin": args.margin,
                "peak_max_v": args.peak_max_v,
                "onsets_v": {r["corner"]: (None if r["onset_v"] is None else round(float(r["onset_v"]), 4))
                             for r in rows},
                # The knee (departure from small-signal gain) is a different quantity from the
                # onset (level at which the cell is saturated) -- at Mesa Orange's quietest
                # corner they are 27x apart, 0.053 V against 1.46 V. Recording both makes that
                # visible instead of leaving a future reader to assume one number means the
                # other, which is the conflation that produced the broken rule in the first place.
                "knees_v": {r["corner"]: (None if r.get("knee_v") is None else round(float(r["knee_v"]), 5))
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
