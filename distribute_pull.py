#!/usr/bin/env python3
"""Pull-based work scheduling for distributed renders: chunks are handed out as workers
free up, instead of the whole grid being pre-assigned.

WHY THIS EXISTS. distribute_gen.sh splits the grid ONCE, up front, by probed core count or
--weights, and each worker keeps its slice until it is done. That is fine only when every
worker's throughput is known in advance AND stays constant. Neither held on the Mesa Orange
648-combination run:

  worker3 (Linux, 12c)   43.8 combos/hr   finished its shard, then sat IDLE for hours
  worker1 (MacBook Air)   7.6 combos/hr   still grinding, 8.9 h of work left
  worker0 (M4 Pro)       40.4 combos/hr   but measured 6.3/hr while oversubscribed

The Air was slow because it was sharing the machine with an unrelated training run -- a
thing no core count or historical weight could have predicted. Static sharding turned a 1.6 h
job into a 8.9 h one, and the fix was a manual mid-run rebalance (stop the straggler, rsync
its completed outputs, re-shard the remainder). This module makes that automatic.

HOW IT WORKS. The grid is cut into CHUNKS expressed as ordinary --shard specs: with
--chunks 64, chunk i is `i-i/64`, i.e. every index where idx % 64 == i. That needs NO
renderer change -- gen_dataset_from_schx.py already accepts --shard, and already resume-skips
outputs that exist, so a re-dispatched or retried chunk costs only what is genuinely missing.

The controller keeps a queue of chunk specs and dispatches one at a time to whichever worker
is free. A fast worker simply takes more chunks. No worker can be left holding work while
another idles, and no throughput estimate is needed anywhere -- the schedule is the
measurement.

CHUNK SIZE IS THE ONE TUNING KNOB. Too large and the tail is lumpy again (the last chunk
still has to finish); too small and per-invocation overhead dominates -- each dispatch pays
an ssh round-trip plus the renderer's own startup (schx parse and symbolic solve, ~30-60 s
for a full amp). Default 64 chunks over a 648-combination grid is ~10 combinations each,
where startup is a few percent of chunk runtime.

NOT A REPLACEMENT for distribute_gen.sh's setup work (repo sync, --sync-file, gate). Run
those first; this only schedules the rendering.
"""
import argparse, csv, os, re, statistics, subprocess, sys, threading, time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from run_pipeline import load_config


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# Per-COMBINATION pacing, shared across workers. The controller used to judge a worker only
# by completed CHUNKS, so "slow", "stuck" and "legitimately working on a long chunk" were
# indistinguishable -- and gen_dataset_from_schx's own guards do not close the gap: its
# stall detector deliberately tolerates a slow-but-progressing render up to
# TOTAL_CEILING_MULT (20x) the per-rung budget, which on a full amp at oversample 8 is
# 36.7 HOURS for a single combination.
#
# Measured consequence (Mesa RED, 2026-09-04): one worker rendered at 54x real-time, needing
# ~148 min per combination against a 110 min budget. Every render progressed, so nothing
# stalled, nothing failed, and nothing printed. It held a chunk for 7.5 h and produced zero
# combinations while its neighbours finished a whole 16-combination chunk every 40 min.
#
# The fix is to judge a worker by the thing that is actually comparable across machines --
# seconds per COMBINATION -- and to take that measurement from the renderer's own existing
# per-combination progress line rather than adding any protocol.
COMBO_LINE = re.compile(r"^\[\s*\d+/\s*\d+\]\s+[\d.]+%\s+combo_(\d+)\s+(OK|FAIL)")

# grid_adequacy.py's own per-probe heartbeat ("    3/48 probes done", printed by
# render_jobs()) -- the equivalent of COMBO_LINE above, for the OTHER tool this scheduler can
# dispatch. Deliberately looser than COMBO_LINE (no percent, no OK/FAIL, no combo id): a cell
# probe has no success/failure status of its own at print time -- a failed render just yields
# NaN, discovered only when the shard's own fail count is inspected after the fact -- so
# "N/M probes done" is ALL the per-item signal grid_adequacy.py's progress line carries.
GRIDADQ_PROBE_LINE = re.compile(r"^\s*\d+/\d+\s+probes\s+done")


class ComboPace:
    """Is a worker producing combinations at a rate the rest of the fleet makes plausible?

    THE OBVIOUS METRIC IS WRONG, and cost a healthy worker its chunk before this was
    rewritten. Measuring the gap between consecutive completions looks right, but the
    renderer runs `--workers N` combinations CONCURRENTLY, so they finish in a BURST: about
    25 minutes of parallel work, then eight completions inside a few seconds. The gaps are
    [25 min, 0.3 s, 0.2 s, ...], their median collapses toward zero, and the derived deadline
    lands on its own floor -- which then killed a worker that was legitimately still in a
    cold 104-corner coverage gate, having produced nothing yet BY DESIGN.

    So this tracks two different questions with two different baselines:

      STARTUP   -- from chunk start until that worker's FIRST completion. Every worker spends
                   real time here before producing anything: the renderer runs its transient
                   coverage gate first, which on a cold saturation-onset cache is ~100 min on
                   a full amp. Judged against the fleet's median time-to-first-completion,
                   with a generous floor, because a cold cache is legitimate and common.

      STEADY    -- after the first completion. Judged on RATE (completions / elapsed), which
                   is immune to bursts because it divides by the whole elapsed time rather
                   than looking at the space between arrivals.

    Both baselines are medians, so one pathological host cannot move the bar it is judged
    against, and neither exists until `min_samples` workers have contributed -- a cold fleet
    is not evidence about any host.
    """

    def __init__(self, slow_mult=3.0, min_samples=2, startup_floor_s=5400.0,
                 steady_floor_s=1800.0):
        self.ttfc: list[float] = []        # seconds from chunk start to first completion
        self.rates: list[float] = []       # completions per second, per worker-chunk
        self.slow_mult = slow_mult
        self.min_samples = min_samples
        self.startup_floor_s = startup_floor_s
        self.steady_floor_s = steady_floor_s
        self._lock = threading.Lock()

    def record_first(self, seconds: float) -> None:
        with self._lock:
            self.ttfc.append(seconds)

    def record_rate(self, completions: int, elapsed_s: float) -> None:
        if completions <= 0 or elapsed_s <= 0:
            return
        with self._lock:
            self.rates.append(completions / elapsed_s)

    def _median(self, xs):
        return statistics.median(xs) if len(xs) >= self.min_samples else None

    def startup_limit(self) -> "float | None":
        """How long a worker may run having completed NOTHING."""
        with self._lock:
            m = self._median(self.ttfc)
        return None if m is None else max(self.startup_floor_s, m * self.slow_mult)

    def steady_limit(self) -> "float | None":
        """How long a worker that HAS produced may go without producing again."""
        with self._lock:
            m = self._median(self.rates)
        # Convert the fleet's median rate into a per-combination time, then allow a multiple.
        return None if not m else max(self.steady_floor_s, (1.0 / m) * self.slow_mult)

    def verdict(self, completions: int, since_last_s: float, elapsed_s: float):
        """(too_slow, human-readable reason). Reason is None when the worker is fine."""
        if completions == 0:
            lim = self.startup_limit()
            if lim is not None and elapsed_s > lim:
                return True, (f"produced nothing in {elapsed_s/60:.1f} min "
                              f"(fleet median time-to-first-combination "
                              f"{statistics.median(self.ttfc)/60:.1f} min; limit {lim/60:.1f} min)")
            return False, None
        lim = self.steady_limit()
        if lim is not None and since_last_s > lim:
            return True, (f"no combination in {since_last_s/60:.1f} min after producing "
                          f"{completions} (limit {lim/60:.1f} min)")
        return False, None


class Worker:
    def __init__(self, spec, job=None):
        # host:remote_dir:parallel[:env]
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(f"--worker needs host:dir:parallel[:env], got {spec!r}")
        self.host, self.dir, self.parallel = parts[0], parts[1], int(parts[2])
        self.env = parts[3] if len(parts) > 3 else ""
        # Defaulted, not required, so every existing caller that builds a Worker with just a
        # spec (this module's own tests included) keeps today's gen_dataset_from_schx.py
        # behavior byte-for-byte -- see the GEN_DATASET_JOB/GRID_ADEQUACY_JOB Job instances
        # below for what actually differs between the two tools this scheduler can dispatch.
        self.job = job or GEN_DATASET_JOB
        self.done = 0          # chunks completed
        self.combos = 0        # COMBINATIONS completed -- the comparable unit across machines
        self.slow_kills = 0
        self.failed = 0
        self.busy = False
        self.secs = 0.0        # cumulative render time, for the throughput report
        # QUARANTINE. A worker that fails FAST is worse than one that is merely slow: it drains
        # the queue faster than healthy workers can take work, and with a small --retries every
        # chunk it touches twice is dead. Measured on the Duke of Tone 252-combination run
        # (2026-09-04): worker4 could not import the transient-coverage gate (missing spicelib in
        # its venv), failed in under a second, and killed 27 of 31 chunks in ~70s while four
        # healthy machines completed 4 between them. Consecutive failures, reset by any success --
        # so a machine with one flaky chunk is not punished, but a systematically broken one gets
        # benched instead of eating the run.
        self.consec_fail = 0
        self.quarantined = False

    @property
    def rate(self):
        """Chunks per hour, MEASURED. Nothing here is estimated from core count."""
        return self.done / (self.secs / 3600) if self.secs > 0 else 0.0

    def _kill_remote(self, chunk, output):
        """Kill the gen_dataset THIS chunk started, on the worker, and release its lock.

        proc.kill() kills the local ssh CLIENT. The remote process is not signalled -- it is
        reparented to init and keeps running, still holding the renderer's exclusive
        .generation.lock, so every chunk dispatched to that host afterwards fails instantly
        with "another gen_dataset_from_schx is already generating" and the host quarantines
        itself. Abandoning a chunk without this is worse than not abandoning it at all.

        Matched on `--shard <chunk>`, which is unique to this dispatch, so a concurrent
        generation for a different chunk or dataset on the same host is never touched.
        """
        # DO NOT rm the lock file. gen_dataset_from_schx.acquire_generation_lock uses flock,
        # which auto-releases when the fd closes -- process exit, crash, or kill -- so a lock
        # file that still exists means a process is still ALIVE holding it. Deleting it does
        # not release anything: the old process keeps its lock on the now-unlinked inode while
        # a new run creates a fresh file and locks that, and the two then append to one
        # params.csv. That corrupts it SILENTLY -- duplicate rows, .npy files that still look
        # perfect, and a params.csv that no longer lines up 1:1 with outputs.npy, so knobs get
        # paired with the WRONG audio. Kill the holder and the lock takes care of itself.
        #
        # `self.job.script` parameterizes WHICH renderer's process this matches -- grid_adequacy.py
        # holds no comparable exclusive lock (each shard writes its own uniquely-named
        # --shard-out file, never a shared one), so killing it cleanly needs no lock-release
        # story at all; the pattern below is still correct for it, just less consequential.
        pat = f"{re.escape(self.job.script)}.*--shard {re.escape(chunk)}"
        cmd = f"pkill -f '{pat}'; sleep 3; pkill -9 -f '{pat}'; exit 0"
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                        self.host, cmd], capture_output=True, text=True, timeout=90)

    def run_chunk(self, chunk, gen_args, output, pace=None, on_combo=None):
        """Run one chunk, watching it ITEM BY ITEM (a combination, or a grid_adequacy cell
        probe -- see self.job) as it goes.

        The renderer already prints one line per finished item and is already invoked with
        `python -u`, so the signal exists and is unbuffered -- it was simply thrown away,
        because subprocess.run(capture_output=True) does not return until the child exits. A
        worker whose items never finish therefore said NOTHING for as long as it took the
        whole chunk to end, which for a slow host is effectively never.
        """
        env = f"export {self.env} && " if self.env else ""
        chunk_output = self.job.chunk_output(output, chunk)
        cmd = (f"cd {self.dir} && {env}./.venv/bin/python -u {self.job.script} "
               f"{gen_args} --shard {chunk} {self.job.output_flag} {chunk_output}")
        t0 = time.time()
        proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60", self.host, cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines: list[str] = []
        done = 0                       # combinations THIS chunk has completed
        last = t0                      # when the most recent one landed
        killed_slow = False

        def pump():
            nonlocal last, done
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))
                if self.job.progress_re.match(line.strip()):
                    now = time.time()
                    done += 1
                    if pace is not None and done == 1:
                        pace.record_first(now - t0)
                    last = now
                    self.combos += 1
                    if on_combo:
                        on_combo(self, line.strip())

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        while proc.poll() is None:
            t.join(timeout=5)
            if not t.is_alive():
                break
            if pace is not None:
                slow, why = pace.verdict(done, time.time() - last, time.time() - t0)
                if slow:
                    killed_slow = True
                    self.slow_kills += 1
                    lines.append(f"[controller] {why} -- abandoning this chunk")
                    self._kill_remote(chunk, output)   # BEFORE dropping the ssh, not after
                    proc.kill()
                    break
        proc.wait()
        t.join(timeout=10)
        dt = time.time() - t0
        self.secs += dt
        if pace is not None and done and not killed_slow:
            pace.record_rate(done, dt)
        rc = proc.returncode if not killed_slow else 1
        return rc, dt, "\n".join(lines)


def _relpath_or_warn(dest_label: str, v, repo_root: Path) -> str:
    """Rewrite an absolute path relative to repo_root, warning if it travels too far to be
    portable. Shared by gen_args_from_config (schx/input, below) and
    grid_adequacy_args_from_config (--config itself) -- both need the SAME reasoning applied,
    since run_chunk cds into each worker's own checkout before running: an absolute path from
    the controller can be a DIFFERENT user's home on a worker (/Users/gene, /Users/chewie,
    /home/gene), or absent entirely.
    """
    try:
        rel = os.path.relpath(Path(v).expanduser().resolve(), repo_root.resolve())
    except ValueError:                     # different drive (Windows) -- keep absolute
        rel = str(v)
    if rel.startswith(".." + os.sep + ".."):
        print(f"WARNING: {dest_label} is {rel} relative to the repo -- that is unlikely to "
              f"resolve the same way on every worker. Put it in a sibling directory of "
              f"the repo, or pass --{dest_label} yourself after --.", file=sys.stderr)
    return rel


def gen_args_from_config(config_path: Path, repo_root: Path) -> "list[str]":
    """Expand a per-circuit config.toml into gen_dataset_from_schx.py arguments.

    THE POINT IS THAT THERE IS ONE SOURCE OF TRUTH. run_pipeline.py takes --config; this
    scheduler took eleven hand-written flags instead, so rendering the same device on one
    machine and across four used two different descriptions of it that nothing reconciled.
    Retyping them is not a theoretical hazard: the Mesa RED launch (2026-09-05) omitted
    --backend, which defaults to `cpp`, and every worker quarantined in under a second.
    load_config() is run_pipeline's own loader, so the two paths cannot drift.

    PATHS ARE MADE RELATIVE to the repo root -- see _relpath_or_warn.
    """
    cfg = load_config(config_path)
    out: list[str] = []
    if cfg.get("backend"):
        out += ["--backend", str(cfg["backend"])]
    for dest, flag in (("schx", "--schx"), ("input", "--input")):
        v = cfg.get(dest)
        if v is None:
            continue
        out += [flag, _relpath_or_warn(dest, v, repo_root)]
    if cfg.get("knobs"):
        out += ["--knobs", str(cfg["knobs"])]
    for r in cfg.get("ranges") or []:
        out += ["--range", str(r)]
    if cfg.get("fixed_params"):
        out += ["--fixed-params", str(cfg["fixed_params"])]
    if cfg.get("oversample") is not None:
        out += ["--oversample", str(cfg["oversample"])]
    # Device-model overrides (a real datasheet-fitted transistor's bjt_vaf/bjt_rb/...) --
    # without this, a sharded dispatch renders through the generic model regardless of what
    # the config declares, silently disagreeing with a single-machine run_pipeline.py render
    # of the SAME config (which does forward --conv -- see its own gen_cmd construction).
    if cfg.get("conv"):
        out += ["--conv", str(cfg["conv"])]
    # Same reasoning for the capture chain: run_pipeline.py forwards it explicitly (see
    # capture_chain.resolve()'s own docstring on why config.toml alone isn't enough), but
    # this scheduler's own config expansion never did.
    if cfg.get("no_capture_chain"):
        out += ["--no-capture-chain"]
    if cfg.get("capture_hp_hz") is not None:
        out += ["--capture-hp-hz", str(cfg["capture_hp_hz"])]
    if cfg.get("capture_order") is not None:
        out += ["--capture-order", str(cfg["capture_order"])]
    return out


def grid_adequacy_args_from_config(config_path: Path, repo_root: Path) -> "list[str]":
    """--config <repo-relative path> -- the whole equivalent of gen_args_from_config for
    grid_adequacy.py, which is much simpler because it IS the one-flag-does-it-all interface
    gen_args_from_config exists to imitate: schx/input/knobs/backend/oversample all live
    inside the config file already, so there is nothing to expand into separate flags. Only
    the config path itself needs the same repo-relative treatment (see _relpath_or_warn) --
    each worker still cds into its own checkout, and this config commonly lives in a sibling
    repo (parametric-nam-models), not this one.
    """
    return ["--config", _relpath_or_warn("config", config_path, repo_root)]


def _warn_chunk_aliasing(gen_args, chunks):
    """Warn when --chunks shares a factor with a knob axis, freezing that knob inside every shard.

    Modulo sharding takes every index where idx % chunks == k. The knob grid is a product with
    the LAST knob varying fastest, so an axis of cardinality c is constant within every shard
    whenever c divides chunks -- the shard steps by `chunks`, which is a whole number of that
    axis's cycles, so it lands on the same value every time.

    That does not corrupt anything: the union of shards is still the whole grid, and every
    combination is rendered exactly once. What it breaks is gen_dataset_from_schx.py's own
    per-shard KNOB SENSITIVITY check, which measures each knob's effect across the rows it can
    see. A frozen knob shows 0.00% spread and is reported as
        WARNING <knob>: RMS varies only 0.00% -- knob may have no effect (check param_map name)
    which reads exactly like a dead knob or a param_map typo. Duke of Tone hit this on
    2026-09-04: --chunks 32 against a 4-value Volume axis (4 | 32) froze Volume in all 32 shards
    and cried wolf on a knob that had just been measured moving output by 28x.

    A chunk count coprime with every axis cardinality avoids it -- a prime is the easy answer.
    """
    ranges = []
    for i, a in enumerate(gen_args):
        if a == "--range" and i + 1 < len(gen_args):
            ranges.append(gen_args[i + 1])
        elif a.startswith("--range="):
            ranges.append(a.split("=", 1)[1])
    axes = []
    for r in ranges:
        if "=" not in r:
            continue
        name, vals = r.split("=", 1)
        n_vals = len([v for v in vals.split(",") if v.strip()])
        if n_vals >= 2:
            axes.append((name, n_vals))
    if not axes:
        return
    frozen = [(n, c) for n, c in axes if chunks % c == 0]
    if not frozen:
        return
    log("WARNING chunk-count aliasing: --chunks %d is divisible by %s"
        % (chunks, ", ".join(f"{n}'s {c} values" for n, c in frozen)))
    log("        Those knobs are CONSTANT inside every shard, so each shard's own "
        "knob-sensitivity check goes blind and reports them as 0.00% / 'may have no effect'.")
    log("        The dataset itself is unaffected -- every combination is still rendered once.")
    total_grid = 1
    for _, c in axes:
        total_grid *= c
    for cand in range(chunks, chunks + 24):
        if all(cand % c for _, c in axes):
            log(f"        Use --chunks {cand} instead (coprime with every axis; "
                f"~{total_grid / cand:.1f} combinations per chunk).")
            break


def merge_params(shard_csvs, out_path):
    """Merge per-shard params.csv files into one, keyed on the GLOBAL grid index.

    Header once, body rows appended -- the same rule distribute_gen.sh follows, and for the
    same reason: a header buried mid-table is read downstream as a combination. Deduped on
    idx because a chunk re-dispatched after a failure, or left over from an aborted run, is
    rendered by two workers under the same global index and the rows are equivalent. Sorted
    so the file is diffable and reads in grid order. Returns the row count.
    """
    rows, hdr = {}, None
    for f in shard_csvs:
        with open(f, newline="") as fh:
            rdr = csv.DictReader(fh)
            if rdr.fieldnames:
                hdr = hdr or rdr.fieldnames
            for r in rdr:
                rows[int(r["idx"])] = r
    if not hdr:
        return 0
    with open(out_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=hdr)
        wtr.writeheader()
        for i in sorted(rows):
            wtr.writerow(rows[i])
    return len(rows)


def _collect(workers, remote_out, local_dir):
    """Pull every worker's shard into one local directory, merging params.csv correctly.

    THE HALF THIS MODULE USED TO LEAVE OUT. distribute_pull schedules renders; it never
    gathered them, so collection was left to whoever ran it -- and the obvious move,
    rsyncing each worker's output dir onto the same local path, is WRONG. sig/ merges
    cleanly because its filenames are the GLOBAL grid index, but params.csv is a whole
    file per worker containing only that worker's rows, so each rsync OVERWRITES the last
    and you end up with one shard's metadata claiming to describe the entire grid.
    distribute_gen.sh has always merged it properly (header once, append body rows); this
    is that logic, for the pull scheduler.

    Worse when a worker is localhost, which is easy to arrange and easy to miss: its output
    dir IS the merge target, so the other workers' rsyncs clobber its params.csv in place.
    That is not a hypothetical -- it happened on the Duke of Tone run (2026-09-04) and cost
    48 of 252 metadata rows while all 252 .npy files sat there looking complete.

    Ordering is the fix, not detection: every worker's params.csv is copied to its own
    scratch name BEFORE any sig/ transfer, and the merged file is written LAST. That is
    correct even when a worker's remote dir and local_dir are the same directory.
    """
    local_dir = Path(local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    scratch = local_dir / ".shard_params"
    scratch.mkdir(exist_ok=True)
    for f in scratch.glob("*.csv"):
        f.unlink()

    # 1. params.csv FIRST, to per-worker names -- before anything can overwrite them.
    got = []
    for w, out in zip(workers, remote_out):
        dst = scratch / f"{w.host}.csv"
        r = subprocess.run(["rsync", "-a", f"{w.host}:{out}/params.csv", str(dst)],
                           capture_output=True, text=True)
        if r.returncode == 0 and dst.exists():
            got.append((w.host, dst))
        else:
            log(f"  collect: {w.host} has no params.csv (empty shard?) -- skipped")

    # 2. sig/ trees and the once-only artifacts. Safe in any order: global-index filenames.
    for w, out in zip(workers, remote_out):
        subprocess.run(["rsync", "-a", f"{w.host}:{out}/", str(local_dir) + "/"],
                       capture_output=True, text=True)

    # 3. merged params.csv LAST, so step 2 cannot clobber it.
    n_rows = merge_params([f for _, f in got], local_dir / "params.csv")
    if n_rows == 0:
        log("  collect: no params.csv found on any worker -- nothing merged")
        return False   # explicit: a bare `return` gave None, which is only ACCIDENTALLY falsy
    rows = range(n_rows)
    n_npy = sum(1 for _ in local_dir.glob("sig/**/*.npy"))
    log(f"  collect: {len(rows)} params rows, {n_npy} .npy files -> {local_dir}")
    if len(rows) != n_npy:
        log(f"  collect: WARNING rows != .npy ({len(rows)} vs {n_npy}) -- "
            f"gen_dataset_from_schx.py --combine will refuse this, correctly.")
    for f in scratch.glob("*.csv"):
        f.unlink()
    scratch.rmdir()
    return len(rows) == n_npy


def should_combine(consistent: bool, no_combine: bool):
    """None => combine. A string => the reason not to (logged verbatim).

    Kept separate from main() so the decision is testable without a fleet.
    """
    if no_combine:
        return "--no-combine given; run 'gen_dataset_from_schx.py --combine <dir>' when ready."
    if not consistent:
        return "rows != .npy; fix the shards first (combine would refuse this, correctly)."
    return None


def _combine(local_dir: Path) -> bool:
    """Build outputs.npy in the collected dir, using gen_dataset_from_schx's own combine().

    WHY HERE. Combining needs EVERY shard present, so it cannot live in the renderer:
    `gen_dataset_from_schx.py --shard 3/37` sees one slice of the grid and could not do it
    correctly. --collect is by definition the moment all shards exist in one directory, and it
    already merges params.csv and checks rows-vs-.npy -- the exact precondition combine needs.
    It used to do that check and then stop, leaving a directory that LOOKS finished but that
    param_train.py refuses ("outputs.npy not found"). That cost Mesa Orange and Duke of Tone
    (Overdrive) a manual step each on 2026-09-07; run_pipeline.py has had a Combine step all
    along, so only the distributed path was missing it.

    local_dir is wrapped in Path() here for the same reason _collect() already does it: the
    type hint says Path, but the real caller is main()'s args.collect, a bare argparse string
    (no type=Path on that flag). gen_dataset_from_schx.combine() does `out_dir / "params.csv"`,
    which TypeErrors on a str -- caught on the Duke of Tone (Distortion) 63-combo run
    (2026-09-11): --collect succeeded (63/63 rows and .npy files, all present on disk) and only
    the auto-combine step after it crashed, so the fix here is purely to combine() what --collect
    already built correctly, not to re-render or re-collect anything.
    """
    local_dir = Path(local_dir).expanduser()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_dataset_from_schx import combine as _do_combine
    _do_combine(local_dir)
    return True


def _collect_grid_adequacy(workers, remote_out, local_dir, config_path, extra_args):
    """Pull every worker's shard_*.json into one local directory, then run grid_adequacy.py
    --merge on them so this prints the exact same report --suggest would print unsharded.

    No params.csv-style clobber hazard here, unlike _collect: each shard file's name embeds
    its own --shard spec (e.g. shard_3-3_16.json), so every worker's output is a globally
    unique filename and a plain whole-tree rsync from each worker is safe in any order.
    """
    local_dir = Path(local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    for w, out in zip(workers, remote_out):
        subprocess.run(["rsync", "-a", f"{w.host}:{out}/", str(local_dir) + "/"],
                       capture_output=True, text=True)
    shards = sorted(local_dir.glob("shard_*.json"))
    if not shards:
        log("  collect: no shard_*.json found on any worker -- nothing to merge")
        return
    log(f"  collect: merging {len(shards)} shard file(s) ...")
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "grid_adequacy.py"),
           "--merge", *[str(s) for s in shards], "--config", str(config_path), *extra_args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        log(f"  {line}")
    if r.returncode != 0:
        log(f"  collect: merge exited {r.returncode}")
        if r.stderr.strip():
            log(f"  {r.stderr.strip()}")


@dataclass
class Job:
    """Everything distribute_pull.py's scheduler needs to know about ONE renderer/analysis
    tool, so Worker and main() stay tool-agnostic. Adding a new --tool means adding one more
    Job instance below -- no change to the queue, pacing, quarantine, or retry logic, all of
    which only ever deal with chunk specs and exit codes.
    """
    name: str
    script: str                # relative to the repo root on each worker
    progress_re: "re.Pattern"  # matched against each stripped stdout line for the heartbeat
    output_flag: str           # e.g. "--output" or "--shard-out"
    build_args: "callable"     # (config_path, repo_root, extra_args) -> list[str]
    chunk_output: "callable"   # (base_output, chunk_spec) -> str, passed after output_flag
    collect: "callable"        # (workers, remote_out, local_dir, config_path, extra_args, no_combine) -> None


def _collect_gen_dataset(workers, remote_out, local_dir, config_path, extra_args, no_combine):
    """GEN_DATASET_JOB's own collect step: merge shards, then build outputs.npy by default.

    --collect used to stop right after merging, leaving a directory that LOOKS finished but
    that param_train.py refuses ("outputs.npy not found"). Combining needs EVERY shard
    present, so it cannot live in the renderer (a single --shard sees only one slice of the
    grid) -- --collect is by definition the moment all shards exist in one directory, and it
    already checks rows-vs-.npy, the exact precondition combine needs. Cost Mesa Orange and
    Duke of Tone (Overdrive) a manual step each on 2026-09-07; run_pipeline.py has had a
    Combine step all along, so only this distributed path was missing it.
    """
    consistent = _collect(workers, remote_out, local_dir)
    why = should_combine(consistent, no_combine)
    if why is None:
        log("  combining -> outputs.npy ...")
        _combine(local_dir)
    else:
        log(f"  collect: NOT combining -- {why}")


GEN_DATASET_JOB = Job(
    name="gen_dataset",
    script="gen_dataset_from_schx.py",
    progress_re=COMBO_LINE,
    output_flag="--output",
    build_args=lambda config_path, repo_root, extra_args:
        gen_args_from_config(config_path, repo_root) + extra_args,
    chunk_output=lambda base_output, chunk: base_output,
    collect=lambda workers, remote_out, local_dir, config_path, extra_args, no_combine:
        _collect_gen_dataset(workers, remote_out, local_dir, config_path, extra_args, no_combine),
)

GRID_ADEQUACY_JOB = Job(
    name="grid_adequacy",
    script="grid_adequacy.py",
    progress_re=GRIDADQ_PROBE_LINE,
    output_flag="--shard-out",
    build_args=lambda config_path, repo_root, extra_args:
        grid_adequacy_args_from_config(config_path, repo_root) + extra_args,
    chunk_output=lambda base_output, chunk: f"{base_output}/shard_{chunk.replace('/', '_')}.json",
    # no_combine is GEN_DATASET_JOB-specific (grid_adequacy has no "combine" concept at all) --
    # accepted and ignored here so both jobs share one call site in main().
    collect=lambda workers, remote_out, local_dir, config_path, extra_args, no_combine:
        _collect_grid_adequacy(workers, remote_out, local_dir, config_path, extra_args),
)

JOBS = {j.name: j for j in (GEN_DATASET_JOB, GRID_ADEQUACY_JOB)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="append", required=True, metavar="HOST:DIR:PARALLEL[:ENV]",
                    help="repeatable. ENV is an optional 'VAR=value' exported before the run "
                         "(e.g. DOTNET_ROOT=$HOME/.dotnet on a box where dotnet is not on the "
                         "non-interactive PATH).")
    ap.add_argument("--tool", choices=sorted(JOBS), default="gen_dataset",
                    help="which script each chunk runs (default gen_dataset, i.e. "
                         "gen_dataset_from_schx.py -- unchanged behavior for existing callers). "
                         "'grid_adequacy' dispatches grid_adequacy.py --shard instead, and "
                         "--collect runs its --merge step so the final report is identical to "
                         "an unsharded grid_adequacy.py run.")
    ap.add_argument("--chunks", type=int, default=64,
                    help="how many pieces to cut the grid into (default 64). Each is dispatched "
                         "as --shard i-i/CHUNKS. See module docstring on sizing.")
    ap.add_argument("--output", required=True, help="output dir ON EACH WORKER")
    ap.add_argument("--retries", type=int, default=1,
                    help="re-queue a failed chunk this many times, on a DIFFERENT worker where "
                         "possible -- a chunk that fails on one machine and succeeds on another "
                         "is a machine problem, not a data problem, and the resume-skip means "
                         "the retry only renders what is still missing (default 1)")
    ap.add_argument("--collect", metavar="LOCAL_DIR", default=None,
                    help="after rendering, pull every worker's shard into LOCAL_DIR and merge "
                         "params.csv properly. Do NOT hand-roll this with rsync: sig/ merges "
                         "cleanly (global-index filenames) but params.csv is one file per worker "
                         "holding only that worker's rows, so a naive rsync leaves you the LAST "
                         "worker's metadata describing the whole grid.")
    ap.add_argument("--no-combine", action="store_true",
                    help="--collect: stop after merging, without building outputs.npy. Combine "
                         "NORMALISES the data and records output_scale into config.json, and for "
                         "a big grid it is a multi-GB write, so this exists for inspecting or "
                         "re-combining with a different --output-peak/--raw.")
    ap.add_argument("--config", type=Path, default=None,
                    help="per-circuit TOML, the SAME file run_pipeline.py --config takes. "
                         "Expands to the renderer's --backend/--schx/--input/--knobs/--range/"
                         "--fixed-params/--oversample so the sharded and single-machine paths "
                         "describe the device identically. Paths are rewritten relative to this "
                         "repo, since each worker runs from its own checkout. Anything you pass "
                         "after -- is appended and wins.")
    ap.add_argument("--slow-mult", type=float, default=3.0,
                    help="abandon a chunk when a worker goes this many times the FLEET MEDIAN "
                         "seconds-per-combination with nothing finished (default 3; 0 disables). "
                         "Judges a worker by the unit that is comparable across machines -- "
                         "gen_dataset's own stall detector deliberately tolerates a slow but "
                         "PROGRESSING render for 20x its per-rung budget, which on a full amp "
                         "at oversample 8 is 36.7 hours for one combination.")
    ap.add_argument("--slow-min-samples", type=int, default=2,
                    help="how many WORKER-CHUNKS the fleet must have contributed before any host "
                         "can be called slow (default 2). A cold fleet is not evidence about a "
                         "host.")
    ap.add_argument("--slow-startup-floor-min", type=float, default=90.0,
                    help="never abandon a worker that has completed NOTHING before this many "
                         "minutes (default 90). Producing nothing early is normal: the renderer "
                         "runs its transient-coverage gate first, which on a cold saturation-onset "
                         "cache is ~100 min on a full amp.")
    ap.add_argument("--slow-steady-floor-min", type=float, default=30.0,
                    help="once a worker HAS produced, never abandon it for going fewer than this "
                         "many minutes without producing again (default 30).")
    ap.add_argument("--quarantine-after", type=int, default=3,
                    help="bench a worker after this many CONSECUTIVE failures with no successes "
                         "(default 3). A fast-failing worker drains the queue faster than healthy "
                         "ones can take work -- see Worker's quarantine comment for the run where "
                         "one killed 27 of 31 chunks in ~70s. 0 disables.")
    ap.add_argument("--", dest="_sep", nargs="?", help=argparse.SUPPRESS)
    args, gen_args = ap.parse_known_args()
    if gen_args and gen_args[0] == "--":
        gen_args = gen_args[1:]
    job = JOBS[args.tool]
    extra_args = gen_args   # raw, un-expanded "after --" flags -- collect() wants these,
                             # not the config-expanded/path-relativized form built below,
                             # since collect runs locally against the ORIGINAL --config path.
    if args.config:
        # Config first, explicit flags second: argparse-style "last wins" for the renderer,
        # so --  --oversample 4  still overrides the config without editing it.
        gen_args = job.build_args(args.config, Path(__file__).resolve().parent, gen_args)
    gen_args_str = " ".join(f"'{a}'" if " " in a else a for a in gen_args)
    if not gen_args_str:
        ap.error("pass --config, or the renderer's own arguments after --")

    workers = [Worker(w, job=job) for w in args.worker]
    queue = deque(f"{i}-{i}/{args.chunks}" for i in range(args.chunks))
    attempts = {c: 0 for c in queue}
    tried_on = {c: set() for c in queue}   # chunk -> hosts that have already failed it
    pace = ComboPace(slow_mult=args.slow_mult, min_samples=args.slow_min_samples,
                     startup_floor_s=args.slow_startup_floor_min * 60.0,
                     steady_floor_s=args.slow_steady_floor_min * 60.0
                     ) if args.slow_mult > 0 else None
    lock = threading.Lock()
    total = len(queue)
    completed = failed_final = 0
    t_start = time.time()

    _warn_chunk_aliasing(gen_args, args.chunks)

    log(f"{total} chunks over {len(workers)} worker(s): "
        + ", ".join(f"{w.host}(par={w.parallel})" for w in workers))

    def worker_loop(w):
        nonlocal completed, failed_final
        while True:
            with lock:
                if w.quarantined or not queue:
                    return
                # Honour the retry contract the docstring already promises -- "on a DIFFERENT
                # worker where possible". Appending to the back of the queue does not achieve
                # that by itself: a worker failing in under a second grabs the chunk again
                # before anyone else is free. Skip past chunks this host has already failed,
                # and only fall back to one of them if nothing else is left.
                chunk = None
                for _ in range(len(queue)):
                    cand = queue.popleft()
                    if w.host in tried_on.get(cand, ()):
                        queue.append(cand)
                        continue
                    chunk = cand
                    break
                if chunk is None:
                    if not queue:
                        return
                    chunk = queue.popleft()
                attempts[chunk] = attempts.get(chunk, 0) + 1
                n_try = attempts[chunk]
            w.busy = True
            rc, dt, out = w.run_chunk(chunk, f"{gen_args_str} --workers {w.parallel}",
                                      args.output, pace=pace)
            w.busy = False
            with lock:
                if rc == 0:
                    w.consec_fail = 0
                    w.done += 1; completed += 1
                    rate = f"  {w.combos} combos" + (
                        f" @ {w.combos/(w.secs/3600):.1f}/h" if w.secs > 0 else "")
                    log(f"  {w.host:<10} chunk {chunk:<10} OK   {dt/60:5.1f} min   "
                        f"[{completed + failed_final}/{total}]{rate}")
                else:
                    w.failed += 1
                    w.consec_fail += 1
                    tried_on.setdefault(chunk, set()).add(w.host)
                    if (args.quarantine_after and w.done == 0
                            and w.consec_fail >= args.quarantine_after):
                        w.quarantined = True
                        log(f"  {w.host:<10} QUARANTINED after {w.consec_fail} consecutive "
                            f"failures and no successes -- draining the queue, not doing work. "
                            f"Remaining chunks go to the other workers. Last error:")
                        for line in out.strip().splitlines()[-4:]:
                            log(f"      {line[:110]}")
                    slow = out.rstrip().endswith("abandoning this chunk")
                    why = "TOO SLOW" if slow else f"FAIL rc={rc}"
                    if slow:
                        log(f"      {out.strip().splitlines()[-1][:150]}")
                    if n_try <= args.retries:
                        queue.append(chunk)   # back of the queue: likely a different worker
                        log(f"  {w.host:<10} chunk {chunk:<10} {why} -- requeued "
                            f"(attempt {n_try}/{args.retries + 1})")
                    else:
                        failed_final += 1
                        log(f"  {w.host:<10} chunk {chunk:<10} FAIL rc={rc} -- giving up")
                        for line in out.strip().splitlines()[-3:]:
                            log(f"      {line[:110]}")

    threads = [threading.Thread(target=worker_loop, args=(w,), daemon=True) for w in workers]
    for t in threads: t.start()
    for t in threads: t.join()

    elapsed = (time.time() - t_start) / 3600
    log(f"done in {elapsed:.2f} h -- {completed} chunk(s) ok, {failed_final} failed")

    if args.collect:
        log(f"collecting shards into {args.collect} ...")
        remote_out = []
        for w in workers:
            # resolve the output path ON THE WORKER: --output is commonly '~/dir', and a tilde
            # inside an rsync host:path argument is NOT expanded (that silently transferred
            # nothing on an earlier run), so ask the remote shell what it means.
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", w.host,
                                f"cd ~ && echo {args.output}"], capture_output=True, text=True)
            remote_out.append(r.stdout.strip() or args.output)
        job.collect(workers, remote_out, args.collect, args.config, extra_args, args.no_combine)
    else:
        log("NOTE: no --collect given. Merging by hand is a trap -- sig/ rsyncs cleanly "
            "(global-index filenames) but params.csv is ONE FILE PER WORKER holding only that "
            "worker's rows, so rsyncing each output dir onto one path leaves the LAST worker's "
            "metadata describing the whole grid. Use --collect DIR, or merge params.csv bodies "
            "yourself. gen_dataset_from_schx.py --combine refuses a row/.npy mismatch, so a "
            "botched merge fails loudly rather than silently training on a hole.")
    log("MEASURED throughput (use these for any future static weighting):")
    for w in sorted(workers, key=lambda x: -x.rate):
        log(f"  {w.host:<12} {w.done:3d} chunks  {w.rate:6.2f} chunks/h"
            + (f"  ({w.failed} failure(s))" if w.failed else ""))
    return 1 if failed_final else 0


if __name__ == "__main__":
    sys.exit(main())
