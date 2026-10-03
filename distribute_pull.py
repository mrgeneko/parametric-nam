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

It also supersedes distribute_gen.sh (deprecated): `--sync-file PATH` (repeatable) pushes
inputs git does not carry to each worker's checkout before dispatch. Repo sync and the gate are
still separate steps (gate_config.py / run_pipeline.py fleet mode); this schedules the rendering.
"""
import argparse, csv, json, os, posixpath, re, shlex, statistics, subprocess, sys, threading, time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from run_pipeline import load_config
from cpu_topology import physical_cpu_count
import ssh_target


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

# measure_truncation.py's own per-setting heartbeat ("  3/15 settings done", printed by
# measure() once per knob setting -- see its own comment on why this is per-setting rather
# than per-render), the equivalent of GRIDADQ_PROBE_LINE for the oversample-measurement job.
MEASURE_TRUNC_LINE = re.compile(r"^\s*\d+/\d+\s+settings\s+done")

# check_transient_coverage.py's own per-corner completion line ("  all-min          onset=
# 1.234 V  OK", printed by _measure() once per corner, in BOTH sharded and unsharded mode --
# see that function's own docstring comment). No running N/M count the way GRIDADQ_PROBE_LINE
# has, but "onset=" only ever appears on this one per-CORNER-COMPLETION line within a single
# shard's own stdout (the OTHER "onset=" print, in --merge-onsets' final report, runs in a
# separate subprocess collect() invokes directly -- run_chunk's pump() never sees it), so it
# is an equally reliable per-item signal for the stall detector/pacing to key on.
TCOV_CORNER_LINE = re.compile(r"^\S+.*\bonset=")


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
        # host:remote_dir[:parallel[:env]] -- parallel is OPTIONAL (2026-09-26): omit it, or
        # leave it empty/"auto" (host:dir::env needs the empty form to still reach env), to
        # auto-detect the worker's PHYSICAL core count over SSH via cpu_topology.py. Explicit
        # values keep working byte-for-byte -- this only fills in what used to be required.
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"--worker needs host:dir[:parallel[:env]], got {spec!r}")
        self.host, self.dir = parts[0], parts[1]
        parallel_str = parts[2] if len(parts) > 2 else ""
        if parallel_str.strip() == "" or parallel_str.strip().lower() == "auto":
            probe_host = None if self.host in ("localhost", "127.0.0.1") else self.host
            self.parallel = physical_cpu_count(probe_host)
            print(f"[controller] {self.host}: --worker parallel not given, auto-detected "
                  f"{self.parallel} physical core(s)", file=sys.stderr)
        else:
            self.parallel = int(parallel_str)
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
        subprocess.run(ssh_target.ssh_argv(self.host, "-o", "BatchMode=yes", "-o",
                                           "ConnectTimeout=20") + [cmd],
                       capture_output=True, text=True, timeout=90)

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
            ssh_target.ssh_argv(self.host, "-o", "BatchMode=yes", "-o",
                                "ServerAliveInterval=60") + [cmd],
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
        # Any non-zero rc that DIDN'T already go through the killed_slow branch above still
        # needs _kill_remote(): an SSH-level drop (broken pipe -> the pump thread's readline
        # loop hits EOF -> `if not t.is_alive(): break` -> proc.wait() picks up ssh's own
        # rc=255) leaves the remote gen_dataset process running, reparented to init, still
        # holding .generation.lock -- exactly what _kill_remote's docstring describes, just
        # reached from a different branch than the one that originally called it. Without
        # this, the NEXT chunk dispatched to this host (immediately, since the scheduler has
        # no idea the previous holder is still alive) fails instantly on the stale lock, and
        # so does every chunk after that -- one dropped connection cascading into the whole
        # remainder of a host's work (30 of 64 AC30 Top Boost chunks, 2026-09-24). Safe to
        # call unconditionally on failure: if the remote process already exited cleanly (a
        # real application error, not a connection drop), its own flock release already
        # dropped the lock, and pkill finds nothing to match -- a harmless no-op, not a
        # redundant kill of something still needed.
        if rc != 0 and not killed_slow:
            self._kill_remote(chunk, output)
        return rc, dt, "\n".join(lines)



def configure_ssh(inventory_arg) -> "Path | None":
    """--inventory: None = flag absent = plain `ssh HOST` (the operator's own ~/.ssh/config);
    "__DEFAULT__" = bare flag = the default inventory file; anything else = that path."""
    if inventory_arg is None:
        return None
    cfg = ssh_target.configure(
        None if inventory_arg == "__DEFAULT__" else Path(inventory_arg).expanduser())
    log(f"ssh: inventory config {cfg}" if cfg
        else "ssh: --inventory given but no host in it sets address/user/port/identity_file "
             "-- using plain ssh")
    return cfg

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


# ---------------------------------------------------------------------------
# Input sync (--sync-file). Pushes inputs a render needs that git does not carry -- the
# excitation wav (gitignored), a .schx or --pedal-dir module that lives outside the worker's
# checkout -- into each worker's OWN repo dir at the same repo-relative path, which is what
# run_chunk's `cd <worker dir> && ...` expects. Replaces distribute_gen.sh's --sync-file.
# ---------------------------------------------------------------------------
_RSYNC_SAFE = re.compile(r"^[A-Za-z0-9_@%+=:,./~-]+$")


def _remote_quote(p: str) -> str:
    """shlex.quote for a REMOTE shell, keeping a leading `~/` unquoted so it still expands."""
    if p.startswith("~/"):
        return "~/" + shlex.quote(p[2:])
    return shlex.quote(p)


def resolve_remote_dir(host: str, path: str, timeout: float = 30.0) -> str:
    """`path` as the worker's own shell resolves it (`~` -> its real home). rsync's host:path
    does not reliably expand `~` (it depends on the rsync version's argument protection), so
    resolve once and use the absolute path. Falls back to `path` unchanged if ssh fails --
    the caller's next command then fails with a real error instead of this guessing."""
    try:
        r = subprocess.run(ssh_target.ssh_argv(host, "-o", "BatchMode=yes")
                           + [f"cd ~ && echo {path}"], capture_output=True, text=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return path
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else path


def sync_path_to_worker(host: str, worker_dir: str, local_path, rel: str,
                        timeout: float = 300.0) -> "tuple[bool, str]":
    """Copy `local_path` (file or directory) to `<worker_dir>/<rel>` on `host`. Never raises.

    A file lands at exactly that path; a directory's CONTENTS land in that directory. Names
    with characters outside a conservative safe set (spaces, parentheses -- e.g. "Big Muff
    (v2).schx", which distribute_gen.sh's --sync-file once had to be fixed for) go through
    tar-over-ssh instead of rsync: rsync's remote-path quoting differs between versions
    (old-style args are re-parsed by the remote shell, 3.2.4+ escape them), so no single
    spelling is right for both, while a quoted `tar -xf -` command is.
    """
    local = Path(local_path).expanduser()
    if not local.exists():
        return False, f"not found locally: {local}"
    base = resolve_remote_dir(host, worker_dir).rstrip("/")
    dest = f"{base}/{rel}"
    is_dir = local.is_dir()
    dest_dir = dest if is_dir else posixpath.dirname(dest)
    try:
        mk = subprocess.run(ssh_target.ssh_argv(host, "-o", "BatchMode=yes")
                            + [f"mkdir -p {_remote_quote(dest_dir)}"],
                            capture_output=True, text=True, timeout=30)
        if mk.returncode != 0:
            return False, f"mkdir failed: {mk.stderr.strip()[:150]}"
        if _RSYNC_SAFE.match(dest):
            src = str(local) + ("/" if is_dir else "")
            r = subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), src,
                                f"{host}:{dest}{'/' if is_dir else ''}"],
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0, "ok" if r.returncode == 0 else r.stderr.strip()[:150]
        tar_src = ["-C", str(local), "."] if is_dir else ["-C", str(local.parent), local.name]
        env = dict(os.environ, COPYFILE_DISABLE="1")      # bsdtar: no ._AppleDouble sidecars
        tar = subprocess.Popen(["tar", "-cf", "-", *tar_src], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
        try:
            r = subprocess.run(ssh_target.ssh_argv(host, "-o", "BatchMode=yes")
                               + [f"tar -C {_remote_quote(dest_dir)} -xf -"], stdin=tar.stdout,
                               capture_output=True, text=True, timeout=timeout)
        finally:
            tar.stdout.close()
            tar.wait()
        if tar.returncode != 0 or r.returncode != 0:
            return False, (r.stderr.strip() or tar.stderr.read().decode(errors="replace").strip()
                           or f"tar exit {tar.returncode}/{r.returncode}")[:150]
        return True, "ok"
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"


def sync_files(workers: "list", paths: "list[str]", repo_root: "Path | None" = None) -> "list":
    """Push every `paths` entry to every worker (see sync_path_to_worker); returns the workers
    that received ALL of them. One that failed is EXCLUDED with a WARNING, not left in the
    pool: a worker missing an input fails or quarantines every chunk it touches, which is
    worse than not having it (the same reasoning verify_workers applies to a stale checkout)."""
    repo_root = repo_root or Path(__file__).resolve().parent
    rels = {p: _relpath_or_warn("sync-file", p, repo_root) for p in paths}
    ok_workers = []
    for w in workers:
        failures = []
        for p in paths:
            ok, detail = sync_path_to_worker(w.host, w.dir, p, rels[p])
            if not ok:
                failures.append(f"{p}: {detail}")
        if failures:
            log(f"WARNING: {w.host}: --sync-file failed, EXCLUDING this worker: "
                + "; ".join(failures))
        else:
            log(f"{w.host}: synced {len(paths)} file(s)")
            ok_workers.append(w)
    return ok_workers


# ---------------------------------------------------------------------------
# Dispatch-time version verification (fleet-deployment-proposal.md step 3,
# docs/implementation-roadmap.md item 5). Checked ONCE per worker before any chunk is
# dispatched to it, not per chunk -- the same "one-time setup, not a per-item cost" reasoning
# the gate check and physical-core auto-detect already apply. Would have caught two real
# incidents fleet-deployment-proposal.md's own "what actually went wrong" table records: a
# worker 472 commits stale, and a render that silently began before a fix landed.
# ---------------------------------------------------------------------------
def local_commit_sha() -> "str | None":
    """This machine's own checkout HEAD -- what every worker's checkout is compared against.
    None (not raised) if this isn't a git checkout at all; the caller decides what that means
    for the check as a whole rather than this function guessing."""
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def extract_backend(gen_args: "list[str]") -> "str | None":
    """Pulls --backend's value back out of an already-built gen_args list (from --config
    expansion or the raw passthrough) -- there is exactly one source of truth for what backend
    a dispatch uses, and this reads it rather than asking the caller to say it twice."""
    try:
        return gen_args[gen_args.index("--backend") + 1]
    except (ValueError, IndexError):
        return None


def version_check_command(worker_dir: str, backend: "str | None") -> str:
    """One remote command, one ssh round trip: the worker's own commit SHA, then its own
    prepare_excitation.solver_identity() -- self-invoked on the worker rather than
    re-implemented here in shell, so this automatically stays in sync with that function's own
    logic and output format. Self-invocation is safe here (unlike fleet_inventory.py's own
    bootstrapping probes, which deliberately avoid it): a dispatch is about to send real render
    work to this exact checkout, so it is guaranteed to already exist as a working venv, not
    something this check needs to discover the hard way."""
    return (f"cd {worker_dir} && git rev-parse HEAD && "
           f"./.venv/bin/python3 -c \"from prepare_excitation import solver_identity; "
           f"print(solver_identity({(backend or 'cpp')!r}))\"")


def parse_version_check_output(stdout: str) -> "tuple[str, str] | tuple[None, None]":
    lines = [ln.strip() for ln in (stdout or "").strip().splitlines() if ln.strip()]
    return (lines[0], lines[1]) if len(lines) >= 2 else (None, None)


def compare_versions(worker_sha, controller_sha, worker_solver, controller_solver) -> "tuple[bool, str]":
    """The refuse-or-not decision, isolated from the ssh round trip so it's testable without a
    network. controller_sha=None means the controller itself isn't a git checkout -- callers
    should skip the whole check rather than call this, since nothing is verifiable then; this
    function does not special-case it.

    A solver-identity mismatch is only a refusal when BOTH sides produced a DETERMINATE string
    that actually differs -- "livespice:UNKNOWN" (git commands failed on one side) means
    "couldn't tell", not "different", and refusing on that basis would treat not-knowing as
    wrong. Two workers on different, non-livespice backends both report an identical
    "<backend>:unidentified" marker, so this composes with no per-backend special-casing.
    """
    if worker_sha is None:
        return False, "could not read the worker's commit SHA (ssh/git failed, or bad output)"
    if worker_sha != controller_sha:
        return False, f"commit mismatch: worker {worker_sha[:12]}, controller {controller_sha[:12]}"
    if (worker_solver and controller_solver and worker_solver != controller_solver
            and "UNKNOWN" not in worker_solver and "UNKNOWN" not in controller_solver):
        return False, f"solver mismatch: worker {worker_solver}, controller {controller_solver}"
    return True, f"commit {worker_sha[:12]} matches; solver {worker_solver or '(undetermined)'}"


def probe_worker_version(host: str, worker_dir: str, backend: "str | None",
                         timeout: float = 20.0) -> "tuple[str | None, str | None]":
    """The real ssh round trip. Never raises -- an unreachable/broken worker comes back as
    (None, None), which compare_versions() already treats as a refusal."""
    cmd = version_check_command(worker_dir, backend)
    try:
        r = subprocess.run(ssh_target.ssh_argv(host, "-o", "BatchMode=yes", "-o",
                                               f"ConnectTimeout={int(timeout)}") + [cmd],
                           capture_output=True, text=True, timeout=timeout + 10)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if r.returncode != 0:
        return None, None
    return parse_version_check_output(r.stdout)


def verify_workers(workers: "list", backend: "str | None") -> "list":
    """Filters `workers` down to those that pass the dispatch-time version check, logging every
    decision. Returns the input UNCHANGED (a no-op, not a filter) if this controller checkout
    itself has no determinable commit SHA -- nothing here would be verifiable against, and
    refusing every worker for a controller-side limitation would be worse than not checking."""
    controller_sha = local_commit_sha()
    if controller_sha is None:
        log("WARNING: could not determine this controller's own commit SHA (not a git "
            "checkout?) -- skipping dispatch-time version verification for this run.")
        return workers
    controller_solver = None
    if backend:
        from prepare_excitation import solver_identity   # local: keeps this module's own
        controller_solver = solver_identity(backend)      # import footprint light otherwise
    kept = []
    for w in workers:
        wsha, wsolver = probe_worker_version(w.host, w.dir, backend)
        ok, reason = compare_versions(wsha, controller_sha, wsolver, controller_solver)
        if ok:
            log(f"{w.host}: version check OK -- {reason}")
            kept.append(w)
        else:
            log(f"WARNING: {w.host}: version check FAILED -- {reason} -- excluding from this "
                f"run. Pass --skip-version-check to bypass (or fix the checkout).")
    return kept


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
    for dest, flag in (("schx", "--schx"), ("input", "--input"), ("pedal_dir", "--pedal-dir")):
        v = cfg.get(dest)
        if v is None:
            continue
        out += [flag, _relpath_or_warn(dest, v, repo_root)]
    # ngspice-deck fields (2026-09-26) -- this scheduler had never dispatched an ngspice-deck
    # device before JC-120's sag/reactive-speaker render, so this gap was never exercised:
    # every chunk quarantined in seconds with "--pedal-dir and --module are required for
    # --backend ngspice-deck", the exact Mesa RED "omitted --backend" failure mode this
    # function's own docstring already warns about, just for a newer set of fields. module/
    # probe-node/maxstep are plain values (not paths), unlike pedal-dir above.
    if cfg.get("module"):
        out += ["--module", str(cfg["module"])]
    if cfg.get("probe_node"):
        out += ["--probe-node", str(cfg["probe_node"])]
    if cfg.get("maxstep") is not None:
        out += ["--maxstep", str(cfg["maxstep"])]
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


def check_transient_coverage_args_from_config(config_path: Path, repo_root: Path) -> "list[str]":
    """--config <repo-relative path> -- identical shape to grid_adequacy_args_from_config,
    since check_transient_coverage.py takes the same one-flag-does-it-all --config interface
    (its own --transient-peak, when not given, is auto-read from the excitation's own
    recipe.json sidecar -- see that tool's docstring). Closes the "transient-coverage isn't
    [sharded via distribute_pull.py] yet" gap config-gate-proposal.md and
    per-item-sharding-proposal.md both flag, needed for gate_config.py's fleet mode
    (docs/implementation-roadmap.md item 8) to shard it the same way it already shards
    grid_adequacy.
    """
    return ["--config", _relpath_or_warn("config", config_path, repo_root)]


def prepare_excitation_args_from_config(config_path: Path, repo_root: Path) -> "list[str]":
    """--config/--backend from the config -- --backend is derived explicitly (unlike
    check_transient_coverage_args_from_config's bare --config) because prepare_excitation.py's
    own --backend is `required=True` at the argparse level and is NOT read from --config the
    way check_transient_coverage.py's is; omitting it fails every shard identically, the same
    Mesa RED "omitted --backend" failure mode gen_args_from_config's docstring describes.

    --sweep-file has no config.toml equivalent -- a device's persistent config does not
    remember which capture clip it was last sized against -- and --output is required
    unconditionally by prepare_excitation.py's own argparse even for a --emit-onsets shard
    that never writes it (see its main(): `if worst is None: return 0` right after the shard
    file is written). Both must be passed after -- by the caller, same as a single-machine
    `prepare_excitation.py --config ... -- --sweep-file ... --output ...` already needs them.
    _collect_prepare_excitation reuses this same function for its own --merge-onsets command,
    so --backend is derived identically for the shard dispatch and the final sizing/build.
    """
    cfg = load_config(config_path)
    out = ["--config", _relpath_or_warn("config", config_path, repo_root)]
    if cfg.get("backend"):
        out += ["--backend", str(cfg["backend"])]
    return out


def measure_truncation_args_from_config(config_path: Path, repo_root: Path) -> "list[str]":
    """--config and --input, repo-relative -- the equivalent of gen_args_from_config for
    measure_truncation.py. Its [knobs]/schx come from --config already (load_device() reads
    them the same way run_pipeline.py's load_config() does); --input is still its own flag
    (measure_truncation.py takes it separately, so the value can be the sweep the DATASET was
    actually rendered with even if a config's own `input` field was later resized -- see
    docs/new-circuit-walkthrough.md step 4) but defaults to the config's own value here so a
    dispatch does not have to repeat it.
    """
    cfg = load_config(config_path)
    out = ["--config", _relpath_or_warn("config", config_path, repo_root)]
    if cfg.get("input"):
        out += ["--input", _relpath_or_warn("input", cfg["input"], repo_root)]
    return out


def _collect_measure_truncation(workers, remote_out, local_dir, config_path, extra_args):
    """Pull every worker's shard_*.json into one local directory, then run
    measure_truncation.py --merge on them so this prints the exact same table an unsharded run
    would -- same no-clobber-hazard reasoning as _collect_grid_adequacy: each shard file's name
    embeds its own --shard spec, so a plain whole-tree rsync from every worker is safe in any
    order.

    Runs LOCALLY (no ssh, no repo_root translation) -- config_path is used as given, same as
    _collect_grid_adequacy.
    """
    local_dir = Path(local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    for w, out in zip(workers, remote_out):
        subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), f"{w.host}:{out}/", str(local_dir) + "/"],
                       capture_output=True, text=True)
    shards = sorted(local_dir.glob("shard_*.json"))
    if not shards:
        log("  collect: no shard_*.json found on any worker -- nothing to merge")
        return
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "measure_truncation.py"),
           "--merge", *[str(s) for s in shards], "--config", str(config_path)]
    # --input is required by measure_truncation.py's own CLI even in --merge mode (the
    # post-merge reference-convergence check re-renders), so supply it from the config unless
    # extra_args already overrides it -- same fallback gen_args_from_config's --input relies on.
    if "--input" not in extra_args:
        input_wav = load_config(config_path).get("input")
        if input_wav:
            cmd += ["--input", str(input_wav)]
        else:
            log(f"  collect: {config_path} has no [input] and --input was not passed -- the "
                f"post-merge reference-convergence check will fail to start; pass --input "
                f"yourself after --")
    cmd += extra_args
    log(f"  collect: merging {len(shards)} shard file(s) ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        log(f"  {line}")
    if r.returncode != 0:
        log(f"  collect: merge exited {r.returncode}")
        if r.stderr.strip():
            log(f"  {r.stderr.strip()}")


def _parse_range_axes(gen_args: "list[str]") -> "list[tuple[str, int]]":
    """[(knob_name, cardinality), ...] from every --range in an already-built gen_args list.
    Shared by _warn_chunk_aliasing and derive_item_count -- one parser, one source of truth for
    what a --range argument means. Handles both `--range X=...` and `--range=X=...` forms.

    A repeated --range for the SAME knob keeps only the last occurrence, not both: that
    matches gen_dataset_from_schx.py's own last-one-wins parsing (values_per_knob[kname] = ...
    in a loop), which is how a config's own --range is meant to be overridden by an extra
    --range passed after `--` (documented usage, e.g. `fleet_ctl.py submit --config ... --
    --range Drive=0,1,2`). Treating both as separate axes silently multiplied the derived
    item count instead of overriding it.
    """
    ranges = []
    for i, a in enumerate(gen_args):
        if a == "--range" and i + 1 < len(gen_args):
            ranges.append(gen_args[i + 1])
        elif a.startswith("--range="):
            ranges.append(a.split("=", 1)[1])
    axes = {}
    order = []
    for r in ranges:
        if "=" not in r:
            continue
        name, vals = r.split("=", 1)
        n_vals = len([v for v in vals.split(",") if v.strip()])
        if n_vals >= 1:
            if name not in axes:
                order.append(name)
            axes[name] = n_vals
    return [(name, axes[name]) for name in order]


def derive_item_count(gen_args: "list[str]", items_override: "int | None" = None) -> int:
    """The real combination count a per-item run needs (per-item-sharding-proposal.md Phase 2),
    from the SAME expanded gen_args used to dispatch -- one description of the grid, not a
    second one this function invents independently. `items_override` (--items) is for a job
    whose grid isn't expressed via --range at all.

    MUST FAIL LOUDLY rather than guess: the proposal is explicit that too high just wastes
    ~80ms per empty dispatch, but too low SILENTLY DROPS combinations -- there is no safe
    default to fall back to, so an unparseable or absent grid description raises.
    """
    if items_override is not None:
        if items_override <= 0:
            raise ValueError(f"--items must be positive, got {items_override}")
        return items_override
    axes = _parse_range_axes(gen_args)
    if not axes:
        raise ValueError("could not derive an item count: no --range found in the dispatched "
                         "arguments, and no --items override was given -- pass --items N "
                         "explicitly for a job whose grid isn't expressed via --range")
    count = 1
    for _, n in axes:
        count *= n
    return count


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
    axes = [(n, c) for n, c in _parse_range_axes(gen_args) if c >= 2]
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
    so the file is diffable and reads in grid order. Returns (row count, sorted index list) --
    the index list matters as much as the count: a count alone cannot tell a caller WHICH
    indices are missing when it doesn't match the .npy count (see _collect).
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
        return 0, []
    with open(out_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=hdr)
        wtr.writeheader()
        for i in sorted(rows):
            wtr.writerow(rows[i])
    return len(rows), sorted(rows)


def _repair_missing(local_dir: Path, config_path, extra_args, missing_npy, repo_root: Path):
    """Regenerate exactly the combinations whose .npy is missing a params.csv row (or vice
    versa), LOCALLY and one index at a time.

    One at a time is not a style choice: gen_dataset_from_schx.py takes an exclusive,
    non-blocking lock on --output for the whole run (acquire_generation_lock) -- a second
    concurrent invocation against the SAME --output is refused outright, loudly, by design
    (two concurrent generations into one dir corrupt params.csv silently otherwise). Running
    these sequentially is simply working with that lock instead of fighting it.

    Reuses gen_args_from_config -- the SAME expansion the original dispatch used -- so a
    repair render cannot silently drift onto different knob ranges/fixed-params/oversample
    than the run it's patching. --shard IDX-IDX/TOTAL selects exactly one global index
    (modulo TOTAL is a no-op for any IDX < TOTAL); TOTAL comes from the collected dir's own
    config.json (combination_count), falling back to one past the highest index this run has
    ever seen if that key or file is missing.
    """
    total = None
    cfg_json = local_dir / "config.json"
    if cfg_json.exists():
        try:
            total = json.loads(cfg_json.read_text()).get("combination_count")
        except (json.JSONDecodeError, OSError):
            pass
    if not total:
        total = max(missing_npy) + 1
        log(f"  repair: config.json has no combination_count -- using {total} "
            f"(one past the highest index seen)")
    base_args = gen_args_from_config(config_path, repo_root) + list(extra_args)
    ok = []
    for idx in missing_npy:
        log(f"  repair: regenerating index {idx} (--shard {idx}-{idx}/{total}) ...")
        cmd = [sys.executable, str(repo_root / "gen_dataset_from_schx.py"), *base_args,
               "--output", str(local_dir), "--shard", f"{idx}-{idx}/{total}"]
        r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"  repair: index {idx} FAILED (exit {r.returncode}) -- "
                f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '(no stderr)'}")
        else:
            ok.append(idx)
    log(f"  repair: {len(ok)}/{len(missing_npy)} index(es) regenerated successfully")
    return ok


def _check_expected(csv_idx, npy_idx, sizes, expected_count) -> bool:
    """The exact-count and uniform-size assertions, shared by the local and sink collects.
    Logs each violation; returns True only when there are none."""
    ok = True
    if len(csv_idx) != expected_count or len(npy_idx) != expected_count:
        ok = False
        log(f"  collect: WARNING expected {expected_count} combination(s), got "
            f"{len(csv_idx)} params row(s) / {len(npy_idx)} .npy file(s) -- some "
            f"combination was never rendered anywhere, not just misfiled between workers.")
    if len(set(sizes)) > 1:
        ok = False
        log(f"  collect: WARNING .npy files are not all the same byte size ({sorted(set(sizes))}) "
            f"-- one device's renders should be uniform length; this usually means a "
            f"truncated or corrupted transfer.")
    return ok


def _collect(workers, remote_out, local_dir, config_path=None, extra_args=(), repair_missing=False,
            labels=None, expected_count=None):
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

    A row-count-vs-.npy-count MISMATCH used to be reported as just that -- two numbers, no
    indication of which combinations were actually affected. That cost a manual npy-vs-csv
    diff on the Ceriatone Captain Reverb (sag) run (2026-09-21): two rc=255 SSH-drop retries
    left index 801 and 842 with a rendered .npy but no params.csv row (the resume-skip check
    treats an existing .npy as "done" and never re-appends a row for it on a retry whose
    OWN attempt is what got interrupted before its row was written). Now the exact indices
    on each side of the mismatch are computed and logged by number, and --repair-missing lets
    a caller regenerate exactly those indices automatically instead of doing it by hand.

    `labels` (per-item-sharding-proposal.md Phase 3): the params.csv SCRATCH FILENAME per
    entry, defaulting to each worker's `.host`. Needed once one host can appear MULTIPLE times
    in `workers`/`remote_out` (per-item mode's K slots, one entry per (worker, slot) sharing
    the SAME `.host`) -- without a distinct label, slot 1's scratch copy would silently
    overwrite slot 0's before either was merged, exactly the clobber-hazard this function
    exists to prevent in the first place, just recreated one level down.

    `expected_count` (Phase 3's other assertion): when given, `consistent` also requires the
    merged row count and .npy count to equal it exactly, and every .npy to be the SAME byte
    size (a real render's outputs of one device are the same duration/format; the reference
    run this was checked against had all 24 at 39,379,328 bytes, so truncation shows up
    immediately). Phase 0's atomic .npy writes should make a genuinely partial file
    unreachable, so this is a second, independent line of defense, not a substitute for that.
    """
    local_dir = Path(local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    scratch = local_dir / ".shard_params"
    scratch.mkdir(exist_ok=True)
    for f in scratch.glob("*.csv"):
        f.unlink()

    labels = list(labels) if labels is not None else [w.host for w in workers]

    # 1. params.csv FIRST, to per-entry scratch names -- before anything can overwrite them.
    got = []
    for w, out, label in zip(workers, remote_out, labels):
        dst = scratch / f"{label}.csv"
        r = subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), f"{w.host}:{out}/params.csv", str(dst)],
                           capture_output=True, text=True)
        if r.returncode == 0 and dst.exists():
            got.append((w.host, dst))
        else:
            reason = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
            log(f"  collect: {label} has no params.csv -- skipped ({reason[0][:150]})")

    # 2. sig/ trees and the once-only artifacts. Safe in any order: global-index filenames.
    for w, out in zip(workers, remote_out):
        subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), f"{w.host}:{out}/", str(local_dir) + "/"],
                       capture_output=True, text=True)

    # 3. merged params.csv LAST, so step 2 cannot clobber it.
    n_rows, csv_idx = merge_params([f for _, f in got], local_dir / "params.csv")
    if n_rows == 0:
        log("  collect: no params.csv found on any worker -- nothing merged")
        return False   # explicit: a bare `return` gave None, which is only ACCIDENTALLY falsy
    npy_idx = sorted(int(p.stem) for p in local_dir.glob("sig/**/*.npy"))
    log(f"  collect: {len(csv_idx)} params rows, {len(npy_idx)} .npy files -> {local_dir}")
    csv_set, npy_set = set(csv_idx), set(npy_idx)
    orphan_npy = sorted(npy_set - csv_set)     # .npy exists, no params.csv row for it
    orphan_csv = sorted(csv_set - npy_set)     # params.csv row exists, no .npy for it
    consistent = not orphan_npy and not orphan_csv
    if not consistent:
        if orphan_npy:
            log(f"  collect: WARNING {len(orphan_npy)} .npy file(s) with no params.csv row: "
                f"{orphan_npy}")
        if orphan_csv:
            log(f"  collect: WARNING {len(orphan_csv)} params.csv row(s) with no .npy file: "
                f"{orphan_csv}")
        log("  collect: gen_dataset_from_schx.py --combine will refuse this, correctly.")
        if repair_missing and orphan_npy and config_path is not None:
            fixed = _repair_missing(local_dir, config_path, extra_args, orphan_npy,
                                     Path(__file__).resolve().parent)
            if fixed:
                # Re-check from scratch, reading local_dir/params.csv directly (the repair
                # renders append to it in place) -- rather than assume success, since a
                # render that exits 0 is not proof its row landed (that is exactly how
                # orphan_npy happens in the first place).
                with open(local_dir / "params.csv", newline="") as fh:
                    csv_idx = sorted(int(r["idx"]) for r in csv.DictReader(fh))
                npy_idx = sorted(int(p.stem) for p in local_dir.glob("sig/**/*.npy"))
                csv_set, npy_set = set(csv_idx), set(npy_idx)
                consistent = csv_set == npy_set
                log(f"  collect: after repair -- {len(csv_idx)} params rows, "
                    f"{len(npy_idx)} .npy files, consistent={consistent}")
        elif orphan_npy and config_path is None:
            log("  collect: --repair-missing needs --config to know how to re-render -- "
                "not attempting a repair.")

    # Phase 3's other assertion (per-item-sharding-proposal.md): the exact expected count, not
    # just internal 1:1 consistency -- an orphan-free 40-row merge from a 65-combination job is
    # "consistent" by the checks above and still means 25 combinations were never rendered
    # anywhere. Only enforced when the caller knows N (per-item mode always does; the legacy
    # --chunks path does not always have one, so this stays opt-in).
    if expected_count is not None:
        sizes = {p.stat().st_size for p in local_dir.glob("sig/**/*.npy")}
        if not _check_expected(csv_idx, npy_idx, sizes, expected_count):
            consistent = False
    for f in scratch.glob("*.csv"):
        f.unlink()
    scratch.rmdir()
    return consistent


# ---------------------------------------------------------------------------
# Destination-aware collect (fleet-deployment-proposal.md §5, implementation-roadmap.md item
# 10). `--collect HOST:DIR` merges the shards ON `HOST` -- the machine that will train on the
# dataset -- instead of on this controller. The motivating run collected 19 GB onto the
# controller and then copied it again to the trainer; nothing said where the dataset belonged.
# Per shard, in order of preference: (a) the worker IS the sink: a local copy on that host,
# no network at all; (b) direct: the worker rsyncs to the sink itself (needs worker->sink ssh,
# which nothing guarantees, so it falls through on any failure); (c) relay: tar piped
# worker -> controller -> sink, one pass, nothing written to this controller's disk.
# ---------------------------------------------------------------------------
_SINK_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*):(.+)$")


def parse_collect_dest(arg: str) -> "tuple[str | None, str]":
    """`HOST:DIR` -> (HOST, DIR); anything else -> (None, arg), i.e. a local directory. A local
    directory whose name contains a colon needs a `./` prefix (`./data:v2`)."""
    m = _SINK_RE.match(arg)
    return (m.group(1), m.group(2)) if m else (None, arg)


@dataclass
class Sink:
    host: str
    dir: str
    repo: "str | None" = None       # sink's checkout, for the remote --combine
    target: "str | None" = None     # how a WORKER should address the sink: [user@]address
    port: "int | None" = None


def sink_from_inventory(host: str, dir_: str, inv: dict, repo: "str | None" = None) -> Sink:
    e = inv.get(host) or {}
    addr = e.get("address") or host
    target = f"{e['user']}@{addr}" if e.get("user") else addr
    port = int(e["port"]) if e.get("port") and int(e["port"]) != 22 else None
    return Sink(host, dir_, repo or e.get("repo"), target, port)


def build_sink(host, path, inventory_arg, sink_repo, workers) -> Sink:
    """Sink for `--collect HOST:DIR`. Checkout for the remote --combine: --sink-repo, else the
    --worker dir if the sink is also a worker, else the inventory's `repo` (only consulted when
    --inventory was given, like every other inventory use here)."""
    import fleet_inventory
    inv = (fleet_inventory.load_inventory(
               None if inventory_arg == "__DEFAULT__" else Path(inventory_arg).expanduser())
           if inventory_arg is not None else {})
    same = next((w.dir for w in workers if w.host == host), None)
    return sink_from_inventory(host, path, inv, sink_repo or same)


def _ssh_run(host: str, cmd: str, timeout: float) -> "subprocess.CompletedProcess":
    try:
        return subprocess.run(ssh_target.ssh_argv(host, "-o", "BatchMode=yes") + [cmd],
                              capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess([], 255, "", f"{type(e).__name__}: {e}")


def transfer_shard_to_sink(worker_host: str, out: str, sink: Sink, sink_dir: str,
                           direct: bool = True, timeout: float = 6 * 3600.0) -> "tuple[bool, str]":
    """Move one worker output dir's contents (minus params.csv, which the caller merges and
    writes LAST) into `sink_dir` on the sink. Returns (ok, how)."""
    out = out.rstrip("/")
    excl = "--exclude=/params.csv"
    same_host = worker_host == sink.host
    if same_host and out == sink_dir:
        return True, "already in place"
    if same_host:
        r = _ssh_run(worker_host, f"rsync -a {excl} {_remote_quote(out)}/ {_remote_quote(sink_dir)}/",
                     timeout)
        if r.returncode == 0:
            return True, "local copy on the sink"
    elif direct and _RSYNC_SAFE.match(out) and _RSYNC_SAFE.match(sink_dir):
        ssh_e = "ssh -o BatchMode=yes -o ConnectTimeout=8" + (f" -p {sink.port}" if sink.port else "")
        r = _ssh_run(worker_host, f"rsync -a {excl} -e {shlex.quote(ssh_e)} {out}/ "
                                  f"{sink.target or sink.host}:{sink_dir}/", timeout)
        if r.returncode == 0:
            return True, "direct worker -> sink"
        log(f"  collect: {worker_host} could not reach the sink directly "
            f"({r.stderr.strip()[:100] or 'rc ' + str(r.returncode)}) -- relaying through this machine")
    try:
        src = subprocess.Popen(
            ssh_target.ssh_argv(worker_host, "-o", "BatchMode=yes")
            + [f"cd {_remote_quote(out)} && COPYFILE_DISABLE=1 tar --exclude=params.csv -cf - ."],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            dst = _ssh_run_stdin(sink.host, f"tar -C {_remote_quote(sink_dir)} -xf -", src.stdout,
                                 timeout)
        finally:
            src.stdout.close()
            src.wait()
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"
    if src.returncode != 0 or dst.returncode != 0:
        return False, (dst.stderr.strip() or src.stderr.read().decode(errors="replace").strip()
                       or f"tar exit {src.returncode}/{dst.returncode}")[:150]
    return True, "relayed through this machine"


def _ssh_run_stdin(host: str, cmd: str, stdin, timeout: float) -> "subprocess.CompletedProcess":
    return subprocess.run(ssh_target.ssh_argv(host, "-o", "BatchMode=yes") + [cmd], stdin=stdin,
                          capture_output=True, text=True, timeout=timeout)


def list_sink_npys(sink_host: str, sink_dir: str) -> "tuple[list[int], list[int]] | None":
    """(indices, byte sizes) of every sig/**/*.npy on the sink, or None if it can't be asked."""
    r = _ssh_run(sink_host, f"cd {_remote_quote(sink_dir)} && [ -d sig ] && "
                            f"find sig -name '*.npy' -exec wc -c {{}} + || true", 600)
    if r.returncode != 0:
        return None
    idx, sizes = [], []
    for ln in r.stdout.splitlines():
        parts = ln.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        stem = posixpath.basename(parts[1].strip())[:-4]
        if stem.isdigit():
            idx.append(int(stem))
            sizes.append(int(parts[0]))
    return sorted(idx), sizes


def combine_command(sink: Sink) -> "str | None":
    if not sink.repo:
        return None
    return (f"cd {_remote_quote(sink.repo)} && ./.venv/bin/python -u gen_dataset_from_schx.py "
            f"--combine {_remote_quote(sink.dir)}")


def _collect_to_sink(workers, remote_out, sink: Sink, no_combine=False, labels=None,
                     expected_count=None, direct=True) -> bool:
    """`_collect` + combine, with the merged dataset landing on `sink` instead of here. Same
    ordering guarantee (every params.csv is read BEFORE the merged one is written, and the
    merged one goes LAST), same consistency checks; --repair-missing is not supported because
    a repair renders locally."""
    import shutil
    import tempfile
    labels = list(labels) if labels is not None else [w.host for w in workers]
    sink_dir = resolve_remote_dir(sink.host, sink.dir).rstrip("/")
    mk = _ssh_run(sink.host, f"mkdir -p {_remote_quote(sink_dir)}", 30)
    if mk.returncode != 0:
        log(f"  collect: cannot create {sink.host}:{sink_dir}: {mk.stderr.strip()[:150]}")
        return False
    scratch = Path(tempfile.mkdtemp(prefix="collect_"))
    try:
        got = []
        for w, out, label in zip(workers, remote_out, labels):
            dst = scratch / f"{label}.csv"
            r = subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), f"{w.host}:{out}/params.csv",
                                str(dst)], capture_output=True, text=True)
            if r.returncode == 0 and dst.exists():
                got.append(dst)
            else:
                reason = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
                log(f"  collect: {label} has no params.csv -- skipped ({reason[0][:150]})")
        xfer_failed = False
        for w, out, label in zip(workers, remote_out, labels):
            ok, how = transfer_shard_to_sink(w.host, out, sink, sink_dir, direct=direct)
            log(f"  collect: {label} -> {sink.host}:{sink_dir}: {how if ok else 'FAILED: ' + how}")
            xfer_failed |= not ok
        n_rows, csv_idx = merge_params(got, scratch / "params.csv")
        if n_rows == 0:
            log("  collect: no params.csv found on any worker -- nothing merged")
            return False
        ok, detail = sync_path_to_worker(sink.host, sink_dir, scratch / "params.csv", "params.csv")
        if not ok:
            log(f"  collect: could not write merged params.csv on the sink: {detail}")
            return False
        listing = list_sink_npys(sink.host, sink_dir)
        if listing is None:
            log("  collect: could not list the sink's sig/ -- not combining")
            return False
        npy_idx, sizes = listing
        log(f"  collect: {len(csv_idx)} params rows, {len(npy_idx)} .npy files -> "
            f"{sink.host}:{sink_dir}")
        orphan_npy = sorted(set(npy_idx) - set(csv_idx))
        orphan_csv = sorted(set(csv_idx) - set(npy_idx))
        consistent = not orphan_npy and not orphan_csv and not xfer_failed
        if orphan_npy:
            log(f"  collect: WARNING {len(orphan_npy)} .npy file(s) with no params.csv row: {orphan_npy}")
        if orphan_csv:
            log(f"  collect: WARNING {len(orphan_csv)} params.csv row(s) with no .npy file: {orphan_csv}")
        if xfer_failed:
            log("  collect: WARNING at least one shard did not transfer -- see above.")
        if expected_count is not None and not _check_expected(csv_idx, npy_idx, sizes, expected_count):
            consistent = False
        why = should_combine(consistent, no_combine)
        if why is not None:
            log(f"  collect: NOT combining -- {why}")
            return consistent
        cmd = combine_command(sink)
        if cmd is None:
            log(f"  collect: NOT combining -- no checkout known for {sink.host} (pass --sink-repo); "
                f"run 'gen_dataset_from_schx.py --combine {sink_dir}' there when ready.")
            return consistent
        log(f"  combining on {sink.host} -> outputs.npy ...")
        r = _ssh_run(sink.host, cmd, 6 * 3600.0)
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-5:])
        if r.returncode != 0:
            log(f"  collect: combine FAILED on {sink.host} (rc {r.returncode}):\n{tail}")
            return False
        log(f"  combine on {sink.host} done:\n{tail}")
        return consistent
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


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
        subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), f"{w.host}:{out}/", str(local_dir) + "/"],
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


def _collect_check_transient_coverage(workers, remote_out, local_dir, config_path, extra_args):
    """Pull every worker's tcov_shard_*.json into one local directory, then run
    check_transient_coverage.py --merge-onsets on them -- same shape as
    _collect_grid_adequacy, different flag name (--merge-onsets, not --merge) and a distinct
    filename prefix (tcov_shard_, not shard_) so the two tools' shard files can never collide
    if a caller ever points both at the same --collect directory.

    Same no-clobber-hazard reasoning as _collect_grid_adequacy: each shard file's name embeds
    its own --shard spec, so every worker's output is a globally unique filename and a plain
    whole-tree rsync from each worker is safe in any order.
    """
    local_dir = Path(local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    for w, out in zip(workers, remote_out):
        subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), f"{w.host}:{out}/", str(local_dir) + "/"],
                       capture_output=True, text=True)
    shards = sorted(local_dir.glob("tcov_shard_*.json"))
    if not shards:
        log("  collect: no tcov_shard_*.json found on any worker -- nothing to merge")
        return
    log(f"  collect: merging {len(shards)} shard file(s) ...")
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "check_transient_coverage.py"),
           "--merge-onsets", *[str(s) for s in shards], "--config", str(config_path), *extra_args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        log(f"  {line}")
    if r.returncode != 0:
        log(f"  collect: merge exited {r.returncode}")
        if r.stderr.strip():
            log(f"  {r.stderr.strip()}")


def _collect_prepare_excitation(workers, remote_out, local_dir, config_path, extra_args):
    """Pull every worker's pexc_shard_*.json into one local directory, then run
    prepare_excitation.py --merge-onsets on them -- same shape as
    _collect_check_transient_coverage, distinct filename prefix (pexc_shard_, not shard_ or
    tcov_shard_) so no two of this module's four sharded tools can ever collide in one
    --collect directory.

    UNLIKE check_transient_coverage's merge (verify + report), this one ALSO SIZES AND BUILDS:
    prepare_excitation.py's own --merge-onsets branch computes the worst-case onset across the
    verified union and calls build_excitation.py from it (see that branch's own docstring
    comment, "MERGE MODE ... skip rendering entirely and size from the verified union") -- so
    the finished excitation.wav and the config.toml update are a side effect of this collect
    step, not a separate manual run afterward.

    --backend is derived from --config directly here (not via
    prepare_excitation_args_from_config, whose repo-relative path rewriting is for a WORKER's
    `cd <worker dir> && ...` dispatch -- this subprocess runs locally, same as every other
    collect function's plain `str(config_path)`), so the merge command cannot silently omit it
    even if the caller's own -- args forgot to repeat it.
    """
    local_dir = Path(local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    for w, out in zip(workers, remote_out):
        subprocess.run(["rsync", "-a", *ssh_target.rsync_e(), f"{w.host}:{out}/", str(local_dir) + "/"],
                       capture_output=True, text=True)
    shards = sorted(local_dir.glob("pexc_shard_*.json"))
    if not shards:
        log("  collect: no pexc_shard_*.json found on any worker -- nothing to merge")
        return
    log(f"  collect: merging {len(shards)} shard file(s) and sizing the excitation ...")
    cfg = load_config(config_path)
    backend_args = ["--backend", str(cfg["backend"])] if cfg.get("backend") else []
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "prepare_excitation.py"),
           "--merge-onsets", *[str(s) for s in shards], "--config", str(config_path),
           *backend_args, *extra_args]
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
    collect: "callable"        # (workers, remote_out, local_dir, config_path, extra_args, no_combine,
                               #  repair_missing, labels=None, expected_count=None) -> None
    workers_flag: str = "--workers"   # the CLI flag this script uses for its own internal
                                       # concurrency -- gen_dataset_from_schx.py/grid_adequacy.py/
                                       # measure_truncation.py all happen to spell it "--workers",
                                       # but prepare_excitation.py spells it "--corner-workers"
                                       # (find_saturation_point() is per-CORNER, not per-combination).
                                       # worker_loop appends f"{workers_flag} {count}" generically --
                                       # without this override every PREPARE_EXCITATION_JOB dispatch
                                       # fails argparse with "unrecognized arguments: --workers N"
                                       # (found 2026-10-02 scaffolding Ampeg SVT Full sag reactive,
                                       # the first real device to shard this tool).


def _collect_gen_dataset(workers, remote_out, local_dir, config_path, extra_args, no_combine,
                          repair_missing, labels=None, expected_count=None):
    """GEN_DATASET_JOB's own collect step: merge shards, then build outputs.npy by default.

    --collect used to stop right after merging, leaving a directory that LOOKS finished but
    that param_train.py refuses ("outputs.npy not found"). Combining needs EVERY shard
    present, so it cannot live in the renderer (a single --shard sees only one slice of the
    grid) -- --collect is by definition the moment all shards exist in one directory, and it
    already checks rows-vs-.npy, the exact precondition combine needs. Cost Mesa Orange and
    Duke of Tone (Overdrive) a manual step each on 2026-09-07; run_pipeline.py has had a
    Combine step all along, so only this distributed path was missing it.

    `labels`/`expected_count`: per-item-sharding-proposal.md Phase 3 -- see _collect's own
    docstring. Both default to None (legacy behavior, one entry per worker, no count check).
    """
    consistent = _collect(workers, remote_out, local_dir, config_path, extra_args, repair_missing,
                          labels=labels, expected_count=expected_count)
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
    collect=lambda workers, remote_out, local_dir, config_path, extra_args, no_combine, repair_missing,
                  labels=None, expected_count=None:
        _collect_gen_dataset(workers, remote_out, local_dir, config_path, extra_args, no_combine,
                              repair_missing, labels=labels, expected_count=expected_count),
)

GRID_ADEQUACY_JOB = Job(
    name="grid_adequacy",
    script="grid_adequacy.py",
    progress_re=GRIDADQ_PROBE_LINE,
    output_flag="--shard-out",
    build_args=lambda config_path, repo_root, extra_args:
        grid_adequacy_args_from_config(config_path, repo_root) + extra_args,
    chunk_output=lambda base_output, chunk: f"{base_output}/shard_{chunk.replace('/', '_')}.json",
    # no_combine/repair_missing are GEN_DATASET_JOB-specific (grid_adequacy has no "combine" or
    # per-index-repair concept at all) -- accepted and ignored here so all three jobs share one
    # call site in main().
    collect=lambda workers, remote_out, local_dir, config_path, extra_args, no_combine, repair_missing,
                  labels=None, expected_count=None:
        _collect_grid_adequacy(workers, remote_out, local_dir, config_path, extra_args),
)

MEASURE_TRUNCATION_JOB = Job(
    name="measure_truncation",
    script="measure_truncation.py",
    progress_re=MEASURE_TRUNC_LINE,
    output_flag="--emit",
    build_args=lambda config_path, repo_root, extra_args:
        measure_truncation_args_from_config(config_path, repo_root) + extra_args,
    chunk_output=lambda base_output, chunk: f"{base_output}/shard_{chunk.replace('/', '_')}.json",
    # no_combine/repair_missing are GEN_DATASET_JOB-specific, same as GRID_ADEQUACY_JOB --
    # accepted and ignored here so all three jobs share one call site in main().
    collect=lambda workers, remote_out, local_dir, config_path, extra_args, no_combine, repair_missing,
                  labels=None, expected_count=None:
        _collect_measure_truncation(workers, remote_out, local_dir, config_path, extra_args),
)

CHECK_TRANSIENT_COVERAGE_JOB = Job(
    name="check_transient_coverage",
    script="check_transient_coverage.py",
    progress_re=TCOV_CORNER_LINE,
    output_flag="--emit-onsets",
    build_args=lambda config_path, repo_root, extra_args:
        check_transient_coverage_args_from_config(config_path, repo_root) + extra_args,
    chunk_output=lambda base_output, chunk: f"{base_output}/tcov_shard_{chunk.replace('/', '_')}.json",
    # no_combine/repair_missing are GEN_DATASET_JOB-specific, same as GRID_ADEQUACY_JOB and
    # MEASURE_TRUNCATION_JOB -- accepted and ignored here so all four jobs share one call site
    # in main(). Closes the "transient-coverage isn't [sharded] yet" gap config-gate-
    # proposal.md and per-item-sharding-proposal.md both flag -- needed for gate_config.py's
    # fleet mode (docs/implementation-roadmap.md item 8) to shard this the same way it already
    # shards grid_adequacy.
    collect=lambda workers, remote_out, local_dir, config_path, extra_args, no_combine, repair_missing,
                  labels=None, expected_count=None:
        _collect_check_transient_coverage(workers, remote_out, local_dir, config_path, extra_args),
)

PREPARE_EXCITATION_JOB = Job(
    name="prepare_excitation",
    script="prepare_excitation.py",
    progress_re=TCOV_CORNER_LINE,   # same "onset=" per-corner line, see that constant's own
                                     # docstring on why it is already tool-agnostic.
    output_flag="--emit-onsets",
    build_args=lambda config_path, repo_root, extra_args:
        prepare_excitation_args_from_config(config_path, repo_root) + extra_args,
    chunk_output=lambda base_output, chunk: f"{base_output}/pexc_shard_{chunk.replace('/', '_')}.json",
    # no_combine/repair_missing are GEN_DATASET_JOB-specific, same as the other three jobs --
    # accepted and ignored here so all five jobs share one call site in main(). This is the
    # SIZING pass CHECK_TRANSIENT_COVERAGE_JOB's own addition left undone (that job only
    # covers the pre-generation VERIFICATION gate, re-checking an excitation already built) --
    # some devices' corner-onset sizing has itself been slow single-machine (e.g. AC30's own
    # 107-corner pass), and this is the same fix for that half: shard the corners, not just the
    # check of them, across the fleet instead of serially on the controller alone.
    collect=lambda workers, remote_out, local_dir, config_path, extra_args, no_combine, repair_missing,
                  labels=None, expected_count=None:
        _collect_prepare_excitation(workers, remote_out, local_dir, config_path, extra_args),
    workers_flag="--corner-workers",
)

JOBS = {j.name: j for j in (GEN_DATASET_JOB, GRID_ADEQUACY_JOB, MEASURE_TRUNCATION_JOB,
                            CHECK_TRANSIENT_COVERAGE_JOB, PREPARE_EXCITATION_JOB)}


def run_collect(job, workers, output, collect_dest, *, per_item, item_count, config, extra_args,
                no_combine, repair_missing, inventory, sink_repo, no_direct_sink,
                slot_pairs=None):
    """Pull every worker's shard into `collect_dest` ([HOST:]DIR) and merge/combine.

    Shared by distribute_pull's own main() and fleet_ctl.py's `collect`. `slot_pairs` (per-item
    only) is the explicit [(Worker, slot)] list whose <output>/slot-K holds finished work; the
    default is every slot of every worker, which is what a push run created up front. The pull
    fleet passes only the (worker, slot) pairs that actually completed a chunk, since its
    agents create slot dirs lazily.
    """
    log(f"collecting shards into {collect_dest} ...")
    sink_host, sink_path = parse_collect_dest(collect_dest)
    sink = None
    if sink_host:
        sink = build_sink(sink_host, sink_path, inventory, sink_repo, workers)
    resolved = {}
    for w in workers:
        if w.host in resolved:
            continue
        # resolve the output path ON THE WORKER: --output is commonly '~/dir', and a tilde
        # inside an rsync host:path argument is NOT expanded (that silently transferred
        # nothing on an earlier run), so ask the remote shell what it means.
        r = subprocess.run(ssh_target.ssh_argv(w.host, "-o", "BatchMode=yes")
                           + [f"cd ~ && echo {output}"], capture_output=True, text=True)
        resolved[w.host] = r.stdout.strip() or output
    if per_item:
        # host x slot (Phase 3), not one entry per host: each slot rendered into its own
        # <output>/slot-K, and needs its own scratch-csv LABEL -- collect_workers repeats
        # the SAME Worker object per slot (only .host is read below it), which is why the
        # label can't just be w.host again, or slot 1's params.csv would silently clobber
        # slot 0's before either was merged. See _collect's own docstring.
        if slot_pairs is None:
            slot_pairs = [(w, k) for w in workers for k in range(w.slots)]
        collect_workers, collect_remote_out, collect_labels = [], [], []
        for w, k in slot_pairs:
            collect_workers.append(w)
            collect_remote_out.append(f"{resolved[w.host]}/slot-{k}")
            collect_labels.append(f"{w.host}-slot{k}")
        if sink:
            _collect_to_sink(collect_workers, collect_remote_out, sink, no_combine,
                             labels=collect_labels, expected_count=item_count,
                             direct=not no_direct_sink)
        else:
            job.collect(collect_workers, collect_remote_out, collect_dest, config, extra_args,
                        no_combine, repair_missing, labels=collect_labels,
                        expected_count=item_count)
    else:
        remote_out = [resolved[w.host] for w in workers]
        if sink:
            _collect_to_sink(workers, remote_out, sink, no_combine, direct=not no_direct_sink)
        else:
            job.collect(workers, remote_out, collect_dest, config, extra_args, no_combine,
                        repair_missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="append", required=True,
                    metavar="HOST:DIR[:PARALLEL[:ENV]]",
                    help="repeatable. PARALLEL is optional (2026-09-26) -- omit it, or leave "
                         "it empty/'auto' (HOST:DIR::ENV to reach ENV without one), to "
                         "auto-detect the worker's PHYSICAL core count over SSH (see "
                         "cpu_topology.py's docstring for why physical, not logical/SMT, is "
                         "the right default for this CPU-bound render workload). ENV is an "
                         "optional 'VAR=value' exported before the run (e.g. "
                         "DOTNET_ROOT=$HOME/.dotnet on a box where dotnet is not on the "
                         "non-interactive PATH).")
    ap.add_argument("--tool", choices=sorted(JOBS), default="gen_dataset",
                    help="which script each chunk runs (default gen_dataset, i.e. "
                         "gen_dataset_from_schx.py -- unchanged behavior for existing callers). "
                         "'grid_adequacy' dispatches grid_adequacy.py --shard instead, and "
                         "--collect runs its --merge step so the final report is identical to "
                         "an unsharded grid_adequacy.py run. 'measure_truncation' dispatches "
                         "measure_truncation.py --shard (scaffold_config.py's oversample-"
                         "measurement step) the same way -- --chunks should usually be much "
                         "smaller than the default 64 here: the grid being cut is knob "
                         "SETTINGS (probe_settings(knobs), ~2x knob count), not combinations, "
                         "so 64 chunks over ~15 settings hands most workers nothing. "
                         "'check_transient_coverage' dispatches check_transient_coverage.py "
                         "--shard --emit-onsets (the pre-generation saturation-coverage gate) "
                         "the same way as grid_adequacy -- the grid being cut is CORNERS "
                         "(the reduced hypercube set, see that tool's docstring), not "
                         "combinations either. 'prepare_excitation' shards THAT tool's own "
                         "worst-case-onset SIZING pass the same way (not just the later "
                         "verification gate) -- --collect's merge also runs build_excitation.py "
                         "from the verified union, so the finished excitation.wav and the "
                         "config.toml update are produced by --collect itself, not a separate "
                         "manual step. Needs --sweep-file (and usually --output) after --, "
                         "same as a single-machine prepare_excitation.py invocation would.")
    ap.add_argument("--chunks", type=int, default=64,
                    help="how many pieces to cut the grid into (default 64). Each is dispatched "
                         "as --shard i-i/CHUNKS. See module docstring on sizing. Ignored when "
                         "--chunk-size is given -- see per-item-sharding-proposal.md.")
    ap.add_argument("--chunk-size", type=int, default=None, metavar="N",
                    help="per-item-sharding-proposal.md Phase 1: dispatch ONE combination at a "
                         "time per slot instead of one multi-combination chunk per worker. Only "
                         "1 is implemented today (any other value is a hard error). Requires "
                         "--slots (or lets it default to each worker's own PARALLEL/core count) "
                         "and derives the real item count via --items or --range (see "
                         "derive_item_count). Default (omitted): the original --chunks path, "
                         "unchanged.")
    ap.add_argument("--slots", type=int, default=None, metavar="K",
                    help="--chunk-size 1 only: concurrent per-item dispatches PER WORKER, each "
                         "into its own <output>/slot-K subdirectory (so K concurrent generations "
                         "never collide on gen_dataset_from_schx's exclusive per-directory lock). "
                         "Default: each worker's own PARALLEL/auto-detected core count -- same "
                         "total concurrency as the legacy --workers-N-internal-to-one-chunk "
                         "model, just restructured into K independent single-item dispatches.")
    ap.add_argument("--items", type=int, default=None, metavar="N",
                    help="--chunk-size 1 only: override the derived combination count instead "
                         "of reading it from --range (needed for a job whose grid isn't "
                         "expressed that way). Getting this WRONG is dangerous in opposite "
                         "directions: too high just wastes ~80ms per empty dispatch, too low "
                         "SILENTLY DROPS combinations -- see derive_item_count.")
    ap.add_argument("--output", required=True, help="output dir ON EACH WORKER")
    ap.add_argument("--retries", type=int, default=1,
                    help="re-queue a failed chunk this many times, on a DIFFERENT worker where "
                         "possible -- a chunk that fails on one machine and succeeds on another "
                         "is a machine problem, not a data problem, and the resume-skip means "
                         "the retry only renders what is still missing (default 1)")
    ap.add_argument("--collect", metavar="[HOST:]DIR", default=None,
                    help="after rendering, pull every worker's shard into DIR (a local directory, "
                         "or HOST:DIR to collect ON the machine that will train -- see "
                         "docs/scripts.md) and merge "
                         "params.csv properly. Do NOT hand-roll this with rsync: sig/ merges "
                         "cleanly (global-index filenames) but params.csv is one file per worker "
                         "holding only that worker's rows, so a naive rsync leaves you the LAST "
                         "worker's metadata describing the whole grid.")
    ap.add_argument("--sink-repo", metavar="DIR", default=None,
                    help="--collect HOST:DIR: the sink's parametric-nam checkout, used to run "
                         "--combine there. Default: that host's --worker dir, else its "
                         "inventory `repo` (needs --inventory).")
    ap.add_argument("--no-direct-sink", action="store_true",
                    help="--collect HOST:DIR: skip the worker->sink rsync attempt and always "
                         "relay through this machine.")
    ap.add_argument("--no-combine", action="store_true",
                    help="--collect: stop after merging, without building outputs.npy. Combine "
                         "NORMALISES the data and records output_scale into config.json, and for "
                         "a big grid it is a multi-GB write, so this exists for inspecting or "
                         "re-combining with a different --output-peak/--raw.")
    ap.add_argument("--repair-missing", action="store_true",
                    help="--collect (--tool gen_dataset only, needs --config): if params.csv "
                         "rows and .npy files don't match 1:1, regenerate exactly the missing "
                         "combinations locally (one at a time, reusing the same "
                         "gen_args_from_config expansion the original dispatch used) before "
                         "deciding whether to combine. Without this flag, a mismatch is still "
                         "detected and every affected index is logged by number -- this only "
                         "controls whether the tool then fixes it automatically or leaves that "
                         "to you.")
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
    ap.add_argument("--skip-gate-check", action="store_true",
                    help="don't check for a gate_config.py sidecar at all (--tool gen_dataset "
                         "with --config only). Default is WARN and continue when it's missing "
                         "or stale -- see docs/config-gate-proposal.md, "
                         "docs/implementation-roadmap.md item 3.")
    ap.add_argument("--require-gate", action="store_true",
                    help="ABORT instead of warning when gate_config.py's sidecar is missing or "
                         "stale (--tool gen_dataset with --config only). Opt-in for now "
                         "(2026-09-28); expected to become the default later. Run `gate_config.py "
                         "--config <config>` on the CONTROLLER before dispatching, not per "
                         "worker -- it needs the sized excitation, which workers may not have.")
    ap.add_argument("--skip-version-check", action="store_true",
                    help="don't verify each worker's commit SHA / solver revision against this "
                         "controller before dispatching (docs/implementation-roadmap.md item "
                         "5). Default is to check once per worker and EXCLUDE a mismatched one "
                         "from this run (not abort the whole run) -- see verify_workers(). Pass "
                         "this only when a mismatch is a known false positive; it is not a "
                         "warn-only default the way --skip-gate-check is.")
    ap.add_argument("--sync-file", action="append", default=[], metavar="PATH",
                    help="repeatable. Push this file (or directory's contents) to every worker "
                         "at the SAME path relative to the repo before dispatching -- for "
                         "inputs git doesn't carry: a gitignored excitation wav, a .schx or "
                         "--pedal-dir module outside the checkout. A worker that can't receive "
                         "one is excluded from the run. Replaces distribute_gen.sh's flag of "
                         "the same name. Provisioning (clone, venv, oracle build) is still a "
                         "per-machine setup step this does not do.")
    ap.add_argument("--inventory", nargs="?", const="__DEFAULT__", default=None,
                    help="reach each --worker HOST the way the fleet inventory says: its "
                         "address/user/port/identity_file become an ssh config (ssh_target.py) "
                         "used for EVERY ssh/rsync call, winning over ~/.ssh/config for the "
                         "fields the inventory sets. Bare --inventory = "
                         "~/.config/parametric-nam/fleet.toml. Absent = plain `ssh HOST` "
                         "(your own ~/.ssh/config), as before. HOST must be the inventory's "
                         "host name.")
    ap.add_argument("--", dest="_sep", nargs="?", help=argparse.SUPPRESS)
    args, gen_args = ap.parse_known_args()
    configure_ssh(args.inventory)
    sink_host, sink_path = parse_collect_dest(args.collect) if args.collect else (None, None)
    if sink_host:
        if args.tool != "gen_dataset":
            ap.error("--collect HOST:DIR is only supported for --tool gen_dataset")
        if args.repair_missing:
            ap.error("--repair-missing renders locally, so it can't be combined with --collect HOST:DIR")
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

        # Gate check: has gate_config.py verified this circuit/grid/excitation recently?
        # WARN-ONLY for now (2026-09-28), same rationale and flags as run_pipeline.py -- see
        # docs/implementation-roadmap.md item 3. Only meaningful for gen_dataset (what the gate
        # protects); grid_adequacy/measure_truncation are themselves pre-generation checks.
        # Checked on the CONTROLLER against args.config as given -- one check per dispatch, not
        # per worker/chunk, and it says nothing about whether a worker's own checkout has the
        # sized excitation synced (see fleet-deployment-proposal.md's open item on that).
        if args.tool == "gen_dataset" and not args.skip_gate_check:
            from gate_config import verify_gate
            from run_pipeline import gate_check_outcome
            try:
                ok, gate_msg = verify_gate(args.config)
            except Exception as e:
                ok, gate_msg = False, f"gate check raised {type(e).__name__}: {e}"
            line, abort = gate_check_outcome(args.config, ok, gate_msg, args.require_gate)
            log(line)
            if abort:
                sys.exit(2)
    gen_args_str = " ".join(f"'{a}'" if " " in a else a for a in gen_args)
    if not gen_args_str:
        ap.error("pass --config, or the renderer's own arguments after --")

    missing = [p for p in args.sync_file if not Path(p).expanduser().exists()]
    if missing:
        ap.error(f"--sync-file not found locally: {', '.join(missing)}")
    workers = [Worker(w, job=job) for w in args.worker]
    if not args.skip_version_check:
        workers = verify_workers(workers, extract_backend(gen_args))
        if not workers:
            ap.error("every worker failed the dispatch-time version check -- nothing to "
                     "dispatch to (see the WARNINGs above). Pass --skip-version-check to "
                     "bypass, or fix the checkout(s).")
    if args.sync_file:
        workers = sync_files(workers, args.sync_file)
        if not workers:
            ap.error("no worker could receive every --sync-file -- nothing to dispatch to "
                     "(see the WARNINGs above).")
    # Per-item dispatch (per-item-sharding-proposal.md, docs/implementation-roadmap.md item 6):
    # --chunk-size 1 replaces the multi-combination --chunks path with K single-item dispatches
    # per worker ("slots"), each into its own <output>/slot-K so K concurrent generations never
    # collide on gen_dataset_from_schx's exclusive per-directory lock. Everything below this
    # block -- the queue, attempts/tried_on, quarantine, retry -- is UNCHANGED either way: a
    # per-item "chunk" is just a shard spec with TOTAL == the real combination count, so it
    # reuses the exact same `i-i/N` machinery legacy chunking already has.
    per_item = args.chunk_size is not None
    item_count = None
    if per_item:
        if args.chunk_size != 1:
            ap.error(f"--chunk-size {args.chunk_size}: only 1 (per-item dispatch) is "
                     "implemented today -- see per-item-sharding-proposal.md")
        try:
            item_count = derive_item_count(gen_args, args.items)
        except ValueError as e:
            ap.error(str(e))
        for w in workers:
            w.slots = args.slots if args.slots else w.parallel
        n_specs = item_count
    else:
        n_specs = args.chunks

    queue = deque(f"{i}-{i}/{n_specs}" for i in range(n_specs))
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

    if per_item:
        log(f"{total} combination(s) over {len(workers)} worker(s), "
            f"{sum(w.slots for w in workers)} slot(s) total: "
            + ", ".join(f"{w.host}(slots={w.slots})" for w in workers))
    else:
        _warn_chunk_aliasing(gen_args, args.chunks)
        log(f"{total} chunks over {len(workers)} worker(s): "
            + ", ".join(f"{w.host}(par={w.parallel})" for w in workers))

    def worker_loop(w, output_dir, workers_flag):
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
            rc, dt, out = w.run_chunk(chunk, f"{gen_args_str} {job.workers_flag} {workers_flag}",
                                      output_dir, pace=pace)
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

    if per_item:
        # Create every slot dir on its worker BEFORE dispatching to it. Found for real, not
        # hypothetically: gen_dataset_from_schx.py's own disk-space check
        # (shutil.disk_usage(args.output.parent if not args.output.exists() else args.output))
        # only defends against the LEAF being missing -- it falls back one level, to the
        # leaf's parent, and assumes THAT exists. Per-item mode nests a second level
        # (<output>/slot-K) that legacy mode never had, so a brand-new --output path (the
        # common case for a device's first-ever render) leaves BOTH the leaf and its parent
        # missing, and shutil.disk_usage() raises FileNotFoundError outright. mkdir -p sidesteps
        # the gap rather than patching the renderer's one-level fallback to be two-level.
        for w in workers:
            for k in range(w.slots):
                subprocess.run(ssh_target.ssh_argv(w.host, "-o", "BatchMode=yes")
                               + [f"mkdir -p {args.output}/slot-{k}"],
                               capture_output=True, text=True)
        threads = [threading.Thread(target=worker_loop,
                                    args=(w, f"{args.output}/slot-{k}", 1), daemon=True)
                  for w in workers for k in range(w.slots)]
    else:
        # Same gap as per-item mode's own mkdir above, just one level shallower: a brand-new
        # --output path with no PARENT either (e.g. a namespace no prior run ever used --
        # found for real via run_pipeline.py's fleet mode dispatching into its own fresh
        # PIPELINE_FLEET_WORK_ROOT) leaves shutil.disk_usage()'s one-level fallback
        # (args.output.parent) ALSO missing, and it raises FileNotFoundError outright before
        # any render starts. mkdir -p per worker sidesteps it the same way.
        for w in workers:
            subprocess.run(ssh_target.ssh_argv(w.host, "-o", "BatchMode=yes")
                           + [f"mkdir -p {args.output}"], capture_output=True, text=True)
        threads = [threading.Thread(target=worker_loop, args=(w, args.output, w.parallel),
                                    daemon=True) for w in workers]
    for t in threads: t.start()
    for t in threads: t.join()

    elapsed = (time.time() - t_start) / 3600
    log(f"done in {elapsed:.2f} h -- {completed} chunk(s) ok, {failed_final} failed")

    if args.collect:
        run_collect(job, workers, args.output, args.collect, per_item=per_item,
                    item_count=item_count, config=args.config, extra_args=extra_args,
                    no_combine=args.no_combine, repair_missing=args.repair_missing,
                    inventory=args.inventory, sink_repo=args.sink_repo,
                    no_direct_sink=args.no_direct_sink)
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
