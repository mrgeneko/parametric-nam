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
import argparse, subprocess, sys, threading, time
from collections import deque
from datetime import datetime


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Worker:
    def __init__(self, spec):
        # host:remote_dir:parallel[:env]
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(f"--worker needs host:dir:parallel[:env], got {spec!r}")
        self.host, self.dir, self.parallel = parts[0], parts[1], int(parts[2])
        self.env = parts[3] if len(parts) > 3 else ""
        self.done = 0          # chunks completed
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

    def run_chunk(self, chunk, gen_args, output):
        env = f"export {self.env} && " if self.env else ""
        cmd = (f"cd {self.dir} && {env}./.venv/bin/python -u gen_dataset_from_schx.py "
               f"{gen_args} --shard {chunk} --output {output}")
        t0 = time.time()
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=60",
                            self.host, cmd], capture_output=True, text=True)
        dt = time.time() - t0
        self.secs += dt
        return r.returncode, dt, (r.stdout + r.stderr)



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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="append", required=True, metavar="HOST:DIR:PARALLEL[:ENV]",
                    help="repeatable. ENV is an optional 'VAR=value' exported before the run "
                         "(e.g. DOTNET_ROOT=$HOME/.dotnet on a box where dotnet is not on the "
                         "non-interactive PATH).")
    ap.add_argument("--chunks", type=int, default=64,
                    help="how many pieces to cut the grid into (default 64). Each is dispatched "
                         "as --shard i-i/CHUNKS. See module docstring on sizing.")
    ap.add_argument("--output", required=True, help="output dir ON EACH WORKER")
    ap.add_argument("--retries", type=int, default=1,
                    help="re-queue a failed chunk this many times, on a DIFFERENT worker where "
                         "possible -- a chunk that fails on one machine and succeeds on another "
                         "is a machine problem, not a data problem, and the resume-skip means "
                         "the retry only renders what is still missing (default 1)")
    ap.add_argument("--quarantine-after", type=int, default=3,
                    help="bench a worker after this many CONSECUTIVE failures with no successes "
                         "(default 3). A fast-failing worker drains the queue faster than healthy "
                         "ones can take work -- see Worker's quarantine comment for the run where "
                         "one killed 27 of 31 chunks in ~70s. 0 disables.")
    ap.add_argument("--", dest="_sep", nargs="?", help=argparse.SUPPRESS)
    args, gen_args = ap.parse_known_args()
    if gen_args and gen_args[0] == "--":
        gen_args = gen_args[1:]
    gen_args_str = " ".join(f"'{a}'" if " " in a else a for a in gen_args)
    if not gen_args_str:
        ap.error("pass the renderer's own arguments after --")

    workers = [Worker(w) for w in args.worker]
    queue = deque(f"{i}-{i}/{args.chunks}" for i in range(args.chunks))
    attempts = {c: 0 for c in queue}
    tried_on = {c: set() for c in queue}   # chunk -> hosts that have already failed it
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
            rc, dt, out = w.run_chunk(chunk, f"{gen_args_str} --workers {w.parallel}", args.output)
            w.busy = False
            with lock:
                if rc == 0:
                    w.consec_fail = 0
                    w.done += 1; completed += 1
                    log(f"  {w.host:<10} chunk {chunk:<10} OK   {dt/60:5.1f} min   "
                        f"[{completed + failed_final}/{total}]")
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
                    if n_try <= args.retries:
                        queue.append(chunk)   # back of the queue: likely a different worker
                        log(f"  {w.host:<10} chunk {chunk:<10} FAIL rc={rc} -- requeued "
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
    log("MEASURED throughput (use these for any future static weighting):")
    for w in sorted(workers, key=lambda x: -x.rate):
        log(f"  {w.host:<12} {w.done:3d} chunks  {w.rate:6.2f} chunks/h"
            + (f"  ({w.failed} failure(s))" if w.failed else ""))
    return 1 if failed_final else 0


if __name__ == "__main__":
    sys.exit(main())
