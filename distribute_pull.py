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
    lock = threading.Lock()
    total = len(queue)
    completed = failed_final = 0
    t_start = time.time()

    log(f"{total} chunks over {len(workers)} worker(s): "
        + ", ".join(f"{w.host}(par={w.parallel})" for w in workers))

    def worker_loop(w):
        nonlocal completed, failed_final
        while True:
            with lock:
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
                    w.done += 1; completed += 1
                    log(f"  {w.host:<10} chunk {chunk:<10} OK   {dt/60:5.1f} min   "
                        f"[{completed + failed_final}/{total}]")
                else:
                    w.failed += 1
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
