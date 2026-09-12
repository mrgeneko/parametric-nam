#!/usr/bin/env python3
"""start / status / stop for ONE dataset-generation shard, addressed by its output directory.

WHY THIS EXISTS. A sharded generation used to be addressed by a PID, tracked by hand, per
machine, over ssh. Every incident we have had with sharding traces to that:

  `nohup cmd &` over ssh leaves a bash wrapper holding the python child. Killing the wrapper
  re-parents the child to init (ppid=1) and it keeps rendering -- with whatever code it
  started with. On 2026-09-12 an orphan like that spent 12 minutes writing old-metric results
  into the same log a replacement run had truncated, producing interleaved output that read
  like a cache bug and cost several rounds of misdiagnosis.

  Clearing a shard for a restart with `rm -rf out_dir` used to unlink the live generation
  lock (now fixed in acquire_generation_lock, which is the other half of this).

The fix is that you never name a process. You name the output directory, and the tool keeps
the mapping. Launch is always in its OWN PROCESS GROUP (start_new_session=True, i.e. setsid),
so stopping signals the group -- `kill(-pgid)` -- and children die with the parent instead of
outliving it. That is the single behavioural difference between the shard that had to be
killed three times that day and the one that stopped cleanly on the first try.

  ./shard_ctl.py start  --output ~/runs/dev/shard1 -- --backend livespice --schx ... --shard 1-1/3
  ./shard_ctl.py status --output ~/runs/dev/shard1
  ./shard_ctl.py stop   --output ~/runs/dev/shard1

NOT a scheduler. distribute_pull.py already dispatches chunks across machines and rebalances;
this is the per-machine primitive underneath it, and is equally usable by hand on a worker
that cannot be reached over ssh.
"""
import argparse, json, os, signal, socket, subprocess, sys, time
from datetime import datetime
from pathlib import Path

RUNFILE = ".run.json"
HERE = Path(__file__).resolve().parent


def _runfile(out_dir: Path) -> Path:
    return out_dir / RUNFILE


def _cmdline(pid: int) -> str:
    """Best-effort process cmdline, '' if it is gone. Used to confirm identity before signalling."""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=10)
        else:
            return Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf8", "replace").replace("\0", " ")
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _alive(pid: int) -> bool:
    """True only if the pid is a process that can still DO something.

    A ZOMBIE IS NOT ALIVE. os.kill(pid, 0) succeeds against a zombie -- the pid is still in the
    table, awaiting a wait() from its parent -- so a naive check reports a finished renderer as
    running. `stop` then waits out its whole grace period and escalates to SIGKILL against a
    group whose only remaining member is that zombie, which macOS answers with EPERM: a stop
    that did its job reports failure. Only shows up when the caller is the process's parent
    (a test harness, or a launcher that waits); a detached run is reaped by init and never
    lingers as one. Found while testing shard_ctl itself, 2026-09-12."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return "Z" not in _proc_state(pid)


def _proc_state(pid: int) -> str:
    """Process state flags ('Z' for zombie), '' if unknown. Best effort, never raises."""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                               capture_output=True, text=True, timeout=10)
            return r.stdout.strip()
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def _is_ours(pid: int, marker: str) -> bool:
    """PID REUSE GUARD. A recorded pgid can be recycled by an unrelated process, and signalling
    a whole group on a stale number is how a stop command turns into collateral damage. Confirm
    the target still looks like the job we started before sending anything."""
    return _alive(pid) and marker in _cmdline(pid)


def cmd_start(args, passthrough):
    out_dir = Path(args.output).expanduser().resolve()
    rf = _runfile(out_dir)
    if rf.exists():
        try:
            prev = json.loads(rf.read_text())
        except (OSError, ValueError):
            prev = {}
        pid = prev.get("pid")
        if pid and _is_ours(pid, prev.get("marker", "gen_dataset_from_schx")):
            sys.exit(f"ERROR: a generation is already running for {out_dir}\n"
                     f"       pid={pid} pgid={prev.get('pgid')} started={prev.get('started')}\n"
                     f"       Stop it with: {sys.argv[0]} stop --output {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # LOG PATHS ARE NEVER REUSED. Two runs sharing one log is what produced the interleaved,
    # truncated-then-overwritten output that made an orphan look like a caching bug. Logs live
    # BESIDE out_dir, not inside it, so clearing a shard does not destroy the evidence of why.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = out_dir.parent / f"{out_dir.name}-{socket.gethostname().split('.')[0]}-{stamp}.log"

    renderer = str(HERE / "gen_dataset_from_schx.py")
    # -u (and PYTHONUNBUFFERED for any child) because stdout to a FILE is block-buffered, not
    # line-buffered. Without it a shard can be 20 of 24 combinations in with a one-line log:
    # the coverage gate's verdict and every progress line sit in an 8 KB buffer, invisible.
    # Observed on all three workers of the Mesa Orange run, 2026-09-12, and it actively misled
    # the diagnosis -- an empty log was read as "the gate passed from cache" when the gate had
    # simply not flushed. A launcher whose job is making a detached run observable cannot ship
    # with its output withheld until the process exits. FAILED lines go to stderr and appear
    # promptly either way, which is why failures still showed up while progress did not.
    cmd = [sys.executable, "-u", renderer, "--output", str(out_dir), *passthrough]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log, "wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True,   # own process group: kill(-pgid) works
                                cwd=str(HERE), env=env)
    rf.write_text(json.dumps({
        "pid": proc.pid, "pgid": os.getpgid(proc.pid), "host": socket.gethostname(),
        "started": datetime.now().isoformat(timespec="seconds"), "log": str(log),
        "marker": "gen_dataset_from_schx", "argv": cmd,
    }, indent=1))
    print(f"started pid={proc.pid} pgid={os.getpgid(proc.pid)}\n  output {out_dir}\n  log    {log}")
    return 0


def cmd_status(args, _passthrough):
    out_dir = Path(args.output).expanduser().resolve()
    rf = _runfile(out_dir)
    if not rf.exists():
        print(f"no run recorded for {out_dir}")
        return 1
    info = json.loads(rf.read_text())
    pid = info.get("pid")
    running = _is_ours(pid, info.get("marker", "gen_dataset_from_schx"))
    n = len(list((out_dir / "sig").glob("*.npy"))) if (out_dir / "sig").is_dir() else 0
    print(f"{'RUNNING' if running else 'not running'}  pid={pid} pgid={info.get('pgid')} "
          f"host={info.get('host')} started={info.get('started')}")
    print(f"  output   {out_dir}  ({n} combination(s) rendered)")
    print(f"  log      {info.get('log')}")
    if not running and _alive(pid):
        print("  NOTE: that pid exists but is NOT this job -- the number was recycled; not ours to signal.")
    return 0 if running else 2


def cmd_stop(args, _passthrough):
    out_dir = Path(args.output).expanduser().resolve()
    rf = _runfile(out_dir)
    if not rf.exists():
        print(f"no run recorded for {out_dir} -- nothing to stop")
        return 1
    info = json.loads(rf.read_text())
    pid, pgid = info.get("pid"), info.get("pgid")
    marker = info.get("marker", "gen_dataset_from_schx")
    if not _is_ours(pid, marker):
        print(f"not running (pid {pid} is gone or recycled) -- nothing to stop")
        return 0

    # Signal the GROUP, not the pid: that is what takes the renderer's backend children with
    # it instead of leaving them orphaned and still writing.
    for sig, label, grace in ((signal.SIGTERM, "TERM", args.grace), (signal.SIGKILL, "KILL", 10)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            # The group holds only remnants we cannot signal (typically a zombie awaiting its
            # parent's wait()). Nothing is still rendering, which is what the caller asked for.
            break
        print(f"sent SIG{label} to process group {pgid}; waiting up to {grace}s")
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not _alive(pid):
                break
            time.sleep(1)
        if not _alive(pid):
            break

    time.sleep(1)
    survivors = []
    try:
        os.killpg(pgid, 0)
        survivors.append(pgid)
    except (ProcessLookupError, PermissionError):
        pass
    if survivors:
        print(f"WARNING: process group {pgid} still present after SIGKILL -- inspect by hand")
        return 3
    print(f"stopped cleanly; no survivors in group {pgid}")
    print(f"  the output dir is intact at {out_dir}")
    print(f"  a re-run resume-skips whatever finished, so you usually do NOT need to delete it")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["start", "status", "stop"])
    ap.add_argument("--output", required=True, help="the shard's output directory -- its identity")
    ap.add_argument("--grace", type=int, default=30,
                    help="seconds to wait after SIGTERM before SIGKILL (default 30)")
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    return {"start": cmd_start, "status": cmd_status, "stop": cmd_stop}[args.action](args, passthrough)


if __name__ == "__main__":
    sys.exit(main())
