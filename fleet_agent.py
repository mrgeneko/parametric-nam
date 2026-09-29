#!/usr/bin/env python3
"""Pull agent: runs ON a worker, leases chunks from the coordinator and renders them locally
(docs/implementation-roadmap.md item 11; docs/fleet-deployment-proposal.md section 4).

    ./.venv/bin/python fleet_agent.py --coordinator http://HOST:8765 --token-file F --name HOST

Compared with distribute_pull.py's ssh push model, the renderer is a LOCAL child here, so:

  * there is no ssh connection whose drop orphans a renderer that still holds the exclusive
    .generation.lock -- the agent owns the process group and kills it deliberately;
  * the coordinator can restart, or the network can blip, without losing the chunk in flight:
    heartbeats and the final report are retried, and a renderer never depends on either;
  * an agent that dies mid-chunk simply stops heartbeating; the lease expires and the chunk is
    re-offered (renderers resume-skip finished items and write outputs atomically).

The agent runs from the repo checkout it lives in (cwd = repo root, the way distribute_pull.py's
`cd DIR && ./.venv/bin/python ...` did) and uses its own interpreter for the renderer.
--name must be the host name the OPERATOR uses to ssh here (inventory name): --collect still
pulls shards over ssh under that name.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from fleet_client import CoordinatorDown, CoordinatorError, FleetClient

REPO = Path(__file__).resolve().parent
TAIL_LINES = 40


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def chunk_paths(job: dict, slot: int, chunk: str, tool) -> "tuple[str, str]":
    """(base output dir, the path passed after the tool's output flag) for this slot/chunk.
    Per-item jobs render into <output>/slot-K so concurrent slots never share the renderer's
    exclusive per-directory lock."""
    out = os.path.expanduser(job["output"])
    base = f"{out}/slot-{slot}" if job.get("per_item") else out
    return base, tool.chunk_output(base, chunk)


def renderer_argv(python: str, job: dict, tool, chunk: str, chunk_output: str, parallel: int):
    workers = 1 if job.get("per_item") else parallel
    return ([python, "-u", tool.script] + list(job["gen_args"]) + ["--workers", str(workers),
            "--shard", chunk, tool.output_flag, chunk_output])


def stale_pattern(tool, chunk_output: str) -> str:
    return f"{re.escape(tool.script)}.*{re.escape(tool.output_flag)} {re.escape(chunk_output)}"


def kill_group(proc, grace: float = 5.0):
    """SIGTERM the renderer's whole process group, then SIGKILL after `grace`. flock is released
    when the holder dies, so this is all the lock cleanup there is (see distribute_pull's
    _kill_remote for why the lock FILE must never be deleted)."""
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def kill_stale(tool, chunk_output: str):
    """A renderer left by a PREVIOUS agent incarnation (agent killed -9) still holds this
    directory's flock. Anything matching this exact script + output path is stale by
    construction: this slot is the only thing that renders into it."""
    pat = stale_pattern(tool, chunk_output)
    try:
        subprocess.run(["pkill", "-f", pat], capture_output=True, timeout=15)
        time.sleep(0.2)
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass


class Agent:
    def __init__(self, client: FleetClient, name: str, slots: int, parallel: int,
                 repo_dir: Path = REPO, env: "dict | None" = None, python: str = sys.executable,
                 poll_s: float = 5.0, heartbeat_s: float = 20.0, exit_when_idle: bool = False,
                 report_retry_s: float = 900.0, jobs: "dict | None" = None, check_version=True):
        self.client, self.name, self.slots, self.parallel = client, name, slots, parallel
        self.repo_dir, self.env, self.python = Path(repo_dir), env or {}, python
        self.poll_s, self.heartbeat_s = poll_s, heartbeat_s
        self.exit_when_idle, self.report_retry_s = exit_when_idle, report_retry_s
        self.check_version = check_version
        if jobs is None:
            from distribute_pull import JOBS
            jobs = JOBS
        self.jobs = jobs
        self.stop = threading.Event()
        self._verified: "dict[int, tuple[bool, str]]" = {}
        self._vlock = threading.Lock()
        self.ran = 0

    # ---- coordinator calls that must survive an outage ---------------------------------
    def _retry(self, fn, what, give_up_s: "float | None" = None, honor_stop: bool = True):
        delay, t0 = 1.0, time.time()
        while not (honor_stop and self.stop.is_set()):
            try:
                return fn()
            except CoordinatorDown as e:
                if give_up_s is not None and time.time() - t0 > give_up_s:
                    log(f"{what}: coordinator unreachable for {give_up_s:.0f}s, giving up ({e})")
                    raise
                log(f"{what}: coordinator unreachable ({e}); retrying in {delay:.0f}s")
                time.sleep(delay) if not honor_stop else self.stop.wait(delay)
                delay = min(delay * 2, 30.0)
        raise CoordinatorDown("agent stopping")

    def register(self):
        from distribute_pull import local_commit_sha
        info = {"sha": local_commit_sha(), "host": socket.gethostname(), "parallel": self.parallel,
                "python": sys.version.split()[0]}
        self._retry(lambda: self.client.post("/api/register", {
            "name": self.name, "dir": str(self.repo_dir), "slots": self.slots, "info": info}),
            "register")

    # ---- version verification (once per job) -------------------------------------------
    def _verify_job(self, jid: int, job: dict) -> "tuple[bool, str]":
        with self._vlock:
            if jid in self._verified:
                return self._verified[jid]
            if not self.check_version or not job.get("sha"):
                res = (True, "no version pinned")
            else:
                from distribute_pull import compare_versions, extract_backend, local_commit_sha
                backend = extract_backend(job["gen_args"])
                solver = None
                if backend and job.get("solver"):
                    from prepare_excitation import solver_identity
                    solver = solver_identity(backend)
                res = compare_versions(local_commit_sha(), job["sha"], solver, job.get("solver"))
            self._verified[jid] = res
            return res

    # ---- one chunk ---------------------------------------------------------------------
    def run_leased(self, slot: int, lease: dict) -> str:
        """Runs one lease to a reported outcome: 'done' | 'failed' | 'released' | 'lost'."""
        cid, jid, job, chunk = lease["chunk_id"], lease["job_id"], lease["job"], lease["spec"]
        tool = self.jobs.get(job["tool"])
        if tool is None:
            self._report(lambda: self.client.post("/api/release", {
                "chunk_id": cid, "worker": self.name, "block": True,
                "reason": f"unknown tool {job['tool']!r}"}))
            return "released"
        ok, why = self._verify_job(jid, job)
        if not ok:
            log(f"job {jid}: refusing -- {why}")
            self._report(lambda: self.client.post("/api/release", {
                "chunk_id": cid, "worker": self.name, "block": True, "reason": why}))
            return "released"

        base, chunk_output = chunk_paths(job, slot, chunk, tool)
        os.makedirs(base, exist_ok=True)
        kill_stale(tool, chunk_output)
        argv = renderer_argv(self.python, job, tool, chunk, chunk_output, self.parallel)
        env = dict(os.environ)
        env.update({k: os.path.expandvars(v) for k, v in self.env.items()})
        log(f"slot {slot}: job {jid} chunk {chunk} (attempt {lease['attempt']})")
        t0 = time.time()
        try:
            proc = subprocess.Popen(argv, cwd=self.repo_dir, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=True)
        except OSError as e:
            self._report(lambda: self.client.post("/api/fail", {
                "chunk_id": cid, "worker": self.name, "error": f"could not start renderer: {e}"}))
            return "failed"

        lines, st = [], {"done": 0, "last": t0, "first": None}

        def pump():
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))
                del lines[:-TAIL_LINES]
                if tool.progress_re.match(line.strip()):
                    now = time.time()
                    st["done"] += 1
                    st["last"] = now
                    if st["first"] is None:
                        st["first"] = now - t0
        pt = threading.Thread(target=pump, daemon=True)
        pt.start()

        verdict = None            # None | ("lost",) | ("cancel", reason, slow) | ("stop",)
        next_beat = time.time() + self.heartbeat_s
        while proc.poll() is None:
            if self.stop.is_set():
                verdict = ("stop",)
                break
            if time.time() >= next_beat:
                next_beat = time.time() + self.heartbeat_s
                try:
                    r = self.client.post("/api/heartbeat", {
                        "chunk_id": cid, "worker": self.name, "progress": st["done"],
                        "elapsed": time.time() - t0, "since_last": time.time() - st["last"]})
                except (CoordinatorDown, CoordinatorError) as e:
                    log(f"heartbeat failed ({e}); renderer keeps running")
                else:
                    if not r.get("ok"):
                        verdict = ("lost",)
                        break
                    if r.get("cancel"):
                        verdict = ("cancel", r.get("reason", "job cancelled"), bool(r.get("slow")))
                        break
            time.sleep(0.2)
        if verdict:
            kill_group(proc)
        proc.wait()
        pt.join(timeout=10)
        dt = time.time() - t0
        tail = "\n".join(lines)

        if verdict and verdict[0] == "lost":
            log(f"slot {slot}: lease on {chunk} was lost -- abandoned, nothing to report")
            return "lost"
        if verdict and verdict[0] == "stop":
            self._report(lambda: self.client.post("/api/release", {
                "chunk_id": cid, "worker": self.name, "reason": "agent stopping"}),
                give_up_s=10.0)
            return "released"
        if verdict and verdict[0] == "cancel":
            if verdict[2]:
                msg = f"{verdict[1]} -- abandoning this chunk\n{tail}"
                self._report(lambda: self.client.post("/api/fail", {
                    "chunk_id": cid, "worker": self.name, "error": msg}))
                return "failed"
            self._report(lambda: self.client.post("/api/release", {
                "chunk_id": cid, "worker": self.name, "reason": verdict[1]}))
            return "released"
        if proc.returncode == 0:
            body = {"chunk_id": cid, "worker": self.name, "duration": dt, "combos": st["done"]}
            if st["first"] is not None:
                body["first_s"] = st["first"]
            self._report(lambda: self.client.post("/api/complete", body))
            self.ran += 1
            return "done"
        self._report(lambda: self.client.post("/api/fail", {
            "chunk_id": cid, "worker": self.name,
            "error": f"rc={proc.returncode}\n{tail}"}))
        return "failed"

    def _report(self, fn, give_up_s: "float | None" = None):
        """A finished chunk's result is worth waiting out a coordinator restart for -- even
        while stopping, where the wait is bounded so shutdown cannot hang on it."""
        try:
            return self._retry(fn, "report", give_up_s=give_up_s or self.report_retry_s,
                               honor_stop=False)
        except CoordinatorDown:
            return None
        except CoordinatorError as e:
            log(f"report rejected: {e}")
            return None

    # ---- loops -------------------------------------------------------------------------
    def slot_loop(self, slot: int):
        while not self.stop.is_set():
            try:
                r = self._retry(lambda: self.client.post("/api/lease", {"worker": self.name,
                                                                        "slot": slot}), "lease")
            except CoordinatorError as e:
                if e.code == 404:                       # coordinator lost its worker table
                    self.register()
                    continue
                log(f"lease rejected: {e}")
                self.stop.wait(self.poll_s)
                continue
            except CoordinatorDown:
                return
            lease = r.get("chunk")
            if lease is None:
                if self.exit_when_idle:
                    return
                self.stop.wait(self.poll_s)
                continue
            try:
                self.run_leased(slot, lease)
            except Exception as e:      # noqa: BLE001 -- one bad chunk must not end the slot
                log(f"slot {slot}: unexpected {type(e).__name__}: {e}")
                self._report(lambda: self.client.post("/api/fail", {
                    "chunk_id": lease["chunk_id"], "worker": self.name,
                    "error": f"agent error: {type(e).__name__}: {e}"}))

    def run(self):
        self.register()
        log(f"agent {self.name}: {self.slots} slot(s), repo {self.repo_dir}")
        ts = [threading.Thread(target=self.slot_loop, args=(k,), daemon=True)
              for k in range(self.slots)]
        for t in ts:
            t.start()
        try:
            while any(t.is_alive() for t in ts):
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop.set()
        for t in ts:
            t.join(timeout=60)


def parse_env(pairs) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--env needs VAR=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coordinator", required=True, metavar="URL")
    ap.add_argument("--token-file", type=Path, default=None)
    ap.add_argument("--name", default=socket.gethostname().split(".")[0],
                    help="host name the operator ssh's to (default: this machine's short name)")
    ap.add_argument("--slots", type=int, default=None,
                    help="concurrent chunks (default: physical core count; per-item jobs use one "
                         "core per slot, whole-chunk jobs use --parallel cores in ONE slot)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="--workers passed to a whole-chunk (non per-item) renderer "
                         "(default: physical core count)")
    ap.add_argument("--env", action="append", default=[], metavar="VAR=value",
                    help="repeatable; exported to the renderer ($VARs in the value expand)")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--heartbeat-s", type=float, default=20.0)
    ap.add_argument("--exit-when-idle", action="store_true",
                    help="exit once the queue has nothing for this agent (for scripted runs)")
    ap.add_argument("--skip-version-check", action="store_true")
    a = ap.parse_args(argv)
    token = os.environ.get("FLEET_TOKEN")
    if a.token_file:
        token = a.token_file.expanduser().read_text().strip()
    if not token:
        ap.error("need --token-file (or FLEET_TOKEN in the environment)")
    from cpu_topology import physical_cpu_count
    cores = physical_cpu_count()
    agent = Agent(FleetClient(a.coordinator, token), a.name, a.slots or cores,
                  a.parallel or cores, env=parse_env(a.env), poll_s=a.poll_s,
                  heartbeat_s=a.heartbeat_s, exit_when_idle=a.exit_when_idle,
                  check_version=not a.skip_version_check)
    signal.signal(signal.SIGTERM, lambda *_: agent.stop.set())
    agent.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
