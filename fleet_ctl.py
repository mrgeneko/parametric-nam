#!/usr/bin/env python3
"""Operator CLI for the pull-agent fleet (docs/implementation-roadmap.md item 11).

    fleet_ctl.py serve                          # run the coordinator + dashboard (foreground)
    fleet_ctl.py start-agents --worker HOST:DIR[:SLOTS[:ENV]] ... --agent-url URL
    fleet_ctl.py submit --config C --output DIR --chunk-size 1 --slots ... [--wait --collect DEST]
    fleet_ctl.py status [--watch]
    fleet_ctl.py collect JOB --dest [HOST:]DIR  # same merge/combine as distribute_pull --collect
    fleet_ctl.py cancel JOB | unquarantine WORKER | stop-agents --worker HOST:DIR ...

`submit` does the controller-side work distribute_pull.py's main() does before dispatching --
expands --config into renderer arguments, runs the (warn-only) gate check, derives the item
count -- then hands the finished description to the coordinator instead of ssh'ing chunks out.
Shards stay on the workers until `collect`, which reuses distribute_pull.run_collect, so
merge/verify/combine and the sink transfer paths are byte-for-byte the ones already in use.

Connection: --url (env FLEET_URL, default http://127.0.0.1:8765) and --token-file (env
FLEET_TOKEN, default the coordinator's own token file). `--agent-url` on start-agents is the URL
the WORKERS use, which is usually not the one you use (loopback here, a LAN/tunnel address there).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import ssh_target
from fleet_client import CoordinatorDown, CoordinatorError, FleetClient

REPO = Path(__file__).resolve().parent
AGENT_TOKEN_PATH = "~/.config/parametric-nam/fleet-agent.token"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- client plumbing -------------------------------------------------------------------
def make_client(args) -> FleetClient:
    from fleet_coordinator import DEFAULT_TOKEN_FILE
    token = os.environ.get("FLEET_TOKEN")
    tf = args.token_file or (None if token else DEFAULT_TOKEN_FILE)
    if tf:
        p = Path(tf).expanduser()
        if not p.exists():
            sys.exit(f"token file {p} not found (start the coordinator once, or pass --token-file)")
        token = p.read_text().strip()
    return FleetClient(args.url or os.environ.get("FLEET_URL", "http://127.0.0.1:8765"), token)


# ---- submit ----------------------------------------------------------------------------
def chunk_specs(n: int) -> "list[str]":
    return [f"{i}-{i}/{n}" for i in range(n)]


def build_job_spec(args, extra_args: "list[str]") -> "tuple[dict, list[str]]":
    """(job spec, chunk specs). Raises SystemExit with a message on operator error -- the same
    checks and the same expansion distribute_pull.main() applies before it dispatches."""
    import distribute_pull as dp
    job = dp.JOBS[args.tool]
    gen_args = list(extra_args)
    if args.config:
        gen_args = job.build_args(args.config, REPO, gen_args)
        if args.tool == "gen_dataset" and not args.skip_gate_check:
            from gate_config import verify_gate
            from run_pipeline import gate_check_outcome
            try:
                ok, gate_msg = verify_gate(args.config)
            except Exception as e:      # noqa: BLE001
                ok, gate_msg = False, f"gate check raised {type(e).__name__}: {e}"
            line, abort = gate_check_outcome(args.config, ok, gate_msg, args.require_gate)
            log(line)
            if abort:
                sys.exit(2)
    if not gen_args:
        sys.exit("pass --config, or the renderer's own arguments after --")

    per_item = args.chunk_size is not None
    item_count = None
    if per_item:
        if args.chunk_size != 1:
            sys.exit(f"--chunk-size {args.chunk_size}: only 1 (per-item dispatch) is implemented")
        try:
            item_count = dp.derive_item_count(gen_args, args.items)
        except ValueError as e:
            sys.exit(str(e))
        n = item_count
    else:
        n = args.chunks

    spec = {"tool": args.tool, "gen_args": gen_args, "output": args.output, "per_item": per_item,
            "item_count": item_count, "extra_args": list(extra_args),
            "config": str(Path(args.config).resolve()) if args.config else None,
            "label": args.label}
    if not args.skip_version_check:
        spec["sha"] = dp.local_commit_sha()
        backend = dp.extract_backend(gen_args)
        if backend and spec["sha"]:
            from prepare_excitation import solver_identity
            spec["solver"] = solver_identity(backend)
    return spec, chunk_specs(n)


def cmd_submit(args, extra_args):
    client = make_client(args)
    spec, chunks = build_job_spec(args, extra_args)
    jid = client.post("/api/jobs", {"spec": spec, "chunks": chunks,
                                    "retries": args.retries})["job_id"]
    log(f"submitted job {jid}: {len(chunks)} chunk(s), {spec['tool']} -> {spec['output']}")
    if args.wait or args.collect:
        rc = wait_for_job(client, jid, args.poll_s)
        if args.collect:
            return do_collect(client, jid, args, args.collect) or rc
        return rc
    return 0


def wait_for_job(client, jid, poll_s=5.0) -> int:
    last = None
    while True:
        try:
            info = client.get(f"/api/jobs/{jid}")
        except CoordinatorDown as e:
            log(f"coordinator unreachable ({e}); still waiting")
            time.sleep(poll_s)
            continue
        c = info["counts"]
        line = (f"job {jid}: {c['done']}/{c['total']} done, {c['leased']} running, "
                f"{c['pending']} queued, {c['failed']} failed")
        if line != last:
            log(line)
            last = line
        if info["finished"]:
            return 1 if c["failed"] else 0
        time.sleep(poll_s)


# ---- collect ---------------------------------------------------------------------------
def do_collect(client, jid, args, dest) -> int:
    import distribute_pull as dp
    info = client.get(f"/api/jobs/{jid}")
    spec, c = info["spec"], info["counts"]
    if not info["finished"] and not args.partial:
        sys.exit(f"job {jid} is not finished ({c['done']}/{c['total']} done, {c['leased']} "
                 f"running, {c['pending']} queued); wait, or pass --partial")
    if c["failed"]:
        log(f"WARNING: {c['failed']} chunk(s) failed for good -- collecting what exists; "
            f"the merge will report what is missing")
    slots = info["worker_slots"]
    if not slots:
        sys.exit(f"job {jid}: no chunk completed, nothing to collect")
    job = dp.JOBS[spec["tool"]]
    per_item = bool(spec.get("per_item"))
    by_host: dict = {}
    for name, slot, wdir in slots:
        by_host.setdefault(name, (dp.Worker(f"{name}:{wdir or '.'}:1", job=job), []))[1].append(slot)
    workers = [w for w, _ in by_host.values()]
    pairs = [(w, s) for w, ss in by_host.values() for s in ss]
    dp.configure_ssh(args.inventory)
    try:
        dp.run_collect(job, workers, spec["output"], dest, per_item=per_item,
                       item_count=spec.get("item_count"),
                       config=Path(spec["config"]) if spec.get("config") else None,
                       extra_args=spec.get("extra_args") or [], no_combine=args.no_combine,
                       repair_missing=args.repair_missing, inventory=args.inventory,
                       sink_repo=args.sink_repo, no_direct_sink=args.no_direct_sink,
                       slot_pairs=pairs if per_item else None)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    return 0


# ---- status ----------------------------------------------------------------------------
def _dur(s):
    if s is None:
        return "-"
    return f"{s:.0f}s" if s < 90 else f"{s/60:.1f}m" if s < 5400 else f"{s/3600:.1f}h"


def render_status(st: dict) -> str:
    out = []
    for j in st["jobs"]:
        c = j["counts"]
        state = ("cancelled" if j["cancelled"] else
                 ("finished with failures" if c["failed"] else "complete") if j["finished"]
                 else "running")
        out.append(f"job {j['id']}  {j['tool']} -> {j['output']}  [{state}]  {c['done']}/{c['total']} "
                   f"done, {c['leased']} running, {c['pending']} queued, {c['failed']} failed, "
                   f"{_dur(j['elapsed'])} since submit")
    if not st["jobs"]:
        out.append("no jobs")
    out.append("")
    out.append(f"{'worker':<14}{'state':<13}{'busy':<7}{'chunks':<8}{'combos':<8}{'chunks/h':<10}"
               f"{'failed':<7}last seen")
    for w in st["workers"]:
        state = "QUARANTINED" if w["quarantined"] else "online" if w["online"] else "offline"
        out.append(f"{w['name']:<14}{state:<13}{str(w['busy']) + '/' + str(w['slots']):<7}"
                   f"{w['done']:<8}{w['combos']:<8}{w['chunks_per_hour']:<10.2f}{w['failed']:<7}"
                   f"{_dur(w['last_seen_age'])} ago")
    if st["inflight"]:
        out.append("")
        for c in st["inflight"]:
            out.append(f"  running: job {c['job_id']} chunk {c['spec']} on {c['worker']} slot "
                       f"{c['slot']} (attempt {c['attempts']}, {c['progress']} combos, "
                       f"{_dur(st['now'] - c['started'])})")
    for p in st["problems"][:5]:
        last = (p["error"] or "").strip().splitlines()[-1:] or [""]
        out.append(f"  problem: job {p['job_id']} chunk {p['spec']} [{p['state']}, "
                   f"{p['attempts']} attempt(s)] {last[0][:110]}")
    return "\n".join(out)


def cmd_status(args):
    client = make_client(args)
    while True:
        text = render_status(client.get("/api/status"))
        if args.watch:
            print("\033[2J\033[H", end="")
        print(text, flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch_s)


# ---- agents over ssh -------------------------------------------------------------------
def agent_start_command(dir: str, url: str, name: str, slots: int, env: "list[str]",
                        skip_version_check: bool = False, python: str = "./.venv/bin/python") -> str:
    """One remote shell command: no-op if an agent is already alive here, else launch detached
    and record its pid. The token is NOT on this command line (it would show in `ps`); see
    agent_token_command."""
    parts = [python, "-u", "fleet_agent.py", "--coordinator", url, "--token-file", AGENT_TOKEN_PATH,
             "--name", name, "--slots", str(slots)]
    for e in env:
        parts += ["--env", e]
    if skip_version_check:
        parts.append("--skip-version-check")
    # the token path keeps its ~ unquoted so the remote shell expands it
    cmd = " ".join(p if p == AGENT_TOKEN_PATH else shlex.quote(p) for p in parts)
    return (f"cd {dir} && if [ -f .fleet_agent.pid ] && kill -0 $(cat .fleet_agent.pid) 2>/dev/null; "
            f"then echo 'agent already running (pid '$(cat .fleet_agent.pid)')'; else "
            f"nohup {cmd} > .fleet_agent.log 2>&1 < /dev/null & echo $! > .fleet_agent.pid; "
            f"echo 'agent started (pid '$!')'; fi")


def agent_token_command() -> str:
    return (f"umask 077 && mkdir -p ~/.config/parametric-nam && cat > {AGENT_TOKEN_PATH}")


def agent_stop_command(dir: str) -> str:
    return (f"cd {dir} && if [ -f .fleet_agent.pid ] && "
            f"ps -p $(cat .fleet_agent.pid) -o command= 2>/dev/null | grep -q fleet_agent.py; then "
            f"kill $(cat .fleet_agent.pid) && echo stopped; else echo 'no agent running'; fi; "
            f"rm -f .fleet_agent.pid")


def _ssh(host, cmd, stdin=None, timeout=120):
    return subprocess.run(ssh_target.ssh_argv(host, "-o", "BatchMode=yes") + [cmd], input=stdin,
                          capture_output=True, text=True, timeout=timeout)


def cmd_start_agents(args, _extra):
    import distribute_pull as dp
    dp.configure_ssh(args.inventory)
    token = (Path(args.agent_token_file or args.token_file
                  or os.path.expanduser("~/.config/parametric-nam/fleet-coordinator.token"))
             .expanduser().read_text().strip())
    workers = [dp.Worker(w) for w in args.worker]
    if not args.skip_version_check:
        workers = dp.verify_workers(workers, None)
    if args.sync_file:
        workers = dp.sync_files(workers, args.sync_file)
    rc = 0
    for w in workers:
        r = _ssh(w.host, agent_token_command(), stdin=token + "\n")
        if r.returncode != 0:
            log(f"{w.host}: could not install the agent token: {r.stderr.strip()[:200]}")
            rc = 1
            continue
        slots = args.slots or w.parallel
        r = _ssh(w.host, agent_start_command(w.dir, args.agent_url, w.host, slots,
                                             [w.env] if w.env else [], args.skip_version_check))
        log(f"{w.host}: {(r.stdout.strip() or r.stderr.strip())[:200]}")
        rc = rc or (1 if r.returncode else 0)
    return rc


def cmd_stop_agents(args, _extra):
    import distribute_pull as dp
    dp.configure_ssh(args.inventory)
    rc = 0
    for spec in args.worker:
        host, dir = spec.split(":")[:2]
        r = _ssh(host, agent_stop_command(dir))
        log(f"{host}: {(r.stdout.strip() or r.stderr.strip())[:200]}")
        rc = rc or (1 if r.returncode else 0)
    return rc


# ---- argument parsing ------------------------------------------------------------------
def add_conn(p):
    p.add_argument("--url", default=None, help="coordinator URL (env FLEET_URL)")
    p.add_argument("--token-file", type=Path, default=None)


def add_collect_flags(p):
    p.add_argument("--no-combine", action="store_true")
    p.add_argument("--repair-missing", action="store_true")
    p.add_argument("--sink-repo", default=None)
    p.add_argument("--no-direct-sink", action="store_true")
    p.add_argument("--inventory", nargs="?", const="__DEFAULT__", default=None,
                   help="see distribute_pull.py --inventory (needed for per-host logins)")
    p.add_argument("--partial", action="store_true",
                   help="collect even though the job has not finished")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the coordinator (arguments after -- go to it)")

    p = sub.add_parser("submit")
    add_conn(p)
    p.add_argument("--tool", default="gen_dataset")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output", required=True, help="output dir ON EACH WORKER")
    p.add_argument("--chunks", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--items", type=int, default=None)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--label", default=None)
    p.add_argument("--skip-gate-check", action="store_true")
    p.add_argument("--require-gate", action="store_true")
    p.add_argument("--skip-version-check", action="store_true",
                   help="don't pin this checkout's commit/solver: agents run the job regardless")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-s", type=float, default=5.0)
    p.add_argument("--collect", metavar="[HOST:]DIR", default=None,
                   help="wait for the job, then collect (implies --wait)")
    add_collect_flags(p)

    p = sub.add_parser("status")
    add_conn(p)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--watch-s", type=float, default=3.0)

    p = sub.add_parser("collect")
    add_conn(p)
    p.add_argument("job", type=int)
    p.add_argument("--dest", required=True, metavar="[HOST:]DIR")
    add_collect_flags(p)

    p = sub.add_parser("cancel")
    add_conn(p)
    p.add_argument("job", type=int)

    p = sub.add_parser("unquarantine")
    add_conn(p)
    p.add_argument("worker")

    p = sub.add_parser("start-agents")
    p.add_argument("--worker", action="append", required=True, metavar="HOST:DIR[:SLOTS[:ENV]]")
    p.add_argument("--agent-url", required=True, help="coordinator URL as the WORKERS reach it")
    p.add_argument("--token-file", type=Path, default=None,
                   help="the coordinator's token (default: its own token file)")
    p.add_argument("--agent-token-file", type=Path, default=None)
    p.add_argument("--slots", type=int, default=None)
    p.add_argument("--sync-file", action="append", default=[])
    p.add_argument("--skip-version-check", action="store_true")
    p.add_argument("--inventory", nargs="?", const="__DEFAULT__", default=None)

    p = sub.add_parser("stop-agents")
    p.add_argument("--worker", action="append", required=True, metavar="HOST:DIR")
    p.add_argument("--inventory", nargs="?", const="__DEFAULT__", default=None)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = build_parser().parse_args(argv)
    if args.cmd == "serve":
        import fleet_coordinator
        return fleet_coordinator.main(extra)
    if args.cmd == "submit":
        if args.collect:
            import distribute_pull as dp
            dp.configure_ssh(args.inventory)
        return cmd_submit(args, extra)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "collect":
        return do_collect(make_client(args), args.job, args, args.dest)
    if args.cmd == "cancel":
        make_client(args).post(f"/api/jobs/{args.job}/cancel")
        log(f"cancelled job {args.job}")
        return 0
    if args.cmd == "unquarantine":
        make_client(args).post(f"/api/workers/{args.worker}/unquarantine")
        log(f"{args.worker} released from quarantine")
        return 0
    if args.cmd == "start-agents":
        return cmd_start_agents(args, extra)
    if args.cmd == "stop-agents":
        return cmd_stop_agents(args, extra)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CoordinatorDown, CoordinatorError) as e:
        sys.exit(f"fleet_ctl: {e}")
