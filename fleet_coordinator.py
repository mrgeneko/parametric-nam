"""HTTP front for the fleet queue (docs/implementation-roadmap.md item 11).

Agents (fleet_agent.py) and the operator's CLI (fleet_ctl.py) speak JSON over HTTP to this
process; only it touches the SQLite queue (fleet_queue.py). The dashboard is `GET /`.

AUTH: every /api/ request needs `Authorization: Bearer <token>`. The dashboard page itself is
static and carries no data; it reads the token from the URL fragment (`http://host:8765/#TOKEN`,
never sent to any server) and uses it for its own /api/status polling.

TRANSPORT IS PLAIN HTTP. The token authenticates, it does not encrypt, and an authenticated
caller can make agents run this repo's renderers with arbitrary arguments. Bind it to
127.0.0.1 (default) and reach it from workers through an ssh reverse tunnel, or bind a LAN /
Tailscale address you trust. Do not expose it to the internet.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from distribute_pull import ComboPace, log
from fleet_queue import FleetQueue

DEFAULT_TOKEN_FILE = Path("~/.config/parametric-nam/fleet-coordinator.token")
MAX_BODY = 8 * 1024 * 1024        # a job with a huge --items list is still far below this


def load_or_create_token(path: Path) -> str:
    path = Path(path).expanduser()
    if path.exists():
        tok = path.read_text().strip()
        if tok:
            return tok
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok + "\n")
    return tok


class Coordinator:
    """The queue plus the fleet-wide pacing statistics (in memory: a restart just makes the
    fleet 'cold' again, which ComboPace already treats as 'no evidence, do not kill')."""

    def __init__(self, queue: FleetQueue, pace: "ComboPace | None" = None):
        self.queue = queue
        self.pace = pace

    def heartbeat(self, b: dict) -> dict:
        r = self.queue.heartbeat(int(b["chunk_id"]), b["worker"], int(b.get("progress", 0)))
        if r["ok"] and not r["cancel"] and self.pace is not None and "elapsed" in b:
            slow, why = self.pace.verdict(int(b.get("progress", 0)),
                                          float(b.get("since_last", 0)), float(b["elapsed"]))
            if slow:
                return {"ok": True, "cancel": True, "reason": why, "slow": True}
        return r

    def complete(self, b: dict) -> dict:
        accepted = self.queue.complete(int(b["chunk_id"]), b["worker"],
                                       float(b.get("duration", 0)), int(b.get("combos", 0)))
        if accepted and self.pace is not None:
            if b.get("first_s"):
                self.pace.record_first(float(b["first_s"]))
            self.pace.record_rate(int(b.get("combos", 0)), float(b.get("duration", 0)))
        return {"accepted": accepted}

    def handle(self, method: str, path: str, body: dict) -> "tuple[int, dict]":
        q = self.queue
        if method == "GET":
            if path == "/api/status":
                return 200, q.status()
            m = re.fullmatch(r"/api/jobs/(\d+)", path)
            if m:
                jid = int(m.group(1))
                return 200, {"id": jid, "spec": q.job_spec(jid), "counts": q.job_counts(jid),
                             "finished": q.job_finished(jid),
                             "worker_slots": q.worker_slots(jid)}
            return 404, {"error": "not found"}
        if method != "POST":
            return 405, {"error": "method not allowed"}
        if path == "/api/register":
            q.register_worker(body["name"], body.get("dir", ""), int(body.get("slots", 1)),
                              body.get("info"))
            return 200, {"ok": True}
        if path == "/api/lease":
            return 200, {"chunk": q.lease(body["worker"], int(body.get("slot", 0)))}
        if path == "/api/heartbeat":
            return 200, self.heartbeat(body)
        if path == "/api/complete":
            return 200, self.complete(body)
        if path == "/api/fail":
            return 200, {"result": q.fail(int(body["chunk_id"]), body["worker"],
                                          body.get("error", ""))}
        if path == "/api/release":
            return 200, {"released": q.release(int(body["chunk_id"]), body["worker"],
                                               bool(body.get("block")), body.get("reason", ""))}
        if path == "/api/jobs":
            jid = q.submit_job(body["spec"], body["chunks"], int(body.get("retries", 2)))
            return 200, {"job_id": jid}
        m = re.fullmatch(r"/api/jobs/(\d+)/cancel", path)
        if m:
            q.cancel_job(int(m.group(1)))
            return 200, {"ok": True}
        m = re.fullmatch(r"/api/workers/([^/]+)/unquarantine", path)
        if m:
            q.unquarantine(m.group(1))
            return 200, {"ok": True}
        return 404, {"error": "not found"}


def make_handler(coord: Coordinator, token: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):      # per-heartbeat access lines would drown the log
            pass

        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            got = self.headers.get("Authorization", "")
            return hmac.compare_digest(got.encode(), f"Bearer {token}".encode())

        def _dispatch(self, method):
            path = self.path.split("?", 1)[0]
            if method == "GET" and path in ("/", "/index.html"):
                return self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            if not path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            if not self._authed():
                return self._send(401, {"error": "bad or missing bearer token"})
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self._send(413, {"error": "body too large"})
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return self._send(400, {"error": "body must be a JSON object"})
            try:
                code, payload = coord.handle(method, path, body)
            except KeyError as e:
                code, payload = 404, {"error": f"not found: {e}"}
            except (ValueError, TypeError) as e:
                code, payload = 400, {"error": f"bad request: {e}"}
            except Exception as e:          # noqa: BLE001 -- a handler bug must not kill the server
                log(f"coordinator: {method} {path} raised {type(e).__name__}: {e}")
                code, payload = 500, {"error": f"{type(e).__name__}: {e}"}
            self._send(code, payload)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def make_server(queue: FleetQueue, token: str, host: str = "127.0.0.1", port: int = 0,
                pace: "ComboPace | None" = None) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(Coordinator(queue, pace), token))
    srv.daemon_threads = True
    return srv


def serve_in_thread(queue, token, host="127.0.0.1", port=0, pace=None):
    """For tests and embedding: (server, url). Stop with server.shutdown()."""
    srv = make_server(queue, token, host, port, pace)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://{host}:{srv.server_address[1]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("~/.local/share/parametric-nam/fleet.db"),
                    help="SQLite queue (created if missing; survives restarts)")
    ap.add_argument("--bind", default="127.0.0.1", help="address to listen on (default loopback)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE,
                    help="bearer token; created (mode 600) if absent")
    ap.add_argument("--lease-s", type=float, default=120.0,
                    help="an agent silent this long loses its chunk (default 120)")
    ap.add_argument("--quarantine-after", type=int, default=3)
    ap.add_argument("--slow-mult", type=float, default=3.0, help="0 disables slow-chunk abandon")
    ap.add_argument("--slow-min-samples", type=int, default=2)
    ap.add_argument("--slow-startup-floor-min", type=float, default=90.0)
    ap.add_argument("--slow-steady-floor-min", type=float, default=30.0)
    a = ap.parse_args(argv)
    db = a.db.expanduser()
    db.parent.mkdir(parents=True, exist_ok=True)
    token = load_or_create_token(a.token_file)
    pace = ComboPace(a.slow_mult, a.slow_min_samples, a.slow_startup_floor_min * 60.0,
                     a.slow_steady_floor_min * 60.0) if a.slow_mult > 0 else None
    queue = FleetQueue(db, lease_s=a.lease_s, quarantine_after=a.quarantine_after)
    srv = make_server(queue, token, a.bind, a.port, pace)
    log(f"coordinator on http://{a.bind}:{srv.server_address[1]}  db={db}")
    log(f"dashboard:   http://{a.bind}:{srv.server_address[1]}/#{token}")
    if a.bind not in ("127.0.0.1", "localhost", "::1"):
        log("WARNING: bound to a non-loopback address over plain HTTP -- see this module's "
            "docstring; use a trusted network only.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fleet</title>
<style>
:root{--bg:#f6f7f9;--fg:#1c2026;--mut:#66707c;--card:#fff;--line:#dfe3e8;--ok:#1f8a4c;--warn:#b7791f;--bad:#c53030;--bar:#3b82f6}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e6e9ee;--mut:#8b95a1;--card:#1d2127;--line:#2c333b;--ok:#4cc38a;--warn:#e0a94a;--bad:#f06a6a;--bar:#5a9bff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:22px 0 8px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
.sub{color:var(--mut);font-size:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:10px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;display:flex;margin:8px 0 4px}
.bar i{display:block;height:100%}.b-done{background:var(--ok)}.b-run{background:var(--bar)}.b-fail{background:var(--bad)}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:500;font-size:12px}.wrap{overflow-x:auto}
.pill{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;border:1px solid var(--line)}
.on{color:var(--ok)}.off{color:var(--mut)}.q{color:var(--bad);border-color:var(--bad)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px ui-monospace,monospace;color:var(--mut)}
#err{color:var(--bad);margin:8px 0}
</style></head><body>
<h1>Fleet</h1><div class="sub" id="stamp">connecting...</div><div id="err"></div>
<h2>Jobs</h2><div id="jobs"></div>
<h2>Workers</h2><div class="card wrap"><table id="workers"></table></div>
<h2>In flight</h2><div class="card wrap"><table id="inflight"></table></div>
<h2>Recent problems</h2><div id="problems"></div>
<script>
const token = decodeURIComponent(location.hash.slice(1));
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const dur = s => s == null ? "-" : s < 90 ? Math.round(s)+"s" : s < 5400 ? (s/60).toFixed(1)+"m" : (s/3600).toFixed(1)+"h";
async function tick(){
  try{
    const r = await fetch("/api/status",{headers:{Authorization:"Bearer "+token}});
    if(r.status===401){document.getElementById("err").textContent="Unauthorized: open this page as /#<token>";return}
    const s = await r.json();
    document.getElementById("err").textContent="";
    document.getElementById("stamp").textContent="updated "+new Date().toLocaleTimeString();
    document.getElementById("jobs").innerHTML = s.jobs.slice().reverse().map(j=>{
      const c=j.counts, p=x=>c.total?100*x/c.total:0;
      const state = j.cancelled?"cancelled":j.finished?(c.failed?"finished with failures":"complete"):"running";
      return `<div class="card"><b>job ${j.id}</b> ${esc(j.tool)} &rarr; ${esc(j.output)} <span class="pill">${state}</span>
      <div class="bar"><i class="b-done" style="width:${p(c.done)}%"></i><i class="b-run" style="width:${p(c.leased)}%"></i><i class="b-fail" style="width:${p(c.failed)}%"></i></div>
      <div class="sub">${c.done}/${c.total} done &middot; ${c.leased} running &middot; ${c.pending} queued &middot; ${c.failed} failed &middot; ${dur(j.elapsed)} since submit</div></div>`}).join("")||'<div class="sub">no jobs</div>';
    document.getElementById("workers").innerHTML = "<tr><th>worker</th><th>state</th><th>slots busy</th><th>chunks</th><th>combos</th><th>chunks/h</th><th>failed</th><th>last seen</th></tr>"+
      s.workers.map(w=>`<tr><td>${esc(w.name)}</td><td>${w.quarantined?'<span class="pill q">quarantined</span>':w.online?'<span class="pill on">online</span>':'<span class="pill off">offline</span>'}</td>
      <td>${w.busy}/${w.slots}</td><td>${w.done}</td><td>${w.combos}</td><td>${w.chunks_per_hour.toFixed(2)}</td><td>${w.failed}</td><td>${dur(w.last_seen_age)} ago</td></tr>`).join("");
    document.getElementById("inflight").innerHTML = "<tr><th>job</th><th>chunk</th><th>worker</th><th>slot</th><th>attempt</th><th>combos</th><th>running</th><th>last beat</th></tr>"+
      s.inflight.map(c=>`<tr><td>${c.job_id}</td><td>${esc(c.spec)}</td><td>${esc(c.worker)}</td><td>${c.slot}</td><td>${c.attempts}</td><td>${c.progress}</td><td>${dur(s.now-c.started)}</td><td>${dur(s.now-c.last_beat)} ago</td></tr>`).join("");
    document.getElementById("problems").innerHTML = s.problems.map(p=>`<div class="card"><b>job ${p.job_id} chunk ${esc(p.spec)}</b> <span class="pill">${esc(p.state)}</span> <span class="sub">attempts ${p.attempts}, tried on ${esc(JSON.parse(p.tried_on).join(", "))}</span><pre>${esc((p.error||"").split("\n").slice(-4).join("\n"))}</pre></div>`).join("")||'<div class="sub">none</div>';
  }catch(e){document.getElementById("err").textContent="Coordinator unreachable: "+e}
}
tick(); setInterval(tick,3000);
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
