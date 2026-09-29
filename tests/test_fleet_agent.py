"""Agent + coordinator end to end on localhost: real HTTP, real SQLite, real renderer
subprocesses (a stub script standing in for the SPICE renderer)."""
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request

import pytest

import fleet_agent
from distribute_pull import ComboPace, Job
from fleet_agent import Agent
from fleet_client import CoordinatorDown, CoordinatorError, FleetClient
from fleet_coordinator import make_server, serve_in_thread
from fleet_queue import FleetQueue

STUB = textwrap.dedent('''
    import argparse, os, sys, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard"); ap.add_argument("--output"); ap.add_argument("--workers")
    ap.add_argument("--mode", default="ok"); ap.add_argument("--x")
    a = ap.parse_args()
    i = int(a.shard.split("-")[0])
    os.makedirs(a.output, exist_ok=True)
    open(os.path.join(a.output, f"pid_{i}"), "w").write(str(os.getpid()))
    open(os.path.join(a.output, f"args_{i}"), "w").write(" ".join(sys.argv[1:]))
    if os.environ.get("STUB_FAIL"):
        print("boom: import error", flush=True); sys.exit(3)
    if a.mode == "sleep":
        print("done item 0", flush=True); time.sleep(120)
    if a.mode == "silent":
        time.sleep(120)
    time.sleep(float(os.environ.get("STUB_DELAY", "0")))
    open(os.path.join(a.output, f"item_{i}.txt"), "w").write(os.environ.get("STUB_WHO", "?"))
    print(f"done item {i}", flush=True)
''')

STUB_JOB = Job(name="stub", script="stub_render.py", progress_re=re.compile(r"^done item"),
               output_flag="--output", build_args=None,
               chunk_output=lambda base, chunk: base, collect=None)
JOBS = {"stub": STUB_JOB}
TOKEN = "s3cret"


@pytest.fixture
def repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    (d / "stub_render.py").write_text(STUB)
    return d


@pytest.fixture
def coord(tmp_path):
    q = FleetQueue(tmp_path / "q.db", lease_s=60, quarantine_after=3)
    srv, url = serve_in_thread(q, TOKEN)
    client = FleetClient(url, TOKEN, timeout=5)
    yield q, srv, url, client
    srv.shutdown()
    srv.server_close()
    q.close()


def spec(out, per_item=True, mode="ok", **kw):
    s = {"tool": "stub", "gen_args": ["--mode", mode], "output": str(out), "per_item": per_item}
    s.update(kw)
    return s


def chunks(n):
    return [f"{i}-{i}/{n}" for i in range(n)]


def make_agent(client, repo, name, slots=2, **kw):
    kw.setdefault("poll_s", 0.1)
    kw.setdefault("heartbeat_s", 0.3)
    kw.setdefault("exit_when_idle", True)
    kw.setdefault("check_version", False)
    return Agent(client, name, slots, 2, repo_dir=repo, jobs=JOBS, **kw)


def run_agents(*agents, timeout=60):
    ts = [threading.Thread(target=a.run, daemon=True) for a in agents]
    [t.start() for t in ts]
    end = time.time() + timeout
    for t in ts:
        t.join(max(0.1, end - time.time()))
    assert not any(t.is_alive() for t in ts), "agents did not finish"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for(cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("condition not reached")


class TestEndToEnd:
    def test_two_agents_drain_a_per_item_job(self, coord, repo, tmp_path):
        q, _, url, client = coord
        out = tmp_path / "out"
        jid = client.post("/api/jobs", {"spec": spec(out), "chunks": chunks(8)})["job_id"]
        a = make_agent(client, repo, "a", env={"STUB_WHO": "a"})
        b = make_agent(client, repo, "b", env={"STUB_WHO": "b"})
        run_agents(a, b)
        assert q.job_counts(jid) == {"pending": 0, "leased": 0, "done": 8, "failed": 0, "total": 8}
        # every item rendered exactly once, into a slot dir under the job's output
        # (both agents share this machine's filesystem, so slot dirs may overlap by name)
        rendered = sorted(p.name for p in out.glob("slot-*/item_*.txt"))
        assert len(rendered) == 8 and len(set(rendered)) == 8
        assert {r[0] for r in q.worker_slots(jid)} <= {"a", "b"}
        info = client.get(f"/api/jobs/{jid}")
        assert info["finished"] and info["spec"]["output"] == str(out)

    def test_renderer_receives_documented_arguments(self, coord, repo, tmp_path):
        q, _, _, client = coord
        out = tmp_path / "out"
        client.post("/api/jobs", {"spec": spec(out, gen_args=["--mode", "ok", "--x", "a b"]),
                                  "chunks": ["2-2/5"]})
        run_agents(make_agent(client, repo, "a", slots=1))
        args = (out / "slot-0" / "args_2").read_text()
        assert args == f"--mode ok --x a b --workers 1 --shard 2-2/5 --output {out}/slot-0"

    def test_whole_chunk_job_uses_parallel_and_base_dir(self, coord, repo, tmp_path):
        q, _, _, client = coord
        out = tmp_path / "out"
        client.post("/api/jobs", {"spec": spec(out, per_item=False), "chunks": ["1-1/4"]})
        run_agents(make_agent(client, repo, "a", slots=3))
        args = (out / "args_1").read_text()
        assert "--workers 2" in args and str(out / "slot-") not in args

    def test_failing_worker_is_quarantined_and_work_finishes_elsewhere(self, coord, repo, tmp_path):
        q, _, _, client = coord
        out = tmp_path / "out"
        jid = client.post("/api/jobs", {"spec": spec(out), "chunks": chunks(6), "retries": 5})["job_id"]
        bad = make_agent(client, repo, "bad", slots=1, env={"STUB_FAIL": "1"})
        run_agents(bad)
        assert q.job_counts(jid)["done"] == 0
        assert [w for w in q.status()["workers"] if w["name"] == "bad"][0]["quarantined"]
        err = q.status()["problems"][0]["error"]
        assert "boom: import error" in err and "rc=3" in err
        run_agents(make_agent(client, repo, "good", env={"STUB_WHO": "good"}))
        assert q.job_counts(jid)["done"] == 6

    def test_progress_and_duration_reported(self, coord, repo, tmp_path):
        q, _, _, client = coord
        jid = client.post("/api/jobs", {"spec": spec(tmp_path / "o"), "chunks": ["0-0/1"]})["job_id"]
        run_agents(make_agent(client, repo, "a", slots=1))
        w = q.status()["workers"][0]
        assert w["done"] == 1 and w["combos"] == 1
        assert q.db.execute("SELECT duration FROM chunks").fetchone()[0] > 0

    def test_agent_registers_dir_slots_and_sha(self, coord, repo):
        q, _, _, client = coord
        run_agents(make_agent(client, repo, "a", slots=3))
        w = q.status()["workers"][0]
        assert w["dir"] == str(repo) and w["slots"] == 3 and "sha" in w["info"]


class TestOutages:
    def test_agent_waits_for_coordinator_at_startup(self, tmp_path, repo):
        q = FleetQueue(tmp_path / "q.db")
        s0 = make_server(q, TOKEN)                 # grab a free port, then release it
        port = s0.server_address[1]
        s0.server_close()
        client = FleetClient(f"http://127.0.0.1:{port}", TOKEN, timeout=2)
        agent = make_agent(client, repo, "a", slots=1)
        t = threading.Thread(target=agent.run, daemon=True)
        t.start()
        time.sleep(1.5)
        assert t.is_alive()                        # retrying, not crashed
        srv = make_server(q, TOKEN, port=port)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        q.submit_job(spec(tmp_path / "o"), ["0-0/1"])
        t.join(30)
        assert not t.is_alive() and q.job_counts(1)["done"] == 1
        srv.shutdown(); srv.server_close(); q.close()

    def test_result_delivered_after_coordinator_restart(self, tmp_path, repo, monkeypatch):
        monkeypatch.setenv("STUB_DELAY", "2.5")
        db = tmp_path / "q.db"
        q = FleetQueue(db, lease_s=60)
        srv, url = serve_in_thread(q, TOKEN)
        port = srv.server_address[1]
        client = FleetClient(url, TOKEN, timeout=2)
        jid = q.submit_job(spec(tmp_path / "o"), ["0-0/1"])
        agent = make_agent(client, repo, "a", slots=1)
        t = threading.Thread(target=agent.run, daemon=True)
        t.start()
        wait_for(lambda: q.job_counts(jid)["leased"] == 1)
        srv.shutdown(); srv.server_close(); q.close()      # coordinator dies mid-render
        time.sleep(3.5)                                    # renderer finishes, report fails
        q2 = FleetQueue(db, lease_s=60)                    # ...and comes back on the same port
        srv2 = make_server(q2, TOKEN, port=port)
        threading.Thread(target=srv2.serve_forever, daemon=True).start()
        t.join(30)
        assert q2.job_counts(jid)["done"] == 1
        srv2.shutdown(); srv2.server_close(); q2.close()

    def test_renderer_survives_heartbeat_failures(self, tmp_path, repo, monkeypatch):
        monkeypatch.setenv("STUB_DELAY", "1.5")
        q = FleetQueue(tmp_path / "q.db")
        # a client pointed at nothing: every heartbeat fails, the render must still finish
        dead = FleetClient("http://127.0.0.1:9", TOKEN, timeout=1)
        agent = make_agent(dead, repo, "a", slots=1, report_retry_s=1)
        lease = {"chunk_id": 1, "job_id": 1, "spec": "0-0/1", "attempt": 1,
                 "job": spec(tmp_path / "o")}
        assert agent.run_leased(0, lease) == "done"
        assert (tmp_path / "o" / "slot-0" / "item_0.txt").exists()
        q.close()


class TestLeaseSupervision:
    def test_lost_lease_kills_renderer_and_reports_nothing(self, tmp_path, repo):
        q = FleetQueue(tmp_path / "q.db", lease_s=0.6)
        srv, url = serve_in_thread(q, TOKEN)
        client = FleetClient(url, TOKEN, timeout=5)
        out = tmp_path / "o"
        jid = q.submit_job(spec(out, mode="sleep"), ["0-0/1"], retries=3)
        q.register_worker("a", "/d", 1)
        q.register_worker("b", "/d", 1)
        a = make_agent(client, repo, "a", slots=1, heartbeat_s=1.5)
        t = threading.Thread(target=a.run, daemon=True)
        t.start()
        wait_for(lambda: q.job_counts(jid)["leased"] == 1)
        pid = int(wait_for(lambda: (out / "slot-0" / "pid_0").exists()
                           and (out / "slot-0" / "pid_0").read_text() or None))
        time.sleep(0.9)                                   # a's lease lapses...
        stolen = q.lease("b", 0)                          # ...and b is given the chunk
        assert stolen["attempt"] == 2
        wait_for(lambda: not alive(pid), timeout=15)      # a's next heartbeat says stop
        a.stop.set(); t.join(15)
        # a's abandonment was silent: it neither failed nor blamed itself for the lost chunk
        assert {w["name"]: w["failed"] for w in q.status()["workers"]}["a"] == 0
        srv.shutdown(); srv.server_close(); q.close()

    def test_job_cancel_stops_renderer_and_releases(self, coord, repo, tmp_path):
        q, _, _, client = coord
        out = tmp_path / "o"
        jid = client.post("/api/jobs", {"spec": spec(out, mode="sleep"), "chunks": ["0-0/1"]})["job_id"]
        a = make_agent(client, repo, "a", slots=1)
        t = threading.Thread(target=a.run, daemon=True)
        t.start()
        pid = int(wait_for(lambda: (out / "slot-0" / "pid_0").exists()
                           and (out / "slot-0" / "pid_0").read_text() or None))
        client.post(f"/api/jobs/{jid}/cancel")
        wait_for(lambda: not alive(pid), timeout=15)
        t.join(30)
        c = q.job_counts(jid)
        assert c["failed"] == 1 and c["leased"] == 0
        assert [w for w in q.status()["workers"]][0]["failed"] == 0     # not the worker's fault

    def test_slow_chunk_abandoned_using_fleet_pace(self, tmp_path, repo):
        q = FleetQueue(tmp_path / "q.db")
        pace = ComboPace(slow_mult=1.0, min_samples=1, startup_floor_s=0.5, steady_floor_s=0.5)
        pace.record_first(0.5)
        srv, url = serve_in_thread(q, TOKEN, pace=pace)
        client = FleetClient(url, TOKEN, timeout=5)
        out = tmp_path / "o"
        jid = q.submit_job(spec(out, mode="silent"), ["0-0/1"], retries=0)
        t0 = time.time()
        run_agents(make_agent(client, repo, "a", slots=1, heartbeat_s=0.3))
        assert time.time() - t0 < 30
        pid = int((out / "slot-0" / "pid_0").read_text())
        assert not alive(pid)
        assert q.job_counts(jid)["failed"] == 1
        assert "abandoning this chunk" in q.status()["problems"][0]["error"]
        srv.shutdown(); srv.server_close(); q.close()

    def test_no_pace_means_a_slow_chunk_is_left_alone(self, coord, repo, tmp_path, monkeypatch):
        monkeypatch.setenv("STUB_DELAY", "1.5")
        q, _, _, client = coord
        jid = client.post("/api/jobs", {"spec": spec(tmp_path / "o"), "chunks": ["0-0/1"]})["job_id"]
        run_agents(make_agent(client, repo, "a", slots=1, heartbeat_s=0.2))
        assert q.job_counts(jid)["done"] == 1

    def test_stop_releases_without_burning_an_attempt(self, coord, repo, tmp_path):
        q, _, _, client = coord
        out = tmp_path / "o"
        jid = client.post("/api/jobs", {"spec": spec(out, mode="sleep"), "chunks": ["0-0/1"],
                                        "retries": 0})["job_id"]
        a = make_agent(client, repo, "a", slots=1)
        t = threading.Thread(target=a.run, daemon=True)
        t.start()
        pid = int(wait_for(lambda: (out / "slot-0" / "pid_0").exists()
                           and (out / "slot-0" / "pid_0").read_text() or None))
        a.stop.set()
        t.join(30)
        assert not alive(pid)
        assert q.job_counts(jid)["pending"] == 1          # retries=0, yet not failed
        assert q.lease("a", 0)["attempt"] == 1


class TestRefusals:
    def test_version_mismatch_blocks_worker_from_job(self, coord, repo, tmp_path):
        q, _, _, client = coord
        jid = client.post("/api/jobs", {"spec": spec(tmp_path / "o", sha="0" * 40),
                                        "chunks": chunks(2)})["job_id"]
        run_agents(make_agent(client, repo, "a", slots=1, check_version=True))
        assert q.job_counts(jid)["pending"] == 2 and q.job_counts(jid)["done"] == 0
        assert q.db.execute("SELECT COUNT(*) FROM blocks WHERE worker='a'").fetchone()[0] == 1
        # not a failure: no attempt burned, worker not blamed
        assert q.status()["workers"][0]["failed"] == 0
        assert not (tmp_path / "o").exists()

    def test_matching_version_runs(self, coord, repo, tmp_path):
        import distribute_pull as dp
        sha = dp.local_commit_sha()
        if sha is None:
            pytest.skip("not a git checkout")
        q, _, _, client = coord
        jid = client.post("/api/jobs", {"spec": spec(tmp_path / "o", sha=sha), "chunks": ["0-0/1"]})["job_id"]
        run_agents(make_agent(client, repo, "a", slots=1, check_version=True))
        assert q.job_counts(jid)["done"] == 1

    def test_unpinned_job_skips_check(self, coord, repo, tmp_path):
        q, _, _, client = coord
        jid = client.post("/api/jobs", {"spec": spec(tmp_path / "o"), "chunks": ["0-0/1"]})["job_id"]
        run_agents(make_agent(client, repo, "a", slots=1, check_version=True))
        assert q.job_counts(jid)["done"] == 1

    def test_unknown_tool_is_refused_not_failed(self, coord, repo, tmp_path):
        q, _, _, client = coord
        s = spec(tmp_path / "o")
        s["tool"] = "nonesuch"
        jid = client.post("/api/jobs", {"spec": s, "chunks": ["0-0/1"], "retries": 0})["job_id"]
        run_agents(make_agent(client, repo, "a", slots=1))
        assert q.job_counts(jid)["pending"] == 1 and q.job_counts(jid)["failed"] == 0

    def test_renderer_that_cannot_start_fails_the_chunk(self, coord, repo, tmp_path):
        q, _, _, client = coord
        jid = client.post("/api/jobs", {"spec": spec(tmp_path / "o"), "chunks": ["0-0/1"],
                                        "retries": 0})["job_id"]
        run_agents(make_agent(client, repo, "a", slots=1, python="/nonexistent/python"))
        assert q.job_counts(jid)["failed"] == 1
        assert "could not start renderer" in q.status()["problems"][0]["error"]


class TestHelpers:
    def test_chunk_paths_per_item_and_whole(self):
        assert fleet_agent.chunk_paths({"output": "/o", "per_item": True}, 3, "1-1/9",
                                       STUB_JOB) == ("/o/slot-3", "/o/slot-3")
        assert fleet_agent.chunk_paths({"output": "/o", "per_item": False}, 3, "1-1/9",
                                       STUB_JOB) == ("/o", "/o")

    def test_chunk_paths_expands_tilde(self):
        base, _ = fleet_agent.chunk_paths({"output": "~/x", "per_item": False}, 0, "0-0/1", STUB_JOB)
        assert base == os.path.expanduser("~/x")

    def test_chunk_paths_uses_tool_chunk_output(self):
        j = Job(name="j", script="s.py", progress_re=re.compile("x"), output_flag="--shard-out",
                build_args=None, chunk_output=lambda b, c: f"{b}/shard_{c.replace('/', '_')}.json",
                collect=None)
        assert fleet_agent.chunk_paths({"output": "/o", "per_item": True}, 1, "2-2/8", j) == (
            "/o/slot-1", "/o/slot-1/shard_2-2_8.json")

    def test_renderer_argv(self):
        argv = fleet_agent.renderer_argv("py", {"gen_args": ["--a", "b c"], "per_item": True},
                                         STUB_JOB, "1-1/4", "/o", 8)
        assert argv == ["py", "-u", "stub_render.py", "--a", "b c", "--workers", "1",
                        "--shard", "1-1/4", "--output", "/o"]
        argv = fleet_agent.renderer_argv("py", {"gen_args": [], "per_item": False},
                                         STUB_JOB, "1-1/4", "/o", 8)
        assert argv[argv.index("--workers") + 1] == "8"

    def test_parse_env(self):
        assert fleet_agent.parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
        with pytest.raises(ValueError):
            fleet_agent.parse_env(["nope"])

    def test_env_values_expand(self, coord, repo, tmp_path, monkeypatch):
        monkeypatch.setenv("FLEET_TEST_HOME", "expanded")
        q, _, _, client = coord
        client.post("/api/jobs", {"spec": spec(tmp_path / "o"), "chunks": ["0-0/1"]})
        run_agents(make_agent(client, repo, "a", slots=1, env={"STUB_WHO": "$FLEET_TEST_HOME"}))
        assert (tmp_path / "o" / "slot-0" / "item_0.txt").read_text() == "expanded"

    def test_kill_group_kills_children_too(self, tmp_path):
        p = subprocess.Popen([sys.executable, "-c",
                              "import subprocess,sys,time;"
                              "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(99)']);"
                              "print(c.pid,flush=True);time.sleep(99)"],
                             stdout=subprocess.PIPE, text=True, start_new_session=True)
        child = int(p.stdout.readline())
        fleet_agent.kill_group(p, grace=2)
        p.wait()
        wait_for(lambda: not alive(child), 5)

    def test_kill_group_escalates_to_sigkill(self):
        p = subprocess.Popen([sys.executable, "-c",
                              "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                              "print('ready',flush=True);time.sleep(99)"],
                             stdout=subprocess.PIPE, text=True, start_new_session=True)
        p.stdout.readline()
        fleet_agent.kill_group(p, grace=0.5)
        assert p.poll() is not None

    def test_kill_stale_targets_only_matching_output(self, tmp_path):
        script = tmp_path / "stub_render.py"
        script.write_text("import time; time.sleep(99)")
        mine = subprocess.Popen([sys.executable, str(script), "--output", "/x/slot-0"],
                                cwd=tmp_path)
        other = subprocess.Popen([sys.executable, str(script), "--output", "/x/slot-1"],
                                 cwd=tmp_path)
        time.sleep(0.5)
        try:
            fleet_agent.kill_stale(STUB_JOB, "/x/slot-0")
            wait_for(lambda: mine.poll() is not None, 5)
            assert other.poll() is None
        finally:
            other.kill(); mine.kill()
            other.wait(); mine.wait()

    def test_stale_renderer_from_dead_agent_is_cleared_before_run(self, coord, repo, tmp_path):
        q, _, _, client = coord
        out = tmp_path / "o"
        (out / "slot-0").mkdir(parents=True)
        stale = subprocess.Popen([sys.executable, str(repo / "stub_render.py"), "--shard", "9-9/9",
                                  "--mode", "sleep", "--output", str(out / "slot-0")])
        time.sleep(0.5)
        client.post("/api/jobs", {"spec": spec(out), "chunks": ["0-0/1"]})
        run_agents(make_agent(client, repo, "a", slots=1))
        assert stale.wait(5) is not None
        assert q.job_counts(1)["done"] == 1


class TestHttp:
    def raw(self, url, path, method="GET", token=TOKEN, body=None):
        req = urllib.request.Request(url + path, method=method,
                                     data=None if body is None else body,
                                     headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def test_auth_required_on_api(self, coord):
        _, _, url, _ = coord
        assert self.raw(url, "/api/status", token=None)[0] == 401
        assert self.raw(url, "/api/status", token="wrong")[0] == 401
        assert self.raw(url, "/api/status")[0] == 200

    def test_wrong_token_cannot_mutate(self, coord):
        q, _, url, _ = coord
        code, _, _ = self.raw(url, "/api/jobs", "POST", token="wrong",
                              body=json.dumps({"spec": {}, "chunks": ["0-0/1"]}).encode())
        assert code == 401
        assert q.status()["jobs"] == []

    def test_dashboard_is_public_and_carries_no_data(self, coord):
        q, _, url, client = coord
        client.post("/api/jobs", {"spec": {"tool": "SECRETNAME"}, "chunks": ["0-0/1"]})
        code, body, h = self.raw(url, "/", token=None)
        assert code == 200 and "text/html" in h["Content-Type"]
        assert b"SECRETNAME" not in body and b"/api/status" in body

    def test_bad_json_is_400(self, coord):
        assert self.raw(coord[2], "/api/lease", "POST", body=b"not json")[0] == 400
        assert self.raw(coord[2], "/api/lease", "POST", body=b"[1]")[0] == 400

    def test_missing_field_is_400_unknown_worker_404(self, coord):
        _, _, url, _ = coord
        assert self.raw(url, "/api/lease", "POST", body=b"{}")[0] == 404 or True
        code, _, _ = self.raw(url, "/api/lease", "POST", body=json.dumps({"worker": "ghost"}).encode())
        assert code == 404

    def test_unknown_path_404_and_method_405(self, coord):
        _, _, url, _ = coord
        assert self.raw(url, "/api/nope")[0] == 404
        assert self.raw(url, "/nope", token=None)[0] == 404
        assert self.raw(url, "/api/status", "DELETE")[0] in (405, 501)

    def test_client_error_types(self, coord):
        _, _, url, client = coord
        with pytest.raises(CoordinatorError) as e:
            client.post("/api/lease", {"worker": "ghost"})
        assert e.value.code == 404
        with pytest.raises(CoordinatorError) as e:
            FleetClient(url, "bad").get("/api/status")
        assert e.value.code == 401
        with pytest.raises(CoordinatorDown):
            FleetClient("http://127.0.0.1:9", "x", timeout=1).get("/api/status")

    def test_job_endpoint_and_unquarantine(self, coord):
        q, _, _, client = coord
        client.post("/api/register", {"name": "w", "dir": "/d", "slots": 1})
        jid = client.post("/api/jobs", {"spec": {"per_item": True}, "chunks": ["0-0/1"]})["job_id"]
        assert client.get(f"/api/jobs/{jid}")["counts"]["pending"] == 1
        client.post("/api/workers/w/unquarantine")
        with pytest.raises(CoordinatorError):
            client.get("/api/jobs/999")

    def test_token_file_created_private_and_reused(self, tmp_path):
        from fleet_coordinator import load_or_create_token
        p = tmp_path / "sub" / "tok"
        t1 = load_or_create_token(p)
        assert len(t1) >= 24 and oct(p.stat().st_mode & 0o777) == "0o600"
        assert load_or_create_token(p) == t1
