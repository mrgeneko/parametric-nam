"""fleet_ctl: job building, status text, agent start/stop commands, and a REAL localhost
end to end -- coordinator, agent, stub renderer writing genuine shard files, then
`submit --wait --collect` through the same run_collect the push scheduler uses."""
import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import distribute_pull as dp
import fleet_ctl
import ssh_target
from fleet_agent import Agent
from fleet_client import FleetClient
from fleet_coordinator import serve_in_thread
from fleet_queue import FleetQueue

TOKEN = "tok"
HERE = Path(__file__).resolve().parent.parent

STUB_GEN = textwrap.dedent('''
    import argparse, json, os, sys
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard"); ap.add_argument("--output"); ap.add_argument("--workers")
    ap.add_argument("--range", action="append", default=[])
    a = ap.parse_args()
    i, n = int(a.shard.split("-")[0]), int(a.shard.split("/")[1])
    out = a.output
    os.makedirs(os.path.join(out, "sig"), exist_ok=True)
    np.save(os.path.join(out, "sig", f"{i}.npy"), np.full(64, 0.5 + i, dtype=np.float32))
    csv = os.path.join(out, "params.csv")
    new = not os.path.exists(csv)
    with open(csv, "a") as f:
        if new:
            f.write("idx,ok,peak,error\\n")
        f.write(f"{i},1,{0.5 + i},\\n")
    open(os.path.join(out, "config.json"), "w").write("{}")
    print(f"[{i+1:3d}/{n:3d}]  {100*(i+1)/n:.1f}%  combo_{i}  OK", flush=True)
''')


def ns(**kw):
    d = dict(tool="gen_dataset", config=None, output="/o", chunks=64, chunk_size=None, items=None,
             retries=2, label=None, skip_gate_check=True, require_gate=False,
             skip_version_check=True)
    d.update(kw)
    return argparse.Namespace(**d)


class TestBuildJobSpec:
    def test_per_item_from_range(self):
        spec, chunks = fleet_ctl.build_job_spec(
            ns(chunk_size=1), ["--range", "A=1,2,3", "--range", "B=1,2", "--backend", "x"])
        assert len(chunks) == 6 and chunks[0] == "0-0/6" and chunks[-1] == "5-5/6"
        assert spec["per_item"] and spec["item_count"] == 6 and spec["output"] == "/o"
        assert spec["gen_args"][:2] == ["--range", "A=1,2,3"]
        assert spec["extra_args"] == spec["gen_args"]      # no --config: nothing expanded

    def test_legacy_chunks(self):
        spec, chunks = fleet_ctl.build_job_spec(ns(chunks=4), ["--range", "A=1,2"])
        assert chunks == ["0-0/4", "1-1/4", "2-2/4", "3-3/4"]
        assert not spec["per_item"] and spec["item_count"] is None

    def test_items_override(self):
        spec, chunks = fleet_ctl.build_job_spec(ns(chunk_size=1, items=5), ["--x", "1"])
        assert len(chunks) == 5

    def test_chunk_size_other_than_1_refused(self):
        with pytest.raises(SystemExit):
            fleet_ctl.build_job_spec(ns(chunk_size=4), ["--range", "A=1,2"])

    def test_no_grid_is_refused_rather_than_guessed(self):
        with pytest.raises(SystemExit):
            fleet_ctl.build_job_spec(ns(chunk_size=1), ["--x", "1"])

    def test_nothing_to_run_is_refused(self):
        with pytest.raises(SystemExit):
            fleet_ctl.build_job_spec(ns(), [])

    def test_pins_this_checkouts_sha_unless_skipped(self):
        sha = dp.local_commit_sha()
        spec, _ = fleet_ctl.build_job_spec(ns(skip_version_check=False), ["--range", "A=1"])
        assert spec["sha"] == sha
        spec, _ = fleet_ctl.build_job_spec(ns(skip_version_check=True), ["--range", "A=1"])
        assert "sha" not in spec

    def test_config_is_expanded_like_distribute_pull(self, tmp_path, monkeypatch):
        cfg = tmp_path / "c.toml"
        cfg.write_text("")
        monkeypatch.setitem(dp.JOBS, "gen_dataset", dataclasses.replace(
            dp.JOBS["gen_dataset"], build_args=lambda c, root, extra: ["--from-config"] + extra))
        spec, _ = fleet_ctl.build_job_spec(ns(config=cfg, chunks=2), ["--tail", "1"])
        assert spec["gen_args"] == ["--from-config", "--tail", "1"]
        assert spec["extra_args"] == ["--tail", "1"]             # raw form, for collect
        assert spec["config"] == str(cfg.resolve())

    def test_gate_abort_exits(self, tmp_path, monkeypatch):
        cfg = tmp_path / "c.toml"
        cfg.write_text("")
        monkeypatch.setitem(dp.JOBS, "gen_dataset", dataclasses.replace(
            dp.JOBS["gen_dataset"], build_args=lambda c, root, extra: ["--a"]))
        import gate_config
        monkeypatch.setattr(gate_config, "verify_gate", lambda c: (False, "stale"))
        with pytest.raises(SystemExit) as e:
            fleet_ctl.build_job_spec(ns(config=cfg, skip_gate_check=False, require_gate=True), [])
        assert e.value.code == 2
        # warn-only by default: no exit
        fleet_ctl.build_job_spec(ns(config=cfg, skip_gate_check=False, require_gate=False), [])


class TestStatusText:
    def status(self):
        return {"now": 1000.0, "jobs": [{"id": 3, "tool": "gen_dataset", "output": "/o",
                "counts": {"pending": 1, "leased": 2, "done": 5, "failed": 0, "total": 8},
                "cancelled": False, "finished": False, "elapsed": 7200.0}],
                "workers": [{"name": "w1", "slots": 4, "busy": 2, "done": 5, "combos": 5,
                             "chunks_per_hour": 2.5, "failed": 0, "quarantined": False,
                             "online": True, "last_seen_age": 4.0},
                            {"name": "w2", "slots": 4, "busy": 0, "done": 0, "combos": 0,
                             "chunks_per_hour": 0, "failed": 3, "quarantined": True,
                             "online": True, "last_seen_age": 6000.0}],
                "inflight": [{"job_id": 3, "spec": "6-6/8", "worker": "w1", "slot": 1,
                              "attempts": 1, "progress": 2, "started": 900.0}],
                "problems": [{"job_id": 3, "spec": "2-2/8", "state": "pending", "attempts": 1,
                              "error": "line1\nImportError: no spicelib"}]}

    def test_renders_everything_an_operator_needs(self):
        t = fleet_ctl.render_status(self.status())
        assert "job 3" in t and "5/8 done" in t and "2 running" in t and "running]" in t
        assert "QUARANTINED" in t and "w2" in t and "2/4" in t
        assert "chunk 6-6/8 on w1 slot 1" in t
        assert "ImportError: no spicelib" in t

    def test_states(self):
        s = self.status()
        s["jobs"][0].update(finished=True)
        assert "[complete]" in fleet_ctl.render_status(s)
        s["jobs"][0]["counts"]["failed"] = 1
        assert "finished with failures" in fleet_ctl.render_status(s)
        s["jobs"][0]["cancelled"] = True
        assert "[cancelled]" in fleet_ctl.render_status(s)
        assert "no jobs" in fleet_ctl.render_status({**s, "jobs": []})


class TestAgentCommands:
    @pytest.fixture
    def fake_repo(self, tmp_path):
        d = tmp_path / "repo dir"           # a space: the dir must survive the remote shell
        d.mkdir()
        (d / "fleet_agent.py").write_text(
            "import sys,time\nopen('argv.txt','w').write(repr(sys.argv[1:]))\ntime.sleep(60)\n")
        return d

    def run(self, cmd, cwd):
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=cwd,
                              timeout=30)

    def test_start_is_idempotent_and_records_pid(self, fake_repo):
        cmd = fleet_ctl.agent_start_command(str(fake_repo).replace(" ", "\\ "), "http://h:1", "w1", 3,
                                            ["DOTNET_ROOT=$HOME/.dotnet"], python=sys.executable)
        r = self.run(cmd, "/")
        assert "agent started" in r.stdout, r
        pid = int((fake_repo / ".fleet_agent.pid").read_text())
        try:
            time.sleep(1)
            argv = eval((fake_repo / "argv.txt").read_text())
            assert argv[argv.index("--name") + 1] == "w1"
            assert argv[argv.index("--slots") + 1] == "3"
            assert argv[argv.index("--coordinator") + 1] == "http://h:1"
            assert argv[argv.index("--env") + 1] == "DOTNET_ROOT=$HOME/.dotnet"   # unexpanded
            assert "--token-file" in argv and "s3cret" not in " ".join(argv)
            again = self.run(cmd, "/")
            assert "already running" in again.stdout
            assert (fake_repo / ".fleet_agent.pid").read_text().strip() == str(pid)
        finally:
            os.kill(pid, 9)

    def test_stale_pidfile_does_not_block_a_start(self, fake_repo):
        (fake_repo / ".fleet_agent.pid").write_text("999999")
        cmd = fleet_ctl.agent_start_command(str(fake_repo).replace(" ", "\\ "), "u", "w", 1, [],
                                            python=sys.executable)
        assert "agent started" in self.run(cmd, "/").stdout
        os.kill(int((fake_repo / ".fleet_agent.pid").read_text()), 9)

    def test_stop_only_kills_a_real_agent(self, fake_repo):
        d = str(fake_repo).replace(" ", "\\ ")
        start = fleet_ctl.agent_start_command(d, "u", "w", 1, [], python=sys.executable)
        self.run(start, "/")
        pid = int((fake_repo / ".fleet_agent.pid").read_text())
        time.sleep(0.5)
        assert "stopped" in self.run(fleet_ctl.agent_stop_command(d), "/").stdout
        time.sleep(0.5)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not (fake_repo / ".fleet_agent.pid").exists()

    def test_stop_refuses_a_recycled_pid(self, fake_repo):
        other = subprocess.Popen(["sleep", "30"])
        try:
            (fake_repo / ".fleet_agent.pid").write_text(str(other.pid))
            out = self.run(fleet_ctl.agent_stop_command(str(fake_repo).replace(" ", "\\ ")), "/")
            assert "no agent running" in out.stdout
            assert other.poll() is None
        finally:
            other.kill(); other.wait()

    def test_token_command_writes_private_file(self, tmp_path):
        env = {**os.environ, "HOME": str(tmp_path)}
        r = subprocess.run(["bash", "-c", fleet_ctl.agent_token_command()], input="abc\n",
                           text=True, env=env, capture_output=True)
        p = tmp_path / ".config/parametric-nam/fleet-agent.token"
        assert r.returncode == 0 and p.read_text() == "abc\n"
        assert oct(p.stat().st_mode & 0o777) == "0o600"


def _ssh_ok():
    try:
        return shutil.which("rsync") and subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "localhost", "true"],
            capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _ssh_ok(), reason="needs passwordless `ssh localhost` + rsync")
class TestRealEndToEnd:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        inv = tmp_path / "fleet.toml"
        inv.write_text('[hosts."fw"]\naddress = "localhost"\n[hosts."fs"]\naddress = "localhost"\n')
        ssh_target.configure(inv, cache_dir=tmp_path / "sshcfg")
        base = Path.home() / f".fleet_e2e_{os.getpid()}"
        base.mkdir()
        repo = base / "repo"
        repo.mkdir()
        (repo / "stub_gen.py").write_text(STUB_GEN)
        job = dataclasses.replace(dp.GEN_DATASET_JOB, name="stubgen", script="stub_gen.py")
        monkeypatch.setitem(dp.JOBS, "stubgen", job)
        q = FleetQueue(tmp_path / "q.db")
        srv, url = serve_in_thread(q, TOKEN)
        tokfile = tmp_path / "tok"
        tokfile.write_text(TOKEN)
        yield type("E", (), dict(base=base, repo=repo, q=q, url=url, tok=tokfile, inv=inv,
                                 tmp=tmp_path))
        srv.shutdown(); srv.server_close(); q.close()
        shutil.rmtree(base, ignore_errors=True)
        ssh_target.reset()

    def agent(self, env, name="fw", slots=2):
        a = Agent(FleetClient(env.url, TOKEN), name, slots, 2, repo_dir=env.repo, poll_s=0.1,
                  heartbeat_s=0.3, check_version=False, jobs={"stubgen": dp.JOBS["stubgen"]})
        t = threading.Thread(target=a.run, daemon=True)
        t.start()
        return a, t

    def submit_argv(self, env, out, *more):
        return ["submit", "--url", env.url, "--token-file", str(env.tok), "--tool", "stubgen",
                "--output", str(out), "--chunk-size", "1", "--skip-version-check",
                "--skip-gate-check", "--poll-s", "0.2", "--inventory", str(env.inv), *more,
                "--", "--range", "A=1,2", "--range", "B=1,2"]

    def test_submit_wait_collect_local(self, env, tmp_path):
        a, t = self.agent(env)
        dest = tmp_path / "collected"
        rc = fleet_ctl.main(self.submit_argv(env, env.base / "out", "--collect", str(dest),
                                             "--no-combine"))
        a.stop.set(); t.join(20)
        assert rc == 0
        rows = (dest / "params.csv").read_text().splitlines()
        assert rows[0] == "idx,ok,peak,error"
        assert sorted(int(r.split(",")[0]) for r in rows[1:]) == [0, 1, 2, 3]
        assert sorted(p.name for p in (dest / "sig").iterdir()) == [f"{i}.npy" for i in range(4)]
        np.testing.assert_allclose(np.load(dest / "sig" / "2.npy"), 2.5)

    def test_collect_to_a_sink_host(self, env, tmp_path):
        a, t = self.agent(env)
        sink = env.base / "sink"
        rc = fleet_ctl.main(self.submit_argv(env, env.base / "out", "--collect", f"fs:{sink}",
                                             "--no-combine"))
        a.stop.set(); t.join(20)
        assert rc == 0
        assert len((sink / "params.csv").read_text().splitlines()) == 5
        assert len(list((sink / "sig").iterdir())) == 4

    def test_collect_is_refused_until_the_job_finishes(self, env, tmp_path, capsys):
        client = FleetClient(env.url, TOKEN)
        spec, chunks = fleet_ctl.build_job_spec(
            ns(tool="stubgen", chunk_size=1, output=str(env.base / "out")),
            ["--range", "A=1,2"])
        jid = client.post("/api/jobs", {"spec": spec, "chunks": chunks})["job_id"]
        args = argparse.Namespace(partial=False, inventory=None, no_combine=True,
                                  repair_missing=False, sink_repo=None, no_direct_sink=False)
        with pytest.raises(SystemExit) as e:
            fleet_ctl.do_collect(client, jid, args, str(tmp_path / "d"))
        assert "not finished" in str(e.value)

    def test_status_and_cancel_commands(self, env, capsys):
        client = FleetClient(env.url, TOKEN)
        spec, chunks = fleet_ctl.build_job_spec(
            ns(tool="stubgen", chunk_size=1, output="/o"), ["--range", "A=1,2"])
        jid = client.post("/api/jobs", {"spec": spec, "chunks": chunks})["job_id"]
        conn = ["--url", env.url, "--token-file", str(env.tok)]
        assert fleet_ctl.main(["status", *conn]) == 0
        assert f"job {jid}" in capsys.readouterr().out
        assert fleet_ctl.main(["cancel", str(jid), *conn]) == 0
        assert client.get(f"/api/jobs/{jid}")["finished"]
        client.post("/api/register", {"name": "w", "dir": "/d", "slots": 1})
        assert fleet_ctl.main(["unquarantine", "w", *conn]) == 0

    def test_start_and_stop_agents_over_ssh(self, env, monkeypatch, capsys):
        (env.repo / "fleet_agent.py").write_text("import time\ntime.sleep(60)\n")
        tokdst = env.base / "installed.token"
        monkeypatch.setattr(fleet_ctl, "agent_token_command",
                            lambda: f"cat > {tokdst}")     # never touch this machine's real token
        (env.tmp / "agent.tok").write_text("s3cret\n")
        common = ["--inventory", str(env.inv)]
        assert fleet_ctl.main(["start-agents", "--worker", f"fw:{env.repo}:2",
                               "--agent-url", "http://c:1", "--agent-token-file",
                               str(env.tmp / "agent.tok"), "--skip-version-check", *common]) == 0
        pid = int((env.repo / ".fleet_agent.pid").read_text())
        try:
            assert tokdst.read_text() == "s3cret\n"
            os.kill(pid, 0)
        finally:
            assert fleet_ctl.main(["stop-agents", "--worker", f"fw:{env.repo}", *common]) == 0
        time.sleep(0.5)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
