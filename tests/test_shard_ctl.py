"""shard_ctl addresses a shard by its OUTPUT DIRECTORY, never by a pid the operator carries.

Both behaviours pinned here are ones we got wrong by hand during the Mesa Orange gain/master
shard run on 2026-09-12: a kill aimed at a wrapper left the renderer orphaned (ppid=1) and
still writing, and two runs shared one log path so the survivor's output interleaved with the
replacement's and read like a caching bug.
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
CTL = HERE / "shard_ctl.py"


def _run(*a):
    return subprocess.run([sys.executable, str(CTL), *a], capture_output=True, text=True, timeout=60)


class TestShardCtl:

    def test_status_without_a_run_is_not_an_error_story(self, tmp_path):
        r = _run("status", "--output", str(tmp_path / "nope"))
        assert r.returncode == 1
        assert "no run recorded" in r.stdout

    def test_stop_without_a_run_says_so(self, tmp_path):
        r = _run("stop", "--output", str(tmp_path / "nope"))
        assert r.returncode == 1
        assert "nothing to stop" in r.stdout

    def test_start_records_its_own_process_group(self, tmp_path):
        """The whole point: a recorded pgid is what lets stop signal children too."""
        out = tmp_path / "shard0"
        r = _run("start", "--output", str(out), "--", "--help")
        assert r.returncode == 0, r.stderr
        info = json.loads((out / ".run.json").read_text())
        assert info["pgid"] == info["pid"], "launch must be its OWN group leader (setsid)"
        assert info["pgid"] != os.getpgid(0), "must not share the launcher's group"
        time.sleep(2)

    def test_log_path_is_unique_per_run_and_outside_out_dir(self, tmp_path):
        out = tmp_path / "shard1"
        _run("start", "--output", str(out), "--", "--help"); time.sleep(1.5)
        first = json.loads((out / ".run.json").read_text())["log"]
        time.sleep(1.1)   # the stamp has 1s resolution
        _run("start", "--output", str(out), "--", "--help"); time.sleep(1.5)
        second = json.loads((out / ".run.json").read_text())["log"]
        assert first != second, "a second run must never reuse the first run's log path"
        assert out not in Path(first).parents, "logs live beside out_dir, not inside it"
        assert Path(first).exists() and Path(second).exists()

    def test_pid_reuse_guard_refuses_to_signal_a_foreign_process(self, tmp_path):
        """A recycled pgid must not be signalled: that turns `stop` into collateral damage.
        os.getpid() here stands in for a recycled number -- it is alive, but it is not ours."""
        out = tmp_path / "shard2"; out.mkdir()
        (out / ".run.json").write_text(json.dumps({
            "pid": os.getpid(), "pgid": os.getpgid(0), "host": "x", "started": "now",
            "log": "/dev/null", "marker": "gen_dataset_from_schx", "argv": [],
        }))
        r = _run("stop", "--output", str(out))
        assert r.returncode == 0
        assert "gone or recycled" in r.stdout
        assert os.getpid() > 0   # still here: nothing was signalled

    def test_start_refuses_while_a_run_is_live(self, tmp_path):
        out = tmp_path / "shard3"; out.mkdir()
        proc = subprocess.Popen([sys.executable, "-c",
                                 "# gen_dataset_from_schx\nimport time; time.sleep(30)"],
                                start_new_session=True)
        try:
            (out / ".run.json").write_text(json.dumps({
                "pid": proc.pid, "pgid": os.getpgid(proc.pid), "host": "x", "started": "now",
                "log": "/dev/null", "marker": "gen_dataset_from_schx", "argv": [],
            }))
            r = _run("start", "--output", str(out), "--", "--help")
            assert r.returncode != 0
            assert "already running" in r.stderr or "already running" in r.stdout
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)

    def test_stop_kills_the_whole_group_not_just_the_leader(self, tmp_path):
        """The orphan case, reproduced: a leader with a child, stopped via the group."""
        out = tmp_path / "shard4"; out.mkdir()
        script = ("# gen_dataset_from_schx\n"
                  "import subprocess,sys,time; "
                  "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                  "print(c.pid, flush=True); time.sleep(60)")
        proc = subprocess.Popen([sys.executable, "-c", script], start_new_session=True,
                                stdout=subprocess.PIPE, text=True)
        child_pid = int(proc.stdout.readline().strip())
        (out / ".run.json").write_text(json.dumps({
            "pid": proc.pid, "pgid": os.getpgid(proc.pid), "host": "x", "started": "now",
            "log": "/dev/null", "marker": "gen_dataset_from_schx", "argv": [],
        }))
        r = _run("stop", "--output", str(out), "--grace", "5")
        assert r.returncode == 0, r.stdout
        time.sleep(1)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)          # the CHILD must be gone, not orphaned to init
        assert out.exists(), "stop must not delete the output directory"


class TestShardCtlOutputIsObservable:
    """A detached run's log must be readable WHILE it runs, not after it exits.

    stdout to a file is block-buffered. On 2026-09-12 all three Mesa Orange workers sat with
    one-line logs while 20 of 24 combinations completed, and the empty log was misread as the
    coverage gate passing from cache. A launcher for detached work cannot withhold its output.
    """

    def test_renderer_is_launched_unbuffered(self, tmp_path):
        out = tmp_path / "shard5"
        r = _run("start", "--output", str(out), "--", "--help")
        assert r.returncode == 0, r.stderr
        argv = json.loads((out / ".run.json").read_text())["argv"]
        assert "-u" in argv, f"renderer must be launched unbuffered: {argv}"
        assert argv.index("-u") < argv.index(str(HERE / "gen_dataset_from_schx.py")), \
            "-u is an interpreter flag and must precede the script"
        time.sleep(2)

    def test_output_appears_in_the_log_before_the_process_exits(self, tmp_path):
        """The behaviour, not just the flag: write slowly, read mid-run."""
        out = tmp_path / "shard6"; out.mkdir()
        log = tmp_path / "slow.log"
        with open(log, "wb") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 "import time\nfor i in range(20): print('line', i); time.sleep(0.4)"],
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            time.sleep(2.5)
            assert log.read_text().strip(), "unbuffered output must be visible mid-run"
            assert proc.poll() is None, "process should still be running when we read it"
        finally:
            proc.kill(); proc.wait(timeout=10)
