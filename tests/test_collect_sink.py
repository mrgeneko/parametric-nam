"""distribute_pull --collect HOST:DIR (roadmap item 10): merge the shards on the machine that
will train. Unit tests with faked ssh, plus REAL transfers through localhost ssh using two
inventory aliases that both resolve to localhost (skipped without passwordless `ssh localhost`)."""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import distribute_pull as dp  # noqa: E402
import ssh_target  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def _cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


class TestParseCollectDest:
    @pytest.mark.parametrize("arg,expected", [
        ("trainer:/data/ds", ("trainer", "/data/ds")),
        ("trainer:~/ds", ("trainer", "~/ds")),
        ("mac-1.local:ds", ("mac-1.local", "ds")),
        ("/abs/dir", (None, "/abs/dir")),
        ("~/ds", (None, "~/ds")),
        ("./data:v2", (None, "./data:v2")),
        ("relative/dir", (None, "relative/dir")),
        ("plain", (None, "plain")),
        ("host:", (None, "host:")),
    ])
    def test_cases(self, arg, expected):
        assert dp.parse_collect_dest(arg) == expected


class TestSinkFromInventory:
    def test_target_port_repo_from_inventory(self):
        inv = {"t": {"address": "10.0.0.5", "user": "chewie", "port": 2222, "repo": "~/r"}}
        s = dp.sink_from_inventory("t", "/d", inv)
        assert (s.target, s.port, s.repo) == ("chewie@10.0.0.5", 2222, "~/r")

    def test_explicit_repo_wins_and_defaults(self):
        s = dp.sink_from_inventory("t", "/d", {"t": {"repo": "~/r"}}, repo="~/other")
        assert (s.target, s.port, s.repo) == ("t", None, "~/other")

    def test_unknown_host_and_port_22(self):
        assert dp.sink_from_inventory("x", "/d", {}).target == "x"
        assert dp.sink_from_inventory("x", "/d", {"x": {"port": 22}}).port is None


class TestBuildSink:
    class Wk:
        def __init__(self, host, dir): self.host, self.dir = host, dir

    @pytest.fixture
    def inv(self, tmp_path):
        f = tmp_path / "inv.toml"
        f.write_text('[hosts."t"]\naddress = "10.0.0.5"\nuser = "u"\nrepo = "~/inv-repo"\n')
        return str(f)

    def test_flag_beats_worker_beats_inventory(self, inv):
        w = [self.Wk("t", "~/worker-dir")]
        assert dp.build_sink("t", "/d", inv, "~/flag", w).repo == "~/flag"
        assert dp.build_sink("t", "/d", inv, None, w).repo == "~/worker-dir"
        assert dp.build_sink("t", "/d", inv, None, []).repo == "~/inv-repo"

    def test_inventory_only_consulted_when_given(self, inv):
        s = dp.build_sink("t", "/d", None, None, [])
        assert (s.repo, s.target) == (None, "t")

    def test_inventory_gives_the_workers_address(self, inv):
        assert dp.build_sink("t", "/d", inv, None, []).target == "u@10.0.0.5"


class TestCheckExpected:
    def test_ok(self):
        assert dp._check_expected([1, 2], [1, 2], [4, 4], 2) is True

    def test_count_mismatch(self):
        assert dp._check_expected([1], [1], [4], 2) is False

    def test_size_mismatch(self):
        assert dp._check_expected([1, 2], [1, 2], [4, 5], 2) is False


class TestListSinkNpys:
    def test_parses_wc_with_total_lines(self, monkeypatch):
        out = "  400 sig/0/3.npy\n  400 sig/0/1.npy\n  800 total\n  400 sig/2.npy\n 1200 total\n"
        monkeypatch.setattr(dp, "_ssh_run", lambda h, c, t: _cp(0, out))
        assert dp.list_sink_npys("h", "/d") == ([1, 2, 3], [400, 400, 400])

    def test_no_sig_dir_is_empty_not_error(self, monkeypatch):
        monkeypatch.setattr(dp, "_ssh_run", lambda h, c, t: _cp(0, ""))
        assert dp.list_sink_npys("h", "/d") == ([], [])

    def test_ssh_failure_is_none(self, monkeypatch):
        monkeypatch.setattr(dp, "_ssh_run", lambda h, c, t: _cp(255, "", "refused"))
        assert dp.list_sink_npys("h", "/d") is None


class TestCombineCommand:
    def test_uses_sink_checkout_and_dir(self):
        s = dp.Sink("h", "/data/my ds", repo="~/repo")
        assert dp.combine_command(s) == ("cd ~/repo && ./.venv/bin/python -u gen_dataset_from_schx.py "
                                         "--combine '/data/my ds'")

    def test_no_repo_no_command(self):
        assert dp.combine_command(dp.Sink("h", "/d")) is None


class TestTransferCommands:
    """Which mechanism is chosen, and the exact commands, with ssh/tar faked."""

    @pytest.fixture
    def rec(self, monkeypatch):
        calls = []
        rc = {"direct": 0, "local": 0}

        def fake_ssh(host, cmd, timeout):
            calls.append((host, cmd))
            if cmd.startswith("rsync") and "-e" in cmd:
                return _cp(rc["direct"], "", "no route")
            if cmd.startswith("rsync"):
                return _cp(rc["local"])
            return _cp()
        monkeypatch.setattr(dp, "_ssh_run", fake_ssh)
        popen = []

        class FakeTar:
            returncode = 0
            stdout = type("S", (), {"close": lambda self: None})()
            stderr = type("E", (), {"read": lambda self: b""})()
            def wait(self): return 0
        monkeypatch.setattr(dp.subprocess, "Popen",
                            lambda argv, **kw: popen.append(list(argv)) or FakeTar())
        monkeypatch.setattr(dp, "_ssh_run_stdin",
                            lambda host, cmd, stdin, timeout: calls.append((host, cmd)) or _cp())
        return calls, popen, rc

    def test_already_in_place_does_nothing(self, rec):
        calls, popen, _ = rec
        s = dp.Sink("h", "/d")
        assert dp.transfer_shard_to_sink("h", "/d/", s, "/d") == (True, "already in place")
        assert calls == [] and popen == []

    def test_same_host_copies_locally_without_params(self, rec):
        calls, popen, _ = rec
        ok, how = dp.transfer_shard_to_sink("h", "/o", dp.Sink("h", "/d"), "/d")
        assert ok and "local" in how and popen == []
        assert calls == [("h", "rsync -a --exclude=/params.csv /o/ /d/")]

    def test_direct_runs_on_the_worker_and_addresses_the_sink(self, rec):
        calls, popen, _ = rec
        s = dp.Sink("sink", "/d", target="chewie@10.0.0.5", port=2222)
        ok, how = dp.transfer_shard_to_sink("w", "/o", s, "/d")
        assert ok and how == "direct worker -> sink" and popen == []
        host, cmd = calls[0]
        assert host == "w"
        assert cmd.startswith("rsync -a --exclude=/params.csv -e ")
        assert "-p 2222" in cmd and cmd.endswith("/o/ chewie@10.0.0.5:/d/")

    def test_direct_failure_falls_back_to_relay(self, rec):
        calls, popen, r = rec
        r["direct"] = 255
        ok, how = dp.transfer_shard_to_sink("w", "/o", dp.Sink("sink", "/d"), "/d")
        assert ok and "relayed" in how
        assert "tar --exclude=params.csv -cf -" in popen[0][-1]
        assert calls[-1] == ("sink", "tar -C /d -xf -")

    def test_no_direct_flag_relays_immediately(self, rec):
        calls, popen, _ = rec
        ok, how = dp.transfer_shard_to_sink("w", "/o", dp.Sink("sink", "/d"), "/d", direct=False)
        assert "relayed" in how and not any(c[1].startswith("rsync") for c in calls)

    def test_unsafe_path_skips_direct(self, rec):
        calls, popen, _ = rec
        ok, how = dp.transfer_shard_to_sink("w", "/o", dp.Sink("sink", "/my d"), "/my d")
        assert "relayed" in how and not any(c[1].startswith("rsync") for c in calls)
        assert calls[-1] == ("sink", "tar -C '/my d' -xf -")

    def test_relay_reports_remote_tar_failure(self, monkeypatch):
        monkeypatch.setattr(dp.subprocess, "Popen", lambda argv, **kw: type("T", (), {
            "returncode": 0, "stdout": type("S", (), {"close": lambda s: None})(),
            "stderr": type("E", (), {"read": lambda s: b""})(), "wait": lambda s: 0})())
        monkeypatch.setattr(dp, "_ssh_run_stdin", lambda *a: _cp(2, "", "tar: disk full"))
        ok, how = dp.transfer_shard_to_sink("w", "/o", dp.Sink("sink", "/d"), "/d", direct=False)
        assert ok is False and "disk full" in how


class TestCli:
    def _run(self, tmp_path, *extra):
        return subprocess.run([sys.executable, str(HERE / "distribute_pull.py"),
                               "--worker", "nope:/r:1", "--output", "/o", *extra],
                              capture_output=True, text=True, timeout=60)

    def test_sink_only_for_gen_dataset(self, tmp_path):
        r = self._run(tmp_path, "--tool", "grid_adequacy", "--collect", "trainer:/d")
        assert r.returncode == 2 and "only supported for --tool gen_dataset" in r.stderr

    def test_repair_missing_rejected_with_sink(self, tmp_path):
        r = self._run(tmp_path, "--collect", "trainer:/d", "--repair-missing")
        assert r.returncode == 2 and "--repair-missing" in r.stderr

    def test_flags_documented(self):
        r = subprocess.run([sys.executable, str(HERE / "distribute_pull.py"), "--help"],
                           capture_output=True, text=True, timeout=30)
        assert "--sink-repo" in r.stdout and "--no-direct-sink" in r.stdout and "[HOST:]DIR" in r.stdout


def _localhost_ssh_ok() -> bool:
    if shutil.which("ssh") is None or shutil.which("rsync") is None:
        return False
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                               "localhost", "true"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


class W:
    def __init__(self, host):
        self.host = host


def _make_shard(d: Path, indices, n=64, bad_csv=()):
    (d / "sig").mkdir(parents=True, exist_ok=True)
    for i in indices:
        np.save(d / "sig" / f"{i}.npy", np.full(n, 0.5 + i, dtype=np.float32))
    with open(d / "params.csv", "w") as fh:
        fh.write("idx,ok,peak,error\n")
        for i in indices:
            if i not in bad_csv:
                fh.write(f"{i},1,{0.5 + i},\n")
    (d / "config.json").write_text("{}")


@pytest.mark.skipif(not _localhost_ssh_ok(), reason="needs passwordless `ssh localhost` + rsync")
class TestRealSinkCollect:
    """Two inventory aliases, `fw` (worker) and `fs` (sink), both resolving to localhost, so
    worker != sink by NAME (the relay/direct code paths) while every byte is real."""

    @pytest.fixture
    def env(self, tmp_path):
        inv = tmp_path / "fleet.toml"
        inv.write_text(textwrap.dedent('''
            [hosts."fw"]
            address = "localhost"
            [hosts."fs"]
            address = "localhost"
            [hosts."fw2"]
            address = "localhost"
        '''))
        ssh_target.configure(inv, cache_dir=tmp_path / "sshcfg")
        base = Path.home() / f".sink_test_{os.getpid()}"
        base.mkdir()
        yield type("E", (), {"base": base, "tmp": tmp_path})
        shutil.rmtree(base, ignore_errors=True)

    def _sink(self, env, **kw):
        return dp.Sink("fs", str(env.base / "sink"), target="localhost", **kw)

    def _two_shards(self, env):
        a, b = env.base / "wa", env.base / "wb"
        _make_shard(a, [0, 2])
        _make_shard(b, [1, 3])
        return [W("fw"), W("fw2")], [str(a), str(b)]

    def _assert_merged(self, sink_dir: Path, n=4):
        rows = (sink_dir / "params.csv").read_text().splitlines()
        assert rows[0] == "idx,ok,peak,error" and len(rows) == n + 1
        assert [r.split(",")[0] for r in rows[1:]] == [str(i) for i in range(n)]

    @pytest.mark.parametrize("direct", [True, False])
    def test_merges_on_the_sink_and_nothing_lands_locally(self, env, direct, monkeypatch, tmp_path):
        workers, outs = self._two_shards(env)
        cwd = tmp_path / "cwd"; cwd.mkdir(); monkeypatch.chdir(cwd)
        ok = dp._collect_to_sink(workers, outs, self._sink(env), no_combine=True,
                                 labels=["a", "b"], expected_count=4, direct=direct)
        assert ok is True
        sd = env.base / "sink"
        self._assert_merged(sd)
        assert sorted(p.name for p in (sd / "sig").iterdir()) == ["0.npy", "1.npy", "2.npy", "3.npy"]
        assert (sd / "config.json").exists()
        assert list(cwd.iterdir()) == []

    def test_unreachable_direct_target_falls_back_to_relay(self, env):
        workers, outs = self._two_shards(env)
        sink = dp.Sink("fs", str(env.base / "sink"), target="no-such-host.invalid")
        assert dp._collect_to_sink(workers, outs, sink, no_combine=True, expected_count=4)
        self._assert_merged(env.base / "sink")

    def test_worker_that_is_the_sink_dir_is_not_clobbered(self, env):
        a, b = env.base / "sink", env.base / "wb"
        _make_shard(a, [0, 2])
        _make_shard(b, [1, 3])
        ok = dp._collect_to_sink([W("fs"), W("fw2")], [str(a), str(b)], self._sink(env),
                                 no_combine=True, labels=["a", "b"], expected_count=4)
        assert ok
        self._assert_merged(a)

    def test_same_host_different_dir_is_a_local_copy(self, env):
        a, b = env.base / "wa", env.base / "wb"
        _make_shard(a, [0, 2]); _make_shard(b, [1, 3])
        ok = dp._collect_to_sink([W("fs"), W("fs")], [str(a), str(b)], self._sink(env),
                                 no_combine=True, labels=["a", "b"], expected_count=4)
        assert ok
        self._assert_merged(env.base / "sink")

    def test_name_with_space_on_the_sink(self, env):
        workers, outs = self._two_shards(env)
        sink = dp.Sink("fs", str(env.base / "my sink"), target="localhost")
        assert dp._collect_to_sink(workers, outs, sink, no_combine=True, expected_count=4)
        self._assert_merged(env.base / "my sink")

    def test_missing_row_is_reported_and_not_combined(self, env):
        a, b = env.base / "wa", env.base / "wb"
        _make_shard(a, [0, 2], bad_csv=(2,)); _make_shard(b, [1, 3])
        sink = self._sink(env, repo=str(HERE))
        assert dp._collect_to_sink([W("fw"), W("fw2")], [str(a), str(b)], sink,
                                   labels=["a", "b"]) is False
        assert not (env.base / "sink" / "outputs.npy").exists()

    def test_expected_count_catches_a_never_rendered_item(self, env):
        a = env.base / "wa"
        _make_shard(a, [0, 1])
        assert dp._collect_to_sink([W("fw")], [str(a)], self._sink(env), no_combine=True,
                                   expected_count=3) is False

    def test_failed_shard_transfer_is_a_failure(self, env):
        workers, outs = self._two_shards(env)
        outs[1] = str(env.base / "does-not-exist")
        assert dp._collect_to_sink(workers, outs, self._sink(env), no_combine=True,
                                   direct=False) is False

    def test_combines_on_the_sink(self, env):
        workers, outs = self._two_shards(env)
        sink = self._sink(env, repo=str(HERE))
        assert dp._collect_to_sink(workers, outs, sink, labels=["a", "b"], expected_count=4)
        sd = env.base / "sink"
        arr = np.load(sd / "outputs.npy")
        assert arr.shape == (4, 64)
        assert not (sd / "sig").exists()

    def test_no_repo_means_no_combine_but_success(self, env):
        workers, outs = self._two_shards(env)
        assert dp._collect_to_sink(workers, outs, self._sink(env), labels=["a", "b"])
        assert not (env.base / "sink" / "outputs.npy").exists()

    def test_orphan_row_makes_the_result_inconsistent_even_without_combining(self, env):
        a, b = env.base / "wa", env.base / "wb"
        _make_shard(a, [0, 2], bad_csv=(2,)); _make_shard(b, [1, 3])
        assert dp._collect_to_sink([W("fw"), W("fw2")], [str(a), str(b)], self._sink(env),
                                   no_combine=True, labels=["a", "b"]) is False

    def test_inconsistent_dataset_never_reaches_the_combine_command(self, env, monkeypatch):
        a, b = env.base / "wa", env.base / "wb"
        _make_shard(a, [0, 2], bad_csv=(2,)); _make_shard(b, [1, 3])
        seen = []
        real = dp._ssh_run
        monkeypatch.setattr(dp, "_ssh_run", lambda h, c, t: seen.append(c) or real(h, c, t))
        dp._collect_to_sink([W("fw"), W("fw2")], [str(a), str(b)], self._sink(env, repo=str(HERE)),
                            labels=["a", "b"])
        assert not any("--combine" in c for c in seen)

    def test_failed_combine_is_a_failure(self, env):
        workers, outs = self._two_shards(env)
        sink = self._sink(env, repo=str(env.base / "no-such-checkout"))
        assert dp._collect_to_sink(workers, outs, sink, labels=["a", "b"], expected_count=4) is False

    def test_merged_params_that_cannot_be_written_is_a_failure(self, env, monkeypatch):
        workers, outs = self._two_shards(env)
        monkeypatch.setattr(dp, "sync_path_to_worker", lambda *a, **k: (False, "nope"))
        assert dp._collect_to_sink(workers, outs, self._sink(env), no_combine=True) is False

    def test_no_params_anywhere_merges_nothing(self, env, monkeypatch):
        a = env.base / "wa"
        _make_shard(a, [0])
        (a / "params.csv").unlink()
        pushed = []
        monkeypatch.setattr(dp, "sync_path_to_worker", lambda *x, **k: pushed.append(x) or (True, "ok"))
        assert dp._collect_to_sink([W("fw")], [str(a)], self._sink(env), no_combine=True) is False
        assert pushed == []
