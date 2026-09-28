"""distribute_pull --sync-file: sync_path_to_worker / sync_files, plus REAL transfers through
localhost ssh (skipped when `ssh localhost` isn't available non-interactively)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import distribute_pull as dp  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def _cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


class TestRemoteQuote:
    def test_tilde_prefix_stays_expandable(self):
        assert dp._remote_quote("~/a b/c") == "~/'a b/c'"

    def test_absolute_and_plain_are_quoted(self):
        assert dp._remote_quote("/a b/(c)") == "'/a b/(c)'"
        assert dp._remote_quote("/a/b") == "/a/b"


class TestResolveRemoteDir:
    def test_uses_remote_shells_answer(self, monkeypatch):
        monkeypatch.setattr(dp.subprocess, "run", lambda a, **k: _cp(0, "/home/x/repo\n"))
        assert dp.resolve_remote_dir("h", "~/repo") == "/home/x/repo"

    def test_falls_back_when_ssh_fails(self, monkeypatch):
        monkeypatch.setattr(dp.subprocess, "run", lambda a, **k: _cp(255, "", "refused"))
        assert dp.resolve_remote_dir("h", "~/repo") == "~/repo"

    def test_ignores_stdout_of_a_failed_ssh(self, monkeypatch):
        monkeypatch.setattr(dp.subprocess, "run", lambda a, **k: _cp(255, "banner text\n", "refused"))
        assert dp.resolve_remote_dir("h", "~/repo") == "~/repo"

    def test_falls_back_when_ssh_cannot_start(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("no ssh")
        monkeypatch.setattr(dp.subprocess, "run", boom)
        assert dp.resolve_remote_dir("h", "~/repo") == "~/repo"


class TestSyncPathToWorker:
    def _run(self, monkeypatch, local, rel, *, home="/home/x/repo", rsync_rc=0, mkdir_rc=0):
        calls = []
        def fake(argv, **kw):
            calls.append(list(argv))
            if argv[0] == "rsync":
                return _cp(rsync_rc, "", "rsync boom")
            cmd = argv[-1]
            if cmd.startswith("cd ~ && echo"):
                return _cp(0, home + "\n")
            if cmd.startswith("mkdir"):
                return _cp(mkdir_rc, "", "mkdir boom")
            return _cp()
        monkeypatch.setattr(dp.subprocess, "run", fake)
        return dp.sync_path_to_worker("h", "~/repo", local, rel), calls

    def test_file_lands_at_repo_relative_path_on_the_workers_own_home(self, tmp_path, monkeypatch):
        f = tmp_path / "e.wav"
        f.write_bytes(b"x")
        (ok, detail), calls = self._run(monkeypatch, f, "amps/e.wav")
        assert (ok, detail) == (True, "ok")
        rs = next(c for c in calls if c[0] == "rsync")
        assert rs[-2:] == [str(f), "h:/home/x/repo/amps/e.wav"]
        mk = next(c for c in calls if c[-1].startswith("mkdir"))
        assert mk[-1] == "mkdir -p /home/x/repo/amps"

    def test_directory_contents_land_in_that_directory(self, tmp_path, monkeypatch):
        d = tmp_path / "mods"
        d.mkdir()
        (d / "a.lib").write_text("x")
        (ok, _), calls = self._run(monkeypatch, d, "pedals/mods")
        rs = next(c for c in calls if c[0] == "rsync")
        assert rs[-2:] == [f"{d}/", "h:/home/x/repo/pedals/mods/"]
        assert next(c for c in calls if c[-1].startswith("mkdir"))[-1] == "mkdir -p /home/x/repo/pedals/mods"

    def test_missing_local_path_fails_without_any_ssh(self, tmp_path, monkeypatch):
        (ok, detail), calls = self._run(monkeypatch, tmp_path / "nope", "x")
        assert ok is False and "not found locally" in detail and calls == []

    def test_mkdir_failure_reported_and_no_transfer_attempted(self, tmp_path, monkeypatch):
        f = tmp_path / "e"
        f.write_text("x")
        (ok, detail), calls = self._run(monkeypatch, f, "e", mkdir_rc=1)
        assert ok is False and "mkdir failed" in detail
        assert not any(c[0] == "rsync" for c in calls)

    def test_rsync_failure_reported(self, tmp_path, monkeypatch):
        f = tmp_path / "e"
        f.write_text("x")
        (ok, detail), _ = self._run(monkeypatch, f, "e", rsync_rc=23)
        assert ok is False and "rsync boom" in detail

    def test_unresolvable_home_keeps_tilde_expandable_for_mkdir(self, tmp_path, monkeypatch):
        f = tmp_path / "e"
        f.write_text("x")
        calls = []
        def fake(argv, **kw):
            calls.append(list(argv))
            return _cp(255 if argv[-1].startswith("cd ~") else 0, "", "")
        monkeypatch.setattr(dp.subprocess, "run", fake)
        dp.sync_path_to_worker("h", "~/repo", f, "sub/e")
        assert next(c for c in calls if c[-1].startswith("mkdir"))[-1] == "mkdir -p ~/repo/sub"

    def test_unsafe_name_goes_through_tar_not_rsync(self, tmp_path, monkeypatch):
        f = tmp_path / "Big Muff (v2).schx"
        f.write_text("x")
        ran = []
        def fake_run(argv, **kw):
            ran.append(list(argv))
            return _cp(0, "/home/x/repo\n" if argv[-1].startswith("cd ~") else "")
        class FakeTar:
            returncode = 0
            stdout = type("S", (), {"close": lambda self: None})()
            stderr = type("E", (), {"read": lambda self: b""})()
            def wait(self): return 0
        popen_args = []
        monkeypatch.setattr(dp.subprocess, "run", fake_run)
        monkeypatch.setattr(dp.subprocess, "Popen",
                            lambda argv, **kw: popen_args.append(list(argv)) or FakeTar())
        ok, _ = dp.sync_path_to_worker("h", "~/repo", f, "amps/Big Muff (v2).schx")
        assert ok is True
        assert not any(c[0] == "rsync" for c in ran)
        assert popen_args[0] == ["tar", "-cf", "-", "-C", str(tmp_path), "Big Muff (v2).schx"]
        ssh_extract = ran[-1][-1]
        assert ssh_extract == "tar -C '/home/x/repo/amps' -xf -" or ssh_extract.startswith("tar -C")
        assert "mkdir -p /home/x/repo/amps" == next(c for c in ran if c[-1].startswith("mkdir"))[-1]

    def test_local_tar_failure_is_a_failure_even_if_remote_tar_exits_zero(self, tmp_path, monkeypatch):
        f = tmp_path / "a b.txt"
        f.write_text("x")
        monkeypatch.setattr(dp.subprocess, "run",
                            lambda argv, **kw: _cp(0, "/home/x/repo\n" if argv[-1].startswith("cd ~") else ""))
        class BadTar:
            returncode = 2
            stdout = type("S", (), {"close": lambda self: None})()
            stderr = type("E", (), {"read": lambda self: b"tar: cannot read"})()
            def wait(self): return 2
        monkeypatch.setattr(dp.subprocess, "Popen", lambda argv, **kw: BadTar())
        ok, detail = dp.sync_path_to_worker("h", "~/repo", f, "a b.txt")
        assert ok is False and detail

    def test_ssh_config_is_used_for_every_call(self, tmp_path, monkeypatch):
        import ssh_target
        ssh_target._active = tmp_path / "c.conf"
        f = tmp_path / "e"
        f.write_text("x")
        _, calls = self._run(monkeypatch, f, "e")
        for c in calls:
            if c[0] == "ssh":
                assert c[:3] == ["ssh", "-F", str(tmp_path / "c.conf")]
            else:
                assert c[2:4] == ["-e", f"ssh -F {tmp_path / 'c.conf'}"]


class W:
    def __init__(self, host, d="~/repo"):
        self.host, self.dir = host, d


class TestSyncFiles:
    def test_workers_that_received_everything_are_kept_others_excluded(self, tmp_path, monkeypatch):
        a = tmp_path / "a"; a.write_text("1")
        b = tmp_path / "b"; b.write_text("2")
        def fake(host, d, p, rel, timeout=300.0):
            return (host != "bad" or Path(p).name != "b"), "boom"
        monkeypatch.setattr(dp, "sync_path_to_worker", fake)
        kept = dp.sync_files([W("good"), W("bad")], [str(a), str(b)], repo_root=tmp_path)
        assert [w.host for w in kept] == ["good"]

    def test_all_fail_returns_empty(self, tmp_path, monkeypatch):
        a = tmp_path / "a"; a.write_text("1")
        monkeypatch.setattr(dp, "sync_path_to_worker", lambda *x, **k: (False, "no"))
        assert dp.sync_files([W("h1"), W("h2")], [str(a)], repo_root=tmp_path) == []

    def test_rel_is_relative_to_the_repo_root_and_each_worker_gets_its_own_dir(self, tmp_path, monkeypatch):
        (tmp_path / "amps").mkdir()
        f = tmp_path / "amps" / "e.wav"; f.write_text("1")
        seen = []
        monkeypatch.setattr(dp, "sync_path_to_worker",
                            lambda host, d, p, rel, timeout=300.0: seen.append((host, d, rel)) or (True, "ok"))
        dp.sync_files([W("h1", "~/a"), W("h2", "/opt/b")], [str(f)], repo_root=tmp_path)
        assert seen == [("h1", "~/a", "amps/e.wav"), ("h2", "/opt/b", "amps/e.wav")]


class TestCli:
    def test_flag_documented(self):
        r = subprocess.run([sys.executable, str(HERE / "distribute_pull.py"), "--help"],
                           capture_output=True, text=True, timeout=30)
        assert "--sync-file PATH" in r.stdout

    def test_missing_local_file_is_a_usage_error_before_any_ssh(self, tmp_path):
        r = subprocess.run([sys.executable, str(HERE / "distribute_pull.py"),
                            "--worker", "definitely-not-a-host:/r:1", "--output", str(tmp_path / "o"),
                            "--sync-file",
                            str(tmp_path / "nope.wav"), "--", "--backend", "x"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "--sync-file not found locally" in r.stderr


def _localhost_ssh_ok() -> bool:
    if shutil.which("ssh") is None:
        return False
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                               "localhost", "true"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _localhost_ssh_ok(), reason="needs passwordless `ssh localhost`")
class TestRealTransfer:
    """Real bytes over real ssh to this machine, as a worker whose repo dir is elsewhere."""

    @pytest.fixture
    def worker_dir(self):
        d = Path.home() / f".sync_files_test_{os.getpid()}"
        d.mkdir()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_plain_file_via_rsync(self, tmp_path, worker_dir):
        (tmp_path / "amps").mkdir()
        f = tmp_path / "amps" / "e.wav"; f.write_bytes(b"RIFFdata")
        ok, detail = dp.sync_path_to_worker("localhost", f"~/{worker_dir.name}", f, "amps/e.wav")
        assert ok, detail
        assert (worker_dir / "amps" / "e.wav").read_bytes() == b"RIFFdata"

    def test_name_with_spaces_and_parens_via_tar(self, tmp_path, worker_dir):
        f = tmp_path / "Big Muff (v2).schx"; f.write_text("<schx/>")
        ok, detail = dp.sync_path_to_worker("localhost", f"~/{worker_dir.name}", f,
                                            "circuits/Big Muff (v2).schx")
        assert ok, detail
        assert (worker_dir / "circuits" / "Big Muff (v2).schx").read_text() == "<schx/>"
        assert [p.name for p in (worker_dir / "circuits").iterdir()] == ["Big Muff (v2).schx"]

    def test_directory_contents(self, tmp_path, worker_dir):
        d = tmp_path / "mods"; d.mkdir()
        (d / "a.lib").write_text("A"); (d / "sub").mkdir(); (d / "sub" / "b.lib").write_text("B")
        ok, detail = dp.sync_path_to_worker("localhost", f"~/{worker_dir.name}", d, "pedals/mods")
        assert ok, detail
        assert (worker_dir / "pedals/mods/a.lib").read_text() == "A"
        assert (worker_dir / "pedals/mods/sub/b.lib").read_text() == "B"

    def test_directory_with_unsafe_name_via_tar(self, tmp_path, worker_dir):
        d = tmp_path / "my mods"; d.mkdir()
        (d / "a.lib").write_text("A")
        ok, detail = dp.sync_path_to_worker("localhost", f"~/{worker_dir.name}", d, "pedals/my mods")
        assert ok, detail
        assert (worker_dir / "pedals/my mods/a.lib").read_text() == "A"

    def test_resync_is_idempotent(self, tmp_path, worker_dir):
        f = tmp_path / "e.wav"; f.write_bytes(b"1")
        for _ in range(2):
            assert dp.sync_path_to_worker("localhost", f"~/{worker_dir.name}", f, "e.wav")[0]

    def test_unreachable_host_fails_cleanly(self, tmp_path):
        f = tmp_path / "e.wav"; f.write_bytes(b"1")
        ok, detail = dp.sync_path_to_worker("no-such-host.invalid", "~/r", f, "e.wav")
        assert ok is False and detail
