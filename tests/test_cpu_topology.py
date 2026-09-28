import pytest
import cpu_topology as ct


def test_parse_proc_cpuinfo_counts_unique_physical_core_pairs():
    # 2 sockets x 2 cores x 2 SMT threads each = 8 logical "processor" entries, 4 physical cores.
    text = "\n\n".join(
        f"processor\t: {i}\nphysical id\t: {phys}\ncore id\t: {core}"
        for i, (phys, core) in enumerate([
            (0, 0), (0, 0), (0, 1), (0, 1),
            (1, 0), (1, 0), (1, 1), (1, 1),
        ])
    )
    assert ct._parse_proc_cpuinfo(text) == 4


def test_parse_proc_cpuinfo_single_socket_no_smt():
    text = "\n\n".join(
        f"processor\t: {i}\nphysical id\t: 0\ncore id\t: {i}" for i in range(6)
    )
    assert ct._parse_proc_cpuinfo(text) == 6


def test_parse_proc_cpuinfo_falls_back_to_processor_count_when_no_ids_present():
    # Some VMs/containers omit "physical id"/"core id" entirely.
    text = "\n\n".join(f"processor\t: {i}\nmodel name\t: fake" for i in range(3))
    assert ct._parse_proc_cpuinfo(text) == 3


def test_parse_proc_cpuinfo_never_returns_zero():
    assert ct._parse_proc_cpuinfo("") >= 1


def test_physical_cpu_count_local_never_raises(monkeypatch):
    # Whatever the real platform is, this must return a positive int without raising --
    # concurrency defaults should degrade, not block the caller.
    n = ct.physical_cpu_count()
    assert isinstance(n, int) and n >= 1


def test_physical_cpu_count_falls_back_when_detection_raises(monkeypatch):
    def boom():
        raise OSError("no sysctl here")
    monkeypatch.setattr(ct, "_physical_cpu_count_local", boom)
    monkeypatch.setattr(ct.os, "cpu_count", lambda: 5)
    assert ct.physical_cpu_count() == 5


def test_physical_cpu_count_remote_falls_back_to_nproc_on_failure(monkeypatch):
    def boom(host):
        raise OSError("unreachable")
    monkeypatch.setattr(ct, "_physical_cpu_count_remote", boom)

    class FakeResult:
        stdout = "8\n"
    monkeypatch.setattr(ct.subprocess, "run", lambda *a, **k: FakeResult())
    assert ct.physical_cpu_count("some-host") == 8


def test_physical_cpu_count_remote_never_raises_when_totally_unreachable(monkeypatch):
    def boom(host):
        raise OSError("unreachable")
    monkeypatch.setattr(ct, "_physical_cpu_count_remote", boom)

    def boom2(*a, **k):
        raise OSError("ssh not found")
    monkeypatch.setattr(ct.subprocess, "run", boom2)
    n = ct.physical_cpu_count("nowhere-host")
    assert isinstance(n, int) and n >= 1


class TestPhysicalCpuCountRemoteOsDetection:
    """The remote probe branches on the TARGET's OS, detected in the same SSH round trip --
    not on the caller's own platform.Darwin has no /proc/cpuinfo and ships no `nproc`, so
    guessing wrong here used to silently return a hardcoded, wrong core count for a remote Mac
    (found via fleet_inventory.py probing a real Mac-to-Mac ssh alias)."""

    def _fake_run(self, monkeypatch, stdout, returncode=0):
        calls = []
        def fake(cmd, **kw):
            calls.append(cmd)
            import subprocess as sp
            return sp.CompletedProcess(cmd, returncode, stdout, "")
        monkeypatch.setattr(ct.subprocess, "run", fake)
        return calls

    def test_darwin_remote_uses_sysctl_not_proc_cpuinfo(self, monkeypatch):
        self._fake_run(monkeypatch, "DARWIN\n10\n")
        assert ct._physical_cpu_count_remote("mbp") == 10

    def test_linux_remote_still_parses_proc_cpuinfo(self, monkeypatch):
        cpuinfo = "\n\n".join(
            f"processor\t: {i}\nphysical id\t: 0\ncore id\t: {i}" for i in range(6))
        self._fake_run(monkeypatch, f"LINUX\n{cpuinfo}\n")
        assert ct._physical_cpu_count_remote("blackbox") == 6

    def test_one_ssh_round_trip_not_two(self, monkeypatch):
        calls = self._fake_run(monkeypatch, "DARWIN\n8\n")
        ct._physical_cpu_count_remote("mac-1")
        assert len(calls) == 1

    def test_darwin_count_never_below_one(self, monkeypatch):
        self._fake_run(monkeypatch, "DARWIN\n0\n")
        assert ct._physical_cpu_count_remote("h") >= 1

    def test_empty_output_raises_rather_than_crashing_on_index_error(self, monkeypatch):
        self._fake_run(monkeypatch, "")
        with pytest.raises(ValueError):
            ct._physical_cpu_count_remote("h")

    def test_unrecognised_marker_falls_back_to_proc_cpuinfo_parsing(self, monkeypatch):
        # uname unavailable/unexpected output -> the shell's own `else` branch already handles
        # this remotely, but if the marker line itself is garbled, don't crash -- degrade to
        # parsing whatever came after it as cpuinfo-shaped text (same "degrade, never raise"
        # contract physical_cpu_count() promises its callers).
        self._fake_run(monkeypatch, "???\nprocessor\t: 0\nphysical id\t: 0\ncore id\t: 0\n")
        assert ct._physical_cpu_count_remote("h") == 1

    def test_full_public_api_returns_the_darwin_count_end_to_end(self, monkeypatch):
        self._fake_run(monkeypatch, "DARWIN\n14\n")
        assert ct.physical_cpu_count("mbp") == 14
