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
