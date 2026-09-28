#!/usr/bin/env python3
"""Physical CPU core count -- the right default for CPU-bound, single-threaded render
concurrency (gen_dataset_from_schx.py --workers, prepare_excitation.py --parallel-sims/
--corner-workers, check_transient_coverage.py --workers, distribute_pull.py --worker's
PARALLEL), which os.cpu_count()/nproc get wrong on any SMT/hyperthreaded x86 machine.

WHY THIS EXISTS (2026-09-26). Benchmarked JC-120's ngspice-deck render (single-threaded,
CPU-bound solve) at parallel_sims in [4, 6, 8, 10, 12] on two real machines:

  blackbox (AMD 9600X, 6 cores / 12 threads):    4=1720/hr  6=2482/hr  8=2321/hr  10=2297/hr
  optiplex7010 (i5-12500, 6 cores / 12 threads):  4=1180/hr  6=1643/hr  8=1561/hr  10=1743/hr
    (optiplex's first pass was noisy -- 3 repeated trials of just 6 vs 10 confirmed 6 wins
    every time, ~5% ahead of 10: 1675/1663, 1758/1613, 1744/1663)

Both machines peak at exactly their PHYSICAL core count, then throughput drops (blackbox)
or plateaus/noisily fluctuates (optiplex) past it -- SMT threads fight over the same
physical core's execution units for a compute-bound single-threaded solve; they do not add
real parallelism. os.cpu_count() (Python) / nproc (shell) report LOGICAL cores -- 12 on
both boxes here -- which is exactly the wrong number to default to for this workload.

Not tested on Apple Silicon (no SMT there, so logical == physical and os.cpu_count() was
already correct) or on any GPU/multi-threaded-solver workload -- this is a validated
default for THIS class of workload (CPU-bound, single-threaded-per-process SPICE solves),
not a universal claim that physical-core-count is always optimal. Every call site wiring
this in keeps its flag overridable.
"""
import os
import platform
import re
import subprocess

import ssh_target


def physical_cpu_count(host: str = None) -> int:
    """Physical (not logical/SMT) core count, local or over SSH to `host`.

    Falls back to os.cpu_count() (logical count -- an overestimate on SMT hardware, but
    still a usable number) if physical-core detection fails for any reason: an unsupported
    platform, a missing tool, a malformed /proc/cpuinfo, or an unreachable host. Never
    raises -- a concurrency default should degrade, not block the caller.
    """
    try:
        if host:
            return _physical_cpu_count_remote(host)
        return _physical_cpu_count_local()
    except Exception:
        pass
    if host:
        try:
            out = subprocess.run(ssh_target.ssh_argv(host, "-o", "ConnectTimeout=8") + ["nproc"],
                                  capture_output=True, text=True, timeout=15)
            return max(1, int(out.stdout.strip()))
        except Exception:
            return 4  # unreachable host, no local os.cpu_count() fallback applies
    return os.cpu_count() or 4


def _physical_cpu_count_local() -> int:
    system = platform.system()
    if system == "Darwin":
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                              capture_output=True, text=True, timeout=5, check=True)
        return max(1, int(out.stdout.strip()))
    if system == "Linux":
        return _parse_proc_cpuinfo(open("/proc/cpuinfo").read())
    return os.cpu_count() or 4


def _physical_cpu_count_remote(host: str) -> int:
    """Physical core count over SSH, Linux or Darwin.

    Originally Linux-only ("every remote worker in this fleet is Linux so far") -- Darwin has
    no /proc/cpuinfo and ships no `nproc` either, so a remote Mac used to fall all the way
    through physical_cpu_count()'s fallback chain to a hardcoded 4, silently wrong (this Air is
    10, the fleet's M3 Max MacBook Pro is 14). Found 2026-09-28 via fleet_inventory.py, whose
    --probe-hosts is the first caller that plausibly probes a remote Mac (the mini's own
    ~/.ssh/config has an `mbp` alias) rather than only ever the fleet's Linux render workers.

    One SSH round trip: the remote shell picks its own branch, rather than this function
    guessing the OS or spending a second round trip on a separate `uname` probe first.
    """
    script = ('if [ "$(uname -s)" = Darwin ]; then echo DARWIN; sysctl -n hw.physicalcpu; '
              'else echo LINUX; cat /proc/cpuinfo; fi')
    out = subprocess.run(ssh_target.ssh_argv(host, "-o", "ConnectTimeout=8") + [script],
                          capture_output=True, text=True, timeout=15, check=True)
    lines = out.stdout.splitlines()
    if not lines:
        raise ValueError(f"empty output from remote core-count probe on {host!r}")
    marker, payload = lines[0].strip(), "\n".join(lines[1:])
    if marker == "DARWIN":
        return max(1, int(payload.strip()))
    return _parse_proc_cpuinfo(payload)


def _parse_proc_cpuinfo(text: str) -> int:
    """Count unique (physical id, core id) pairs -- the standard way to get physical cores
    from /proc/cpuinfo on Linux (logical `processor` entries double up per SMT thread, but
    share their (physical id, core id) pair with their sibling thread)."""
    pairs = set()
    phys, core = None, None
    for line in text.splitlines():
        if line.startswith("physical id"):
            phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core = line.split(":", 1)[1].strip()
            if phys is not None:
                pairs.add((phys, core))
    if pairs:
        return len(pairs)
    # No "physical id"/"core id" lines at all (some VMs/containers omit them) -- fall back
    # to counting unique "processor" entries, which is the logical count, not physical, but
    # still better than raising.
    n = len(re.findall(r"^processor\s*:", text, re.MULTILINE))
    return max(1, n)


if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else None
    print(physical_cpu_count(host))
