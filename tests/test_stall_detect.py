"""_run_with_stall_detect: a render that is SLOW must survive; one that is HUNG must not.

The old contract was a total wall-clock timeout guessed from oversample, which killed
legitimately slow operating points (55 of 448 Mesa combinations, all Master=1.0) while still
letting a real hang burn the entire budget. These cover the three cases that distinction turns on.
"""
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import gen_dataset_from_schx as g  # noqa: E402


def _run(script, base_timeout, stall_s, monkeypatch):
    monkeypatch.setattr(g, "STALL_S", stall_s)
    p = subprocess.Popen([sys.executable, "-u", "-c", script],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    try:
        out, err = g._run_with_stall_detect(p, base_timeout)
        return "ok", time.time() - t0, err.count("PROGRESS")
    except g._StallTimeout as e:
        return "stall", time.time() - t0, str(e)


SLOW = ("import sys,time\n"
        "for i in range(6):\n"
        "    time.sleep(0.5); sys.stderr.write(f'PROGRESS {i+1}/6\\n'); sys.stderr.flush()\n")
HUNG = "import time\ntime.sleep(600)\n"
STOPS = ("import sys,time\n"
         "sys.stderr.write('PROGRESS 1/6\\n'); sys.stderr.flush()\n"
         "time.sleep(600)\n")


def test_slow_but_progressing_is_not_killed(monkeypatch):
    # 3 s of work against a 1 s base timeout: under the OLD total-timeout contract this dies.
    verdict, elapsed, n = _run(SLOW, base_timeout=1, stall_s=4, monkeypatch=monkeypatch)
    assert verdict == "ok", f"a progressing render was killed after {elapsed:.1f}s"
    assert n == 6


def test_a_hang_is_caught_quickly(monkeypatch):
    verdict, elapsed, msg = _run(HUNG, base_timeout=100, stall_s=2, monkeypatch=monkeypatch)
    assert verdict == "stall"
    assert elapsed < 6, "a hang should die on the stall timer, not the total budget"
    assert "no progress" in msg


def test_progress_then_hang_is_caught(monkeypatch):
    verdict, elapsed, msg = _run(STOPS, base_timeout=100, stall_s=2, monkeypatch=monkeypatch)
    assert verdict == "stall"
    assert elapsed < 6


def test_total_ceiling_catches_a_livelock(monkeypatch):
    """A live-lock keeps emitting progress forever -- the stall timer cannot see it, so the
    ceiling is the backstop."""
    monkeypatch.setattr(g, "TOTAL_CEILING_MULT", 2.0)
    live = ("import sys,time\n"
            "while True:\n"
            "    time.sleep(0.2); sys.stderr.write('PROGRESS 1/9\\n'); sys.stderr.flush()\n")
    verdict, elapsed, msg = _run(live, base_timeout=1.5, stall_s=30, monkeypatch=monkeypatch)
    assert verdict == "stall"
    assert "live-lock" in msg


def test_threshold_adapts_to_slow_chunks(monkeypatch):
    """A render whose chunks are slower than the floor must raise its OWN threshold.

    This is the property a fixed constant cannot have: at the ladder's oversample ceiling on
    the slowest machine in this fleet a single chunk takes ~128 s, and the pathological
    operating points are slower still. Anything fixed is either too tight there or uselessly
    loose everywhere else.
    """
    monkeypatch.setattr(g, "STALL_S", 2.0)          # tiny floor, so the ADAPTIVE part is what's tested
    monkeypatch.setattr(g, "STALL_GAP_MULT", 5.0)
    # gaps of ~1.5 s -- well over the 2 s floor once multiplied, so the run must survive
    slow_chunks = ("import sys,time\n"
                   "for i in range(4):\n"
                   "    time.sleep(1.5); sys.stderr.write(f'PROGRESS {i+1}/4\\n'); sys.stderr.flush()\n")
    verdict, elapsed, n = _run(slow_chunks, base_timeout=100, stall_s=2.0, monkeypatch=monkeypatch)
    assert verdict == "ok", "chunks slower than the floor should widen the threshold, not fail"
    assert n == 4


def test_floor_applies_before_any_progress_arrives(monkeypatch):
    """Before the first PROGRESS line there is no gap to learn from -- circuit parse,
    simulation build and JIT all happen there -- so the floor is what protects that window."""
    monkeypatch.setattr(g, "STALL_GAP_MULT", 5.0)
    verdict, elapsed, msg = _run(HUNG, base_timeout=100, stall_s=2.0, monkeypatch=monkeypatch)
    assert verdict == "stall"
    assert elapsed < 6
