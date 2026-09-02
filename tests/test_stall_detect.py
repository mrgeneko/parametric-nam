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
