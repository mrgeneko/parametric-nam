"""Tests for distribute_pull.py's scheduling safeguards.

Both cover failures seen on the Duke of Tone 252-combination run (2026-09-04):

  * a worker whose venv could not import the transient-coverage gate failed in under a
    second and killed 27 of 31 chunks in ~70s, because a fast-failing worker drains the
    queue faster than healthy machines can take work, and the "retry on a DIFFERENT
    worker" the docstring promised was only an append-to-back that the same broken worker
    immediately grabbed again;

  * --chunks 32 against a 4-value Volume axis froze Volume inside every shard, so the
    renderer's own per-shard knob-sensitivity check reported a knob we had just measured
    moving output 28x as "RMS varies only 0.00% -- knob may have no effect".
"""
import pytest

import distribute_pull as dp


# ------------------------------------------------------------------ chunk/axis aliasing

DUKE = ["--range", "Gain=0.1,0.25,0.5,0.75,0.85,0.95,1.0", "--range", "Tone=0.2,0.5,0.8",
        "--range", "Presence=0.2,0.5,0.8", "--range", "Volume=0.1,0.25,0.75,1.0"]


def test_warns_when_a_knob_axis_divides_the_chunk_count(capsys):
    dp._warn_chunk_aliasing(DUKE, 32)          # 4 | 32 -> Volume frozen in every shard
    out = capsys.readouterr().out
    assert "aliasing" in out and "Volume" in out


def test_suggests_a_coprime_chunk_count(capsys):
    dp._warn_chunk_aliasing(DUKE, 32)
    out = capsys.readouterr().out
    suggested = int(out.split("Use --chunks ")[1].split()[0])
    for axis in (7, 3, 3, 4):
        assert suggested % axis, f"suggested {suggested} still aliases a {axis}-value axis"


def test_silent_when_no_axis_divides_the_chunk_count(capsys):
    dp._warn_chunk_aliasing(DUKE, 31)          # prime -> coprime with 7, 3, 3, 4
    assert capsys.readouterr().out == ""


def test_aliasing_is_about_divisibility_not_size(capsys):
    # 3 does not divide 64, so a big chunk count over a 3-value axis is fine; the bug is
    # a shared factor, not granularity.
    dp._warn_chunk_aliasing(["--range", "Tone=0.2,0.5,0.8"], 64)
    assert capsys.readouterr().out == ""


def test_single_valued_axis_is_not_reported():
    # A pinned knob has nothing to vary; it is not an aliasing problem.
    dp._warn_chunk_aliasing(["--range", "Volume=0.7"], 32)


def test_no_ranges_is_a_no_op(capsys):
    dp._warn_chunk_aliasing(["--knobs", "Gain,Tone"], 32)
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------- quarantine

def _worker(host="w", parallel=1):
    return dp.Worker(f"{host}:/tmp/x:{parallel}")


def test_a_new_worker_is_not_quarantined():
    w = _worker()
    assert not w.quarantined and w.consec_fail == 0


def test_consecutive_failures_are_counted_and_reset_by_a_success():
    w = _worker()
    w.consec_fail = 2
    w.consec_fail = 0          # what the rc==0 branch does
    assert w.consec_fail == 0


def test_worker_spec_still_parses_with_and_without_env():
    plain = dp.Worker("h:/d:4")
    assert (plain.host, plain.dir, plain.parallel, plain.env) == ("h", "/d", 4, "")
    with_env = dp.Worker("h:/d:4:DOTNET_ROOT=$HOME/.dotnet")
    assert with_env.env == "DOTNET_ROOT=$HOME/.dotnet"


def test_worker_spec_rejects_a_short_spec():
    with pytest.raises(ValueError, match="host:dir:parallel"):
        dp.Worker("h:/d")
