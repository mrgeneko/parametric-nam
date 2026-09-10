"""find_saturation_point must find the onset even when the sweep's DEFAULT floor
(start_v=0.005V) already sits on the saturated plateau.

Regression: Mesa Dual Rectifier Ch1 (2026-09-10). Its real onset is 2.08 mV, below
start_v, so the default 0.005-40V sweep was FLAT and the crossing search returned
onset=None. That read as "never saturates -- raise --peak-max-v", and raising the
ceiling to 400V only added 20x more plateau (measured: 11.21V -> 11.12V out across an
80000x input range). The fix extends the sweep DOWNWARD instead.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from find_saturation_point import find_saturation_point

SR = 48000


class SoftClipBackend:
    """A hard limiter with settable small-signal gain. Output RMS rises linearly with
    input, then flattens -- the real shape find_saturation_point looks for. Counts
    renders so a test can assert the downward extension actually costs something finite.

    HARD clipping, not tanh: tanh's knee is far too soft to reproduce the condition under
    test. At Ch1's measured gain (29648) a tanh model is still only at 97.6% of its
    ceiling at the 0.005V sweep floor, so the floor is NOT on the plateau and the bug
    never reproduces. The real circuit was at 99.6% there.
    """

    def __init__(self, gain, ceiling=11.2):
        self.gain = gain
        self.ceiling = ceiling
        self.renders = 0
        self.levels = []

    def prepare_input(self, raw, sr, level_v, scratch, tag):
        self.levels.append(level_v)
        return (raw, level_v)

    def render_many(self, jobs, handle, scratch):
        raw, level_v = handle
        self.renders += len(jobs)
        y = np.clip(self.gain * level_v * raw.astype(np.float64),
                    -self.ceiling, self.ceiling)
        return {j["tag"]: y.astype(np.float32) for j in jobs}


def _onset(gain, **kw):
    b = SoftClipBackend(gain)
    sat = find_saturation_point(b, {}, "/tmp/nonexistent-unused", dur=0.2,
                                npoints=20, workers=4, **kw)
    return sat, b


def test_onset_found_when_default_floor_is_above_it():
    """The Ch1 case: onset far below start_v=0.005. Must still be found.

    gain=3e5 rather than Ch1's measured 29648: an ideal hard clipper is slightly LESS
    compressed just past its knee than the real circuit was (0.988 of ceiling at the
    0.005V floor here, vs 0.9958 measured), so at the literal measured gain the floor
    lands just under the 0.99 plateau test and the condition does not reproduce. The
    extra gain buys an unambiguous "floor is on the plateau", which is the state under
    test; the real device's numbers are in the module docstring.
    """
    sat, b = _onset(gain=3e5)
    assert sat is not None
    onset = sat["onset_99pct_input_v"]
    assert onset is not None, "returned None -- the downward extension did not fire"
    assert onset < 0.005, f"onset {onset} should be below the default sweep floor"
    assert b.renders < 200, "extension should terminate, not sweep forever"


def test_ordinary_circuit_does_not_trigger_extension():
    """A device whose onset is inside the default range must not pay for extra renders."""
    sat, b = _onset(gain=5.0)
    assert sat["onset_99pct_input_v"] is not None
    assert 0.005 < sat["onset_99pct_input_v"] < 40.0
    assert b.renders == 20, "no extension expected when the floor is already below onset"


def test_curve_stays_sorted_and_unique():
    sat, _ = _onset(gain=29648.0)
    amps = [a for a, _ in sat["curve"]]
    assert amps == sorted(amps)
    assert len(amps) == len(set(amps)), "duplicate amplitudes across extension rounds"


def test_extension_floor_is_respected():
    """A circuit that never comes off the plateau must give up, not loop forever."""
    b = SoftClipBackend(gain=1e12)
    sat = find_saturation_point(b, {}, "/tmp/nonexistent-unused", dur=0.2, npoints=20,
                                workers=4, max_extend_decades=2)
    assert sat is not None
    assert b.renders <= 20 + 2 * 10, "bounded by max_extend_decades"


def test_all_renders_failing_still_returns_none():
    class DeadBackend(SoftClipBackend):
        def render_many(self, jobs, handle, scratch):
            return {j["tag"]: None for j in jobs}
    assert find_saturation_point(DeadBackend(1.0), {}, "/tmp/nonexistent-unused",
                                 dur=0.2, npoints=8, workers=4) is None
