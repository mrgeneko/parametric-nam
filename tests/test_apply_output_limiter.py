"""Properties apply_output_limiter.py's soft-limit knee and hand-rolled 16-bit WAV
round-trip must have. The whole point of this tool is a SMOOTH ceiling -- no new harmonic
distortion from a hard clip -- so continuity at the threshold and a strictly-bounded (never
exceeded) ceiling are the properties that actually matter for a training target, not just
"some compression happens."

See apply_output_limiter.py.
"""
import math

import pytest

from apply_output_limiter import read_wav16, soft_limit, write_wav16


class TestSoftLimit:
    def test_passes_through_unchanged_below_threshold(self):
        assert soft_limit(0.1, threshold=0.3, ceiling=0.6) == pytest.approx(0.1)

    def test_continuous_at_the_threshold_no_discontinuity(self):
        # tanh(0) == 0, so the knee starts exactly at threshold -- a discontinuity here would
        # inject a new artifact right at the boundary, defeating the "no new distortion" goal.
        assert soft_limit(0.3, threshold=0.3, ceiling=0.6) == pytest.approx(0.3)

    def test_negative_values_mirror_the_positive_curve(self):
        pos = soft_limit(0.5, threshold=0.3, ceiling=0.6)
        neg = soft_limit(-0.5, threshold=0.3, ceiling=0.6)
        assert neg == pytest.approx(-pos)

    def test_never_reaches_or_exceeds_the_ceiling_near_the_knee(self):
        # tanh saturates to exactly 1.0 in float64 once its argument gets large (starting
        # around ax=1.0 here, since span=0.3 makes the knee steep) -- so this bound is only
        # meaningful close to the knee, not asymptotically.
        for ax in (0.31, 0.35, 0.4, 0.5):
            y = soft_limit(ax, threshold=0.3, ceiling=0.6)
            assert 0.3 <= y < 0.6

    def test_approaches_the_ceiling_for_very_loud_input(self):
        y = soft_limit(1000.0, threshold=0.3, ceiling=0.6)
        assert y == pytest.approx(0.6, abs=1e-4)

    def test_monotonically_increasing_above_threshold(self):
        xs = [0.3, 0.4, 0.5, 0.7, 1.0, 2.0]
        ys = [soft_limit(x, threshold=0.3, ceiling=0.6) for x in xs]
        assert ys == sorted(ys)

    def test_below_threshold_is_a_true_identity_not_an_approximation(self):
        # Any value under threshold must be untouched exactly, including values that would
        # otherwise trip floating-point tanh edge cases.
        for x in (-0.29, -0.05, 0.0, 0.05, 0.29):
            assert soft_limit(x, threshold=0.3, ceiling=0.6) == x


class TestWavRoundTrip:
    def test_round_trips_representable_16bit_values(self, tmp_path):
        sr = 48000
        samples = [-1.0, -0.5, 0.0, 0.5, 0.999]
        path = tmp_path / "t.wav"
        write_wav16(str(path), sr, samples)
        sr2, out = read_wav16(str(path))
        assert sr2 == sr
        for a, b in zip(samples, out):
            assert a == pytest.approx(b, abs=1e-4)

    def test_write_clamps_values_outside_full_scale(self, tmp_path):
        path = tmp_path / "t.wav"
        write_wav16(str(path), 48000, [2.0, -3.0])
        _, out = read_wav16(str(path))
        assert out[0] == pytest.approx(1.0, abs=1e-4)
        assert out[1] == pytest.approx(-1.0, abs=1e-4)
