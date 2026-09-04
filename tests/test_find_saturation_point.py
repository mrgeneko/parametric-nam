"""Properties find_saturation_point.py's backend-agnostic saturation sweep must have.
It underlies preflight.py's --find-peak/--clean-probe-peak (the fix for EQ-knob checks getting
swamped by clipping at a moderate Gain default) and prepare_excitation.py's excitation-level
choice -- a wrong ceiling/onset silently mis-sizes both. Exercised here against a FakeBackend
(no real ngspice/livespice) so the algorithm's own logic -- curve assembly, transient exclusion,
ceiling/onset selection, failure handling -- is tested independently of any circuit.

See find_saturation_point.py, preflight.py, prepare_excitation.py.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from find_saturation_point import (
    _linear_region_top,
    _loglog_interp,
    find_saturation_point,
    findpeak_cache_key,
)


class FakeBackend:
    """Ignores the actual raw waveform's shape -- render_many returns an array of the same
    length filled with rms_fn(amp), so the steady-state RMS the real code computes is
    rms_fn(amp) exactly (a constant array's RMS is its own magnitude, no sine-period rounding
    to worry about)."""

    def __init__(self, rms_fn=None, fail_fn=None):
        self.rms_fn = rms_fn or (lambda amp: amp)
        self.fail_fn = fail_fn or (lambda amp: False)
        self.prepared_amps = []

    def prepare_input(self, raw, sr, level_v, scratch, tag):
        self.prepared_amps.append(level_v)
        return {"raw": raw, "level_v": level_v}

    def render_many(self, jobs, input_handle, scratch):
        amp = input_handle["level_v"]
        raw = input_handle["raw"]
        out = {}
        for job in jobs:
            tag = job["tag"]
            if self.fail_fn(amp):
                out[tag] = None
            else:
                out[tag] = np.full_like(raw, self.rms_fn(amp), dtype=np.float32)
        return out


class TestLoglogInterp:
    def test_recovers_exact_point_on_a_y_equals_x_line(self):
        # log-log interpolation of y=x is exact: interpolating for y=10 between (1,1) and
        # (100,100) must return x=10, not some log-log-distorted approximation.
        assert _loglog_interp(1, 1, 100, 100, 10) == pytest.approx(10.0)

    def test_result_falls_between_the_two_x_endpoints(self):
        x = _loglog_interp(2.0, 1.0, 8.0, 4.0, 2.0)
        assert 2.0 < x < 8.0


class TestFindSaturationPointCurveAndCeiling:
    def test_curve_sorted_ascending_by_input_level(self, tmp_path):
        backend = FakeBackend(rms_fn=lambda amp: amp)
        result = find_saturation_point(backend, {}, tmp=str(tmp_path), dur=0.01, start_v=0.1, max_v=10.0,
                                        npoints=6, sr=200, workers=4)
        amps = [a for a, _ in result["curve"]]
        assert amps == sorted(amps)

    def test_ceiling_is_the_first_amp_that_reaches_the_max_rms(self, tmp_path):
        """Once the circuit saturates, every higher input gives the SAME ceiling RMS -- a tie.
        Python's max() keeps the first item at the max value, so ceiling_at_input_v lands on
        the smallest input that already reached the ceiling, not the largest one tested. This
        is the exact behavior preflight.py/prepare_excitation.py rely on to report where
        saturation actually begins, not just "the loudest thing we tried"."""
        backend = FakeBackend(rms_fn=lambda amp: min(2.0 * amp, 3.0))
        result = find_saturation_point(backend, {}, tmp=str(tmp_path), dur=0.01, start_v=0.5, max_v=8.0,
                                        npoints=6, sr=200, workers=4)
        assert result["ceiling_rms"] == pytest.approx(3.0)
        plateau_amps = [a for a, r in result["curve"] if r == pytest.approx(3.0)]
        assert result["ceiling_at_input_v"] == pytest.approx(min(plateau_amps))
        assert result["ceiling_at_input_v"] != pytest.approx(max(plateau_amps))

    def test_onset_is_interpolated_from_the_straddling_segment(self, tmp_path):
        gain, ceiling = 1.0, 2.0
        backend = FakeBackend(rms_fn=lambda amp: min(gain * amp, ceiling))
        start_v, max_v, npoints = 0.1, 3.0, 4
        result = find_saturation_point(backend, {}, tmp=str(tmp_path), dur=0.01, start_v=start_v,
                                        max_v=max_v, npoints=npoints, sr=200, workers=4)

        amps = np.geomspace(start_v, max_v, npoints)
        curve = sorted((float(a), min(gain * a, ceiling)) for a in amps)
        target = 0.99 * ceiling
        a0, r0 = next(p for p in reversed(curve) if p[1] < target)
        a1, r1 = next(p for p in curve if p[1] >= target)
        expected_onset = _loglog_interp(a0, r0, a1, r1, target)

        assert result["onset_99pct_input_v"] == pytest.approx(expected_onset)

    def test_no_saturation_in_swept_range_still_finds_a_ceiling_and_onset(self, tmp_path):
        # A circuit that never saturates across the swept range: the "ceiling" is just the
        # loudest point tried, and onset still resolves against 99% of THAT.
        backend = FakeBackend(rms_fn=lambda amp: 2.0 * amp)
        result = find_saturation_point(backend, {}, tmp=str(tmp_path), dur=0.01, start_v=0.1, max_v=1.0,
                                        npoints=5, sr=200, workers=4)
        assert result["ceiling_rms"] == pytest.approx(2.0)
        assert result["onset_99pct_input_v"] is not None

    def test_params_are_forwarded_to_every_render(self, tmp_path):
        seen = []

        class RecordingBackend(FakeBackend):
            def render_many(self, jobs, input_handle, scratch):
                seen.append(jobs[0]["params"])
                return super().render_many(jobs, input_handle, scratch)

        params = {"Gain": 0.3, "Tone": 0.7}
        find_saturation_point(RecordingBackend(), params, tmp=str(tmp_path), dur=0.01, start_v=0.1,
                               max_v=1.0, npoints=3, sr=200, workers=2)
        assert all(p == params for p in seen)


class TestFindSaturationPointFailureHandling:
    def test_all_renders_failing_returns_none(self, tmp_path):
        backend = FakeBackend(fail_fn=lambda amp: True)
        result = find_saturation_point(backend, {}, tmp=str(tmp_path), dur=0.01, start_v=0.1, max_v=1.0,
                                        npoints=4, sr=200, workers=2)
        assert result is None

    def test_partial_failures_are_dropped_not_fatal(self, tmp_path):
        backend = FakeBackend(rms_fn=lambda amp: amp, fail_fn=lambda amp: amp > 0.5)
        result = find_saturation_point(backend, {}, tmp=str(tmp_path), dur=0.01, start_v=0.1, max_v=1.0,
                                        npoints=6, sr=200, workers=3)
        assert result is not None
        assert all(a <= 0.5 for a, _ in result["curve"])
        assert len(result["curve"]) < 6


class TestFindSaturationPointTransientExclusion:
    def test_transient_before_the_steady_slice_is_excluded_from_rms(self, tmp_path):
        """The real render always starts from a cold state, so the first part of ANY render
        (the lead-silence settle, plus the first half of the tone itself) is excluded before
        computing RMS -- `y[tone_start + int(sr*dur*0.5):]`. A backend returning garbage there
        (a settling transient) must not move the measured ceiling at all."""
        sr, dur, lead_silence_s = 1000, 2.0, 1.0
        clean_val, garbage_val = 5.0, 999.0
        total_len = int(sr * lead_silence_s) + int(sr * dur)
        slice_start = int(sr * lead_silence_s) + int(sr * dur * 0.5)

        class TransientBackend:
            def prepare_input(self, raw, sr_, level_v, scratch, tag):
                return None

            def render_many(self, jobs, input_handle, scratch):
                y = np.full(total_len, clean_val, dtype=np.float32)
                y[:slice_start] = garbage_val
                return {jobs[0]["tag"]: y}

        result = find_saturation_point(TransientBackend(), {}, tmp=str(tmp_path), dur=dur,
                                        lead_silence_s=lead_silence_s, start_v=1.0, max_v=1.0,
                                        npoints=1, sr=sr)
        assert result["ceiling_rms"] == pytest.approx(clean_val)


class TestLinearRegionTop:
    def test_constant_gain_curve_returns_the_highest_point(self):
        curve = [(a, 2.0 * a) for a in (0.1, 0.5, 1.0, 2.0)]
        assert _linear_region_top(curve) == pytest.approx(2.0)

    def test_stops_at_the_last_point_before_gain_deviates(self):
        curve = [(0.1, 0.2), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]  # gain drops 2.0 -> 1.5 at a=2.0
        assert _linear_region_top(curve, tol=0.05) == pytest.approx(1.0)

    def test_ignores_nonpositive_points(self):
        curve = [(0.0, 0.0), (0.1, 0.2), (0.5, 1.0)]
        assert _linear_region_top(curve) == pytest.approx(0.5)

    def test_too_short_returns_none(self):
        assert _linear_region_top([(0.1, 0.2)]) is None
        assert _linear_region_top([]) is None


class TestFindpeakCacheKey:
    @pytest.fixture(autouse=True)
    def fake_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def test_deterministic_for_the_same_inputs(self):
        k1 = findpeak_cache_key(b"circuit-bytes", {"Gain": 0.3}, "extra")
        k2 = findpeak_cache_key(b"circuit-bytes", {"Gain": 0.3}, "extra")
        assert k1 == k2

    def test_param_dict_order_does_not_matter(self):
        k1 = findpeak_cache_key(b"circuit-bytes", {"Gain": 0.3, "Tone": 0.5}, "extra")
        k2 = findpeak_cache_key(b"circuit-bytes", {"Tone": 0.5, "Gain": 0.3}, "extra")
        assert k1 == k2

    def test_different_identity_bytes_give_different_keys(self):
        """identity_bytes is how an edited circuit invalidates a stale cached sweep -- a
        changed gen_*_ngspice.py or .schx must not collide with the old one's cache entry."""
        k1 = findpeak_cache_key(b"circuit-v1", {"Gain": 0.3}, "extra")
        k2 = findpeak_cache_key(b"circuit-v2", {"Gain": 0.3}, "extra")
        assert k1 != k2

    def test_different_params_give_different_keys(self):
        k1 = findpeak_cache_key(b"circuit-bytes", {"Gain": 0.3}, "extra")
        k2 = findpeak_cache_key(b"circuit-bytes", {"Gain": 0.9}, "extra")
        assert k1 != k2

    def test_returns_a_json_path_under_the_findpeak_cache_dir(self, tmp_path):
        path = findpeak_cache_key(b"circuit-bytes", {"Gain": 0.3}, "extra")
        assert path.suffix == ".json"
        assert path.parent == tmp_path / ".cache" / "parametric-nam" / "findpeak"
        assert path.parent.is_dir()


class TestScratchDir:
    """Scratch must not survive the process that made it.

    preflight.py and prepare_excitation.py wrote their intermediate renders to
    ~/.cache/parametric-nam/<tool>_scratch and never cleaned them, next to the real
    content-keyed caches. They reached 123 MB of write-only files under fixed names, for
    devices long finished. These pin the distinction so it can't drift back.
    """

    def _child(self, body, tmp_path):
        """Run `body` in a real subprocess -- atexit cleanup only happens on interpreter exit,
        so an in-process test would prove nothing."""
        import subprocess, sys, textwrap
        code = textwrap.dedent(f"""
            import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
            from pathlib import Path
            from find_saturation_point import scratch_dir
            {body}
        """)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "HOME": str(tmp_path)})
        assert r.returncode == 0, r.stderr
        return r

    def test_it_is_removed_when_the_process_exits(self, tmp_path):
        r = self._child('d = scratch_dir("t"); (d / "probe.wav").write_bytes(b"x" * 10);'
                        ' print(d)', tmp_path)
        d = Path(r.stdout.strip().splitlines()[-1])
        assert not d.exists(), f"scratch survived the process: {d}"

    def test_it_is_removed_even_when_the_tool_exits_nonzero_paths(self, tmp_path):
        """Both callers are long main()s full of sys.exit() -- the case a `with` block around
        one call site would have missed."""
        r = self._child('d = scratch_dir("t"); print(d); '
                        'import sys; sys.exit(0)', tmp_path)
        assert not Path(r.stdout.strip().splitlines()[-1]).exists()

    def test_it_is_not_inside_the_cache_directory(self, tmp_path):
        """The bug was scratch living next to findpeak/gridadq, which made 'clear the cache'
        and 'clear dead scratch' the same destructive gesture."""
        r = self._child('print(scratch_dir("t"))', tmp_path)
        d = Path(r.stdout.strip().splitlines()[-1])
        assert ".cache/parametric-nam" not in str(d), \
            f"scratch must not live among the real caches: {d}"

    def test_keep_scratch_keeps_it_at_a_predictable_path(self, tmp_path):
        """--keep-scratch exists to be looked at afterwards, so it must both survive AND be
        somewhere findable -- not a random tempdir name printed once and lost."""
        r = self._child('d = scratch_dir("t", keep=True); (d / "probe.wav").write_bytes(b"x");'
                        ' print(d)', tmp_path)
        d = Path(r.stdout.strip().splitlines()[-1])
        assert d.exists() and (d / "probe.wav").exists(), "--keep-scratch must not clean up"
        assert d == tmp_path / ".cache" / "parametric-nam" / "t_scratch"
        assert "keep" in r.stderr.lower(), "keeping scratch should say so on stderr"
