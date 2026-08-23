"""Properties tools/prepare_excitation.py must have. This tool closes the human-in-the-loop gap
between find_saturation_point.py and build_excitation.py by deriving --sweep-peaks/
--realistic-peak directly from a MEASURED worst-case onset across the knob grid's corners --
worst_case_onset's "refuse to guess" behavior (raise rather than silently build against a
partial sweep) and its worst-case (not average, not first) selection are the properties an
excitation's whole calibration depends on. Exercised with find_saturation_point stubbed out.

See tools/prepare_excitation.py.
"""
import sys
from pathlib import Path

import pytest

from tools.prepare_excitation import _parse_fixed, _parse_ranges, main, worst_case_onset


class TestParseRanges:
    def test_parses_multiple_entries(self):
        ranges = _parse_ranges(["Gain=0.1,0.5,0.9", "Tone=0.2,0.8"])
        assert ranges == {"Gain": [0.1, 0.5, 0.9], "Tone": [0.2, 0.8]}

    def test_values_are_floats_not_strings(self):
        ranges = _parse_ranges(["Gain=0,1"])
        assert ranges["Gain"] == [0.0, 1.0]
        assert all(isinstance(v, float) for v in ranges["Gain"])

    def test_empty_list_returns_empty_dict(self):
        assert _parse_ranges([]) == {}


class TestParseFixed:
    def test_parses_comma_separated_pairs(self):
        assert _parse_fixed("Volume=1.0,Presence=0.5") == {"Volume": 1.0, "Presence": 0.5}

    def test_empty_string_returns_empty_dict(self):
        assert _parse_fixed("") == {}

    def test_none_returns_empty_dict(self):
        assert _parse_fixed(None) == {}


class TestWorstCaseOnset:
    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def _stub_onset(self, monkeypatch, onset_fn, calls=None):
        def fake(backend, params, tmp, max_v=40.0, lead_silence_s=0.0):
            if calls is not None:
                calls.append(dict(params))
            onset = onset_fn(params)
            if onset is None:
                return None
            return {"onset_99pct_input_v": onset, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("tools.prepare_excitation.find_saturation_point", fake)

    def test_returns_the_highest_onset_not_the_average_or_first(self, monkeypatch, tmp_path):
        self._stub_onset(monkeypatch, lambda p: 5.0 if p.get("Gain") == 1.0 else 0.1)
        worst, rows = worst_case_onset(backend=object(), identity=b"x", cache_extra="e",
                                        knob_ranges={"Gain": [0.0, 1.0]}, fixed={}, tmp=str(tmp_path),
                                        quiet=True)
        assert worst == pytest.approx(5.0)
        assert len(rows) > 1  # more than one corner was actually probed

    def test_raises_when_any_corner_onset_is_unresolved(self, monkeypatch, tmp_path):
        self._stub_onset(monkeypatch, lambda p: None if p.get("Gain") == 1.0 else 0.1)
        with pytest.raises(RuntimeError, match="refusing to build"):
            worst_case_onset(backend=object(), identity=b"x", cache_extra="e",
                              knob_ranges={"Gain": [0.0, 1.0]}, fixed={}, tmp=str(tmp_path), quiet=True)

    def test_fixed_params_are_merged_into_every_corners_params(self, monkeypatch, tmp_path):
        seen = []
        self._stub_onset(monkeypatch, lambda p: 0.1, calls=seen)
        worst_case_onset(backend=object(), identity=b"x", cache_extra="e",
                          knob_ranges={"Gain": [0.0, 1.0]}, fixed={"Volume": 1.0}, tmp=str(tmp_path),
                          quiet=True)
        assert all(c.get("Volume") == 1.0 for c in seen)

    def test_a_cached_onset_is_not_recomputed(self, monkeypatch, tmp_path):
        calls = []
        self._stub_onset(monkeypatch, lambda p: 0.1, calls=calls)
        kwargs = dict(backend=object(), identity=b"x", cache_extra="e",
                      knob_ranges={"Gain": [0.5]}, fixed={}, tmp=str(tmp_path), quiet=True)
        worst_case_onset(**kwargs)
        n_first = len(calls)
        assert n_first > 0
        worst_case_onset(**kwargs)
        assert len(calls) == n_first


class TestMainRealisticPeakVsCheckTransientCoverage:
    """The exact regression this session found: with --realistic-peak-frac below 1.0,
    --realistic-peak comes out LESS than worst-case onset by construction, which guarantees
    check_transient_coverage.py's own default gate (transient_peak >= onset at margin=1.0)
    FAILS at exactly the worst corner -- contradicting this tool's own docstring claim that a
    check run afterward should pass cleanly. --realistic-peak-frac's default (1.0) must not
    regress back below that line."""

    def _write_pedal_module(self, tmp_path, name="gen_fake_ngspice"):
        (tmp_path / f"{name}.py").write_text(
            "KNOB_NAMES = ['Gain']\ndef build_deck(**kw): return ''\n")

    def _run_main_and_capture_cmd(self, tmp_path, monkeypatch, worst_onset, extra_argv=()):
        self._write_pedal_module(tmp_path)
        monkeypatch.setattr("tools.prepare_excitation.worst_case_onset",
                            lambda *a, **kw: (worst_onset, [{"corner": "worst", "onset_v": worst_onset}]))
        captured = {}

        def fake_run(cmd, check=True):
            captured["cmd"] = cmd
            return None
        monkeypatch.setattr("tools.prepare_excitation.subprocess.run", fake_run)

        argv = ["prepare_excitation.py", "--backend", "ngspice-deck",
                "--pedal-dir", str(tmp_path), "--module", "gen_fake_ngspice",
                "--range", "Gain=0.0,1.0", "--real-clip", "clip.wav",
                "--output", str(tmp_path / "out.wav"), *extra_argv]
        monkeypatch.setattr(sys, "argv", argv)
        main()
        return captured["cmd"]

    def _realistic_peak_from_cmd(self, cmd):
        return float(cmd[cmd.index("--realistic-peak") + 1])

    def test_default_realistic_peak_meets_or_exceeds_worst_case_onset(self, tmp_path, monkeypatch):
        cmd = self._run_main_and_capture_cmd(tmp_path, monkeypatch, worst_onset=5.0)
        assert self._realistic_peak_from_cmd(cmd) >= 5.0

    def test_realistic_peak_scales_with_the_explicit_frac(self, tmp_path, monkeypatch):
        cmd = self._run_main_and_capture_cmd(tmp_path, monkeypatch, worst_onset=5.0,
                                             extra_argv=["--realistic-peak-frac", "2.0"])
        assert self._realistic_peak_from_cmd(cmd) == pytest.approx(10.0)
