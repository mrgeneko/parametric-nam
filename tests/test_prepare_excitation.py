"""Properties tools/prepare_excitation.py must have. This tool closes the human-in-the-loop gap
between find_saturation_point.py and build_excitation.py by deriving --sweep-peaks/
--realistic-peak directly from a MEASURED worst-case onset across the knob grid's corners --
worst_case_onset's "refuse to guess" behavior (raise rather than silently build against a
partial sweep) and its worst-case (not average, not first) selection are the properties an
excitation's whole calibration depends on. Exercised with find_saturation_point stubbed out.

See tools/prepare_excitation.py.
"""
from pathlib import Path

import pytest

from tools.prepare_excitation import _parse_fixed, _parse_ranges, worst_case_onset


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
