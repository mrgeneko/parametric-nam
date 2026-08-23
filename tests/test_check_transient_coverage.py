"""Properties tools/check_transient_coverage.py must have. Built after a real miss: Tweed
5F6-A's shipped excitation cleared saturation at its own overall peak, but the transient
(real-playing) segment never did at a corner where two knobs sat at grid-min while three
others sat at grid-max simultaneously -- a corner the "solo extreme" set literally cannot
represent (solo holds every OTHER knob at center). _corners' binary hypercube exists
specifically to catch that. check_coverage is exercised with find_saturation_point/
LiveSpiceBackend stubbed out, so the per-corner pass/fail/skip bookkeeping is tested
independently of any real circuit render.

See tools/check_transient_coverage.py.
"""
import json
from pathlib import Path

import pytest

from tools.check_transient_coverage import (
    _corners,
    _transient_peak_from_recipe,
    check_coverage,
    check_coverage_ngspice_deck,
    main,
)


class TestCornersBaseSet:
    def test_includes_all_min_all_max_and_center(self):
        labels = {label for label, _ in _corners({"Gain": [0.0, 0.5, 1.0]})}
        assert {"all-min", "all-max", "center"} <= labels

    def test_uses_each_knobs_own_grid_extremes_not_a_blanket_0_1(self):
        """A narrowed tone-stack range (e.g. Tweed's 0.2..0.8) must be probed at ITS OWN
        min/max, not some generic 0/1 that isn't even in the swept grid."""
        corners = dict(_corners({"Treble": [0.2, 0.5, 0.8]}, full_hypercube=False))
        assert corners["all-min"]["Treble"] == 0.2
        assert corners["all-max"]["Treble"] == 0.8

    def test_center_is_the_nearest_grid_point_not_the_arithmetic_midpoint(self):
        corners = dict(_corners({"Gain": [0.0, 0.3, 0.9]}, full_hypercube=False))
        assert corners["center"]["Gain"] == 0.3  # vs[len//2], not (0.0+0.9)/2

    def test_solo_extremes_hold_every_other_knob_at_center(self):
        corners = dict(_corners({"A": [0.0, 0.5, 1.0], "B": [0.0, 0.5, 1.0]}, full_hypercube=False))
        assert corners["A=lo-solo"] == {"A": 0.0, "B": 0.5}  # B held at its own center
        assert corners["A=hi-solo"] == {"A": 1.0, "B": 0.5}


class TestCornersBinaryHypercube:
    def test_mixed_extreme_corner_only_exists_with_the_hypercube_enabled(self):
        """The exact Tweed 5F6-A miss: NormalVol/BrightVol at grid-min while Treble/Bass/Middle
        sit at grid-max, all at once. No solo-extreme corner can be this -- solo only ever
        moves ONE knob off center."""
        knob_ranges = {"A": [0.0, 1.0], "B": [0.0, 1.0], "C": [0.0, 1.0]}
        mixed = {"A": 0.0, "B": 1.0, "C": 0.0}

        without = _corners(knob_ranges, full_hypercube=False)
        assert not any(vals == mixed for _, vals in without)

        with_full = _corners(knob_ranges, full_hypercube=True)
        assert any(vals == mixed for _, vals in with_full)

    def test_hypercube_corners_are_deduplicated_against_the_base_set(self):
        corners = _corners({"A": [0.0, 0.5, 1.0], "B": [0.0, 0.5, 1.0]}, full_hypercube=True)
        keys = [tuple(sorted(v.items())) for _, v in corners]
        assert len(keys) == len(set(keys)), "all-min/all-max must not be duplicated by the hypercube"

    def test_raises_above_max_full_corners_instead_of_silently_ballooning(self):
        knob_ranges = {f"K{i}": [0.0, 1.0] for i in range(10)}  # 2**10 = 1024
        with pytest.raises(ValueError, match="max_full_corners"):
            _corners(knob_ranges, full_hypercube=True, max_full_corners=512)

    def test_full_hypercube_false_skips_the_raise(self):
        knob_ranges = {f"K{i}": [0.0, 1.0] for i in range(10)}
        _corners(knob_ranges, full_hypercube=False)  # must not raise


class TestTransientPeakFromRecipe:
    def test_missing_sidecar_returns_none(self, tmp_path):
        assert _transient_peak_from_recipe(tmp_path / "input.wav") is None

    def test_reads_realistic_peak_from_a_valid_recipe(self, tmp_path):
        wav = tmp_path / "input.wav"
        wav.with_suffix(".recipe.json").write_text(json.dumps({"args": {"realistic_peak": 0.42}}))
        assert _transient_peak_from_recipe(wav) == pytest.approx(0.42)

    def test_malformed_json_returns_none_not_a_crash(self, tmp_path):
        wav = tmp_path / "input.wav"
        wav.with_suffix(".recipe.json").write_text("{not json")
        assert _transient_peak_from_recipe(wav) is None

    def test_missing_key_returns_none_not_a_crash(self, tmp_path):
        wav = tmp_path / "input.wav"
        wav.with_suffix(".recipe.json").write_text(json.dumps({"args": {}}))
        assert _transient_peak_from_recipe(wav) is None


class DummyBackend:
    def __init__(self, *args, **kwargs):
        pass


class TestCheckCoverage:
    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)  # keep findpeak_cache_key off the real ~/.cache
        monkeypatch.setattr("tools.check_transient_coverage.LiveSpiceBackend", DummyBackend)
        self.schx = tmp_path / "device.schx"
        self.schx.write_text("dummy circuit")

    def _stub_onset(self, monkeypatch, onset_fn, calls=None):
        def fake(backend, params, scratch, max_v=40.0, lead_silence_s=0.0):
            if calls is not None:
                calls.append(dict(params))
            onset = onset_fn(params)
            if onset is None:
                return None
            return {"onset_99pct_input_v": onset, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("tools.check_transient_coverage.find_saturation_point", fake)

    def test_all_corners_passing_is_ok(self, monkeypatch):
        self._stub_onset(monkeypatch, lambda p: 0.1)
        result = check_coverage(str(self.schx), {"Gain": [0.0, 1.0]}, {}, oversample=8,
                                 transient_peak=1.0, quiet=True)
        assert result["ok"] is True
        assert all(r["ok"] for r in result["rows"])

    def test_a_corner_whose_onset_exceeds_the_transient_peak_fails(self, monkeypatch):
        self._stub_onset(monkeypatch, lambda p: 5.0 if p.get("Gain") == 1.0 else 0.1)
        result = check_coverage(str(self.schx), {"Gain": [0.0, 1.0]}, {}, oversample=8,
                                 transient_peak=1.0, quiet=True)
        assert result["ok"] is False
        failed = [r for r in result["rows"] if r["ok"] is False]
        assert failed and all(r["params"]["Gain"] == 1.0 for r in failed)

    def test_unresolved_onset_is_treated_as_not_passing(self, monkeypatch):
        self._stub_onset(monkeypatch, lambda p: None)
        result = check_coverage(str(self.schx), {"Gain": [0.0, 1.0]}, {}, oversample=8,
                                 transient_peak=1.0, quiet=True)
        assert result["ok"] is False
        assert all(r["ok"] is None for r in result["rows"])

    @pytest.mark.parametrize("margin,expected_ok", [(1.0, True), (1.5, False)])
    def test_margin_scales_the_required_onset(self, monkeypatch, margin, expected_ok):
        self._stub_onset(monkeypatch, lambda p: 1.0)
        result = check_coverage(str(self.schx), {"Gain": [0.5]}, {}, oversample=8,
                                 transient_peak=1.0, margin=margin, quiet=True)
        assert result["ok"] is expected_ok

    def test_a_cached_onset_is_not_recomputed(self, monkeypatch):
        calls = []
        self._stub_onset(monkeypatch, lambda p: 0.1, calls=calls)
        knob_ranges = {"Gain": [0.5]}
        check_coverage(str(self.schx), knob_ranges, {}, oversample=8, transient_peak=1.0, quiet=True)
        n_first = len(calls)
        assert n_first > 0
        check_coverage(str(self.schx), knob_ranges, {}, oversample=8, transient_peak=1.0, quiet=True)
        assert len(calls) == n_first, "second run with the same identity/params must hit the cache"


class TestCheckCoverageNgspiceDeck:
    """The hand-written-deck twin of TestCheckCoverage -- same _check_corners core, reached
    through check_coverage_ngspice_deck() instead. Only the backend construction/identity differ
    (NgspiceBackend + the module's own source bytes, vs. LiveSpiceBackend + the .schx bytes)."""

    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("tools.check_transient_coverage.NgspiceBackend", DummyBackend)
        self.module_file = tmp_path / "gen_fake_ngspice.py"
        self.module_file.write_text("KNOB_NAMES = ['Gain']\n")

    def _stub_onset(self, monkeypatch, onset_fn):
        def fake(backend, params, scratch, max_v=40.0, lead_silence_s=0.0):
            onset = onset_fn(params)
            if onset is None:
                return None
            return {"onset_99pct_input_v": onset, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("tools.check_transient_coverage.find_saturation_point", fake)

    def test_all_corners_passing_is_ok(self, monkeypatch):
        self._stub_onset(monkeypatch, lambda p: 0.1)
        result = check_coverage_ngspice_deck(build_deck=lambda **kw: None, module_file=str(self.module_file),
                                             probe_node="OUT", knob_ranges={"Gain": [0.0, 1.0]},
                                             fixed={}, transient_peak=1.0, quiet=True)
        assert result["ok"] is True

    def test_a_failing_corner_is_reported(self, monkeypatch):
        self._stub_onset(monkeypatch, lambda p: 5.0 if p.get("Gain") == 1.0 else 0.1)
        result = check_coverage_ngspice_deck(build_deck=lambda **kw: None, module_file=str(self.module_file),
                                             probe_node="OUT", knob_ranges={"Gain": [0.0, 1.0]},
                                             fixed={}, transient_peak=1.0, quiet=True)
        assert result["ok"] is False
        failed = [r for r in result["rows"] if r["ok"] is False]
        assert failed and all(r["params"]["Gain"] == 1.0 for r in failed)

    def test_identity_is_keyed_on_the_modules_own_source(self, monkeypatch):
        """Editing the deck (different source bytes) must not silently reuse a stale cached
        onset from before the edit -- same convention as preflight.py's _build_backend."""
        seen_identities = []

        def fake(backend, params, scratch, max_v=40.0, lead_silence_s=0.0):
            return {"onset_99pct_input_v": 0.1, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("tools.check_transient_coverage.find_saturation_point", fake)

        real_findpeak_cache_key = __import__("tools.check_transient_coverage",
                                             fromlist=["findpeak_cache_key"]).findpeak_cache_key

        def spy(identity_bytes, params, extra):
            seen_identities.append(identity_bytes)
            return real_findpeak_cache_key(identity_bytes, params, extra)
        monkeypatch.setattr("tools.check_transient_coverage.findpeak_cache_key", spy)

        check_coverage_ngspice_deck(build_deck=lambda **kw: None, module_file=str(self.module_file),
                                    probe_node="OUT", knob_ranges={"Gain": [0.5]}, fixed={},
                                    transient_peak=1.0, quiet=True)
        assert seen_identities and all(i == self.module_file.read_bytes() for i in seen_identities)


class TestMainNgspiceDeckDispatch:
    """A wiring-level check on main() itself: does an ngspice-deck config.toml's pedal-dir/
    module/probe-node actually reach check_coverage_ngspice_deck(), not silently fall through
    to the .schx path (which would then KeyError on a missing "schx" key, or worse, misread
    the config)."""

    def test_ngspice_deck_config_is_dispatched_correctly(self, tmp_path, monkeypatch):
        import sys as _sys

        import numpy as np
        import soundfile as sf

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("tools.check_transient_coverage.NgspiceBackend",
                            lambda *a, **kw: object())
        monkeypatch.setattr("tools.check_transient_coverage.find_saturation_point",
                            lambda backend, params, scratch, max_v=40.0, lead_silence_s=0.0:
                            {"onset_99pct_input_v": 0.1, "ceiling_rms": 1.0,
                             "ceiling_at_input_v": 1.0, "curve": []})

        (tmp_path / "gen_fake_ngspice.py").write_text(
            "KNOB_NAMES = ['Gain']\ndef build_deck(**kw): return ''\n")
        wav = tmp_path / "input.wav"
        sf.write(str(wav), np.zeros(100, dtype=np.float32), 48000)

        config = tmp_path / "config.toml"
        config.write_text(
            f'input = "{wav}"\n'
            'backend = "ngspice-deck"\n'
            f'pedal-dir = "{tmp_path}"\n'
            'module = "gen_fake_ngspice"\n'
            'probe-node = "OUT"\n'
            "\n[knobs]\nGain = [0.0, 1.0]\n"
        )
        monkeypatch.setattr(_sys, "argv", ["check_transient_coverage.py", "--config", str(config),
                                          "--transient-peak", "1.0"])
        assert main() == 0
