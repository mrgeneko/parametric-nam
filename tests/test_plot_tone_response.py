"""Properties tools/plot_tone_response.py's measurement primitives must have. freq_response is
the actual measurement this whole chart is built on -- a wrong dB number here silently
mis-documents (or mis-scores against the schx oracle) a shipped model's tone-stack response,
not a crash. tiers_from_model's ratio/label derivation feeds every render call's --size
argument, so a wrong ratio renders the WRONG TIER's response and reports it as the right one.

See tools/plot_tone_response.py.
"""
import math

import numpy as np
import pytest

from tools.plot_tone_response import F0, F1, find_oracle_bin, find_render_bin, freq_response, make_sweep, tiers_from_model


class TestMakeSweep:
    def test_deconvolving_with_its_own_inverse_peaks_at_exactly_one(self):
        # The inverse filter is explicitly normalized "unity at the linear-IR peak" -- this is
        # what make_sweep's own contract promises, not an incidental property.
        from scipy.signal import fftconvolve
        x, inv = make_sweep(48000)
        conv = fftconvolve(x, inv)
        assert np.abs(conv).max() == pytest.approx(1.0, rel=1e-9)

    def test_length_matches_the_configured_duration(self):
        from tools.plot_tone_response import DUR
        x, _ = make_sweep(48000)
        assert len(x) == int(48000 * DUR)

    def test_deterministic(self):
        x1, inv1 = make_sweep(48000)
        x2, inv2 = make_sweep(48000)
        assert np.array_equal(x1, x2)
        assert np.array_equal(inv1, inv2)


@pytest.fixture(scope="module")
def sweep():
    return make_sweep(48000)


class TestFreqResponse:
    def test_recovers_a_gain_difference_in_db(self, sweep):
        x, inv = sweep
        sr = 48000
        fpts = np.logspace(math.log10(F0), math.log10(F1), 20)[:-1]  # drop the top edge bin
        h_unity = freq_response(x, inv, x, sr, fpts)
        h_double = freq_response(x, inv, (x * 2.0).astype(np.float32), sr, fpts)
        assert np.allclose(h_double - h_unity, 20 * math.log10(2.0), atol=0.05)

    def test_flat_for_an_identity_system_away_from_band_edges(self, sweep):
        x, inv = sweep
        sr = 48000
        fpts = np.logspace(math.log10(F0), math.log10(F1), 20)[:-1]
        h = freq_response(x, inv, x, sr, fpts)
        assert np.ptp(h) < 0.5  # dB


class TestTiersFromModel:
    def _write(self, tmp_path, submodels):
        import json
        p = tmp_path / "model.nam"
        p.write_text(json.dumps({"config": {"submodels": submodels}}))
        return p

    def test_ratio_is_midpoint_between_consecutive_breakpoints(self, tmp_path):
        subs = [
            {"max_value": 0.5, "model": {"config": {"layers": 3}}},
            {"max_value": 1.0, "model": {"config": {"layers": 8}}},
        ]
        tiers = tiers_from_model(str(self._write(tmp_path, subs)))
        assert tiers == [(0.25, "w3", 0.5), (0.75, "w8", 1.0)]

    def test_layers_given_as_a_list_uses_the_first_element(self, tmp_path):
        subs = [{"max_value": 1.0, "model": {"config": {"layers": [8, 8]}}}]
        tiers = tiers_from_model(str(self._write(tmp_path, subs)))
        assert tiers == [(0.5, "w8", 1.0)]

    def test_missing_layers_falls_back_to_a_positional_label(self, tmp_path):
        subs = [{"max_value": 1.0, "model": {"config": {}}}]
        tiers = tiers_from_model(str(self._write(tmp_path, subs)))
        assert tiers[0][1] == "tier0"

    def test_bare_model_with_no_submodels_raises(self, tmp_path):
        p = tmp_path / "bare.nam"
        import json
        p.write_text(json.dumps({"config": {}}))
        with pytest.raises(SystemExit):
            tiers_from_model(str(p))

    def test_three_tiers_breakpoints_are_ascending(self, tmp_path):
        subs = [
            {"max_value": 1 / 3, "model": {"config": {"layers": 3}}},
            {"max_value": 2 / 3, "model": {"config": {"layers": 5}}},
            {"max_value": 1.0, "model": {"config": {"layers": 8}}},
        ]
        tiers = tiers_from_model(str(self._write(tmp_path, subs)))
        ratios = [t[0] for t in tiers]
        assert ratios == sorted(ratios)


class TestFindRenderBin:
    def test_explicit_path_wins_when_it_exists(self, tmp_path, monkeypatch):
        real = tmp_path / "render_parametric"
        real.write_text("#!/bin/sh\n")
        monkeypatch.delenv("RENDER_PARAMETRIC", raising=False)
        assert find_render_bin(str(real)) == str(real)

    def test_falls_back_to_env_var_when_explicit_is_missing(self, tmp_path, monkeypatch):
        real = tmp_path / "render_parametric"
        real.write_text("x")
        monkeypatch.setenv("RENDER_PARAMETRIC", str(real))
        assert find_render_bin(None) == str(real)

    def test_falls_back_to_which_when_nothing_else_matches(self, monkeypatch):
        monkeypatch.delenv("RENDER_PARAMETRIC", raising=False)
        monkeypatch.setattr("tools.plot_tone_response.shutil.which", lambda name: "/usr/local/bin/render_parametric")
        assert find_render_bin(None) == "/usr/local/bin/render_parametric"

    def test_returns_none_when_nothing_is_found(self, monkeypatch):
        monkeypatch.delenv("RENDER_PARAMETRIC", raising=False)
        monkeypatch.setattr("tools.plot_tone_response.shutil.which", lambda name: None)
        assert find_render_bin(None) is None


class TestFindOracleBin:
    def test_explicit_path_wins(self, tmp_path):
        real = tmp_path / "livespice_cli"
        real.write_text("x")
        assert find_oracle_bin(str(real)) == str(real)

    def test_returns_none_when_nothing_found(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LIVESPICE_CLI", raising=False)
        monkeypatch.setattr("tools.plot_tone_response.shutil.which", lambda name: None)
        monkeypatch.setattr("tools.plot_tone_response.Path.home", lambda: tmp_path)
        assert find_oracle_bin(None) is None
