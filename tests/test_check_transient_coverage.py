"""Properties check_transient_coverage.py must have. Built after a real miss: the tweed-style
amp's shipped excitation cleared saturation at its own overall peak, but the transient
(real-playing) segment never did at a corner where two knobs sat at grid-min while three
others sat at grid-max simultaneously -- a corner the "solo extreme" set literally cannot
represent (solo holds every OTHER knob at center). _corners' binary hypercube exists
specifically to catch that. check_coverage is exercised with find_saturation_point/
LiveSpiceBackend stubbed out, so the per-corner pass/fail/skip bookkeeping is tested
independently of any real circuit render.

See check_transient_coverage.py.
"""
import json
from pathlib import Path

import pytest

from check_transient_coverage import (
    _corners,
    _transient_peak_from_recipe,
    check_coverage,
    check_coverage_ngspice_deck,
    check_coverage_ltspice_deck,
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
        """The exact tweed-style-amp miss: NormalVol/BrightVol at grid-min while Treble/Bass/Middle
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
        monkeypatch.setattr("check_transient_coverage.LiveSpiceBackend", DummyBackend)
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
        monkeypatch.setattr("check_transient_coverage.find_saturation_point", fake)

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
        monkeypatch.setattr("check_transient_coverage.NgspiceBackend", DummyBackend)
        self.module_file = tmp_path / "gen_fake_ngspice.py"
        self.module_file.write_text("KNOB_NAMES = ['Gain']\n")

    def _stub_onset(self, monkeypatch, onset_fn):
        def fake(backend, params, scratch, max_v=40.0, lead_silence_s=0.0):
            onset = onset_fn(params)
            if onset is None:
                return None
            return {"onset_99pct_input_v": onset, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("check_transient_coverage.find_saturation_point", fake)

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
        monkeypatch.setattr("check_transient_coverage.find_saturation_point", fake)

        real_findpeak_cache_key = __import__("check_transient_coverage",
                                             fromlist=["findpeak_cache_key"]).findpeak_cache_key

        def spy(identity_bytes, params, extra):
            seen_identities.append(identity_bytes)
            return real_findpeak_cache_key(identity_bytes, params, extra)
        monkeypatch.setattr("check_transient_coverage.findpeak_cache_key", spy)

        check_coverage_ngspice_deck(build_deck=lambda **kw: None, module_file=str(self.module_file),
                                    probe_node="OUT", knob_ranges={"Gain": [0.5]}, fixed={},
                                    transient_peak=1.0, quiet=True)
        assert seen_identities and all(i == self.module_file.read_bytes() for i in seen_identities)


class TestCheckCoverageLtspiceDeck:
    """The LTspice-hand-deck twin of TestCheckCoverageNgspiceDeck -- same _check_corners core,
    reached through check_coverage_ltspice_deck() instead. Only the backend construction
    differs (LtspiceBackend, keyed on `tap` instead of `probe_node`)."""

    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("check_transient_coverage.LtspiceBackend", DummyBackend)
        self.module_file = tmp_path / "gen_fake_ltspice.py"
        self.module_file.write_text("KNOB_NAMES = ['Gain']\n")

    def _stub_onset(self, monkeypatch, onset_fn):
        def fake(backend, params, scratch, max_v=40.0, lead_silence_s=0.0):
            onset = onset_fn(params)
            if onset is None:
                return None
            return {"onset_99pct_input_v": onset, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("check_transient_coverage.find_saturation_point", fake)

    def test_all_corners_passing_is_ok(self, monkeypatch):
        self._stub_onset(monkeypatch, lambda p: 0.1)
        result = check_coverage_ltspice_deck(build_deck=lambda **kw: None, module_file=str(self.module_file),
                                             tap="spk", knob_ranges={"Gain": [0.0, 1.0]},
                                             fixed={}, transient_peak=1.0, quiet=True)
        assert result["ok"] is True

    def test_a_failing_corner_is_reported(self, monkeypatch):
        self._stub_onset(monkeypatch, lambda p: 5.0 if p.get("Gain") == 1.0 else 0.1)
        result = check_coverage_ltspice_deck(build_deck=lambda **kw: None, module_file=str(self.module_file),
                                             tap="spk", knob_ranges={"Gain": [0.0, 1.0]},
                                             fixed={}, transient_peak=1.0, quiet=True)
        assert result["ok"] is False
        failed = [r for r in result["rows"] if r["ok"] is False]
        assert failed and all(r["params"]["Gain"] == 1.0 for r in failed)

    def test_identity_is_keyed_on_the_modules_own_source(self, monkeypatch):
        seen_identities = []

        def fake(backend, params, scratch, max_v=40.0, lead_silence_s=0.0):
            return {"onset_99pct_input_v": 0.1, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("check_transient_coverage.find_saturation_point", fake)

        real_findpeak_cache_key = __import__("check_transient_coverage",
                                             fromlist=["findpeak_cache_key"]).findpeak_cache_key

        def spy(identity_bytes, params, extra):
            seen_identities.append(identity_bytes)
            return real_findpeak_cache_key(identity_bytes, params, extra)
        monkeypatch.setattr("check_transient_coverage.findpeak_cache_key", spy)

        check_coverage_ltspice_deck(build_deck=lambda **kw: None, module_file=str(self.module_file),
                                    tap="spk", knob_ranges={"Gain": [0.5]}, fixed={},
                                    transient_peak=1.0, quiet=True)
        assert seen_identities and all(i == self.module_file.read_bytes() for i in seen_identities)


class TestMainLtspiceDeckDispatch:
    """A wiring-level check on main() itself: does an ltspice-deck config.toml's pedal-dir/
    module/probe-node actually reach check_coverage_ltspice_deck(), not silently fall through
    to the .schx path."""

    def test_ltspice_deck_config_is_dispatched_correctly(self, tmp_path, monkeypatch):
        import sys as _sys

        import numpy as np
        import soundfile as sf

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("check_transient_coverage.LtspiceBackend",
                            lambda *a, **kw: object())
        monkeypatch.setattr("check_transient_coverage.find_saturation_point",
                            lambda backend, params, scratch, max_v=40.0, lead_silence_s=0.0:
                            {"onset_99pct_input_v": 0.1, "ceiling_rms": 1.0,
                             "ceiling_at_input_v": 1.0, "curve": []})

        (tmp_path / "gen_fake_ltspice.py").write_text(
            "KNOB_NAMES = ['Gain']\ndef build_deck(**kw): return ''\n")
        wav = tmp_path / "input.wav"
        sf.write(str(wav), np.zeros(100, dtype=np.float32), 48000)

        config = tmp_path / "config.toml"
        config.write_text(
            f'input = "{wav}"\n'
            'backend = "ltspice-deck"\n'
            f'pedal-dir = "{tmp_path}"\n'
            'module = "gen_fake_ltspice"\n'
            'probe-node = "spk"\n'
            "\n[knobs]\nGain = [0.0, 1.0]\n"
        )
        monkeypatch.setattr(_sys, "argv", ["check_transient_coverage.py", "--config", str(config),
                                          "--transient-peak", "1.0"])
        assert main() == 0


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
        monkeypatch.setattr("check_transient_coverage.NgspiceBackend",
                            lambda *a, **kw: object())
        monkeypatch.setattr("check_transient_coverage.find_saturation_point",
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


# ----------------------------------------------------------------- corner-set budget
#
# The structural-only set (--no-full-hypercube) holds every OTHER knob at CENTER while moving
# one, so a mixed some-knobs-low-others-high corner is unreachable BY CONSTRUCTION. That blind
# spot shipped the tweed blowup, and on 2026-09-04 sized Duke of Tone's excitation short at
# three Gain=lo,Volume=lo corners. max_corners is the replacement: a budget that keeps the
# structural corners AND fills from the hypercube, sampling deterministically when it does not
# all fit -- so it degrades gracefully instead of going blind.

import io
import contextlib
from check_transient_coverage import _corners


def _ranges(n):
    return {f"K{i}": [0.2, 0.5, 0.8] for i in range(n)}


def _mixed(labels):
    return [l for l in labels if "=lo" in l and "=hi" in l and "solo" not in l]


def test_small_configs_get_the_whole_hypercube_by_default():
    assert len(_corners(_ranges(4))) == 25          # 16 hypercube + all-min/max/center + 8 solos
    assert len(_corners(_ranges(5))) == 43


def test_nine_knobs_still_allowed_by_default_as_before():
    # max_full_corners bounds the HYPERCUBE portion (its "512, i.e. up to 9 knobs" doc counts
    # 2**n only), NOT the total -- conflating them would silently reject configs that work today.
    assert len(_corners(_ranges(9))) == 512 + 19


def test_over_cap_without_a_budget_raises_and_points_at_max_corners():
    with pytest.raises(ValueError, match="max_corners"):
        _corners(_ranges(10))


def test_max_corners_bounds_the_total_and_is_deterministic():
    a = [l for l, _ in _corners(_ranges(16), max_corners=48)]
    b = [l for l, _ in _corners(_ranges(16), max_corners=48)]
    assert len(a) == 48
    assert a == b, "two tools sizing/checking the same config must agree on the corner set"


def test_budget_still_reaches_mixed_corners_where_the_deprecated_set_never_does():
    budgeted = [l for l, _ in _corners(_ranges(16), max_corners=48)]
    with contextlib.redirect_stderr(io.StringIO()):
        reduced = [l for l, _ in _corners(_ranges(16), full_hypercube=False)]
    assert len(_mixed(budgeted)) > 0
    assert len(_mixed(reduced)) == 0, "the deprecated set is structurally blind to mixed corners"


def test_deprecated_reduced_set_warns():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        _corners(_ranges(4), full_hypercube=False)
    assert "max_corners" in err.getvalue()


# ------------------------------------------------------------- interior grid sampling
#
# Corners are a heuristic, and Mesa Dual Rectifier ORANGE proved on 2026-09-04 that they are
# not sufficient in principle: its highest saturation onset (23.177 V, at Bass=min with every
# OTHER knob at its centre grid value) is 27% above the highest of all 32 hypercube vertices
# (18.232 V). Onset is not monotonic in the knobs, so its maximum over the box need not sit at
# a vertex, and vertex enumeration cannot find it. --sample-grid buys bounded coverage of the
# interior; probing all 648 points would guarantee it at ~9h per channel, which is not worth
# paying before every render to choose one scalar.

from check_transient_coverage import _sample_interior


def _grid():
    return {"Gain": [0.1, 0.35, 0.5, 0.8, 1.0], "Master": [0.1, 0.35, 0.5, 0.8, 1.0],
            "Bass": [0.2, 0.8], "Mid": [0.2, 0.8], "Treble": [0.2, 0.8]}


def test_sampling_adds_the_requested_number_of_points():
    base = _corners(_grid())
    got = _sample_interior(_grid(), list(base), 20)
    assert len(got) == len(base) + 20


def test_sampled_points_reach_values_no_corner_can():
    # The whole point: corners only ever use each knob's min or max (plus centre). A sampled
    # point may use ANY grid value, which is where Orange's true worst onset lived.
    base = _corners(_grid())
    sampled = _sample_interior(_grid(), list(base), 40)[len(base):]
    mids = {0.35, 0.8}          # Gain/Master values that are neither min nor max
    assert any(v["Gain"] in mids or v["Master"] in mids for _, v in sampled)


def test_sampling_is_deterministic_so_sizing_and_checking_agree():
    a = [l for l, _ in _sample_interior(_grid(), list(_corners(_grid())), 20)]
    b = [l for l, _ in _sample_interior(_grid(), list(_corners(_grid())), 20)]
    assert a == b


def test_sampling_never_duplicates_an_existing_corner():
    base = _corners(_grid())
    got = _sample_interior(_grid(), list(base), 30)
    keys = [tuple(sorted(v.items())) for _, v in got]
    assert len(keys) == len(set(keys))


def test_zero_is_a_no_op():
    base = _corners(_grid())
    assert len(_sample_interior(_grid(), list(base), 0)) == len(base)


def test_structural_corners_are_deduped_on_an_even_cardinality_axis():
    # `mid` is the nearest-to-centre GRID POINT, so on a 2-value axis it picks the MAX and
    # that axis's hi-solo becomes bit-identical to centre. Mesa Dual Rectifier probed the
    # same setting four times (center + Bass/Mid/Treble hi-solo), all returning 14.041 V.
    grid = {"Gain": [0.1, 0.5, 1.0], "Bass": [0.2, 0.8], "Mid": [0.2, 0.8]}
    got = _corners(grid)
    keys = [tuple(sorted(v.items())) for _, v in got]
    assert len(keys) == len(set(keys)), "an onset sweep is expensive; never probe one twice"


def test_every_backend_branch_forwards_the_corner_selection_flags():
    """--max-corners and --sample-grid must reach ALL THREE backend branches.

    They did not: when --max-corners was added, the ngspice-deck and ltspice-deck call
    sites were updated and the LIVESPICE one -- the primary backend -- was missed, because
    its formatting differed and a blind string replace did not match it. The flag parsed,
    documented itself in --help, and was silently ignored on the path almost every device
    uses. Caught only when --sample-grid 40 came back reporting 25 corners.

    Asserting on the source text is crude, but the alternative is driving main() with a
    real config and a real oracle, and a silently-dropped kwarg is exactly the failure a
    cheap check catches.
    """
    import inspect
    import check_transient_coverage as m
    src = inspect.getsource(m)
    for entry in ("check_coverage(", "check_coverage_ngspice_deck(", "check_coverage_ltspice_deck("):
        i = src.index("result = " + entry)
        call = src[i:src.index(")\n", i) + 1]
        assert "max_corners=args.max_corners" in call, f"{entry} drops --max-corners"
        assert "sample_grid=args.sample_grid" in call, f"{entry} drops --sample-grid"
