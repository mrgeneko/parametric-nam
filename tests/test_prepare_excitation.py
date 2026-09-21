"""Properties prepare_excitation.py must have. This tool closes the human-in-the-loop gap
between find_saturation_point.py and build_excitation.py by deriving --chirp-levels/
--sweep-peak directly from a MEASURED worst-case onset across the knob grid's corners --
worst_case_onset's "refuse to guess" behavior (raise rather than silently build against a
partial sweep) and its worst-case (not average, not first) selection are the properties an
excitation's whole calibration depends on. Exercised with find_saturation_point stubbed out.

See prepare_excitation.py.
"""
import sys
from pathlib import Path

import pytest

from prepare_excitation import method_summary, _parse_fixed, _parse_ranges, _setup, main, worst_case_onset
from render_backends import NgspiceSchxBackend


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
        def fake(backend, params, tmp, max_v=40.0, lead_silence_s=0.0, **kw):
            if calls is not None:
                calls.append(dict(params))
            onset = onset_fn(params)
            if onset is None:
                return None
            return {"onset_99pct_input_v": onset, "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0, "curve": []}
        monkeypatch.setattr("prepare_excitation.find_saturation_point", fake)

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


class TestMainSweepPeakVsCheckTransientCoverage:
    """The exact regression this session found: with --sweep-peak-frac below 1.0,
    --sweep-peak comes out LESS than worst-case onset by construction, which guarantees
    check_transient_coverage.py's own default gate (transient_peak >= onset at margin=1.0)
    FAILS at exactly the worst corner -- contradicting this tool's own docstring claim that a
    check run afterward should pass cleanly. --sweep-peak-frac's default (1.0) must not
    regress back below that line.

    (--real-clip/--realistic-peak/--realistic-peak-frac were renamed --sweep-file/--sweep-peak/
    --sweep-peak-frac on 2026-09-10, to match TONE3000's own "sweep signal" term for this
    style of file -- see build_excitation.py's docstring.)"""

    def _write_pedal_module(self, tmp_path, name="gen_fake_ngspice"):
        (tmp_path / f"{name}.py").write_text(
            "KNOB_NAMES = ['Gain']\ndef build_deck(**kw): return ''\n")

    def _run_main_and_capture_cmd(self, tmp_path, monkeypatch, worst_onset, extra_argv=()):
        self._write_pedal_module(tmp_path)
        monkeypatch.setattr("prepare_excitation.worst_case_onset",
                            lambda *a, **kw: (worst_onset, [{"corner": "worst", "onset_v": worst_onset}]))
        captured = {}

        def fake_run(cmd, check=True):
            captured["cmd"] = cmd
            return None
        monkeypatch.setattr("prepare_excitation.subprocess.run", fake_run)

        argv = ["prepare_excitation.py", "--backend", "ngspice-deck",
                "--pedal-dir", str(tmp_path), "--module", "gen_fake_ngspice",
                "--range", "Gain=0.0,1.0", "--sweep-file", "clip.wav",
                "--output", str(tmp_path / "out.wav"), *extra_argv]
        monkeypatch.setattr(sys, "argv", argv)
        main()
        return captured["cmd"]

    def _sweep_peak_from_cmd(self, cmd):
        return float(cmd[cmd.index("--sweep-peak") + 1])

    def test_default_sweep_peak_meets_or_exceeds_worst_case_onset(self, tmp_path, monkeypatch):
        cmd = self._run_main_and_capture_cmd(tmp_path, monkeypatch, worst_onset=5.0)
        assert self._sweep_peak_from_cmd(cmd) >= 5.0

    def test_sweep_peak_scales_with_the_explicit_frac(self, tmp_path, monkeypatch):
        cmd = self._run_main_and_capture_cmd(tmp_path, monkeypatch, worst_onset=5.0,
                                             extra_argv=["--sweep-peak-frac", "2.0"])
        assert self._sweep_peak_from_cmd(cmd) == pytest.approx(10.0)


class TestSetupLivespiceSolverIdentityInCache:
    """The findpeak cache key must be sensitive to the ORACLE'S OWN BUILD, not just the
    circuit and knob settings -- otherwise two machines on different livespice-cli revisions
    (or a rebuilt oracle on the same machine) silently collide on the same cache entry, and a
    fleet-wide onset-cache sync would mix incompatible measurements with no way to detect it.

    This is a REGRESSION test, not new coverage of new behavior: solver_identity() already
    rides inside cache_extra for every backend (see _setup()'s own f-string) -- confirmed
    2026-09-21 while designing a cross-machine findpeak-cache sync tool, which depends on this
    already being true. Pins it down explicitly so it can't silently regress.
    """

    def _args(self, schx, oversample=8, iterations=256, peak_max_v=40.0, min_start_v=1e-9,
             sweep_start_v=0.005):
        import types
        return types.SimpleNamespace(backend="livespice", schx=str(schx), range=["Fuzz=0.0,1.0"],
                                     config=None, oversample=oversample, iterations=iterations,
                                     peak_max_v=peak_max_v, min_start_v=min_start_v,
                                     sweep_start_v=sweep_start_v, fixed_params="")

    def test_cache_extra_embeds_the_solver_identity(self, tmp_path, monkeypatch):
        import prepare_excitation as pe
        monkeypatch.setattr(pe, "solver_identity", lambda backend: "livespice:deadbeef1234")
        schx = tmp_path / "fake.schx"
        schx.write_text("<Schematic></Schematic>")
        _, _, cache_extra, *_ = pe._setup(self._args(schx))
        assert "livespice:deadbeef1234" in cache_extra

    def test_different_solver_identity_produces_a_different_cache_extra(self, tmp_path, monkeypatch):
        import prepare_excitation as pe
        schx = tmp_path / "fake.schx"
        schx.write_text("<Schematic></Schematic>")
        monkeypatch.setattr(pe, "solver_identity", lambda backend: "livespice:aaaaaaa")
        _, _, a, *_ = pe._setup(self._args(schx))
        monkeypatch.setattr(pe, "solver_identity", lambda backend: "livespice:bbbbbbb")
        _, _, b, *_ = pe._setup(self._args(schx))
        assert a != b

    def test_different_solver_identity_produces_a_different_findpeak_cache_path(self, tmp_path, monkeypatch):
        """End to end: not just cache_extra as a string, but the actual on-disk cache file
        findpeak_cache_key() resolves to -- the property a cross-machine sync tool relies on
        to never merge two builds' entries under one key."""
        import prepare_excitation as pe
        from find_saturation_point import findpeak_cache_key
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        schx = tmp_path / "fake.schx"
        schx.write_text("<Schematic></Schematic>")

        monkeypatch.setattr(pe, "solver_identity", lambda backend: "livespice:aaaaaaa")
        _, identity, extra_a, *_ = pe._setup(self._args(schx))
        path_a = findpeak_cache_key(identity, {"Fuzz": 0.5}, extra_a)

        monkeypatch.setattr(pe, "solver_identity", lambda backend: "livespice:bbbbbbb")
        _, _, extra_b, *_ = pe._setup(self._args(schx))
        path_b = findpeak_cache_key(identity, {"Fuzz": 0.5}, extra_b)

        assert path_a != path_b


class TestSetupNgspiceBackend:
    """--backend ngspice (the GENERIC schx-translated path, added 2026-09-11 for a .schx
    circuit whose LiveSPICE render diverges under real signal, e.g. Arbiter Fuzz Face) must
    build an NgspiceSchxBackend, not silently fall through to LiveSpiceBackend or exit as an
    unknown backend."""

    def _args(self, schx=None, range_=None, config=None, oversample=None, peak_max_v=40.0,
             lead_silence_s=3.0, fixed_params="", conv=None, min_start_v=1e-9,
             sweep_start_v=0.005):
        import types
        return types.SimpleNamespace(backend="ngspice", schx=schx, range=range_ or [],
                                     config=config, oversample=oversample,
                                     peak_max_v=peak_max_v, lead_silence_s=lead_silence_s,
                                     fixed_params=fixed_params, conv=conv,
                                     min_start_v=min_start_v, sweep_start_v=sweep_start_v)

    def test_builds_an_ngspice_schx_backend_from_schx_and_range(self, tmp_path):
        schx = tmp_path / "fake.schx"
        schx.write_text("<Schematic></Schematic>")
        (backend, identity, cache_extra, knob_ranges, fixed, lead_silence_s, label,
         capture) = _setup(self._args(schx=str(schx), range_=["Fuzz=0.0,1.0"]))
        assert isinstance(backend, NgspiceSchxBackend)
        assert identity == schx.read_bytes()
        assert knob_ranges == {"Fuzz": [0.0, 1.0]}
        assert label == "fake.schx"

    def test_cache_extra_names_the_backend_so_it_cannot_collide_with_livespice(self, tmp_path):
        schx = tmp_path / "fake.schx"
        schx.write_text("<Schematic></Schematic>")
        _, _, cache_extra, *_ = _setup(self._args(schx=str(schx), range_=["Fuzz=0.0,1.0"]))
        assert "backend=ngspice" in cache_extra

    def test_missing_schx_or_range_exits(self):
        with pytest.raises(SystemExit):
            _setup(self._args(schx=None, range_=[]))

    def test_conv_reaches_the_backend_and_the_cache_key(self, tmp_path):
        """A device-model override (e.g. a real transistor fit) must reach the actual
        NgspiceSchxBackend AND the cache_extra key -- otherwise an onset measured under one
        --conv could be served to a caller expecting a different (or no) override."""
        schx = tmp_path / "fake.schx"
        schx.write_text("<Schematic></Schematic>")
        backend, _, cache_extra, *_ = _setup(self._args(
            schx=str(schx), range_=["Fuzz=0.0,1.0"], conv="bjt_vaf=102.207"))
        assert backend.conv == {"bjt_vaf": "102.207"}
        assert "bjt_vaf=102.207" in cache_extra

    def test_different_conv_produces_a_different_cache_key(self, tmp_path):
        schx = tmp_path / "fake.schx"
        schx.write_text("<Schematic></Schematic>")
        _, _, a, *_ = _setup(self._args(schx=str(schx), range_=["Fuzz=0.0,1.0"],
                                        conv="bjt_vaf=102.207"))
        _, _, b, *_ = _setup(self._args(schx=str(schx), range_=["Fuzz=0.0,1.0"], conv=None))
        assert a != b


class TestSizingProvenance:
    """The recipe must record HOW onsets were derived, not only what they were.

    On 2026-09-12 find_saturation_point's onset rule changed, and every recipe already on disk
    looked identical to a freshly-correct one -- no field distinguished them. Auditing nine
    devices came down to build dates plus a judgement about whether each peak looked physically
    plausible. That is forensics, not provenance.
    """

    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def _stub(self, monkeypatch, sat_fn):
        def fake(backend, params, tmp, max_v=40.0, lead_silence_s=0.0, **kw):
            return sat_fn(params)
        monkeypatch.setattr("prepare_excitation.find_saturation_point", fake)

    def test_rows_carry_the_method_and_knee_from_each_measurement(self, monkeypatch, tmp_path):
        self._stub(monkeypatch, lambda p: {"onset_99pct_input_v": 0.9, "knee_v": 0.053,
                                            "onset_method": "knee+sat95-v1",
                                            "ceiling_rms": 1.0, "ceiling_at_input_v": 1.0,
                                            "curve": []})
        _worst, rows = worst_case_onset(backend=object(), identity=b"x", cache_extra="e",
                                         knob_ranges={"Gain": [0.0, 1.0]}, fixed={},
                                         tmp=str(tmp_path), quiet=True)
        assert rows and all(r["method"] == "knee+sat95-v1" for r in rows)
        # knee and onset are DIFFERENT quantities and both must survive into the artifact
        assert all(r["knee_v"] == 0.053 and r["onset_v"] == 0.9 for r in rows)
        assert method_summary(rows) == "knee+sat95-v1"

    def test_a_pre_fix_measurement_is_named_not_left_blank(self, monkeypatch, tmp_path):
        """An old cache entry has no method field. Absence is not 'unknown' -- it identifies
        the 99%-of-max rule, which is exactly what a reader needs to see."""
        self._stub(monkeypatch, lambda p: {"onset_99pct_input_v": 21.24, "ceiling_rms": 1.0,
                                            "ceiling_at_input_v": 1.0, "curve": []})
        _worst, rows = worst_case_onset(backend=object(), identity=b"x", cache_extra="e",
                                         knob_ranges={"Gain": [0.0, 1.0]}, fixed={},
                                         tmp=str(tmp_path), quiet=True)
        assert all(r["method"] is None and r["knee_v"] is None for r in rows)
        assert method_summary(rows) == "pre-2026-09-12/99pct-of-max"

    def test_a_run_mixing_methods_is_reported_as_mixed(self):
        """Fresh measurements alongside cache hits from older code. Collapsing that to one
        method would assert a consistency the run does not have."""
        rows = [{"method": "knee+sat95-v1"}, {"method": None}]
        out = method_summary(rows)
        assert out.startswith("MIXED: ")
        assert "knee+sat95-v1" in out and "pre-2026-09-12/99pct-of-max" in out

    def test_summary_is_stable_regardless_of_corner_order(self):
        a = method_summary([{"method": None}, {"method": "knee+sat95-v1"}])
        b = method_summary([{"method": "knee+sat95-v1"}, {"method": None}])
        assert a == b, "provenance must not depend on which corner happened to be probed first"

    def test_empty_rows_do_not_crash_the_recipe_write(self):
        assert method_summary([]) == "pre-2026-09-12/99pct-of-max"
