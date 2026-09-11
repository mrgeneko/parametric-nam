"""Properties the virtual capture chain must have.

Context: Duke of Tone (Distortion), 2026-09-10. 64-74% of its rendered target energy sat
below 19 Hz -- real bias-rail wander (tau 2.0-2.35 s) that a ~52 ms receptive field cannot
model. Both our trainer and the official upstream nam-full plateaued at the same ESR.
A hardware capture never contains this: the interface rolls it off first.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capture_chain import (capture_chain, lf_energy_fraction, context_sensitivity,
                           DEFAULT_CORNER_HZ, DEFAULT_ORDER)

SR = 48000


def _tone(f, dur=2.0, amp=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float64)


class TestItRemovesWhatCannotBeModelled:
    def test_sub_audio_is_attenuated_hard(self):
        """1 Hz must be crushed -- that is the content with multi-second state."""
        y = _tone(1.0, dur=8.0)
        out = capture_chain(y, SR)
        settled = out[4 * SR:]           # past the filter's own transient
        assert np.abs(settled).max() < 0.05 * np.abs(y).max()

    def test_audio_band_is_essentially_untouched(self):
        """The spec this filter is derived from: ~flat above 20 Hz."""
        for f in (100.0, 440.0, 5000.0):
            y = _tone(f, dur=1.0)
            out = capture_chain(y, SR)[SR // 2:]
            ref = y[SR // 2:]
            ratio = np.sqrt((out ** 2).mean()) / np.sqrt((ref ** 2).mean())
            assert 0.99 < ratio < 1.01, f"{f} Hz changed by {20*np.log10(ratio):+.2f} dB"

    def test_preserves_bass_low_e(self):
        """THE governing rule: bass low E (41.2 Hz) must survive within -0.5 dB.

        It is the lowest note anyone plays through these models, so it is the thing the
        filter must not damage. This replaced an earlier anchor of "-0.5 dB at 20 Hz",
        which was a converter datasheet standing in for the musical constraint -- a proxy,
        and one that would have blocked the 3rd @ 18 Hz default that is strictly better.
        """
        from capture_chain import PRESERVE_HZ, PRESERVE_MAX_LOSS_DB
        y = _tone(PRESERVE_HZ, dur=8.0)
        out = capture_chain(y, SR)[4 * SR:]
        loss_db = 20 * np.log10(np.sqrt((out ** 2).mean())
                                / np.sqrt((y[4 * SR:] ** 2).mean()))
        assert loss_db > -PRESERVE_MAX_LOSS_DB, f"low E lost {loss_db:.3f} dB"

    def test_any_override_must_also_preserve_low_e(self):
        """The invariant binds per-device overrides too, not just the default.

        Bounds under the -0.5 dB rule: 1st<=14.4, 2nd<=24.4, 3rd<=29.0, 4th<=31.7 Hz.
        A mic'd-cab-style corner (~70 Hz) violates it by tens of dB -- and would also be
        wrong to bake into a model meant to stay composable with the user's own cab.
        """
        from capture_chain import PRESERVE_HZ, PRESERVE_MAX_LOSS_DB
        y = _tone(PRESERVE_HZ, dur=8.0)
        ref = np.sqrt((y[4 * SR:] ** 2).mean())

        def loss(fc, order):
            out = capture_chain(y, SR, corner_hz=fc, order=order)[4 * SR:]
            return 20 * np.log10(np.sqrt((out ** 2).mean()) / ref)

        for fc, order in ((14.4, 1), (24.4, 2), (29.0, 3), (31.7, 4)):
            assert loss(fc, order) > -PRESERVE_MAX_LOSS_DB - 0.05, f"{order} @ {fc} Hz"
        assert loss(70.0, 2) < -6.0, "a cab-style corner should be obviously disqualified"

    def test_default_beats_the_previous_default_on_every_axis(self):
        """3rd @ 18 Hz vs the retired 2nd @ 11.8 Hz: better rejection, same low-E cost.

        Pins the reason for the change. Raising ORDER buys stopband rejection; raising the
        CORNER buys it by trading away bass. If someone later 'simplifies' this back to a
        2nd-order default, the rejection column regresses ~18 dB at 5 Hz for nothing.
        """
        from capture_chain import PRESERVE_HZ
        for f in (1.0, 5.0):
            y = _tone(f, dur=20.0)
            ref = np.sqrt((y[10 * SR:] ** 2).mean())
            new = np.sqrt((capture_chain(y, SR)[10 * SR:] ** 2).mean()) / ref
            old = np.sqrt((capture_chain(y, SR, corner_hz=11.8, order=2)[10 * SR:] ** 2).mean()) / ref
            assert new < old, f"{f} Hz: default rejects less than the old 2nd @ 11.8 Hz"
        y = _tone(PRESERVE_HZ, dur=8.0)
        ref = np.sqrt((y[4 * SR:] ** 2).mean())
        new = 20 * np.log10(np.sqrt((capture_chain(y, SR)[4 * SR:] ** 2).mean()) / ref)
        assert new > -0.1, f"low E cost {new:.3f} dB -- must stay negligible"

    def test_dc_is_removed(self):
        y = np.full(int(SR * 4), 0.5)
        assert abs(capture_chain(y, SR)[2 * SR:].mean()) < 1e-3


class TestCausality:
    def test_filter_is_causal(self):
        """A zero-phase filter would make the target depend on FUTURE input -- its
        pre-ringing is unpredictable to a causal model, a self-inflicted error floor."""
        y = np.zeros(SR)
        y[SR // 2] = 1.0                      # lone impulse halfway through
        out = capture_chain(y, SR)
        assert np.abs(out[:SR // 2]).max() == 0.0, "output moved BEFORE the impulse"

    def test_impulse_response_fits_inside_the_receptive_field(self):
        """~2483 samples (51.7 ms). If the filter rang longer, it would itself be
        unlearnable -- removing one error floor by adding another."""
        y = np.zeros(SR); y[0] = 1.0
        out = np.abs(capture_chain(y, SR))
        assert out[2483:].max() < 0.01 * out.max()


class TestShape:
    def test_handles_2d_datasets_along_last_axis(self):
        """outputs.npy is (n_combos, n_samples); filter in one call, no caller loop."""
        rows = np.stack([_tone(1.0, dur=1.0), _tone(440.0, dur=1.0)])
        out = capture_chain(rows, SR)
        assert out.shape == rows.shape
        assert np.allclose(out[0], capture_chain(rows[0], SR))
        assert np.allclose(out[1], capture_chain(rows[1], SR))

    def test_dtype_is_preserved(self):
        y = _tone(440.0, dur=0.5).astype(np.float32)
        assert capture_chain(y, SR).dtype == np.float32

    def test_rejects_corner_above_nyquist(self):
        with pytest.raises(ValueError):
            capture_chain(_tone(440.0, dur=0.1), SR, corner_hz=SR)


class TestDiagnostics:
    def test_lf_fraction_separates_the_duke_from_the_fleet(self):
        """Fleet sits at 0.20-9.47%; the Distortion measured 64-74%."""
        healthy = _tone(440.0, dur=4.0) + 0.05 * _tone(2.0, dur=4.0)
        sick = _tone(440.0, dur=4.0) + 4.0 * _tone(2.0, dur=4.0)
        assert lf_energy_fraction(healthy, SR) < 0.10
        assert lf_energy_fraction(sick, SR) > 0.50

    def test_capture_chain_collapses_the_lf_fraction(self):
        sick = _tone(440.0, dur=8.0) + 4.0 * _tone(2.0, dur=8.0)
        before = lf_energy_fraction(sick, SR)
        after = lf_energy_fraction(capture_chain(sick, SR)[4 * SR:], SR)
        assert before > 0.50 and after < 0.02, f"{before:.3f} -> {after:.3f}"

    def test_context_sensitivity_is_zero_for_identical_renders(self):
        y = _tone(440.0, dur=2.0)
        assert context_sensitivity(y, y, SR) == pytest.approx(0.0, abs=1e-12)

    def test_context_sensitivity_flags_a_lingering_offset(self):
        """The class an LF check alone cannot catch: state that outlives the window."""
        base = _tone(440.0, dur=2.0)
        drifted = base + 0.3 * _tone(0.3, dur=2.0)
        assert context_sensitivity(drifted, base, SR) > 0.01


class TestRenderPathWiring:
    """gen_dataset_from_schx must apply the chain, and must DECLARE it in config.json.

    A dataset that does not record its capture chain is indistinguishable from one
    rendered without it, so a later reader silently compares incomparable targets.
    """

    def _args(self, **kw):
        import types
        d = {"no_capture_chain": False, "capture_hp_hz": None, "capture_order": None}
        d.update(kw)
        return types.SimpleNamespace(**d)

    def test_enabled_by_default(self):
        from gen_dataset_from_schx import _capture_cfg
        cfg = _capture_cfg(self._args())
        assert cfg == {"corner_hz": DEFAULT_CORNER_HZ, "order": DEFAULT_ORDER}

    def test_opt_out_yields_none(self):
        from gen_dataset_from_schx import _capture_cfg
        assert _capture_cfg(self._args(no_capture_chain=True)) is None

    def test_overrides_are_honoured(self):
        from gen_dataset_from_schx import _capture_cfg
        assert _capture_cfg(self._args(capture_hp_hz=5.0, capture_order=1)) == \
            {"corner_hz": 5.0, "order": 1}

    def test_cfg_is_json_serialisable(self):
        """It is written verbatim into config.json and shipped across the worker pool."""
        import json
        from gen_dataset_from_schx import _capture_cfg
        assert json.loads(json.dumps(_capture_cfg(self._args()))) == _capture_cfg(self._args())

    def test_finalize_applies_chain_and_restates_stats(self, tmp_path):
        """rms/peak must describe the SAVED signal: combine() sets output_scale from
        params.csv's `peak`, so raw stats beside filtered data mis-scale the dataset."""
        import soundfile as sf
        from gen_dataset_from_schx import _finalize_wav
        sr = SR
        sig = (_tone(440.0, dur=3.0, amp=0.3) + 1.5 * _tone(2.0, dur=3.0)).astype(np.float32)
        raw_peak = float(np.abs(sig[sr:]).max())
        out_wav = tmp_path / "r.wav"; sf.write(str(out_wav), sig, sr, subtype="FLOAT")
        path = tmp_path / "r.npy"
        res = _finalize_wav(0, path, out_wav, 0, 0.0, warmup_s=1.0,
                            capture={"corner_hz": DEFAULT_CORNER_HZ, "order": DEFAULT_ORDER})
        assert res.ok, res.error
        saved = np.load(path)
        assert lf_energy_fraction(saved[sr:], sr) < 0.05, "chain was not applied to saved data"
        assert res.peak < raw_peak * 0.9, "stats still describe the RAW signal"
        assert res.peak == pytest.approx(float(np.abs(saved[sr:]).max()), rel=1e-6)

    def test_finalize_without_capture_is_unchanged(self, tmp_path):
        import soundfile as sf
        from gen_dataset_from_schx import _finalize_wav
        sig = (_tone(440.0, dur=2.0, amp=0.3)).astype(np.float32)
        out_wav = tmp_path / "r2.wav"; sf.write(str(out_wav), sig, SR, subtype="FLOAT")
        path = tmp_path / "r2.npy"
        res = _finalize_wav(0, path, out_wav, 0, 0.0, warmup_s=1.0, capture=None)
        assert res.ok
        assert np.allclose(np.load(path), sig)


class TestMeasurementPathsUseTheChain:
    """A measurement characterises what the MODEL must learn, so it must see the same
    signal the dataset stores -- otherwise find_saturation_point watches output RMS stop
    rising while most of that RMS is sub-audio bias wander, and reports the wander's
    behaviour as the circuit's saturation onset.
    """

    def test_find_saturation_point_measures_through_the_chain(self):
        """Onset must be judged on audio-band content, not on a sub-audio pedestal.

        The fake circuit below is a linear, NON-saturating path plus a large constant-
        amplitude 2 Hz pedestal. Raw, the pedestal dominates RMS at every drive level, so
        the curve looks FLAT and a bogus 'onset' is read off it. Through the chain the
        pedestal is gone and the true linear ramp is visible -- no onset, correctly.
        """
        from find_saturation_point import find_saturation_point

        class Pedestal:
            def prepare_input(self, raw, sr, level_v, scratch, tag):
                return (raw, level_v)

            def render_many(self, jobs, handle, scratch):
                raw, level = handle
                n = len(raw)
                t = np.arange(n) / SR
                # gain chosen so the 2 Hz pedestal still dominates RAW rms at every
                # level (curve looks flat) while the chained residual -- 2 Hz through
                # a 2nd-order 11.8 Hz corner is ~31x down -- leaves clear headroom.
                y = 0.02 * level * raw + 3.0 * np.sin(2 * np.pi * 2.0 * t)
                return {j["tag"]: y.astype(np.float32) for j in jobs}

        raw_sat = find_saturation_point(Pedestal(), {}, "/tmp/unused", dur=1.0,
                                        npoints=8, workers=4)
        chained = find_saturation_point(Pedestal(), {}, "/tmp/unused", dur=1.0,
                                        npoints=8, workers=4,
                                        capture={"corner_hz": DEFAULT_CORNER_HZ,
                                                 "order": DEFAULT_ORDER})
        raw_curve = [r for _, r in raw_sat["curve"]]
        chained_curve = [r for _, r in chained["curve"]]
        assert max(raw_curve) / min(raw_curve) < 1.10, "raw curve should be pedestal-flat"
        assert max(chained_curve) / min(chained_curve) > 5.0, \
            "chained curve should reveal the linear ramp the pedestal was hiding"


class TestCacheKeyCarriesTheChain:
    """cache_extra MUST distinguish chained from raw measurements.

    On 2026-09-10 a cached FAILURE silently defeated a verified fix to the sweep for two
    full runs because the key could not tell the two apart. A chained-vs-raw collision is
    the same bug with a quieter symptom: a plausible onset measured on the wrong signal.
    """

    def test_tag_differs_between_on_and_off(self):
        from capture_chain import cache_tag
        assert cache_tag(None) != cache_tag({"corner_hz": DEFAULT_CORNER_HZ, "order": DEFAULT_ORDER})

    def test_tag_differs_between_settings(self):
        from capture_chain import cache_tag
        assert cache_tag({"corner_hz": 11.8, "order": 2}) != cache_tag({"corner_hz": 5.0, "order": 2})
        assert cache_tag({"corner_hz": 11.8, "order": 2}) != cache_tag({"corner_hz": 11.8, "order": 1})

    def test_tag_is_stable_for_equal_settings(self):
        from capture_chain import cache_tag
        assert cache_tag({"corner_hz": 11.8, "order": 2}) == cache_tag({"corner_hz": 11.80, "order": 2})

    @pytest.mark.parametrize("mod", ["preflight", "prepare_excitation", "check_transient_coverage"])
    def test_every_cache_extra_includes_the_tag(self, mod):
        """Guards against a new backend branch being added without the tag."""
        import inspect, importlib, re
        src = inspect.getsource(importlib.import_module(mod))
        # only CONSTRUCTION sites; `a, b, cache_extra = _build_backend(args)` is a
        # destructuring assignment, not a key being built.
        sites = re.findall(r'cache_extra = f".*', src)
        assert sites, f"{mod}: no cache_extra found -- did it move?"
        for line in sites:
            assert "cache_tag(" in line, f"{mod}: cache_extra without the chain tag: {line.strip()}"


class TestDeclarationAndGuard:
    """Recording the chain in config.json is necessary but not sufficient -- nobody reads a
    JSON file before wondering why an ESR moved. It must be announced, and a resume must
    refuse to continue against re-rendered targets.
    """

    def test_describe_names_the_settings(self):
        from capture_chain import describe
        d = describe({"corner_hz": 11.8, "order": 2})
        assert "11.8" in d and "2nd-order" in d
        assert "DISABLED" in describe(None)

    def test_read_dataset_chain_roundtrips(self, tmp_path):
        import json
        from capture_chain import read_dataset_chain
        (tmp_path / "config.json").write_text(json.dumps(
            {"knobs": [], "capture_chain": {"corner_hz": 11.8, "order": 2}}))
        assert read_dataset_chain(tmp_path) == {"corner_hz": 11.8, "order": 2}

    def test_explicit_none_is_distinct_from_missing(self, tmp_path):
        """'rendered raw on purpose' and 'predates the field' must not collapse together:
        the first is a real mismatch against a chained run, the second is unprovable."""
        import json
        from capture_chain import read_dataset_chain
        a = tmp_path / "a"; a.mkdir()
        (a / "config.json").write_text(json.dumps({"capture_chain": None}))
        b = tmp_path / "b"; b.mkdir()
        (b / "config.json").write_text(json.dumps({"knobs": []}))
        assert read_dataset_chain(a) is None
        assert read_dataset_chain(b) == "unknown"

    def test_missing_config_is_unknown_not_a_crash(self, tmp_path):
        from capture_chain import read_dataset_chain
        assert read_dataset_chain(tmp_path / "nope") == "unknown"

    def test_corrupt_config_is_unknown_not_a_crash(self, tmp_path):
        from capture_chain import read_dataset_chain
        (tmp_path / "config.json").write_text("{ this is not json")
        assert read_dataset_chain(tmp_path) == "unknown"

    def test_mismatch_detects_on_vs_off(self):
        from capture_chain import mismatch_reason
        assert mismatch_reason({"corner_hz": 11.8, "order": 2}, None)
        assert mismatch_reason(None, {"corner_hz": 11.8, "order": 2})

    def test_mismatch_detects_different_settings(self):
        from capture_chain import mismatch_reason
        assert "corner_hz" in mismatch_reason({"corner_hz": 11.8, "order": 2},
                                              {"corner_hz": 5.0, "order": 2})
        assert "order" in mismatch_reason({"corner_hz": 11.8, "order": 2},
                                          {"corner_hz": 11.8, "order": 1})

    def test_agreement_is_not_a_mismatch(self):
        from capture_chain import mismatch_reason
        assert mismatch_reason({"corner_hz": 11.8, "order": 2},
                               {"corner_hz": 11.8, "order": 2}) is None
        assert mismatch_reason(None, None) is None

    def test_unknown_is_compatible_with_anything(self):
        """Hard-failing every pre-2026-09-10 dataset would be worse than the mismatch this
        guards against -- absence of evidence is not evidence of mismatch."""
        from capture_chain import mismatch_reason
        assert mismatch_reason("unknown", {"corner_hz": 11.8, "order": 2}) is None
        assert mismatch_reason({"corner_hz": 11.8, "order": 2}, "unknown") is None
        assert mismatch_reason("unknown", None) is None


class TestCaptureStaticIsConsistent:
    """capture_static must probe the onset through the SAME chain it renders through.

    It measures a saturation onset (find_saturation_point) to decide whether the excitation
    drives the circuit hard enough, then shells out to gen_dataset_from_schx to render. If
    the probe sees the raw node while the render is chained, the excitation is sized against
    a signal that never reaches the trainer -- and on a circuit with sub-audio bias wander
    the probe is measuring the wander, not saturation.
    """

    def test_render_forwards_the_chain_to_the_renderer(self, monkeypatch, tmp_path):
        import capture_static
        seen = {}

        def fake_run(cmd, **kw):
            if "--combine" not in cmd:
                seen["cmd"] = list(cmd)
            import types
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(capture_static.subprocess, "run", fake_run)
        (tmp_path / "config.json").write_text('{"oversample": 8}')
        capture_static.render("x.schx", {"Gain": 0.5, "Tone": 0.2}, "in.wav", "8", 1e-4,
                              tmp_path, capture={"corner_hz": 11.8, "order": 2})
        cmd = seen["cmd"]
        assert "--capture-hp-hz" in cmd and cmd[cmd.index("--capture-hp-hz") + 1] == "11.8"
        assert "--capture-order" in cmd and cmd[cmd.index("--capture-order") + 1] == "2"
        assert "--no-capture-chain" not in cmd

    def test_render_forwards_the_opt_out_explicitly(self, monkeypatch, tmp_path):
        """gen_dataset defaults the chain ON, so 'off' must be passed, not omitted."""
        import capture_static
        seen = {}

        def fake_run(cmd, **kw):
            if "--combine" not in cmd:
                seen["cmd"] = list(cmd)
            import types
            return types.SimpleNamespace(returncode=0)

        monkeypatch.setattr(capture_static.subprocess, "run", fake_run)
        (tmp_path / "config.json").write_text('{"oversample": 8}')
        capture_static.render("x.schx", {"Gain": 0.5, "Tone": 0.2}, "in.wav", "8", 1e-4,
                              tmp_path, capture=None)
        assert "--no-capture-chain" in seen["cmd"]
        assert "--capture-hp-hz" not in seen["cmd"]

    def test_onset_probe_receives_the_chain(self, monkeypatch, tmp_path):
        import capture_static
        seen = {}

        def fake_fsp(backend, params, scratch, **kw):
            seen["capture"] = kw.get("capture")
            return {"onset_99pct_input_v": 0.5, "ceiling_rms": 1.0,
                    "ceiling_at_input_v": 1.0, "curve": []}

        monkeypatch.setattr(capture_static, "find_saturation_point", fake_fsp)
        monkeypatch.setattr(capture_static, "_schx_input_v0dbfs", lambda p: 1.0)
        monkeypatch.setattr(capture_static, "LiveSpiceBackend", lambda *a, **k: object())
        import soundfile as sf
        wav = tmp_path / "in.wav"
        sf.write(str(wav), _tone(440.0, dur=0.2), SR, subtype="FLOAT")
        chain = {"corner_hz": 11.8, "order": 2}
        capture_static.ensure_adequate_excitation("x.schx", {"Gain": 0.5}, str(wav),
                                                  tmp_path, capture=chain)
        assert seen["capture"] == chain, "probe measured a different chain than the render"


def test_describe_ordinals_are_correct():
    """'3nd-order' shipped in the first cut of describe(); the suffix was hardcoded."""
    from capture_chain import describe
    for n, want in ((1, "1st-order"), (2, "2nd-order"), (3, "3rd-order"), (4, "4th-order")):
        assert want in describe({"corner_hz": 18.0, "order": n})


class TestConfigResolution:
    """CLI > config.toml > default, and every tool must reach the same answer.

    Before resolve(), a capture key in a config.toml was honoured by run_pipeline (which
    uses set_defaults) and silently IGNORED by prepare_excitation, check_transient_coverage
    and grid_adequacy, which load the config and cherry-pick individual keys. Half-honoured
    is worse than unsupported: it looks like it works.
    """

    def _args(self, **kw):
        import types
        d = {"no_capture_chain": None, "capture_hp_hz": None, "capture_order": None}
        d.update(kw)
        return types.SimpleNamespace(**d)

    def test_default_when_nothing_given(self):
        from capture_chain import resolve
        assert resolve(self._args()) == {"corner_hz": DEFAULT_CORNER_HZ, "order": DEFAULT_ORDER}

    def test_config_supplies_an_override(self):
        from capture_chain import resolve
        assert resolve(self._args(), {"capture-hp-hz": 15.0, "capture-order": 3}) == \
            {"corner_hz": 15.0, "order": 3}

    def test_cli_beats_config(self):
        from capture_chain import resolve
        assert resolve(self._args(capture_hp_hz=22.0),
                       {"capture-hp-hz": 15.0})["corner_hz"] == 22.0

    def test_either_key_spelling_works(self):
        """run_pipeline.load_config normalises to underscores; grid_adequacy loads raw TOML."""
        from capture_chain import resolve
        assert resolve(self._args(), {"capture_hp_hz": 12.0})["corner_hz"] == 12.0
        assert resolve(self._args(), {"capture-hp-hz": 12.0})["corner_hz"] == 12.0

    def test_config_can_disable_the_chain(self):
        """Needs --no-capture-chain to be store_const/None: with store_true its False
        default is indistinguishable from an explicit off, so config could never win."""
        from capture_chain import resolve
        assert resolve(self._args(), {"no-capture-chain": True}) is None

    def test_cli_disable_beats_a_config_override(self):
        from capture_chain import resolve
        assert resolve(self._args(no_capture_chain=True), {"capture-hp-hz": 15.0}) is None

    def test_absent_flag_is_not_a_disable(self):
        """The regression the store_const change exists to prevent."""
        from capture_chain import resolve
        assert resolve(self._args()) is not None

    def test_values_are_coerced(self):
        """TOML may hand back ints or strings where floats/ints are wanted."""
        from capture_chain import resolve
        r = resolve(self._args(), {"capture-hp-hz": 15, "capture-order": "3"})
        assert r == {"corner_hz": 15.0, "order": 3}
        assert isinstance(r["corner_hz"], float) and isinstance(r["order"], int)


class TestEveryBackendGetsAChain:
    """prepare_excitation._setup must resolve a chain on EVERY backend branch.

    Regression caught during development: initialising `_capture = None` and setting it only
    inside the livespice branch left the ngspice-deck and ltspice-deck paths with None --
    silently DISABLING the chain for them while the renderer still applied it.
    """

    def test_setup_returns_a_capture_for_deck_backends(self, tmp_path, monkeypatch):
        import types, sys
        import prepare_excitation as pe
        mod = types.ModuleType("fake_deck")
        mod.KNOB_NAMES = ["Gain"]
        mod.build_deck = lambda **kw: ""
        mod.__file__ = str(tmp_path / "fake_deck.py")
        (tmp_path / "fake_deck.py").write_text("KNOB_NAMES=['Gain']\n")
        monkeypatch.setitem(sys.modules, "fake_deck", mod)
        monkeypatch.setattr(pe, "NgspiceBackend", lambda *a, **k: object())
        args = types.SimpleNamespace(
            backend="ngspice-deck", pedal_dir=str(tmp_path), module="fake_deck",
            exclude_knob=[], probe_node="OUT", maxstep=3e-6, parallel_sims=8,
            peak_max_v=40.0, lead_silence_s=3.0, range=["Gain=0,1"], fixed_params=None,
            no_capture_chain=None, capture_hp_hz=None, capture_order=None, config=None,
            schx=None, oversample=None, iterations=None)
        out = pe._setup(args)
        assert out[-1] is not None, "deck backend silently lost the capture chain"
        assert out[-1] == {"corner_hz": DEFAULT_CORNER_HZ, "order": DEFAULT_ORDER}
