"""Properties render_backends.py's two backend adapters must have. preflight.py and
find_saturation_point.py are backend-agnostic ONLY because both backends genuinely honor the
same contract (prepare_input/render_many -- see the module docstring); a backend that scales
input wrong, or a tag that goes missing between jobs and results, breaks every checker built on
top of it without looking like a backend bug. Heavy externals (the livespice_cli subprocess,
ngspice_spicelib's render_grid/load_input) are stubbed so the adapter's own logic -- level
scaling, tag bookkeeping, failure-to-None mapping, and the int16-peak-unnormalization math -- is
tested independently of any real render.

See render_backends.py.
"""
import numpy as np
import pytest
import soundfile as sf

from render_backends import LiveSpiceBackend, NgspiceBackend, LtspiceBackend


class TestLiveSpiceBackendPrepareInput:
    def test_scales_raw_so_its_peak_hits_level_v(self, tmp_path):
        backend = LiveSpiceBackend(schx="unused.schx")
        raw = np.array([0.0, 0.5, -0.25, 0.1], dtype=np.float32)
        path = backend.prepare_input(raw, sr=1000, level_v=2.0, scratch=str(tmp_path), tag="t1")
        y, sr = sf.read(path, dtype="float32")
        assert sr == 1000
        assert np.abs(y).max() == pytest.approx(2.0, rel=1e-4)

    def test_written_as_float_so_drive_above_1v_survives(self, tmp_path):
        # A wav written as int PCM would clip any sample above 1.0 -- prepare_input must use a
        # FLOAT subtype so a >1V drive level round-trips intact.
        backend = LiveSpiceBackend(schx="unused.schx")
        raw = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        path = backend.prepare_input(raw, sr=1000, level_v=5.0, scratch=str(tmp_path), tag="t1")
        y, _ = sf.read(path, dtype="float32")
        assert np.abs(y).max() == pytest.approx(5.0, rel=1e-4)

    def test_zero_signal_does_not_divide_by_zero(self, tmp_path):
        backend = LiveSpiceBackend(schx="unused.schx")
        raw = np.zeros(10, dtype=np.float32)
        path = backend.prepare_input(raw, sr=1000, level_v=1.0, scratch=str(tmp_path), tag="t1")
        y, _ = sf.read(path, dtype="float32")
        assert np.all(np.isfinite(y))


class TestLiveSpiceBackendRenderMany:
    def test_empty_jobs_returns_empty_dict(self):
        backend = LiveSpiceBackend(schx="unused.schx")
        assert backend.render_many([], input_handle="in.wav", scratch="/tmp") == {}

    def test_successful_render_returns_audio_keyed_by_tag(self, tmp_path, monkeypatch):
        def fake_run(cmd, capture_output, text):
            out_path = cmd[cmd.index("--output") + 1]
            sf.write(out_path, np.array([0.1, 0.2, 0.3], dtype=np.float32), 1000, subtype="FLOAT")

            class R:
                stderr = ""
            return R()

        monkeypatch.setattr("render_backends.subprocess.run", fake_run)
        backend = LiveSpiceBackend(schx="unused.schx")
        jobs = [{"params": {"Gain": 0.5}, "tag": "a"}, {"params": {"Gain": 0.9}, "tag": "b"}]
        out = backend.render_many(jobs, input_handle="in.wav", scratch=str(tmp_path))
        assert set(out) == {"a", "b"}
        assert np.allclose(out["a"], [0.1, 0.2, 0.3], atol=1e-4)

    def test_render_that_never_produces_output_maps_to_none_not_an_exception(self, tmp_path, monkeypatch):
        def fake_run(cmd, capture_output, text):
            class R:
                stderr = "ngspice: convergence failed"
            return R()  # never writes the output file

        monkeypatch.setattr("render_backends.subprocess.run", fake_run)
        backend = LiveSpiceBackend(schx="unused.schx")
        out = backend.render_many([{"params": {}, "tag": "a"}], input_handle="in.wav", scratch=str(tmp_path))
        assert out == {"a": None}


class TestNgspiceBackendRenderMany:
    def test_empty_jobs_returns_empty_dict(self):
        backend = NgspiceBackend(build_deck=lambda **kw: None, probe_node="OUT")
        assert backend.render_many([], input_handle=(1000, np.zeros(1), ("", "")), scratch="/tmp") == {}

    def test_converged_render_is_unnormalized_back_to_voltage_scale(self, tmp_path, monkeypatch):
        """render_grid hands back int16 wavs peak-normalized to 0.9*32767 -- render_many must
        undo exactly that scaling using the real peak voltage it also got back, or every
        downstream RMS/peak measurement is wrong by the normalization factor."""
        outfile = str(tmp_path / "pf_a.wav")
        real_peak_v = 3.0
        int16_peak = np.array([0, int(0.9 * 32767), -int(0.9 * 32767)], dtype=np.int16)

        monkeypatch.setattr("render_backends.render_grid",
                             lambda *a, **kw: {outfile: real_peak_v})
        monkeypatch.setattr("render_backends.wavfile.read",
                             lambda path: (1000, int16_peak))

        backend = NgspiceBackend(build_deck=lambda **kw: None, probe_node="OUT")
        out = backend.render_many([{"params": {}, "tag": "a"}], input_handle=(1000, np.zeros(1), ("", "")),
                                   scratch=str(tmp_path))
        assert out["a"][1] == pytest.approx(real_peak_v, rel=1e-3)
        assert out["a"][2] == pytest.approx(-real_peak_v, rel=1e-3)
        assert out["a"][0] == pytest.approx(0.0, abs=1e-6)

    def test_nonconverged_render_maps_to_none_without_reading_a_missing_file(self, tmp_path, monkeypatch):
        outfile = str(tmp_path / "pf_a.wav")
        monkeypatch.setattr("render_backends.render_grid", lambda *a, **kw: {outfile: None})

        def boom(path):
            raise AssertionError("wavfile.read must not be called for a non-converged render")
        monkeypatch.setattr("render_backends.wavfile.read", boom)

        backend = NgspiceBackend(build_deck=lambda **kw: None, probe_node="OUT")
        out = backend.render_many([{"params": {}, "tag": "a"}], input_handle=(1000, np.zeros(1), ("", "")),
                                   scratch=str(tmp_path))
        assert out == {"a": None}

    def test_multiple_jobs_each_get_their_own_outfile_and_tag(self, monkeypatch):
        peaks = {"/s/pf_a.wav": 1.0, "/s/pf_b.wav": 2.0}
        monkeypatch.setattr("render_backends.render_grid", lambda *a, **kw: peaks)
        monkeypatch.setattr("render_backends.wavfile.read",
                             lambda path: (1000, np.array([int(0.9 * 32767)], dtype=np.int16)))

        backend = NgspiceBackend(build_deck=lambda **kw: None, probe_node="OUT")
        jobs = [{"params": {"Gain": 0.1}, "tag": "a"}, {"params": {"Gain": 0.9}, "tag": "b"}]
        out = backend.render_many(jobs, input_handle=(1000, np.zeros(1), ("", "")), scratch="/s")
        assert out["a"][0] == pytest.approx(1.0, rel=1e-3)
        assert out["b"][0] == pytest.approx(2.0, rel=1e-3)


class TestLtspiceBackendPrepareInput:
    def test_forwards_raw_as_a_float_wav_to_load_input(self, tmp_path, monkeypatch):
        # LTspice's own wavefile= needs PCM (see ltspice_spicelib.load_input's docstring), but
        # prepare_input's OWN write here is an intermediate handed straight to load_input,
        # which does the PCM-safe rescale itself -- FLOAT here just avoids a second, redundant
        # lossy round-trip before that happens.
        seen = {}

        def fake_load_input(wav_path, vin, tmp, src_name):
            y, sr = sf.read(wav_path, dtype="float32")
            seen.update(wav_path=wav_path, vin=vin, tmp=tmp, src_name=src_name, peak=float(np.abs(y).max()))
            return (sr, 1.0, wav_path, 1.0)

        monkeypatch.setattr("render_backends.ltspice_spicelib.load_input", fake_load_input)
        backend = LtspiceBackend(build_deck=lambda **kw: None)
        raw = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        handle = backend.prepare_input(raw, sr=1000, level_v=2.0, scratch=str(tmp_path), tag="t1")
        assert seen["vin"] == 2.0
        assert seen["peak"] == pytest.approx(1.0, rel=1e-4)  # unscaled -- load_input does the level_v scaling
        assert handle == (1000, 1.0, seen["wav_path"], 1.0)


class TestLtspiceBackendRenderMany:
    def test_empty_jobs_returns_empty_dict(self):
        backend = LtspiceBackend(build_deck=lambda **kw: None)
        assert backend.render_many([], input_handle=(1000, 1.0, "in.wav", 1.0), scratch="/tmp") == {}

    def test_converged_render_returns_audio_keyed_by_tag(self, tmp_path, monkeypatch):
        def fake_render_grid(build_deck, jobs, tap, sr, dur_s, wav_path, in_scale, tmp, **kw):
            out = {}
            for knobs, outfile in jobs:
                sf.write(outfile, np.array([0.1, 0.2, 0.3], dtype=np.float32), sr, subtype="FLOAT")
                out[outfile] = 0.3
            return out

        monkeypatch.setattr("render_backends.ltspice_spicelib.render_grid", fake_render_grid)
        backend = LtspiceBackend(build_deck=lambda **kw: None)
        jobs = [{"params": {"Gain": 0.5}, "tag": "a"}, {"params": {"Gain": 0.9}, "tag": "b"}]
        out = backend.render_many(jobs, input_handle=(1000, 1.0, "in.wav", 1.0), scratch=str(tmp_path))
        assert set(out) == {"a", "b"}
        assert np.allclose(out["a"], [0.1, 0.2, 0.3], atol=1e-4)

    def test_nonconverged_render_maps_to_none_without_reading_a_missing_file(self, tmp_path, monkeypatch):
        outfile = str(tmp_path / "pf_a.wav")
        monkeypatch.setattr("render_backends.ltspice_spicelib.render_grid",
                            lambda *a, **kw: {outfile: None})

        def boom(path, dtype=None):
            raise AssertionError("sf.read must not be called for a non-converged render")
        monkeypatch.setattr("render_backends.sf.read", boom)

        backend = LtspiceBackend(build_deck=lambda **kw: None)
        out = backend.render_many([{"params": {}, "tag": "a"}], input_handle=(1000, 1.0, "in.wav", 1.0),
                                  scratch=str(tmp_path))
        assert out == {"a": None}

    def test_input_handle_fields_are_forwarded_to_render_grid(self, monkeypatch):
        seen = {}

        def fake_render_grid(build_deck, jobs, tap, sr, dur_s, wav_path, in_scale, tmp, **kw):
            seen.update(tap=tap, sr=sr, dur_s=dur_s, wav_path=wav_path, in_scale=in_scale, **kw)
            return {outfile: None for _knobs, outfile in jobs}

        monkeypatch.setattr("render_backends.ltspice_spicelib.render_grid", fake_render_grid)
        backend = LtspiceBackend(build_deck=lambda **kw: None, tap="spk", maxstep=1e-7,
                                 parallel_sims=4, out_scale=0.1)
        backend.render_many([{"params": {}, "tag": "a"}],
                            input_handle=(48000, 2.5, "/s/in.wav", 3.0), scratch="/s")
        assert seen["tap"] == "spk"
        assert seen["sr"] == 48000
        assert seen["dur_s"] == 2.5
        assert seen["wav_path"] == "/s/in.wav"
        assert seen["in_scale"] == 3.0
        assert seen["maxstep"] == 1e-7
        assert seen["parallel_sims"] == 4
        assert seen["out_scale"] == 0.1
