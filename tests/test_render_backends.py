"""Properties tools/render_backends.py's two backend adapters must have. preflight.py and
find_saturation_point.py are backend-agnostic ONLY because both backends genuinely honor the
same contract (prepare_input/render_many -- see the module docstring); a backend that scales
input wrong, or a tag that goes missing between jobs and results, breaks every checker built on
top of it without looking like a backend bug. Heavy externals (the livespice_cli subprocess,
ngspice_spicelib's render_grid/load_input) are stubbed so the adapter's own logic -- level
scaling, tag bookkeeping, failure-to-None mapping, and the int16-peak-unnormalization math -- is
tested independently of any real render.

See tools/render_backends.py.
"""
import numpy as np
import pytest
import soundfile as sf

from tools.render_backends import LiveSpiceBackend, NgspiceBackend


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

        monkeypatch.setattr("tools.render_backends.subprocess.run", fake_run)
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

        monkeypatch.setattr("tools.render_backends.subprocess.run", fake_run)
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

        monkeypatch.setattr("tools.render_backends.render_grid",
                             lambda *a, **kw: {outfile: real_peak_v})
        monkeypatch.setattr("tools.render_backends.wavfile.read",
                             lambda path: (1000, int16_peak))

        backend = NgspiceBackend(build_deck=lambda **kw: None, probe_node="OUT")
        out = backend.render_many([{"params": {}, "tag": "a"}], input_handle=(1000, np.zeros(1), ("", "")),
                                   scratch=str(tmp_path))
        assert out["a"][1] == pytest.approx(real_peak_v, rel=1e-3)
        assert out["a"][2] == pytest.approx(-real_peak_v, rel=1e-3)
        assert out["a"][0] == pytest.approx(0.0, abs=1e-6)

    def test_nonconverged_render_maps_to_none_without_reading_a_missing_file(self, tmp_path, monkeypatch):
        outfile = str(tmp_path / "pf_a.wav")
        monkeypatch.setattr("tools.render_backends.render_grid", lambda *a, **kw: {outfile: None})

        def boom(path):
            raise AssertionError("wavfile.read must not be called for a non-converged render")
        monkeypatch.setattr("tools.render_backends.wavfile.read", boom)

        backend = NgspiceBackend(build_deck=lambda **kw: None, probe_node="OUT")
        out = backend.render_many([{"params": {}, "tag": "a"}], input_handle=(1000, np.zeros(1), ("", "")),
                                   scratch=str(tmp_path))
        assert out == {"a": None}

    def test_multiple_jobs_each_get_their_own_outfile_and_tag(self, monkeypatch):
        peaks = {"/s/pf_a.wav": 1.0, "/s/pf_b.wav": 2.0}
        monkeypatch.setattr("tools.render_backends.render_grid", lambda *a, **kw: peaks)
        monkeypatch.setattr("tools.render_backends.wavfile.read",
                             lambda path: (1000, np.array([int(0.9 * 32767)], dtype=np.int16)))

        backend = NgspiceBackend(build_deck=lambda **kw: None, probe_node="OUT")
        jobs = [{"params": {"Gain": 0.1}, "tag": "a"}, {"params": {"Gain": 0.9}, "tag": "b"}]
        out = backend.render_many(jobs, input_handle=(1000, np.zeros(1), ("", "")), scratch="/s")
        assert out["a"][0] == pytest.approx(1.0, rel=1e-3)
        assert out["b"][0] == pytest.approx(2.0, rel=1e-3)
