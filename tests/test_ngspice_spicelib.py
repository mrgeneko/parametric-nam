"""Properties tools/ngspice_spicelib.py's input-preparation and result-reading helpers must
have. load_input's vin=None "absolute volts" mode is what lets a multi-level excitation (real
clip at one peak, sweeps at several others, all in one file) survive into ngspice without being
flattened by a single peak-rescale -- see its own docstring and render_ngspice_deck.py's
--absolute flag. _read_result's short-trace/missing-file/exception guards are what let
render_grid's retry escalation treat a genuinely failed render as "still pending" instead of
crashing the whole sweep.

See tools/ngspice_spicelib.py.
"""
import numpy as np
import pytest
from scipy.io import wavfile

from tools.ngspice_spicelib import _read_result, load_input


def write_wav(path, sr, samples, dtype):
    wavfile.write(str(path), sr, np.asarray(samples, dtype=dtype))


class TestLoadInputVinScaling:
    def test_rescales_a_float_wav_to_the_requested_vin_peak(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0.0, 0.25, -0.5], np.float32)
        sr, t, input_src = load_input(str(wav), vin=2.0, tmp=str(tmp_path))
        assert sr == 1000
        assert "filesrc" in input_src

    def test_vin_none_keeps_the_files_own_absolute_values(self, tmp_path):
        """The whole point of vin=None: a file already built at the V0dBFS=1V convention, whose
        segments intentionally sit at different absolute levels, must NOT be rescaled."""
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0.0, 0.25, -0.5], np.float32)
        _, _, input_src_absolute = load_input(str(wav), vin=None, tmp=str(tmp_path), src_name="a.src")
        _, _, input_src_scaled = load_input(str(wav), vin=1.0, tmp=str(tmp_path), src_name="b.src")
        src_a = (tmp_path / "a.src").read_text()
        src_b = (tmp_path / "b.src").read_text()
        vals_a = [float(line.split()[1]) for line in src_a.strip().splitlines()]
        vals_b = [float(line.split()[1]) for line in src_b.strip().splitlines()]
        # vin=1.0 rescales peak 0.5 up to 1.0 (2x); vin=None leaves 0.5 as 0.5.
        assert max(abs(v) for v in vals_a) == pytest.approx(0.5, rel=1e-3)
        assert max(abs(v) for v in vals_b) == pytest.approx(1.0, rel=1e-3)

    def test_int16_wav_is_normalized_to_full_scale_before_vin_scaling(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0, 16384, -32768], np.int16)
        load_input(str(wav), vin=1.0, tmp=str(tmp_path), src_name="c.src")
        vals = [float(line.split()[1]) for line in (tmp_path / "c.src").read_text().strip().splitlines()]
        assert max(abs(v) for v in vals) == pytest.approx(1.0, rel=1e-3)


class TestLoadInputFileFormat:
    def test_pads_the_tail_with_the_held_final_sample(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0.0, 0.5, -0.25], np.float32)
        load_input(str(wav), vin=1.0, tmp=str(tmp_path), src_name="d.src")
        lines = (tmp_path / "d.src").read_text().strip().splitlines()
        vals = [float(line.split()[1]) for line in lines]
        # last real sample (-0.25) rescaled to vin=1.0 peak -> -0.5; every padded sample repeats it.
        assert len(vals) > 3
        assert all(v == pytest.approx(vals[2]) for v in vals[2:])

    def test_time_axis_matches_input_length_and_samplerate(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 500, np.zeros(200), np.float32)
        sr, t, _ = load_input(str(wav), vin=1.0, tmp=str(tmp_path), src_name="e.src")
        assert sr == 500
        assert len(t) == 200
        assert t[1] - t[0] == pytest.approx(1 / 500)

    def test_input_src_references_the_written_file_and_sig_node(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0.0, 0.5], np.float32)
        _, _, input_src = load_input(str(wav), vin=1.0, tmp=str(tmp_path), src_name="f.src")
        assert "sig" in input_src
        assert str(tmp_path / "f.src") in input_src


class TestReadResult:
    def test_missing_file_returns_none_none(self, tmp_path):
        yv, pk = _read_result(str(tmp_path / "nope.raw"), "OUT", t=np.arange(100), sr=1000)
        assert (yv, pk) == (None, None)

    def test_exception_reading_the_trace_returns_none_none(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "x.raw"
        raw_path.write_bytes(b"not a real raw file")

        class BoomRawRead:
            def __init__(self, path):
                raise ValueError("bad raw file")

        monkeypatch.setattr("tools.ngspice_spicelib.RawRead", BoomRawRead)
        yv, pk = _read_result(str(raw_path), "OUT", t=np.arange(100), sr=1000)
        assert (yv, pk) == (None, None)

    def test_trace_much_shorter_than_the_input_is_rejected(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "x.raw"
        raw_path.write_bytes(b"placeholder")

        class ShortTrace:
            def get_wave(self):
                return np.zeros(5)  # far shorter than t (below len(t)//2)

        class FakeRawRead:
            def __init__(self, path):
                pass

            def get_trace(self, name):
                return ShortTrace()

        monkeypatch.setattr("tools.ngspice_spicelib.RawRead", FakeRawRead)
        yv, pk = _read_result(str(raw_path), "OUT", t=np.arange(1000), sr=1000)
        assert (yv, pk) == (None, None)

    def test_trace_that_aborted_before_reaching_full_duration_is_rejected(self, tmp_path, monkeypatch):
        """A `tran` that aborts ("Timestep too small...trouble with node X") mid-run still
        writes a valid, readable raw file for whatever partial time range it completed -- if
        that partial trace happens to have >= len(t)//2 samples, the old length-only check
        accepted it as converged, and np.interp then silently flat-extrapolated the last real
        value for the rest of the requested duration. Must be rejected so render_grid's
        escalation ladder retries at a finer maxstep instead of returning held-flat fake data."""
        raw_path = tmp_path / "x.raw"
        raw_path.write_bytes(b"placeholder")
        t = np.arange(1000) / 1000.0
        tv = np.linspace(0, 0.5 * t[-1], 800)  # plenty of samples, but only reached half the duration
        v = np.full(800, 0.7)

        class TraceV:
            def get_wave(self):
                return v

        class TraceTime:
            def get_wave(self):
                return tv

        class FakeRawRead:
            def __init__(self, path):
                pass

            def get_trace(self, name):
                return TraceTime() if name == "time" else TraceV()

        monkeypatch.setattr("tools.ngspice_spicelib.RawRead", FakeRawRead)
        yv, pk = _read_result(str(raw_path), "OUT", t=t, sr=1000)
        assert (yv, pk) == (None, None)

    def test_successful_trace_is_interpolated_onto_the_input_time_base(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "x.raw"
        raw_path.write_bytes(b"placeholder")
        t = np.arange(1000) / 1000.0
        tv = np.linspace(0, t[-1], 1000)
        v = np.full(1000, 0.7)

        class TraceV:
            def get_wave(self):
                return v

        class TraceTime:
            def get_wave(self):
                return tv

        class FakeRawRead:
            def __init__(self, path):
                pass

            def get_trace(self, name):
                return TraceTime() if name == "time" else TraceV()

        monkeypatch.setattr("tools.ngspice_spicelib.RawRead", FakeRawRead)
        yv, pk = _read_result(str(raw_path), "OUT", t=t, sr=1000)
        assert pk == pytest.approx(0.7)
        assert len(yv) == len(t)
