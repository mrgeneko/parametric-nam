"""Properties tools/ltspice_spicelib.py's input-preparation and result-reading helpers must
have -- the LTspice counterpart of test_ngspice_spicelib.py, for the contract differences that
LTspice's PCM-only, +/-1V-bounded wavefile=/.wave I/O force (see that module's own docstring):
load_input's in_scale must keep the written file's peak under 1.0 regardless of how many volts
the excitation represents, and render_grid's truncation check must reject a run that aborted
partway through rather than accept whatever partial data it produced -- the exact bug found and
fixed in ngspice_spicelib.py this session, baked in here from the start instead of discovered
the same way twice.

See tools/ltspice_spicelib.py.
"""
import numpy as np
import pytest
import soundfile as sf

from tools.ltspice_spicelib import (DEFAULT_TIMEOUT_S_PER_AUDIO_S, MIN_TIMEOUT_S, load_input,
                                     _read_result, render_grid, render_one)


def write_wav(path, sr, samples, subtype="FLOAT"):
    sf.write(str(path), np.asarray(samples, dtype=np.float64), sr, subtype=subtype)


class TestLoadInput:
    def test_returns_the_files_own_samplerate_and_duration(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, np.zeros(500))
        sr, dur_s, wav_path, in_scale = load_input(str(wav), vin=None, tmp=str(tmp_path))
        assert sr == 1000
        assert dur_s == pytest.approx((500 + int(0.01 * 1000)) / 1000)  # + PAD_S tail

    def test_vin_none_keeps_the_files_own_absolute_values(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0.0, 0.5, -0.25])
        _, _, wav_path, in_scale = load_input(str(wav), vin=None, tmp=str(tmp_path), src_name="a.wav")
        y, _ = sf.read(wav_path, dtype="float64")
        assert np.abs(y).max() * in_scale == pytest.approx(0.5, rel=1e-3)

    def test_vin_rescales_the_files_peak(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0.0, 0.5, -0.25])
        _, _, wav_path, in_scale = load_input(str(wav), vin=2.0, tmp=str(tmp_path), src_name="b.wav")
        y, _ = sf.read(wav_path, dtype="float64")
        assert np.abs(y).max() * in_scale == pytest.approx(2.0, rel=1e-3)

    def test_written_wav_never_exceeds_pcm_full_scale_for_a_loud_excitation(self, tmp_path):
        # This is THE reason in_scale exists: a real excitation in this pipeline routinely
        # peaks well past 1V (OCD's own excitation peaks at 11.75V). Writing that directly as
        # PCM would silently clip at +/-1.0 with no error -- confirmed directly this session.
        wav = tmp_path / "loud.wav"
        write_wav(wav, 1000, [0.0, 11.75, -11.75])
        _, _, wav_path, in_scale = load_input(str(wav), vin=None, tmp=str(tmp_path), src_name="c.wav")
        y, _ = sf.read(wav_path, dtype="float64")
        assert np.abs(y).max() < 1.0
        assert in_scale >= 11.75

    def test_quiet_signal_does_not_get_scaled_up(self, tmp_path):
        # in_scale must never go BELOW 1.0 -- PCM_24's own resolution is high enough that a
        # quiet signal loses nothing by staying at its own scale; scaling up would just be an
        # unnecessary difference from a straightforward "write it as-is" reading of the file.
        wav = tmp_path / "quiet.wav"
        write_wav(wav, 1000, [0.0, 0.001, -0.001])
        _, _, _, in_scale = load_input(str(wav), vin=None, tmp=str(tmp_path), src_name="d.wav")
        assert in_scale == pytest.approx(1.0, rel=1e-3)

    def test_pads_the_tail_with_the_held_final_sample(self, tmp_path):
        wav = tmp_path / "in.wav"
        write_wav(wav, 1000, [0.0, 0.5, -0.25])
        _, _, wav_path, in_scale = load_input(str(wav), vin=None, tmp=str(tmp_path), src_name="e.wav")
        y, _ = sf.read(wav_path, dtype="float64")
        assert len(y) > 3
        assert all(v == pytest.approx(y[2]) for v in y[2:])


class TestReadResult:
    def test_missing_file_returns_none_none(self, tmp_path):
        yv, pk = _read_result(str(tmp_path / "nope.wav"), dur_target_n=100, out_scale=1.0)
        assert (yv, pk) == (None, None)

    def test_exception_reading_the_file_returns_none_none(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a real wav file")
        yv, pk = _read_result(str(bad), dur_target_n=100, out_scale=1.0)
        assert (yv, pk) == (None, None)

    def test_a_run_that_aborted_before_reaching_full_duration_is_rejected(self, tmp_path):
        """An LTspice run that crashes/aborts mid-transient ("Time step too small...") still
        leaves a valid, readable partial .wave file for whatever it completed. A length-only
        check that doesn't compare against the REQUESTED duration would accept this as
        converged -- the exact bug found and fixed in ngspice_spicelib.py's _read_result this
        session (a truncated ngspice reference silently corrupted an entire cross-engine
        comparison). Must be rejected so render_grid's escalation ladder retries instead."""
        wav = tmp_path / "truncated.wav"
        write_wav(wav, 1000, np.full(400, 0.5))  # only 400 of a requested 1000 samples
        yv, pk = _read_result(str(wav), dur_target_n=1000, out_scale=1.0)
        assert (yv, pk) == (None, None)

    def test_a_full_duration_run_is_unscaled_and_returned(self, tmp_path):
        wav = tmp_path / "ok.wav"
        write_wav(wav, 1000, [0.0, 0.05, -0.05])  # written at out_scale=0.1 -> real peak 0.5V
        yv, pk = _read_result(str(wav), dur_target_n=3, out_scale=0.1)
        assert pk == pytest.approx(0.5, rel=1e-3)
        assert len(yv) == 3


class FakeLtspiceRun:
    """Stands in for tools.ltspice_spicelib._run_ltspice: instead of actually invoking the
    LTspice binary, writes a .wave file at the raw_wav path build_deck's own call embedded --
    mirrors what a real render would leave on disk for _read_result to then pick up."""

    def __init__(self, value_fn, always_short_by=0):
        self.value_fn = value_fn
        self.always_short_by = always_short_by
        self.calls = []

    def __call__(self, net_path, timeout):
        self.calls.append((net_path, timeout))
        return True


def make_build_deck(sr, value_fn, n_samples=100, short_by=0):
    """A fake build_deck that, instead of returning real LTspice netlist text, writes the
    .wave file directly at out_wav and returns a harmless comment -- render_grid only cares
    that build_deck returns a string to write to net_path and that out_wav ends up populated
    after _run_ltspice "runs" (faked to a no-op subprocess by monkeypatching _run_ltspice)."""
    def build_deck(wav_path, dur_s, maxstep, out_wav, knobs=None, tap="spk", out_scale=1.0,
                   in_scale=1.0, method=None):
        n = max(0, n_samples - short_by)
        val = value_fn(knobs, maxstep)
        sf.write(out_wav, np.full(n, val * out_scale), sr, subtype="FLOAT")
        return "* fake deck\n.end\n"
    return build_deck


class TestRenderGridTimeoutScaling:
    """A flat timeout can't be right for both a grid_adequacy 8s probe and a 60s+ full
    excitation capture -- found directly this session: a value tuned for short probes
    silently killed every job partway through a real capture render, which looked exactly
    like a genuine convergence failure (every job escalated through the whole maxstep ladder
    and failed again) rather than the timeout-too-short bug it actually was."""

    def _seen_timeout(self, tmp_path, monkeypatch, dur_s, timeout=None):
        seen = []

        def fake_run(net_path, timeout):
            seen.append(timeout)
            return True

        monkeypatch.setattr("tools.ltspice_spicelib._run_ltspice", fake_run)
        build_deck = make_build_deck(sr=1000, value_fn=lambda k, m: 0.5,
                                     n_samples=int(dur_s * 1000))
        outfile = str(tmp_path / "out.wav")
        kwargs = {} if timeout is None else {"timeout": timeout}
        render_grid(build_deck, [({}, outfile)], tap="spk", sr=1000, dur_s=dur_s,
                   wav_path="in.wav", in_scale=1.0, tmp=str(tmp_path), out_scale=1.0, **kwargs)
        return seen[0]

    def test_default_timeout_scales_with_duration_not_a_flat_constant(self, tmp_path, monkeypatch):
        short = self._seen_timeout(tmp_path, monkeypatch, dur_s=1.0)
        long = self._seen_timeout(tmp_path, monkeypatch, dur_s=60.0)
        assert long > short
        assert long == pytest.approx(60.0 * DEFAULT_TIMEOUT_S_PER_AUDIO_S)

    def test_default_timeout_has_a_floor_for_very_short_clips(self, tmp_path, monkeypatch):
        tiny = self._seen_timeout(tmp_path, monkeypatch, dur_s=0.01)
        assert tiny == pytest.approx(MIN_TIMEOUT_S)

    def test_explicit_timeout_overrides_the_duration_based_default(self, tmp_path, monkeypatch):
        seen = self._seen_timeout(tmp_path, monkeypatch, dur_s=60.0, timeout=42.0)
        assert seen == pytest.approx(42.0)


class TestRenderGrid:
    def test_converged_render_is_read_back_and_unscaled(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.ltspice_spicelib._run_ltspice", lambda net_path, timeout: True)
        build_deck = make_build_deck(sr=1000, value_fn=lambda knobs, maxstep: 0.5, n_samples=100)
        outfile = str(tmp_path / "out.wav")
        results = render_grid(build_deck, [({"Gain": 0.5}, outfile)], tap="spk", sr=1000,
                              dur_s=0.1, wav_path="in.wav", in_scale=1.0, tmp=str(tmp_path),
                              out_scale=0.1)
        assert results[outfile] == pytest.approx(0.5, rel=1e-3)

    def test_every_job_gets_its_own_outfile(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.ltspice_spicelib._run_ltspice", lambda net_path, timeout: True)
        build_deck = make_build_deck(sr=1000, value_fn=lambda knobs, maxstep: knobs["Gain"], n_samples=100)
        jobs = [({"Gain": 0.2}, str(tmp_path / "a.wav")), ({"Gain": 0.8}, str(tmp_path / "b.wav"))]
        results = render_grid(build_deck, jobs, tap="spk", sr=1000, dur_s=0.1, wav_path="in.wav",
                              in_scale=1.0, tmp=str(tmp_path), out_scale=1.0)
        assert results[jobs[0][1]] == pytest.approx(0.2, rel=1e-3)
        assert results[jobs[1][1]] == pytest.approx(0.8, rel=1e-3)

    def test_a_run_that_never_reaches_full_duration_escalates_to_a_finer_rung(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.ltspice_spicelib._run_ltspice", lambda net_path, timeout: True)
        # first rung's build_deck always writes a too-short file; the SECOND rung's maxstep
        # (passed to build_deck) writes the full length -- simulates a finer timestep
        # succeeding where the coarse one aborted.
        def build_deck(wav_path, dur_s, maxstep, out_wav, knobs=None, tap="spk", out_scale=1.0,
                       in_scale=1.0, method=None):
            n = 100 if maxstep <= 1e-6 else 40  # coarse (rung 0) truncates; rung 1 (maxstep/3) doesn't
            sf.write(out_wav, np.full(n, 0.5), 1000, subtype="FLOAT")
            return "* fake\n.end\n"

        outfile = str(tmp_path / "out.wav")
        results = render_grid(build_deck, [({}, outfile)], tap="spk", sr=1000, dur_s=0.1,
                              wav_path="in.wav", in_scale=1.0, tmp=str(tmp_path),
                              maxstep=3e-6, out_scale=1.0)
        assert results[outfile] == pytest.approx(0.5, rel=1e-3)

    def test_all_rungs_failing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.ltspice_spicelib._run_ltspice", lambda net_path, timeout: True)
        build_deck = make_build_deck(sr=1000, value_fn=lambda k, m: 0.5, n_samples=10)  # always too short
        outfile = str(tmp_path / "out.wav")
        results = render_grid(build_deck, [({}, outfile)], tap="spk", sr=1000, dur_s=1.0,
                              wav_path="in.wav", in_scale=1.0, tmp=str(tmp_path), out_scale=1.0)
        assert results[outfile] is None

    def test_rungs_override_gives_a_genuine_single_shot_no_escalation(self, tmp_path, monkeypatch):
        seen_maxsteps = []

        def build_deck(wav_path, dur_s, maxstep, out_wav, knobs=None, tap="spk", out_scale=1.0,
                       in_scale=1.0, method=None):
            seen_maxsteps.append(maxstep)
            sf.write(out_wav, np.full(10, 0.5), 1000, subtype="FLOAT")  # always too short
            return "* fake\n.end\n"

        monkeypatch.setattr("tools.ltspice_spicelib._run_ltspice", lambda net_path, timeout: True)
        outfile = str(tmp_path / "out.wav")
        results = render_grid(build_deck, [({}, outfile)], tap="spk", sr=1000, dur_s=1.0,
                              wav_path="in.wav", in_scale=1.0, tmp=str(tmp_path),
                              maxstep=5e-7, rungs=(5e-7,), out_scale=1.0)
        assert results[outfile] is None
        assert seen_maxsteps == [5e-7]  # never escalated to maxstep/3 or /10


class TestRenderOne:
    def test_returns_the_single_jobs_peak(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.ltspice_spicelib._run_ltspice", lambda net_path, timeout: True)
        build_deck = make_build_deck(sr=1000, value_fn=lambda k, m: 0.7, n_samples=50)
        outfile = str(tmp_path / "out.wav")
        pk = render_one(build_deck, {"Gain": 0.5}, outfile, "spk", 1000, 0.05, "in.wav", 1.0,
                        str(tmp_path), out_scale=1.0)
        assert pk == pytest.approx(0.7, rel=1e-3)
