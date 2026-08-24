"""Properties gen_dataset_from_schx.write_probe_clip()'s lead-in override must have. Found
directly while retraining the Fulltone OCD: write_probe_clip's hardcoded 1.0s default lead-in
is far short of that circuit's ~5s C10/RVOL2 RC settling time, so every probe window built by
measure_truncation.py / tools/grid_adequacy.py / tools/measure_ngspice_timestep.py (all of which
call this function, directly or via probe_clips()) started from an under-settled state -- and
which quasi-stable-but-WRONG DC point the render then landed in depended on tiny solver-path
differences (maxstep, grid density), looking like circuit instability or runaway non-convergence
until traced back to this. `lead_s` is opt-in per-call so every existing caller/device at the
1.0s default is unaffected.

See gen_dataset_from_schx.write_probe_clip, measure_truncation.probe_clips.
"""
import numpy as np
import pytest
import soundfile as sf

from gen_dataset_from_schx import PROBE_LEAD_S, write_probe_clip
from measure_truncation import probe_clips


class TestWriteProbeClipLeadIn:
    def test_default_lead_matches_probe_lead_s_constant(self, tmp_path):
        sig = np.ones(1000, dtype=np.float32)
        lead_n = write_probe_clip(sig, sr=1000, path=tmp_path / "c.wav")
        assert lead_n == int(PROBE_LEAD_S * 1000)

    def test_lead_s_override_changes_the_returned_lead_length(self, tmp_path):
        sig = np.ones(1000, dtype=np.float32)
        lead_n = write_probe_clip(sig, sr=1000, path=tmp_path / "c.wav", lead_s=5.0)
        assert lead_n == 5000

    def test_lead_s_override_actually_writes_that_much_silence(self, tmp_path):
        sig = np.ones(2000, dtype=np.float32)
        path = tmp_path / "c.wav"
        lead_n = write_probe_clip(sig, sr=1000, path=path, lead_s=3.0)
        out, sr = sf.read(str(path))
        assert lead_n == 3000
        assert np.allclose(out[:lead_n], 0.0)
        # real content picks up right after the lead-in (ramped, so not exactly 1.0 at the edge)
        assert len(out) == lead_n + 2000

    def test_lead_s_none_falls_back_to_the_default(self, tmp_path):
        sig = np.ones(1000, dtype=np.float32)
        default_lead = write_probe_clip(sig, sr=1000, path=tmp_path / "a.wav", lead_s=None)
        explicit_lead = write_probe_clip(sig, sr=1000, path=tmp_path / "b.wav")
        assert default_lead == explicit_lead == int(PROBE_LEAD_S * 1000)


class TestProbeClipsForwardsLeadS:
    def test_lead_s_override_is_forwarded_to_write_probe_clip(self, tmp_path):
        sr = 1000
        sig = np.ones(sr * 20, dtype=np.float32)
        inp = tmp_path / "input.wav"
        sf.write(str(inp), sig, sr)

        clips, lead_n, sr_out = probe_clips(inp, probe_s=8.0, n_windows=2, td=tmp_path, lead_s=5.0)
        assert lead_n == 5 * sr
        assert sr_out == sr
        for c in clips:
            out, _ = sf.read(str(c))
            assert np.allclose(out[:lead_n], 0.0)

    def test_default_lead_s_none_uses_write_probe_clips_own_default(self, tmp_path):
        sr = 1000
        sig = np.ones(sr * 20, dtype=np.float32)
        inp = tmp_path / "input.wav"
        sf.write(str(inp), sig, sr)

        _, lead_n, _ = probe_clips(inp, probe_s=8.0, n_windows=2, td=tmp_path)
        assert lead_n == int(PROBE_LEAD_S * sr)
