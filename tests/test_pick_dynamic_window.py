"""Properties tools/pick_dynamic_window.py's provenance and fade helpers must have. This tool
replaced a hand-rolled, undocumented "slide a window and pick the loudest" one-liner used at
least three times in this codebase's history -- the provenance recipe it writes is what makes
a later regen of the SAME window reproducible, and the fade is what keeps the cut from clicking
since it's cut from mid-file, not from silence.

See tools/pick_dynamic_window.py.
"""
import numpy as np
import pytest

from tools.pick_dynamic_window import SR, _audio_provenance, _fade, _tool_git_rev


class TestAudioProvenance:
    def test_reports_frames_and_duration_from_the_array(self):
        x = np.zeros(SR * 2, dtype=np.float32)  # 2 seconds
        prov = _audio_provenance("clip.wav", x=x)
        assert prov["frames"] == SR * 2
        assert prov["duration_s"] == pytest.approx(2.0)
        assert prov["samplerate"] == SR

    def test_sha1_is_deterministic_for_identical_audio(self):
        x = np.linspace(-1, 1, 1000, dtype=np.float32)
        p1 = _audio_provenance("a.wav", x=x.copy())
        p2 = _audio_provenance("b.wav", x=x.copy())
        assert p1["audio_sha1"] == p2["audio_sha1"]

    def test_sha1_differs_for_different_audio(self):
        x1 = np.zeros(1000, dtype=np.float32)
        x2 = np.ones(1000, dtype=np.float32)
        assert _audio_provenance("a.wav", x=x1)["audio_sha1"] != _audio_provenance("a.wav", x=x2)["audio_sha1"]

    def test_stereo_input_is_averaged_to_mono_before_hashing(self):
        left = np.full(1000, 0.2, dtype=np.float32)
        right = np.full(1000, 0.6, dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        mono_equiv = np.full(1000, 0.4, dtype=np.float32)
        assert _audio_provenance("x.wav", x=stereo)["audio_sha1"] == \
            _audio_provenance("x.wav", x=mono_equiv)["audio_sha1"]

    def test_name_and_path_come_from_the_given_path(self):
        prov = _audio_provenance("/some/dir/clip.wav", x=np.zeros(10, dtype=np.float32))
        assert prov["name"] == "clip.wav"
        assert prov["path"] == "/some/dir/clip.wav"


class TestFade:
    def test_does_not_mutate_the_input_array(self):
        y = np.ones(SR, dtype=np.float32)
        original = y.copy()
        _fade(y, ms_in=10, ms_out=10)
        assert np.array_equal(y, original)

    def test_zero_fade_lengths_are_a_no_op(self):
        y = np.random.RandomState(0).randn(1000).astype(np.float32)
        out = _fade(y, ms_in=0, ms_out=0)
        assert np.array_equal(out, y)

    def test_first_sample_of_a_fade_in_is_near_silent(self):
        y = np.ones(SR, dtype=np.float32)
        out = _fade(y, ms_in=10, ms_out=0)
        assert abs(out[0]) < 1e-6

    def test_last_sample_of_a_fade_out_is_near_silent(self):
        y = np.ones(SR, dtype=np.float32)
        out = _fade(y, ms_in=0, ms_out=10)
        assert abs(out[-1]) < 1e-6

    def test_content_outside_the_fade_regions_is_untouched(self):
        y = np.full(SR, 0.7, dtype=np.float32)
        out = _fade(y, ms_in=5, ms_out=5)
        assert out[SR // 2] == pytest.approx(0.7)

    def test_fade_only_scales_never_amplifies(self):
        y = np.full(SR, 0.7, dtype=np.float32)
        out = _fade(y, ms_in=10, ms_out=10)
        assert np.all(np.abs(out) <= 0.7 + 1e-6)


class TestToolGitRev:
    def test_returns_stripped_stdout_on_success(self, monkeypatch):
        class R:
            returncode = 0
            stdout = "abc1234\n"
        monkeypatch.setattr("tools.pick_dynamic_window.subprocess.run", lambda *a, **kw: R())
        assert _tool_git_rev() == "abc1234"

    def test_returns_none_on_nonzero_exit(self, monkeypatch):
        class R:
            returncode = 128
            stdout = ""
        monkeypatch.setattr("tools.pick_dynamic_window.subprocess.run", lambda *a, **kw: R())
        assert _tool_git_rev() is None

    def test_returns_none_if_subprocess_raises(self, monkeypatch):
        def boom(*a, **kw):
            raise FileNotFoundError("git not installed")
        monkeypatch.setattr("tools.pick_dynamic_window.subprocess.run", boom)
        assert _tool_git_rev() is None
