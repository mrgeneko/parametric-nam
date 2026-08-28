"""Tests for gen_dataset_from_captures.py's raw-capture (.wav/.aif/.aiff) path -- specifically
the delay/latency calibration and alignment logic, which is the one thing a raw capture needs
that a .nam capture (pure digital inference, phase-aligned by construction) doesn't. See that
module's detect_delay()/_align_wet() docstrings for the full reasoning.

detect_delay() shells out to nam_delay_helper.py under a sibling neural-amp-modeler
venv's own interpreter (see detect_delay's docstring for why: a numba/numpy version conflict
when importing nam.train.core in-process, confirmed hitting it directly). These tests fake
that subprocess boundary -- monkeypatching _find_nam_site_packages (a fake venv layout under
tmp_path) and subprocess.run (canned JSON, matching the helper's actual output contract) --
so they pass whether or not a real neural-amp-modeler install exists in the environment
running the tests, and without spawning a real subprocess.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

import gen_dataset_from_captures as gdc


# --------------------------------------------------------------------------- _align_wet

def test_align_wet_positive_delay_drops_leading_samples_and_pads():
    # wet lags dry by 3 samples: the first 3 are "dead time" before the response starts.
    y = np.array([9, 9, 9, 1, 2, 3, 4], dtype=np.float32)
    out = gdc._align_wet(y, delay=3, target_len=7)
    np.testing.assert_array_equal(out, [1, 2, 3, 4, 0, 0, 0])


def test_align_wet_negative_delay_prepends_silence():
    # wet LEADS dry by 2 samples (unusual, but the sign convention supports it).
    y = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    out = gdc._align_wet(y, delay=-2, target_len=5)
    np.testing.assert_array_equal(out, [0, 0, 1, 2, 3])


def test_align_wet_zero_delay_just_pads_or_truncates():
    y = np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(gdc._align_wet(y, 0, target_len=5), [1, 2, 3, 0, 0])
    np.testing.assert_array_equal(gdc._align_wet(y, 0, target_len=2), [1, 2])


def test_align_wet_truncates_when_shift_leaves_it_longer_than_target():
    y = np.arange(10, dtype=np.float32)
    out = gdc._align_wet(y, delay=1, target_len=5)
    assert len(out) == 5
    np.testing.assert_array_equal(out, [1, 2, 3, 4, 5])


# --------------------------------------------------------------------------- detect_delay

def _fake_sibling_venv(tmp_path, with_python=True):
    """Build a fake sibling-checkout layout matching what _find_nam_site_packages looks for
    and detect_delay derives its interpreter path from, WITHOUT a real neural-amp-modeler
    install -- these tests fake the subprocess boundary itself (see module docstring)."""
    site_packages = tmp_path / "neural-amp-modeler" / "venv" / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    if with_python:
        bin_dir = site_packages.parent.parent.parent / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python3").touch()
    return site_packages


def test_detect_delay_when_no_sibling_checkout(monkeypatch):
    monkeypatch.setattr(gdc, "_find_nam_site_packages", lambda: None)
    delay, source = gdc.detect_delay(Path("dry.wav"), Path("wet.wav"))
    assert (delay, source) == (0, "nam_unavailable")


def test_detect_delay_when_sibling_venv_has_no_python_binary(monkeypatch, tmp_path):
    site_packages = _fake_sibling_venv(tmp_path, with_python=False)
    monkeypatch.setattr(gdc, "_find_nam_site_packages", lambda: site_packages)
    delay, source = gdc.detect_delay(Path("dry.wav"), Path("wet.wav"))
    assert (delay, source) == (0, "nam_unavailable")


def _run_detect_delay_with_helper_output(monkeypatch, tmp_path, payload, returncode=0, stderr=""):
    """Wire detect_delay's subprocess call to a canned response matching
    nam_delay_helper.py's real JSON-on-stdout contract, without spawning it."""
    site_packages = _fake_sibling_venv(tmp_path)
    monkeypatch.setattr(gdc, "_find_nam_site_packages", lambda: site_packages)

    def fake_run(cmd, capture_output, text, timeout):
        assert cmd[0] == str(site_packages.parent.parent.parent / "bin" / "python3")
        assert cmd[1] == str(gdc._NAM_DELAY_HELPER)
        return SimpleNamespace(stdout=json.dumps(payload) + "\n", stderr=stderr,
                               returncode=returncode)

    monkeypatch.setattr(gdc.subprocess, "run", fake_run)
    return gdc.detect_delay(Path("dry.wav"), Path("wet.wav"))


def test_detect_delay_when_helper_reports_nam_unavailable(monkeypatch, tmp_path):
    delay, source = _run_detect_delay_with_helper_output(
        monkeypatch, tmp_path, {"delay": None, "source": "nam_unavailable", "detail": "no nam"})
    assert (delay, source) == (0, "nam_unavailable")


def test_detect_delay_when_input_is_not_a_nam_standard_file(monkeypatch, tmp_path):
    delay, source = _run_detect_delay_with_helper_output(
        monkeypatch, tmp_path, {"delay": None, "source": "delay_zero_fallback"})
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_uses_real_calibration_when_helper_succeeds(monkeypatch, tmp_path):
    delay, source = _run_detect_delay_with_helper_output(
        monkeypatch, tmp_path, {"delay": 137, "source": "nam_standard_calibration"})
    assert (delay, source) == (137, "nam_standard_calibration")


def test_detect_delay_when_subprocess_times_out(monkeypatch, tmp_path):
    site_packages = _fake_sibling_venv(tmp_path)
    monkeypatch.setattr(gdc, "_find_nam_site_packages", lambda: site_packages)

    import subprocess as _subprocess
    def fake_run(cmd, capture_output, text, timeout):
        raise _subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(gdc.subprocess, "run", fake_run)

    delay, source = gdc.detect_delay(Path("dry.wav"), Path("wet.wav"))
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_when_subprocess_output_is_not_json(monkeypatch, tmp_path):
    site_packages = _fake_sibling_venv(tmp_path)
    monkeypatch.setattr(gdc, "_find_nam_site_packages", lambda: site_packages)
    monkeypatch.setattr(gdc.subprocess, "run",
                         lambda *a, **kw: SimpleNamespace(stdout="", stderr="traceback...",
                                                          returncode=1))
    delay, source = gdc.detect_delay(Path("dry.wav"), Path("wet.wav"))
    assert (delay, source) == (0, "nam_unavailable")


# ---- sanity bound: a "successful" result outside NAM's own search window is untrusted ----

def test_detect_delay_trusts_a_result_at_the_edge_of_the_lookback_window(monkeypatch, tmp_path):
    delay, source = _run_detect_delay_with_helper_output(
        monkeypatch, tmp_path,
        {"delay": gdc._NAM_LOOKBACK_SAMPLES, "source": "nam_standard_calibration"})
    assert (delay, source) == (gdc._NAM_LOOKBACK_SAMPLES, "nam_standard_calibration")


def test_detect_delay_rejects_a_result_beyond_the_lookback_window(monkeypatch, tmp_path):
    """Confirmed hitting this for real: an injected ~1s delay, far outside NAM's own
    lookback=10_000-sample search window, still came back as a plausible-looking "0" instead
    of failing loudly. detect_delay must not trust a match outside the window it's honest
    about searching."""
    delay, source = _run_detect_delay_with_helper_output(
        monkeypatch, tmp_path,
        {"delay": gdc._NAM_LOOKBACK_SAMPLES + 1, "source": "nam_standard_calibration"})
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_rejects_a_result_beyond_the_lookahead_window(monkeypatch, tmp_path):
    delay, source = _run_detect_delay_with_helper_output(
        monkeypatch, tmp_path,
        {"delay": -(gdc._NAM_LOOKAHEAD_SAMPLES + 1), "source": "nam_standard_calibration"})
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_trusts_a_result_at_the_edge_of_the_lookahead_window(monkeypatch, tmp_path):
    delay, source = _run_detect_delay_with_helper_output(
        monkeypatch, tmp_path,
        {"delay": -gdc._NAM_LOOKAHEAD_SAMPLES, "source": "nam_standard_calibration"})
    assert (delay, source) == (-gdc._NAM_LOOKAHEAD_SAMPLES, "nam_standard_calibration")


# --------------------------------------------------------------------------- _find_nam_site_packages

def test_find_nam_site_packages_returns_none_when_no_sibling_checkout(tmp_path):
    assert gdc._find_nam_site_packages(work_dir=tmp_path) is None


def test_find_nam_site_packages_finds_a_real_sibling_venv(tmp_path):
    site_packages = _fake_sibling_venv(tmp_path, with_python=False)
    assert gdc._find_nam_site_packages(work_dir=tmp_path) == site_packages


def test_find_nam_site_packages_env_override_takes_precedence(monkeypatch, tmp_path):
    # work_dir points at a real, discoverable checkout ...
    work_site_packages = _fake_sibling_venv(tmp_path / "work", with_python=False)
    # ... but $NEURAL_AMP_MODELER_HOME points somewhere else entirely, and must win.
    override_root = tmp_path / "elsewhere"
    override_site_packages = override_root / "venv" / "lib" / "python3.12" / "site-packages"
    override_site_packages.mkdir(parents=True)
    monkeypatch.setenv("NEURAL_AMP_MODELER_HOME", str(override_root))

    found = gdc._find_nam_site_packages(work_dir=tmp_path / "work")
    assert found == override_site_packages
    assert found != work_site_packages


def test_find_nam_site_packages_env_override_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    site_packages = tmp_path / "nam" / "venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setenv("NEURAL_AMP_MODELER_HOME", "~/nam")
    assert gdc._find_nam_site_packages() == site_packages


# --------------------------------------------------------------------------- _wet_from_wav

def test_wet_from_wav_aligns_and_resamples_to_the_dry_reference_rate(monkeypatch, tmp_path):
    """End-to-end through _wet_from_wav (not just detect_delay/_align_wet in isolation):
    a capture recorded at a DIFFERENT sample rate than the shared dry sweep must come back
    resampled to sr_in AND with its delay scaled proportionally, not applied in the wrong
    sample-rate's units."""
    sr_wav = 8000
    sr_in = 4000  # half rate, so delay must halve too
    target_len = 20

    # A capture where the first 8 samples (at sr_wav) are pre-response "dead time".
    wet = np.concatenate([np.zeros(8), np.arange(1, 41)]).astype(np.float32)
    wet_path = tmp_path / "capture.wav"
    sf.write(str(wet_path), wet, sr_wav, subtype="FLOAT")
    dry_path = tmp_path / "dry.wav"
    sf.write(str(dry_path), np.zeros(target_len, dtype=np.float32), sr_in, subtype="FLOAT")

    monkeypatch.setattr(gdc, "detect_delay", lambda dry, wetp: (8, "nam_standard_calibration"))

    sig, delay_used, source = gdc._wet_from_wav(wet_path, dry_path, sr_in, target_len)

    assert source == "nam_standard_calibration"
    assert delay_used == 4  # 8 samples at 8kHz -> 4 samples at 4kHz
    assert len(sig) == target_len
    # After resampling 2:1 and dropping the (now 4-sample) lead-in, the signal should be
    # monotonically increasing (the ramp), not still sitting on the dead-time zeros.
    assert sig[0] < sig[-1]
    assert not np.allclose(sig[:3], 0.0)


def test_wet_from_wav_passes_through_unchanged_when_rates_already_match(monkeypatch, tmp_path):
    sr_in = 4000
    target_len = 10
    wet = np.arange(1, 21).astype(np.float32)
    wet_path = tmp_path / "capture.wav"
    sf.write(str(wet_path), wet, sr_in, subtype="FLOAT")
    dry_path = tmp_path / "dry.wav"
    sf.write(str(dry_path), np.zeros(target_len, dtype=np.float32), sr_in, subtype="FLOAT")

    monkeypatch.setattr(gdc, "detect_delay", lambda dry, wetp: (0, "delay_zero_fallback"))

    sig, delay_used, source = gdc._wet_from_wav(wet_path, dry_path, sr_in, target_len)
    assert delay_used == 0
    assert source == "delay_zero_fallback"
    np.testing.assert_array_equal(sig, wet[:target_len])


def test_wet_from_wav_reads_aiff_captures_too(monkeypatch, tmp_path):
    """_wet_from_wav's sf.read is format-agnostic (libsndfile) -- an AIFF capture must read
    identically to the same content saved as WAV. This is what unblocks .aif/.aiff as a
    --captures source kind (see WET_CAPTURE_EXTS): the read path already worked, only the
    CLI's extension whitelist was turning them away."""
    sr_in = 4000
    target_len = 10
    wet = np.arange(1, 21).astype(np.float32)
    wet_path = tmp_path / "capture.aiff"
    sf.write(str(wet_path), wet, sr_in, format="AIFF", subtype="FLOAT")
    dry_path = tmp_path / "dry.wav"
    sf.write(str(dry_path), np.zeros(target_len, dtype=np.float32), sr_in, subtype="FLOAT")

    monkeypatch.setattr(gdc, "detect_delay", lambda dry, wetp: (0, "delay_zero_fallback"))

    sig, delay_used, source = gdc._wet_from_wav(wet_path, dry_path, sr_in, target_len)
    assert delay_used == 0
    np.testing.assert_allclose(sig, wet[:target_len], atol=1e-3)


# --------------------------------------------------------------------------- extension whitelist

def test_wet_capture_exts_accepts_wav_and_aiff_rejects_everything_else():
    assert gdc.WET_CAPTURE_EXTS == (".wav", ".aif", ".aiff")
    accepted = {".nam", *gdc.WET_CAPTURE_EXTS}
    assert Path("Klon G5.aiff").suffix.lower() in accepted
    assert Path("Klon G5.AIF").suffix.lower() in accepted
    assert Path("Klon G5.wav").suffix.lower() in accepted
    assert Path("Klon G5.mp3").suffix.lower() not in accepted
