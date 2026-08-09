"""Tests for gen_dataset_from_captures.py's .wav capture path -- specifically the delay/latency
calibration and alignment logic, which is the one thing a raw .wav capture needs that a .nam
capture (pure digital inference, phase-aligned by construction) doesn't. See that module's
detect_delay()/_align_wet() docstrings for the full reasoning.

detect_delay() is tested entirely through monkeypatched _import_nam_core() -- never against a
real neural-amp-modeler install -- so these pass whether or not that optional package (or its
sibling-checkout venv) is present in the environment running the tests.
"""
from pathlib import Path

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

def test_detect_delay_when_nam_core_unimportable(monkeypatch):
    monkeypatch.setattr(gdc, "_import_nam_core", lambda: None)
    delay, source = gdc.detect_delay(dry_wav=Path("dry.wav"), wet_wav=Path("wet.wav"))
    assert (delay, source) == (0, "nam_unavailable")


def test_detect_delay_falls_back_when_input_is_not_a_nam_standard_file(monkeypatch):
    def fake_detect_input_version(path):
        return None, False  # NAM's own "no match" return shape

    monkeypatch.setattr(gdc, "_import_nam_core",
                         lambda: (fake_detect_input_version, ValueError, None))
    delay, source = gdc.detect_delay(dry_wav=Path("sweepv5.wav"), wet_wav=Path("wet.wav"))
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_falls_back_when_detection_raises_input_validation_error(monkeypatch):
    class _FakeInputValidationError(Exception):
        pass

    def fake_detect_input_version(path):
        raise _FakeInputValidationError("no strong or weak match")

    monkeypatch.setattr(gdc, "_import_nam_core",
                         lambda: (fake_detect_input_version, _FakeInputValidationError, None))
    delay, source = gdc.detect_delay(dry_wav=Path("sweepv5.wav"), wet_wav=Path("wet.wav"))
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_falls_back_on_any_other_detection_error_without_crashing(monkeypatch):
    """A FLOAT-subtype WAV can crash NAM's own reader outright (see capture_static.py's
    identical trap) -- detect_delay must degrade, not propagate."""
    def fake_detect_input_version(path):
        raise RuntimeError("unknown format: 3")

    monkeypatch.setattr(gdc, "_import_nam_core",
                         lambda: (fake_detect_input_version, ValueError, None))
    delay, source = gdc.detect_delay(dry_wav=Path("scaled.wav"), wet_wav=Path("wet.wav"))
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_uses_nam_calibration_when_input_matches_a_standard_file(monkeypatch):
    class _Calibration:
        recommended = 137

    class _LatencyResult:
        calibration = _Calibration()

    def fake_detect_input_version(path):
        return "v3", True

    def fake_analyze_latency(user_latency, input_version, input_path, output_path, silent):
        assert user_latency is None
        assert input_version == "v3"
        assert silent is True
        return _LatencyResult()

    monkeypatch.setattr(gdc, "_import_nam_core",
                         lambda: (fake_detect_input_version, ValueError, fake_analyze_latency))
    delay, source = gdc.detect_delay(dry_wav=Path("sweep-v3.wav"), wet_wav=Path("wet.wav"))
    assert (delay, source) == (137, "nam_standard_calibration")


def test_detect_delay_falls_back_when_calibration_finds_no_impulse_response(monkeypatch):
    class _Calibration:
        recommended = None  # NAM's own "couldn't calibrate" signal

    class _LatencyResult:
        calibration = _Calibration()

    monkeypatch.setattr(
        gdc, "_import_nam_core",
        lambda: (lambda p: ("v3", True), ValueError,
                 lambda **kw: _LatencyResult()))
    delay, source = gdc.detect_delay(dry_wav=Path("sweep-v3.wav"), wet_wav=Path("silent.wav"))
    assert (delay, source) == (0, "delay_zero_fallback")


def test_detect_delay_falls_back_when_analyze_latency_itself_raises(monkeypatch):
    def fake_analyze_latency(**kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        gdc, "_import_nam_core",
        lambda: (lambda p: ("v3", True), ValueError, fake_analyze_latency))
    delay, source = gdc.detect_delay(dry_wav=Path("sweep-v3.wav"), wet_wav=Path("wet.wav"))
    assert (delay, source) == (0, "delay_zero_fallback")


# --------------------------------------------------------------------------- _find_nam_site_packages

def test_find_nam_site_packages_returns_none_when_no_sibling_checkout(tmp_path):
    assert gdc._find_nam_site_packages(work_dir=tmp_path) is None


def test_find_nam_site_packages_finds_a_real_sibling_venv(tmp_path):
    site_packages = tmp_path / "neural-amp-modeler" / "venv" / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    assert gdc._find_nam_site_packages(work_dir=tmp_path) == site_packages


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
