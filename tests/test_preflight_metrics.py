"""Properties preflight.py's pure per-render metric functions must have. These decide
"dead"/"reversed"/"OK" for a knob check (see the module docstring's Fulltone OCD Tone/Klon Tone
war stories) -- a subtle bug in a band ratio or the spike/ESR formula reads as the WRONG KNOB
CONCLUSION on a real circuit, not a crash. Exercised directly against synthetic tones instead
of a real render, so the metric math is verified independently of any circuit or backend.

See preflight.py.
"""
import numpy as np
import pytest

from preflight import SR, _band, esr, metric, spikes


def sine(freq, amp=1.0, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestBand:
    def test_energy_concentrates_inside_the_requested_band(self):
        y = sine(5000)
        inside = _band(y, 2000, 8000)
        outside = _band(y, 40, 800)
        assert inside > 1000 * outside

    def test_silence_has_no_energy_anywhere(self):
        y = np.zeros(SR, dtype=np.float32)
        assert _band(y, 40, 8000) == pytest.approx(0.0, abs=1e-9)


class TestMetricRmsAndDrive:
    @pytest.mark.parametrize("kind", ["rms", "drive"])
    def test_matches_the_amplitude_of_a_sine(self, kind):
        y = sine(440, amp=0.6)
        assert metric(y, kind) == pytest.approx(0.6 / np.sqrt(2), rel=1e-2)

    def test_drops_the_first_tenth_of_a_second_attack_transient(self):
        """metric() slices off y[int(0.1*SR):] before measuring -- a loud attack in the probe's
        first 0.1s (every render's cold-start settle) must not pollute the steady-state reading
        used to judge a knob's direction."""
        attack = np.full(int(0.1 * SR), 50.0, dtype=np.float32)
        tail = sine(440, amp=0.1, dur=1.0)
        y = np.concatenate([attack, tail])
        assert metric(y, "rms") == pytest.approx(0.1 / np.sqrt(2), rel=1e-2)


class TestMetricEqBands:
    def test_hi_is_large_for_a_treble_tone_and_small_for_a_bass_tone(self):
        assert metric(sine(5000), "hi") > 100 * metric(sine(100), "hi")

    def test_lo_is_large_for_a_bass_tone_and_small_for_a_midrange_tone(self):
        assert metric(sine(100), "lo") > 100 * metric(sine(1000), "lo")

    def test_mid_is_large_for_a_midrange_tone_and_small_for_bass_or_treble(self):
        m_mid = metric(sine(700), "mid")
        assert m_mid > 100 * metric(sine(100), "mid")
        assert m_mid > 100 * metric(sine(5000), "mid")

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            metric(sine(440), "bogus")


class TestEsr:
    def test_identical_signals_have_zero_error(self):
        y = sine(440)
        assert esr(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_truncates_to_the_shorter_signal(self):
        a = sine(440, dur=1.0)
        b = np.concatenate([sine(440, dur=1.0), np.full(1000, 999.0, dtype=np.float32)])
        assert esr(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_larger_deviation_gives_larger_error(self):
        ref = sine(440, amp=1.0)
        near = ref + 0.01
        far = ref + 0.5
        assert esr(far, ref) > esr(near, ref)

    def test_all_zero_reference_does_not_raise_or_return_nan(self):
        e = esr(sine(440), np.zeros(SR, dtype=np.float32))
        assert np.isfinite(e)


class TestSpikes:
    def test_smooth_sine_has_no_spikes(self):
        assert spikes(sine(440)) == 0

    def test_flat_constant_signal_has_no_spikes(self):
        # A uniformly large signal must not be flagged: spikes() compares each sample to its
        # NEIGHBORS, not to some absolute threshold.
        assert spikes(np.full(1000, 5.0, dtype=np.float32)) == 0

    def test_an_isolated_large_sample_is_detected(self):
        y = np.zeros(1000, dtype=np.float32)
        y[500] = 100.0
        assert spikes(y) >= 1
