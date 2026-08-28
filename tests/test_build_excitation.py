"""Properties build_excitation.py's synthesis primitives must have. This tool assembles
the actual training excitation the fleet trains on, so a wrong sweep frequency range or a
transient burst that isn't actually broadband/instant-attack silently produces a worse dataset,
not a crash. _fade/_audio_provenance/_tool_git_rev are deliberately duplicated from
pick_dynamic_window.py (see this module's own docstring: independent, no cross-tool import) and
already covered there; this file focuses on what's unique here -- the sweep and synthesized
transient-burst generators.

See build_excitation.py.
"""
import numpy as np
import pytest

from build_excitation import SR, _fade, _log_sweep, _transient_burst


class TestLogSweep:
    def test_peak_amplitude_matches_amp(self):
        s = _log_sweep(f0=40.0, f1=12000.0, dur=3.0, amp=0.7)
        assert np.abs(s).max() == pytest.approx(0.7, rel=1e-3)

    def test_length_matches_duration(self):
        s = _log_sweep(f0=40.0, f1=12000.0, dur=2.0, amp=1.0)
        assert len(s) == int(SR * 2.0)

    def test_deterministic(self):
        a = _log_sweep(40.0, 12000.0, 1.0, 1.0)
        b = _log_sweep(40.0, 12000.0, 1.0, 1.0)
        assert np.array_equal(a, b)

    def test_starts_low_frequency_and_ends_high_frequency(self):
        # A log sweep's instantaneous frequency rises monotonically -- zero-crossing rate in a
        # short window near the start must be far lower than near the end.
        s = _log_sweep(f0=40.0, f1=12000.0, dur=3.0, amp=1.0)
        win = int(SR * 0.01)  # 10ms
        start_crossings = np.sum(np.diff(np.sign(s[:win])) != 0)
        end_crossings = np.sum(np.diff(np.sign(s[-win:])) != 0)
        assert end_crossings > start_crossings * 10


class TestTransientBurst:
    def test_peak_amplitude_equals_amp_exactly(self):
        # The module docstring's own claim: the two-stage normalize makes "peak level == amp"
        # exact, not merely close (unlike normalizing the un-enveloped signal alone would).
        b = _transient_burst(sec=0.5, amp=1.0, decay_tau=0.03)
        assert np.abs(b).max() == pytest.approx(1.0, rel=1e-9)

    def test_scales_linearly_with_amp(self):
        b1 = _transient_burst(0.5, 1.0, 0.03)
        b2 = _transient_burst(0.5, 2.0, 0.03)
        assert np.allclose(b2, b1 * 2.0)

    def test_deterministic_no_rng(self):
        a = _transient_burst(0.5, 1.0, 0.03)
        b = _transient_burst(0.5, 1.0, 0.03)
        assert np.array_equal(a, b)

    def test_default_decay_tau_gives_crest_factor_near_8_5(self):
        # The documented design target: crest ~8.5 matches a real hard pick-attack.
        b = _transient_burst(0.5, 1.0, decay_tau=0.03)
        rms = np.sqrt(np.mean(b ** 2))
        crest = np.abs(b).max() / rms
        assert crest == pytest.approx(8.5, abs=0.2)

    def test_most_energy_stays_in_the_20hz_16khz_band(self):
        # The pre-envelope spectrum is strictly masked to [20, 16000] Hz, but multiplying by
        # the exponential decay envelope afterward necessarily leaks some energy outside it
        # (amplitude modulation broadens a spectrum) -- so "most", not "all".
        b = _transient_burst(0.5, 1.0, decay_tau=0.03)
        spec = np.abs(np.fft.rfft(b))
        freqs = np.fft.rfftfreq(len(b), 1 / SR)
        in_band = spec[(freqs >= 20) & (freqs <= 16000)].sum()
        out_of_band = spec[(freqs < 20) | (freqs > 16000)].sum()
        assert out_of_band < 0.2 * in_band

    def test_longer_decay_tau_sustains_more_energy_later_in_the_burst(self):
        short = _transient_burst(0.5, 1.0, decay_tau=0.01)
        long = _transient_burst(0.5, 1.0, decay_tau=0.1)
        tail = slice(int(SR * 0.2), int(SR * 0.5))
        assert np.sqrt(np.mean(long[tail] ** 2)) > np.sqrt(np.mean(short[tail] ** 2))


class TestFade:
    def test_does_not_mutate_the_input(self):
        y = np.ones(SR, dtype=np.float32)
        original = y.copy()
        _fade(y, 10, 10)
        assert np.array_equal(y, original)

    def test_edges_fade_toward_silence(self):
        y = np.ones(SR, dtype=np.float32)
        out = _fade(y, ms_in=10, ms_out=10)
        assert abs(out[0]) < 1e-6
        assert abs(out[-1]) < 1e-6
