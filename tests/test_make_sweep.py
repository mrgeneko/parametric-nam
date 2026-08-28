"""Properties the training sweep must have. Each one corresponds to something that actually bit us.

See make_sweep.py.
"""
import numpy as np
import pytest

from make_sweep import SR, build_excitation, flat_burst, log_chirp, silence


@pytest.fixture(scope="module")
def exc():
    return build_excitation(SR)


class TestReproducible:
    """A training input that cannot be regenerated is a training input whose datasets cannot be
    reproduced. That is not hypothetical: sweep120s.wav -- the input a SHIPPED model was trained on
    -- existed only in a Downloads folder, and had to be purged anyway because it turned out to be
    unlicensed audio. The generated part of the sweep must be a function of nothing but the code."""

    def test_byte_identical_across_calls(self):
        assert np.array_equal(build_excitation(SR), build_excitation(SR))

    def test_no_rng_in_the_broadband_burst(self):
        """The obvious way to write `flat_burst` is np.random.randn. Then it is noise, and the file
        is unreproducible unless someone remembers to seed it. It is a deterministic Schroeder-phase
        construction instead."""
        assert np.array_equal(flat_burst(0.5, 0.5, SR), flat_burst(0.5, 0.5, SR))


class TestNoDiscontinuities:
    """An abrupt amplitude jump in the INPUT does not merely sound wrong -- it breaks the simulator.
    A volume-jump splice in an earlier sweep caused ngspice convergence failures at exactly that
    timestamp (sweep-files README, sweep45s). Every segment is fade-joined through silence."""

    def test_segments_start_and_end_at_zero(self):
        for seg in (log_chirp(1.0, 20, 20000, 0.9, SR), flat_burst(0.5, 0.9, SR)):
            assert abs(seg[0]) < 1e-6, "segment must fade IN from silence"
            assert abs(seg[-1]) < 1e-6, "segment must fade OUT to silence"

    def test_a_20khz_chirp_has_a_large_sample_delta_and_that_is_NOT_a_splice(self):
        """PINNED so the check is not 'fixed' by capping the chirp.

        max|diff| on this file is ~1.73, which looks alarming and is not. A sine at frequency f and
        amplitude A, sampled at fs, has adjacent samples differing by up to 2*A*sin(pi*f/fs). At
        20 kHz, A=0.9, 48 kHz that is 1.739 -- it is what a 20 kHz sine LOOKS like near Nyquist, not
        a step. Judge joins by whether the LEVEL is continuous, not by max|diff|."""
        c = log_chirp(6.0, 20, 20000, 0.9, SR)
        expected = 2 * 0.9 * np.sin(np.pi * 20000 / SR)
        assert np.abs(np.diff(c)).max() == pytest.approx(expected, rel=0.05)


class TestCoverage:
    """A model can only learn the circuit where the input has energy. sweepv4 is MIDI instruments
    plus guitar, and it puts 71% of its energy in 80 Hz - 1.28 kHz and 0.2% above 10 kHz -- so the
    tone stack's top end, the presence control and the treble taper were unconstrained, and the model
    was free to invent them."""

    def test_reaches_the_top_of_the_band(self, exc):
        X = np.abs(np.fft.rfft(exc))
        f = np.fft.rfftfreq(len(exc), 1 / SR)
        tot = (X ** 2).sum()
        hi = (X[(f >= 10240) & (f < 20000)] ** 2).sum() / tot
        assert hi > 0.03, f"only {hi:.1%} of energy above 10 kHz — the top of the band is unlearnable"

    def test_covers_several_drive_levels(self, exc):
        """The circuit is NONLINEAR: its response is a curve PER LEVEL, not one curve. A sweep that
        only ever hits one amplitude teaches the band at one point on the transfer curve."""
        blocks = [exc[i:i + SR] for i in range(0, len(exc) - SR, SR)]
        rms = np.array([np.sqrt((b ** 2).mean()) for b in blocks])
        loud = rms[rms > 1e-4]
        assert loud.max() / loud.min() > 5, "the sweep must span a wide range of input levels"

    def test_never_clips(self, exc):
        assert np.abs(exc).max() <= 0.9 + 1e-6, "a clipped input teaches the model the clipper"


class TestLeadInSilence:
    """The sweep MUST open with >= 1 s of silence, and for two independent reasons.

    THE SOLVER. A file that starts mid-signal asks the simulator to jump straight to a large-signal
    solution from an uninitialised state at t=0, and ngspice DIVERGES -- "diverged at t~0", every
    retry rung, no output. Verified: the Boss DS-1 at Dist=0.2 fails on an 8 s slice of this sweep and
    succeeds on the whole file, and giving that same slice 250 ms of silence makes it converge.

    THE WARMUP CONVENTION. gen_dataset_from_schx discards the first warmup_s=1.0 when computing stats and ESR.
    A shorter lead means THE SKIP EATS REAL SIGNAL: at 0.25 s of silence it was discarding 0.75 s of
    the clean 20 Hz chirp -- the low end of the quiet sweep -- from every measurement, silently.
    """

    def test_opens_with_at_least_one_second_of_silence(self, exc):
        first = int(np.argmax(np.abs(exc) > 1e-4))
        assert first / SR >= 1.0, (
            f"sweep starts at t={first/SR:.3f}s; it must open with >= 1 s of silence "
            f"(solver operating point + the 1.0 s warmup convention)"
        )
