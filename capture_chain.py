#!/usr/bin/env python3
"""The virtual capture chain: model the audio interface a hardware NAM capture goes through.

WHY THIS EXISTS. A standard .nam is trained on a wet file recorded through an audio
interface, whose input stage rolls off below ~20 Hz. NAM's architecture was designed
against that distribution. Rendering a circuit in a simulator and probing a node directly
skips that stage entirely, so our targets can carry sub-audio content no hardware capture
would ever contain -- and NAM's ~52 ms receptive field cannot model content whose state
evolves over seconds.

Measured on Duke of Tone (Distortion), 2026-09-10: 64-74% of the rendered target's energy
sat BELOW 19 Hz (62% below 5 Hz), against 0.49% in the input. Cause is real and faithful to
the pedal -- an out-of-loop BAT46 shunt clipper into a soft bias rail (VB: 47k||47k with
100 uF, tau 2.35 s), passed by an output network with tau 2.0 s (R11 1M with C8||C9 2 uF,
0.08 Hz corner). Both trainers plateaued at the same place: our ParametricA2 at w8 0.525 and
the OFFICIAL upstream nam-full at a comparable ESR on a static single-setting capture. Two
different architectures, one shared receptive field -- the failure tracked the shared
constraint. The predicted floor from unmodellable content, >=0.64, bracketed the observed
0.525.

For scale: 13 other fleet datasets (Joyo, Mesa RED/Orange, JCM800, Duke OVERDRIVE, statics)
all sit at 0.20-9.47% sub-19 Hz. The Distortion is not the tail of that distribution.

FILTER SHAPE is derived, not picked. A converter spec'd "20 Hz-20 kHz +/-0.5 dB" has its
-3 dB point well below 20 Hz. Solving for exactly -0.5 dB at 20 Hz gives 11.8 Hz at 2nd
order (or 7.0 Hz at 1st). 2nd order is the default: it reaches -42.9 dB at 1 Hz where 1st
order manages only -17.0 dB, while being FLATTER above the corner (-0.01 dB at 50 Hz vs
-0.08 dB) -- it removes more of what we cannot model and disturbs less of what we can.

CAUSAL, ALWAYS. sosfilt, never sosfiltfilt. A zero-phase filter makes the target depend on
FUTURE input, so its pre-ringing is unpredictable to a causal model by construction -- a
self-inflicted error floor in the name of removing one. The 2nd-order 11.8 Hz impulse
response is ~0.1% by 52 ms, so it sits inside the receptive field and is learnable.

THIS IS NOT A UNIVERSAL SAFETY NET. It removes sub-audio, which is the SYMPTOM. A circuit
whose multi-second state reaches into the AUDIO band is still unmodellable and this will not
save it -- the Distortion's own bias modulation shows up as a 0.119 ESR audio-band
difference. Use context_sensitivity() to detect that class; an LF-energy check cannot.
"""
import numpy as np
from scipy.signal import butter, sosfilt

# -0.5 dB at 20 Hz at 2nd order; see FILTER SHAPE above.
DEFAULT_CORNER_HZ = 11.8
DEFAULT_ORDER = 2


def _sos(sr, corner_hz=DEFAULT_CORNER_HZ, order=DEFAULT_ORDER):
    if not (0 < corner_hz < sr / 2):
        raise ValueError(f"corner_hz {corner_hz} must be in (0, {sr/2})")
    return butter(order, corner_hz, "hp", fs=sr, output="sos")


def capture_chain(y, sr, corner_hz=DEFAULT_CORNER_HZ, order=DEFAULT_ORDER):
    """Apply the virtual interface input stage. CAUSAL.

    `y` may be 1-D (one signal) or 2-D (combinations x samples); filtering is along the
    LAST axis either way, so a (n_combos, n_samples) dataset filters in one call without
    the caller transposing or looping.
    """
    a = np.asarray(y)
    out = sosfilt(_sos(sr, corner_hz, order), a.astype(np.float64), axis=-1)
    return out.astype(a.dtype, copy=False)


def lf_energy_fraction(y, sr, below_hz=19.0):
    """Fraction (0-1) of `y`'s energy below `below_hz`. The cheap screen.

    19 Hz is not arbitrary: it is 1/0.0517 s, the reciprocal of the A2 receptive field, so
    it names the slowest periodicity a model could resolve within one window.
    """
    a = np.asarray(y, dtype=np.float64).ravel()
    if a.size < 2:
        return 0.0
    w = np.hanning(a.size)
    S = np.abs(np.fft.rfft(a * w)) ** 2
    f = np.fft.rfftfreq(a.size, 1.0 / sr)
    tot = S.sum()
    return float(S[f < below_hz].sum() / tot) if tot > 0 else 0.0


def context_sensitivity(y_ctx_a, y_ctx_b, sr, skip_s=0.2):
    """ESR between two renders of the SAME input preceded by DIFFERENT context.

    The direct test for state outliving the receptive field, and the one that catches what
    lf_energy_fraction cannot: a circuit whose long memory reaches the audio band. Both
    inputs must be the identical segment; pass the post-context portion of each render.
    `skip_s` drops the first 0.2 s, where a genuine short transient still differs.
    """
    a = np.asarray(y_ctx_a, dtype=np.float64).ravel()
    b = np.asarray(y_ctx_b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    s = min(int(skip_s * sr), max(0, n - 1))
    a, b = a[s:n], b[s:n]
    den = (b ** 2).sum()
    return float(((a - b) ** 2).sum() / den) if den > 0 else 0.0
