"""knob_map.direction_map on a fake backend: a per-tone linear 'circuit' whose band gains are functions of the knobs."""
import numpy as np
import pytest

import knob_map as K


class FakeBackend:
    """prepare_input scales the probe to `level_v` volts peak; render_many applies gain(params, f) to each probe tone,
    then an optional soft clipper (so a high-gain baseline stops being linear)."""
    def __init__(self, gain, clip_at=None):
        self.gain, self.clip_at = gain, clip_at

    def prepare_input(self, raw, sr, level_v, scratch, tag):
        peak = float(np.abs(raw).max()) + 1e-12
        return (raw / peak * level_v).astype(np.float64)

    def render_many(self, jobs, handle, scratch):
        out = {}
        X = np.fft.rfft(handle)
        f = np.fft.rfftfreq(len(handle), 1 / K.SR)
        for j in jobs:
            Y = np.zeros_like(X)
            for tone in K.ALL_TONES:
                i = int(round(tone * len(handle) / K.SR))
                Y[i] = X[i] * self.gain(j["params"], tone)
            y = np.fft.irfft(Y, len(handle)) + 3.0          # a DC offset, as a tube preamp's output has
            if self.clip_at is not None:
                y = 3.0 + self.clip_at * np.tanh((y - 3.0) / self.clip_at)
            out[j["tag"]] = y
        return out


def classify(name):
    return {"Bass": ("lo", "bass"), "Treble": ("hi", "treble"), "Vol": ("rms", "volume"), "Gain": ("drive", "gain")}.get(name, (None, "unknown"))


def run(gain, knobs, **kw):
    return K.direction_map(FakeBackend(gain, kw.pop("clip_at", None)), knobs, classify, {}, ".", workers=1, n_random=3)


def test_baselines_are_distinct_and_start_at_centre():
    b = K.baselines(["a", "b", "c", "d", "e"], 6)
    assert b[0] == {k: 0.5 for k in "abcde"}
    assert len({tuple(sorted(x.items())) for x in b}) == len(b) == 7


def test_ok_reversed_dead():
    def g(p, f):
        base = 30.0
        if f < 300: base *= 0.3 + 1.4 * p["Bass"]                 # bass band rises with Bass
        if f > 2500: base *= 1.7 - 1.4 * p["Treble"]              # treble band FALLS with Treble
        return base
    r = run(g, ["Bass", "Treble", "Vol"])
    assert r["Bass"]["verdict"] == "OK"
    assert r["Treble"]["verdict"] == "REVERSED"
    assert r["Vol"]["verdict"] == "DEAD"


def test_interaction_when_another_path_dominates():
    def g(p, f):
        if f < 300:
            # Bass raises the bass of the tone-stack path but loads the bypass path; which wins depends on Vol
            return 10.0 * (0.2 + p["Bass"]) * p["Vol"] + 10.0 * (1.2 - p["Bass"]) * (1.0 - p["Vol"])
        return 10.0
    r = run(g, ["Bass", "Vol"])
    assert r["Bass"]["verdict"] == "INTERACTION"
    assert r["Bass"]["ok"] and r["Bass"]["reversed"]


def test_input_limited_baseline_is_skipped_not_scored():
    # at Gain > 0.7 the circuit's gain is so low the probe would need more than the 1 V input cap: skipped, not scored
    def g(p, f):
        return 30.0 * (1.0 + 0.4 * p["Bass"]) * (1e-6 if p["Gain"] > 0.7 else 1.0)
    r = run(g, ["Bass", "Gain"])
    assert r["Bass"]["skipped_nonlinear"] > 0
    assert r["Bass"]["usable"] > 0 and r["Bass"]["reversed"] == 0 and r["Bass"]["verdict"] == "OK"


def test_clipping_everywhere_is_untested_not_failed():
    r = run(lambda p, f: 30.0 * (1.0 - 0.5 * p["Bass"]), ["Bass"], clip_at=0.03)   # a REVERSED knob, but never linear
    assert r["Bass"]["verdict"] == "UNTESTED"


def test_travel_concentrated_in_first_tenth():
    def squeezed(p, f):      # the bass band is fully up by knob 0.1, then flat: a mirrored taper
        return 30.0 * (0.05 + 0.95 * min(1.0, p["Bass"] / 0.1) ** 1.0) if f < 300 else 30.0
    def smooth(p, f):
        return 30.0 * (0.1 + 0.9 * p["Bass"]) if f < 300 else 30.0
    # 0.2 -> 0.8 is flat for the squeezed one, so give it a slow ramp on top that keeps the verdict OK
    def squeezed_ok(p, f):
        return squeezed(p, f) * (1.0 + 0.3 * p["Bass"]) if f < 300 else 30.0
    assert run(squeezed_ok, ["Bass"])["Bass"].get("travel_concentrated") is True
    assert run(smooth, ["Bass"])["Bass"].get("travel_concentrated") is False
