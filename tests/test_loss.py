"""Property tests for the training objective.

Every test here corresponds to a bug that was actually shipped. They are written as PROPERTIES
of the loss rather than golden numbers, because the bugs were not "wrong value" bugs — they were
"optimising the wrong thing" bugs, and a golden number would have happily locked them in.

See docs/loss-energy-bias.md.
"""
import numpy as np
import pytest
import torch

from param_train import ParamLoss, esr_per_example, pre_emphasis


def _sig(n, rms, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, n, generator=g)
    return x * (rms / x.pow(2).mean().sqrt())


class TestScaleInvariance:
    """THE bug. MSE is an ABSOLUTE error, so an example's pull on the gradient is proportional
    to its ENERGY. Across the Big Muff's knob grid output RMS spans 0.090..0.755 — a 70x energy
    ratio — so the loudest 8% of permutations took 28% of the gradient and the quietest half got
    17%. The parametric model neglected the quiet settings, which is why its ESR sat above a
    static model's.

    A loss that is fit for a MULTI-SETTING model must not care how loud the setting is."""

    def test_esr_is_scale_invariant(self):
        target = _sig(4096, rms=0.1)
        pred = target + _sig(4096, rms=0.01, seed=1)          # 10% relative error

        quiet = esr_per_example(pred, target, floor=0.05)
        loud = esr_per_example(pred * 10, target * 10, floor=0.05)   # same RELATIVE error, 20 dB up

        assert torch.allclose(quiet, loud, rtol=1e-5), (
            "ESR must be scale-invariant: the same relative error at any level is the same loss."
        )

    def test_mse_is_NOT_scale_invariant(self):
        """Pinned deliberately. This is the defect, expressed as a test — if someone 'fixes' the
        loss back to MSE, this documents exactly what they are buying."""
        target = _sig(4096, rms=0.1)
        pred = target + _sig(4096, rms=0.01, seed=1)

        quiet = torch.nn.functional.mse_loss(pred, target)
        loud = torch.nn.functional.mse_loss(pred * 10, target * 10)

        # 20 dB louder → 100x the loss, for identical relative error.
        assert loud / quiet == pytest.approx(100.0, rel=0.02)

    def test_quiet_and_loud_permutations_contribute_equally(self):
        """The 70x-energy-ratio case, end to end. Under ESR both examples in the batch must pull
        on the gradient by comparable amounts; under MSE the loud one dominates."""
        quiet_t = _sig(4096, rms=0.09)                 # quietest Big Muff permutation
        loud_t = _sig(4096, rms=0.755, seed=2)         # loudest
        quiet_p = quiet_t + _sig(4096, rms=0.009, seed=3)    # both 10% relative error
        loud_p = loud_t + _sig(4096, rms=0.0755, seed=4)

        t = torch.cat([quiet_t, loud_t])
        p = torch.cat([quiet_p, loud_p])

        num = ((p - t) ** 2).sum(dim=(1, 2))
        den = (t ** 2).sum(dim=(1, 2))
        esr_each = num / den
        ratio_esr = (esr_each[1] / esr_each[0]).item()

        mse_each = ((p - t) ** 2).mean(dim=(1, 2))
        ratio_mse = (mse_each[1] / mse_each[0]).item()

        assert ratio_esr == pytest.approx(1.0, rel=0.1), "ESR: equal pull"
        assert ratio_mse > 50, "MSE: the loud permutation dominates (this is the bug)"


class TestSilentCropDoesNotExplode:
    """The second bug, shipped and reverted within the hour. NAM's own trainer warns about it:

        "Be careful when computing ESR on minibatches! ... (Hint: think about what happens if one
         item in the minibatch is all zeroes...)"

    An ABSOLUTE epsilon cannot save you. A 0.5 s crop at RMS 0.09 has energy ~194, so eps=1e-4 is
    irrelevant to it — while a decay-tail crop has energy ~0.02 and a silent one ~0. AND THE CROPS
    WE ARE TRYING TO UPWEIGHT ARE THE NEAR-SILENT ONES, so it detonates exactly where the fix is
    supposed to work. Hence a floor relative to the BATCH MEAN."""

    def test_all_zero_item_does_not_dominate(self):
        t = torch.cat([_sig(4096, rms=0.3, seed=i) for i in range(7)] + [torch.zeros(1, 1, 4096)])
        p = t + _sig(4096, rms=0.01, seed=99) * 0.0
        p = t.clone()
        p += torch.randn_like(p) * 0.01                       # a decent model everywhere

        loss = esr_per_example(p, t, floor=0.05)
        assert torch.isfinite(loss), "a silent batch item must not produce inf/nan"
        assert loss.item() < 1.0, (
            f"a single silent crop must not dominate the batch loss (got {loss.item():.1f})"
        )

    def test_near_silent_tail_crop_is_upweighted_but_bounded(self):
        """It must still COUNT — that is the entire point of the change — just not explode."""
        t = torch.cat([_sig(4096, rms=0.3, seed=i) for i in range(7)] + [_sig(4096, rms=1e-5, seed=8)])
        p = t + torch.randn_like(t) * 0.003

        num = ((p - t) ** 2).sum(dim=(1, 2))
        den = (t ** 2).sum(dim=(1, 2))
        den = torch.clamp(den, min=0.05 * den.mean())
        each = num / den

        assert each[7] > 5 * each[0], "the quiet tail crop must be upweighted"
        assert each[7] < 1000 * each[0], "...but bounded, not a gradient bomb"

    def test_no_floor_IS_a_gradient_bomb(self):
        """Pinned so nobody 'simplifies' the floor away."""
        t = torch.cat([_sig(4096, rms=0.3, seed=i) for i in range(7)] + [torch.zeros(1, 1, 4096)])
        p = t + torch.randn_like(t) * 0.01

        num = ((p - t) ** 2).sum(dim=(1, 2))
        bad = (num / ((t ** 2).sum(dim=(1, 2)) + 1e-4)).mean()      # the old eps
        good = esr_per_example(p, t, floor=0.05)

        assert bad > 100 * good, "documenting the failure mode we removed"


class TestPreEmphasis:
    def test_zero_coef_is_identity(self):
        x = _sig(1024, rms=0.2)
        assert torch.equal(pre_emphasis(x, 0.0), x)

    def test_high_pass_attenuates_dc(self):
        dc = torch.ones(1, 1, 1024)
        out = pre_emphasis(dc, 0.85)
        assert out.abs().mean() < 0.2, "pre-emphasis must attenuate a constant"


class TestParamLossWiring:
    def test_default_is_esr(self):
        """--loss esr must stay the default. Reverting it restores the DS-1 fade-out AND the
        quiet-setting under-fit, while making the headline ESR look better. See docs/RETRAINING.md."""
        import inspect
        sig = inspect.signature(ParamLoss.__init__)
        assert sig.parameters["kind"].default == "esr"

    def test_both_kinds_run(self):
        t = _sig(4096, rms=0.2)
        p = t + _sig(4096, rms=0.02, seed=5)
        for kind in ("esr", "mse"):
            loss = ParamLoss(kind=kind, mrstft_weight=0.1)(p, t)
            assert torch.isfinite(loss), kind
