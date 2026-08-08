"""Property tests for the training objective.

Every test here corresponds to a bug that was actually shipped. They are written as PROPERTIES
of the loss rather than golden numbers, because the bugs were not "wrong value" bugs — they were
"optimising the wrong thing" bugs, and a golden number would have happily locked them in.

See internal engineering notes.
"""
import numpy as np
import pytest
import torch

from param_train import ParamLoss, SlimmableParametricA2, esr_per_example, pre_emphasis, validate


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


class TestValidateReportsPerExampleESR:
    """The third bug. validate() computed its reported ESR as a single batch-aggregate ratio
    (sum of squared error over the WHOLE batch / sum of target energy over the whole batch)
    instead of per-example normalise-then-average -- exactly the minibatch trap NAM's own trainer
    warns about (see TestSilentCropDoesNotExplode above), just in the METRIC instead of the loss.
    Training already used esr_per_example (this file); validate() silently used something else, so
    metrics.csv, best-checkpoint selection, and every published ESR_RECORD.md were measuring
    something training never optimised for -- and, because the dataset's random per-item crop
    position changes which examples in a batch happen to be loud vs quiet on every call, the
    reported number could swing by tens of percent between two evaluations of the SAME checkpoint.
    Matches NAM's own esr() (nam/models/losses.py): per-example mean over the sample axis, THEN
    mean over the batch axis -- not one ratio over the whole batch at once."""

    def test_loud_example_does_not_dominate_reported_esr(self):
        quiet_t = _sig(4096, rms=0.09, seed=10)
        loud_t = _sig(4096, rms=0.755, seed=11)
        # Both examples get the SAME absolute error -- under a batch-aggregate ratio the loud
        # example's huge energy would swamp the quiet one's contribution almost entirely.
        err = _sig(4096, rms=0.05, seed=12)
        quiet_p = quiet_t + err
        loud_p = loud_t + err

        class FakeModel(torch.nn.Module):
            def forward(self, audio, params):
                del params
                return audio   # dataset below hands us the prediction directly as "audio"

        class FakePairDataset(torch.utils.data.Dataset):
            def __init__(self, preds, targets):
                self.preds, self.targets = preds, targets

            def __len__(self):
                return len(self.preds)

            def __getitem__(self, i):
                return self.preds[i], self.targets[i], torch.zeros(1)

        preds = [quiet_p.squeeze(0), loud_p.squeeze(0)]
        targets = [quiet_t.squeeze(0), loud_t.squeeze(0)]
        loader = torch.utils.data.DataLoader(FakePairDataset(preds, targets), batch_size=2)

        _, esr_list = validate(FakeModel(), loader, ParamLoss(floor=0.05), "cpu")
        reported = esr_list[0]

        p_cat = torch.stack(preds)   # already (B, 1, L) -- each item is (1, L) post-squeeze(0)
        t_cat = torch.stack(targets)
        batch_aggregate = (((p_cat - t_cat) ** 2).sum() / (t_cat ** 2).sum()).item()
        per_example = esr_per_example(p_cat, t_cat, floor=0.05).item()

        assert reported == pytest.approx(per_example, rel=1e-4), (
            "validate() must report per-example ESR, matching the training objective"
        )
        assert abs(reported - batch_aggregate) / batch_aggregate > 0.2, (
            "validate() must NOT report the batch-aggregate ratio (the bug) -- "
            f"reported={reported:.4f} batch_aggregate={batch_aggregate:.4f}"
        )

    def test_slimmable_path_also_uses_per_example_esr(self):
        """validate() has two branches -- slimmable (list of per-tier preds) and single-model.
        The test above only exercises the single-model branch; this exercises the actual
        multi-tier production path (every real training run uses SlimmableParametricA2)."""
        torch.manual_seed(0)
        model = SlimmableParametricA2(num_params=2, widths=[2, 3])
        model.eval()

        quiet_t = _sig(2048, rms=0.09, seed=20)
        loud_t = _sig(2048, rms=0.755, seed=21)
        t = torch.cat([quiet_t, loud_t])                      # (2, 1, 2048)
        params = torch.tensor([[0.3, 0.7], [0.3, 0.7]])

        with torch.no_grad():
            preds = model(t, params)                          # list of (2, 1, 2048), one per tier

        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, t, p):
                self.t, self.p = t, p

            def __len__(self):
                return self.t.shape[0]

            def __getitem__(self, i):
                return self.t[i], self.t[i], self.p[i]        # same signal as both input and target

        loader = torch.utils.data.DataLoader(FakeDataset(t, params), batch_size=2)
        _, esr_list = validate(model, loader, ParamLoss(floor=0.05), "cpu")

        for tier_idx, pred in enumerate(preds):
            expected = esr_per_example(pred, t, floor=0.05).item()
            assert esr_list[tier_idx] == pytest.approx(expected, rel=1e-4), (
                f"tier {tier_idx}: validate() must report per-example ESR on the slimmable path too"
            )

    def test_val_passes_averages_distinct_draws(self):
        """--val-passes repeats the full validation pass and averages, trading epoch wall-clock
        time for a less noisy reading -- worthwhile because ParamDataset re-crops randomly on
        every __getitem__ call (a fresh np.random.randint), so a second pass draws genuinely
        different windows, not the same ones again. This simulates that with a dataset that
        hands back a DIFFERENT (pred, target) pair on each successive call, and checks
        val_passes=2 actually draws twice and averages them -- not just re-reads one cached pass."""
        t1 = _sig(2048, rms=0.2, seed=30)
        t2 = _sig(2048, rms=0.2, seed=31)
        p1 = t1 + _sig(2048, rms=0.02, seed=32)
        p2 = t2 + _sig(2048, rms=0.05, seed=33)   # a different, larger error on the second draw

        calls = {"n": 0}
        pairs = [(p1.squeeze(0), t1.squeeze(0)), (p2.squeeze(0), t2.squeeze(0))]

        class FakeModel(torch.nn.Module):
            def forward(self, audio, params):
                del params
                return audio   # dataset hands us the prediction directly, like the test above

        class FakeDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, i):
                pred, target = pairs[calls["n"] % len(pairs)]
                calls["n"] += 1
                return pred, target, torch.zeros(1)

        loader = torch.utils.data.DataLoader(FakeDataset(), batch_size=1)
        _, esr_list = validate(FakeModel(), loader, ParamLoss(floor=0.05), "cpu", val_passes=2)

        assert calls["n"] == 2, "val_passes=2 must draw the loader twice, not reuse one pass"
        expected = (esr_per_example(p1, t1, floor=0.05).item()
                    + esr_per_example(p2, t2, floor=0.05).item()) / 2
        assert esr_list[0] == pytest.approx(expected, rel=1e-4), (
            "val_passes must average genuinely distinct draws, not just repeat the first"
        )


class TestParamLossWiring:
    def test_default_is_esr(self):
        """--loss esr must stay the default. Reverting it restores the DS-1 fade-out AND the
        quiet-setting under-fit, while making the headline ESR look better. See internal engineering notes."""
        import inspect
        sig = inspect.signature(ParamLoss.__init__)
        assert sig.parameters["kind"].default == "esr"

    def test_both_kinds_run(self):
        t = _sig(4096, rms=0.2)
        p = t + _sig(4096, rms=0.02, seed=5)
        for kind in ("esr", "mse"):
            loss = ParamLoss(kind=kind, mrstft_weight=0.1)(p, t)
            assert torch.isfinite(loss), kind
