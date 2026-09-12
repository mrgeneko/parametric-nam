"""--steps-per-epoch replaces --target-steps, which was never a budget.

In open-ended mode (epochs=0) there is no horizon, so the "total steps" --target-steps named
did not exist: it reached `repeats` through an invented 450-epoch schedule. What it actually
set was steps/epoch, which with SGDR is the length of a restart cycle. Renaming it makes the
quantity honest AND makes the repeats-floor divergence visible instead of silent.
"""
import sys, math
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_pipeline import _derive_repeats, DEFAULT_STEPS_PER_EPOCH


class TestDefaultReproducesTheFleet:
    """Every fleet config used --target-steps 25000, which implied 52-61 steps/epoch on every
    grid. The default must reproduce that, not silently retune every device."""

    def test_default_matches_the_old_implied_value(self):
        assert DEFAULT_STEPS_PER_EPOCH == round(25000 / 450)

    @pytest.mark.parametrize("n_combos", [8, 36, 60, 165])
    def test_deprecated_alias_is_exactly_unchanged(self, n_combos):
        """--target-steps must derive EXACTLY the repeats it always did, so deprecating it
        cannot silently retune a device. It passes target_steps/450 UNROUNDED for this
        reason: 25000/450 = 55.56, and rounding to 56 shifts repeats by one on small grids."""
        old = max(1, round(25000 * 64 / max(1, 450 * n_combos * 0.95)))
        new, _ = _derive_repeats(25000 / 450.0, n_combos, 64, 0.05)
        assert new == old, f"{n_combos} combos: {new} != {old}"

    @pytest.mark.parametrize("n_combos", [8, 36, 60, 165])
    def test_integer_default_drifts_under_two_percent(self, n_combos):
        """The integer default is a rounded 55.56, so it drifts slightly from the old implied
        value. The bound must be RELATIVE: at 8 combos repeats is ~470, so an absolute
        "within one repeat" test is far stricter than at 165 combos where repeats is 23.
        Measured drift: +0.9% / +1.0% / +1.6% / 0.0%. Acceptable for a NEW flag; the
        deprecated alias above stays exact."""
        old = max(1, round(25000 * 64 / max(1, 450 * n_combos * 0.95)))
        new, _ = _derive_repeats(DEFAULT_STEPS_PER_EPOCH, n_combos, 64, 0.05)
        assert abs(new - old) / old < 0.02, f"{n_combos} combos: {new} vs {old}"


class TestTheFloorIsReported:
    """The floor keeps a 5% val split per-combination, but above ~165 combos it means the
    request CANNOT be honoured -- and that divergence used to be silent. Mesa Orange asked
    for 52 steps/epoch and ran 171: SGDR cycles 3.3x longer than --restart-period implied."""

    def test_large_grid_hits_the_floor_and_says_so(self):
        r, achieved = _derive_repeats(56, 576, 64, 0.05)
        assert r == 20, "floor should bind at 576 combos"
        assert achieved > 56 * 2, f"achieved {achieved} should far exceed the request"
        assert achieved == math.ceil(576 * 20 * 0.95 / 64)

    def test_achieved_is_returned_not_the_request(self):
        """Callers must log what RAN, not what was asked for."""
        for nc in (432, 576, 1296):
            r, achieved = _derive_repeats(56, nc, 64, 0.05)
            assert achieved == math.ceil(nc * r * 0.95 / 64)

    def test_no_floor_no_divergence(self):
        r, achieved = _derive_repeats(56, 60, 64, 0.05)
        assert abs(achieved - 56) <= 1, "small grid should hit the request"

    def test_floor_tracks_val_split(self):
        """floor = 1/val_split -- a looser split needs fewer repeats to stay representative."""
        r10, _ = _derive_repeats(56, 576, 64, 0.10)
        r05, _ = _derive_repeats(56, 576, 64, 0.05)
        assert r10 < r05, "a 10% split should floor lower than a 5% one"


class TestDerivationShape:
    def test_repeats_scales_inversely_with_grid_size(self):
        """The whole point: hold steps/epoch constant as the grid changes. Regridding used to
        silently change how long a model trained (126->60 combos HALVED Large Muffin's run)."""
        small, _ = _derive_repeats(56, 30, 64, 0.05)
        large, _ = _derive_repeats(56, 120, 64, 0.05)
        assert small == pytest.approx(large * 4, rel=0.1)

    def test_never_returns_zero(self):
        r, _ = _derive_repeats(1, 100000, 64, 0.05)
        assert r >= 1
