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


class TestDefaults:
    """The default is a DELIBERATE round number, not a behaviour-preserving one.

    Every fleet config used --target-steps 25000, which through the nominal-450 formula
    implied 55.6 steps/epoch. The default is 50 -- about 10% shorter cycles. That is a real
    change, and it is fine because it only reaches NEW configs: every existing one specifies
    target-steps, which takes the deprecated-alias path and is exact (below).
    """

    def test_default_is_fifty(self):
        assert DEFAULT_STEPS_PER_EPOCH == 50

    def test_default_is_within_ten_percent_of_the_fleets_implied_value(self):
        implied = 25000 / 450
        assert abs(DEFAULT_STEPS_PER_EPOCH - implied) / implied < 0.11

    @pytest.mark.parametrize("n_combos", [8, 36, 60, 165])
    def test_deprecated_alias_is_exactly_unchanged(self, n_combos):
        """--target-steps must derive EXACTLY the repeats it always did, so deprecating it
        cannot silently retune an existing device. It passes target_steps/450 UNROUNDED for
        this reason: rounding to an int shifts repeats on small grids."""
        old = max(1, round(25000 * 64 / max(1, 450 * n_combos * 0.95)))
        new, _ = _derive_repeats(25000 / 450.0, n_combos, 64, 0.05)
        assert new == old, f"{n_combos} combos: {new} != {old}"


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


class TestParamTrainOwnsTheDerivation:
    """repeats is derived in param_train.py and NOWHERE else.

    It used to be computed in run_pipeline.py from --target-steps and then silently
    overridden by param_train's val-split floor, so the two disagreed: Mesa Orange was
    handed repeats 6 (52 steps/epoch) and trained at 20 (171), making every SGDR cycle
    3.3x longer than --restart-period 50 was calibrated for. Nothing reported it.
    """

    def test_param_train_defaults_repeats_to_derive_not_one(self):
        """The old default of 1 gave an EMPTY val split and runs that self-stopped on a
        false plateau having barely trained (72-combo config, 2026-08-30). Running
        param_train.py directly must no longer land there."""
        import param_train, argparse, inspect
        src = inspect.getsource(param_train.main) if hasattr(param_train, "main") else ""
        # the flag itself is what matters
        ap = argparse.ArgumentParser()
        found = [l for l in inspect.getsource(param_train).splitlines()
                 if '"--repeats"' in l]
        assert found, "--repeats flag not found"
        assert "default=None" in found[0], f"repeats must default to derive, got: {found[0]}"

    def test_param_train_exposes_steps_per_epoch(self):
        import inspect, param_train
        src = inspect.getsource(param_train)
        assert '"--steps-per-epoch"' in src
        assert param_train.DEFAULT_STEPS_PER_EPOCH == DEFAULT_STEPS_PER_EPOCH, \
            "run_pipeline and param_train must agree on the default"


class TestTemplateUsesTheNewFlag:
    """A freshly scaffolded device must not take the DEPRECATED path.

    The template kept writing `target-steps = 25000`, so every newly scaffolded config
    silently used the alias -- which rather defeats deprecating it. Caught on the Arbiter
    Fuzz Face config, scaffolded after the rename and still carrying target-steps.
    """

    def _template(self):
        from pathlib import Path
        import scaffold_config
        return Path(scaffold_config.TEMPLATE).read_text()

    def test_template_sets_steps_per_epoch(self):
        import re
        t = self._template()
        m = re.search(r"^steps-per-epoch\s*=\s*(\d+)", t, re.M)
        assert m, "template must set steps-per-epoch"
        assert int(m.group(1)) == DEFAULT_STEPS_PER_EPOCH

    def test_template_no_longer_sets_target_steps(self):
        import re
        t = self._template()
        assert not re.search(r"^target-steps\s*=", t, re.M), \
            "template still emits the deprecated target-steps"

    def test_template_value_round_trips_through_the_config_loader(self):
        """The key must be one run_pipeline.load_config maps to the real argparse dest --
        a near-miss name would look set and be silently ignored."""
        import tomllib
        from run_pipeline import load_config
        import tempfile, os
        t = self._template()
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(t); path = f.name
        try:
            cfg = load_config(path)
            assert cfg.get("steps_per_epoch") == DEFAULT_STEPS_PER_EPOCH, \
                f"loader did not pick it up: {[k for k in cfg if 'step' in k]}"
        finally:
            os.unlink(path)
