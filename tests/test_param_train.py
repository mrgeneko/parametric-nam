"""Tests for param_train.py's restore_scheduler_on_resume() -- the fix for a real bug
found 2026-09-01: reconstructing CosineAnnealingWarmRestarts directly via
`last_epoch=<big number>` does not replay restart history (torch's own constructor just
sets T_cur=last_epoch, T_i=T_0 with no re-derivation), so every --resume past the first
SGDR cycle produced a scrambled scheduler state that only self-corrected after several
epochs of spurious wrap-arounds -- confirmed empirically on a real resume at epoch 2499
(5 bogus wraps, LR bouncing 0.5/0.86/0.32/0.91/0.35x eta_max before settling on an
arbitrary ~1600-epoch cycle nobody asked for).

See param_train.py.
"""
import math

import pytest
import torch

import param_train as pt


def _make_optimizer(lr=3e-4):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for g in opt.param_groups:
        g["initial_lr"] = lr
    return opt


def _make_scheduler_factory(optimizer, T_0=50, T_mult=2, open_ended=True):
    def make_scheduler(last_epoch):
        if open_ended:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, last_epoch=last_epoch)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 450, last_epoch=last_epoch)
    return make_scheduler


# --------------------------------------------------------- open-ended, new-format ckpt

def test_new_format_checkpoint_restores_exact_position_no_bogus_wraps():
    """A checkpoint saved by the fixed code (scheduler_T_cur/T_i present) must restore
    T_cur/T_i exactly, with no spurious wrap-arounds on the epochs that follow."""
    opt = _make_optimizer()
    make_scheduler = _make_scheduler_factory(opt, T_0=50, T_mult=2)
    scheduler = make_scheduler(-1)
    ckpt = {"scheduler_last_epoch": 2499, "scheduler_T_cur": 37, "scheduler_T_i": 50}

    scheduler = pt.restore_scheduler_on_resume(scheduler, opt, ckpt, True, make_scheduler)

    assert scheduler.T_cur == 37
    assert scheduler.T_i == 50
    expected_lr = 3e-4 * (1 + math.cos(math.pi * 37 / 50)) / 2
    assert opt.param_groups[0]["lr"] == pytest.approx(expected_lr, abs=1e-12)

    # Step forward to the cycle boundary (13 epochs: 50 - 37) -- must land cleanly on
    # T_cur=0 with T_i grown by T_mult, never wrapping early or overshooting.
    for _ in range(13):
        scheduler.step()
    assert scheduler.T_cur == 0
    assert scheduler.T_i == 100          # T_mult=2 growth applied at the boundary


def test_new_format_restore_keeps_current_cycle_length_when_mult_changes():
    """Changing --restart-mult on resume must NOT retroactively alter the cycle already
    in progress -- only the NEXT restart should grow by the new mult."""
    opt = _make_optimizer()
    # Simulate: original run used T_mult=1 (T_i stayed 50 forever); resuming now with
    # T_mult=2 for the first time, mid-cycle.
    make_scheduler = _make_scheduler_factory(opt, T_0=50, T_mult=2)
    scheduler = make_scheduler(-1)
    ckpt = {"scheduler_last_epoch": 2499, "scheduler_T_cur": 10, "scheduler_T_i": 50}

    scheduler = pt.restore_scheduler_on_resume(scheduler, opt, ckpt, True, make_scheduler)
    assert scheduler.T_i == 50            # current cycle length preserved, not retroactively 100

    for _ in range(40):                   # reach the boundary (50 - 10)
        scheduler.step()
    assert scheduler.T_cur == 0
    assert scheduler.T_i == 100           # NEW mult applies starting from this restart


# ------------------------------------------------------------- open-ended, old-format

def test_old_format_checkpoint_falls_back_to_exact_closed_form_for_mult1_history():
    """A checkpoint saved before this fix (no scheduler_T_cur/T_i) must fall back to
    torch's own correct closed-form re-derivation. For a real history that was always
    T_mult=1, this reduces to (and must exactly equal) epoch % T_0."""
    opt = _make_optimizer()
    make_scheduler = _make_scheduler_factory(opt, T_0=50, T_mult=1)
    scheduler = make_scheduler(-1)
    ckpt = {"scheduler_last_epoch": 2499}   # old format: no T_cur/T_i keys at all

    scheduler = pt.restore_scheduler_on_resume(scheduler, opt, ckpt, True, make_scheduler)

    assert scheduler.T_cur == 2499 % 50
    assert scheduler.T_i == 50


def test_old_format_checkpoint_with_none_values_also_falls_back():
    """A checkpoint saved by the FIXED code but for a run that never had a real
    scheduler_T_cur (shouldn't happen for open_ended, but guards the None case
    defensively) must not crash and must still fall back correctly."""
    opt = _make_optimizer()
    make_scheduler = _make_scheduler_factory(opt, T_0=50, T_mult=1)
    scheduler = make_scheduler(-1)
    ckpt = {"scheduler_last_epoch": 2499, "scheduler_T_cur": None, "scheduler_T_i": None}

    scheduler = pt.restore_scheduler_on_resume(scheduler, opt, ckpt, True, make_scheduler)

    assert scheduler.T_cur == 2499 % 50


# -------------------------------------------------------- fixed-epoch (CosineAnnealingLR)

def test_fixed_epoch_scheduler_resume_is_byte_identical_to_pre_fix_behavior():
    """CosineAnnealingLR (fixed --epochs > 0, the more common training mode across this
    fleet) must NOT be touched by this fix at all -- confirmed torch's own -1 + explicit
    .step(epoch) reconstruction is NOT numerically identical to direct last_epoch
    construction for this scheduler (~0.06% LR difference, torch's own closed-form vs
    chainable-form distinction)."""
    opt_direct = _make_optimizer()
    direct = torch.optim.lr_scheduler.CosineAnnealingLR(opt_direct, 450, last_epoch=200)

    opt_via_fix = _make_optimizer()
    make_scheduler = _make_scheduler_factory(opt_via_fix, open_ended=False)
    scheduler = make_scheduler(-1)
    ckpt = {"scheduler_last_epoch": 200, "scheduler_T_cur": None, "scheduler_T_i": None}
    scheduler = pt.restore_scheduler_on_resume(scheduler, opt_via_fix, ckpt, False, make_scheduler)

    assert opt_via_fix.param_groups[0]["lr"] == pytest.approx(
        opt_direct.param_groups[0]["lr"], abs=1e-12)


def test_no_checkpoint_data_returns_scheduler_unchanged():
    opt = _make_optimizer()
    make_scheduler = _make_scheduler_factory(opt)
    scheduler = make_scheduler(-1)
    result = pt.restore_scheduler_on_resume(scheduler, opt, {}, True, make_scheduler)
    assert result is scheduler


# --------------------------------------------------------- grouped_random_split()
#
# Tests for the fix to a real bug found 2026-09-02: `--repeats 1` (the default) combined
# with the old flat `torch.utils.data.random_split` over the repeats-expanded index range
# could -- and, on the Joyo American Sound v3 run (675 combos, seed 42), DID -- send a
# combo's only rendered example entirely to val, so 33/675 combos trained on nothing at
# all while every other combo trained fine, with nothing reporting it. See param_train.py.

def _combo_of(indices, n_groups):
    """Maps dataset indices back to their real combo id, matching
    ParamDataset.__getitem__'s `real_idx = idx % len(self.samples)`."""
    return {i % n_groups for i in indices}


def test_grouped_split_partitions_exhaustively_and_disjointly():
    n_groups, repeats = 37, 5
    train_idx, val_idx = pt.grouped_random_split(n_groups * repeats, n_groups, 0.05, seed=1)
    assert sorted(train_idx + val_idx) == list(range(n_groups * repeats))
    assert set(train_idx).isdisjoint(val_idx)


def test_grouped_split_never_fully_excludes_a_combo_from_train():
    """The actual bug: at repeats=1 a flat random_split could zero out a combo's only
    training example. Grouped splitting must make that impossible at ANY repeats."""
    for repeats in (1, 2, 3, 8, 32):
        n_groups = 675
        train_idx, _ = pt.grouped_random_split(n_groups * repeats, n_groups, 0.05, seed=42)
        assert _combo_of(train_idx, n_groups) == set(range(n_groups)), f"repeats={repeats}"


def test_grouped_split_gives_every_combo_a_val_example_once_it_has_room():
    """count >= 2 is the minimum for a combo to have both sides represented (ceil() of
    any positive fraction is >= 1) -- verify that guarantee actually holds."""
    n_groups, repeats = 40, 2
    train_idx, val_idx = pt.grouped_random_split(n_groups * repeats, n_groups, 0.05, seed=7)
    assert _combo_of(train_idx, n_groups) == set(range(n_groups))
    assert _combo_of(val_idx, n_groups) == set(range(n_groups))


def test_grouped_split_repeats_1_degrades_val_to_fully_empty():
    """Documents the one case grouped splitting cannot rescue on its own: with exactly one
    example per combo, giving any of it to val would zero out that combo's training --
    kept in train instead, so val collapses to empty rather than starving anyone. This is
    why main() enforces a --repeats floor before constructing the dataset."""
    n_groups = 675
    train_idx, val_idx = pt.grouped_random_split(n_groups, n_groups, 0.05, seed=42)
    assert val_idx == []
    assert len(train_idx) == n_groups


def test_grouped_split_val_split_zero_disables_val():
    train_idx, val_idx = pt.grouped_random_split(40 * 8, 40, 0.0, seed=42)
    assert val_idx == []
    assert len(train_idx) == 40 * 8


def test_grouped_split_is_reproducible_for_a_fixed_seed():
    a = pt.grouped_random_split(40 * 8, 40, 0.05, seed=42)
    b = pt.grouped_random_split(40 * 8, 40, 0.05, seed=42)
    assert a == b


# ------------------------------------------------- --restart-max-period (cap_sgdr_cycle)
#
# Motivated by replaying every --restart-mult=2 run's metrics.csv (2026-09-15, 7 runs /
# 29 cycles): doubling keeps paying off much longer than expected -- the second half of
# each cycle delivers a near-constant ~1.2x ESR gain whether that half is 50 or 674
# epochs -- but the run that reached cycles of 2400 and 4800 epochs wasted the last 30%
# of its final cycle (last new best at 70% of it). The cap stops growth at the largest
# cycle length still observed to have a live tail. See --restart-max-period's help text.


def _drive(scheduler, n_epochs, max_period):
    """Run the real training-loop sequence: bare step(), then cap at each cycle end."""
    lengths, lrs = [], []
    for _ in range(n_epochs):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
        if scheduler.T_cur == 0:                      # cycle boundary, as main() detects it
            pt.cap_sgdr_cycle(scheduler, max_period)
            lengths.append(scheduler.T_i)
    return lengths, lrs


def test_cap_stops_geometric_growth_and_holds_the_ceiling():
    """50, 100, 200, 400, 800, then 1200 forever -- not 1600, 3200, ..."""
    opt = _make_optimizer()
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    lengths, _ = _drive(sched, 5000, max_period=1200)
    assert lengths[:7] == [100, 200, 400, 800, 1200, 1200, 1200]
    assert all(v == 1200 for v in lengths[4:]), "cap must hold, not decay or re-grow"


def test_uncapped_behavior_is_unchanged_when_opted_out():
    """--restart-max-period 0 is the opt-out and must reproduce pre-cap lengths exactly."""
    opt = _make_optimizer()
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    lengths, _ = _drive(sched, 5000, max_period=0)
    assert lengths[:6] == [100, 200, 400, 800, 1600, 3200]


def test_capped_run_never_produces_a_nonpositive_lr():
    """The failure mode the max(T_cur + 1, ...) floor exists to prevent: a T_i clamped
    below T_cur pushes get_lr()'s cosine argument past pi and drives LR negative."""
    opt = _make_optimizer()
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    _, lrs = _drive(sched, 5000, max_period=1200)
    assert min(lrs) > 0.0
    assert max(lrs) <= 3e-4 + 1e-12


def test_cap_is_a_noop_below_the_ceiling_and_at_mult_1():
    opt = _make_optimizer()
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1)
    lengths, _ = _drive(sched, 600, max_period=1200)
    assert set(lengths) == {50}, "mult=1 never grows, so the cap can never engage"


def test_cap_survives_a_checkpoint_roundtrip():
    """T_i is already persisted (scheduler_T_i) and restored verbatim, so a capped cycle
    length needs no separate persistence mechanism -- this pins that."""
    opt = _make_optimizer()
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    _drive(sched, 1600, max_period=1200)
    assert sched.T_i == 1200
    ckpt = {"scheduler_last_epoch": sched.last_epoch,
            "scheduler_T_cur": sched.T_cur, "scheduler_T_i": sched.T_i}

    opt2 = _make_optimizer()
    fresh = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=50, T_mult=2)
    restored = pt.restore_scheduler_on_resume(
        fresh, opt2, ckpt, True, _make_scheduler_factory(opt2), max_period=1200)
    assert restored.T_i == 1200
    assert restored.T_cur == sched.T_cur


def test_turning_the_cap_on_mid_cycle_defers_to_the_next_boundary():
    """Matches what restore_scheduler_on_resume already does for a changed --restart-mult:
    the in-flight cycle keeps its length; the cap applies once it completes. Clamping into
    a live cycle instead would be the negative-LR bug above."""
    opt = _make_optimizer()
    # Mid-way through an uncapped 1600-epoch cycle, as a pre-cap run would have saved it.
    ckpt = {"scheduler_last_epoch": 2350, "scheduler_T_cur": 800, "scheduler_T_i": 1600}
    fresh = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    restored = pt.restore_scheduler_on_resume(
        fresh, opt, ckpt, True, _make_scheduler_factory(opt), max_period=1200)
    assert restored.T_i == 1200, "cap applies, but only down to a length that still holds T_cur"
    assert restored.T_cur == 800
    assert restored.get_last_lr()[0] > 0.0

    # And a cap BELOW the live T_cur must not invert the cosine.
    opt3 = _make_optimizer()
    fresh3 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt3, T_0=50, T_mult=2)
    tight = pt.restore_scheduler_on_resume(
        fresh3, opt3, dict(ckpt), True, _make_scheduler_factory(opt3), max_period=400)
    assert tight.T_i == 801, "floored at T_cur + 1, never below"
    assert tight.get_last_lr()[0] > 0.0


def test_old_format_checkpoint_gets_the_cap_reapplied():
    """The old-format path uses torch's step(epoch) branch, the one that recomputes T_i as
    T_0 * T_mult**n with no knowledge of the cap -- so the cap must be re-applied after it,
    along with a get_lr() refresh (step(epoch) already pushed an uncapped-T_i LR)."""
    # last_epoch=1700 re-derives to T_cur=150, T_i=1600 -- an over-ceiling cycle whose
    # position is still low enough that the cap can engage immediately.
    opt = _make_optimizer()
    ckpt = {"scheduler_last_epoch": 1700}             # no scheduler_T_cur/T_i => old format
    fresh = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    restored = pt.restore_scheduler_on_resume(
        fresh, opt, ckpt, True, _make_scheduler_factory(opt), max_period=1200)
    assert restored.T_i == 1200
    assert restored.get_last_lr()[0] > 0.0
    assert opt.param_groups[0]["lr"] == restored.get_last_lr()[0], "param groups refreshed"

    # last_epoch=3000 re-derives to T_cur=1450, T_i=1600 -- already PAST the ceiling, so
    # the floor defers the cap to the next boundary rather than inverting the cosine.
    opt2 = _make_optimizer()
    fresh2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=50, T_mult=2)
    late = pt.restore_scheduler_on_resume(
        fresh2, opt2, {"scheduler_last_epoch": 3000}, True,
        _make_scheduler_factory(opt2), max_period=1200)
    assert late.T_cur == 1450 and late.T_i == 1451, "floored at T_cur + 1"
    assert late.get_last_lr()[0] > 0.0
    late.step()                                        # completes the deferred cycle
    assert late.T_cur == 0
    pt.cap_sgdr_cycle(late, 1200)                      # what the training loop then does
    assert late.T_i == 1200, "cap engages at the next boundary"


def test_cap_bounds_stale_cycles_patience_in_epoch_terms():
    """The side benefit --restart-max-period's help text claims: uncapped mult=2 makes
    --stale-cycles' patience grow without bound in epochs; capped, it is a fixed budget."""
    opt = _make_optimizer()
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    lengths, _ = _drive(sched, 20000, max_period=1200)
    worst_case_3_cycles = sum(sorted(lengths)[-3:])
    assert worst_case_3_cycles == 3600


# ---------------------------------------- --restart-max-period default resolution
#
# The cap is ON by default (DEFAULT_RESTART_MAX_PERIOD); 0 opts out. Two ways a default
# could misbehave that these pin down: it must not nag on the default mult=1 run, and it
# must not silently override a --restart-period the user chose deliberately.


def test_default_is_on_at_the_documented_ceiling():
    value, notice, error = pt.resolve_restart_max_period(None, restart_period=50, restart_mult=2)
    assert value == pt.DEFAULT_RESTART_MAX_PERIOD == 1200
    assert error is None and notice is None


def test_zero_opts_out_silently():
    value, notice, error = pt.resolve_restart_max_period(0, restart_period=50, restart_mult=2)
    assert value == 0
    assert error is None and notice is None, "the documented opt-out must be quiet"


def test_default_does_not_nag_on_an_explicit_mult1_run():
    """--restart-mult 1 is an explicit opt-out (default is 2 as of 2026-09-20), but a default
    cap that warned 'no-op at mult 1' on every such run would still be unwanted noise --
    only an EXPLICIT --restart-max-period may say that."""
    _, notice, error = pt.resolve_restart_max_period(None, restart_period=50, restart_mult=1)
    assert notice is None and error is None

    _, explicit_notice, _ = pt.resolve_restart_max_period(1200, restart_period=50, restart_mult=1)
    assert explicit_notice is not None and "no-op" in explicit_notice


def test_default_defers_to_a_larger_user_chosen_restart_period():
    """A default ceiling at or below the user's own --restart-period would pin every cycle
    to that period and silently turn their --restart-mult 2 into a no-op. The default must
    disable itself and say so rather than override a value they typed."""
    value, notice, error = pt.resolve_restart_max_period(None, restart_period=2000, restart_mult=2)
    assert value == 0, "default cap disables itself rather than neutering --restart-mult"
    assert error is None
    assert notice is not None and "UNCAPPED" in notice

    # Exactly at the ceiling is the same situation (cap == period => no growth at all).
    value, notice, _ = pt.resolve_restart_max_period(None, restart_period=1200, restart_mult=2)
    assert value == 0 and notice is not None


def test_explicit_cap_below_restart_period_is_an_error():
    """Explicit is different from default: if the user types a cap below their period they
    have asked for something incoherent, so say so instead of quietly disabling it."""
    value, _, error = pt.resolve_restart_max_period(20, restart_period=50, restart_mult=2)
    assert value is None
    assert error is not None and "below --restart-period" in error


def test_explicit_cap_equal_to_restart_period_is_allowed():
    """cap == period pins cycles at the period. Incoherent as a silent default, but a
    legitimate explicit request ('grow no further than where I started')."""
    value, _, error = pt.resolve_restart_max_period(50, restart_period=50, restart_mult=2)
    assert value == 50 and error is None


# ---------------------------------------------------------------------------
# --freeze-tiers (train_epoch's per-tier frozen-loss guard)
#
# Motivated by a real case: scan_film_runaway.py's full-grid scan found a FiLM/
# LeakyReLU runaway in the tweed-5f6-a-full-sag-ac w4 (lite) tier only -- w8
# (full) was clean on the same grid/reference. A targeted fine-tune needs to
# update lite without disturbing full, which requires excluding full's params
# from the optimizer entirely. train_epoch's per-tier loop always called
# .backward() on every tier's loss; a fully-frozen tier's loss has no grad_fn
# (every input a non-trainable leaf), so that raises "does not require grad"
# unless the loop skips backward for tiers with nothing trainable.
# ---------------------------------------------------------------------------

def _tiny_slimmable_batch(n=4, t=480):
    inp = torch.randn(n, 1, t)
    out = torch.randn(n, 1, t)
    params = torch.rand(n, 1)
    return inp, out, params


def test_freeze_tier_does_not_crash_and_leaves_its_weights_untouched():
    torch.manual_seed(0)
    model = pt.SlimmableParametricA2(num_params=1, widths=[2, 3])
    for p in model.lite.parameters():
        p.requires_grad_(False)

    before_lite = [p.clone() for p in model.lite.parameters()]
    before_full = [p.clone() for p in model.full.parameters()]

    inp, out, params = _tiny_slimmable_batch()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inp, out, params), batch_size=2)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)

    pt.train_epoch(model, loader, optimizer, torch.nn.MSELoss(), device="cpu",
                   epoch=1, total_epochs=1, log_interval=0)

    for before, after in zip(before_lite, model.lite.parameters()):
        assert torch.equal(before, after), "frozen tier's weights changed"
    assert any(not torch.equal(before, after)
              for before, after in zip(before_full, model.full.parameters())), \
        "trainable tier never updated -- test isn't exercising anything"


def test_freeze_tier_joint_clip_does_not_crash_on_frozen_tier_with_no_grad():
    """clip_grad_norm_ over ALL model.parameters() (the default, non-per-tier-clip
    path) must tolerate a frozen tier whose .grad stays None the whole epoch."""
    torch.manual_seed(1)
    model = pt.SlimmableParametricA2(num_params=1, widths=[2, 3])
    for p in model.full.parameters():
        p.requires_grad_(False)

    inp, out, params = _tiny_slimmable_batch()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inp, out, params), batch_size=2)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)

    # per_tier_clip=False is the default -- joint clip_grad_norm_ over every
    # param, frozen ones included.
    pt.train_epoch(model, loader, optimizer, torch.nn.MSELoss(), device="cpu",
                   epoch=1, total_epochs=1, log_interval=0, per_tier_clip=False)


# ------------------------------------------- plateau-stop defaults (resolve_stale_rules)
#
# The 2026-09-16 default flip: --stale-epochs (was 0) becomes the plateau rule at
# max(1500, 1.25 * cap), and --stale-cycles (was 3) goes to 0, whenever the cycle cap is
# active. Simulating both rules against 41 distinct real runs showed --stale-cycles 3
# would have fired early on 14 of them (worst: stopping at epoch 3198 a run that kept
# improving to 9321, a 2.615x better model), while the epoch rule fired early on none.
#
# 2026-09-20: --stale-epochs lowered from 1500 to a FLAT 750, decoupled from max_period
# on request. The 41-run drought distribution (longest ever followed by improvement 1164
# epochs, next worst 664, everything else <=303) means 750 forfeits only the one 1164
# outlier and clears every other run's real drought with margin -- but it also drops the
# patience > max_period trough-coverage guarantee the 1.25x coupling used to provide (see
# DEFAULT_STALE_EPOCHS's own comment and resolve_stale_rules' docstring for the tradeoff).


def test_capped_run_uses_the_epoch_rule_and_disables_the_cycle_rule():
    sc, se, _ = pt.resolve_stale_rules(None, None, max_period=1200)
    assert (sc, se) == (0, 750)


def test_uncapped_run_keeps_the_historical_cycle_rule():
    """With no cap, cycles grow without bound, so a fixed epoch counter has no guarantee of
    spanning an LR trough -- the cosine-tail artifact the cycle rule exists to avoid."""
    sc, se, notice = pt.resolve_stale_rules(None, None, max_period=0)
    assert (sc, se) == (3, 0)
    assert notice is not None and "uncapped" in notice


def test_patience_is_flat_regardless_of_cap():
    """Since 2026-09-20, patience is DEFAULT_STALE_EPOCHS (750) flat -- NOT scaled with
    max_period. This is a deliberate regression from the prior 1.25x-coupled behavior
    (patience used to stay > cap to guarantee trough coverage); a raised cap no longer
    raises patience to match, so a very long cycle can now be stopped mid-cycle."""
    _, se, _ = pt.resolve_stale_rules(None, None, max_period=4000)
    assert se == pt.DEFAULT_STALE_EPOCHS == 750
    assert se < 4000


def test_patience_stays_flat_on_a_lowered_cap_too():
    """No floor coupling left to test against a lowered cap -- 750 is 750 regardless."""
    _, se, _ = pt.resolve_stale_rules(None, None, max_period=400)
    assert se == pt.DEFAULT_STALE_EPOCHS == 750


def test_explicit_values_are_honored_including_zero():
    assert pt.resolve_stale_rules(5, 900, max_period=1200)[:2] == (5, 900)
    assert pt.resolve_stale_rules(0, 900, max_period=1200)[:2] == (0, 900)
    # Explicitly re-enabling the cycle rule alongside the epoch rule is allowed; the loop
    # ORs them, so the cycle rule would fire first -- the caller's business, not ours.
    assert pt.resolve_stale_rules(3, None, max_period=1200)[:2] == (3, 750)


def test_warns_when_both_rules_end_up_disabled():
    sc, se, notice = pt.resolve_stale_rules(0, 0, max_period=1200)
    assert (sc, se) == (0, 0)
    assert notice is not None and "STOP file" in notice


# ---------------------------------------------------------------------------
# SlimmableParametricA2.enable_spectral_norm(skip_tiers=...)
#
# Real bug found by a peer session (2026-09-17), independently confirmed here: the
# reparametrization spectral_norm applies CLIPS a layer's weight the moment it is wrapped,
# using whatever values are already loaded -- that happens regardless of requires_grad, so
# --freeze-tiers (which only sets requires_grad_(False)) did NOT protect a "frozen" tier from
# this: it still got wrapped and clipped, just never trained further after that. Measured on
# a real --init-from + --freeze-tiers full + --spectral-norm run: the "frozen" full tier's
# peak output moved -9% to +30% across knob corners from the wrap alone, before a single
# training step. skip_tiers makes a named tier's weights exactly what was loaded, period.
# ---------------------------------------------------------------------------

def test_enable_spectral_norm_skip_tiers_leaves_named_tier_completely_unwrapped():
    torch.manual_seed(0)
    model = pt.SlimmableParametricA2(num_params=2, widths=[4, 8])
    before_full = [p.clone() for p in model.full.parameters()]

    model.enable_spectral_norm(skip_tiers=["full"])

    after_full = list(model.full.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before_full, after_full)), \
        "skip_tiers must leave the named tier's weights byte-identical"
    assert not hasattr(model.full.layers[0].conv, "parametrizations"), \
        "skip_tiers must not even WRAP the named tier -- not just avoid changing its values"


def test_enable_spectral_norm_skip_tiers_still_wraps_and_clips_the_others():
    torch.manual_seed(0)
    model = pt.SlimmableParametricA2(num_params=2, widths=[4, 8])
    before_lite = [p.clone() for n, p in model.lite.named_parameters()]

    model.enable_spectral_norm(skip_tiers=["full"])

    assert hasattr(model.lite.layers[0].conv, "parametrizations")
    after_lite = [p for n, p in model.lite.named_parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before_lite, after_lite)), \
        "the non-skipped tier must still actually be wrapped/clipped"


def test_enable_spectral_norm_no_skip_tiers_wraps_every_tier_as_before():
    """Backward compatibility: omitting skip_tiers must reproduce the pre-fix behavior
    exactly -- every tier wrapped, none exempted."""
    torch.manual_seed(0)
    model = pt.SlimmableParametricA2(num_params=2, widths=[4, 8])
    model.enable_spectral_norm()
    assert hasattr(model.lite.layers[0].conv, "parametrizations")
    assert hasattr(model.full.layers[0].conv, "parametrizations")
