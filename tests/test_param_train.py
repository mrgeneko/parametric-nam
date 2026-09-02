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
