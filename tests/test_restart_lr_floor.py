"""--restart-lr-floor: end an SGDR cycle early once its LR would fall below the floor."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from param_train import (DEFAULT_RESTART_LR_FLOOR, RESTART_FLOOR_MIN_RATIO, cap_sgdr_cycle,  # noqa: E402
                         early_restart_at_floor, restart_decay_factor)


def make(lr=3e-4, t0=50, mult=2):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=lr)
    return opt, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t0, T_mult=mult)


def run(epochs, floor, lr=3e-4, t0=50, mult=2, decay=1.0, cap=0):
    """Mimic main()'s order: step -> early restart -> (restart) decay -> cap. Returns the lr each
    epoch TRAINS at, plus the completed length of every cycle."""
    opt, sch = make(lr, t0, mult)
    lrs, lens = [opt.param_groups[0]["lr"]], []
    for _ in range(epochs - 1):
        sch.step()
        early = early_restart_at_floor(sch, opt, floor)
        if sch.T_cur == 0:                                    # cycle_ended
            lens.append(early[0] if early else sch.T_i / mult)
            if decay != 1.0:
                f = restart_decay_factor(lens[-1], t0, decay)
                for g in opt.param_groups:
                    g["initial_lr"] *= f
                sch.base_lrs = [g["initial_lr"] for g in opt.param_groups]
            if cap:
                cap_sgdr_cycle(sch, cap)
        lrs.append(opt.param_groups[0]["lr"])
    return lrs, lens


def test_default_is_one_micro():
    assert DEFAULT_RESTART_LR_FLOOR == 1e-6


def test_floor_zero_changes_nothing():
    a, la = run(400, 0.0)
    b, lb = run(400, None)
    opt, sch = make()
    plain = [opt.param_groups[0]["lr"]]
    for _ in range(399):
        sch.step(); plain.append(opt.param_groups[0]["lr"])
    assert a == plain and b == plain


def test_no_epoch_trains_below_the_floor():
    lrs, _ = run(3000, 1e-6, lr=3e-4)
    assert min(lrs) >= 1e-6


def test_cycle_is_cut_exactly_where_cosine_crosses_the_floor():
    T, eta, floor = 800, 2.4e-4, 1e-6
    # first cycle length that actually runs 800 under mult 2: 50,100,200,400,800
    expect = next(t for t in range(1, T) if eta * (1 + math.cos(math.pi * t / T)) / 2 < floor)
    opt, sch = make(lr=eta, t0=T, mult=1)
    ran = 1
    while True:
        sch.step()
        r = early_restart_at_floor(sch, opt, floor)
        if r:
            break
        ran += 1
    assert r[0] == expect and r[1] == T and sch.T_cur == 0
    assert ran == expect
    # the cut saves the ~4-6% the data predicted for a long cycle
    assert 0.03 < (T - expect) / T < 0.07


def test_restart_resets_lr_to_peak_and_grows_t_i_like_a_natural_restart():
    opt, sch = make(lr=3e-4, t0=100, mult=2)
    while not early_restart_at_floor(sch, opt, 1e-6):
        sch.step()
    assert sch.T_cur == 0 and sch.T_i == 200
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert sch._last_lr[0] == pytest.approx(3e-4)


def test_cap_still_applies_after_an_early_restart():
    _, lens = run(7000, 1e-6, lr=3e-4, cap=1200)
    # nominal lengths 50,100,200,400,800 then CAPPED at 1200 (not 1600/3200): the cycles that
    # reach the cap run ~1200 minus the sub-floor tail (3.7% at this peak), and stay there
    assert len(lens) >= 7
    assert all(1100 < l <= 1200 for l in lens[5:])
    assert lens[4] < 800 and lens[4] > 740

def test_decay_uses_the_actual_shorter_cycle():
    _, lens = run(2000, 1e-6, lr=3e-4, decay=0.97)
    nominal = [50, 100, 200, 400, 800]
    assert all(l < n for l, n in zip(lens, nominal))          # every cycle was cut
    assert restart_decay_factor(lens[3], 50, 0.97) > restart_decay_factor(400, 50, 0.97)


def test_guard_skips_when_peak_is_within_ten_times_the_floor():
    floor = 1e-6
    lrs, lens = run(600, floor, lr=RESTART_FLOOR_MIN_RATIO * floor * 0.9, t0=200, mult=1)
    assert lens[0] == 200                                     # ran the full cycle
    assert min(lrs) < floor                                   # and went below it, by design


def test_no_restart_storm_with_a_heavily_decayed_ceiling():
    lrs, lens = run(20000, 1e-6, lr=3e-4, decay=0.97, cap=1200)
    assert min(lens) >= 10                                    # never degenerates to 1-2 epoch cycles


def test_returns_none_when_nothing_to_do():
    opt, sch = make()
    assert early_restart_at_floor(sch, opt, 1e-6) is None     # T_cur == 0
    sch.step()
    assert early_restart_at_floor(sch, opt, 1e-6) is None     # lr still high
    assert early_restart_at_floor(sch, opt, 0.0) is None      # disabled
