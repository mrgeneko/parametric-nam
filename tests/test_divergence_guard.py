"""Tests for the training divergence guard (--skip-nonfinite-grad, --rollback-factor).

Claims: a non-finite-gradient step is dropped without touching the weights and the epoch loss
stays finite; DivergenceGuard flags a tier whose val ESR is non-finite or beyond factor x its
best, snapshots only while healthy, restores model and optimizer exactly, cuts the lr each
time, and stops after max_rollbacks; disabled (factor 0) it does nothing.
"""
import math

import torch
import torch.nn as nn

from param_train import DivergenceGuard, train_epoch


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x, p):
        return self.lin(x)


def _loader(poison_step=None, n=4):
    out = []
    for i in range(n):
        x = torch.randn(8, 4)
        if i == poison_step:
            x[0, 0] = float("inf")
        out.append((x, torch.randn(8, 1), torch.zeros(8, 1)))
    return out


def _crit(pred, target):
    return ((pred - target) ** 2).mean()


def test_nonfinite_step_is_dropped_and_weights_untouched():
    torch.manual_seed(0)
    m = Tiny(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    stats = {}
    before = {k: v.clone() for k, v in m.state_dict().items()}
    loss = train_epoch(m, _loader(poison_step=0, n=1), opt, _crit, "cpu", guard_stats=stats)
    assert stats["skipped"] == 1
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert not any(torch.isnan(p).any() for p in m.parameters())


def test_epoch_loss_stays_finite_when_a_step_is_skipped():
    torch.manual_seed(0)
    m = Tiny(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    stats = {}
    loss = train_epoch(m, _loader(poison_step=1, n=4), opt, _crit, "cpu", guard_stats=stats)
    assert stats["skipped"] == 1 and math.isfinite(loss)


def test_without_the_guard_a_nonfinite_step_poisons_the_weights():
    torch.manual_seed(0)
    m = Tiny(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    train_epoch(m, _loader(poison_step=0, n=1), opt, _crit, "cpu")
    assert any(not torch.isfinite(p).all() for p in m.parameters())


def test_finite_run_skips_nothing():
    torch.manual_seed(0)
    m = Tiny(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    stats = {}
    train_epoch(m, _loader(n=3), opt, _crit, "cpu", guard_stats=stats)
    assert not stats.get("skipped")


def _setup():
    torch.manual_seed(0)
    m = Tiny(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    m(torch.randn(2, 4), None).sum().backward(); opt.step()
    return m, opt


def test_diverged_flags_nonfinite_and_jumps_only():
    g = DivergenceGuard(5.0)
    best = {"lite": 0.03, "full": 0.01}
    assert g.diverged({"lite": 0.12, "full": 0.012}, best) == []
    assert g.diverged({"lite": 0.2, "full": 0.012}, best) == ["lite"]
    assert g.diverged({"lite": 0.03, "full": float("nan")}, best) == ["full"]
    assert g.diverged({"lite": float("inf"), "full": 0.01}, best) == ["lite"]


def test_disabled_guard_does_nothing():
    g = DivergenceGuard(0.0)
    m, opt = _setup()
    assert g.diverged({"lite": 1e9}, {"lite": 0.03}) == []
    assert not g.maybe_snapshot(1, m, opt, {"lite": 0.03}, {"lite": 0.03})
    assert not g.can_rollback()


def test_snapshot_only_while_healthy_and_every_n_epochs():
    g = DivergenceGuard(5.0, every=10)
    m, opt = _setup(); best = {"lite": 0.03}
    assert g.maybe_snapshot(1, m, opt, {"lite": 0.03}, best)
    assert not g.maybe_snapshot(5, m, opt, {"lite": 0.03}, best)         # too soon
    assert not g.maybe_snapshot(11, m, opt, {"lite": 0.08}, best)        # 2.7x best: not healthy
    assert g.maybe_snapshot(11, m, opt, {"lite": 0.04}, best)
    assert g.snap_epoch == 11


def test_rollback_restores_model_and_optimizer_and_cuts_lr():
    g = DivergenceGuard(5.0, lr_scale=0.5)
    m, opt = _setup(); best = {"lite": 0.03}
    g.maybe_snapshot(7, m, opt, {"lite": 0.03}, best)
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    steps = [s["step"].clone() for s in opt.state.values()]
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(1e6)
    m(torch.randn(2, 4), None).sum().backward(); opt.step()              # moves weights and Adam state
    assert g.can_rollback()
    assert g.rollback(m, opt) == 7
    assert all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    assert [s["step"] for s in opt.state.values()] == steps
    assert opt.param_groups[0]["lr"] == 1e-2 * 0.5
    g.snap["model"]["lin.weight"].zero_()                                # the snapshot is not aliased
    assert not torch.equal(m.lin.weight, torch.zeros_like(m.lin.weight))


def test_rollbacks_are_capped():
    g = DivergenceGuard(5.0, max_rollbacks=2)
    m, opt = _setup()
    g.maybe_snapshot(1, m, opt, {"lite": 0.03}, {"lite": 0.03})
    for _ in range(2):
        assert g.can_rollback(); g.rollback(m, opt)
    assert not g.can_rollback()


def test_no_rollback_without_a_snapshot():
    assert not DivergenceGuard(5.0).can_rollback()
