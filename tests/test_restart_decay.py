"""restart_decay_factor scales the SGDR ceiling-decay by the completed cycle's length.

Before this, --restart-decay applied flat at every restart regardless of cycle length, so
under --restart-mult 2 the decay EVENT stayed the same size (0.97) while firing less and
less often as cycles doubled -- 5 restarts by epoch 1550 (0.97^5 = 0.86x the ceiling) versus
the 31 restarts a flat --restart-period 50 schedule would have had by then (0.97^31 = 0.17x).
OR120's own log showed ESR at lr~1e-4 already within 6% of its eventual best at lr~1e-6, with
lr below ~1e-6 buying nothing further -- so a ceiling decaying this slowly spends much of a
long cycle re-descending through a range that wasn't earning its keep.

restart_decay_factor treats --restart-decay as a decay rate PER --restart-period-worth of
epochs, so the per-epoch decay rate stays constant regardless of how --restart-mult grows
cycle lengths, matching the historical flat-50 behaviour exactly at --restart-mult 1.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from param_train import restart_decay_factor, cap_sgdr_cycle
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class TestRestartDecayFactor:
    """Pure math, matching OR120's actual cycle sequence (50, 100, 200, 400, 800, 1200-capped)."""

    def test_one_restart_period_is_flat_decay(self):
        # completed cycle length == restart_period -- historical flat-per-restart behaviour,
        # exact (this is --restart-mult 1's case at every restart).
        assert restart_decay_factor(50, 50, 0.97) == pytest.approx(0.97)

    def test_doubled_cycle_squares_the_factor(self):
        assert restart_decay_factor(100, 50, 0.97) == pytest.approx(0.97 ** 2)

    def test_matches_or120s_measured_cycle_sequence(self):
        expected = {50: 0.9700, 100: 0.9409, 200: 0.8853, 400: 0.7837,
                    800: 0.6143, 1200: 0.4814}
        for cycle_len, factor in expected.items():
            assert restart_decay_factor(cycle_len, 50, 0.97) == pytest.approx(factor, abs=1e-4)

    def test_longer_cycle_decays_more_than_shorter_cycle(self):
        assert restart_decay_factor(1200, 50, 0.97) < restart_decay_factor(50, 50, 0.97)

    def test_no_op_decay_rate_is_unaffected_by_cycle_length(self):
        assert restart_decay_factor(1200, 50, 1.0) == 1.0


class TestIntegrationWithRealScheduler:
    """Drives an actual CosineAnnealingWarmRestarts through several restarts, using the same
    T_cur==0 cycle-boundary detection and post-step() T_i-already-grown division the real
    training loop uses -- catches ordering/off-by-one bugs a pure math test of the formula
    alone cannot, without waiting on real training to reach a restart.
    """

    def _run(self, restart_period, restart_mult, restart_decay, n_restarts, max_period=0):
        import torch
        model = torch.nn.Linear(2, 2)
        optimizer = SGD([{"params": model.parameters(), "lr": 0.0003}])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=restart_period,
                                                 T_mult=restart_mult)
        observed = []
        epoch = 0
        while len(observed) < n_restarts:
            epoch += 1
            scheduler.step()
            cycle_ended = getattr(scheduler, "T_cur", None) == 0
            if cycle_ended and restart_decay != 1.0:
                completed_len = scheduler.T_i / max(1, restart_mult)
                factor = restart_decay_factor(completed_len, restart_period, restart_decay)
                for group in optimizer.param_groups:
                    group["initial_lr"] *= factor
                scheduler.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
                observed.append((completed_len, factor, optimizer.param_groups[0]["initial_lr"]))
            if cycle_ended and max_period:
                cap_sgdr_cycle(scheduler, max_period)
        return observed

    def test_flat_cycles_match_historical_flat_decay(self):
        # restart_mult=1: every cycle is restart_period long, so every restart applies
        # exactly restart_decay -- the pre-existing behaviour, unchanged.
        observed = self._run(restart_period=5, restart_mult=1, restart_decay=0.97,
                              n_restarts=4)
        ceiling = 0.0003
        for completed_len, factor, new_ceiling in observed:
            assert completed_len == pytest.approx(5)
            assert factor == pytest.approx(0.97)
            ceiling *= 0.97
            assert new_ceiling == pytest.approx(ceiling)

    def test_growing_cycles_decay_faster_per_restart(self):
        # restart_mult=2: cycle lengths 5, 10, 20, 40 -- each restart's factor should shrink
        # (longer cycle -> more decay), and match restart_decay_factor exactly.
        observed = self._run(restart_period=5, restart_mult=2, restart_decay=0.97,
                              n_restarts=4)
        lengths = [c for c, _, _ in observed]
        factors = [f for _, f, _ in observed]
        assert lengths == pytest.approx([5, 10, 20, 40])
        assert factors == pytest.approx(
            [restart_decay_factor(l, 5, 0.97) for l in lengths])
        # Strictly decreasing: each successive (longer) cycle decays the ceiling harder.
        assert factors == sorted(factors, reverse=True)

    def test_cumulative_decay_matches_per_epoch_rate_regardless_of_chunking(self):
        # The whole point: total epochs elapsed should determine cumulative decay, not how
        # many restarts happened to occur along the way.
        observed = self._run(restart_period=5, restart_mult=2, restart_decay=0.97,
                              n_restarts=4)
        total_epochs = sum(c for c, _, _ in observed)
        cumulative = 1.0
        for _, factor, _ in observed:
            cumulative *= factor
        expected = 0.97 ** (total_epochs / 5)
        assert cumulative == pytest.approx(expected)

    def test_respects_restart_max_period_cap(self):
        # Cap at 20: cycle lengths become 5, 10, 20, 20 (mult=2 growth stops at the cap) --
        # completed_len must reflect the CAPPED length, not the uncapped 5,10,20,40.
        observed = self._run(restart_period=5, restart_mult=2, restart_decay=0.97,
                              n_restarts=4, max_period=20)
        lengths = [c for c, _, _ in observed]
        assert lengths == pytest.approx([5, 10, 20, 20])
