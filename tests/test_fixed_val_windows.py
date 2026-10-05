"""FixedWindowVal: validation that scores the same audio every epoch."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from param_train import FixedWindowVal, ParamDataset, grouped_random_split, validate  # noqa: E402
from test_dataset import fake_dataset  # noqa: E402,F401  (fixture)

CROP = 4096


@pytest.fixture
def ds(fake_dataset):
    d, n_combos = fake_dataset
    return ParamDataset(str(d), crop_len=CROP, repeats=20, mmap=False), n_combos


def _val_indices(ds_, n_combos):
    _, val = grouped_random_split(len(ds_), n_combos, 0.05, seed=42)
    return val


def _grab(fv, k):
    fv.set_pass(k)
    return [fv[i][0].numpy().copy() for i in range(len(fv))]


def test_random_dataset_really_is_noisy_for_the_same_index(ds):
    """The premise: ParamDataset re-crops randomly on every call."""
    d, _ = ds
    a, b = d[0][0].numpy(), d[0][0].numpy()
    assert not np.array_equal(a, b)


def test_same_windows_every_epoch(ds):
    d, n = ds
    fv = FixedWindowVal(d, _val_indices(d, n), seed=7, passes=3)
    for k in range(3):
        a, b = _grab(fv, k), _grab(fv, k)          # two "epochs"
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_passes_are_different_windows(ds):
    d, n = ds
    fv = FixedWindowVal(d, _val_indices(d, n), seed=7, passes=3)
    p0, p1 = _grab(fv, 0), _grab(fv, 1)
    assert any(not np.array_equal(x, y) for x, y in zip(p0, p1))


def test_seed_changes_the_windows(ds):
    d, n = ds
    idx = _val_indices(d, n)
    a = _grab(FixedWindowVal(d, idx, seed=1, passes=1), 0)
    b = _grab(FixedWindowVal(d, idx, seed=2, passes=1), 0)
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


def test_every_knob_combo_is_in_validation(ds):
    d, n = ds
    fv = FixedWindowVal(d, _val_indices(d, n), seed=0, passes=1)
    assert {i % n for i in fv.indices} == set(range(n))


def test_target_stays_aligned_with_input(ds):
    """A pinned window must slice input and target at the SAME start."""
    d, n = ds
    fv = FixedWindowVal(d, _val_indices(d, n), seed=3, passes=1)
    for i in range(len(fv)):
        inp, out, params = fv[i]
        # fake_dataset targets are x * (0.2 + A), A = the first knob
        expect = inp[0].numpy() * (0.2 + float(params[0]))
        assert np.allclose(out[0].numpy(), expect, atol=1e-4)   # fixture sweep.wav is PCM-16; real misalignment ~1e-2


class _Echo(torch.nn.Module):
    def forward(self, inp, params):
        return inp * 0.9


class _Crit:
    def __call__(self, p, o, *a):
        return ((p - o) ** 2).mean()


def test_validate_is_reproducible_with_fixed_windows_and_not_without(ds):
    d, n = ds
    idx = _val_indices(d, n)
    m, c = _Echo(), _Crit()
    fixed = torch.utils.data.DataLoader(FixedWindowVal(d, idx, seed=5, passes=4),
                                        batch_size=64, shuffle=False)
    f1, _ = validate(m, fixed, c, "cpu", val_passes=4)
    f2, _ = validate(m, fixed, c, "cpu", val_passes=4)
    assert f1 == pytest.approx(f2, rel=0, abs=0)

    rnd = torch.utils.data.DataLoader(torch.utils.data.Subset(d, idx), batch_size=64, shuffle=False)
    r1, _ = validate(m, rnd, c, "cpu", val_passes=4)
    r2, _ = validate(m, rnd, c, "cpu", val_passes=4)
    assert r1 != r2                                # the noise this flag removes
