"""Dataset integrity, and what the train/val split ACTUALLY measures.

The second test class here is not a "does it work" test. It PINS A KNOWN LIMITATION so that
nobody reads a val-ESR number as something it is not. See docs/architecture.md.
"""
import csv
import json

import numpy as np
import pytest
import torch


@pytest.fixture
def fake_dataset(tmp_path):
    """A minimal but STRUCTURALLY REAL dataset: 6 permutations, 2 knobs."""
    import soundfile as sf

    sr, n = 48000, 48000
    d = tmp_path / "ds"
    d.mkdir()

    x = (np.random.default_rng(0).standard_normal(n) * 0.1).astype(np.float32)
    sf.write(d / "sweep.wav", x, sr)

    perms = [(a, b) for a in (0.0, 0.5, 1.0) for b in (0.0, 1.0)]
    outs = np.stack([x * (0.2 + a) for a, b in perms]).astype(np.float32)
    np.save(d / "outputs.npy", outs)

    with open(d / "params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "A", "B", "ok", "error"])
        for i, (a, b) in enumerate(perms):
            w.writerow([i, a, b, 1, ""])

    json.dump({"knobs": ["A", "B"], "backend": "test"}, open(d / "config.json", "w"))
    return d, len(perms)


class TestIntegrity:
    """Cheap invariants. A dataset that violates any of these will train silently and produce a
    model that is quietly wrong — which is strictly worse than a crash."""

    def test_rows_match_outputs(self, fake_dataset):
        d, n = fake_dataset
        rows = list(csv.DictReader(open(d / "params.csv")))
        outs = np.load(d / "outputs.npy", mmap_mode="r")
        assert len(rows) == outs.shape[0], "params.csv rows must match outputs.npy rows"

    def test_no_nonfinite(self, fake_dataset):
        d, _ = fake_dataset
        outs = np.load(d / "outputs.npy")
        assert np.isfinite(outs).all(), "a diverged SPICE permutation must not reach training"

    def test_no_dead_permutation(self, fake_dataset):
        d, _ = fake_dataset
        outs = np.load(d / "outputs.npy")
        rms = np.sqrt((outs.astype(np.float64) ** 2).mean(axis=1))
        assert (rms > 1e-9).all(), (
            "an all-silent permutation is a failed render, not a data point. Under a per-example "
            "ESR loss it is also a gradient hazard — see tests/test_loss.py."
        )

    def test_knobs_declared(self, fake_dataset):
        d, _ = fake_dataset
        cfg = json.load(open(d / "config.json"))
        rows = list(csv.DictReader(open(d / "params.csv")))
        for k in cfg["knobs"]:
            assert k in rows[0], f"knob {k!r} declared in config.json but absent from params.csv"


class TestValSplitIsNotInterpolation:
    """PINNED KNOWN LIMITATION — do not "fix" this test, fix the splitter.

    ParamDataset.__len__ is n_perms * repeats and __getitem__ does `real_idx = idx % n_perms`,
    so torch.utils.data.random_split shuffles CROPS, not knob settings. The same permutation
    therefore appears in BOTH train and val.

    Consequence: every val ESR we have ever published measures RECONSTRUCTION of knob settings the
    model trained on, evaluated on unseen time-crops. It does NOT measure INTERPOLATION to unseen
    knob settings.

    Mostly benign for a dense 126-point SPICE sweep — you are never far from a training point.
    NOT benign for a sparse hardware capture (~9 settings), where interpolation is the entire
    question and a model could memorise all nine, be garbage between them, and report a great ESR.
    """

    def test_the_same_permutation_lands_in_both_splits(self, fake_dataset):
        d, n_perms = fake_dataset
        from param_train import ParamDataset

        ds = ParamDataset(str(d), crop_len=4096, repeats=8, mmap=False)
        assert len(ds) == n_perms * 8, "items are (permutation x repeat), not permutations"

        n_val = max(1, int(len(ds) * 0.1))
        train, val = torch.utils.data.random_split(
            ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0)
        )
        train_perms = {i % n_perms for i in train.indices}
        val_perms = {i % n_perms for i in val.indices}

        assert val_perms & train_perms, (
            "If this ever passes cleanly, someone has made the split knob-disjoint — good! "
            "Update docs/architecture.md and delete this test."
        )
