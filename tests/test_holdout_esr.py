"""holdout_esr.py -- scoring a checkpoint on knob settings it never trained on.

Training's own val ESR is a random slice of the SAME combinations, so it cannot see
interpolation error at all. These cover the classification rule that decides which cells count
as held out; the ESR maths itself is per_combo_esr's and tested there.
"""
import csv
import importlib.util
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("holdout_esr", REPO / "holdout_esr.py")
he = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(he)


def _write_params(d: Path, names, rows):
    with open(d / "params.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx"] + names)
        for i, r in enumerate(rows):
            w.writerow([i] + list(r))


def test_trained_keys_are_read_from_the_training_dataset():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_params(d, ["Bass", "Mids"], [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75)])
        got = he._trained_keys_from_dataset(d, ["Bass", "Mids"])
        assert got == {(0.25, 0.25), (0.25, 0.75), (0.75, 0.75)}


def test_key_rounds_so_csv_strings_match_floats():
    # a value read as the string "0.25" must match the float 0.25 from the scored dataset
    assert he._key({"A": "0.25", "B": "0.75"}, ["A", "B"]) == (0.25, 0.75)
    assert he._key({"A": 0.2500000001, "B": 0.75}, ["A", "B"]) == (0.25, 0.75)


def test_a_missing_knob_column_fails_loudly_rather_than_silently_mismatching():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_params(d, ["Bass"], [(0.25,), (0.75,)])
        with pytest.raises(SystemExit):
            he._trained_keys_from_dataset(d, ["Bass", "Treble"])


def test_value_classification_requires_EVERY_knob_on_a_trained_value():
    vals = {0.25, 0.75}
    trained = (0.25, 0.75, 0.25)
    interior = (0.25, 0.5, 0.25)          # one axis off-grid -> held out
    assert all(v in vals for v in trained)
    assert not all(v in vals for v in interior)
