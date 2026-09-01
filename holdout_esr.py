#!/usr/bin/env python3
"""Did the model actually LEARN the knob space, or just memorise the settings it saw?

THE QUESTION, AND WHY THE USUAL NUMBER CANNOT ANSWER IT.
A parametric model is trained at a finite set of knob settings and expected to interpolate
everything between them. Training's own val ESR cannot tell you whether it does: the val split is
a random slice of the SAME combinations, so a model that fits its training settings perfectly and
interpolates badly scores well. per_combo_esr.py has the same blind spot -- it reports per-cell
ESR, but only over cells the model was trained on.

The only way to measure interpolation is to score cells the model has NEVER SEEN. That means
rendering a denser grid than you train on, training on a subset, and scoring the remainder.

RELATIONSHIP TO grid_adequacy.py. That tool answers "is this grid dense enough?" BEFORE training
and with no model in the loop -- it compares the circuit's true output at a cell midpoint against
interpolating its neighbours, so the residual is a property of the circuit and the sampling alone.
This tool answers the same question AFTER training, WITH the model: it measures what the trained
network actually does between its training points, which includes the model's own capacity limits,
not just the grid's. Use grid_adequacy to choose a grid; use this to check the choice was right.

WORKED RESULT (Joyo American Sound, 2026-08-30). Rendered the full 3^3 = 27 EQ grid, trained on
only the 8 corners of [0.25, 0.75]^3, scored the 19 interior cells:

    w9  TRAINED corners     mean 0.00817
        HELD-OUT interior   mean 0.02547     3.1x worse
        dead-centre                0.04002   4.9x worse

Two grid points per axis gave accurate end stops and a mushy middle -- and the middle is where
tone controls actually get set. The training val ESR for that run was 0.00299, which shows none
of this.

INTERPRETING THE GAP. A large ratio is not automatically audible. Decompose it before acting: on
that run, 54% of the dead-centre error was a static level/tone offset (+0.79 dB), the kind of
error a player dials out with a small knob nudge rather than hears as wrongness. ESR scores a
benign 0.8 dB tilt identically to 0.8 dB of wrong harmonics.

Usage:
    # derive the trained cells from the training dataset itself (preferred -- no guessing)
    python holdout_esr.py --checkpoint ckpt/best.pt --dataset full_27_cell_ds \
        --trained-dataset corners_8_cell_ds

    # or classify by value: a cell is "trained" when EVERY knob sits on one of these
    python holdout_esr.py --checkpoint ckpt/best.pt --dataset full_27_cell_ds \
        --trained-values 0.25,0.75

    # write the per-cell rows too
    python holdout_esr.py --checkpoint ckpt/best.pt --dataset ds --trained-values 0.25,0.75 \
        -o holdout.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from param_train import ParamDataset, SlimmableParametricA2, resolve_device  # noqa: E402
from per_combo_esr import compute_per_combo_esr  # noqa: E402

ROUND = 4          # knob values are compared rounded, so 0.25 from a CSV matches 0.25 from a float


def _key(row, names):
    return tuple(round(float(row[n]), ROUND) for n in names)


def _trained_keys_from_dataset(ds_dir: Path, names: list) -> set:
    """The exact set of combinations a training dataset contains. Preferred over --trained-values:
    it cannot disagree with what was actually trained, and it handles grids whose trained cells are
    not describable as a simple per-axis value set."""
    with open(ds_dir / "params.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = [n for n in names if rows and n not in rows[0]]
    if missing:
        raise SystemExit(f"--trained-dataset {ds_dir} has no column(s) {missing}; its knobs are "
                         f"{[c for c in rows[0] if c not in ('idx',)][:8]}")
    return {_key(r, names) for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="best.pt / latest.pt from the training run under test. --widths is read "
                         "from its own args_dict, so it cannot disagree with how it was trained.")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="the FULL (denser) dataset to score against -- must contain the held-out "
                         "cells, not just the trained ones")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--trained-dataset", type=Path,
                   help="the dataset actually trained on; its combinations are the trained set")
    g.add_argument("--trained-values",
                   help="comma-separated values, e.g. 0.25,0.75 -- a cell counts as trained when "
                        "EVERY knob sits on one of them")
    ap.add_argument("--tier", default=None,
                    help="score only this tier label (e.g. full, lite, w5); default: every tier")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write per-cell rows (knobs, esr, trained/held-out) to this CSV")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    widths = ck.get("args_dict", {}).get("widths")
    if isinstance(widths, str):
        widths = [int(x) for x in widths.split(",")]
    if not widths:
        raise SystemExit(f"{args.checkpoint} has no widths in args_dict -- cannot rebuild the model")

    ds = ParamDataset(str(args.dataset), crop_len=48000, mmap=True)
    names = ds.param_names
    dev = resolve_device(args.device)
    model = SlimmableParametricA2(num_params=len(names), widths=widths).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    if args.trained_dataset:
        trained = _trained_keys_from_dataset(args.trained_dataset, names)
        how = f"--trained-dataset {args.trained_dataset} ({len(trained)} combinations)"
    else:
        vals = {round(float(v), ROUND) for v in args.trained_values.split(",")}
        trained = None                      # classified per row below
        how = f"--trained-values {sorted(vals)}"

    print(f"checkpoint {args.checkpoint.name}  epoch {ck.get('epoch')}  widths {widths}")
    print(f"dataset    {args.dataset}  {len(ds.samples)} combinations  knobs {names}")
    print(f"trained by {how}\n")

    rows_out = []
    for sub, lbl in zip(model.submodels, model.tier_labels()):
        if args.tier and lbl != args.tier:
            continue
        res = compute_per_combo_esr(sub, ds.inp, ds.outputs, ds.samples, names, dev)
        tr, held = [], []
        for r in res:
            k = _key(r, names)
            is_tr = (k in trained) if trained is not None else all(v in vals for v in k)
            (tr if is_tr else held).append(r)
            rows_out.append({**{n: r[n] for n in names}, "tier": lbl, "esr": r["esr"],
                             "set": "trained" if is_tr else "held-out"})
        if not held:
            print(f"  [{lbl}] every cell counts as TRAINED -- nothing held out, so this says "
                  f"nothing about interpolation. Score against a DENSER dataset than you trained on.")
            continue
        for name, rs in (("TRAINED", tr), ("HELD-OUT", held)):
            if not rs:
                print(f"  [{lbl}] {name:<9} none"); continue
            e = sorted(x["esr"] for x in rs)
            print(f"  [{lbl}] {name:<9} n={len(e):<4d} mean={np.mean(e):.5f} "
                  f"median={np.median(e):.5f} best={e[0]:.5f} worst={e[-1]:.5f}")
        if tr and held:
            print(f"  [{lbl}] held-out / trained mean ratio: "
                  f"{np.mean([x['esr'] for x in held]) / np.mean([x['esr'] for x in tr]):.2f}x")
        # the cell farthest from any trained point is the worst case for interpolation
        if trained is not None or True:
            far = max(held, key=lambda r: min(
                sum((a - b) ** 2 for a, b in zip(_key(r, names), t)) for t in
                ({_key(x, names) for x in tr} or {tuple([0] * len(names))})))
            print(f"  [{lbl}] farthest held-out cell from any trained point: "
                  f"{ {n: far[n] for n in names} }  ESR={far['esr']:.5f}")
        print()

    if args.output and rows_out:
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {args.output} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
