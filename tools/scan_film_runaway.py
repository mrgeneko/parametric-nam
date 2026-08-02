#!/usr/bin/env python3
"""Scan a published .nam bundle for the FiLM/LeakyReLU runaway instability.

Background: see docs/film_runaway_investigation.md. Two published models (Tweed
5F6-A Full sag, and the old pre-fix Boss DS-1) were found to blow up 80-260x at a
narrow (knob-corner x real-transient) combination the training excitation
under-covered. This tool reproduces that check generically, against ANY
published .nam -- no training checkpoint or dataset required, since it
reconstructs the model directly from the exported weights (ParametricA2's own
_load_weight_block round-trips this exactly).

Method: enumerate the knob-space hypercube corners (all-min, all-max, each
knob solo-extreme -- the same reduced corner set already used elsewhere in
this fleet for knob-grid design, e.g. docs/tweed5f6a_training_notes.md) plus
the center, run each corner against a real-transient reference clip
([redacted]-sweep-v3.wav by convention -- local-only, licensed, already the fleet's
standard "real playing with hard attacks" reference) in windowed chunks, and
flag any window where the predicted peak is anomalous relative to that
model's OWN typical output level.

Usage:
  python tools/scan_film_runaway.py --nam PATH/TO/model.param.nam \
      [--reference ~/Downloads/[redacted]-sweep-v3.wav] [--chunk-s 5.0] \
      [--flag-ratio 8.0] [--flag-abs 3.0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from param_train import ParametricA2

SR = 48000


def load_widest_submodel(nam_path: str):
    d = json.loads(Path(nam_path).read_text())
    subs = d["config"]["submodels"]
    sub = subs[-1]  # widest tier (SlimmableContainer stores ascending by max_value)
    channels = sub["model"]["config"]["layers"]
    param_metas = sub["model"]["config"]["parametric"]["parameters"]
    param_names = [p["name"] for p in param_metas]
    weights = sub["model"]["weights"]
    model = ParametricA2(channels=channels, num_params=len(param_names))
    expected = model.weight_count()
    if len(weights) != expected:
        raise SystemExit(f"{nam_path}: weight count mismatch ({len(weights)} vs {expected}) "
                          f"-- export format assumption may not hold for this file")
    model.load_weights(weights)
    model.eval()
    return model, param_names, channels


def hypercube_corners(param_names):
    """All-min, all-max, each knob solo-extreme (rest at 0.5), and the center --
    the same reduced corner set already used for knob-grid design in this fleet
    (see docs/tweed5f6a_training_notes.md), not the full 2^n hypercube."""
    n = len(param_names)
    corners = [("all-min", [0.0] * n), ("all-max", [1.0] * n), ("center", [0.5] * n)]
    for i, name in enumerate(param_names):
        lo = [0.5] * n; lo[i] = 0.0
        hi = [0.5] * n; hi[i] = 1.0
        corners.append((f"{name}=0-solo", lo))
        corners.append((f"{name}=1-solo", hi))
    return corners


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nam", required=True)
    ap.add_argument("--reference", default=str(Path.home() / "Downloads/[redacted]-sweep-v3.wav"))
    ap.add_argument("--chunk-s", type=float, default=5.0)
    ap.add_argument("--flag-ratio", type=float, default=8.0,
                    help="flag a window if its peak exceeds this multiple of the model's own median peak")
    ap.add_argument("--flag-abs", type=float, default=3.0,
                    help="...AND exceeds this absolute volts floor (avoids flagging near-silent models)")
    args = ap.parse_args()

    model, param_names, channels = load_widest_submodel(args.nam)
    print(f"{Path(args.nam).name}: channels={channels} params={param_names}")

    x, sr = sf.read(args.reference, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    if sr != SR:
        raise SystemExit(f"reference sr {sr} != {SR}")

    corners = hypercube_corners(param_names)
    chunk_n = int(args.chunk_s * SR)
    n_chunks = len(x) // chunk_n

    results = []
    with torch.no_grad():
        for label, vals in corners:
            cond = torch.tensor([vals], dtype=torch.float32)
            for c in range(n_chunks):
                seg = x[c * chunk_n:(c + 1) * chunk_n]
                inp = torch.from_numpy(seg).float().reshape(1, 1, -1)
                pred = model(inp, cond)[0, 0].numpy()
                pk = float(np.abs(pred).max())
                results.append((pk, label, c))

    peaks = np.array([r[0] for r in results])
    median = float(np.median(peaks))
    results.sort(key=lambda r: -r[0])
    flagged = [r for r in results if r[0] > args.flag_ratio * median and r[0] > args.flag_abs]

    print(f"  {len(results)} (corner x {args.chunk_s:.0f}s-chunk) probes, "
          f"{len(corners)} corners x {n_chunks} chunks over {len(x)/SR:.0f}s")
    print(f"  median peak={median:.4f}  max peak={peaks.max():.4f}  min peak={peaks.min():.4f}")
    if flagged:
        print(f"  FLAGGED ({len(flagged)} windows > {args.flag_ratio}x median AND > {args.flag_abs}V):")
        for pk, label, c in flagged[:10]:
            print(f"    peak={pk:10.3f}  corner={label:16}  t={c*args.chunk_s:.0f}-{(c+1)*args.chunk_s:.0f}s")
    else:
        print("  clean -- no anomalous windows")
    return len(flagged)


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
