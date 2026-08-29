#!/usr/bin/env python3
"""
level_band_esr.py — ESR split by OUTPUT LEVEL BAND. The metric that decides whether the
Boss DS-1 fade-out bug is actually fixed.

Why this exists
---------------
Headline ESR is energy-weighted: it is a single ratio of summed error to summed signal, so a
note's decay tail contributes almost nothing to it. On the most distorted Big Muff setting,
windows below -40 dB of peak are 8.2% of the DURATION and 0.00% of the ENERGY. A model can drop
the distortion entirely as a note fades -- which is the DS-1 complaint -- and the headline ESR
will barely move.

So headline ESR cannot see the bug, and therefore cannot tell you if you fixed it.

This splits ESR by how loud the TARGET is in each window. The fade-out lives in the bottom bands.

    python level_band_esr.py --checkpoint best.pt --dataset <dir> --tier full

Read the output like this:

  * The `>-6 dB` and `-20..-6 dB` bands are what headline ESR already told you.
  * The `-40..-20` and `<-40` bands are the fade-out. THAT is where a loss fix must show up.
  * If the bottom bands do not improve, the loss change did not do what it was meant to,
    regardless of what the headline number does.

EXPECT THE HEADLINE TO GET SLIGHTLY WORSE. A model that now spends capacity on the quiet 20% of
a note has less to spend on the loud 80%. That is the trade being bought deliberately. Judge it
on the bands, not the average.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from checkpoint_infer import load_model

BANDS = [
    (-6.0, 999.0, "louder than -6 dB"),
    (-20.0, -6.0, "-20 .. -6 dB"),
    (-40.0, -20.0, "-40 .. -20 dB"),
    (-999.0, -40.0, "below -40 dB  (tail)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tier", default="full")
    ap.add_argument("--window-ms", type=float, default=100.0)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--target-dbfs", type=float, default=None,
                    help="Rescale the input to this RMS before inference. Default: none -- "
                         "ParamDataset no longer rescales its input (see param_train.py; RMS "
                         "rescaling taught a linearly-wrong input/output relationship for a "
                         "nonlinear circuit, and broke on real-playing/high-crest-factor sweep "
                         "files), so a bare invocation now matches current training. Pass this "
                         "explicitly only to evaluate a checkpoint trained under the OLD -18dBFS "
                         "convention.")
    ap.add_argument("--max-perms", type=int, default=0,
                    help="Evaluate only the N most distorted permutations (0 = all). "
                         "The fade-out bug is worst where the gain is highest.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    d = Path(args.dataset)
    model, _ = load_model(args.checkpoint)
    model.to(args.device).eval()
    labels = model.tier_labels()
    if args.tier not in labels:
        sys.exit(f"--tier {args.tier!r} not in {labels}")
    sub = model.submodels[labels.index(args.tier)]

    cfg = json.load(open(d / "config.json"))
    knobs = cfg["knobs"]

    x, sr = sf.read(str(d / "sweep.wav"), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if args.target_dbfs is not None:
        rms = float(np.sqrt(np.mean(x ** 2)))
        x = x * (10 ** (args.target_dbfs / 20.0) / (rms + 1e-8))

    outs = np.load(d / "outputs.npy", mmap_mode="r")
    import csv
    rows = list(csv.DictReader(open(d / "params.csv")))
    if args.max_perms:
        rows = sorted(rows, key=lambda r: -sum(float(r[k]) for k in knobs))[:args.max_perms]

    w = int(args.window_ms * sr / 1000)
    skip = int(args.warmup_s * sr)

    # accumulate error and signal energy per band, over all permutations
    num = {b[2]: 0.0 for b in BANDS}
    den = {b[2]: 0.0 for b in BANDS}
    dur = {b[2]: 0 for b in BANDS}

    for r in rows:
        i = int(r["idx"])
        p = [float(r[k]) for k in knobs]
        y = np.asarray(outs[i], dtype=np.float64)
        n = min(len(y), len(x))
        with torch.no_grad():
            xt = torch.from_numpy(x[:n]).float().view(1, 1, -1).to(args.device)
            pt = torch.tensor([p], dtype=torch.float32, device=args.device)
            yh = sub(xt, pt).squeeze().cpu().numpy().astype(np.float64)
        m = min(len(yh), n)
        y, yh = y[skip:m], yh[skip:m]
        if len(y) < w:
            continue

        nb = len(y) // w
        yy = y[:nb * w].reshape(nb, w)
        ee = (y - yh)[:nb * w].reshape(nb, w)
        e_sig = (yy ** 2).sum(axis=1)
        e_err = (ee ** 2).sum(axis=1)
        peak = np.sqrt(e_sig / w).max() + 1e-12
        db = 20 * np.log10(np.sqrt(e_sig / w) / peak + 1e-12)

        for lo, hi, lbl in BANDS:
            sel = (db > lo) & (db <= hi)
            num[lbl] += e_err[sel].sum()
            den[lbl] += e_sig[sel].sum()
            dur[lbl] += int(sel.sum())

    tot_dur = sum(dur.values()) or 1
    tot_e = sum(den.values()) or 1.0

    print(f"\n  {args.checkpoint}")
    print(f"  tier={args.tier}   {len(rows)} permutation(s)   {args.window_ms:.0f} ms windows\n")
    print(f"  {'band':<24}{'% duration':>12}{'% energy':>11}{'ESR in band':>14}")
    print(f"  {'-'*24}{'-'*12}{'-'*11}{'-'*14}")
    for _, _, lbl in BANDS:
        esr = num[lbl] / den[lbl] if den[lbl] > 0 else float("nan")
        print(f"  {lbl:<24}{100*dur[lbl]/tot_dur:>11.1f}%{100*den[lbl]/tot_e:>10.2f}%{esr:>14.4f}")
    overall = sum(num.values()) / tot_e
    print(f"\n  headline (energy-weighted) ESR = {overall:.4f}")
    print("  ^ this is the number that CANNOT SEE the fade-out. Judge on the bands.")

    if args.output:
        import csv as _csv
        with open(args.output, "w", newline="") as f:
            wr = _csv.writer(f)
            wr.writerow(["band", "pct_duration", "pct_energy", "esr"])
            for _, _, lbl in BANDS:
                e = num[lbl] / den[lbl] if den[lbl] > 0 else ""
                wr.writerow([lbl, 100*dur[lbl]/tot_dur, 100*den[lbl]/tot_e, e])
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
