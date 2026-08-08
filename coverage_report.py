#!/usr/bin/env python3
"""Coverage report for a sampled dataset's params.csv.

Read-only analysis of which knob combinations were actually sampled (and which
sims succeeded), to surface 'holes' in a random-sampled high-dimensional knob
space. In N-D, fine-grained full-grid occupancy is meaningless (500 points in
6-D is ~2.8 points/axis), so this focuses on what matters for FiLM interpolation:

  - Success/failure summary (failed sims are real holes)
  - Boundary coverage: is each knob's min AND max actually present in the ok set?
  - Per-knob 1-D histograms (even coverage across each knob's range?)
  - Pairwise 2-D occupancy (worst-covered knob pairs)
  - Nearest-neighbour gap analysis -> the largest empty regions ('holes'),
    optionally emitted as suggested fill points to append (the sampler is
    seed-extensible, so you can grow the dataset toward the gaps).

Usage:
  python coverage_report.py --dataset /path/to/dataset [--bins 10] \
      [--suggest 10] [--emit-fill fill.csv] [--candidates 20000] [--seed 0]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _ok(row):
    return str(row.get("ok", "")).strip().lower() in ("1", "true", "yes")


def load(dataset: Path):
    cfg = json.loads((dataset / "config.json").read_text())
    knobs = cfg["knobs"]
    b = cfg.get("bounds", {})
    lo = np.array([float(b.get(k, [0.0, 1.0])[0]) for k in knobs])
    hi = np.array([float(b.get(k, [0.0, 1.0])[1]) for k in knobs])
    rows = list(csv.DictReader(open(dataset / "params.csv")))
    if not rows:
        return knobs, lo, hi, np.zeros((0, len(knobs))), np.zeros(0, bool), rows
    P = np.array([[float(r[k]) for k in knobs] for r in rows], dtype=float)
    ok = np.array([_ok(r) for r in rows], dtype=bool)
    return knobs, lo, hi, P, ok, rows


def bar(frac, width=34):
    n = int(round(frac * width))
    return "#" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--bins", type=int, default=10, help="1-D histogram bins per knob")
    ap.add_argument("--pair-bins", type=int, default=5, help="grid resolution per axis for 2-D occupancy")
    ap.add_argument("--suggest", type=int, default=0, help="emit N suggested fill points at the largest gaps")
    ap.add_argument("--candidates", type=int, default=20000, help="random candidates probed for gap-finding")
    ap.add_argument("--emit-fill", type=Path, default=None, help="write suggested fill points to this CSV")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    knobs, lo, hi, P, ok, rows = load(args.dataset)
    d = len(knobs)
    span = np.where(hi > lo, hi - lo, 1.0)
    n_total = len(rows)
    n_ok = int(ok.sum())
    Pok = P[ok] if n_total else P

    print(f"\nCoverage report — {args.dataset}")
    print(f"  knobs ({d}): {', '.join(knobs)}")
    print(f"  bounds: " + ", ".join(f"{k}[{lo[i]:.2f},{hi[i]:.2f}]" for i, k in enumerate(knobs)))
    print(f"  permutations: {n_total} total, {n_ok} ok, {n_total - n_ok} failed")
    if n_ok == 0:
        print("  no successful perms yet — nothing to analyse."); return

    # --- failures ---
    if n_total - n_ok:
        print("\nFAILED sims (holes — region produced no valid data):")
        for r in rows:
            if not _ok(r):
                vals = " ".join(f"{k}={float(r[k]):.2f}" for k in knobs)
                print(f"  idx {r.get('idx','?'):>4}: {vals}   {str(r.get('error',''))[:50]}")

    # --- boundary coverage (does each knob's min AND max appear in the ok set?) ---
    print("\nBoundary coverage (each knob's trained min/max present in ok data?):")
    for i, k in enumerate(knobs):
        col = Pok[:, i]
        nlo = int(np.isclose(col, lo[i], atol=1e-6).sum())
        nhi = int(np.isclose(col, hi[i], atol=1e-6).sum())
        smin, smax = col.min(), col.max()
        flag = "" if (nlo and nhi) else "   <-- MISSING an extreme"
        print(f"  {k:<12} min={lo[i]:.2f}:{'ok' if nlo else 'ABSENT'}(x{nlo})  "
              f"max={hi[i]:.2f}:{'ok' if nhi else 'ABSENT'}(x{nhi})  "
              f"[sampled {smin:.2f}..{smax:.2f}]{flag}")

    # --- per-knob 1-D histograms ---
    print(f"\nPer-knob coverage ({args.bins} bins across each range):")
    for i, k in enumerate(knobs):
        edges = np.linspace(lo[i], hi[i], args.bins + 1)
        cnt, _ = np.histogram(Pok[:, i], bins=edges)
        empties = int((cnt == 0).sum())
        peak = max(cnt.max(), 1)
        print(f"  {k}:  {bar(0)}  (empty bins: {empties}/{args.bins})")
        for b in range(args.bins):
            print(f"    {edges[b]:.2f}-{edges[b+1]:.2f} {bar(cnt[b]/peak, 28)} {cnt[b]}")

    # --- normalise for distance/pair analysis ---
    U = (Pok - lo) / span  # -> [0,1]^d

    # --- pairwise 2-D occupancy (worst-covered pairs) ---
    pb = args.pair_bins
    print(f"\nPairwise 2-D occupancy ({pb}x{pb} cells, worst-covered pairs):")
    pairs = []
    for a in range(d):
        for b in range(a + 1, d):
            ia = np.clip((U[:, a] * pb).astype(int), 0, pb - 1)
            ib = np.clip((U[:, b] * pb).astype(int), 0, pb - 1)
            filled = len(set(zip(ia.tolist(), ib.tolist())))
            pairs.append((filled / (pb * pb), knobs[a], knobs[b], filled))
    for frac, ka, kb, filled in sorted(pairs)[:min(8, len(pairs))]:
        print(f"  {ka:<10} x {kb:<10}  {filled:>3}/{pb*pb} cells filled  ({frac*100:.0f}%)")

    # --- nearest-neighbour gaps + suggested fill points ---
    if cKDTree is None:
        print("\n(scipy unavailable — skipping gap/fill analysis)"); return
    tree = cKDTree(U)
    dd, _ = tree.query(U, k=min(2, len(U)))
    if U.shape[0] > 1:
        nn = dd[:, 1]
        print(f"\nNearest-neighbour spacing (normalised): "
              f"median={np.median(nn):.3f}  max={nn.max():.3f}  (large max = an isolated sample)")

    rng = np.random.default_rng(args.seed)
    cand = rng.random((args.candidates, d))
    cdist, _ = tree.query(cand)  # distance from each candidate to nearest real sample
    order = np.argsort(-cdist)
    # greedy farthest-point: spread suggestions across distinct holes
    k_sug = args.suggest if args.suggest else (5 if args.emit_fill else 0)
    print(f"\nLargest empty regions (biggest holes), radius = dist to nearest sample:")
    chosen = []
    grown = U.copy()
    for _ in range(min(k_sug if k_sug else 3, len(order))):
        gt = cKDTree(grown)
        cd, _ = gt.query(cand)
        j = int(np.argmax(cd))
        pt = cand[j] * span + lo
        chosen.append((cd[j], pt))
        grown = np.vstack([grown, cand[j]])
        vals = " ".join(f"{knobs[m]}={pt[m]:.2f}" for m in range(d))
        print(f"  hole radius {cd[j]:.3f}: {vals}")

    if args.emit_fill and chosen:
        with open(args.emit_fill, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(knobs)
            for _, pt in chosen:
                w.writerow([f"{v:.4f}" for v in pt])
        print(f"\nWrote {len(chosen)} suggested fill points -> {args.emit_fill}")
        print("  (feeding these back needs a --perms-csv ingest in gen_dataset_from_schx; "
              "or just widen --random and re-run, the sampler is seed-extensible.)")


if __name__ == "__main__":
    main()
