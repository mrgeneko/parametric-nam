#!/usr/bin/env python3
"""Is the knob grid dense enough? Answer it BEFORE training, and without training anything.

THE QUESTION.
A parametric model is fitted to the circuit at a finite set of knob settings and is expected to fill
in everything between them. So the grid has to be dense enough that "everything between them" is
recoverable. Too coarse and the model cannot win no matter how long you train it -- the information
simply is not in the data. Too dense and you pay for renders and epochs that teach it nothing.

Nobody could say which we had. The obvious way to find out -- train on grid A, train on grid B,
compare -- is expensive, slow, and confounded by optimiser noise (we tried: nine overnight runs whose
ESR got monotonically WORSE as the grid got denser, which is physically impossible and told us only
that the runs were undertrained).

THE MEASUREMENT.
Render the circuit's TRUE output at the midpoint of a grid cell. Compare it against interpolating the
two neighbouring grid points. The residual is the interpolation error THE GRID IMPOSES: a property of
the circuit and the sampling, with no model in it at all. Compare that residual to the ESR you are
chasing:

    residual >> target ESR   the GRID is the limiting factor. No capacity, no training time, and no
                             architecture can recover what was never sampled. Refine the cell.
    residual << target ESR   the cell is OVERSAMPLED. Those points cost render time and lengthen
                             every epoch (dataset items = perms x repeats) and teach the model
                             nothing it could not have interpolated.

It is cheap: 3 renders per cell probe, and cells are independent so they run concurrently.

WHAT IT FOUND ON THE BIG MUFF (14 x 9 = 126 perms, the shipped grid):

    Sustain 0.001-0.70   11 cells, all <= 0.0010     oversampled 10-100x
    Sustain 0.70-0.85                     0.0091     at the limit
    Sustain 0.85-1.00                     0.0813     9x OVER -- the model cannot win here
    Tone    every one of 8 cells          0.0000     all nine values interpolate perfectly

So the grid was simultaneously too dense (almost everywhere) and too coarse (in one place), and the
one coarse cell is where a Big Muff actually lives. The Sustain axis had been hand-tuned dense at the
BOTTOM, on the reasoning that a linear-taper pot puts most of its audible change in the bottom decade.
That is true of LEVEL, and backwards for WAVEFORM SHAPE, which changes fastest at the top where the
pedal is deep into clipping -- and waveform shape is what the model has to learn.

    ./tools/grid_adequacy.py --config configs/big-muff-v1.toml --target 0.009
    ./tools/grid_adequacy.py --config ... --suggest       # propose a regrid
"""
from __future__ import annotations

import argparse
import os as _os
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batch_harness import LIVESPICE_CLI


def esr(a: np.ndarray, b: np.ndarray) -> float:
    m = min(len(a), len(b))
    den = float(np.sum(b[:m] ** 2))
    return float(np.sum((a[:m] - b[:m]) ** 2) / den) if den > 0 else float("nan")


def load_config(p: Path) -> dict:
    with open(p, "rb") as f:
        return tomllib.load(f)


class Renderer:
    """Renders a knob setting on STRATIFIED WINDOWS spanning the input, not on one clip.

    WHICH PART OF THE INPUT YOU PROBE DECIDES THE ANSWER, and both obvious choices are wrong. The
    first probe_s seconds understates everything (our sweeps open quiet). The single highest-slew
    window OVERSTATES it: on sweepv5 that window lands inside the chirp section, and measured there
    the Big Muff's Sustain 0.03-0.12 cell reads 0.0079 against a whole-file truth of 0.0008 -- a 10x
    exaggeration that would have bought points the model does not need.

    The quantity that matters is the whole-file one, because that is what the model is fitted against
    and scored on. So: several short windows spanning the input, rendered SEPARATELY (never spliced --
    a concatenation manufactures discontinuities that are not in the signal and the probe then
    measures its own splice), with the ESR numerator and denominator POOLED across them. sum(err) /
    sum(sig) is exactly the whole-file ESR restricted to the sampled windows, and an unbiased estimate
    of it.

    This is the same trap, and the same fix, as choose_oversample in batch_harness.py.
    """

    def __init__(self, schx, inp, oversample, iterations, fixed, td, probe_s, n_windows=4):
        self.schx, self.os_, self.it = schx, oversample, iterations
        self.fixed, self.td = fixed, td
        self.cache: dict = {}

        x, sr = sf.read(str(inp))
        total = len(x)
        win = min(total, int(probe_s * sr) // n_windows or total)
        if total <= win * n_windows:
            starts, win = [0], total
        else:
            span = total - win
            starts = [int(round(i * span / (n_windows - 1))) for i in range(n_windows)]

        self.clips = []
        for i, st in enumerate(starts):
            c = td / f"probe{i}.wav"
            sf.write(str(c), x[st:st + win], sr)
            self.clips.append(c)
        self.win_s = win / sr

    def __call__(self, params: dict):
        """-> list of rendered windows (one array per probe window)."""
        key = tuple(sorted(params.items()))
        if key in self.cache:
            return self.cache[key]
        allp = ",".join(f"{k}={v}" for k, v in params.items())
        if self.fixed:
            allp = f"{self.fixed},{allp}"
        out = []
        for i, clip in enumerate(self.clips):
            w = self.td / f"{abs(hash(key))}_{i}.wav"
            r = subprocess.run(
                [str(LIVESPICE_CLI), "--input", str(clip), "--output", str(w),
                 "--circuit", str(self.schx), "--params", allp,
                 "--oversample", str(self.os_), "--iterations", str(self.it)],
                capture_output=True, text=True)
            if r.returncode == 0 and w.exists():
                d, _ = sf.read(str(w))
                out.append(np.asarray(d, dtype=np.float64))
            else:
                tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
                print(f"      render failed at {allp}: {tail}", file=sys.stderr)
                out.append(None)
        self.cache[key] = out
        return out


def cell_error(render, knobs: dict, axis: str, lo: float, hi: float, others: dict) -> float:
    """Interpolation error the grid imposes on the cell [lo, hi] of `axis`.

    Pooled across the probe windows: sum of squared error over sum of squared signal, which IS the
    whole-file ESR restricted to those windows. Averaging per-window ESRs instead would let one quiet
    window dominate the mean.
    """
    mid = 0.5 * (lo + hi)
    A = render({**others, axis: lo})
    B = render({**others, axis: hi})
    M = render({**others, axis: mid})
    num = den = 0.0
    for a, b, m in zip(A, B, M):
        if a is None or b is None or m is None:
            continue
        n = min(len(a), len(b), len(m))
        sk = n // 10                                  # drop each window's start transient
        interp = 0.5 * (a[:n] + b[:n])                # the best a grid-limited learner can recover
        num += float(np.sum((interp[sk:] - m[sk:n]) ** 2))
        den += float(np.sum(m[sk:n] ** 2))
    return num / den if den > 0 else float("nan")


def suggest_axis(values: list, worst: dict, target: float) -> list:
    """Refine the cells that fail, drop points that are not earning their place.

    A cell whose error is FAR under target (<= target/10) is oversampled: its interior is recoverable
    by interpolation, so the point between two such cells is redundant. A cell over target gets
    bisected. Applied once -- run again to iterate.
    """
    out = [values[0]]
    i = 0
    while i < len(values) - 1:
        lo, hi = values[i], values[i + 1]
        e = worst.get((lo, hi), float("nan"))
        if np.isfinite(e) and e > target:
            out.append(round(0.5 * (lo + hi), 4))        # bisect: the grid is losing here
            out.append(hi)
            i += 1
        elif (i + 2 < len(values)
              and np.isfinite(e) and e <= target / 10
              and np.isfinite(worst.get((values[i + 1], values[i + 2]), float("nan")))
              and worst[(values[i + 1], values[i + 2])] <= target / 10):
            out.append(values[i + 2])                     # drop values[i+1]: both cells are trivial
            i += 2
        else:
            out.append(hi)
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the same TOML the pipeline uses")
    ap.add_argument("--target", type=float, default=0.009,
                    help="the model ESR you are chasing (default 0.009). A cell whose interpolation "
                         "error exceeds this is the LIMITING FACTOR -- no training fixes it.")
    ap.add_argument("--probe-s", type=float, default=8.0,
                    help="seconds of the input to probe with, taken from its highest-slew window")
    ap.add_argument("--iterations", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 4) - 2))
    ap.add_argument("--suggest", action="store_true", help="propose a regrid and print it as TOML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    knobs: dict = cfg["knobs"]
    schx = Path(_os.path.expanduser(cfg["schx"]))
    inp = Path(_os.path.expanduser(cfg["input"]))
    fixed = ",".join(f"{k}={v}" for k, v in (cfg.get("fixed") or {}).items())
    oversample = cfg.get("oversample", 2)

    n_perms = int(np.prod([len(v) for v in knobs.values()]))
    print(f"  config     {args.config}")
    print(f"  grid       {' x '.join(str(len(v)) for v in knobs.values())} = {n_perms} permutations")
    print(f"  target ESR {args.target}   (a cell above this is the limiting factor)")
    print(f"  probe      {args.probe_s:.0f}s @ oversample {oversample}, {args.iterations} iters\n")

    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        render = Renderer(schx, inp, oversample, args.iterations, fixed, td, args.probe_s)

        # Probe each cell at several positions of the OTHER knobs -- the knobs interact, and a cell
        # that interpolates cleanly at one Tone can fail at another. (On the Big Muff the Sustain
        # 0.85-1.0 cell is 0.0044 at Tone=0 and 0.0813 at Tone=1: an 18x spread. Probing one slice
        # would have missed it.)
        jobs = []
        for axis, vals in knobs.items():
            other_axes = {k: v for k, v in knobs.items() if k != axis}
            slices = [{}] if not other_axes else [
                {k: v[i] for k, v in other_axes.items()}
                for i in (0, len(next(iter(other_axes.values()))) // 2, -1)
            ]
            for lo, hi in zip(vals, vals[1:]):
                for oth in slices:
                    jobs.append((axis, lo, hi, oth))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(lambda j: (j[0], j[1], j[2], cell_error(render, knobs, j[0], j[1], j[2], j[3])),
                              jobs))

    worst_by_axis: dict = {}
    n_coarse = n_over = 0
    for axis, vals in knobs.items():
        rows = [r for r in res if r[0] == axis]
        worst = {}
        for lo, hi in zip(vals, vals[1:]):
            es = [r[3] for r in rows if r[1] == lo and r[2] == hi and np.isfinite(r[3])]
            worst[(lo, hi)] = max(es) if es else float("nan")
        worst_by_axis[axis] = worst

        print(f"  {axis}:")
        for (lo, hi), e in worst.items():
            if not np.isfinite(e):
                verdict = "?"
            elif e > args.target:
                verdict = f"**TOO COARSE** ({e/args.target:.1f}x over) — the grid is the limit here"
                n_coarse += 1
            elif e <= args.target / 10:
                verdict = "oversampled — these points teach the model nothing"
                n_over += 1
            else:
                verdict = "ok"
            print(f"    {lo:>7.4g} - {hi:<7.4g}  {e:>8.4f}   {verdict}")
        print()

    print(f"  {n_coarse} cell(s) too coarse, {n_over} oversampled.")
    if n_coarse:
        print("  A cell over target cannot be fixed by training: the information is not in the data.")

    if args.suggest:
        print("\n  suggested regrid:\n\n  [knobs]")
        tot = 1
        for axis, vals in knobs.items():
            new = suggest_axis(list(vals), worst_by_axis[axis], args.target)
            tot *= len(new)
            print(f"  {axis:<8} = {new}")
        print(f"\n  -> {tot} permutations (was {n_perms})")

    sys.exit(1 if n_coarse else 0)


if __name__ == "__main__":
    main()
