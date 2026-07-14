#!/usr/bin/env python3
"""Measure BDF2 truncation error across the device fleet.

WHY THIS FILE EXISTS AS A FILE.
The first fleet-wide truncation table (docs/convergence.md, commit 9d986c5) was produced by an
ad-hoc script that was never committed. When its Big Muff number (2.25e-02) later failed to
reproduce -- a careful re-measurement on the same input, same circuit, same worst corner gives
9.33e-03 -- there was nothing to audit and no way to find out which of the two was wrong. A
measurement that decides whether we retrain the entire model fleet has to be re-runnable by
someone who doubts it. So it lives here, in the repo, and the docs quote it rather than the
other way round.

WHAT IT MEASURES.
oversample is a DISCRETISATION choice. LiveSPICE integrates with BDF2, which is O(h^2), so the
simulated circuit is not quite the real circuit; the gap is TRUNCATION ERROR. It is not a bug and
it is never zero. But its SIZE decides whether it matters: if the truncation in the training target
is the same order as the ESR of the model you fit to that target, the model is spending capacity
learning the integrator's mistakes. A model cannot be more right than its target.

HOW (and the trap).
Against a REFERENCE at high oversample -- NOT against the next rung up. ESR(os, 2*os) is the
tempting cheap estimate and it UNDERSTATES the error ~3x, which is the dangerous direction: you
pick too low an oversample and ship a contaminated target. The finer render is not the truth, it
is merely less wrong, and the two errors are correlated and largely cancel. Measured on the Big
Muff, worst corner, whole file:

    os      ESR vs 2*os      ESR vs os=32     understated by
     2       2.83e-03         9.33e-03            3.3x
     4       6.45e-04         1.98e-03            3.1x
     8       1.63e-04         3.70e-04            2.3x

All renders at 256 Newton iterations, so under-convergence is not a confound (that is a different
failure and docs/convergence.md covers it).

KNOB SETTINGS. Both ends of every knob, plus both corners, plus the mid point -- and we report
WHERE the worst was. "All knobs at max" is not reliably the stiff setting: the Boss DS-1's Dist pot
is ReverseLinear, so all-max is MINIMUM drive.

Usage:
    ./measure_truncation.py --input ../sweep-files/sweep60_composite.wav          # whole fleet
    ./measure_truncation.py --input ... --device big-muff-pi-v1-66-5              # one device
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf

from batch_harness import LIVESPICE_CLI

REGISTRY = Path(__file__).resolve().parent.parent / "spice-circuits" / "devices.toml"


def esr(a: np.ndarray, b: np.ndarray, skip_frac: float = 0.1) -> float:
    """ESR of `a` against reference `b`, dropping the leading transient."""
    m = min(len(a), len(b))
    sk = int(m * skip_frac)
    den = float(np.sum(b[sk:m] ** 2))
    if den <= 0:
        return float("nan")
    return float(np.sum((a[sk:m] - b[sk:m]) ** 2) / den)


def probe_settings(knobs: list[str]) -> list[dict]:
    """Both ends of each knob, both corners, and the mid point."""
    picks: list[dict] = []
    for k in knobs:
        for v in (0.0, 1.0):
            p = {j: 0.5 for j in knobs}
            p[k] = v
            picks.append(p)
    picks.append({k: 0.0 for k in knobs})
    picks.append({k: 1.0 for k in knobs})
    picks.append({k: 0.5 for k in knobs})
    seen, uniq = set(), []
    for p in picks:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def load_fleet(only: str | None) -> list[tuple[str, Path, list[str]]]:
    reg = tomllib.loads(REGISTRY.read_text())
    root = REGISTRY.parent
    out = []
    for dev_id, d in reg.items():
        if only and dev_id != only:
            continue
        # Only real knobs. A switch changes the TOPOLOGY, so it is not a point on a knob sweep.
        knobs = [k["name"] for k in d.get("knobs", []) if k.get("kind") != "switch"]
        if not knobs:
            continue
        out.append((d["name"], root / d["schx"], knobs))
    return out


def measure(schx: Path, knobs: list[str], input_wav: Path, ref_os: int,
            candidates: tuple[int, ...], iterations: int, speaker: str | None,
            td: Path) -> dict:
    """Worst-over-knob-settings truncation ESR at each candidate oversample."""
    cache: dict[tuple, np.ndarray | None] = {}

    def render(p: dict, os_: int):
        key = (tuple(sorted(p.items())), os_)
        if key in cache:
            return cache[key]
        w = td / f"{abs(hash((str(schx), key)))}.wav"
        cmd = [str(LIVESPICE_CLI), "--input", str(input_wav), "--output", str(w),
               "--circuit", str(schx),
               "--params", ",".join(f"{k}={v}" for k, v in p.items()),
               "--oversample", str(os_), "--iterations", str(iterations)]
        if speaker:
            cmd += ["--speaker", speaker]
        r = subprocess.run(cmd, capture_output=True, text=True)
        v = None
        if r.returncode == 0 and w.exists():
            d, _ = sf.read(str(w))
            v = np.asarray(d, dtype=np.float64)
        else:
            print(f"      render failed (os={os_}): {r.stderr.strip().splitlines()[-1:]}",
                  file=sys.stderr)
        cache[key] = v
        return v

    settings = probe_settings(knobs)
    res: dict = {"n_settings": len(settings)}
    for os_ in candidates:
        worst, at = 0.0, None
        for p in settings:
            a, ref = render(p, os_), render(p, ref_os)
            if a is None or ref is None:
                continue
            e = esr(a, ref)
            if np.isfinite(e) and e >= worst:
                worst, at = e, p
        res[os_] = (worst, at)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="the sweep to measure on. Truncation ESR is INPUT-DEPENDENT (a quiet tail "
                         "shrinks the denominator), so the table is only meaningful with the input "
                         "named. Use the sweep the datasets were actually rendered with.")
    ap.add_argument("--device", help="one device id from devices.toml (default: the whole fleet)")
    ap.add_argument("--ref-os", type=int, default=32, help="reference oversample (default 32)")
    ap.add_argument("--candidates", default="2,4,8",
                    help="oversamples to score against the reference (default 2,4,8)")
    ap.add_argument("--iterations", type=int, default=256,
                    help="Newton iterations; high, so convergence is not a confound (default 256)")
    ap.add_argument("--speaker", help="speaker to capture on multi-speaker circuits")
    args = ap.parse_args()

    cands = tuple(int(c) for c in args.candidates.split(","))
    if max(cands) >= args.ref_os:
        ap.error(f"--candidates must all be below --ref-os ({args.ref_os}): you cannot measure the "
                 f"reference against itself")

    fleet = load_fleet(args.device)
    if not fleet:
        ap.error(f"no devices matched (registry: {REGISTRY})")

    print(f"input:      {args.input.name}")
    print(f"reference:  oversample={args.ref_os}, {args.iterations} Newton iterations")
    print(f"devices:    {len(fleet)}\n")

    rows = []
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        for name, schx, knobs in fleet:
            if not schx.exists():
                print(f"  {name}: MISSING {schx}", file=sys.stderr)
                continue
            print(f"  measuring {name} ({len(knobs)} knobs) ...", flush=True, file=sys.stderr)
            r = measure(schx, knobs, args.input, args.ref_os, cands,
                        args.iterations, args.speaker, td)
            rows.append((name, r))

    hdr = " | ".join(f"@ os={c}" for c in cands)
    print(f"\n| device | {hdr} | worst setting @ os={cands[0]} | falls per doubling |")
    print("|---|" + "---:|" * len(cands) + "---|---|")
    for name, r in rows:
        cells = []
        for c in cands:
            v, _ = r.get(c, (float('nan'), None))
            cells.append(f"{v:.2e}")
        v0, at0 = r.get(cands[0], (float('nan'), None))
        v1, _ = r.get(cands[1], (float('nan'), None)) if len(cands) > 1 else (float('nan'), None)
        fall = f"{v0 / v1:.1f}x" if (np.isfinite(v0) and np.isfinite(v1) and v1 > 0) else "-"
        atstr = ", ".join(f"{k}={v:g}" for k, v in sorted(at0.items())) if at0 else "-"
        smooth = "" if (np.isfinite(v0) and np.isfinite(v1) and v1 > 0 and v0 / v1 >= 2.0) \
                 else "  **NOT SMOOTH**"
        print(f"| {name} | " + " | ".join(cells) + f" | {atstr} | {fall}{smooth} |")

    print("\nO(h^2) demands the error fall ~4x per doubling. A device that falls <2x does not "
          "converge in the timestep:\nsomething in it is not smooth, and oversampling will not fix "
          "its dataset.")


if __name__ == "__main__":
    main()
