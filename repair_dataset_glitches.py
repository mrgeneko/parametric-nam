#!/usr/bin/env python3
"""Repair sporadic solver glitches in a combined dataset's outputs.npy, using two more renders.

WHY THIS EXISTS.
A fixed-timestep implicit solver (LiveSPICE) driven hard by a full-level excitation can produce
short, isolated glitches -- a few ms, one to three per hot combination -- at ANY oversample, and
raising the oversample does not cure them: they come and go per (oversample, transient) instead of
shrinking with the timestep. Found on the Marshall Bluesbreaker (2026-09-25): rows rendered at
oversample 8 differed from an independent adaptive-timestep ngspice render and from oversample-32
LiveSPICE by 0.15-0.47 (normalised units) for ~10 ms, while oversample 16 blew up entirely on
three other combinations. Because the glitches are independent from render to render, the
MEDIAN of three renders removes any single render's glitch.

HOW.
For each combination: x = the dataset row, a and b = two further renders of the SAME combination
on the SAME input at other oversamples. m = median(x, a, b) sample by sample. x is flagged only
where it is the OUTLIER (|x - m| > tol), i.e. outside the [min(a,b), max(a,b)] band by more than
tol; those samples are padded, merged into spans, and replaced by m with a short crossfade. Every
sample outside a span is left BIT-IDENTICAL, so the dataset keeps its original render everywhere
it was already right. A glitch in a or b (not x) flags nothing.

a and b are RAW renders; they go through the same post-processing the dataset got (the virtual
capture chain declared in config.json, then its output_scale), so all three are comparable in
the dataset's own units.

SAFETY. The original outputs.npy is copied to outputs.pre_repair.npy first (never overwritten),
the repair is recorded in config.json under "glitch_repair" (method, tol, per-combination spans,
sha256 of the pre-repair file), and a second run refuses unless --force. --dry-run reports
without writing anything. A combination is skipped (and reported) if either extra render is
missing.

    ./repair_dataset_glitches.py --dataset DIR \\
        --render-a "/path/c{idx:02d}_os16.wav" --render-b "/path/c{idx:02d}_os32.wav" --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from capture_chain import capture_chain


def process_render(y: np.ndarray, sr: int, capture: dict | None, scale: float) -> np.ndarray:
    """A raw render -> the dataset's units: capture chain (if the dataset declares one), then
    output_scale. Same order as gen_dataset_from_schx: chain in _finalize_wav, scale in combine()."""
    y = np.asarray(y, dtype=np.float32)
    if capture:
        y = capture_chain(y, sr, corner_hz=capture["corner_hz"], order=capture["order"])
    return (y * scale).astype(np.float64)


def detect_spans(x: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float, pad: int,
                 merge_gap: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """([start, end) spans where x is the outlier of the three, median array m).

    All three arrays must be the same length. `pad` widens every flagged run on both sides;
    runs closer than `merge_gap` samples are joined so one glitch is one span."""
    m = np.median(np.stack([x, a, b]), axis=0)
    bad = np.flatnonzero(np.abs(x - m) > tol)
    if bad.size == 0:
        return [], m
    brk = np.flatnonzero(np.diff(bad) > merge_gap)
    firsts, lasts = bad[np.r_[0, brk + 1]], bad[np.r_[brk, bad.size - 1]]
    spans: list[list[int]] = []
    for s, e in zip(firsts, lasts):
        s, e = max(0, int(s) - pad), min(len(x), int(e) + pad + 1)
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    return [(s, e) for s, e in spans], m


def apply_patch(x: np.ndarray, m: np.ndarray, spans: list[tuple[int, int]], fade: int) -> np.ndarray:
    """x with each span replaced by m, crossfaded over `fade` samples at both ends of the span
    (so the switch itself cannot create a step). Samples outside every span are untouched."""
    y = x.copy()
    for s, e in spans:
        w = np.ones(e - s)
        f = min(fade, (e - s) // 2)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, endpoint=False)
            w[:f], w[-f:] = ramp, ramp[::-1]
        y[s:e] = x[s:e] + w * (m[s:e] - x[s:e])
    return y


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def repair(dataset: Path, pat_a: str, pat_b: str, tol: float, pad_ms: float, merge_ms: float,
           fade_ms: float, dry_run: bool = False, force: bool = False,
           combos: list[int] | None = None, log=print) -> dict:
    cfg_path, out_path = dataset / "config.json", dataset / "outputs.npy"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("glitch_repair") and not force:
        raise SystemExit(f"{dataset} was already repaired ({cfg['glitch_repair'].get('at')}). The "
                         f"original is outputs.pre_repair.npy; pass --force to repair again.")
    sr = int((cfg.get("input") or {}).get("samplerate") or 48000)
    scale = float(cfg.get("output_scale") or 1.0)
    capture = cfg.get("capture_chain")
    pad, gap, fade = (int(v * sr / 1000) for v in (pad_ms, merge_ms, fade_ms))
    out = np.load(out_path, mmap_mode="r")
    todo = list(range(out.shape[0])) if combos is None else combos
    new, report, skipped = {}, {}, []
    for i in todo:
        pa, pb = Path(pat_a.format(idx=i)), Path(pat_b.format(idx=i))
        if not (pa.exists() and pb.exists()):
            skipped.append(i)
            continue
        x = np.asarray(out[i], dtype=np.float64)
        a = process_render(sf.read(str(pa))[0], sr, capture, scale)
        b = process_render(sf.read(str(pb))[0], sr, capture, scale)
        if not (len(a) == len(b) == len(x)):
            raise SystemExit(f"combination {i}: lengths differ (dataset {len(x)}, a {len(a)}, b {len(b)})")
        spans, m = detect_spans(x, a, b, tol, pad, gap)
        worst = float(np.abs(x - m).max())
        report[i] = {"spans": [[int(s), int(e)] for s, e in spans],
                     "samples": int(sum(e - s for s, e in spans)), "max_outlier": worst}
        log(f"  combo {i:3d}: {len(spans):2d} span(s), {report[i]['samples']:7d} samples patched, "
            f"largest outlier {worst:.3g}")
        if spans:
            new[i] = apply_patch(x, m, spans, fade)
    if skipped:
        log(f"  skipped (a render is missing): {skipped}")
    summary = {"repaired": sorted(new), "skipped": skipped, "combos": report}
    if dry_run or not new:
        log("dry run -- nothing written" if dry_run else "nothing to patch")
        return summary

    backup = dataset / "outputs.pre_repair.npy"
    if backup.exists():
        raise SystemExit(f"{backup} already exists; refusing to overwrite the only pre-repair copy")
    digest = _sha256(out_path)
    shutil.copy2(out_path, backup)
    tmp = dataset / ".outputs.repair.tmp.npy"
    arr = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=out.dtype, shape=out.shape)
    arr[:] = out
    for i, y in new.items():
        arr[i] = y.astype(out.dtype)
    arr.flush()
    del arr
    os.replace(tmp, out_path)
    cfg["glitch_repair"] = {
        "tool": "repair_dataset_glitches.py", "method": "median-of-three outlier patch",
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tol": tol, "pad_ms": pad_ms,
        "merge_ms": merge_ms, "fade_ms": fade_ms, "render_a": pat_a, "render_b": pat_b,
        "pre_repair_file": backup.name, "pre_repair_sha256": digest,
        "combos": {str(i): report[i] for i in sorted(new)}}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log(f"patched {len(new)} combination(s); original kept as {backup.name}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="a combined dataset dir (has outputs.npy)")
    ap.add_argument("--render-a", required=True, help='raw render of combo i, "{idx}" is the row index')
    ap.add_argument("--render-b", required=True, help="second raw render, same pattern")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="flag x where |x - median| exceeds this, in the dataset's own units "
                         "(default 0.01; choose it just above the inter-render noise plateau)")
    ap.add_argument("--pad-ms", type=float, default=2.0, help="widen each flagged run (default 2)")
    ap.add_argument("--merge-ms", type=float, default=5.0, help="join runs closer than this (default 5)")
    ap.add_argument("--fade-ms", type=float, default=0.5, help="crossfade at span edges (default 0.5)")
    ap.add_argument("--combos", help="comma-separated row indices to consider (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--force", action="store_true", help="repair even if already repaired")
    a = ap.parse_args()
    combos = [int(c) for c in a.combos.split(",")] if a.combos else None
    repair(a.dataset, a.render_a, a.render_b, a.tol, a.pad_ms, a.merge_ms, a.fade_ms,
           dry_run=a.dry_run, force=a.force, combos=combos)


if __name__ == "__main__":
    main()
