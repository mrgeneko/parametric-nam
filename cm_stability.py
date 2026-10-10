#!/usr/bin/env python3
"""Solver stability of a cm device across its knob grid, on the excitation the dataset will render.

cm has no spike detector or retry ladder (stability_sweep.py is livespice-only). Its solver
reports divergences and unconverged Newton solves through cm_run --metrics. The dataset
generator and CmBackend both gate on them; this gives the full per-corner picture up front.
It renders the same corners prepare_excitation.py sizes against, with CmBackend's settings
(--prepared off, 2 s lead-in; the dataset generator uses a prepared state when one is usable,
which settles the same supply), and reports per corner:

  FAIL  divergences > 0, non-finite output, or the render produced no audio
  WARN  severe solves > 0, or unconverged solves above --max-unconverged of all solves

Exit status is 1 on any FAIL, so a generation script can gate on it.

  python cm_stability.py --config path/to/device/config.toml [--input excitation.wav] [--workers 4] [-o report.json]
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_transient_coverage import _corners, _sample_interior, resolve_sample_grid  # noqa: E402
from render_backends import _find_cm_run_exe  # noqa: E402


def render_corner(circuit, in_wav, scratch, idx, params, oversample, iterations, lead_in):
    out = os.path.join(scratch, f"c{idx}.wav")
    metrics = os.path.join(scratch, f"c{idx}.json")
    args = [_find_cm_run_exe(), circuit, in_wav, out, "--prepared", "off", "--os", str(oversample),
            "--tol-rel", "1e-4", "--tables", "on", "--resampler", "fir-linear", "--iterations", str(iterations),
            "--metrics", metrics]
    if lead_in > 0:
        args += ["--lead-in", f"{lead_in:g}"]
    for k, v in params.items():
        args += ["--knob", f"{k}={v}"]
    r = subprocess.run(args, capture_output=True, text=True)
    rec = {"rc": r.returncode, "params": params, "stderr_tail": " | ".join((r.stderr or "").strip().splitlines()[-2:])}
    try:
        m = json.load(open(metrics))
        rec.update({k: m[k] for k in ("solves", "unconverged", "severe", "divergences", "peak", "realtime_factor")})
    except (OSError, ValueError, KeyError):
        rec["error"] = "no metrics written"
    try:
        y, _ = sf.read(out, dtype="float64")
        rec["finite"] = bool((y == y).all() and abs(y).max() < float("inf"))
    except Exception as e:  # noqa: BLE001
        rec["finite"] = False
        rec.setdefault("error", f"no audio: {e}")
    return rec


def verdict(rec, max_unconverged):
    if "error" in rec and "solves" not in rec:
        return "FAIL", rec["error"]
    if rec.get("divergences", 0) > 0:
        return "FAIL", f"{rec['divergences']} divergence(s)"
    if not rec.get("finite", False):
        return "FAIL", "non-finite or missing output"
    notes = []
    if rec.get("severe", 0) > 0:
        notes.append(f"{rec['severe']} severe solve(s)")
    solves = rec.get("solves", 0)
    if solves and rec.get("unconverged", 0) / solves > max_unconverged:
        notes.append(f"{100.0 * rec['unconverged'] / solves:.4f}% unconverged")
    return ("WARN", ", ".join(notes)) if notes else ("OK", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--input", type=Path, help="excitation to render (default: the config's input)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=256)
    ap.add_argument("--lead-in", type=float, default=2.0, help="as CmBackend's default")
    ap.add_argument("--max-unconverged", type=float, default=1e-4)
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    schx = Path(cfg["schx"]).expanduser()
    circuit = str(schx.with_suffix(".cm.json"))
    in_wav = str(args.input or cfg["input"])
    oversample = cfg.get("oversample", 8)
    fixed = cfg.get("fixed", {})
    knob_ranges = {k: [float(x) for x in v] for k, v in cfg["knobs"].items()}

    corners = _corners(knob_ranges, full_hypercube=None, max_corners=None)
    corners = _sample_interior(knob_ranges, corners, resolve_sample_grid(None, knob_ranges))
    print(f"{len(corners)} corners, oversample {oversample}, input {in_wav}")

    scratch = tempfile.mkdtemp(prefix="cm_stability_")
    jobs = []
    for i, (label, c) in enumerate(corners):
        params = dict(fixed)
        params.update(c)
        jobs.append((i, label, params))

    results = [None] * len(jobs)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(render_corner, circuit, in_wav, scratch, i, p, oversample, args.iterations, args.lead_in): i
                for i, _, p in jobs}
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
    for (i, label, _), rec in zip(jobs, results):
        rec["corner"] = label

    worst = "OK"
    rank = {"OK": 0, "WARN": 1, "FAIL": 2}
    for i, rec in enumerate(results):
        v, note = verdict(rec, args.max_unconverged)
        rec["verdict"], rec["note"] = v, note
        if rank[v] > rank[worst]:
            worst = v
        label = rec["corner"]
        if "solves" in rec:
            print(f"  {v:4} {label:<36} solves {rec['solves']:>9}  unconv {rec['unconverged']:>6}  "
                  f"severe {rec['severe']:>4}  div {rec['divergences']:>3}  peak {rec['peak']:8.3g}  "
                  f"rt {rec['realtime_factor']:.2f}  {note}")
        else:
            print(f"  {v:4} {label:<36} {note}  [rc {rec['rc']}: {rec['stderr_tail'][:160]}]")

    print(f"overall: {worst}")
    if args.output:
        args.output.write_text(json.dumps({"config": str(args.config), "input": in_wav, "oversample": oversample,
                                           "overall": worst, "corners": results}, indent=1))
    sys.exit(1 if worst == "FAIL" else 0)


if __name__ == "__main__":
    main()
