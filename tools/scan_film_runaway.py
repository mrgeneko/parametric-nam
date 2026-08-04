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

  # full trained grid instead of the reduced corner set (see docs/film_runaway_investigation.md
  # for measured cost: the default reduced set is ~2 min for 13 corners on a 190s reference;
  # --config scans the FULL grid batched, e.g. ~2-3 min for Tweed's 972 permutations at the
  # default --batch-size, vs. hours if it re-used the old one-forward-call-per-corner loop):
  python tools/scan_film_runaway.py --nam PATH/TO/model.param.nam \
      --config configs/tweed-5f6-a-full-sag.toml
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from param_train import ParametricA2
from run_pipeline import load_config
from batch_harness import grid_permutations

SR = 48000


def load_widest_submodel(nam_path: str):
    d = json.loads(Path(nam_path).read_text())
    subs = d["config"]["submodels"]
    sub = subs[-1]  # widest tier (SlimmableContainer stores ascending by max_value)
    channels = sub["model"]["config"]["layers"]
    parametric = sub["model"]["config"]["parametric"]
    param_metas = parametric["parameters"]
    param_names = [p["name"] for p in param_metas]
    weights = sub["model"]["weights"]
    # film_gamma_bound is a forward-pass formula parameter (docs/film_runaway_investigation.md,
    # "A1"), not a weight -- must be read back from the export or this reconstruction would
    # silently apply the wrong (unbounded) gamma to a model actually trained bounded, which
    # would corrupt exactly the thing this scanner exists to check. 0.0 (off) for older exports.
    film_gamma_bound = float(parametric.get("film_gamma_bound", 0.0))
    model = ParametricA2(channels=channels, num_params=len(param_names),
                         film_gamma_bound=film_gamma_bound)
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


def full_grid_corners(config_path, param_names):
    """The FULL Cartesian product of --config's own [knobs] grid -- the exact permutations
    batch_harness.py rendered for training (e.g. 972 for Tweed), not a reduced sample.
    Needs --config because a .nam's own metadata only carries min/max/default per knob
    (config.parametric.parameters), not the discrete grid VALUES actually trained on --
    that's a batch_harness.py/config.toml-level fact this tool has no other way to recover.
    """
    cfg = load_config(Path(config_path))
    knob_ranges = {}
    for entry in cfg.get("ranges", []):
        name, vals = entry.split("=", 1)
        knob_ranges[name.strip()] = [float(v) for v in vals.split(",")]
    missing = [n for n in param_names if n not in knob_ranges]
    if missing:
        raise SystemExit(f"--config's [knobs] is missing {missing} (has: {list(knob_ranges)}) "
                         f"-- the .nam's own parameters are {param_names}")
    perms = grid_permutations(param_names, knob_ranges)  # dicts keyed by param_names already
    return [(",".join(f"{n}={p[n]:g}" for n in param_names), [p[n] for n in param_names])
            for p in perms]


def _batched(seq, n):
    it = iter(seq)
    while batch := list(islice(it, n)):
        yield batch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nam", required=True)
    ap.add_argument("--reference", default=str(Path.home() / "Downloads/[redacted]-sweep-v3.wav"))
    ap.add_argument("--chunk-s", type=float, default=5.0)
    ap.add_argument("--flag-ratio", type=float, default=8.0,
                    help="flag a window if its peak exceeds this multiple of the model's own median peak")
    ap.add_argument("--flag-abs", type=float, default=3.0,
                    help="...AND exceeds this absolute volts floor (avoids flagging near-silent models)")
    ap.add_argument("--config", default=None,
                    help="per-circuit TOML -- if given, scan the FULL trained grid (exact "
                         "permutations batch_harness.py rendered, e.g. 972 for Tweed) instead "
                         "of the reduced hypercube corner set. Needs --batch-size's batching "
                         "to stay fast at that scale -- see the module docstring for measured cost.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="corners per batched forward call (default: %(default)s). Batching "
                         "across corners (not just chunks) is what makes --config's full-grid "
                         "mode tractable -- one forward call per (chunk, batch-of-corners) pair "
                         "instead of one per (chunk, corner). PROPERLY INVESTIGATED 2026-08-04 "
                         "after an earlier fix attempt (thread-pool 'wave' teardown) turned out "
                         "to be treating the wrong mechanism -- see the long comment at this "
                         "module's job-scheduling loop for the full writeup. Short version: this "
                         "was never an unbounded leak, it's a genuinely large per-job working set "
                         "(batch_size x chunk_s x SR-sized tensors, x23 WaveNet layers) that at "
                         "the old default (64) x 14 workers pushed close enough to a 36GB "
                         "machine's physical RAM to trigger macOS memory-pressure pathology "
                         "(looks like unbounded growth, isn't). Confirmed via a clean, complete, "
                         "isolated repro: workers=14 batch_size=8 ran all 60/60 test iterations "
                         "at a rock-stable ~16GB peak (this module's own peak-RSS tracking never "
                         "moved off its first-reached value the entire run) -- vs. batch_size=64 "
                         "which climbed unboundedly past 34GB on the exact same machine/model. "
                         "Raise this only if you have headroom to spare -- workers x batch_size "
                         "is the number that matters, not either alone.")
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent (chunk, corner-batch) forward calls (default: cpu_count). "
                         "This model is too narrow (few channels) for PyTorch's own intra-op "
                         "threading to bother splitting a single conv1d across cores -- confirmed "
                         "empirically (2026-08-02): a single scan used ~1.3 of 14 cores despite "
                         "torch.get_num_threads()==10. Running independent forward calls "
                         "concurrently across threads works instead, because PyTorch's C++ conv/ "
                         "matmul kernels release the GIL during compute -- this is the same 'many "
                         "small independent ops, not one big one' shape as batch_harness.py's "
                         "render_many() ThreadPoolExecutor pattern, just for inference instead of "
                         "subprocesses. torch.set_num_threads(1) below is required alongside this: "
                         "without it, each of --workers concurrent calls would ALSO try to spawn "
                         "its own intra-op threads, oversubscribing the machine.")
    args = ap.parse_args()

    workers = args.workers or os.cpu_count() or 4
    torch.set_num_threads(1)  # see --workers help: avoid oversubscribing against the thread pool

    model, param_names, channels = load_widest_submodel(args.nam)
    print(f"{Path(args.nam).name}: channels={channels} params={param_names}")

    x, sr = sf.read(args.reference, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    if sr != SR:
        raise SystemExit(f"reference sr {sr} != {SR}")

    corners = (full_grid_corners(args.config, param_names) if args.config
              else hypercube_corners(param_names))
    chunk_n = int(args.chunk_s * SR)
    n_chunks = len(x) // chunk_n
    n_batches = -(-len(corners) // args.batch_size)  # ceil
    print(f"  scanning {len(corners)} corners x {n_chunks} chunks "
          f"({n_batches} corner-batches/chunk, batch-size={args.batch_size}, workers={workers})"
          f"{' [full grid]' if args.config else ' [reduced hypercube]'}")

    def score(job):
        c, batch = job
        seg = x[c * chunk_n:(c + 1) * chunk_n]
        B = len(batch)
        inp = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)  # [B,1,T]
        cond = torch.tensor([vals for _, vals in batch], dtype=torch.float32)          # [B,num_params]
        with torch.no_grad():
            pred = model(inp, cond)                                                    # [B,1,T]
        pk = pred.abs().amax(dim=(1, 2))                                                # [B]
        return [(float(pk[i]), label, c) for i, (label, _) in enumerate(batch)]

    jobs = [(c, batch) for c in range(n_chunks) for batch in _batched(corners, args.batch_size)]
    # PROPERLY INVESTIGATED 2026-08-04 (see --batch-size's help above for the short version).
    # This used to tear down and recreate the ThreadPoolExecutor every 20 jobs ("wave"
    # teardown), on a theory that OS-level per-thread allocator caching was compounding over
    # a long-lived pool's lifetime. That fix was a plausible-sounding GUESS, explicitly never
    # confirmed with a real memory profiler, and it turned out to be wrong: it only partially
    # bounded growth (still climbed to 20GB by t=400s on a 266-job scan), and a SMALLER wave
    # size -- which the "more teardown = more relief" theory predicted should help MORE --
    # instead made things WORSE (25.6GB within 20s), which should have been the signal the
    # theory was broken, not something to keep tuning.
    #
    # Properly isolated with two controlled, non-live repros instead of more tuning attempts:
    # (1) the SAME workload run single-threaded (no ThreadPoolExecutor at all) stayed stable,
    # ruling out anything Python-reference-related (no_grad already ensures no autograd graph
    # retention; the returned `results` only ever holds plain floats/strings, never tensors).
    # (2) workers=14 at the OLD batch_size=64 vs. a SMALLER batch_size=8 (same worker count,
    # same total iterations, same model): batch_size=8 ran clean and FLAT the entire way (see
    # --batch-size's help) while batch_size=64 is what produced the original unbounded-looking
    # climb past 34GB. Varying batch_size (not worker count, not teardown frequency) was the
    # actual controlling variable -- this was never a leak, it's a genuinely large per-job
    # working set that only becomes pathological once workers x batch_size pushes close enough
    # to the machine's physical RAM to trigger macOS memory-pressure behavior (which looks like
    # runaway growth from the outside, but is fundamentally bounded per-job, not compounding).
    # Fixed at the source (--batch-size's default) instead of papering over it with executor
    # churn -- a single plain ThreadPoolExecutor is both simpler and, per the repro above,
    # exactly as safe as the wave version once batch_size is sane.
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for batch_results in ex.map(score, jobs):
            results.extend(batch_results)
            done += 1
            if len(jobs) > 10 and done % max(1, len(jobs) // 10) == 0:
                print(f"    {done}/{len(jobs)} corner-batches done")

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
