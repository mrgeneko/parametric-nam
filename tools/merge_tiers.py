#!/usr/bin/env python3
"""merge_tiers.py — assemble a multi-tier SlimmableContainer .param.nam from the
submodels of one or more existing .param.nam files.

Because slimmable tiers share NO weights, a container is just N independent
ParametricWaveNet submodels selected at runtime by ascending max_value
breakpoints. So building e.g. [3,5,8] from an existing [3,8] plus a separately
trained w5 is pure JSON surgery: collect the submodels, key them by channel
width, sort ascending, RECOMPUTE the max_value breakpoints evenly ((i+1)/N to
match param_train's max_values()), and write a new container. No training, no
PyTorch, and w3/w8 come through byte-for-byte from the file you already shipped.

  # add a separately-trained w5 tier to an existing [3,8] model -> [3,5,8]
  python tools/merge_tiers.py \
      jcm800-2203-preamp.composite.param.nam  jcm800preamp_w5.best_full.param.nam \
      --out jcm800-2203-preamp.3-5-8.param.nam

Each input may be a SlimmableContainer (all its submodels are taken) or a bare
ParametricWaveNet (taken as one submodel — e.g. a single-width run's export).
Inputs are read left to right; a later input offering a width already seen is an
error unless --replace is given (then it overrides — e.g. to swap in a better
tier). All tiers must be the same product (version, sample rate, head_mode, param
set) differing only in channel width; mismatches are refused.
"""
import argparse, json, sys
from pathlib import Path


def submodels_of(nam: dict, src: str) -> list[dict]:
    """Submodel dicts ({max_value, model}) from a .param.nam, container or bare."""
    arch = nam.get("architecture")
    if arch == "SlimmableContainer":
        subs = nam.get("config", {}).get("submodels")
        if not subs:
            raise SystemExit(f"{src}: SlimmableContainer has no submodels")
        return subs
    if arch == "ParametricWaveNet":
        return [{"max_value": 1.0, "model": nam}]   # whole file == one submodel
    raise SystemExit(f"{src}: unsupported architecture {arch!r} "
                     f"(want SlimmableContainer or ParametricWaveNet)")


def width_of(sub: dict) -> int:
    return int(sub["model"]["config"]["layers"])


def par_of(sub: dict) -> dict:
    return sub["model"]["config"].get("parametric", {})


def signature(sub: dict):
    """What must match across tiers — everything but the channel width."""
    m, p = sub["model"], par_of(sub)
    return (m.get("version"), m.get("sample_rate"), p.get("head_mode"),
            p.get("schema_version"), json.dumps(p.get("parameters"), sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge .param.nam tiers into one SlimmableContainer")
    ap.add_argument("inputs", nargs="+", help=".param.nam files (containers or bare ParametricWaveNet)")
    ap.add_argument("--out", required=True, help="output .param.nam")
    ap.add_argument("--replace", action="store_true",
                    help="let a later input override an earlier same-width tier (default: error)")
    args = ap.parse_args()

    by_width: dict[int, tuple] = {}          # width -> (submodel, source path)
    for path in args.inputs:
        nam = json.loads(Path(path).read_text())
        for sub in submodels_of(nam, path):
            w = width_of(sub)
            if w in by_width:
                if not args.replace:
                    raise SystemExit(f"width {w} appears in both {by_width[w][1]} and {path}; "
                                     f"pass --replace to let the later input override")
                print(f"  replacing width-{w} tier: {by_width[w][1]} -> {path}", file=sys.stderr)
            by_width[w] = (sub, path)

    widths = sorted(by_width)
    subs = [by_width[w][0] for w in widths]
    n = len(subs)

    # --- compatibility: every tier must be the SAME product, differing only in width ---
    base_w = widths[-1]
    base_sig = signature(by_width[base_w][0])
    for w in widths:
        sub = by_width[w][0]
        if signature(sub) != base_sig:
            raise SystemExit(
                f"tier w{w} ({by_width[w][1]}) is incompatible with w{base_w} "
                f"({by_width[base_w][1]}): version / sample_rate / head_mode / schema_version / "
                f"parameters must all match.")
        p = par_of(sub)
        if p.get("head_mode") != "skip":
            raise SystemExit(f"tier w{w} ({by_width[w][1]}): head_mode={p.get('head_mode')!r} "
                             f"(must be 'skip' — [redacted] rejects other heads)")
        if sub["model"].get("version") != "0.7.0":
            raise SystemExit(f"tier w{w} ({by_width[w][1]}): version="
                             f"{sub['model'].get('version')!r} (must be 0.7.0)")
        if not sub["model"].get("weights"):
            raise SystemExit(f"tier w{w} ({by_width[w][1]}): no weights")

    # --- recompute even max_value breakpoints ((i+1)/N), matching max_values() ---
    for i, sub in enumerate(subs):
        sub["max_value"] = round((i + 1) / n, 6)

    widest = subs[-1]["model"]
    container = {
        "version": widest.get("version", "0.7.0"),
        "architecture": "SlimmableContainer",
        "config": {"submodels": subs},
        "weights": [],
        "metadata": dict(widest.get("metadata", {})),   # container metadata from widest tier
        "sample_rate": widest.get("sample_rate"),
    }
    Path(args.out).write_text(json.dumps(container, separators=(",", ":")))
    tiers = ", ".join(f"{w}ch@{by_width[w][0]['max_value']}" for w in widths)
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  {n}-tier SlimmableContainer: [{tiers}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
