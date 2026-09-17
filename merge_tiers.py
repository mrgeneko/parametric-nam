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
  python merge_tiers.py \
      myamp.composite.param.nam  myamp_w5.best_full.param.nam \
      --out myamp.3-5-8.param.nam

Each input may be a SlimmableContainer (all its submodels are taken) or a bare
ParametricWaveNet (taken as one submodel — e.g. a single-width run's export).
Inputs are read left to right; a later input offering a width already seen is an
error unless --replace is given (then it overrides — e.g. to swap in a better
tier). All tiers must be the same product (version, sample rate, head_mode, param
set) differing only in channel width; mismatches are refused.

MIXED MERGE (path:width): a bare path contributes EVERY submodel it has, so two
multi-tier containers that both offer the same widths can't be combined into "w4
from A, w8 from B" via file order + --replace alone -- with --replace, B's LATER
input simply overrides EVERY width it also offers, not just the one you wanted (a
real sharp edge, found 2026-09-17 fixing a --freeze-tiers case where the "kept"
tier needed to come from the ORIGINAL checkpoint, not the one that just trained
alongside it). Suffix a path with :<width> to take ONLY that one submodel from it:

  # w4 from the just-trained fix, w8 unchanged from the original shipped model
  python merge_tiers.py \
      original.best_full.param.nam:8  fixed.best_lite.param.nam:4 \
      --out patched.param.nam

A bare path (no suffix) still contributes every submodel it has, as before.
"""
import argparse, json, re, sys
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


_INPUT_RE = re.compile(r"^(.*):(\d+)$")


def parse_input(arg: str) -> tuple[str, int | None]:
    """'path' -> (path, None) [every submodel]; 'path:8' -> (path, 8) [only that width]."""
    m = _INPUT_RE.match(arg)
    return (m.group(1), int(m.group(2))) if m else (arg, None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge .param.nam tiers into one SlimmableContainer")
    ap.add_argument("inputs", nargs="+",
                    help=".param.nam files (containers or bare ParametricWaveNet). Suffix "
                         "with :<width> (e.g. model.param.nam:8) to take only that one "
                         "submodel from a multi-tier file -- see module docstring's MIXED "
                         "MERGE section for why bare paths can't express this alone.")
    ap.add_argument("--out", required=True, help="output .param.nam")
    ap.add_argument("--replace", action="store_true",
                    help="let a later input override an earlier same-width tier (default: error)")
    args = ap.parse_args()

    by_width: dict[int, tuple] = {}          # width -> (submodel, source path)
    for raw in args.inputs:
        path, only_width = parse_input(raw)
        nam = json.loads(Path(path).read_text())
        subs = submodels_of(nam, path)
        if only_width is not None:
            subs = [s for s in subs if width_of(s) == only_width]
            if not subs:
                raise SystemExit(f"{path}: no width-{only_width} submodel found "
                                 f"(has: {sorted(width_of(s) for s in submodels_of(nam, path))})")
        for sub in subs:
            w = width_of(sub)
            if w in by_width:
                if not args.replace:
                    raise SystemExit(f"width {w} appears in both {by_width[w][1]} and {path}; "
                                     f"pass --replace to let the later input override, or suffix "
                                     f"an input with :<width> to pick just one tier from it")
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
                             f"(must be 'skip' — the host app rejects other heads)")
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
