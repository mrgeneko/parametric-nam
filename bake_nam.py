#!/usr/bin/env python3
"""Bake a knob setting from a (parametric) .param.nam into a STANDARD NAM .nam.

Standard NAM plugins have no runtime knob input, so to deliver a specific tone we
freeze the FiLM at a chosen setting → an identical static A2 → an official "WaveNet"
.nam that loads in any NAM plugin. Runs offline on an already-trained parametric file
(pure weight transform; no retrain). See docs/spice_static_plan.md.

  python bake_nam.py --in model.param.nam --params "Gain=0.7,Tone=0.5" -o tone.nam
  python bake_nam.py --in model.param.nam --params "..." --width 8       -o tone.nam

Omitted knobs fall back to each knob's DECLARED default (config.parametric.
parameters[].default) — not a blanket 0.5, which is wrong for many circuits
(e.g. a Timmy's controls don't center at noon).

BY DEFAULT this emits a DUAL-payload file: the baked static tone at top level (so any
stock plugin plays it) PLUS the full original parametric model under the
"[redacted]_parametric" key (which stock loaders ignore, and a parametric-aware host reads
for live knobs). One artifact, best experience where supported — and safe to hand anyone.

That default matters: a RAW .param.nam handed to a stock plugin throws
("No config parser registered for architecture: ParametricWaveNet") — verified against
upstream NeuralAmpModelerCore. A dual file never does. Pass --no-embed-parametric to emit
a static-only file (for capture-pack members, where embedding the master in every tone
just multiplies size).

HEAD MODE: a baked .nam is played with a SKIP-accumulating head everywhere it can run
(stock NAM plugins, nam_wavenet.metal, [redacted]'s static a2_fast path). So only a
SKIP-trained source bakes faithfully. This tool reads config.parametric.head_mode and
REFUSES a residual source by default (it would load but sound wrong, corr ~0.31);
retrain with `param_train.py --head-mode skip`, or pass --allow-unfaithful for A/B.
"""
import argparse
import json
from pathlib import Path

from param_train import ParametricA2, check_parametric_schema
import nam_standard

EMBED_KEY = "[redacted]_parametric"


def _submodels(nam: dict, source: str = ".param.nam"):
    """Per-submodel (channels, num_params, param_metas, weights, head_mode) for a
    ParametricWaveNet or SlimmableContainer .param.nam. param_metas is the list of
    {name,min,max,default} dicts as declared in the file. head_mode defaults to
    "residual" for pre-tagging files."""
    entries = ([s["model"] for s in nam["config"]["submodels"]]
               if nam.get("architecture") == "SlimmableContainer" else [nam])
    out = []
    for m in entries:
        cfg = m["config"]
        par = cfg.get("parametric", {}) or {}
        check_parametric_schema(par, source=source)
        metas = par.get("parameters", [])
        out.append((int(cfg["layers"]),
                    int(par.get("condition_size", len(metas))),
                    metas, m["weights"],
                    par.get("head_mode", "residual")))
    return out


def _knob_default(meta: dict) -> float:
    """A knob's declared default; else the midpoint of its range; else 0.5."""
    if "default" in meta and meta["default"] is not None:
        return float(meta["default"])
    lo, hi = meta.get("min"), meta.get("max")
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2.0
    return 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="parametric .param.nam")
    ap.add_argument("--params", default="", help="knob=value,... (values as trained, [0,1]); "
                                                 "omitted knobs use their declared default")
    ap.add_argument("--width", type=int, default=None, help="tier width to bake (default: widest)")
    ap.add_argument("-o", "--out", required=True, type=Path, help="output standard .nam")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--embed-parametric", action=argparse.BooleanOptionalAction, default=True,
                    help=f"embed the full parametric model under '{EMBED_KEY}' (DEFAULT). The "
                         "result is safe to hand anyone: a stock plugin plays the baked tone and "
                         "ignores the extra key, while a parametric-aware host unlocks live knobs. "
                         "Use --no-embed-parametric for capture-pack members, where embedding the "
                         "master in every tone just multiplies the file size.")
    ap.add_argument("--allow-unfaithful", action="store_true",
                    help="bake a non-skip (residual) model anyway. The output will LOAD "
                         "in stock plugins but SOUND WRONG. A/B and experiments only.")
    a = ap.parse_args()

    nam = json.loads(a.inp.read_text())
    subs = _submodels(nam, source=str(a.inp))
    widths = sorted(c for c, *_ in subs)
    target = a.width if a.width is not None else widths[-1]
    match = next((s for s in subs if s[0] == target), None)
    if match is None:
        ap.error(f"width {target} not found; available: {widths}")
    channels, num_params, metas, weights, head_mode = match
    names = [m["name"] for m in metas]

    # A baked standard .nam is played with a SKIP-accumulating head EVERYWHERE it can
    # run (stock NAM plugins, nam_wavenet.metal, and [redacted]'s static a2_fast path).
    # So a residual-trained model bakes to a file that LOADS but sounds WRONG (measured
    # corr ~0.31 vs the real model). Refuse by default rather than ship a silent dud.
    if head_mode != "skip" and not a.allow_unfaithful:
        ap.error(
            f"source head_mode='{head_mode}' cannot be baked faithfully.\n"
            f"  A baked .nam is run with a skip-accumulating head, but this model was "
            f"trained '{head_mode}'.\n"
            f"  It would load and sound WRONG (corr ~0.31).\n"
            f"  Fix: retrain with `param_train.py --head-mode skip`.\n"
            f"  Override (A/B or experiments only): --allow-unfaithful")

    a2 = ParametricA2(channels=channels, num_params=num_params)
    a2.load_weights(weights)

    kv = dict(x.split("=", 1) for x in a.params.split(",") if "=" in x)
    unknown = [k for k in kv if k not in names]
    if unknown:
        ap.error(f"unknown knob(s) {unknown}; model knobs are {names}")
    # Per-knob: user override if given, else the DECLARED default for this circuit.
    vec = ([float(kv[m["name"]]) if m["name"] in kv else _knob_default(m)
            for m in metas] if num_params else None)

    out = nam_standard.export_nam_standard(a2, params=vec, sample_rate=a.sample_rate,
                                           metadata=nam.get("metadata"))

    baked = dict(zip(names, vec)) if vec else "static"
    md = out.setdefault("metadata", {}) or {}
    md["baked_setting"] = baked                    # self-describing: which knob values
    md["baked_from_head_mode"] = head_mode         # provenance for the faithfulness check
    if a.embed_parametric:
        out[EMBED_KEY] = nam                       # verbatim original → host loads live knobs
        md["parametric_embedded"] = True
    out["metadata"] = md

    a.out.write_text(json.dumps(out, separators=(",", ":")))
    defaulted = [n for n in names if n not in kv]
    dflt_note = "" if not vec else f"  (defaulted: {defaulted or 'none'})"
    embed_note = (f"  +embedded parametric ('{EMBED_KEY}') — loads in stock AND parametric hosts"
                  if a.embed_parametric else "  [static-only: --no-embed-parametric]")
    print(f"baked width={channels}ch  head={head_mode}  params={baked}{dflt_note}"
          f"{embed_note}  ->  {a.out}")
    if head_mode != "skip":
        print("  WARNING: --allow-unfaithful — this baked tone will SOUND WRONG in any "
              "stock plugin.")


if __name__ == "__main__":
    main()
