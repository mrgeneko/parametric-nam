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

--embed-parametric emits a DUAL-payload file: the baked static tone at top level
(so any stock plugin plays it) PLUS the full original parametric model under the
"[redacted]_parametric" key (which stock loaders ignore, and a parametric-aware host
reads for live knobs). One artifact, best experience where supported. NOTE: the
stock-faithful top-level requires a SKIP-trained source (see --head-mode / the
rearchitecture doc); a residual-trained source bakes a top-level that loads but
sounds wrong under stock's skip-accumulating head.
"""
import argparse
import json
from pathlib import Path

from param_train import ParametricA2
import nam_standard

EMBED_KEY = "[redacted]_parametric"


def _submodels(nam: dict):
    """Per-submodel (channels, num_params, param_metas, weights) for a
    ParametricWaveNet or SlimmableContainer .param.nam. param_metas is the list of
    {name,min,max,default} dicts as declared in the file."""
    entries = ([s["model"] for s in nam["config"]["submodels"]]
               if nam.get("architecture") == "SlimmableContainer" else [nam])
    out = []
    for m in entries:
        cfg = m["config"]
        par = cfg.get("parametric", {}) or {}
        metas = par.get("parameters", [])
        out.append((int(cfg["layers"]),
                    int(par.get("condition_size", len(metas))),
                    metas, m["weights"]))
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
    ap.add_argument("--embed-parametric", action="store_true",
                    help="also embed the full parametric model under "
                         f"'{EMBED_KEY}' (stock plays the baked tone; a parametric-aware "
                         "host unlocks live knobs). Requires a skip-trained source to be "
                         "stock-faithful.")
    a = ap.parse_args()

    nam = json.loads(a.inp.read_text())
    subs = _submodels(nam)
    widths = sorted(c for c, *_ in subs)
    target = a.width if a.width is not None else widths[-1]
    match = next((s for s in subs if s[0] == target), None)
    if match is None:
        ap.error(f"width {target} not found; available: {widths}")
    channels, num_params, metas, weights = match
    names = [m["name"] for m in metas]

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
    if a.embed_parametric:
        out[EMBED_KEY] = nam                       # verbatim original → host loads live knobs
        md["parametric_embedded"] = True
    out["metadata"] = md

    a.out.write_text(json.dumps(out, separators=(",", ":")))
    defaulted = [n for n in names if n not in kv]
    dflt_note = "" if not vec else f"  (defaulted: {defaulted or 'none'})"
    embed_note = f"  +embedded parametric ('{EMBED_KEY}')" if a.embed_parametric else ""
    print(f"baked width={channels}ch  params={baked}{dflt_note}{embed_note}  ->  {a.out}")
    if a.embed_parametric:
        print("  note: top-level baked tone is stock-faithful ONLY if the source was "
              "skip-trained.")


if __name__ == "__main__":
    main()
