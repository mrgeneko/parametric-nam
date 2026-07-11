#!/usr/bin/env python3
"""Bake a knob setting from a (parametric) .param.nam into a STANDARD NAM .nam.

Standard NAM plugins have no runtime knob input, so to deliver a specific tone we
freeze the FiLM at a chosen setting → an identical static A2 → an official "WaveNet"
.nam that loads in any NAM plugin. Runs offline on an already-trained parametric file
(pure weight transform; no retrain). See docs/spice_static_plan.md.

  python bake_nam.py --in model.param.nam --params "Gain=0.7,Tone=0.5" -o tone.nam
  python bake_nam.py --in model.param.nam --params "..." --width 8       -o tone.nam
"""
import argparse
import json
from pathlib import Path

from param_train import ParametricA2
import nam_standard


def _submodels(nam: dict):
    """(channels, num_params, param_names, weights) per submodel of a
    ParametricWaveNet or SlimmableContainer .param.nam."""
    entries = ([s["model"] for s in nam["config"]["submodels"]]
               if nam.get("architecture") == "SlimmableContainer" else [nam])
    out = []
    for m in entries:
        cfg = m["config"]
        par = cfg.get("parametric", {}) or {}
        names = [p["name"] for p in par.get("parameters", [])]
        out.append((int(cfg["layers"]),
                    int(par.get("condition_size", len(names))),
                    names, m["weights"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="parametric .param.nam")
    ap.add_argument("--params", default="", help="knob=value,... (values as trained, [0,1]); "
                                                 "omitted knobs default to 0.5")
    ap.add_argument("--width", type=int, default=None, help="tier width to bake (default: widest)")
    ap.add_argument("-o", "--out", required=True, type=Path, help="output standard .nam")
    ap.add_argument("--sample-rate", type=int, default=48000)
    a = ap.parse_args()

    nam = json.loads(a.inp.read_text())
    subs = _submodels(nam)
    widths = sorted(c for c, *_ in subs)
    target = a.width if a.width is not None else widths[-1]
    match = next((s for s in subs if s[0] == target), None)
    if match is None:
        ap.error(f"width {target} not found; available: {widths}")
    channels, num_params, names, weights = match

    a2 = ParametricA2(channels=channels, num_params=num_params)
    a2.load_weights(weights)

    kv = dict(x.split("=", 1) for x in a.params.split(",") if "=" in x)
    unknown = [k for k in kv if k not in names]
    if unknown:
        ap.error(f"unknown knob(s) {unknown}; model knobs are {names}")
    vec = [float(kv.get(n, 0.5)) for n in names] if num_params else None

    out = nam_standard.export_nam_standard(a2, params=vec, sample_rate=a.sample_rate,
                                           metadata=nam.get("metadata"))
    a.out.write_text(json.dumps(out, separators=(",", ":")))
    baked = dict(zip(names, vec)) if vec else "static"
    print(f"baked width={channels}ch  params={baked}  ->  {a.out}")


if __name__ == "__main__":
    main()
