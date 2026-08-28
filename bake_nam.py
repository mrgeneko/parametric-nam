#!/usr/bin/env python3
"""Bake a knob setting from a (parametric) .param.nam into a STANDARD NAM .nam.

Standard NAM plugins have no runtime knob input, so to deliver a specific tone we
freeze the FiLM at a chosen setting → an identical static A2 → an official "WaveNet"
.nam that loads in any NAM plugin. Runs offline on an already-trained parametric file
(pure weight transform; no retrain). See internal engineering notes.

  python bake_nam.py --in model.param.nam --params "Gain=0.7,Tone=0.5" -o tone.nam
  python bake_nam.py --in model.param.nam --params "..." --width 8       -o tone.nam
  python bake_nam.py --in model.param.nam --params "Gain=0.7" -o packdir/   # auto-named
  python bake_nam.py --in model.param.nam --params "Gain=0.7"              # -> beside the input

Omitted knobs fall back to each knob's DECLARED default (config.parametric.
parameters[].default) — not a blanket 0.5, which is wrong for many circuits
(e.g. a Timmy's controls don't center at noon).

BY DEFAULT this emits a DUAL-payload file: the baked static tone at top level (so any
stock plugin plays it) PLUS the full original parametric model under the
"embedded_parametric" key (which stock loaders ignore, and a parametric-aware host reads
for live knobs). One artifact, best experience where supported — and safe to hand anyone.

That default matters: a RAW .param.nam handed to a stock plugin throws
("No config parser registered for architecture: ParametricWaveNet") — verified against
upstream NeuralAmpModelerCore. A dual file never does. Pass --no-embed-parametric to emit
a static-only file (for capture-pack members, where embedding the master in every tone
just multiplies size).

SKIP-ONLY: a baked .nam is played with a skip-accumulating head everywhere it can run
(stock NAM plugins, nam_wavenet.metal, the host app's static a2_fast path), which is the only
head we support. A legacy residual/untagged source is REJECTED (its weights would sound wrong,
corr ~0.31) — retrain it.
"""
import argparse
import json
from pathlib import Path

from param_train import ParametricA2, check_parametric_schema
import nam_standard

EMBED_KEY = "embedded_parametric"


def _submodels(nam: dict, source: str = ".param.nam"):
    """Per-submodel (channels, num_params, param_metas, weights) for a ParametricWaveNet
    or SlimmableContainer .param.nam. check_parametric_schema rejects legacy
    residual/untagged models, so anything we return here is a supported skip model.
    """
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


def _model_stem(inp: Path) -> str:
    """'timmy_v4_optimal.param.nam' -> 'timmy_v4_optimal' (strip the DOUBLE extension)."""
    n = inp.name
    for suf in (".param.nam", ".nam"):
        if n.endswith(suf):
            return n[: -len(suf)]
    return inp.stem


def auto_name(inp: Path, baked: dict, channels=None) -> str:
    """Encode the RESOLVED setting into the filename.

    Resolved, not the user's --params string: knobs left out fall back to their declared
    default, so a name built from the raw argument would omit knobs that were nonetheless
    baked in -- and two different settings could collide on one name.

    Values are formatted with %g, so 0.50 -> '0.5' and two spellings of the same setting give
    the same file rather than two. The tier width is included only when the caller pinned one,
    since tiers are genuinely different models and a pack mixing them would otherwise collide.

        timmy_v4_optimal_Gain0.55_Bass0.5_Treble0.2.nam
    """
    parts = [_model_stem(inp)]
    if channels is not None:
        parts.append(f"w{channels}")
    parts += [f"{k}{v:g}" for k, v in baked.items()]
    return "_".join(parts) + ".nam"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="parametric .param.nam")
    ap.add_argument("--params", default="", help="knob=value,... (values as trained, [0,1]); "
                                                 "omitted knobs use their declared default")
    ap.add_argument("--width", type=int, default=None, help="tier width to bake (default: widest)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output standard .nam. A DIRECTORY (or omitted, meaning the input's "
                         "own directory) auto-names the file from the RESOLVED knob values -- "
                         "see auto_name(). The caller already stated the setting; making every "
                         "caller reinvent a filename for it is how one pack ends up with ten "
                         "naming conventions, and how a name drifts from the setting it claims.")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--embed-parametric", action=argparse.BooleanOptionalAction, default=True,
                    help=f"embed the full parametric model under '{EMBED_KEY}' (DEFAULT). The "
                         "result is safe to hand anyone: a stock plugin plays the baked tone and "
                         "ignores the extra key, while a parametric-aware host unlocks live knobs. "
                         "Use --no-embed-parametric for capture-pack members, where embedding the "
                         "master in every tone just multiplies the file size.")
    a = ap.parse_args()

    nam = json.loads(a.inp.read_text())
    subs = _submodels(nam, source=str(a.inp))
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

    dest = a.out
    if dest is None or dest.is_dir():
        base = dest if dest is not None else a.inp.parent
        dest = base / auto_name(a.inp, baked if isinstance(baked, dict) else {},
                                channels if a.width is not None else None)
    md = out.setdefault("metadata", {}) or {}
    md["baked_setting"] = baked                    # self-describing: which knob values
    if a.embed_parametric:
        out[EMBED_KEY] = nam                       # verbatim original → host loads live knobs
        md["parametric_embedded"] = True
    out["metadata"] = md

    dest.write_text(json.dumps(out, separators=(",", ":")))
    defaulted = [n for n in names if n not in kv]
    dflt_note = "" if not vec else f"  (defaulted: {defaulted or 'none'})"
    embed_note = (f"  +embedded parametric ('{EMBED_KEY}') — loads in stock AND parametric hosts"
                  if a.embed_parametric else "  [static-only: --no-embed-parametric]")
    print(f"baked width={channels}ch  params={baked}{dflt_note}{embed_note}  ->  {dest}")


if __name__ == "__main__":
    main()
