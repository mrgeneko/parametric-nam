#!/usr/bin/env python3
"""
nam_infer.py — Run inference directly from an exported .param.nam file, no training
checkpoint needed. See checkpoint_infer.py for the equivalent that loads from a .pt
training checkpoint instead (and supports multiple knob-setting combinations per run).

Usage:
    # One or more named knobs; anything omitted falls back to its declared default
    # (same convention as bake_nam.py's --params, not a blanket 0.5)
    python nam_infer.py --model model.param.nam --input limelight.wav --output out.wav \
        --params "Gain=0.7,Tone=0.5"

    # Sweep one knob, holding any others at --params (or their declared default)
    python nam_infer.py --model model.param.nam --input limelight.wav \
        --sweep-param Gain --sweep 0.1,0.3,0.5,0.7,0.9 --out-dir /tmp/sweep_out
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from param_train import ParametricA2, check_parametric_schema

CHUNK = 131072  # process in chunks to avoid MPS memory pressure


def load_model(nam_path: str, device: torch.device, quality: str = "full"):
    data = json.loads(Path(nam_path).read_text())
    arch = data["architecture"]

    if arch == "SlimmableContainer":
        # Pick submodel by quality: "lite" → max_value<=0.5, "full" → max_value>0.5
        submodels = data["config"]["submodels"]
        if quality == "lite":
            sm = next(s for s in submodels if s["max_value"] <= 0.5)
        else:
            sm = next(s for s in reversed(submodels) if s["max_value"] > 0.5)
        model_data = sm["model"]
        print(f"Loaded: SlimmableContainer [{quality}] "
              f"(max_value={sm['max_value']})", file=sys.stderr)
    elif arch == "ParametricWaveNet":
        model_data = data
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    cfg = model_data["config"]
    channels = cfg["layers"]
    par = cfg["parametric"]
    check_parametric_schema(par, source=str(nam_path))   # rejects legacy/residual models
    num_params = par["condition_size"]
    model = ParametricA2(channels, num_params)
    model.load_weights(model_data["weights"])
    model.to(device).eval()
    param_defs = cfg["parametric"]["parameters"]
    print(f"  A2 {channels}ch, params={[p['name'] for p in param_defs]}",
          file=sys.stderr)
    return model, param_defs


def _knob_default(meta: dict) -> float:
    """A knob's declared default; else the midpoint of its range; else 0.5.

    Same convention as bake_nam.py's `_knob_default` -- kept as its own small copy here
    rather than importing a private helper cross-module."""
    if "default" in meta and meta["default"] is not None:
        return float(meta["default"])
    lo, hi = meta.get("min"), meta.get("max")
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2.0
    return 0.5


def resolve_params(params_str: str, param_defs: list[dict]) -> dict[str, float]:
    """--params "Name=val,..." -> {every declared knob's name: resolved value}. A knob
    left out of params_str falls back to its declared default (_knob_default above),
    not a blanket 0.5 -- wrong for many circuits (e.g. a knob that doesn't center at
    noon). Raises on an unrecognized knob name, same "refuse to guess" convention used
    elsewhere in this toolchain."""
    kv = dict(p.split("=", 1) for p in params_str.split(",") if "=" in p)
    names = {p["name"] for p in param_defs}
    unknown = set(kv) - names
    if unknown:
        raise ValueError(f"unrecognized knob name(s) {sorted(unknown)} -- "
                          f"this model's knobs are {sorted(names)}")
    return {p["name"]: (float(kv[p["name"]]) if p["name"] in kv else _knob_default(p))
            for p in param_defs}


def process(model, audio: np.ndarray, param_values: list[float],
            device: torch.device) -> np.ndarray:
    cond = torch.tensor([param_values], dtype=torch.float32, device=device)
    out = np.zeros(len(audio), dtype=np.float32)
    i = 0
    while i < len(audio):
        chunk = torch.from_numpy(audio[i:i+CHUNK]).float().to(device)
        with torch.no_grad():
            pred = model(chunk.unsqueeze(0).unsqueeze(0), cond)
        out[i:i+len(chunk)] = pred.squeeze().cpu().numpy()
        i += CHUNK
    return out


def run_one(model, param_defs, audio, sr, resolved: dict, out_path, device):
    param_values = [resolved[p["name"]] for p in param_defs]
    result = process(model, audio, param_values, device)
    sf.write(out_path, result, sr, subtype="FLOAT")
    labels = ", ".join(f"{p['name']}={resolved[p['name']]}" for p in param_defs)
    print(f"  {labels} → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True)
    ap.add_argument("--input",   required=True)
    ap.add_argument("--params",  default="",
                    help="knob=value,... (values as trained, [0,1]); omitted knobs fall "
                         "back to their declared default (same convention as bake_nam.py)")
    ap.add_argument("--output",  default=None, help="single-run mode")
    ap.add_argument("--sweep-param", default=None,
                    help="knob to sweep --sweep's values over; default: the model's sole "
                         "knob if it has exactly one, else required")
    ap.add_argument("--sweep",   default=None, help="comma-separated values for --sweep-param")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quality", choices=["lite", "full"], default="full",
                    help="For SlimmableParametricContainer: which submodel to use (default: full)")
    args = ap.parse_args()

    # auto-detect GPU: cuda covers both NVIDIA and AMD/ROCm (ROCm exposes AMD GPUs
    # through the torch.cuda API), then Apple MPS, else CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")
    model, param_defs = load_model(args.model, device, quality=args.quality)

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"Input: {args.input} ({len(audio)/sr:.2f}s @ {sr}Hz)", file=sys.stderr)

    if args.sweep:
        sweep_param = args.sweep_param
        if sweep_param is None:
            if len(param_defs) != 1:
                ap.error("--sweep-param is required when the model has more than one knob "
                         f"({[p['name'] for p in param_defs]})")
            sweep_param = param_defs[0]["name"]
        values = [float(v) for v in args.sweep.split(",")]
        out_dir = Path(args.out_dir or Path(args.input).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        for v in values:
            resolved = resolve_params(args.params, param_defs)
            resolved[sweep_param] = v
            out_path = out_dir / f"{stem}_{sweep_param}_{v:.2f}.wav"
            run_one(model, param_defs, audio, sr, resolved, str(out_path), device)
    else:
        if args.output is None:
            ap.error("Provide --output (with --params), or use --sweep-param/--sweep and --out-dir")
        resolved = resolve_params(args.params, param_defs)
        run_one(model, param_defs, audio, sr, resolved, args.output, device)


if __name__ == "__main__":
    main()
