#!/usr/bin/env python3
"""
Run inference with a .param.nam model.

Usage:
    # Single run
    python infer.py --model poc_a2.param.nam --input limelight.wav --output out.wav --gain2 0.7

    # Sweep
    python infer.py --model poc_a2.param.nam --input limelight.wav \
        --sweep 0.1,0.3,0.5,0.7,0.9 --out-dir /tmp/sweep_out
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from param_train import ParametricA2

CHUNK = 131072  # process in chunks to avoid MPS memory pressure


def load_model(nam_path: str, device: torch.device, quality: str = "full"):
    data = json.loads(Path(nam_path).read_text())
    arch = data["architecture"]

    if arch == "SlimmableParametricContainer":
        # Pick submodel by quality: "lite" → max_value<=0.5, "full" → max_value>0.5
        submodels = data["config"]["submodels"]
        if quality == "lite":
            sm = next(s for s in submodels if s["max_value"] <= 0.5)
        else:
            sm = next(s for s in reversed(submodels) if s["max_value"] > 0.5)
        model_data = sm["model"]
        print(f"Loaded: SlimmableParametricContainer [{quality}] "
              f"(max_value={sm['max_value']})", file=sys.stderr)
    elif arch == "ParametricWaveNet":
        model_data = data
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    cfg = model_data["config"]
    channels = cfg["layers"]
    num_params = cfg["parametric"]["condition_size"]
    model = ParametricA2(channels, num_params)
    model.load_weights(model_data["weights"])
    model.to(device).eval()
    param_defs = cfg["parametric"]["parameters"]
    print(f"  A2 {channels}ch, params={[p['name'] for p in param_defs]}", file=sys.stderr)
    return model, param_defs


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


def run_one(model, param_defs, audio, sr, param_values, out_path, device):
    result = process(model, audio, param_values, device)
    sf.write(out_path, result, sr, subtype="FLOAT")
    labels = ", ".join(f"{p['name']}={v}" for p, v in zip(param_defs, param_values))
    print(f"  {labels} → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True)
    ap.add_argument("--input",   required=True)
    ap.add_argument("--gain2",   type=float, default=None)
    ap.add_argument("--output",  default=None)
    ap.add_argument("--sweep",   default=None, help="Comma-separated gain_2 values")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quality", choices=["lite", "full"], default="full",
                    help="For SlimmableParametricContainer: which submodel to use (default: full)")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, param_defs = load_model(args.model, device, quality=args.quality)

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"Input: {args.input} ({len(audio)/sr:.2f}s @ {sr}Hz)", file=sys.stderr)

    if args.sweep:
        values = [float(v) for v in args.sweep.split(",")]
        out_dir = Path(args.out_dir or Path(args.input).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        for g2 in values:
            out_path = out_dir / f"{stem}_gain2_{g2:.2f}.wav"
            run_one(model, param_defs, audio, sr, [g2], str(out_path), device)
    else:
        if args.gain2 is None or args.output is None:
            ap.error("Provide --gain2 and --output, or use --sweep and --out-dir")
        run_one(model, param_defs, audio, sr, [args.gain2], args.output, device)


if __name__ == "__main__":
    main()
