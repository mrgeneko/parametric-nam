#!/usr/bin/env python3
"""
param_infer.py — Run inference with a trained parametric NAM model.

Usage:
  python param_infer.py --checkpoint best.pt --input audio.wav --output-dir /path/to/out \
    --params "volume=0.5,mid=0.5,treble=0.5,middle=0.5,bass=0.5,clean_master=0.5" \
    --params "volume=0.8,mid=0.5,treble=0.7,middle=0.5,bass=0.3,clean_master=0.7"
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Import model classes from param_train
sys.path.insert(0, str(Path(__file__).parent))
from param_train import SlimmableParametricA2, ParametricA2


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = ckpt["args_dict"]
    # Default True, not False: param_train.py no longer has a --slimmable flag
    # (it's unconditional -- see the "drop the non-slimmable path" commit), so
    # checkpoints saved after that change have no "slimmable" key in args_dict
    # at all. Older checkpoints still have an explicit key and use it as-is.
    slimmable = args_dict.get("slimmable", True)

    # Load param_names from the dataset config
    dataset_dir = Path(str(args_dict["dataset"]))
    with open(dataset_dir / "config.json") as f:
        config = json.load(f)
    param_names = config["knobs"]
    num_params = len(param_names)
    args_dict["param_names"] = param_names

    if slimmable:
        model = SlimmableParametricA2(num_params=num_params, widths=args_dict.get("widths"))
    else:
        channels = args_dict.get("channels", 8)
        model = ParametricA2(num_params=num_params, channels=channels)

    state = ckpt.get("model") or ckpt.get("best_state")
    model.load_state_dict(state)
    model.eval()
    return model, args_dict


def run_inference(model, audio: np.ndarray, param_vec: list[float],
                  device: str, chunk_size: int = 48000 * 10) -> np.ndarray:
    model.to(device)
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
    params_t = torch.tensor([param_vec], dtype=torch.float32).to(device)            # [1,N]

    with torch.no_grad():
        if isinstance(model, SlimmableParametricA2):
            out = model(audio_t, params_t)[-1]   # use full (widest) tier output
        else:
            out = model(audio_t, params_t)

    return out.squeeze().cpu().numpy()


def parse_params(s: str, param_names: list[str]) -> list[float]:
    kv = dict(p.split("=") for p in s.split(","))
    return [float(kv[n]) for n in param_names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--params", action="append", default=[],
                    help="knob=value,... (repeat for multiple outputs)")
    ap.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available()          # NVIDIA or AMD/ROCm
        else "mps" if torch.backends.mps.is_available()
        else "cpu"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model, args_dict = load_model(args.checkpoint)
    param_names = args_dict["param_names"]
    print(f"  Params: {param_names}", flush=True)

    print(f"Loading input {args.input} ...", flush=True)
    audio, sr = sf.read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # Normalize to -18 dBFS (same as training)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    target_rms = 10 ** (-18.0 / 20.0)
    scale = target_rms / (rms + 1e-8)
    audio_norm = audio * scale

    if not args.params:
        # Default: a few representative combinations
        defaults = {n: 0.5 for n in param_names}
        combos = [
            {**defaults},
            {**defaults, param_names[0]: 0.2},
            {**defaults, param_names[0]: 0.8},
        ]
        param_strs = [",".join(f"{k}={v}" for k, v in c.items()) for c in combos]
    else:
        param_strs = args.params

    for param_str in param_strs:
        vec = parse_params(param_str, param_names)
        label = param_str.replace(",", "_").replace("=", "")
        out_path = out_dir / f"out_{label}.wav"

        print(f"  Running {param_str} ...", flush=True)
        out_audio = run_inference(model, audio_norm, vec, args.device)

        # Scale back (undo normalization)
        out_audio = out_audio / scale
        sf.write(str(out_path), out_audio.astype(np.float32), sr, subtype="FLOAT")
        print(f"  → {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
