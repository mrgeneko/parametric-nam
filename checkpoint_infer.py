#!/usr/bin/env python3
"""
checkpoint_infer.py — Run inference from a training checkpoint (.pt) with a trained
parametric NAM model. See nam_infer.py for the equivalent that loads directly from an
exported .param.nam file instead, with no checkpoint needed.

Usage:
  python checkpoint_infer.py --checkpoint best.pt --input audio.wav --output-dir /path/to/out \
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

    # A2 (internal engineering notes): args_dict already captures this from the
    # training CLI (it's saved from vars(args)), so this recovers it for free -- no separate
    # detection needed the way export_checkpoint.py has to (it doesn't get an args_dict when
    # composing across checkpoints from possibly different runs).
    spectral_norm = args_dict.get("spectral_norm", False)
    # Same free recovery as spectral_norm above: --lora-rank is captured in args_dict
    # automatically (the whole argparse namespace is saved verbatim), no extra plumbing.
    lora_rank = args_dict.get("lora_rank", 0)
    if slimmable:
        model = SlimmableParametricA2(num_params=num_params, widths=args_dict.get("widths"),
                                      spectral_norm=spectral_norm, lora_rank=lora_rank)
    else:
        channels = args_dict.get("channels", 8)
        model = ParametricA2(num_params=num_params, channels=channels,
                             spectral_norm=spectral_norm, lora_rank=lora_rank)

    state = ckpt.get("model") or ckpt.get("best_state")
    model.load_state_dict(state)
    model.eval()
    return model, args_dict


def run_inference(model, audio: np.ndarray, param_vec: list[float],
                  device: str, tier: str = None, chunk_size: int = 48000 * 10) -> np.ndarray:
    """`tier`: which submodel's output to return for a SlimmableParametricA2 model
    (must match model.tier_labels(), e.g. 'lite'/'full'/'w4'/'w8'). REQUIRED for a
    slimmable model -- there is no safe default. Silently always returning the
    widest tier regardless of which checkpoint was actually loaded (the previous
    behavior here) meant inspecting a best_lite.pt checkpoint via this function
    quietly evaluated the full/w8 submodel instead -- caught this comparing
    against per_perm_esr.py's own (tier-correct) result for the same checkpoint
    and permutation: 0.037 here vs. the real 1.30 for that lite-tier corner."""
    model.to(device)
    audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
    params_t = torch.tensor([param_vec], dtype=torch.float32).to(device)            # [1,N]

    with torch.no_grad():
        if isinstance(model, SlimmableParametricA2):
            if tier is None:
                raise ValueError("run_inference: --tier is required for a slimmable "
                                 "model (no safe default -- see this function's docstring)")
            labels = model.tier_labels()
            if tier not in labels:
                raise ValueError(f"tier {tier!r} not in {labels}")
            out = model(audio_t, params_t)[labels.index(tier)]
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
    ap.add_argument("--tier", default=None,
                    help="which submodel to run, for a slimmable checkpoint (e.g. "
                         "'lite'/'full'). Default: inferred from the checkpoint's own "
                         "filename (best.pt -> full, best_<label>.pt -> <label>), matching "
                         "how param_train.py names them -- pass explicitly to override.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model, args_dict = load_model(args.checkpoint)
    param_names = args_dict["param_names"]
    print(f"  Params: {param_names}", flush=True)

    tier = args.tier
    if isinstance(model, SlimmableParametricA2) and tier is None:
        stem = Path(args.checkpoint).stem
        tier = "full" if stem == "best" else stem.removeprefix("best_")
        labels = model.tier_labels()
        if tier not in labels:
            sys.exit(f"Couldn't infer --tier from checkpoint filename {stem!r} "
                     f"(guessed {tier!r}, not in {labels}) -- pass --tier explicitly.")
        print(f"  Tier (inferred from filename): {tier}", flush=True)

    print(f"Loading input {args.input} ...", flush=True)
    audio, sr = sf.read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    # No level normalization -- training no longer rescales its input either (see
    # ParamDataset in param_train.py), so this stays at native level to match.
    audio_norm = audio

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
        out_audio = run_inference(model, audio_norm, vec, args.device, tier)

        sf.write(str(out_path), out_audio.astype(np.float32), sr, subtype="FLOAT")
        print(f"  → {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
