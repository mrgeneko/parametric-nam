#!/usr/bin/env python3
"""Export a training checkpoint (.pt) to a .param.nam without re-training.

Reconstructs the model, loads weights from the checkpoint, and writes the
SlimmableContainer / ParametricWaveNet NAM. Useful for exporting a specific
epoch (e.g. the final `latest.pt` weights, or the stored `best_state`) after
training has finished.

    python export_checkpoint.py \
        --checkpoint checkpoints/latest.pt \
        --dataset    /path/to/dataset \
        --output     model_final.param.nam \
        --state      model            # or 'best_state'
"""
import argparse
import json
from pathlib import Path

import torch

from param_train import SlimmableParametricA2, ParametricA2, ParamDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="dataset dir (supplies config.json + input audio for gain calc)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--state", default="model", choices=["model", "best_state"],
                    help="which weights in the checkpoint to export (default: model)")
    ap.add_argument("--sample-rate", type=int, default=48000)
    args = ap.parse_args()

    ds = ParamDataset(str(args.dataset), mmap=True)  # config + input; mmap avoids loading outputs into RAM
    ck = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if args.state not in ck:
        raise SystemExit(f"checkpoint has no '{args.state}' key; available: {list(ck.keys())}")
    state = ck[args.state]

    slimmable = any(k.startswith("lite.") for k in state)
    if slimmable:
        model = SlimmableParametricA2(ds.num_params)
    else:
        # infer channel count from the first layer's conv weight (out_channels)
        ch = next(v.shape[0] for k, v in state.items() if k.endswith("conv.weight"))
        model = ParametricA2(ch, ds.num_params)
    model.load_state_dict(state)
    model.eval()

    nam = model.export_nam(ds.config, {"version": "0.7.0"},
                           sample_rate=args.sample_rate, input_audio=ds.inp)
    args.output.write_text(json.dumps(nam, separators=(",", ":")))
    print(f"wrote {args.output}  (state={args.state}, "
          f"epoch={ck.get('epoch')}, best_esr={ck.get('best_esr')})")


if __name__ == "__main__":
    main()
