#!/usr/bin/env python3
"""Export a training checkpoint (.pt) to a .param.nam without re-training.

Two modes:

1. Single checkpoint — reconstruct the model, load one checkpoint's weights,
   write the SlimmableContainer / ParametricWaveNet NAM:

       python export_checkpoint.py \
           --checkpoint checkpoints/latest.pt \
           --dataset    /path/to/dataset \
           --output     model_final.param.nam \
           --state      model            # or 'best_state'

2. Composite (--compose) — build ONE container whose every tier is at its own
   best epoch, by splicing each tier's submodel weights from a different
   checkpoint. Valid because slimmable tiers are independent (no shared weights).

       python export_checkpoint.py \
           --compose 'full=checkpoints/best.pt,lite=checkpoints/latest.pt' \
           --dataset /path/to/dataset --output model_optimal.param.nam

   The spec is comma-separated `tier=ckpt.pt` (tiers: lite / full / w<N>); every
   tier of the model must be covered. Widths are inferred from the checkpoints.
"""
import argparse
import json
from pathlib import Path

import torch

from param_train import SlimmableParametricA2, ParametricA2, ParamDataset


def infer_widths(state) -> list[int]:
    """Per-tier channel widths from a slimmable state dict (rechannel.weight is
    (channels,1,1) per tier) — round-trips any --widths ([3,8], [3,4,8], …)."""
    def ch_of(prefix):
        return next(v.shape[0] for k, v in state.items()
                    if k.startswith(prefix + ".") and k.endswith("rechannel.weight"))
    mids = sorted({int(k.split(".")[1]) for k in state
                   if k.startswith("mid.") and k.split(".")[1].isdigit()})
    return [ch_of("lite")] + [ch_of(f"mid.{i}") for i in mids] + [ch_of("full")]


def model_state(ck):
    """Extract a weights dict from a loaded checkpoint (tolerates a raw state)."""
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"]
    if isinstance(ck, dict) and ck.get("best_state"):
        return ck["best_state"]
    return ck


def ckpt_head_mode(ck) -> str:
    """The head this checkpoint was TRAINED with. Must be carried into the model so
    export_nam stamps the .nam correctly — exporting a skip-trained checkpoint as
    'residual' (or vice versa) makes every downstream reader run the wrong head."""
    if isinstance(ck, dict):
        return (ck.get("args_dict") or {}).get("head_mode", "residual")
    return "residual"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="single-checkpoint mode")
    ap.add_argument("--compose", type=str, default=None,
                    help="composite mode: 'tier=ckpt.pt,...' (e.g. "
                         "'full=best.pt,lite=latest.pt'); every tier must be covered")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="dataset dir (supplies config.json + input audio for gain calc)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--state", default="model", choices=["model", "best_state"],
                    help="single-checkpoint mode: which weights to export (default: model)")
    ap.add_argument("--sample-rate", type=int, default=48000)
    args = ap.parse_args()
    if bool(args.checkpoint) == bool(args.compose):
        raise SystemExit("pass exactly one of --checkpoint or --compose")

    ds = ParamDataset(str(args.dataset), mmap=True)  # config + input; mmap avoids loading outputs

    if args.compose:
        # --- composite: each tier's submodel weights from its own checkpoint ---
        specs = {}
        for tok in args.compose.split(","):
            tier, _, path = tok.partition("=")
            if not tier.strip() or not path.strip():
                raise SystemExit(f"bad --compose spec: {tok!r} (want tier=path)")
            specs[tier.strip()] = Path(path.strip())
        raw = {t: torch.load(str(p), map_location="cpu", weights_only=False)
               for t, p in specs.items()}
        loaded = {t: model_state(ck) for t, ck in raw.items()}
        modes = {t: ckpt_head_mode(ck) for t, ck in raw.items()}
        if len(set(modes.values())) > 1:
            raise SystemExit(f"--compose checkpoints disagree on head_mode: {modes}; "
                             "all tiers must be trained with the same head")
        head_mode = next(iter(modes.values()))
        widths = infer_widths(next(iter(loaded.values())))
        model = SlimmableParametricA2(ds.num_params, widths=widths, head_mode=head_mode)
        labels = model.tier_labels()
        missing = [l for l in labels if l not in specs]
        if missing:
            raise SystemExit(f"--compose must cover all tiers {labels}; missing {missing}")
        composite = {k: v.clone() for k, v in model.state_dict().items()}
        for i, lbl in enumerate(labels):
            pref = model.tier_state_prefix(i)
            for k, v in loaded[lbl].items():
                if k.startswith(pref):
                    composite[k] = v.clone() if hasattr(v, "clone") else v
        model.load_state_dict(composite)
        provenance = f"composite [{', '.join(f'{l}<-{specs[l].name}' for l in labels)}]"
    else:
        # --- single checkpoint ---
        ck = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        if args.state not in ck:
            raise SystemExit(f"checkpoint has no '{args.state}' key; available: {list(ck.keys())}")
        state = ck[args.state]
        head_mode = ckpt_head_mode(ck)
        if any(k.startswith("lite.") for k in state):
            model = SlimmableParametricA2(ds.num_params, widths=infer_widths(state),
                                          head_mode=head_mode)
        else:
            ch = next(v.shape[0] for k, v in state.items() if k.endswith("conv.weight"))
            model = ParametricA2(ch, ds.num_params, head_mode=head_mode)
        model.load_state_dict(state)
        provenance = (f"state={args.state}, epoch={ck.get('epoch')}, "
                      f"best_esr={ck.get('best_esr')}, head={head_mode}")

    model.eval()
    nam = model.export_nam(ds.config, {"version": "0.7.0"},
                           sample_rate=args.sample_rate, input_audio=ds.inp)
    args.output.write_text(json.dumps(nam, separators=(",", ":")))
    print(f"wrote {args.output}  ({provenance})")


if __name__ == "__main__":
    main()
