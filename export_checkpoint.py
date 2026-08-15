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
    def has(prefix):
        return any(k.startswith(prefix + ".") and k.endswith("rechannel.weight") for k in state)
    if not has("lite"):        # single-width model: only the `full` tier exists
        return [ch_of("full")]
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


def detect_spectral_norm(state) -> bool:
    """Was this checkpoint trained with A2Layer(spectral_norm=True)? Detectable directly
    from the state dict: PyTorch's parametrize system renames conv/mixin/l1x1's plain
    `.weight` key to `.parametrizations.weight.original` (+ power-iteration `._u`/`._v`
    buffers) -- their presence/absence is unambiguous."""
    return any(".parametrizations.weight" in k for k in state)


def detect_lora_rank(state) -> int:
    """Model-wide LoRA rank (uniform across all tiers/layers by construction -- see
    SlimmableParametricA2.__init__'s `kw` dict), or 0 if this checkpoint has no LoRA
    layers. Purely structural, like detect_spectral_norm above: finds any
    "...layers.<i>.lora.net_A.weight" key, reads that SAME tier's own rechannel.weight
    (or, for a non-slimmable single ParametricA2 state with no tier prefix, the
    top-level rechannel.weight) to recover channels, and divides -- net_A.weight is
    nn.Linear(cond_dim, channels*rank), shape (channels*rank, cond_dim), so rank is
    only recoverable once channels is known. No separate CLI re-entry needed, avoiding
    the mistake `--film-gamma-bound` made (a scalar hyperparameter with no state-dict
    trace, requiring the caller to re-specify it by hand and get it right)."""
    for k, v in state.items():
        if not k.endswith("lora.net_A.weight"):
            continue
        # "" (non-slimmable, key starts with "layers.") or "lite"/"full"/"mid.0" etc.
        # (slimmable, key is "<prefix>.layers...."). k.split(".layers.") only works for the
        # latter -- the non-slimmable key has no leading dot before "layers." to split on.
        idx = k.find("layers.")
        prefix = k[:idx].rstrip(".")
        rechannel_key = f"{prefix}.rechannel.weight" if prefix else "rechannel.weight"
        if rechannel_key not in state:
            raise SystemExit(f"found {k} but no matching {rechannel_key} to recover "
                             f"channels from -- corrupt or hand-edited checkpoint?")
        channels = state[rechannel_key].shape[0]
        out_features = v.shape[0]
        if out_features % channels != 0:
            raise SystemExit(f"can't infer LoRA rank from {k}: {out_features} output "
                             f"features not a multiple of channels={channels}")
        return out_features // channels
    return 0


def require_tier_agreement(per_tier: dict, what: str):
    """--compose's tiers must all agree on a model-wide setting (spectral_norm presence,
    LoRA rank, ...) -- these aren't per-tier knobs, so a mismatch means the checkpoints
    can't actually be spliced into one consistent container. Returns the single agreed
    value, or raises SystemExit citing exactly which tiers disagreed."""
    if len(set(per_tier.values())) > 1:
        raise SystemExit(f"--compose checkpoints disagree on {what} "
                         f"(detected {per_tier}) -- can't mix tiers with different {what}")
    return next(iter(per_tier.values()))


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
        widths = infer_widths(next(iter(loaded.values())))
        sn_per_tier = {t: detect_spectral_norm(s) for t, s in loaded.items()}
        spectral_norm = require_tier_agreement(sn_per_tier, "spectral_norm")
        lora_per_tier = {t: detect_lora_rank(s) for t, s in loaded.items()}
        lora_rank = require_tier_agreement(lora_per_tier, "LoRA rank")
        model = SlimmableParametricA2(ds.num_params, widths=widths, spectral_norm=spectral_norm,
                                      lora_rank=lora_rank)
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
        spectral_norm = detect_spectral_norm(state)
        lora_rank = detect_lora_rank(state)
        if any(k.startswith("lite.") for k in state):
            model = SlimmableParametricA2(ds.num_params, widths=infer_widths(state),
                                          spectral_norm=spectral_norm, lora_rank=lora_rank)
        else:
            ch = next(v.shape[0] for k, v in state.items()
                     if k.endswith("conv.parametrizations.weight.original") or k.endswith("conv.weight"))
            model = ParametricA2(ch, ds.num_params, spectral_norm=spectral_norm, lora_rank=lora_rank)
        model.load_state_dict(state)
        provenance = (f"state={args.state}, epoch={ck.get('epoch')}, "
                      f"best_esr={ck.get('best_esr')}")

    model.eval()
    nam = model.export_nam(ds.config, {"version": "0.7.0"},
                           sample_rate=args.sample_rate, input_audio=ds.inp)
    args.output.write_text(json.dumps(nam, separators=(",", ":")))
    print(f"wrote {args.output}  ({provenance})")


if __name__ == "__main__":
    main()
