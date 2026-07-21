#!/usr/bin/env python3
"""
per_perm_esr.py — Per-permutation validation ESR for a trained parametric NAM.

Training's own reported ESR is a single number per tier, averaged over a
random ~10% slice of the pooled (permutation x repeat-crop) dataset -- it
doesn't say where in the knob grid the model is more or less accurate. This
runs the trained model over each permutation's FULL ground-truth audio (not
a training crop) and reports ESR per (knob...) combination, so you can see
which regions of the grid fit well and which don't.

compute_per_perm_esr() is the reusable core -- param_train.py calls it directly
at the end of training (reusing the already-loaded ParamDataset, no re-reading
from disk). main() below is a thin standalone-CLI wrapper around the same
function, for re-running this against an already-saved checkpoint later.

Usage:
    python per_perm_esr.py --checkpoint best.pt --dataset /path/to/dataset_dir \
        --tier full --warmup-s 1.0 -o per_perm_esr.csv
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
# Deferred to inside main() (CLI-only, not used by compute_per_perm_esr/summarize/write_csv):
# param_infer imports param_train, which imports THIS module for compute_per_perm_esr/summarize/
# write_csv -- a module-level import here would be circular.


def compute_per_perm_esr(submodel, inp: np.ndarray, outputs, samples: list,
                         param_names: list[str], device: str, scale: float = 1.0,
                         warmup_s: float = 1.0) -> list[dict]:
    """Per-permutation ESR for `submodel` against full-length ground truth.

    `inp` must already be scaled to the same target_dbfs the model was trained
    on (ParamDataset.inp / ParamDataset._scale) -- this does not renormalize it.
    `outputs` is the RAW (unscaled) ground truth array (ParamDataset.outputs,
    which may be memory-mapped) -- `scale` (ParamDataset._scale) is applied to
    each permutation's row lazily inside the loop, not to the whole array up
    front, so an mmap'd outputs.npy is never pulled fully into RAM here.
    `samples` is a list of (row_index_into_outputs, {param_name: value}) pairs,
    matching ParamDataset.samples exactly.

    Returns a list of {**params, "esr": float}, one entry per permutation,
    UNSORTED (caller sorts if it wants best/worst).
    """
    submodel.eval()
    warmup_n = int(warmup_s * 48000)
    inp_t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
    results = []
    with torch.no_grad():
        for row_i, params_dict in samples:
            target = (np.asarray(outputs[row_i]) * scale).astype(np.float32)
            sig_len = min(len(inp), len(target))
            params = torch.tensor([[params_dict[n] for n in param_names]],
                                  dtype=torch.float32, device=device)
            pred = submodel(inp_t[:, :, :sig_len], params).squeeze().cpu().numpy()
            t = target[:sig_len]
            p, t = pred[warmup_n:], t[warmup_n:]
            esr = float(np.sum((p - t) ** 2) / (np.sum(t ** 2) + 1e-12))
            results.append({**params_dict, "esr": esr})
    return results


def summarize(results: list[dict], tier_label: str = "") -> str:
    """Human-readable best/worst/mean/median/std summary, sorted worst-first
    is NOT assumed -- caller's `results` order is preserved; this just reports
    stats. Matches the print format main() used to emit inline."""
    ordered = sorted(results, key=lambda d: d["esr"])
    esrs = np.array([d["esr"] for d in ordered])
    prefix = f"{tier_label} tier, " if tier_label else ""
    lines = [
        f"{prefix}{len(ordered)} permutations",
        f"  best:   {ordered[0]}",
        f"  worst:  {ordered[-1]}",
        f"  mean={esrs.mean():.6f}  median={np.median(esrs):.6f}  std={esrs.std():.6f}",
    ]
    return "\n".join(lines)


def write_csv(results: list[dict], param_names: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*param_names, "esr"])
        w.writeheader()
        w.writerows(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="best.pt / best_<tier>.pt")
    ap.add_argument("--dataset", required=True, help="Dataset dir (has outputs.npy, sweep.wav, params.csv)")
    ap.add_argument("--tier", default="full", help="Which tier to evaluate (default: %(default)s)")
    ap.add_argument("--warmup-s", type=float, default=1.0,
                    help="Seconds to exclude from the start of each permutation before "
                         "computing ESR (matches the SPICE-generation warmup convention -- "
                         "avoids a startup transient skewing the number). (default: %(default)s)")
    ap.add_argument("--target-dbfs", type=float, default=-18.0,
                    help="Must match param_train.py's ParamDataset normalization (default: %(default)s)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Write per-perm CSV here")
    args = ap.parse_args()

    from param_infer import load_model
    dataset_dir = Path(args.dataset)
    model, args_dict = load_model(args.checkpoint)
    model.to(args.device)
    model.eval()

    labels = model.tier_labels()
    if args.tier not in labels:
        sys.exit(f"--tier {args.tier!r} not in {labels}")
    submodel = model.submodels[labels.index(args.tier)]

    with open(dataset_dir / "config.json") as f:
        config = json.load(f)
    param_names = config["knobs"]

    # Same normalization ParamDataset.__init__ applies -- must match to get a
    # comparable ESR to what training reported.
    inp_raw, _ = sf.read(str(dataset_dir / "sweep.wav"))
    if inp_raw.ndim > 1:
        inp_raw = inp_raw.mean(axis=1)
    inp_raw = inp_raw.astype(np.float32)
    rms = float(np.sqrt(np.mean(inp_raw ** 2)))
    scale = (10 ** (args.target_dbfs / 20.0)) / (rms + 1e-8)
    inp = inp_raw * scale

    outputs = np.load(str(dataset_dir / "outputs.npy"), mmap_mode="r")

    # Same idx-compaction ParamDataset.__init__ does: outputs.npy row i is the
    # i-th successful permutation in sorted-idx order, NOT row `idx` (see the
    # param_train commit fixing the IndexError this caused when any perm fails).
    with open(dataset_dir / "params.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["ok"] == "1"]
    rows.sort(key=lambda r: int(r["idx"]))
    samples = [(i, {n: float(r[n]) for n in param_names}) for i, r in enumerate(rows)]

    results = compute_per_perm_esr(submodel, inp, outputs, samples, param_names,
                                   args.device, scale, args.warmup_s)
    for r in sorted(results, key=lambda d: d["esr"]):
        print(f"  {', '.join(f'{n}={r[n]}' for n in param_names)}  ESR={r['esr']:.6f}",
              file=sys.stderr)
    print(f"\n{summarize(results, args.tier)}", file=sys.stderr)

    if args.output:
        write_csv(results, param_names, args.output)
        print(f"\nWrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
