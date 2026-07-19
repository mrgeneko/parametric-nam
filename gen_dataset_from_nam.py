#!/usr/bin/env python3
"""
Build a parametric training dataset from a SET OF EXISTING .nam FILES, each a fixed/captured
knob-setting of a real device (e.g. [REDACTED] exports), instead of rendering from a .schx circuit.

    python gen_dataset_from_nam.py \\
        --nam "/Users/USER/Downloads/5150 DST *.nam" \\
        --output /tmp/5150_ds --gear-make "EVH" --gear-model "5150 Iconic EL34 15w"

Each input filename encodes its knob settings as comma-separated tokens, e.g.
"5150 DST G2, B5, M5, T5, Rvb0, Rsn0, Prsn0.nam" -> Gain=0.2, Bass=0.5, Mids=0.5, Treble=0.5,
Reverb=0.0, Resonance=0.0, Presence=0.0 (single-digit token -> value/10). The prefix->knob-name
map and the digit->value scale are BOTH overridable (--knob-map, --knob-scale) because "not every
set of .nam files will follow the same naming convention" -- there is also a --mapping-csv escape
hatch for conventions too irregular for filename-token parsing.

A knob that takes only ONE distinct value across the provided files is auto-treated as FIXED
(recorded in config.json's "fixed", excluded from training) rather than swept -- per-file captures
are usually a scattered handful of points, not a dense grid, and most filename tokens in any given
batch will not vary at all.

Each .nam file is loaded and run once against the shared sweep input (same convention as the
.schx pipeline: batch_harness.py's `input` config, e.g. sweep-files/sweepv5.wav), producing exactly
one permutation per file. This is a SCATTERED point-sample dataset, not a Cartesian grid --
tools/grid_adequacy.py's interpolation-adequacy reasoning does not apply here; there is nothing to
interpolate between systematically with only a handful of arbitrary points.

Output directory has the exact same contract batch_harness.py produces (config.json, sweep.wav,
params.csv, sig/<shard>/<idx>.npy -> outputs.npy after --combine), so param_train.py's
ParamDataset reads it unmodified -- this file only builds config.json + params.csv + audio, then
calls batch_harness.combine()/run_post_generation_checks() directly rather than reimplementing
either.

    python gen_dataset_from_nam.py --combine /tmp/5150_ds     # same --combine flag as batch_harness.py
"""
import argparse
import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import batch_harness as bh
from nam.models import init_from_nam

# Default prefix -> knob-name map, matching the 5150 DST [REDACTED] captures the user provided.
# A token's ALPHA PREFIX (not just its first letter) is the dict key, so "Rvb"/"Rsn"/"Prsn" never
# collide with "R" or with each other -- filename splitting isolates the whole prefix run before
# lookup, see _parse_filename_tokens.
DEFAULT_PREFIX_MAP = {
    "G": "Gain", "B": "Bass", "M": "Mids", "T": "Treble",
    "Rvb": "Reverb", "Rsn": "Resonance", "Prsn": "Presence",
}

_TOKEN_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _parse_filename_tokens(stem: str, prefix_map: dict, scale_overrides: dict) -> tuple:
    """Split "G2, B5, M5, T5, Rvb0, Rsn0, Prsn0" (after stripping any leading device-name words)
    into {knob_name: float_value}, skipping any token whose prefix isn't in prefix_map (unknown/
    device-name tokens are expected and silently ignored, not an error).

    Also returns {knob_name: digit_string} (the raw token digits, e.g. "10") so callers can catch
    the DEFAULT SCALE'S AMBIGUITY: 10**(-len(digits)) assumes every extra digit is another decimal
    place (a PERCENTAGE convention: "50" -> 0.50), which silently collides with a DIAL-POSITION
    convention (1..10 written out, where "10" means the max, 1.0, not 0.10) -- "G1" and "G10" both
    scale to 0.1 by default. There is no way to tell which convention a filename uses from the
    digit string alone, so this can only be caught after the fact by comparing outputs across the
    whole batch (see main()'s collision check), not fixed inside this function.
    """
    out, raw_digits = {}, {}
    for raw in stem.split(","):
        tok = raw.strip()
        m = _TOKEN_RE.match(tok.split()[-1]) if tok else None
        if not m:
            continue
        prefix, digits = m.group(1), m.group(2)
        name = prefix_map.get(prefix)
        if name is None:
            continue
        scale = scale_overrides.get(prefix, scale_overrides.get(name))
        if scale is None:
            scale = 10 ** (-len(digits))   # single digit -> /10, matches the user's spec exactly
        out[name] = round(int(digits) * scale, 6)
        raw_digits[name] = digits
    return out, raw_digits


def _load_mapping_csv(path: Path) -> dict:
    """filename,<Knob1>,<Knob2>,... escape hatch -- bypasses regex/token parsing entirely for
    conventions that don't fit the "comma-separated PREFIXdigits tokens" shape."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fname = row.pop("filename")
            out[fname] = {k: float(v) for k, v in row.items() if v not in (None, "")}
    return out


def load_nam_model(path: Path, tier: str = "full"):
    """Load a .nam file for inference, transparently unwrapping SlimmableContainer exports
    ([REDACTED] and this repo's own export_checkpoint.py --compose both use this wrapper: multiple
    width-tier submodels gated by `max_value`). Picks the highest max_value ("full", best-quality)
    submodel by default -- these captures have condition_size=1 per submodel (no FiLM knob
    conditioning inside the file itself; each file IS one fixed setting), so tier selection only
    trades inference cost for fidelity, it doesn't change what knob-setting is being captured."""
    d = json.loads(path.read_text())
    if d.get("architecture") == "SlimmableContainer":
        submodels = d["config"]["submodels"]
        picker = max if tier == "full" else min
        d = picker(submodels, key=lambda s: s["max_value"])["model"]
    model = init_from_nam(d)
    model.eval()
    return model, int(d.get("sample_rate") or 48000)


def _run_model(model, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(x, dtype=torch.float32).view(1, -1)
        y = model(t)
    return np.asarray(y).reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--combine", type=Path, help="combine sharded .npy files (delegates to batch_harness.combine)")
    ap.add_argument("--nam", nargs="+", help="glob pattern(s) or explicit .nam file paths")
    ap.add_argument("--output", type=Path, help="output dataset directory")
    ap.add_argument("--input", default="~/work/sweep-files/sweepv5.wav",
                     help="shared sweep input, same convention as the .schx pipeline")
    ap.add_argument("--tier", choices=["full", "lite"], default="full",
                     help="which SlimmableContainer submodel to run when a .nam is container-wrapped")
    ap.add_argument("--knob-map", action="append", default=[],
                     help="PREFIX=Name, overrides/extends the default filename-token prefix map "
                          "(e.g. --knob-map D=Drive). Repeatable.")
    ap.add_argument("--knob-scale", action="append", default=[],
                     help="PREFIX_OR_NAME=FACTOR, overrides the default digit->value scale "
                          "(default: 1/10^len(digits), i.e. a single digit -> value/10). Repeatable.")
    ap.add_argument("--mapping-csv", type=Path,
                     help="filename,<Knob1>,<Knob2>,... CSV; bypasses filename-token parsing "
                          "entirely for a batch that doesn't follow the token convention")
    ap.add_argument("--gear-make", default=None)
    ap.add_argument("--gear-model", default=None)
    ap.add_argument("--gear-type", default=None,
                     help="default: the .nam metadata's own gear_type (e.g. 'pedal', 'amp_cab'), "
                          "falling back to 'amp_cab' only if the metadata has none")
    ap.add_argument("--device-name", default=None,
                     help="circuit/device label; default: first file's metadata gear_model, "
                          "falling back to its (per-setting) metadata name")
    ap.add_argument("--max-crest", type=float, default=20.0,
                     help="reject a capture whose crest factor exceeds this (see batch_harness._finalize_wav)")
    ap.add_argument("--restricted-input", action="store_true",
                     help="mark --input as licensed for local training only, not redistribution "
                          "(e.g. a [redacted]-derived sweep) -- a trained/shipped model is fine, the raw "
                          "sweep audio itself is not. Writes a NOTICE file and an "
                          "input_restricted flag in config.json so the dataset directory is "
                          "self-documenting, matching sweep-files/README.md's [redacted]-DERIVED convention. "
                          "Does not skip writing sweep.wav -- ParamDataset needs it for training -- "
                          "it only makes the restriction explicit and hard to miss.")
    args = ap.parse_args()

    if args.combine:
        bh.combine(args.combine)
        return

    if not args.nam or not args.output:
        ap.error("--nam and --output are required (unless using --combine)")

    files = []
    for pat in args.nam:
        matches = sorted(glob.glob(str(Path(pat).expanduser())))
        files.extend(Path(m) for m in matches) if matches else files.append(Path(pat).expanduser())
    files = sorted(set(files))
    missing = [f for f in files if not f.exists()]
    if missing:
        ap.error(f"not found: {missing}")
    if not files:
        ap.error(f"no .nam files matched {args.nam}")

    prefix_map = dict(DEFAULT_PREFIX_MAP)
    for kv in args.knob_map:
        k, v = kv.split("=", 1)
        prefix_map[k.strip()] = v.strip()
    scale_overrides = {}
    for kv in args.knob_scale:
        k, v = kv.split("=", 1)
        scale_overrides[k.strip()] = float(v)
    mapping = _load_mapping_csv(args.mapping_csv) if args.mapping_csv else {}

    # 1. Parse every filename into a {knob: value} dict.
    per_file, per_file_raw = {}, {}
    for f in files:
        if f.name in mapping:
            per_file[f], per_file_raw[f] = mapping[f.name], {}
        else:
            per_file[f], per_file_raw[f] = _parse_filename_tokens(f.stem, prefix_map, scale_overrides)
    empty = [f.name for f, p in per_file.items() if not p]
    if empty:
        print(f"WARNING: no knobs parsed from filename for: {empty}\n"
              f"         (use --knob-map / --mapping-csv if this batch uses a different convention)",
              file=sys.stderr)

    # 1b. Catch the default scale's DIAL-POSITION-vs-PERCENTAGE ambiguity (see
    # _parse_filename_tokens' docstring): if two files carry DIFFERENT raw digit strings for the
    # same knob (e.g. "1" and "10") but the default 10**(-len(digits)) scale collapsed them to the
    # SAME float value, that's not a legitimate duplicate capture -- it's the scale guessing wrong.
    # Fail loudly with the exact colliding files rather than silently training on a corrupted grid.
    for name in sorted({k for p in per_file_raw.values() for k in p}):
        by_value: dict = {}
        for f, raw in per_file_raw.items():
            if name in raw:
                by_value.setdefault(per_file[f][name], []).append((f.name, raw[name]))
        for value, hits in by_value.items():
            distinct_digits = {digits for _, digits in hits}
            if len(hits) >= 2 and len(distinct_digits) >= 2:
                ap.error(
                    f"'{name}' scale collision: {[h[0] for h in hits]} carry DIFFERENT filename "
                    f"tokens ({sorted(distinct_digits)}) but all scaled to the SAME value "
                    f"({value}) under the default 10**(-len(digits)) rule. This is the classic "
                    f"dial-position-vs-percentage ambiguity (\"1\" and \"10\" both -> 0.1) -- pass "
                    f"--knob-scale {name}=FACTOR (a fixed factor applied to the raw digits, e.g. "
                    f"0.1 for a 1..10 dial) to disambiguate.")

    # 2. Auto-detect fixed vs swept: a knob with exactly one distinct value across the batch is
    # fixed, not trained -- per the source instruction, a batch this small routinely has several
    # knobs that never move at all.
    all_knob_names = sorted({k for p in per_file.values() for k in p})
    values_per_knob = {k: {round(p[k], 6) for p in per_file.values() if k in p} for k in all_knob_names}
    swept = [k for k in all_knob_names if len(values_per_knob[k]) >= 2]
    fixed = {k: sorted(v)[0] for k, v in values_per_knob.items() if len(v) == 1}
    if any(k not in p for p in per_file.values() for k in swept):
        missing_rows = [f.name for f, p in per_file.items() if any(k not in p for k in swept)]
        ap.error(f"some files are missing a swept knob's value entirely (not just a fixed one): "
                 f"{missing_rows} -- every file must supply every swept knob")
    print(f"Swept knobs ({len(swept)}): {swept}")
    print(f"Fixed knobs ({len(fixed)}): {fixed}")

    in_wav = Path(args.input).expanduser()
    inp, sr_in = sf.read(str(in_wav), dtype="float32")
    if inp.ndim > 1:
        inp = inp.mean(axis=1)

    out_dir = args.output.expanduser()
    (out_dir / "sig").mkdir(parents=True, exist_ok=True)

    # 3. Run each captured .nam against the shared sweep, one permutation per file.
    perms, rows, meta0 = [], [], {}
    for idx, f in enumerate(sorted(per_file)):
        params = per_file[f]
        model, sr_model = load_nam_model(f, tier=args.tier)
        x = inp
        if sr_model != sr_in:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr_model, sr_in)
            x = resample_poly(x, sr_model // g, sr_in // g).astype(np.float32)
        t0 = time.time()
        sig = _run_model(model, x)
        proc_t = time.time() - t0

        if not meta0:
            meta0 = json.loads(f.read_text()).get("metadata", {}) or {}

        path = bh.sig_path(out_dir, idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_wav = path.with_suffix(".wav")
        sf.write(str(out_wav), sig, sr_model, subtype="FLOAT")
        r = bh._finalize_wav(idx, path, out_wav, expected_frames=len(sig),
                              max_crest=args.max_crest, proc_t=proc_t)
        r.rung, r.settings = 0, "nam-capture"
        perms.append(params)
        rows.append((idx, params, r))
        status = "OK" if r.ok else "FAIL"
        print(f"[{idx+1}/{len(files)}] {f.name}  {status}  {' '.join(f'{k}={v}' for k, v in params.items())}"
              f"{'  ' + r.error if r.error else ''}")

    ok_count = sum(1 for _, _, r in rows if r.ok)
    if ok_count == 0:
        sys.exit("ERROR: every capture failed integrity checks -- nothing to write")

    # 4. config.json + params.csv, matching batch_harness.py's own schema exactly so
    # param_train.py's ParamDataset (which only requires config.json["knobs"], sweep.wav,
    # params.csv, outputs.npy) reads this unmodified.
    device_name = args.device_name or meta0.get("gear_model") or meta0.get("name") or out_dir.name
    knob_bounds = {k: [min(p[k] for p in per_file.values()), max(p[k] for p in per_file.values())]
                   for k in swept}
    cfg = {
        "backend": "nam-capture",
        "circuit": device_name,
        "schx": None,
        "knobs": swept,
        "fixed": fixed,
        "steps": {},
        "bounds": knob_bounds,
        "defaults": {},
        "oversample": 1,
        "param_map": {k: k for k in swept},
        "fixed_params": None,
        "speaker": None,
        "input_wav": str(in_wav),
        "input": bh.input_provenance(in_wav),
        "permutation_count": len(files),
        "values": None,
        "workers": 1,
        "gear_make": args.gear_make or meta0.get("gear_make") or device_name,
        "gear_model": args.gear_model or meta0.get("gear_model") or device_name,
        "gear_type": args.gear_type or meta0.get("gear_type") or "amp_cab",
        "source_nam_files": [str(f) for f in sorted(per_file)],
        "tier": args.tier,
    }
    if args.restricted_input:
        # Local training/a shipped trained model is fine; the raw sweep audio itself is not
        # redistributable (matches sweep-files/README.md's [redacted]-DERIVED convention). sweep.wav is
        # still written below -- ParamDataset needs it on disk to train -- this just makes the
        # restriction impossible to miss later, the same way that repo's file-list table does.
        cfg["input_restricted"] = True
        cfg["input_restricted_note"] = (
            "input_wav is NOT licensed for redistribution (e.g. a [redacted]-derived sweep). "
            "Training on it / shipping the resulting .nam is fine; committing, publishing, or "
            "bundling this dataset directory or its sweep.wav is NOT. See sweep-files/README.md."
        )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "sweep.wav").write_bytes(in_wav.read_bytes())
    if args.restricted_input:
        (out_dir / "NOTICE_RESTRICTED_INPUT.md").write_text(
            f"# Restricted input -- do not redistribute\n\n"
            f"This dataset's `sweep.wav` is a copy of `{in_wav}`, which is NOT licensed for "
            f"redistribution (see sweep-files/README.md's [redacted]-DERIVED table).\n\n"
            f"- OK: training locally on this dataset, shipping/publishing a model trained from it.\n"
            f"- NOT OK: committing this directory, publishing it, or bundling `sweep.wav` itself "
            f"into any released dataset or model package.\n"
        )
        print(f"NOTE: --restricted-input set -- wrote {out_dir/'NOTICE_RESTRICTED_INPUT.md'} "
              f"(sweep.wav must not be redistributed; the trained model is fine)")

    with open(out_dir / "params.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx"] + swept + ["dsp_load", "proc_time", "rms", "peak", "ok", "error", "rung", "solver"])
        for idx, params, r in rows:
            w.writerow([idx] + [params[k] for k in swept] +
                       [r.dsp_load, r.proc_time, f"{r.rms:.6f}", f"{r.peak:.6f}", int(r.ok), r.error,
                        r.rung, r.settings])

    bh.combine(out_dir)
    bh.run_post_generation_checks(out_dir, swept, perms, values_per_knob)


if __name__ == "__main__":
    main()
