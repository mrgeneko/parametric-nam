#!/usr/bin/env python3
"""
Build a parametric training dataset from a SET OF EXISTING per-setting captures, each a
fixed/captured knob-setting of a real device, instead of rendering from a .schx circuit. Two
source kinds, auto-detected per file by extension, freely mixable in one run:

  .nam   an already-trained/exported fixed-setting model (real hardware or amp-modeler export).
         Loaded and run once against the shared sweep input to produce its wet render --
         digital inference, exactly phase-aligned to the input by construction.
  .wav   an already-recorded wet capture: the shared sweep played through the real device at
         this setting and captured directly (e.g. via an audio interface). Unlike .nam
         inference, a REAL analog signal chain (ADC/DAC, cable, buffering) has its own latency
         relative to the reference sweep -- so each .wav is time-aligned via best-effort NAM
         blip-based calibration before use (see detect_delay). Only NAM's OWN recognized
         standard sweeps (e.g. sweep-v3.wav) get real calibration; capturing against this
         repo's bundled sweepv5.wav always falls back to a disclosed delay=0, same as
         capture_static.py's identical fallback for a non-standard excitation.

    python gen_dataset_from_captures.py \\
        --captures "~/Downloads/5150 DST *.nam" \\
        --output /tmp/5150_ds --gear-make "EVH" --gear-model "5150 Iconic EL34 15w"

    python gen_dataset_from_captures.py \\
        --captures "~/Downloads/Klon *.wav" --output /tmp/klon_ds

Each input filename encodes its knob settings as comma-separated tokens, e.g.
"5150 DST G2, B5, M5, T5, Rvb0, Rsn0, Prsn0.nam" -> Gain=0.2, Bass=0.5, Mids=0.5, Treble=0.5,
Reverb=0.0, Resonance=0.0, Presence=0.0 (single-digit token -> value/10). The prefix->knob-name
map and the digit->value scale are BOTH overridable (--knob-map, --knob-scale) because "not every
set of captures will follow the same naming convention" -- there is also a --mapping-csv escape
hatch for conventions too irregular for filename-token parsing. Same convention, same parser
(capture_common.py), for either source kind.

A knob that takes only ONE distinct value across the provided files is auto-treated as FIXED
(recorded in config.json's "fixed", excluded from training) rather than swept -- per-file captures
are usually a scattered handful of points, not a dense grid, and most filename tokens in any given
batch will not vary at all.

Output directory has the exact same contract batch_harness.py produces (config.json, sweep.wav,
params.csv, sig/<shard>/<idx>.npy -> outputs.npy after --combine), so param_train.py's
ParamDataset reads it unmodified -- this file only builds config.json + params.csv + audio, then
calls batch_harness.combine()/run_post_generation_checks() directly rather than reimplementing
either. This is a SCATTERED point-sample dataset, not a Cartesian grid -- tools/grid_adequacy.py's
interpolation-adequacy reasoning does not apply here; there is nothing to interpolate between
systematically with only a handful of arbitrary points.

    python gen_dataset_from_captures.py --combine /tmp/5150_ds     # same --combine flag as batch_harness.py
"""
import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import batch_harness as bh
from capture_common import add_knob_parsing_args, check_scale_collisions, parse_all_filenames, resolve_knob_maps


def _find_nam_site_packages(work_dir: Path = None):
    """Best-effort: locate a sibling neural-amp-modeler checkout's venv site-packages (same
    layout capture_static.py assumes), so the OPTIONAL nam.models/nam.train.core imports below
    -- needed only for .nam inference and .wav delay calibration respectively, not for this
    module's own import or for a .wav-only run without calibration -- can find them. Returns
    None if no such checkout exists; callers degrade gracefully rather than crash.

    `work_dir` (default ~/work) is a parameter rather than hardcoded so tests can point it at
    an empty directory instead of monkeypatching Path.home() -- stdlib global state pytest's
    own internals rely on too."""
    venv = (work_dir or Path.home() / "work") / "neural-amp-modeler" / "venv"
    site_packages = next((venv / "lib").glob("python*/site-packages"), None)
    return site_packages if site_packages and site_packages.exists() else None


def _import_nam_models():
    """Lazy, best-effort import of nam.models.init_from_nam -- isolated into its own function
    (rather than a bare module-level import) so (a) this script imports cleanly even without
    the external neural-amp-modeler package installed, needed only when a .nam file is actually
    present, and (b) tests can monkeypatch this exact function instead of faking sys.path."""
    site_packages = _find_nam_site_packages()
    if site_packages and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    try:
        from nam.models import init_from_nam
        return init_from_nam
    except ImportError:
        return None


def load_nam_model(path: Path, tier: str = "full"):
    """Load a .nam file for inference, transparently unwrapping SlimmableContainer exports
    (community capture tools and this repo's own export_checkpoint.py --compose both use this wrapper: multiple
    width-tier submodels gated by `max_value`). Picks the highest max_value ("full", best-quality)
    submodel by default -- these captures have condition_size=1 per submodel (no FiLM knob
    conditioning inside the file itself; each file IS one fixed setting), so tier selection only
    trades inference cost for fidelity, it doesn't change what knob-setting is being captured."""
    init_from_nam = _import_nam_models()
    if init_from_nam is None:
        sys.exit(f"can't load {path}: the 'nam' package (neural-amp-modeler) isn't importable. "
                 f"Point $PYTHONPATH at its venv's site-packages, or install it "
                 f"(pip install neural-amp-modeler) -- only needed for .nam-sourced captures; "
                 f".wav-sourced ones don't need it.")
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


# nam.train.core._calibrate_latency_v_all's own search window around the expected blip
# location (lookahead=1_000, lookback=10_000 -- not exported as constants, so mirrored here).
# A delay outside [-lookahead, +lookback] can still come back as a "successful" calibration
# -- confirmed directly: an injected ~1s delay, far outside the window, returned a plausible-
# looking but WRONG ~0. detect_delay() below treats anything outside this range as untrusted.
#
# THIS BOUND IS NOT A COMPLETE GUARANTEE. It only catches results that violate NAM's own
# documented search range -- a spurious match that happens to land INSIDE that range (e.g.
# the algorithm triggering on some incidental transient in the actual rendered content,
# rather than the real calibration blip) isn't caught by it, and isn't caught by NAM's own
# exposed warnings either (checked directly: `matches_lookahead`/`disagreement_too_high` were
# both False on exactly such a spurious match -- V3 has only one blip location, so there's
# nothing for `disagreement_too_high` to cross-check against). A latency this far outside
# realistic audio-interface territory (tested: ~1s) is an extreme case, not a scenario this
# tool can fully defend against from NAM's calibration output alone.
_NAM_LOOKAHEAD_SAMPLES = 1_000
_NAM_LOOKBACK_SAMPLES = 10_000

_NAM_DELAY_HELPER = Path(__file__).resolve().parent / "tools" / "nam_delay_helper.py"


def detect_delay(dry_wav: Path, wet_wav: Path) -> tuple:
    """Best-effort NAM blip-based latency calibration between a wet capture and the shared dry
    reference sweep. Returns (delay_samples, source):

      "nam_standard_calibration"  real calibration -- dry_wav matched a NAM-recognized standard
                                   sweep (e.g. sweep-v3.wav)
      "delay_zero_fallback"       dry_wav isn't a recognized standard file (e.g. this repo's own
                                   sweepv5.wav), calibration found no usable impulse response, or
                                   a "successful" result fell outside NAM's own trusted search
                                   window (see _NAM_LOOKAHEAD_SAMPLES/_NAM_LOOKBACK_SAMPLES)
      "nam_unavailable"           the optional neural-amp-modeler package isn't importable, or
                                   its subprocess (see below) couldn't be run at all

    A REAL capture's latency (analog signal chain: ADC/DAC, cable, buffering) has no digital-
    inference equivalent -- batch_harness.py's SPICE renders and this file's own .nam-inference
    path are both exactly phase-aligned to the input by construction, so this problem is unique
    to the .wav path. Never a correlation-based guess (see capture_static.py's module docstring
    for why that actively misleads) -- only NAM's own impulse-based calibration, or a disclosed
    delay=0.

    RUNS IN A SUBPROCESS (tools/nam_delay_helper.py), under the sibling neural-amp-modeler
    venv's OWN interpreter -- not a lazy in-process import like load_nam_model's. nam.train.core
    depends on numba, which checks numpy's version at import time; importing it in-process (even
    after adding the sibling venv's site-packages to sys.path) hands numba whatever numpy module
    THIS process already loaded (parametric-nam's own), not the sibling venv's compatible one --
    confirmed hitting exactly this ("Numba needs NumPy 2.4 or less. Got NumPy 2.5.") in normal
    use. A fresh subprocess under that venv's own python has no such contamination."""
    site_packages = _find_nam_site_packages()
    if site_packages is None:
        return 0, "nam_unavailable"
    python = site_packages.parent.parent.parent / "bin" / "python3"  # venv/lib/pyX/site-packages -> venv/bin/python3
    if not python.exists():
        return 0, "nam_unavailable"

    try:
        r = subprocess.run([str(python), str(_NAM_DELAY_HELPER), str(dry_wav), str(wet_wav)],
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"  WARNING: NAM latency calibration timed out on {wet_wav.name} -- "
              f"falling back to delay=0.")
        return 0, "delay_zero_fallback"

    try:
        result = json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  WARNING: NAM latency calibration subprocess produced no usable output on "
              f"{wet_wav.name} (exit {r.returncode}): {r.stderr[-300:] or '(no stderr)'} -- "
              f"falling back to delay=0.")
        return 0, "nam_unavailable"

    if "detail" in result:
        print(f"  WARNING: {result['detail']} -- falling back to delay=0.")
    delay, source = result.get("delay"), result.get("source")
    if delay is None:
        return 0, source or "delay_zero_fallback"

    if delay < -_NAM_LOOKAHEAD_SAMPLES or delay > _NAM_LOOKBACK_SAMPLES:
        print(f"  WARNING: NAM calibration on {wet_wav.name} returned delay={delay}, outside "
              f"its own [-{_NAM_LOOKAHEAD_SAMPLES}, +{_NAM_LOOKBACK_SAMPLES}]-sample search "
              f"window -- treating as an unreliable match rather than a real detection, "
              f"falling back to delay=0.")
        return 0, "delay_zero_fallback"

    return int(delay), source


def _align_wet(y: np.ndarray, delay: int, target_len: int) -> np.ndarray:
    """Shift a wet capture so index i lines up with the shared dry sweep.wav's index i, using
    NAM's own delay-sign convention (positive = wet lags dry; see nam.data.Data's docstring:
    "we get rid of the start of x, end of y"). Unlike NAM's own _apply_delay_int (which
    shortens BOTH x and y), sweep.wav is shared by every permutation in this dataset and is
    never trimmed here -- only `y` is shifted, then padded or truncated to exactly `target_len`
    (the dataset's fixed N_samples, from sweep.wav's own length)."""
    if delay > 0:
        y = y[delay:]
    elif delay < 0:
        y = np.concatenate([np.zeros(-delay, dtype=y.dtype), y])
    if len(y) < target_len:
        y = np.concatenate([y, np.zeros(target_len - len(y), dtype=y.dtype)])
    else:
        y = y[:target_len]
    return y


def _wet_from_wav(wet_path: Path, dry_path: Path, sr_in: int, target_len: int) -> tuple:
    """Load a raw capture WAV as this permutation's wet signal: delay-detected against the
    shared dry reference (native sample rate), then resampled to sr_in if needed -- the
    detected delay is scaled proportionally so it still lines up after resampling -- and
    finally time-aligned via _align_wet. Returns (aligned_signal, delay_used_at_sr_in,
    calibration_source) for the caller's params.csv row."""
    delay_native, source = detect_delay(dry_path, wet_path)

    y, sr_wav = sf.read(str(wet_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    delay = delay_native
    if sr_wav != sr_in:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_wav, sr_in)
        y = resample_poly(y, sr_in // g, sr_wav // g).astype(np.float32)
        delay = round(delay_native * sr_in / sr_wav)

    return _align_wet(y, delay, target_len), delay, source


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--combine", type=Path, help="combine sharded .npy files (delegates to batch_harness.combine)")
    ap.add_argument("--captures", nargs="+",
                     help="glob pattern(s) or explicit .nam/.wav file paths -- freely mixable")
    ap.add_argument("--output", type=Path, help="output dataset directory")
    ap.add_argument("--input", default="examples/sweepv5.wav",
                     help="shared dry sweep, same convention as the .schx pipeline. For .wav "
                          "captures this is also the delay-calibration reference -- see "
                          "detect_delay's docstring for why only a NAM-recognized standard "
                          "sweep (not this default) gets real calibration")
    ap.add_argument("--tier", choices=["full", "lite"], default="full",
                     help="which SlimmableContainer submodel to run when a .nam is container-wrapped")
    add_knob_parsing_args(ap)
    ap.add_argument("--gear-make", default=None)
    ap.add_argument("--gear-model", default=None)
    ap.add_argument("--gear-type", default=None,
                     help="default: a .nam file's own metadata gear_type (e.g. 'pedal', "
                          "'amp_cab'), falling back to 'amp_cab' if there's none (.wav "
                          "captures carry no metadata at all)")
    ap.add_argument("--device-name", default=None,
                     help="circuit/device label; default: first .nam file's metadata "
                          "gear_model, falling back to its (per-setting) metadata name, "
                          "falling back to the output directory name")
    ap.add_argument("--max-crest", type=float, default=20.0,
                     help="reject a capture whose crest factor exceeds this (see batch_harness._finalize_wav)")
    ap.add_argument("--restricted-input", action="store_true",
                     help="mark --input as licensed for local training only, not redistribution "
                          "(e.g. a third-party-derived sweep) -- a trained/shipped model is fine, the raw "
                          "sweep audio itself is not. Writes a NOTICE file and an "
                          "input_restricted flag in config.json so the dataset directory is "
                          "self-documenting, matching sweep-files/README.md's third-party-derived convention. "
                          "Does not skip writing sweep.wav -- ParamDataset needs it for training -- "
                          "it only makes the restriction explicit and hard to miss.")
    args = ap.parse_args()

    if args.combine:
        bh.combine(args.combine)
        return

    if not args.captures or not args.output:
        ap.error("--captures and --output are required (unless using --combine)")

    files = []
    for pat in args.captures:
        matches = sorted(glob.glob(str(Path(pat).expanduser())))
        files.extend(Path(m) for m in matches) if matches else files.append(Path(pat).expanduser())
    files = sorted(set(files))
    missing = [f for f in files if not f.exists()]
    if missing:
        ap.error(f"not found: {missing}")
    if not files:
        ap.error(f"no files matched {args.captures}")
    unknown_ext = [f for f in files if f.suffix.lower() not in (".nam", ".wav")]
    if unknown_ext:
        ap.error(f"not a .nam or .wav file: {unknown_ext}")

    prefix_map, scale_overrides, mapping = resolve_knob_maps(args)

    # 1. Parse every filename into a {knob: value} dict.
    per_file, per_file_raw = parse_all_filenames(files, prefix_map, scale_overrides, mapping)

    # 1b. Catch the default scale's DIAL-POSITION-vs-PERCENTAGE ambiguity (see
    # capture_common.parse_filename_tokens' docstring) -- fail loudly with the exact colliding
    # files rather than silently training on a corrupted grid.
    check_scale_collisions(per_file, per_file_raw, ap.error)

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

    # 3. Get each permutation's wet render: run a .nam through the shared sweep (inference,
    # exactly phase-aligned by construction), or time-align a .wav capture against it
    # (detect_delay -- a real analog signal chain has latency inference doesn't).
    perms, rows, meta0 = [], [], {}
    for idx, f in enumerate(sorted(per_file)):
        params = per_file[f]
        t0 = time.time()
        if f.suffix.lower() == ".wav":
            sig, delay, source = _wet_from_wav(f, in_wav, sr_in, target_len=len(inp))
            sig_sr, tag, rung = sr_in, f"wav-capture:{source}", delay
        else:
            model, sr_model = load_nam_model(f, tier=args.tier)
            x = inp
            if sr_model != sr_in:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr_model, sr_in)
                x = resample_poly(x, sr_model // g, sr_in // g).astype(np.float32)
            sig = _run_model(model, x)
            sig_sr, tag, rung = sr_model, "nam-capture", 0
            if not meta0:
                meta0 = json.loads(f.read_text()).get("metadata", {}) or {}
        proc_t = time.time() - t0

        path = bh.sig_path(out_dir, idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_wav = path.with_suffix(".wav")
        sf.write(str(out_wav), sig, sig_sr, subtype="FLOAT")
        r = bh._finalize_wav(idx, path, out_wav, expected_frames=len(sig),
                              max_crest=args.max_crest, proc_t=proc_t)
        r.rung, r.settings = rung, tag
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
        "backend": "capture",
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
        "source_files": [str(f) for f in sorted(per_file)],
        "tier": args.tier,
    }
    if args.restricted_input:
        # Local training/a shipped trained model is fine; the raw sweep audio itself is not
        # redistributable (matches sweep-files/README.md's third-party-derived convention). sweep.wav is
        # still written below -- ParamDataset needs it on disk to train -- this just makes the
        # restriction impossible to miss later, the same way that repo's file-list table does.
        cfg["input_restricted"] = True
        cfg["input_restricted_note"] = (
            "input_wav is NOT licensed for redistribution (e.g. a third-party-derived sweep). "
            "Training on it / shipping the resulting .nam is fine; committing, publishing, or "
            "bundling this dataset directory or its sweep.wav is NOT. See sweep-files/README.md."
        )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "sweep.wav").write_bytes(in_wav.read_bytes())
    if args.restricted_input:
        (out_dir / "NOTICE_RESTRICTED_INPUT.md").write_text(
            f"# Restricted input -- do not redistribute\n\n"
            f"This dataset's `sweep.wav` is a copy of `{in_wav}`, which is NOT licensed for "
            f"redistribution (see sweep-files/README.md's third-party-derived table).\n\n"
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
