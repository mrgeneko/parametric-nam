#!/usr/bin/env python3
"""
batch_generate.py — Synthetic training data generator for parametric NAM.

Latin Hypercube Sampling over knob space → harness subprocess per sample
→ paired (input_sweep, output_sweep, knob_values) dataset.

Usage:
  python batch_generate.py --output-dir /tmp/param_dataset [options]
"""

import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from scipy.stats.qmc import LatinHypercube
except ImportError:
    LatinHypercube = None

GDRIVE_FILE_ID = "1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9"
SWEEP_FILENAME = "standard_test_sweep_48k.wav"

# Known param ranges per circuit. Extend as circuits are added.
CIRCUIT_PARAM_RANGES: dict[str, dict[str, list[float]]] = {
    "subcircuit_tone_stack": {
        "mid":      [0.0, 1.0],
        "treble":   [0.0, 1.0],
    },
    "c_59_bassman_tone_stack": {
        "mid":      [0.0, 1.0],
        "treble":   [0.0, 1.0],
    },
    "subcircuit_tone_stacks": {
        "mid":      [0.0, 1.0],
        "treble":   [0.0, 1.0],
    },
    "marshall_jcm800_2203_preamp_modded": {
        "gain_1":   [0.0, 1.0],
        "gain_2":   [0.0, 1.0],
        "bass":     [0.0, 1.0],
        "middle":   [0.0, 1.0],
        "treble":   [0.0, 1.0],
        "volume":   [0.0, 1.0],
    },
    "marshall_jcm800_2203_preamp": {
        "gain":     [0.0, 1.0],
        "middle":   [0.0, 1.0],
        "treble":   [0.0, 1.0],
        "bass1":    [0.0, 1.0],
        "volume":   [0.0, 1.0],
    },
}

# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def generate_sweep(length_sec=3.0, sr=48000, f0=20.0, f1=20000.0):
    """Exponential sine sweep (log-swept sine)."""
    t = np.arange(int(length_sec * sr)) / sr
    rate = np.log(f1 / f0) / length_sec
    phase = 2 * np.pi * f0 / rate * (np.exp(rate * t) - 1)
    sig = np.sin(phase)
    fade = int(0.005 * sr)
    sig[:fade] *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    return sig


def download_sweep(dst: Path, timeout=30):
    """Download the standard NAM training sweep from Google Drive."""
    import requests
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    print(f"  Downloading NAM sweep...", file=sys.stderr, end=" ", flush=True)
    r = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
    # Handle Google Drive confirmation page
    if "confirm" in r.url or "download-warning" in r.text:
        import re
        confirm_token = re.search(r'confirm=([0-9A-Za-z_-]+)', r.text)
        if confirm_token:
            url = f"https://drive.google.com/uc?export=download&confirm={confirm_token.group(1)}&id={GDRIVE_FILE_ID}"
            r = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"OK ({dst.name})", file=sys.stderr)


def load_or_create_sweep(path: Path | None, sr: int, length_sec: float):
    """
    Return (sweep_array, sr, source_label).
    If path is given and exists, load it.
    Otherwise try to download the standard NAM sweep.
    Fall back to generating our own.
    """
    if path and path.exists():
        data, file_sr = sf.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, file_sr, path.name

    # Try download
    download_dir = Path("/tmp")
    sweep_path = download_dir / SWEEP_FILENAME
    if not sweep_path.exists():
        try:
            download_sweep(sweep_path)
        except Exception as e:
            print(f"  Download failed ({e}); generating sweep instead", file=sys.stderr)
            sig = generate_sweep(length_sec=length_sec, sr=sr)
            return sig, sr, "generated"
    data, file_sr = sf.read(str(sweep_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, file_sr, sweep_path.name


def apply_gain(audio: np.ndarray, db: float):
    """Scale audio to a given RMS level in dBFS (relative to FS sine)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    target = 10 ** (db / 20) / np.sqrt(2)
    return audio * (target / rms)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_lhs(param_ranges: dict, n_samples: int, seed=42):
    """Latin Hypercube Sampling over the parameter space."""
    names = list(param_ranges.keys())
    bounds = np.array([param_ranges[n] for n in names])
    lo, hi = bounds[:, 0], bounds[:, 1]
    if LatinHypercube is not None:
        sampler = LatinHypercube(d=len(names), seed=seed, optimization="random-cd")
        samples = sampler.random(n=n_samples)
    else:
        rng = np.random.default_rng(seed)
        samples = rng.uniform(0, 1, size=(n_samples, len(names)))
    scaled = lo + samples * (hi - lo)
    return [dict(zip(names, row)) for row in scaled]


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------

def run_harness(harness_exe: str, circuit: str, sr: int, oversample: int,
                input_wav: Path, output_wav: Path, params: dict) -> bool:
    cmd = [
        harness_exe,
        "--input", str(input_wav),
        "--output", str(output_wav),
        "--circuit", circuit,
        "--sr", str(sr),
        "--oversample", str(oversample),
    ]
    if params:
        cmd += ["--params", ",".join(f"{k}={v:.10g}" for k, v in params.items())]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic training data for parametric NAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --output-dir ./dataset --samples 40\n"
            "  %(prog)s --circuit marshall_jcm800_2203_preamp_modded --samples 50\n"
            "  %(prog)s --sweep /path/to/sweep.wav --samples 30\n"
        ),
    )
    ap.add_argument("--circuit", default="marshall_jcm800_2203_preamp_modded",
                    help="Circuit ID (default: %(default)s)")
    ap.add_argument("--samples", type=int, default=40,
                    help="Number of LHS samples (default: %(default)s)")
    ap.add_argument("--sr", type=int, default=48000,
                    help="Sample rate (default: %(default)s)")
    ap.add_argument("--oversample", type=int, default=2, choices=[2, 4],
                    help="Oversample factor (default: %(default)s)")
    ap.add_argument("--harness",
                    default="/Users/USER/work/livespice-emitter/harness/build_release/harness",
                    help="Path to harness binary")
    ap.add_argument("--sweep", type=Path, default=None,
                    help="Path to sweep WAV (downloads NAM standard if not given)")
    ap.add_argument("--length", type=float, default=3.0,
                    help="Sweep length in seconds when generating (default: %(default)s)")
    ap.add_argument("--trim-seconds", type=float, default=None,
                    help="Trim loaded sweep to first N seconds (for quick prototyping)")
    ap.add_argument("--input-levels", nargs="+", type=float, default=[-12, -6],
                    help="Input levels in dBFS (default: %(default)s)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for the dataset")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for LHS (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without running harness")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        print(f"Output directory {out_dir} already exists; will overwrite", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.circuit not in CIRCUIT_PARAM_RANGES:
        print(f"Unknown circuit '{args.circuit}'. Known: {list(CIRCUIT_PARAM_RANGES.keys())}",
              file=sys.stderr)
        sys.exit(1)
    param_ranges = CIRCUIT_PARAM_RANGES[args.circuit]
    param_names = list(param_ranges.keys())

    # 1. Load / download / generate sweep
    print("=== Sweep ===", file=sys.stderr)
    sweep_data, sweep_sr, sweep_label = load_or_create_sweep(args.sweep, args.sr, args.length)
    if sweep_sr != args.sr:
        print(f"  Resampling from {sweep_sr} to {args.sr} Hz", file=sys.stderr)
        from scipy import signal as sp_signal
        ratio = args.sr / sweep_sr
        new_len = int(len(sweep_data) * ratio)
        sweep_data = sp_signal.resample(sweep_data, new_len)
        sweep_sr = args.sr
    print(f"  Source: {sweep_label}, {sweep_sr} Hz, {len(sweep_data)/sweep_sr:.2f}s",
          file=sys.stderr)

    if args.trim_seconds is not None:
        trim_samples = int(args.trim_seconds * sweep_sr)
        if trim_samples < len(sweep_data):
            sweep_data = sweep_data[:trim_samples]
            print(f"  Trimmed to first {args.trim_seconds:.1f}s ({len(sweep_data)} samples)",
                  file=sys.stderr)

    # 2. Save level-adjusted sweeps
    sweep_files: dict[float, Path] = {}
    for db in args.input_levels:
        sig = apply_gain(sweep_data, db)
        path = out_dir / f"input_{db:.0f}dBFS.wav"
        sf.write(str(path), sig, sweep_sr, subtype="FLOAT")
        sweep_files[db] = path
        print(f"  Input level {db:+.0f} dBFS -> {path.name}", file=sys.stderr)

    # 3. Sample param space
    print(f"\n=== Sampling ({args.samples} LHS samples, {len(param_names)} params) ===",
          file=sys.stderr)
    params_list = sample_lhs(param_ranges, args.samples, seed=args.seed)

    # 4. Prepare output directory structure
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Save config
    config = {
        "circuit":         args.circuit,
        "sample_rate":     sweep_sr,
        "oversample":      args.oversample,
        "num_params":      len(param_names),
        "param_names":     param_names,
        "param_ranges":    param_ranges,
        "n_samples":       args.samples,
        "seed":            args.seed,
        "input_levels_dbfs": args.input_levels,
        "sweep_source":    sweep_label,
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Save master params CSV
    csv_path = out_dir / "params.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id"] + param_names)
        for i, p in enumerate(params_list):
            w.writerow([f"{i:06d}"] + [f"{p[n]:.10g}" for n in param_names])
    print(f"  params.csv written ({args.samples} rows)", file=sys.stderr)

    # 5. Generate output for each sample
    total = args.samples * len(args.input_levels)
    print(f"\n=== Generating {args.samples} samples × {len(args.input_levels)} levels"
          f" = {total} WAVs ===", file=sys.stderr)

    for i, params in enumerate(params_list):
        sample_id = f"{i:06d}"
        sample_dir = samples_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        # Save per-sample params
        (sample_dir / "params.json").write_text(json.dumps(params, indent=2))

        for db in args.input_levels:
            out_wav = sample_dir / f"output_{db:.0f}dBFS.wav"
            if args.dry_run:
                print(f"  [{i+1}/{args.samples}] {sample_id} @ {db:+.0f} dBFS "
                      f"(DRY RUN, would run harness)", file=sys.stderr)
                continue
            ok = run_harness(args.harness, args.circuit, sweep_sr, args.oversample,
                             sweep_files[db], out_wav, params)
            if ok:
                print(f"  [{i+1}/{args.samples}] {sample_id} @ {db:+.0f} dBFS", file=sys.stderr)
            else:
                print(f"  [{i+1}/{args.samples}] FAIL {sample_id} @ {db:+.0f} dBFS", file=sys.stderr)

    print(f"\nDone. Dataset in {out_dir.resolve()}", file=sys.stderr)
    print(f"  Config:   {out_dir / 'config.json'}")
    print(f"  Params:   {out_dir / 'params.csv'}")
    print(f"  Input:    {sweep_files[next(iter(sweep_files))].parent / 'input_*dBFS.wav'}")
    print(f"  Samples:  {samples_dir}/{'{sample_id}'}/output_*.wav")
    print(f"  Size:     {total} WAV files")


if __name__ == "__main__":
    main()
