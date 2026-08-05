#!/usr/bin/env python3
"""Capture a static (non-parametric) NAM model from a LiveSpice circuit at one fixed
knob/switch setting, training through the OFFICIAL upstream `neural-amp-modeler`
trainer (`nam-full`) instead of this repo's own FiLM-conditioned `param_train.py`.

WHY A SEPARATE PATH, NOT "BAKE A PARAMETRIC MODEL". See docs/spice_static_plan.md:
this repo's own ParametricA2 architecture is residual-only (the head reads only the
final layer's output); official NAM's WaveNet is skip-accumulating (the head reads the
SUM of every layer's output). Same weights, different forward pass -- a baked/
reserialized model loads in a stock plugin but sounds wrong (measured round-trip
correlation ~0.18). Training directly through nam-full sidesteps this entirely: there
is no translation step, so nothing can get lost in translation.

WHAT THIS REUSES, UNCHANGED. Rendering goes through batch_harness.py's own CLI (the
documented "pin everything but one knob" workaround for its lack of a literal 0-knob
mode) -- not by importing its internals, to stay decoupled from its process-owning
thread-pool/signal-handler assumptions. preflight.py's calibration logic is not
imported directly here; batch_harness.py's own post-generation checks (crest-factor
divergence detection, convergence audit) already run as part of the render.

DELAY: PREFER NAM'S OWN BLIP-BASED CALIBRATION, DON'T HAND-ROLL IT. A prototype run's
own envelope cross-correlation found a suspiciously clean, consistent ~24-sample peak
and (wrongly) trusted it: applying it as `nam-full`'s `delay` made training an order
of magnitude worse (ESR ~0.10 vs. ~0.004, ran to completion, not just slow to
converge). The correlation "peak" was reproducible (robust to envelope window
size/parity) but not real -- almost certainly a swept-tone excitation's own
self-similarity structure, reshaped by the circuit's frequency-dependent nonlinear
compression, producing a stable but meaningless envelope-correlation lag. Turned out
`sweepv5.wav` wasn't the right tool for this: `[redacted]-sweep-v3.wav` (this ecosystem's
usual real-playing excitation source) is, byte-for-byte (MD5 match), NAM's own
official "Version 3.0.0" standard input file -- the exact file `nam.train.core`'s
input-version detection and blip-based latency calibration are built around. Running
that calibration for real (`nam.train.core._calibrate_latency_v3`) on this circuit's
output found delay=0 (recommended=-1 after NAM's own conservative safety margin) --
matching the direct A/B result, not the hand-rolled correlation. Lesson: when the
input is one of NAM's recognized standard files, use `nam.train.core`'s own
detection/calibration/split-point logic (`_detect_input_version`, `_get_data_config`)
directly -- it's purpose-built, impulse-based, and already trusted by the wider
ecosystem. Only fall back to a bare `delay=0` (never a guess) for a non-standard,
custom-built excitation, and say so loudly in the manifest.
"""
import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from batch_harness import parse_schx_controls, resolve_knobs, input_provenance  # noqa: E402

NAM_VENV = Path.home() / "work" / "neural-amp-modeler" / "venv"
NAM_FULL = NAM_VENV / "bin" / "nam-full"
NAM_SITE_PACKAGES = next((NAM_VENV / "lib").glob("python*/site-packages"), None)
if NAM_SITE_PACKAGES:
    sys.path.insert(0, str(NAM_SITE_PACKAGES))

# The fleet's real "A2" architecture (param_train.py K_KERNEL_SIZES/K_DILATIONS),
# copied as literals -- NOT imported from param_train.py, so this script carries no
# runtime dependency on that (residual-head) training code. Verified against NAM's own
# reference config (nam/train/_resources/config_model_packed.json) to be identical.
K_KERNEL_SIZES = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6]
K_DILATIONS = [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13,
               1, 3, 7, 17, 41, 101, 239]


def parse_setting(s: str) -> dict:
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def setting_to_filename(setting: dict, gear_model: str = "") -> str:
    """'Bass=0.5, Gain=0.7' -> 'Bass0.50_Gain0.70.nam' -- sorted alphabetically for
    determinism (two captures of the same setting always produce the same name,
    regardless of --setting's argument order), 2 decimal places (avoids ambiguous
    trailing-zero differences, e.g. 0.7 vs 0.70 meaning the same knob position)."""
    prefix = re.sub(r"[^A-Za-z0-9]+", "", gear_model)
    parts = [f"{k}{v:.2f}" for k, v in sorted(setting.items())]
    name = "_".join(([prefix] if prefix else []) + parts)
    return f"{name}.nam"


def validate_setting_complete(schx: str, setting: dict):
    """Every real control must be pinned -- a static capture has no 'left at
    default' concept; each control's value is part of the run's identity."""
    controls = parse_schx_controls(schx)
    real_names = set(controls.values())
    given_names = set()
    resolved = resolve_knobs(list(setting.keys()), controls)
    given_names = set(resolved.values())
    missing = real_names - given_names
    if missing:
        raise SystemExit(f"--setting is missing control(s): {', '.join(sorted(missing))}\n"
                          f"Every real control must be pinned for a static capture.")


def render(schx: str, setting: dict, input_wav: str, oversample: str,
           trunc_target: float, output_dir: Path) -> dict:
    """Shell out to batch_harness.py's documented single-permutation workaround:
    pin every control but one via --fixed-params, sweep that one remaining control
    with a single value via --knobs/--values. Produces exactly 1 permutation through
    the harness's normal render/retry/spike-detection/normalization path, unmodified."""
    items = list(setting.items())
    sweep_knob, sweep_val = items[0]
    fixed = ",".join(f"{k}={v}" for k, v in items[1:])

    cmd = [sys.executable, str(REPO_ROOT / "batch_harness.py"),
           "--backend", "livespice", "--schx", schx,
           "--knobs", sweep_knob, "--values", str(sweep_val),
           "--input", input_wav, "--output", str(output_dir),
           "--skip-transient-check"]
    if fixed:
        cmd += ["--fixed-params", fixed]
    if oversample == "auto":
        cmd += ["--oversample", "auto", "--trunc-target", str(trunc_target)]
    else:
        cmd += ["--oversample", str(oversample)]

    print(f"[capture_static] rendering: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "batch_harness.py"),
                    "--combine", str(output_dir)], check=True)

    cfg = json.loads((output_dir / "config.json").read_text())
    return cfg


def extract_wet_wav(output_dir: Path, out_path: Path) -> Path:
    data = np.load(output_dir / "outputs.npy")
    assert data.shape[0] == 1, f"expected exactly 1 permutation, got {data.shape[0]}"
    sf.write(str(out_path), data[0].astype(np.float32), 48000)
    return out_path


def detect_standard_input(input_wav: Path):
    """Check whether input_wav is one of NAM's own recognized standard input files
    (content-hash match, nam.train.core._detect_input_version). If so, we get its
    real blip-based latency calibration and correct train/val split points for free
    instead of guessing at either."""
    from nam.train.core import _detect_input_version, _InputValidationError
    try:
        version, strong_match = _detect_input_version(str(input_wav))
    except _InputValidationError:
        # Despite the (version, strong_match) return signature implying a graceful
        # None for "no match", it actually RAISES when neither a strong nor weak
        # match is found -- confirmed by hitting this directly on sweepv5.wav.
        return None
    if version is None:
        return None
    print(f"[capture_static] input matches NAM standard version {version} "
          f"({'strong' if strong_match else 'weak'} match)")
    return version


def calibrate_and_write_data_config(input_wav: Path, dry_wav: Path, wet_wav: Path,
                                     configs_dir: Path, val_seconds: float) -> dict:
    """Prefer NAM's own calibration for a recognized standard input file (impulse-
    based, not correlation-based -- see module docstring for why this matters).
    Falls back to a bare delay=0 for a non-standard/custom excitation, never a guess."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    version = detect_standard_input(dry_wav)

    if version is not None:
        from nam.train.core import _analyze_latency, _get_data_config
        latency_result = _analyze_latency(
            user_latency=None, input_version=version,
            input_path=str(dry_wav), output_path=str(wet_wav), silent=True)
        delay = latency_result.calibration.recommended
        if delay is None:
            print("[capture_static] WARNING: NAM's blip calibration found no impulse "
                  "response -- falling back to delay=0. Check the render for issues.")
            delay = 0
        print(f"[capture_static] NAM blip-based calibration: delay={delay}")
        data_config = _get_data_config(
            input_version=version, input_path=str(dry_wav), output_path=str(wet_wav),
            ny=8192, latency=delay)
        source = "nam_standard_calibration"
    else:
        print("[capture_static] input is not a NAM-recognized standard file -- "
              "using delay=0 (not a guess: matches param_train.py's own convention "
              "for LiveSpice renders, and this ecosystem's envelope-correlation "
              "delay-guessing has been shown to actively mislead). For a rigorously "
              "calibrated capture, use [redacted]-sweep-v3.wav or another NAM standard file.")
        delay = 0
        data_config = {
            "train": {"start_seconds": None, "stop_seconds": -val_seconds, "ny": 8192},
            "validation": {"start_seconds": -val_seconds, "stop_seconds": None, "ny": None},
            "common": {
                "x_path": str(dry_wav), "y_path": str(wet_wav), "delay": delay,
                # The stock 0.4s pre-silence check assumes a deliberate silent gap
                # before the val split, which a NAM standard file has and a custom
                # fleet excitation doesn't (yet -- build_excitation.py doesn't build
                # one in). Disabled here for the non-standard path only.
                "require_input_pre_silence": None,
            },
        }
        source = "delay_zero_fallback"

    with open(configs_dir / "data.json", "w") as f:
        json.dump(data_config, f, indent=4)
    return {"delay": delay, "input_version": str(version) if version else None,
            "calibration_source": source}


def write_model_and_learning_configs(channels: int, max_epochs: int, configs_dir: Path):
    model_config = {
        "net": {"name": "WaveNet", "config": {
            "layers_configs": [{
                "input_size": 1, "condition_size": 1, "channels": channels,
                "kernel_sizes": K_KERNEL_SIZES, "dilations": K_DILATIONS,
                "activation": "LeakyReLU", "gated": False,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
            }],
            "head_scale": 0.01,
        }},
        "loss": {"val_loss": "esr", "mrstft_weight": 0.0005},
        "optimizer": {"lr": 0.004, "weight_decay": 3.17e-07},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.994}},
    }
    with open(configs_dir / "model.json", "w") as f:
        json.dump(model_config, f, indent=4)

    learning_config = {
        "train_dataloader": {"batch_size": 16, "shuffle": True, "pin_memory": True,
                              "drop_last": True, "num_workers": 0},
        "val_dataloader": {},
        # "auto" (Lightning's own default) picks CUDA/MPS/CPU at runtime -- was
        # hardcoded "mps" (Apple-Silicon-only), broken on Linux/CUDA machines.
        "trainer": {"accelerator": "auto", "devices": 1, "max_epochs": max_epochs},
        "trainer_fit_kwargs": {},
    }
    with open(configs_dir / "learning.json", "w") as f:
        json.dump(learning_config, f, indent=4)

    return configs_dir / "model.json", configs_dir / "learning.json"


def _find_result(outdir: Path, wall_s: float) -> dict:
    run_dirs = sorted(glob.glob(str(outdir / "*")))
    if not run_dirs:
        raise SystemExit("nam-full produced no output directory")
    run_dir = Path(run_dirs[-1])
    nam_files = list(run_dir.glob("*.nam")) + list(run_dir.glob("**/*.nam"))
    if not nam_files:
        raise SystemExit(f"no .nam file found under {run_dir}")

    best_esr = None
    ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
    if ckpt_dir.exists():
        esrs = []
        for p in ckpt_dir.glob("*ESR=*"):
            m = re.search(r"ESR=([0-9.e+-]+)", p.name)
            if m:
                esrs.append(float(m.group(1)))
        if esrs:
            best_esr = min(esrs)

    return {"nam_path": nam_files[0], "run_dir": run_dir, "wall_s": wall_s, "best_esr": best_esr}


def run_nam_full(data_cfg: Path, model_cfg: Path, learning_cfg: Path, outdir: Path,
                  threshold_esr: float = None, poll_interval_s: float = 15.0) -> dict:
    """Run nam-full as a subprocess (kept as a genuine subprocess boundary, not
    reimplemented in-process -- full.py's own callback list is hardcoded in Python
    with no way to inject a threshold-stop via config, and reimplementing its
    training loop by hand risks drifting from the real thing's edge-case handling).

    Open-ended in practice: learning_config's max_epochs is a generous ceiling, not
    the real stop condition. Poll checkpoint filenames (which already embed ESR,
    e.g. '..._ESR=6.771e-03_...ckpt') and send SIGINT once the best crosses
    threshold_esr. nam-full's own main() catches KeyboardInterrupt specifically and
    ALWAYS exports the best checkpoint in its `finally` block -- confirmed by reading
    it -- so this is a graceful stop, not a kill, and behaves exactly like a normal
    completion. Same idea as this ecosystem's own `touch <ckpt-dir>/STOP` convention
    for open-ended parametric training: an external signal, not a baked-in trainer
    feature."""
    import signal

    if not NAM_FULL.exists():
        raise SystemExit(f"nam-full not found at {NAM_FULL} -- is neural-amp-modeler set up there?")
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(NAM_FULL), str(data_cfg), str(model_cfg), str(learning_cfg), str(outdir), "--no-show"]
    print(f"[capture_static] training: {' '.join(cmd)}")
    t0 = time.time()

    if threshold_esr is None:
        subprocess.run(cmd, check=True)
        return _find_result(outdir, time.time() - t0)

    proc = subprocess.Popen(cmd)
    stopped_early = False
    try:
        while proc.poll() is None:
            time.sleep(poll_interval_s)
            run_dirs = sorted(glob.glob(str(outdir / "*")))
            if not run_dirs:
                continue
            ckpt_dir = Path(run_dirs[-1]) / "lightning_logs" / "version_0" / "checkpoints"
            if not ckpt_dir.exists():
                continue
            esrs = [float(m.group(1)) for p in ckpt_dir.glob("*ESR=*")
                    if (m := re.search(r"ESR=([0-9.e+-]+)", p.name))]
            if esrs and min(esrs) <= threshold_esr:
                print(f"[capture_static] best ESR {min(esrs):.4g} <= threshold "
                      f"{threshold_esr:.4g} -- sending SIGINT for a graceful stop")
                proc.send_signal(signal.SIGINT)
                stopped_early = True
                break
        proc.wait(timeout=300)
    finally:
        if proc.poll() is None:
            print("[capture_static] WARNING: nam-full didn't exit after SIGINT within "
                  "300s, terminating (may not have exported cleanly)")
            proc.terminate()
            proc.wait(timeout=30)

    wall_s = time.time() - t0
    result = _find_result(outdir, wall_s)
    result["stopped_early"] = stopped_early
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schx", required=True)
    ap.add_argument("--setting", required=True, help="'Gain=0.7,Bass=0.5,...' -- every real control, no omissions")
    ap.add_argument("--input", required=True)
    ap.add_argument("--oversample", default="auto")
    ap.add_argument("--trunc-target", type=float, default=1e-3)
    ap.add_argument("--output", required=True, help="working/output directory")
    ap.add_argument("--channels", type=int, default=8, help="fleet 'full' tier width")
    ap.add_argument("--threshold-esr", type=float, default=0.01,
                     help="stop training once best val ESR crosses this (default 0.01 -- "
                          "looser than the fleet's typical 0.007-0.009 parametric-model "
                          "target, appropriate for a quick static capture). Set to 0/negative "
                          "to disable and just run --max-epochs.")
    ap.add_argument("--max-epochs", type=int, default=1000,
                     help="ceiling, not the primary stop condition when --threshold-esr is "
                          "set (default) -- training stops at whichever comes first")
    ap.add_argument("--val-seconds", type=float, default=9.0)
    ap.add_argument("--gear-make", default="")
    ap.add_argument("--gear-model", default="")
    ap.add_argument("--gear-type", default="pedal")
    args = ap.parse_args()

    out = Path(args.output).expanduser()
    ds_dir = out / "ds"
    configs_dir = out / "configs"
    nam_out_dir = out / "nam_out"
    out.mkdir(parents=True, exist_ok=True)

    setting = parse_setting(args.setting)
    validate_setting_complete(args.schx, setting)

    cfg = render(args.schx, setting, args.input, args.oversample, args.trunc_target, ds_dir)

    wet_wav = extract_wet_wav(ds_dir, out / "wet.wav")
    dry_wav = ds_dir / "sweep.wav"

    calibration = calibrate_and_write_data_config(
        Path(args.input), dry_wav, wet_wav, configs_dir, args.val_seconds)
    model_cfg, learning_cfg = write_model_and_learning_configs(
        args.channels, args.max_epochs, configs_dir)
    data_cfg = configs_dir / "data.json"

    threshold_esr = args.threshold_esr if args.threshold_esr > 0 else None
    result = run_nam_full(data_cfg, model_cfg, learning_cfg, nam_out_dir, threshold_esr=threshold_esr)

    # nam-full always names its output "model.nam" -- rename in place (same dir) to
    # encode the setting, so a directory of captures is browsable without opening
    # each manifest.json.
    named_path = result["nam_path"].with_name(setting_to_filename(setting, args.gear_model))
    result["nam_path"].rename(named_path)
    result["nam_path"] = named_path

    manifest = {
        "schx": args.schx,
        "setting": setting,
        "oversample_config": cfg.get("oversample"),
        "input_provenance": input_provenance(dry_wav),
        "output_scale": cfg.get("output_scale"),
        "delay_samples": calibration["delay"],
        "input_version": calibration["input_version"],
        "calibration_source": calibration["calibration_source"],
        "channels": args.channels,
        "max_epochs_ceiling": args.max_epochs,
        "threshold_esr": threshold_esr,
        "stopped_early_at_threshold": result.get("stopped_early", False),
        "val_seconds": args.val_seconds,
        "best_val_esr": result["best_esr"],
        "wall_clock_s": result["wall_s"],
        "gear_make": args.gear_make, "gear_model": args.gear_model, "gear_type": args.gear_type,
        "nam_path": str(result["nam_path"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[capture_static] done in {result['wall_s']/60:.1f} min, "
          f"best val ESR={result['best_esr']}, model at {result['nam_path']}")
    print(f"[capture_static] manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
