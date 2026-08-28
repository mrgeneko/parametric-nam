#!/usr/bin/env python3
"""Capture a static (non-parametric) NAM model from a LiveSpice circuit at one fixed
knob/switch setting, training through the OFFICIAL upstream `neural-amp-modeler`
trainer (`nam-full`) instead of this repo's own FiLM-conditioned `param_train.py`.

WHY A SEPARATE PATH, NOT "BAKE A PARAMETRIC MODEL". See internal engineering notes:
this repo's own ParametricA2 architecture is residual-only (the head reads only the
final layer's output); official NAM's WaveNet is skip-accumulating (the head reads the
SUM of every layer's output). Same weights, different forward pass -- a baked/
reserialized model loads in a stock plugin but sounds wrong (measured round-trip
correlation ~0.18). Training directly through nam-full sidesteps this entirely: there
is no translation step, so nothing can get lost in translation.

WHAT THIS REUSES, UNCHANGED. Rendering goes through gen_dataset_from_schx.py's own CLI (the
documented "pin everything but one knob" workaround for its lack of a literal 0-knob
mode) -- not by importing its internals, to stay decoupled from its process-owning
thread-pool/signal-handler assumptions. gen_dataset_from_schx.py's own post-generation checks
(crest-factor divergence detection, convergence audit) already run as part of the
render. preflight.py's find_saturation_point IS imported directly (see
ensure_adequate_excitation) -- see EXCITATION ADEQUACY below for why that one's worth
the coupling.

EXCITATION ADEQUACY: A GIVEN INPUT FILE ISN'T AUTOMATICALLY A GOOD FIT FOR A GIVEN
CIRCUIT. Hit this for real on the JCM800 2203 power amp (sag): trained "in" 2.5
minutes (18 epochs) to a suspiciously good val ESR. Cause: T3K-sweep-v3.wav peaks at
0.99 V, but a preflight sweep of that exact circuit showed it doesn't reach 99% of its
own saturation ceiling until ~11.64 V input -- the standard excitation is calibrated
for a PREAMP's input level, and this schx has no preceding gain stage to drive it
that hard. The whole 190s render stayed in the linear region; the network learned an
easy near-linear function and never saw the amp's actual nonlinearity. Same failure
mode as the JCM800 gain-only PARAMETRIC excitation issue (see that config's blurb),
just discovered later because a static capture has no grid_adequacy-style pre-check.

Fix: ensure_adequate_excitation() runs a saturation sweep (preflight.py's own
find_saturation_point, at the EXACT pinned --setting -- not a generic probe) before
every render, compares the given excitation's peak against the measured onset, and
transparently scales the excitation up if it falls short. This is NOT opt-in; it runs
by default for every capture, because "does this excitation actually exercise this
circuit's nonlinearity" is not something a caller should have to remember to check by
hand -- that's exactly how the power-amp incident happened. A scaled excitation loses
NAM's exact-hash standard-input match (delay falls back to 0 -- an already-existing,
disclosed path, not a new one), which is a reasonable trade for training on a model
that has actually seen saturation.

DELAY: PREFER NAM'S OWN BLIP-BASED CALIBRATION, DON'T HAND-ROLL IT. A prototype run's
own envelope cross-correlation found a suspiciously clean, consistent ~24-sample peak
and (wrongly) trusted it: applying it as `nam-full`'s `delay` made training an order
of magnitude worse (ESR ~0.10 vs. ~0.004, ran to completion, not just slow to
converge). The correlation "peak" was reproducible (robust to envelope window
size/parity) but not real -- almost certainly a swept-tone excitation's own
self-similarity structure, reshaped by the circuit's frequency-dependent nonlinear
compression, producing a stable but meaningless envelope-correlation lag. Turned out
`sweepv5.wav` wasn't the right tool for this: the widely-used `sweep-v3.wav` (this ecosystem's
standard capture-sweep excitation source -- a synthesized sweep, not a real-playing recording,
despite its high crest factor) is, byte-for-byte (MD5 match), NAM's own
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
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from gen_dataset_from_schx import parse_schx_controls, resolve_knobs, input_provenance  # noqa: E402
from param_train import _input_level_dbu, _schx_input_v0dbfs  # noqa: E402
from find_saturation_point import find_saturation_point  # noqa: E402
from render_backends import LiveSpiceBackend  # noqa: E402

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
    """Shell out to gen_dataset_from_schx.py's documented single-permutation workaround:
    pin every control but one via --fixed-params, sweep that one remaining control
    with a single value via --knobs/--values. Produces exactly 1 permutation through
    the harness's normal render/retry/spike-detection/normalization path, unmodified."""
    items = list(setting.items())
    sweep_knob, sweep_val = items[0]
    fixed = ",".join(f"{k}={v}" for k, v in items[1:])

    cmd = [sys.executable, str(REPO_ROOT / "gen_dataset_from_schx.py"),
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
    subprocess.run([sys.executable, str(REPO_ROOT / "gen_dataset_from_schx.py"),
                    "--combine", str(output_dir)], check=True)

    cfg = json.loads((output_dir / "config.json").read_text())
    return cfg


def ensure_adequate_excitation(schx: str, setting: dict, input_wav: str, work_dir: Path,
                                margin: float = 1.2, probe_oversample: int = 8,
                                probe_iterations: int = 256) -> tuple:
    """Check whether `input_wav` actually drives this circuit (at the exact pinned
    `setting`) past its own saturation onset -- and if not, scale it up until it does.
    See the module docstring's EXCITATION ADEQUACY section for why this exists and why
    it isn't opt-in. Returns (report_dict_for_the_manifest, effective_input_path).

    margin=1.2: target the excitation's peak at 1.2x the measured 99%-onset voltage --
    comfortably past the point output stops growing (the JCM800 gain-only parametric
    fix used a similar ~1.1x margin against its own worst-case onset; 1.2x here since
    a static capture has just one setting to clear, not a whole knob grid's worst
    corner)."""
    v0dbfs = _schx_input_v0dbfs(schx)
    x, sr = sf.read(input_wav, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    peak_raw = float(np.abs(x).max())

    report = {"checked": False, "adjusted": False, "excitation_peak_v": None,
              "onset_99pct_v": None, "ceiling_v": None, "scale_applied": None}

    if v0dbfs is None:
        print("[capture_static] WARNING: schx has no V0dBFS -- skipping the "
              "excitation-adequacy check (can't reliably interpret levels).")
        return report, input_wav

    peak_v = peak_raw * v0dbfs
    print(f"[capture_static] excitation-adequacy check: probing saturation onset at "
          f"setting={setting} ...")
    def _progress(done, total, elapsed):
        print(f"  {done}/{total} amplitude probes rendered ({elapsed:.0f}s)", flush=True)
    backend = LiveSpiceBackend(schx, oversample=probe_oversample, iterations=probe_iterations)
    with tempfile.TemporaryDirectory() as scratch:
        sat = find_saturation_point(backend, setting, scratch, progress=_progress)
    report["checked"] = True
    report["excitation_peak_v"] = peak_v

    if sat is None or sat.get("onset_99pct_input_v") is None:
        print("[capture_static] WARNING: could not measure a saturation onset (all "
              "probe renders failed, or the circuit never plateaus within the probe "
              "range) -- proceeding with the excitation as given, unchecked.")
        return report, input_wav

    onset_v = sat["onset_99pct_input_v"]
    report["onset_99pct_v"] = onset_v
    report["ceiling_v"] = sat["ceiling_rms"]
    target_v = margin * onset_v
    print(f"[capture_static] saturation onset ~{onset_v:.3g} V, excitation peaks at "
          f"{peak_v:.3g} V (target >= {target_v:.3g} V, {margin}x onset)")

    if peak_v >= target_v:
        print("[capture_static] excitation adequately drives this circuit into saturation.")
        return report, input_wav

    scale = target_v / peak_v if peak_v > 1e-9 else 1.0
    scaled_path = work_dir / "excitation_scaled.wav"
    # subtype='FLOAT' is NOT optional: sf.write's default WAV subtype is PCM_16,
    # which silently clips anything outside +/-1.0 -- exactly the amplitude range this
    # scaled excitation needs to represent. Caught this the hard way in
    # find_saturation_point (preflight.py) -- see that fix's comment.
    sf.write(str(scaled_path), (x * scale).astype(np.float32), sr, subtype="FLOAT")
    print(f"[capture_static] excitation too quiet for this circuit at this setting -- "
          f"scaling x{scale:.3g} ({peak_v:.3g} V -> {target_v:.3g} V peak), using the "
          f"scaled copy. This breaks NAM's exact-hash standard-input match; delay "
          f"calibration falls back to 0 (an existing, disclosed path -- see "
          f"calibrate_and_write_data_config), not a new failure mode.")
    report["adjusted"] = True
    report["scale_applied"] = scale
    report["scaled_excitation_path"] = str(scaled_path)
    return report, str(scaled_path)


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
        # match is found -- confirmed by hitting this directly on the old sweepv5.wav (the
        # excitation this repo shipped before 2026-08-28).
        return None
    except Exception as e:
        # A FLOAT-subtype WAV (e.g. ensure_adequate_excitation's scaled copy) isn't a
        # NAM standard file anyway -- that's already the documented, accepted tradeoff
        # -- but detecting that "isn't a match" can itself crash first: nam.data's
        # primary reader (wavio -> stdlib `wave`) doesn't support IEEE-float WAVs
        # (wave.Error: unknown format: 3), and its own librosa fallback is broken in
        # THIS repo's venv from a cross-venv cffi version mismatch (this script mixes
        # parametric-nam/.venv's compiled _cffi_backend with neural-amp-modeler/venv's
        # pure-Python cffi via NAM_SITE_PACKAGES on sys.path -- a real, separate
        # environment wart, not worth fixing just to reach the same "not a standard
        # file" conclusion this except clause already reaches for the expected case).
        # Treat any such failure the same as "no match": fall back to delay=0.
        print(f"[capture_static] WARNING: standard-input detection failed ({type(e).__name__}: "
              f"{e}) -- treating as not-a-standard-file, falling back to delay=0.")
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
              "calibrated capture, use sweep-v3.wav or another NAM standard file.")
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


# Human-readable submodel names for the fleet's usual two widths -- purely cosmetic
# (shows up in nam-full's own logging/checkpoint bookkeeping, not in the exported
# .nam), falls back to "w<N>" for anything outside the usual pair.
_SUBMODEL_NAMES = {3: "lite", 8: "full"}


def _layer_config(channels: int) -> dict:
    return {
        "input_size": 1, "condition_size": 1, "channels": channels,
        "kernel_sizes": K_KERNEL_SIZES, "dilations": K_DILATIONS,
        "activation": "LeakyReLU", "gated": False,
        "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
    }


def write_model_and_learning_configs(widths: list, max_epochs: int, configs_dir: Path):
    """widths=[8] (a single value) trains one plain WaveNet, same as before. Two or
    more widths (default widths=[4, 8]) trains a PackedWaveNet instead: every listed
    width jointly, as one masked model in one nam-full run (net.name="PackedWaveNet",
    per nam_full_configs/models/wavenet_packed.json), exporting a single
    SlimmableContainer .nam with one discrete submodel per width -- this export shape
    was verified against a real released model (Deluxe Reverb.nam: 2 submodels,
    channels 3 and 8, max_value 0.5/1.0). Matches the fleet's own lite/full split
    without needing a separate capture per width.

    NOTE on --threshold-esr with 2+ widths: nam-full's own best-checkpoint monitor
    (whose filename used to be what this script's SIGINT-threshold polling read)
    tracks val_loss = the MEAN of every submodel's own val ESR for a packed run, not
    one width's ESR. Confirmed this actually mattered on a real run: the mean crossed
    0.005 while the narrower tier was still at 0.0066 and hadn't converged, stopping
    training early on the wider tier's good score masking the narrower one. Fixed by
    having run_nam_full's polling read each tier's own ESR from the tfevents log
    directly and require the WORST of them below threshold -- see _worst_tier_esr."""
    if len(widths) == 1:
        model_config = {
            "net": {"name": "WaveNet", "config": {
                "layers_configs": [_layer_config(widths[0])],
                "head_scale": 0.01,
            }},
            "loss": {"val_loss": "esr", "mrstft_weight": 0.0005},
            "optimizer": {"lr": 0.004, "weight_decay": 3.17e-07},
            "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.994}},
        }
    else:
        model_config = {
            "net": {"name": "PackedWaveNet", "config": {
                "submodels": [
                    {
                        "name": _SUBMODEL_NAMES.get(w, f"w{w}"),
                        "config": {
                            "layers_configs": [_layer_config(w)],
                            "head": None,
                            "head_scale": 0.01,
                        },
                    }
                    for w in widths
                ],
                "export": {"container_max_values": "uniform"},
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


def _worst_tier_esr(run_dir: Path) -> "float | None":
    """The checkpoint FILENAME's ESR (what the polling loop used to check directly) is
    only the whole run's aggregate val_loss -- for a packed [3,8]-style run, nam-full
    logs it as the MEAN of every submodel's own ESR (ESR_packed_0, ESR_packed_1, ... in
    the tfevents), not any one tier's real accuracy. Confirmed on a real run: averaged
    ESR dipped to 0.00488 (crossing a 0.005 threshold) while the narrower tier was
    still at 0.00663 and hadn't converged -- the wider tier's good score was masking it,
    and training stopped early on a lucky dip in the average, not real convergence.

    Reads the tfevents file directly for the latest value of every ESR_packed_N tag and
    returns the WORST (max) of them -- crossing the threshold now means every tier
    actually cleared the bar, not just their average. Falls back to the plain 'ESR' tag
    (single-width runs have no _packed_ tags at all, so there's no ambiguity to correct)
    if none are found. Returns None if the tfevents file doesn't exist yet or has no
    usable scalars (too early in training -- caller should just keep waiting)."""
    tf_glob = list(run_dir.glob("lightning_logs/version_0/events.out.tfevents.*"))
    if not tf_glob:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(tf_glob[0]), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        packed_tags = [t for t in tags if re.match(r"ESR_packed_\d+$", t)]
        use_tags = packed_tags if packed_tags else (["ESR"] if "ESR" in tags else [])
        if not use_tags:
            return None
        values = []
        for t in use_tags:
            pts = ea.Scalars(t)
            if pts:
                values.append(pts[-1].value)
        return max(values) if values else None
    except Exception:
        return None


def _current_epoch(ckpt_dir: Path) -> int:
    """Latest epoch reached, from the per-epoch checkpoint nam-full writes every epoch
    (checkpoint_epoch_epoch=NNNN.ckpt) -- unlike the top-k 'best' checkpoints, this one
    always exists and always reflects the current epoch, even once training has
    plateaued and stopped producing new bests."""
    epochs = [int(m.group(1)) for p in ckpt_dir.glob("checkpoint_epoch_epoch=*.ckpt")
              if (m := re.search(r"epoch=(\d+)", p.name))]
    return max(epochs) if epochs else 0


def run_nam_full(data_cfg: Path, model_cfg: Path, learning_cfg: Path, outdir: Path,
                  threshold_esr: float = None, poll_interval_s: float = 15.0,
                  min_epochs: int = 500) -> dict:
    """Run nam-full as a subprocess (kept as a genuine subprocess boundary, not
    reimplemented in-process -- full.py's own callback list is hardcoded in Python
    with no way to inject a threshold-stop via config, and reimplementing its
    training loop by hand risks drifting from the real thing's edge-case handling).

    Open-ended in practice: learning_config's max_epochs is a generous ceiling, not
    the real stop condition. Poll for the WORST tier's own ESR (see _worst_tier_esr)
    once at least min_epochs have run, and send SIGINT once it crosses threshold_esr.
    nam-full's own main() catches KeyboardInterrupt specifically and ALWAYS exports the
    best checkpoint in its `finally` block -- confirmed by reading it -- so this is a
    graceful stop, not a kill, and behaves exactly like a normal completion. Same idea
    as this ecosystem's own `touch <ckpt-dir>/STOP` convention for open-ended
    parametric training: an external signal, not a baked-in trainer feature.

    min_epochs=500 default: a threshold crossing in the first handful of epochs is far
    more likely to be an early noisy dip (exactly what happened on a real packed run --
    see _worst_tier_esr) than genuine, stable convergence; this is a floor under the
    threshold check, not a substitute for it -- crossing still requires threshold_esr,
    just not before min_epochs has given training a real chance to settle."""
    import signal

    if not NAM_FULL.exists():
        raise SystemExit(f"nam-full not found at {NAM_FULL} -- is neural-amp-modeler set up there?")
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(NAM_FULL), str(data_cfg), str(model_cfg), str(learning_cfg), str(outdir), "--no-show"]
    print(f"[capture_static] training: {' '.join(cmd)}")
    t0 = time.time()

    # SIGINT silently did nothing on a real run: sent it twice, 15 min apart, while
    # training kept advancing through 45+ epochs with no "KeyboardInterrupt" ever
    # printed -- nam-full's own except/finally in full.py (which DOES catch it and
    # export) never even ran. Root cause, confirmed by checking a live process's own
    # signal.getsignal(signal.SIGINT): it was SIG_IGN from the moment the process
    # started, before any of nam's code ran. POSIX shells set SIGINT (and SIGQUIT) to
    # be ignored for background jobs (this project routinely launches long captures
    # via `cmd &`/nohup) -- CPython's own startup respects an inherited SIG_IGN rather
    # than overriding it with the normal KeyboardInterrupt-raising handler, and nothing
    # downstream (nam-full, Lightning) ever restores it, since Lightning's own signal
    # handling only touches SIGTERM/SIGUSR1 (SLURM requeue), not SIGINT. So the
    # ordinary case -- capture_static.py itself started as (or descended from) a
    # background job -- silently produces a subprocess that can NEVER be interrupted
    # gracefully. Fix: force the child back to the default disposition explicitly,
    # regardless of what it inherited. Confirmed fixed: with this, a real run exited
    # in <1s of SIGINT with "Detected KeyboardInterrupt, attempting graceful
    # shutdown ..." printed and a real .nam exported.
    restore_sigint = lambda: signal.signal(signal.SIGINT, signal.SIG_DFL)

    if threshold_esr is None:
        subprocess.run(cmd, check=True, preexec_fn=restore_sigint)
        return _find_result(outdir, time.time() - t0)

    proc = subprocess.Popen(cmd, preexec_fn=restore_sigint)
    stopped_early = False
    while proc.poll() is None:
        time.sleep(poll_interval_s)
        run_dirs = sorted(glob.glob(str(outdir / "*")))
        if not run_dirs:
            continue
        run_dir = Path(run_dirs[-1])
        ckpt_dir = run_dir / "lightning_logs" / "version_0" / "checkpoints"
        if not ckpt_dir.exists():
            continue
        epoch = _current_epoch(ckpt_dir)
        if epoch < min_epochs:
            continue
        worst = _worst_tier_esr(run_dir)
        if worst is not None and worst <= threshold_esr:
            print(f"[capture_static] worst tier's ESR {worst:.4g} <= threshold "
                  f"{threshold_esr:.4g} at epoch {epoch} (>= min_epochs {min_epochs}) "
                  f"-- sending SIGINT for a graceful stop")
            proc.send_signal(signal.SIGINT)
            stopped_early = True
            break

    # Graceful export after SIGINT (checkpoint reload + comparison plots + .nam write)
    # took >300s on a heavy circuit (EVH 5150 Lead Full sag v30: still legitimately
    # working, not stuck -- confirmed by the .nam existing and loading cleanly once it
    # did finish). A single blocking proc.wait(timeout=...) that raises
    # TimeoutExpired is also the wrong shape here regardless of the exact number: it's
    # not exception-safe (a naive try/finally around it re-raises after cleanup runs,
    # crashing the script even when the export subsequently succeeds -- hit exactly
    # this). Poll in a loop instead, generous grace period, only escalate if it's
    # truly still running after that, and never let a wait() timeout propagate.
    grace_s, poll_s, waited = 1800, 15, 0
    while proc.poll() is None and waited < grace_s:
        time.sleep(poll_s)
        waited += poll_s
        if waited % 60 == 0:
            print(f"[capture_static] waiting for nam-full's graceful export... {waited}s")
    if proc.poll() is None:
        print(f"[capture_static] WARNING: nam-full still running {grace_s}s after SIGINT -- "
              "terminating. This likely means an incomplete/missing export, unlike the "
              "normal case above (which just needs patience, not intervention).")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[capture_static] WARNING: didn't exit after terminate(), killing")
            proc.kill()
            proc.wait(timeout=30)

    wall_s = time.time() - t0
    result = _find_result(outdir, wall_s)
    result["stopped_early"] = stopped_early
    return result


def patch_input_level_dbu(nam_path: Path, schx: str) -> "float | None":
    """Set input_level_dbu on every metadata block in an already-exported .nam (top-level
    AND per-submodel, if it's a container -- param_train.py's export_composite_nam writes
    the identical value into both, since it's a property of the schx's input stage, not
    of any one tier). nam-full's own export path (nam/train/full.py's main(), calling
    model.net.export()/.export_container() with no metadata argument at all) has no
    mechanism to set this -- it has to be patched in after the fact.

    Same derivation as param_train.py's _input_level_dbu/_schx_input_v0dbfs: read the
    schx's Circuit.Input V0dBFS and convert to dBu. Returns None (and leaves every
    metadata block's input_level_dbu as null, matching the parametric side's own
    "omitted" behavior) if the schx has no readable V0dBFS."""
    v0dbfs = _schx_input_v0dbfs(schx)
    input_level_dbu = _input_level_dbu(v0dbfs) if v0dbfs else None
    if input_level_dbu is None:
        print("[capture_static] WARNING: schx has no V0dBFS -- input_level_dbu will be "
              "omitted from the .nam (same as the parametric side's behavior).")

    def patch(obj):
        if isinstance(obj, dict):
            if "metadata" in obj and isinstance(obj["metadata"], dict):
                obj["metadata"]["input_level_dbu"] = input_level_dbu
            for v in obj.values():
                patch(v)
        elif isinstance(obj, list):
            for v in obj:
                patch(v)

    d = json.loads(nam_path.read_text())
    patch(d)
    nam_path.write_text(json.dumps(d, separators=(",", ":")))
    return input_level_dbu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schx", required=True)
    ap.add_argument("--setting", required=True, help="'Gain=0.7,Bass=0.5,...' -- every real control, no omissions")
    ap.add_argument("--input", required=True)
    ap.add_argument("--oversample", default="auto")
    ap.add_argument("--trunc-target", type=float, default=1e-3)
    ap.add_argument("--output", required=True, help="working/output directory")
    ap.add_argument("--widths", default="4,8",
                     help="comma-separated channel widths to train (default '4,8', the "
                          "fleet's lite/full split) -- 2+ widths trains one PackedWaveNet "
                          "jointly and exports a single SlimmableContainer .nam with one "
                          "submodel per width; a single width (e.g. '8') trains a plain "
                          "WaveNet, matching this script's pre-slimmable behavior")
    ap.add_argument("--threshold-esr", type=float, default=0.005,
                     help="stop training once the WORST tier's own val ESR crosses this "
                          "(default 0.005) -- for a packed (--widths with 2+ values) run, "
                          "checked per-tier via the tfevents log, not nam-full's own "
                          "checkpoint-filename ESR (that's the MEAN across tiers, which lets "
                          "a strong wide tier mask a still-undertrained narrow one -- see "
                          "_worst_tier_esr's docstring). Set to 0/negative to disable and "
                          "just run --max-epochs.")
    ap.add_argument("--min-epochs", type=int, default=500,
                     help="don't act on --threshold-esr before this many epochs (default "
                          "500) -- a crossing in the first handful of epochs is far more "
                          "likely to be an early noisy dip than genuine convergence")
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

    widths = sorted({int(w.strip()) for w in args.widths.split(",") if w.strip()})
    if not widths:
        raise SystemExit("--widths produced no valid channel counts")

    setting = parse_setting(args.setting)
    validate_setting_complete(args.schx, setting)

    excitation_report, effective_input = ensure_adequate_excitation(
        args.schx, setting, args.input, out)

    cfg = render(args.schx, setting, effective_input, args.oversample, args.trunc_target, ds_dir)

    wet_wav = extract_wet_wav(ds_dir, out / "wet.wav")
    dry_wav = ds_dir / "sweep.wav"

    calibration = calibrate_and_write_data_config(
        Path(args.input), dry_wav, wet_wav, configs_dir, args.val_seconds)
    model_cfg, learning_cfg = write_model_and_learning_configs(
        widths, args.max_epochs, configs_dir)
    data_cfg = configs_dir / "data.json"

    threshold_esr = args.threshold_esr if args.threshold_esr > 0 else None
    result = run_nam_full(data_cfg, model_cfg, learning_cfg, nam_out_dir,
                           threshold_esr=threshold_esr, min_epochs=args.min_epochs)

    # nam-full always names its output "model.nam" -- rename in place (same dir) to
    # encode the setting, so a directory of captures is browsable without opening
    # each manifest.json.
    named_path = result["nam_path"].with_name(setting_to_filename(setting, args.gear_model))
    result["nam_path"].rename(named_path)
    result["nam_path"] = named_path

    # nam-full's own export has no metadata hook for this (see patch_input_level_dbu's
    # docstring) -- same derivation param_train.py uses for the parametric side, patched
    # in after the fact since it isn't one.
    input_level_dbu = patch_input_level_dbu(result["nam_path"], args.schx)

    manifest = {
        "schx": args.schx,
        "setting": setting,
        "excitation_adequacy": excitation_report,
        "oversample_config": cfg.get("oversample"),
        "input_provenance": input_provenance(dry_wav),
        "output_scale": cfg.get("output_scale"),
        "delay_samples": calibration["delay"],
        "input_version": calibration["input_version"],
        "calibration_source": calibration["calibration_source"],
        "widths": widths,
        "max_epochs_ceiling": args.max_epochs,
        "threshold_esr": threshold_esr,
        "stopped_early_at_threshold": result.get("stopped_early", False),
        "val_seconds": args.val_seconds,
        "best_val_esr": result["best_esr"],
        "wall_clock_s": result["wall_s"],
        "gear_make": args.gear_make, "gear_model": args.gear_model, "gear_type": args.gear_type,
        "input_level_dbu": input_level_dbu,
        "nam_path": str(result["nam_path"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[capture_static] done in {result['wall_s']/60:.1f} min, "
          f"best val ESR={result['best_esr']}, model at {result['nam_path']}")
    print(f"[capture_static] manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
