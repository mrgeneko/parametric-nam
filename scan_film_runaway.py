#!/usr/bin/env python3
"""Scan a published .nam bundle for the FiLM/LeakyReLU runaway instability.

Background: see internal engineering notes. Two published models (the tweed-style
amp's Full sag, and the old pre-fix reverse-linear-drive pedal) were found to blow up 80-260x at a
narrow (knob-corner x real-transient) combination the training excitation
under-covered. This tool reproduces that check generically, against ANY
published .nam -- no training checkpoint or dataset required, since it
reconstructs the model directly from the exported weights (ParametricA2's own
_load_weight_block round-trips this exactly).

Method: enumerate the knob-space hypercube corners (all-min, all-max, each
knob solo-extreme -- the same reduced corner set already used elsewhere in
this fleet for knob-grid design, per internal engineering notes) plus
the center, run each corner against a reference clip with varied, hard
transient dynamics (--reference -- a synthesized capture sweep like
T3K-sweep-v3.wav works as well as a real recording; no bundled default,
bring your own) in windowed chunks, and flag any window where the
predicted peak is anomalous relative to that model's OWN typical output level.

Scans EVERY submodel in the container, not just the widest tier. The runaway mechanism is a
property of a tier's own layer weights, which differ per tier -- a clean widest-tier scan says
nothing about a narrower tier's weights (found the hard way: this tool used to load only
subs[-1], and every best_<tier>.param.nam in this fleet is a full multi-tier snapshot from
that tier's own best epoch, not a size-reduced single-tier export -- so there was previously no
way to reach a narrower tier's weights through this CLI at all). Works for any tier count (a
3-way [3,5,8] slimmable container scans all three), not just 2.

Usage:
  python scan_film_runaway.py --nam PATH/TO/model.param.nam \
      --reference PATH/TO/dynamic_clip.wav [--chunk-s 5.0] \
      [--flag-ratio 8.0] [--flag-abs 3.0]

  # full trained grid instead of the reduced corner set (see internal engineering notes
  # for measured cost: the default reduced set is ~2 min for 13 corners on a 190s reference;
  # --config scans the FULL grid batched, e.g. ~2-3 min for Tweed's 972 combinations at the
  # default --batch-size, vs. hours if it re-used the old one-forward-call-per-corner loop):
  python scan_film_runaway.py --nam PATH/TO/model.param.nam \
      --config ~/work/parametric-nam-models/amps/myamp-full-sag/config.toml
"""
import argparse
import csv
import itertools
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from param_train import ParametricA2
from run_pipeline import load_config
from gen_dataset_from_schx import grid_combinations

SR = 48000


def load_all_submodels(nam_path: str):
    """Every submodel in the container, narrowest first (SlimmableContainer's own storage
    order, ascending by max_value) -- see the module docstring for why ALL of them, not just
    the widest. Returns a list of (model, param_names, channels) tuples, one per tier."""
    d = json.loads(Path(nam_path).read_text())
    subs = d["config"]["submodels"]
    out = []
    for sub in subs:
        channels = sub["model"]["config"]["layers"]
        parametric = sub["model"]["config"]["parametric"]
        param_metas = parametric["parameters"]
        param_names = [p["name"] for p in param_metas]
        weights = sub["model"]["weights"]
        # LoRA-enabled export (schema_version 2, "type":"film+lora") declares its rank
        # directly in the config -- no state-dict detection needed the way
        # export_checkpoint.py's detect_lora_rank() has to do for a training checkpoint.
        # Every pre-LoRA export has no "lora" key at all -> rank 0, unchanged behavior.
        lora_rank = parametric.get("lora", {}).get("rank", 0)
        model = ParametricA2(channels=channels, num_params=len(param_names), lora_rank=lora_rank)
        expected = model.weight_count()
        if len(weights) != expected:
            raise SystemExit(f"{nam_path}: weight count mismatch ({len(weights)} vs {expected}) "
                              f"-- export format assumption may not hold for this file")
        model.load_weights(weights)
        model.eval()
        out.append((model, param_names, channels))
    return out


def hypercube_corners(param_names, max_full_corners: int = 512):
    """All-min, all-max, each knob solo-extreme (rest at 0.5), center, PLUS the full
    binary hypercube (every knob independently at 0.0 or 1.0, 2**n corners; all-min/
    all-max are 2 of them, deduped below).

    Used to be solo-only. That missed a real corner: the tweed-style amp's Full shipped blowup
    was two volume knobs at min held SIMULTANEOUSLY with three tone knobs at max -- solo
    holds every OTHER knob at 0.5 (center), never at another extreme, so a mixed
    some-low-some-high corner was never scanned. 2**n is exponential; capped at
    max_full_corners (default 512, up to 9 params) so a many-knob device doesn't silently
    balloon this -- pass a smaller cap (it'll raise) if that's ever hit intentionally."""
    n = len(param_names)
    corners = [("all-min", [0.0] * n), ("all-max", [1.0] * n), ("center", [0.5] * n)]
    for i, name in enumerate(param_names):
        lo = [0.5] * n; lo[i] = 0.0
        hi = [0.5] * n; hi[i] = 1.0
        corners.append((f"{name}=0-solo", lo))
        corners.append((f"{name}=1-solo", hi))

    if n:
        n_full = 2 ** n
        if n_full > max_full_corners:
            raise ValueError(f"full hypercube would be {n_full} corners (> max_full_corners="
                             f"{max_full_corners}) for {n} params")
        seen = {tuple(v) for _, v in corners}
        for bits in itertools.product((0.0, 1.0), repeat=n):
            vals = list(bits)
            key = tuple(vals)
            if key in seen:
                continue
            seen.add(key)
            corners.append((",".join(f"{nm}={'1' if b else '0'}"
                                     for nm, b in zip(param_names, bits)), vals))
    return corners


def device_sweep_floor(nam_path):
    """The lowest frequency this device's excitation deliberately covers, in Hz.

    Returns (floor_hz, source) or (None, None).

    WHY THE PROBE MUST BE HIGH-PASSED PER DEVICE (2026-09-17). build_excitation.py sweeps its
    amplitude-stepped chirps from --chirp-f0 upward; below that the excitation carries only
    whatever incidental content the --sweep-file segment happens to have, at whatever level it
    happens to be. So chirp_f0 is the bottom of the band training deliberately covers, and a
    reference clip reaching below it probes a band the model was never taught -- which reads as
    instability but is just absence of training signal.

    That is not hypothetical: a fixed 20 Hz probe floor made a published Mesa Orange w4 read 341
    on a SUSTAINED 20 Hz tone while being clean from 45 Hz up and clean on real playing, because
    20 Hz sits in the transition band of the 18 Hz capture-chain high-pass. Raising the clip's own
    floor to 40 Hz fixed that -- but 40 is not right either: this fleet already records floors of
    40 AND 80 Hz (marshall-the-guv-nor, proco-rat), and build_excitation.py's default moved to 15
    in 98cbb16. Any single hardcoded number is wrong for part of the fleet in one direction or the
    other, so derive it.

    Reads the bundle's dataset_config.json recipe. Accepts both key spellings: `chirp_f0` (current)
    and `sweep_f0` (pre-c03f54b rename), since published bundles carry a mix.
    """
    cfg_path = Path(nam_path).parent / "dataset_config.json"
    if not cfg_path.exists():
        return None, None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return None, None
    args = ((cfg.get("input") or {}).get("build_recipe") or {}).get("args") or {}
    for key in ("chirp_f0", "sweep_f0"):
        v = args.get(key)
        if v:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f > 0:
                return f, f"bundle recipe {key}={f:g} Hz"
    return None, None


def device_input_level(nam_path):
    """The peak input level this device was TRAINED at, in volts. Returns (level, source).

    WHY THE SCAN MUST SCALE PER DEVICE (2026-09-17). A reference clip has one absolute
    amplitude, but this fleet's excitations are sized per circuit by the knee+sat95 onset
    metric and span 122x -- 0.44 V for a Mesa RED gain-master, 53.8 V for a 5-knob Orange.
    Running every model against the same raw clip therefore over-drives some and barely
    tickles others. Measured: the bundled 1.5 V reference is 3.4x hot for the RED, whose w4
    median peak reads 2291 instead of 0.569 -- a ~4000x artefact of out-of-distribution level,
    not a property of the model -- while for a 53.8 V device the same clip is 36x too QUIET.
    Scaling restored the check's discrimination: 8 flagged windows vs 222.

    Source order, most authoritative first:
      1. the bundle's dataset_config.json `input.peak` -- the actual excitation the model was
         fit to. Authoritative, and present in every release_run.sh bundle.
      2. the .nam's own metadata.input_level_dbu -> V0dBFS. Works for a bare .nam, but it is a
         DIFFERENT QUANTITY (volts per digital full-scale, read off the .schx's Circuit.Input)
         and param_train._input_level_dbu's own docstring calls it an assumption rather than a
         measurement -- so it is a fallback that warns, not a peer.
    Returns (None, None) when neither is available; the caller then leaves the clip alone.
    """
    cfg_path = Path(nam_path).parent / "dataset_config.json"
    if cfg_path.exists():
        try:
            peak = (json.loads(cfg_path.read_text()).get("input") or {}).get("peak")
            if peak and float(peak) > 0:
                return float(peak), "bundle dataset_config.json input.peak"
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    try:
        meta = json.loads(Path(nam_path).read_text()).get("metadata") or {}
        dbu = meta.get("input_level_dbu")
        if dbu is not None:
            v0dbfs = (0.7746 * (10.0 ** (float(dbu) / 20.0))) * math.sqrt(2.0)
            if v0dbfs > 0:
                return v0dbfs, "metadata.input_level_dbu (ASSUMED nominal, not the training level)"
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return None, None


def bundle_grid_corners(nam_path, param_names):
    """The exact trained combinations, read from the published bundle's own dataset_params.csv.

    WHY THIS IS THE DEFAULT (2026-09-17). The reduced hypercube_corners() set probes every knob
    at literal 0.0/1.0 regardless of what the device was actually trained on -- and most grids
    do not reach those values. Scanning there measures EXTRAPOLATION, not instability, and
    reports it in the same units as a real defect. Measured cost of that confusion on one
    afternoon's fleet scan: a Dumble variant read 43x and a 5-knob Mesa read 47,461x at
    out-of-grid corners, while both are 0/16 and 0/576 across their real grids. Three of nine
    flagged bundles were false alarms on that basis alone.

    A published bundle already carries the answer -- release_run.sh stages dataset_params.csv
    beside the .nam -- so unlike full_grid_corners() this needs no --config and no access to
    the training repo. Returns None when there is no bundle CSV to read (a bare .nam), leaving
    the caller to fall back and SAY SO rather than silently extrapolate.
    """
    csv_path = Path(nam_path).parent / "dataset_params.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows or any(n not in rows[0] for n in param_names):
        return None
    out, seen = [], set()
    for r in rows:
        try:
            vals = [float(r[n]) for n in param_names]
        except (TypeError, ValueError):
            return None
        key = tuple(vals)
        if key in seen:
            continue
        seen.add(key)
        out.append((",".join(f"{n}={v:g}" for n, v in zip(param_names, vals)), vals))
    return out or None


def full_grid_corners(config_path, param_names):
    """The FULL Cartesian product of --config's own [knobs] grid -- the exact combinations
    gen_dataset_from_schx.py rendered for training (e.g. 972 for Tweed), not a reduced sample.
    Needs --config because a .nam's own metadata only carries min/max/default per knob
    (config.parametric.parameters), not the discrete grid VALUES actually trained on --
    that's a gen_dataset_from_schx.py/config.toml-level fact this tool has no other way to recover.
    """
    cfg = load_config(Path(config_path))
    knob_ranges = {}
    for entry in cfg.get("ranges", []):
        name, vals = entry.split("=", 1)
        knob_ranges[name.strip()] = [float(v) for v in vals.split(",")]
    missing = [n for n in param_names if n not in knob_ranges]
    if missing:
        raise SystemExit(f"--config's [knobs] is missing {missing} (has: {list(knob_ranges)}) "
                         f"-- the .nam's own parameters are {param_names}")
    combos = grid_combinations(param_names, knob_ranges)  # dicts keyed by param_names already
    return [(",".join(f"{n}={p[n]:g}" for n in param_names), [p[n] for n in param_names])
            for p in combos]


def _batched(seq, n):
    it = iter(seq)
    while batch := list(islice(it, n)):
        yield batch



#: Log-spaced probe frequencies, Hz. Floored at 20 Hz deliberately: the A2 stack's receptive
#: field is 6332 samples / 131.9 ms, so the slowest periodicity it can resolve at all is
#: SR/RF ~= 7.58 Hz (see param_train.RECEPTIVE_FIELD_SAMPLES). Content below that completes
#: less than one cycle inside the model's window and is structurally indistinguishable from a
#: slow DC drift -- probing there measures undefined behavior, not instability.
PROBE_FREQS = [20, 30, 45, 65, 95, 140, 200, 290, 420, 600, 880, 1300, 1900, 2800, 4000, 6000, 8800, 12000]


def frequency_probe(model, vals, level, flag_abs, dur=0.4):
    """Which frequencies drive this (model, corner) past `flag_abs`, and is it transient-gated?

    Runs each frequency twice at the same peak level:
      * SUSTAINED -- 50 ms raised fade-in, so there is no attack transient at all.
      * GATED     -- hard on/off, the sharpest attack representable at this amplitude.

    The two together separate the failure modes this fleet actually produces, which the
    windowed scan alone cannot tell apart (both just read as "a big peak"):

      * gated >> sustained  -> TRANSIENT-triggered. The narrow (corner x transient-shape)
        instability the FiLM runaway investigation documents -- a real defect.
      * gated ~= sustained, broad across frequency -> the model is simply producing too much
        everywhere at this corner. In practice that has meant EXTRAPOLATION: a corner outside
        the trained grid, where no training signal constrains the output (measured 2026-09-17:
        a Dumble variant read 43x at an out-of-grid corner yet 0/16 across its real grid).
        Benign as a model defect; check the corner is in-grid before reading anything into it.

    Returns (rows, verdict) where rows is [(freq, sustained_peak, gated_peak), ...].
    """
    t = np.arange(int(dur * SR)) / SR
    fade_n = int(0.05 * SR)
    env = np.minimum(1.0, np.arange(len(t)) / max(fade_n, 1))
    gate = np.zeros(len(t)); gate[int(0.25 * len(t)):int(0.75 * len(t))] = 1.0
    cond = torch.tensor([vals], dtype=torch.float32)
    rows = []
    for f in PROBE_FREQS:
        sine = np.sin(2 * np.pi * f * t)
        pk = []
        for shape in (env, gate):
            sig = (sine * shape * level).astype(np.float32)
            with torch.no_grad():
                o = model(torch.from_numpy(sig).unsqueeze(0).unsqueeze(0), cond)
            pk.append(float(o.abs().max()))
        rows.append((f, pk[0], pk[1]))
    over = [r for r in rows if max(r[1], r[2]) > flag_abs]
    if not over:
        verdict = "no probe frequency exceeds the threshold"
    else:
        gated_only = [r for r in over if r[2] > 2.0 * max(r[1], 1e-9)]
        frac = len(over) / len(rows)
        # ORDER MATTERS: breadth is tested BEFORE attack-dependence. A corner that fails at
        # essentially every frequency is over-producing unconditionally, which in practice has
        # meant extrapolation outside the trained grid -- and it can still show a gated/sustained
        # ratio > 2 in the bands where the sustained tone happens to be quiet, so an
        # attack-first test misreads it as the narrow transient defect. Measured 2026-09-17:
        # a Dumble out-of-grid corner failed 18/18 bands (sustained 30.2 at 20 Hz) yet an
        # attack-first classifier called it TRANSIENT; the real transient defect (Mesa RED w4)
        # fails 9/18 bands with sustained clean at EVERY frequency.
        if frac > 0.75:
            verdict = (f"BROADBAND ({len(over)}/{len(rows)} bands, sustained and gated alike) -- "
                       f"over-producing unconditionally. CHECK THIS CORNER IS IN THE TRAINED GRID "
                       f"before treating it as instability")
        elif len(gated_only) >= max(1, len(over) // 2):
            verdict = (f"TRANSIENT-triggered ({len(gated_only)}/{len(over)} bands need the attack; "
                       f"sustained tones clean) -- the narrow (corner x transient) defect")
        else:
            verdict = f"band-limited ({len(over)}/{len(rows)} bands), not attack-gated"
    return rows, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nam", required=True)
    ap.add_argument("--reference", required=True,
                    help="a local reference clip with varied, hard transient dynamics -- a real "
                         "recording (e.g. a guitar DI) and a synthesized capture sweep (e.g. "
                         "T3K-sweep-v3.wav) work equally well. No bundled default; bring your own.")
    ap.add_argument("--chunk-s", type=float, default=5.0)
    ap.add_argument("--flag-ratio", type=float, default=8.0,
                    help="flag a window if its peak exceeds this multiple of the model's own median peak")
    ap.add_argument("--flag-abs", type=float, default=3.0,
                    help="...AND exceeds this absolute volts floor (avoids flagging near-silent models)")
    ap.add_argument("--config", default=None,
                    help="per-circuit TOML -- scan the FULL trained grid (exact "
                         "combinations gen_dataset_from_schx.py rendered, e.g. 972 for Tweed) instead "
                         "of the reduced hypercube corner set. Auto-discovered from a config.toml "
                         "next to --nam if not given (every release bundle ships one) -- pass "
                         "--no-auto-config to disable that and force the reduced set. Needs --batch-size's batching "
                         "to stay fast at that scale -- see the module docstring for measured cost.")
    ap.add_argument("--no-auto-config", action="store_true",
                    help="skip trained-grid discovery entirely and force the reduced 0/1 "
                         "hypercube corner set. Discovery order is otherwise: the bundle's "
                         "dataset_params.csv (the combinations actually RENDERED) > a config.toml "
                         "beside the .nam (the grid INTENDED, which can differ -- Mesa Orange's "
                         "config lists a Master=0.1 combination that was dropped from the render) "
                         "> the hypercube. Forcing the hypercube probes every knob at literal "
                         "0.0/1.0, which most grids never reach, so it measures EXTRAPOLATION "
                         "beyond the trained range rather than instability within it. That is "
                         "worth knowing deliberately -- a plugin host can send any knob value -- "
                         "but it must not be read as a defect: on 2026-09-17 it produced 47,461x "
                         "and 43x readings on two models that are 0/576 and 0/16 across their "
                         "actual grids.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="corners per batched forward call (default: %(default)s). Batching "
                         "across corners (not just chunks) is what makes --config's full-grid "
                         "mode tractable -- one forward call per (chunk, batch-of-corners) pair "
                         "instead of one per (chunk, corner). PROPERLY INVESTIGATED 2026-08-04 "
                         "after an earlier fix attempt (thread-pool 'wave' teardown) turned out "
                         "to be treating the wrong mechanism -- see the long comment at this "
                         "module's job-scheduling loop for the full writeup. Short version: this "
                         "was never an unbounded leak, it's a genuinely large per-job working set "
                         "(batch_size x chunk_s x SR-sized tensors, x23 WaveNet layers) that at "
                         "the old default (64) x 14 workers pushed close enough to a 36GB "
                         "machine's physical RAM to trigger macOS memory-pressure pathology "
                         "(looks like unbounded growth, isn't). Confirmed via a clean, complete, "
                         "isolated repro: workers=14 batch_size=8 ran all 60/60 test iterations "
                         "at a rock-stable ~16GB peak (this module's own peak-RSS tracking never "
                         "moved off its first-reached value the entire run) -- vs. batch_size=64 "
                         "which climbed unboundedly past 34GB on the exact same machine/model. "
                         "Raise this only if you have headroom to spare -- workers x batch_size "
                         "is the number that matters, not either alone.")
    ap.add_argument("--sweep-floor", type=float, default=None, metavar="HZ",
                    help="high-pass the reference to this frequency before scanning. Default: "
                         "derived from the bundle's own excitation recipe (chirp_f0/sweep_f0) -- "
                         "the lowest frequency training deliberately covers. Probing below it "
                         "measures a band the model was never taught, which reads as instability "
                         "but is absence of training signal: a 20 Hz probe made a published Mesa "
                         "Orange w4 read 341 on a sustained tone while clean from 45 Hz up and "
                         "clean on real playing. No single constant is right -- this fleet records "
                         "floors of 40 AND 80 Hz, and build_excitation.py's default moved to 15 -- "
                         "so it is derived, not hardcoded. 0 disables the high-pass.")
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="scale the reference to (this device's trained peak x HEADROOM). "
                         "Default 1.0 = drive it exactly as hard as training did. Raising this "
                         "probes margin beyond the trained range, but do NOT go far: the "
                         "flag-abs test implicitly assumes the circuit saturates, so on a "
                         "NON-saturating device (output still growing past typical input -- "
                         "a clean channel, a high-onset amp) a correct, perfectly linear model "
                         "trips --flag-abs by construction somewhere above ~3x. 0 disables "
                         "scaling and uses the clip as-is.")
    ap.add_argument("--input-level", type=float, default=None, metavar="VOLTS",
                    help="override the detected per-device training peak (see --headroom). "
                         "Use when scanning a bare .nam whose bundle isn't to hand.")
    ap.add_argument("--freq-probe", type=int, default=0, metavar="N",
                    help="after scanning, run a sine-sweep probe on the N worst-flagged corners "
                         "of each tier (0 = off). Reports, per frequency, the peak under a "
                         "SUSTAINED tone (50 ms fade-in, no attack) and under a hard-GATED burst. "
                         "Separates the two failure shapes the windowed scan reports identically: "
                         "gated >> sustained is the narrow transient-triggered instability that is "
                         "a real defect; broadband-and-sustained has in practice meant the corner "
                         "is OUTSIDE the trained grid, where nothing constrains the model. Probe "
                         "frequencies start at 20 Hz -- below the ~7.58 Hz receptive-field floor "
                         "the model cannot represent the input at all, so anything there measures "
                         "undefined behavior rather than instability.")
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent (chunk, corner-batch) forward calls (default: cpu_count). "
                         "This model is too narrow (few channels) for PyTorch's own intra-op "
                         "threading to bother splitting a single conv1d across cores -- confirmed "
                         "empirically (2026-08-02): a single scan used ~1.3 of 14 cores despite "
                         "torch.get_num_threads()==10. Running independent forward calls "
                         "concurrently across threads works instead, because PyTorch's C++ conv/ "
                         "matmul kernels release the GIL during compute -- this is the same 'many "
                         "small independent ops, not one big one' shape as gen_dataset_from_schx.py's "
                         "render_many() ThreadPoolExecutor pattern, just for inference instead of "
                         "subprocesses. torch.set_num_threads(1) below is required alongside this: "
                         "without it, each of --workers concurrent calls would ALSO try to spawn "
                         "its own intra-op threads, oversubscribing the machine.")
    args = ap.parse_args()

    workers = args.workers or os.cpu_count() or 4
    torch.set_num_threads(1)  # see --workers help: avoid oversubscribing against the thread pool

    submodels = load_all_submodels(args.nam)
    param_names = submodels[0][1]
    for _, p, ch in submodels[1:]:
        if p != param_names:
            raise SystemExit(f"{args.nam}: submodels disagree on params ({param_names} at "
                              f"channels={submodels[0][2]} vs {p} at channels={ch}) -- "
                              f"container may be malformed")
    tier_channels = [ch for _, _, ch in submodels]
    print(f"{Path(args.nam).name}: {len(submodels)} submodel(s), channels={tier_channels} "
          f"params={param_names}")

    x, sr = sf.read(args.reference, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    if sr != SR:
        raise SystemExit(f"reference sr {sr} != {SR}")

    # PER-DEVICE SWEEP FLOOR -- see device_sweep_floor() for why this cannot be a constant.
    # Applied BEFORE scaling so the level is measured on the band actually being scanned.
    floor, floor_src = ((args.sweep_floor, "--sweep-floor") if args.sweep_floor is not None
                        else device_sweep_floor(args.nam))
    if floor:
        from scipy.signal import butter, sosfilt
        x = sosfilt(butter(4, floor, "highpass", fs=SR, output="sos"), x).astype(np.float32)
        print(f"  sweep floor: high-passed at {floor:g} Hz ({floor_src})")
    elif args.sweep_floor is None:
        print("  WARNING: no excitation recipe found beside the .nam -- reference NOT high-passed. "
              "If it\n           carries content below this device's own chirp floor, findings "
              "there are absence of\n           training signal, not instability. Pass "
              "--sweep-floor to set it explicitly.")

    # PER-DEVICE INPUT SCALING -- see device_input_level() for why a fixed absolute level
    # makes this check meaningless across a fleet whose excitations span 122x.
    ref_peak = float(np.abs(x).max())
    if args.headroom and ref_peak > 0:
        level, src = ((args.input_level, "--input-level") if args.input_level
                      else device_input_level(args.nam))
        if level:
            target = level * args.headroom
            x = (x / ref_peak * target).astype(np.float32)
            note = "" if args.headroom == 1.0 else f" x{args.headroom:g} headroom"
            print(f"  input level: {ref_peak:.4g} -> {target:.4g} V peak ({src}{note})")
            if src and src.startswith("metadata."):
                print("           NOTE: that is a volts-per-full-scale ASSUMPTION from the .schx, "
                      "not the level\n           this model was trained at. Prefer scanning the "
                      "bundle, or pass --input-level.")
        else:
            print(f"  WARNING: no per-device input level found (no bundle dataset_config.json, no "
                  f"metadata.input_level_dbu)\n           -- using the clip as-is at {ref_peak:.4g} "
                  f"V peak. If that is far from what this device\n           was trained at, both "
                  f"the peaks and the flag counts below are unreliable.")
    else:
        print(f"  input level: clip as-is, {ref_peak:.4g} V peak (scaling disabled)")

    # Corner-set selection, most authoritative first. See bundle_grid_corners() for why the
    # hypercube is a LAST resort rather than the default it used to be.

    # CORNER-SET SELECTION, most authoritative first. Merged from two independent fixes to
    # the same problem (74af6c5 auto-discovered config.toml; d80e5c8 read dataset_params.csv):
    # the ACTUALLY-RENDERED combinations outrank the INTENDED ones, because they differ in
    # practice. Mesa Orange's config.toml lists Or Master=0.1, but that combination was dropped
    # from the render -- LiveSPICE emitted isolated single-sample spikes there at oversample
    # 8/16/32 alike -- so the model never saw it. Scanning it would be extrapolation wearing a
    # trained grid's clothes. dataset_params.csv records what was really rendered; config.toml
    # is the fallback for a bundle that lacks it; the 0/1 hypercube is a last resort.
    grid_source = None
    if args.config:
        corners = full_grid_corners(args.config, param_names)
        grid_source = "--config trained grid"
    elif args.no_auto_config:
        corners = hypercube_corners(param_names)
        grid_source = "0/1 hypercube (EXPLICITLY REQUESTED -- probes beyond the trained grid)"
    else:
        corners = bundle_grid_corners(args.nam, param_names)
        if corners is not None:
            grid_source = "bundle dataset_params.csv (exact rendered combinations)"
        else:
            candidate = Path(args.nam).parent / "config.toml"
            if candidate.exists():
                corners = full_grid_corners(str(candidate), param_names)
                grid_source = f"auto-discovered {candidate.name} (intended grid; no "
                grid_source += "dataset_params.csv beside the .nam)"
            else:
                corners = hypercube_corners(param_names)
    if grid_source is None:
        print("  WARNING: no trained grid available (no --config, no dataset_params.csv and no "
              "config.toml\n           beside the .nam) -- falling back to the 0/1 hypercube, "
              "which probes knob values\n           this model may never have been trained on. "
              "The reduced set has also concretely\n           missed real defects before "
              "(interior grid points, not just vertices, can be the\n           worst corner). "
              "Findings here are EXTRAPOLATION, not a verdict on the model.")
        grid_source = "0/1 hypercube (FALLBACK -- EXTRAPOLATION, not the trained grid)"
    print(f"  corner set: {grid_source}")
    corner_vals = {label: vals for label, vals in corners}   # for --freq-probe
    chunk_n = int(args.chunk_s * SR)
    n_chunks = len(x) // chunk_n
    n_batches = -(-len(corners) // args.batch_size)  # ceil
    print(f"  {len(corners)} corners x {n_chunks} chunks each "
          f"({n_batches} corner-batches/chunk, batch-size={args.batch_size}, workers={workers})"
          f"{' [reduced hypercube]' if 'hypercube' in grid_source else ' [full grid]'}")

    def make_score(model):
        def score(job):
            c, batch = job
            seg = x[c * chunk_n:(c + 1) * chunk_n]
            B = len(batch)
            inp = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)  # [B,1,T]
            cond = torch.tensor([vals for _, vals in batch], dtype=torch.float32)          # [B,num_params]
            with torch.no_grad():
                pred = model(inp, cond)                                                    # [B,1,T]
            pk = pred.abs().amax(dim=(1, 2))                                                # [B]
            return [(float(pk[i]), label, c) for i, (label, _) in enumerate(batch)]
        return score

    jobs = [(c, batch) for c in range(n_chunks) for batch in _batched(corners, args.batch_size)]
    # PROPERLY INVESTIGATED 2026-08-04 (see --batch-size's help above for the short version).
    # An earlier fix attempt tore down and recreated the ThreadPoolExecutor every 20 jobs
    # ("wave" teardown), on a theory that OS-level per-thread allocator caching was compounding
    # over a long-lived pool's lifetime. That fix was a plausible-sounding GUESS, explicitly
    # never confirmed with a real memory profiler, and it turned out to be wrong: it only
    # partially bounded growth (still climbed to 20GB by t=400s on a 266-job scan), and a
    # SMALLER wave size -- which the "more teardown = more relief" theory predicted should help
    # MORE -- instead made things WORSE (25.6GB within 20s), which should have been the signal
    # the theory was broken, not something to keep tuning.
    #
    # Properly isolated with two controlled, non-live repros instead of more tuning attempts:
    # (1) the SAME workload run single-threaded (no ThreadPoolExecutor at all) stayed stable,
    # ruling out anything Python-reference-related (no_grad already ensures no autograd graph
    # retention; the returned `results` only ever holds plain floats/strings, never tensors).
    # (2) workers=14 at the OLD batch_size=64 vs. a SMALLER batch_size=8 (same worker count,
    # same total iterations, same model): batch_size=8 ran clean and FLAT the entire way (see
    # --batch-size's help) while batch_size=64 is what produced the original unbounded-looking
    # climb past 34GB. Varying batch_size (not worker count, not teardown frequency) was the
    # actual controlling variable -- this was never a leak, it's a genuinely large per-job
    # working set that only becomes pathological once workers x batch_size pushes close enough
    # to the machine's physical RAM to trigger macOS memory-pressure behavior (which looks like
    # runaway growth from the outside, but is fundamentally bounded per-job, not compounding).
    # Fixed at the source (--batch-size's default) instead of papering over it with executor
    # churn -- a single plain ThreadPoolExecutor is both simpler and, per the repro above,
    # exactly as safe as the wave version once batch_size is sane. Applies per-tier below too:
    # each tier gets its own fresh ThreadPoolExecutor over the identical `jobs` list, so the
    # working set (batch_size x workers) never grows across tiers, and each pool is fully torn
    # down (not just idled) before the next tier's begins.

    total_flagged = 0
    tier_summaries = []
    for tier_i, (model, _, channels) in enumerate(submodels):
        tag = f"w{channels}"
        print(f"  -- tier {tier_i + 1}/{len(submodels)} ({tag}) --")
        results = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for batch_results in ex.map(make_score(model), jobs):
                results.extend(batch_results)
                done += 1
                if len(jobs) > 10 and done % max(1, len(jobs) // 10) == 0:
                    print(f"    {done}/{len(jobs)} corner-batches done")

        peaks = np.array([r[0] for r in results])
        median = float(np.median(peaks))
        results.sort(key=lambda r: -r[0])
        flagged = [r for r in results if r[0] > args.flag_ratio * median and r[0] > args.flag_abs]
        total_flagged += len(flagged)
        tier_summaries.append((tag, len(flagged), median, float(peaks.max()), float(peaks.min())))

        print(f"    {len(results)} (corner x {args.chunk_s:.0f}s-chunk) probes, "
              f"{len(corners)} corners x {n_chunks} chunks over {len(x)/SR:.0f}s")
        print(f"    median peak={median:.4f}  max peak={peaks.max():.4f}  min peak={peaks.min():.4f}")
        if flagged:
            print(f"    FLAGGED ({len(flagged)} windows > {args.flag_ratio}x median AND > {args.flag_abs}V):")
            for pk, label, c in flagged[:10]:
                print(f"      peak={pk:10.3f}  corner={label:16}  t={c*args.chunk_s:.0f}-{(c+1)*args.chunk_s:.0f}s")
        else:
            print("    clean -- no anomalous windows")

        # --freq-probe: characterise the SHAPE of each flagged corner's failure, not just its
        # size. Runs on the worst corners only (the probe is cheap per corner, but there is no
        # point sweeping a corner that never flagged).
        if args.freq_probe and flagged:
            seen, probe_corners = set(), []
            for pk, label, c in flagged:
                if label in seen:
                    continue
                seen.add(label)
                probe_corners.append(label)
                if len(probe_corners) >= args.freq_probe:
                    break
            level = float(np.abs(x).max())
            for label in probe_corners:
                vals = corner_vals.get(label)
                if vals is None:
                    continue
                rows, verdict = frequency_probe(model, vals, level, args.flag_abs)
                print(f"    frequency probe @ {label} (input peak {level:.3f}): {verdict}")
                print(f"      {'Hz':>7} {'sustained':>11} {'gated':>10}")
                for f, sus, gat in rows:
                    mark = " **" if max(sus, gat) > args.flag_abs else ""
                    print(f"      {f:7d} {sus:11.3f} {gat:10.3f}{mark}")

    print(f"\n  summary ({len(submodels)} tier(s)):")
    for tag, n_flagged, median, pk_max, pk_min in tier_summaries:
        verdict = f"FLAGGED ({n_flagged})" if n_flagged else "clean"
        print(f"    {tag:>6}  median={median:.4f}  max={pk_max:.4f}  min={pk_min:.4f}  {verdict}")
    if total_flagged:
        flagged_tiers = [tag for tag, n, *_ in tier_summaries if n]
        print(f"  {total_flagged} total flagged window(s) across {len(flagged_tiers)} "
              f"tier(s): {', '.join(flagged_tiers)}")
    else:
        print("  all tiers clean -- no anomalous windows")
    return total_flagged


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
