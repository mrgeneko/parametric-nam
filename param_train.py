#!/usr/bin/env python3
"""
param_train.py — Train a parametric A2 + FiLM Neural Amp Modeler.

Loads a dataset produced by batch_harness.py, trains an A2 architecture
with FiLM conditioning on knob parameters, and exports a .param.nam file
loadable by the C++ ParametricWaveNet subclass.

Usage:
  python param_train.py --dataset /path/to/dataset --output model.param.nam
"""

import argparse, csv, datetime, faulthandler, json, math, os, signal, sys, time, warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from per_perm_esr import compute_per_perm_esr, summarize as summarize_per_perm_esr, write_csv as write_per_perm_esr_csv

# ---------------------------------------------------------------------------
# Hang watchdog. 2026-07-21: MPS training on this fleet has hung silently
# (no exception, no error) at the epoch boundary -- observed in both the
# checkpoint save's .cpu() device copy AND the next epoch's first backward
# pass, via manual `sample <pid>` stack traces (a slow, live-only diagnostic).
# This arms a timer before each such operation and cancels it on success; if
# an operation doesn't return in time, faulthandler dumps every thread's
# Python-level traceback to <ckpt-or-cwd>/watchdog.log -- so a recurrence is
# caught automatically, unattended, instead of needing a human to `sample` it
# during the narrow window before the next kill/relaunch.
_watchdog_file = None


def _watchdog_open(ckpt_dir):
    global _watchdog_file
    path = (ckpt_dir if ckpt_dir is not None else Path(".")) / "watchdog.log"
    _watchdog_file = open(path, "a")
    print(f"Hang watchdog armed -- stalls dump to {path}", file=sys.stderr)


def _watchdog_arm(label: str, timeout: float = 45.0):
    if _watchdog_file is None:
        return
    _watchdog_file.write(f"\n--- watchdog armed for {label!r} at "
                          f"{datetime.datetime.now().isoformat()} (timeout {timeout}s) ---\n")
    _watchdog_file.flush()
    faulthandler.dump_traceback_later(timeout, exit=False, file=_watchdog_file)


def _watchdog_disarm():
    if _watchdog_file is None:
        return
    faulthandler.cancel_dump_traceback_later()

# ---------------------------------------------------------------------------
# A2 architecture constants (num layers / kernel sizes / dilations / LeakyReLU) — match
# C++ a2_fast.h and NAM's A2 config.
#
# HEAD — SKIP-ONLY (migrated 2026-07; see docs/rearchitecture_skip_accumulation.md):
#   * The head reads the SUM of every layer's post-activation — the standard WaveNet skip
#     connections that NAM, nam_wavenet.metal, the static a2_fast path and every stock NAM
#     plugin already compute. It is the only head that exports faithfully to stock plugins
#     (corr 1.0; the old residual head scored 0.31), and it costs no accuracy.
#   * The legacy "residual" head (head reads the final residual) was a [redacted]-only
#     convention. It is GONE — one head, no mode to choose. Models carrying it are REJECTED
#     on load (check_parametric_schema), never silently played: residual weights under a skip
#     head produce garbage with no error.
#   * The head is CAUSAL. It used to be padding=K//2, which let output[n] peek 7 samples into
#     the future — information a real amp does not have, and a 7-sample misalignment against
#     every other (causal) NAM capture.
# Models self-describe via config.parametric.{head_mode:"skip", schema_version}.
# ---------------------------------------------------------------------------
K_NUM_LAYERS = 23
K_HEAD_KERNEL = 16
K_LEAKY_SLOPE = 0.01

# Schema version of OUR config["parametric"] block — independent of the .nam file's
# top-level "version" (0.7.0), which is NAM's FILE-FORMAT version and belongs to NAM
# (bumping it would make loaders report the file unsupported). This one versions the
# parametric extension so future changes can't be silently misread.
#   0 = pre-versioning (implicit): no schema_version, no head_mode => legacy residual model,
#       which is REJECTED (the residual head no longer exists).
#   1 = declares "head_mode": "skip".
# Readers MUST fail loudly on a version they don't know rather than guess — guessing
# a conditioning/head change produces wrong audio with no error.
K_PARAM_SCHEMA_VERSION = 1


def check_parametric_schema(par: dict, source: str = ".param.nam") -> int:
    """Reject any model we cannot faithfully run. SKIP-ONLY.

    Two hard failures, never a guess — guessing a head silently produces wrong audio:
      * schema_version newer than we understand;
      * head_mode other than "skip". An ABSENT head_mode means a pre-tagging model, which is
        residual by definition. Residual weights under a skip head give garbage (corr ~0.3),
        so we refuse rather than run them.
    """
    v = int(par.get("schema_version", 0))
    if v > K_PARAM_SCHEMA_VERSION:
        raise SystemExit(
            f"{source}: parametric schema_version {v} is newer than this build supports "
            f"(max {K_PARAM_SCHEMA_VERSION}). Upgrade parametric-nam — proceeding would "
            f"silently misread the model and produce wrong audio.")
    head_mode = par.get("head_mode", "residual")
    if head_mode != "skip":
        raise SystemExit(
            f"{source}: this model uses the legacy '{head_mode}' head and is no longer "
            f"supported. Parametric models are skip-only; the residual head was removed. Its "
            f"weights would produce garbage under a skip head, so it is rejected rather than "
            f"run. Retrain it, or re-download a current model.")
    return v

K_KERNEL_SIZES = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    15, 15,
    6, 6, 6, 6, 6, 6, 6,
]

K_DILATIONS = [
    1, 3, 7, 17, 41, 101, 239,
    1, 3, 7, 17, 41, 101, 239,
    1, 13,
    1, 3, 7, 17, 41, 101, 239,
]

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta."""
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.net = nn.Linear(cond_dim, 2 * channels)
        # Small NON-zero weight so the knob modulates from step 0. A pure-zero init
        # leaves FiLM at identity for ALL cond, and it then under-learns subtle /
        # level-invariant knob effects (the 5150 gain knob changes tone but not
        # level — exactly that case), producing a near-dead knob. Bias still starts
        # near-identity: gamma≈1, beta≈0.
        nn.init.normal_(self.net.weight, std=0.1)
        nn.init.zeros_(self.net.bias)
        self.net.bias.data[:channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * x + beta


def parse_knob_boost(spec: str) -> dict[str, float]:
    """Parse '--knob-boost' syntax: 'NAME=mult,NAME2=mult2,...' -> {NAME: mult}."""
    boosts: dict[str, float] = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, sep, mult = tok.partition("=")
        if not sep:
            raise SystemExit(f"--knob-boost: bad entry {tok!r} (want NAME=mult)")
        boosts[name.strip()] = float(mult)
    return boosts


def register_knob_boost_hooks(model: nn.Module, param_names: list[str],
                              boosts: dict[str, float]) -> int:
    """Scale the backward-pass gradient flowing into specific knobs' FiLM
    conditioning columns, in every FiLM layer of the model.

    FiLM.net is nn.Linear(num_params, 2*channels): its weight matrix has one
    INPUT column per knob (column i is entirely and only responsible for knob
    param_names[i]'s contribution to that layer's gamma/beta). Registering a
    gradient hook on that column lets one knob's conditioning pathway learn
    faster without touching the loss function (which stays an honest,
    unweighted measure of fit quality) or any other knob's learning rate --
    unlike --film-lr-mult, which boosts the whole conditioning vector at once.
    Generic over any knob name / any device: this has no TS-9-, Drive-, or
    circuit-specific logic, it only needs a name present in param_names.
    """
    col_mult: dict[int, float] = {}
    for name, mult in boosts.items():
        if name not in param_names:
            raise SystemExit(f"--knob-boost: unknown knob {name!r}; "
                             f"dataset knobs are {param_names}")
        col_mult[param_names.index(name)] = mult

    def make_hook(cols: dict[int, float]):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            g = grad.clone()
            for col, mult in cols.items():
                g[:, col] *= mult
            return g
        return hook

    n = 0
    for module in model.modules():
        if isinstance(module, FiLM):
            module.net.weight.register_hook(make_hook(col_mult))
            n += 1
    return n


class A2Layer(nn.Module):
    """Single A2 dilated conv layer with optional FiLM."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, cond_dim: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation)
        self.mixin = nn.Conv1d(1, channels, 1, bias=False)
        self.film = FiLM(channels, cond_dim) if cond_dim > 0 else None
        self.l1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, inp_audio: torch.Tensor, cond: torch.Tensor):
        """Returns (residual_out, post_activation). post_activation is the pre-layer1x1
        LeakyReLU output = the 'skip' term NAM/a2_fast accumulate into the head."""
        h = self.conv(x)
        # .contiguous(): this crop is non-contiguous (each row stays stride-1 internally,
        # but rows are no longer packed at the original conv output's stride) -- suspected
        # trigger for the 2026-07-21 MPS backward-pass hangs (a historical class of MPS
        # deadlock when backpropagating through non-contiguous tensors). Cheap either way.
        h = h[:, :, :inp_audio.shape[-1]].contiguous()
        h = h + self.mixin(inp_audio)
        if self.film is not None:
            h = self.film(h, cond)
        post_act = F.leaky_relu(h, K_LEAKY_SLOPE)
        residual_out = x + self.l1x1(post_act)
        return residual_out, post_act


class ParametricA2(nn.Module):
    """A2 (LeakyReLU, mixed kernels) with FiLM conditioning on knob params.

    Architecture matches the C++ A2FastModel with additional FiLM layers
    for parametric knob control.
    """
    def __init__(self, channels: int, num_params: int):
        super().__init__()
        self.channels = channels
        self.num_params = num_params
        self.rechannel = nn.Conv1d(1, channels, 1, bias=False)
        self.layers = nn.ModuleList([
            A2Layer(channels, K_KERNEL_SIZES[i], K_DILATIONS[i], num_params)
            for i in range(K_NUM_LAYERS)
        ])
        # CAUSAL head: no built-in padding; we left-pad K-1 in forward(). A centered
        # `padding=K_HEAD_KERNEL // 2` (the old behavior) makes output[n] read inputs
        # [n-8, n+7] — i.e. 7 samples of LOOKAHEAD, which a real amp does not have. NAM and
        # a2_fast use a strictly causal head ([n-15, n]), so the same weights played there
        # come out 7 samples late. That constant offset is inaudible on its own, but it
        # comb-filters against any causally-aligned parallel path (dry blend, a second amp,
        # another NAM capture) with a first null near 3.4 kHz. Causal = 0 added latency and
        # bit-alignment with the whole NAM ecosystem. Weight shapes are unchanged.
        self.head = nn.Conv1d(channels, 1, K_HEAD_KERNEL)
        self.head_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, audio: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = self.rechannel(audio)
        inp_audio = audio
        # SKIP head: reads the SUM of every layer's post-activation — the standard WaveNet
        # skip connections that NAM, a2_fast and every stock plugin compute.
        skip = None
        for layer in self.layers:
            x, post_act = layer(x, inp_audio, params)
            skip = post_act if skip is None else skip + post_act
        head_in = skip
        # Causal: left-pad K-1 so output[n] reads only [n-(K-1), n] — no lookahead.
        head_in = F.pad(head_in, (K_HEAD_KERNEL - 1, 0))
        x = self.head(head_in)
        x = x[:, :, :audio.shape[-1]].contiguous()   # see A2Layer.forward's .contiguous() note
        x = x * self.head_scale
        return x

    def weight_count(self) -> int:
        """Total flat float count for .nam export matching C++ order."""
        total = 0
        total += self.rechannel.weight.numel()
        for layer in self.layers:
            total += layer.conv.weight.numel()
            total += layer.conv.bias.numel()
            total += layer.mixin.weight.numel()
            total += layer.l1x1.weight.numel()
            total += layer.l1x1.bias.numel()
            if layer.film is not None:
                total += layer.film.net.weight.numel()
                total += layer.film.net.bias.numel()
        total += self.head.weight.numel()
        total += self.head.bias.numel()
        total += 1  # head_scale
        return total

    def _export_weight_block(self) -> list[float]:
        """C++ weight order — Conv layers: out_ch→in_ch→tap (r-major)."""
        w = []
        w.extend(self.rechannel.weight.detach().flatten().tolist())
        for layer in self.layers:
            w.extend(layer.conv.weight.detach().flatten().tolist())
            w.extend(layer.conv.bias.detach().flatten().tolist())
            w.extend(layer.mixin.weight.detach().flatten().tolist())
            w.extend(layer.l1x1.weight.detach().flatten().tolist())
            w.extend(layer.l1x1.bias.detach().flatten().tolist())
            if layer.film is not None:
                w.extend(layer.film.net.weight.detach().flatten().tolist())
                w.extend(layer.film.net.bias.detach().flatten().tolist())
        w.extend(self.head.weight.detach().flatten().tolist())
        w.extend(self.head.bias.detach().flatten().tolist())
        w.append(self.head_scale.detach().item())
        return w

    def _load_weight_block(self, it):
        """Consume flat iterator in C++ order, populate weights in-place."""
        self.rechannel.weight.data.copy_(
            torch.tensor([next(it) for _ in range(self.channels)]).view_as(self.rechannel.weight))
        for layer in self.layers:
            C = self.channels
            K = layer.conv.weight.shape[-1]
            layer.conv.weight.data.copy_(
                torch.tensor([next(it) for _ in range(K * C * C)]).view_as(layer.conv.weight))
            layer.conv.bias.data.copy_(
                torch.tensor([next(it) for _ in range(C)]).view_as(layer.conv.bias))
            layer.mixin.weight.data.copy_(
                torch.tensor([next(it) for _ in range(C)]).view_as(layer.mixin.weight))
            layer.l1x1.weight.data.copy_(
                torch.tensor([next(it) for _ in range(C * C)]).view_as(layer.l1x1.weight))
            layer.l1x1.bias.data.copy_(
                torch.tensor([next(it) for _ in range(C)]).view_as(layer.l1x1.bias))
            if layer.film is not None:
                np_ = self.num_params
                layer.film.net.weight.data.copy_(
                    torch.tensor([next(it) for _ in range(2 * C * np_)]).view_as(layer.film.net.weight))
                layer.film.net.bias.data.copy_(
                    torch.tensor([next(it) for _ in range(2 * C)]).view_as(layer.film.net.bias))
        self.head.weight.data.copy_(
            torch.tensor([next(it) for _ in range(K_HEAD_KERNEL * self.channels)]).view_as(self.head.weight))
        self.head.bias.data.copy_(
            torch.tensor([next(it) for _ in range(1)]).view_as(self.head.bias))
        self.head_scale.data.copy_(torch.tensor(next(it)))

    def export_weights(self) -> list[float]:
        return self._export_weight_block()

    def load_weights(self, weights: list[float]):
        self._load_weight_block(iter(weights))

    def export_nam(self, config: dict, metadata: dict, sample_rate: int,
                   input_audio: "np.ndarray | None" = None) -> dict:
        weights = self.export_weights()

        # Compute gain: run at default knobs (all 0.5), target -18 dBFS output.
        # gain is the scalar the plugin multiplies output by for level normalisation.
        gain = 1.0
        if input_audio is not None:
            device = next(self.parameters()).device
            clip = input_audio[:sample_rate * 10]  # 10 s is enough for RMS
            with torch.no_grad():
                inp = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(device)
                params = torch.full((1, self.num_params), 0.5, device=device)
                out = self(inp, params)
            out_rms = float(out.squeeze().cpu().pow(2).mean().sqrt().item())
            target_rms = 10 ** (-18.0 / 20.0)
            if out_rms > 1e-8:
                gain = float(target_rms / out_rms)

        param_map = config.get("param_map", {})
        bounds = config.get("bounds", {})
        defaults = config.get("defaults", {}) or {}
        steps_map = config.get("steps", {}) or {}
        param_defs = []
        for name in config.get("param_names", []):
            lo, hi = bounds.get(name, [0.0, 1.0])
            lo, hi = float(lo), float(hi)
            # Declared default: the circuit's real default knob position (used by
            # bake_nam when no --params is given). Falls back to the range midpoint
            # only if the circuit doesn't declare one — midpoint is wrong for many
            # controls (e.g. a Timmy's don't center at noon).
            dflt = defaults.get(name)
            dflt = float(dflt) if dflt is not None else round((lo + hi) / 2, 4)
            entry = {
                "name": param_map.get(name, name).strip(),
                "min": lo, "max": hi, "default": dflt,
            }
            # Discrete switch metadata (from --steps at dataset-generation time), for
            # readers (NeuralAmpModelerCore's DSPParamDef.steps) that render an N-position
            # selector instead of a continuous knob. Omitted for ordinary continuous knobs.
            steps = int(steps_map.get(name, 0) or 0)
            if steps >= 2:
                entry["steps"] = steps
            param_defs.append(entry)
        layer_config = {
            "layers": self.channels,
            "head_scale": self.head_scale.item(),
            "parametric": {
                "type": "film",
                "schema_version": K_PARAM_SCHEMA_VERSION,
                "condition_size": self.num_params,
                # Always "skip" — the one head we support. Declared so readers can REJECT
                # legacy residual/untagged models instead of silently playing garbage.
                "head_mode": "skip",
                "film_layers": [f"layer_{i}" for i in range(K_NUM_LAYERS)],
                "parameters": param_defs,
            },
        }
        now = datetime.datetime.now()
        nam_metadata = {
            "date": {
                "year": now.year, "month": now.month, "day": now.day,
                "hour": now.hour, "minute": now.minute, "second": now.second,
            },
            "loudness": -18.0,
            "gain": gain,
            "name": config.get("name", config.get("circuit", "")),
            "modeled_by": config.get("modeled_by", ""),
            "gear_make": config.get("gear_make", ""),
            "gear_model": config.get("gear_model", config.get("circuit", "")),
            "gear_type": config.get("gear_type", "amp"),
            "tone_type": config.get("tone_type", ""),
        }
        # Strip empty strings so absent fields don't clutter the file
        nam_metadata = {k: v for k, v in nam_metadata.items() if v != ""}
        return {
            "version": metadata.get("version", "0.7.0"),
            "architecture": "ParametricWaveNet",
            "config": layer_config,
            "weights": weights,
            "metadata": nam_metadata,
            "sample_rate": sample_rate,
        }


# ---------------------------------------------------------------------------
# Slimmable model (A2 Lite 3ch + A2 Full 8ch, trained jointly)
# ---------------------------------------------------------------------------

K_LITE_CHANNELS = 3
K_FULL_CHANNELS = 8
K_DEFAULT_WIDTHS = [K_LITE_CHANNELS, K_FULL_CHANNELS]   # 2-tier default (back-compat)


def parse_widths(spec) -> list[int]:
    """'3,4,8' -> [3,4,8]; None/'' -> the default [3,8]. Sorted, de-duplicated,
    ascending (narrowest first = the 'lite' tier, widest last = the 'full' tier)."""
    if not spec:
        return list(K_DEFAULT_WIDTHS)
    if isinstance(spec, (list, tuple)):
        vals = list(spec)
    else:
        vals = [int(x) for x in str(spec).split(",") if x.strip()]
    ws = sorted(set(int(w) for w in vals))
    if len(ws) < 2:
        raise ValueError(f"--widths needs >=2 distinct channel counts, got {ws}")
    return ws


class SlimmableParametricA2(nn.Module):
    """N independent ParametricA2 models (one per channel width) trained jointly,
    exported as one SlimmableContainer whose submodels are selected at runtime by
    ascending `max_value` breakpoints.

    Tiers are ordered ascending by width: index 0 = narrowest (the 'lite' tier),
    index -1 = widest (the 'full' tier). Default widths [3, 8] reproduce the prior
    2-tier lite/full behavior exactly (max_value 0.5 / 1.0).
    """
    def __init__(self, num_params: int, widths=None):
        super().__init__()
        self.widths = parse_widths(widths)
        self.num_params = num_params
        # Endpoints keep the attribute names `lite`/`full` so state-dict keys stay
        # `lite.*` / `full.*` — a default [3, 8] model is byte-compatible with old
        # 2-tier checkpoints (resume + export_checkpoint keep working). Any middle
        # tiers live in `mid` (empty ModuleList when widths has just 2 entries).
        self.lite = ParametricA2(self.widths[0], num_params)
        self.full = ParametricA2(self.widths[-1], num_params)
        self.mid = nn.ModuleList(ParametricA2(w, num_params) for w in self.widths[1:-1])

    @property
    def submodels(self):
        """All tiers in ascending-width order."""
        return [self.lite, *self.mid, self.full]

    def forward(self, audio: torch.Tensor, params: torch.Tensor):
        return [m(audio, params) for m in self.submodels]   # ascending by width

    # --- tier addressing -----------------------------------------------------
    def tier_labels(self) -> list[str]:
        """Endpoints keep role names (lite/full) so existing tooling — metrics
        columns, best-checkpoint filenames, release naming — keeps working;
        middle tiers are labeled by width (e.g. 'w4')."""
        n = len(self.widths)
        labels = []
        for i, w in enumerate(self.widths):
            if i == 0:
                labels.append("lite")
            elif i == n - 1:
                labels.append("full")
            else:
                labels.append(f"w{w}")
        return labels

    def max_values(self) -> list[float]:
        n = len(self.widths)
        return [round((i + 1) / n, 6) for i in range(n)]   # last == 1.0

    def tier_state_prefix(self, i: int) -> str:
        """State-dict key prefix for the i-th tier (ascending width): the
        endpoints live under 'lite.'/'full.', middle tiers under 'mid.<k>.'."""
        n = len(self.widths)
        if i == 0:
            return "lite."
        if i == n - 1:
            return "full."
        return f"mid.{i - 1}."

    def export_nam(self, config: dict, metadata: dict, sample_rate: int,
                   input_audio: "np.ndarray | None" = None) -> dict:
        subs = [m.export_nam(config, metadata, sample_rate, input_audio)
                for m in self.submodels]
        full_nam = subs[-1]   # widest tier supplies the container-level metadata
        return {
            "version": metadata.get("version", "0.7.0"),
            "architecture": "SlimmableContainer",
            "config": {
                "submodels": [
                    {"max_value": mv, "model": s}
                    for mv, s in zip(self.max_values(), subs)
                ]
            },
            "weights": [],
            "metadata": {
                **{k: v for k, v in full_nam["metadata"].items()
                   if k not in ("gain",)},
                "gain": full_nam["metadata"]["gain"],
            },
            "sample_rate": sample_rate,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParamDataset(torch.utils.data.Dataset):
    """Loads dataset produced by batch_harness.py (livespice backend).

    Expected directory structure:
        dataset/
            sweep.wav        — input audio used during capture
            outputs.npy      — float32 [N_perms, N_samples] (run --combine first)
            params.csv       — idx, knob1, ..., ok, error columns
            config.json      — must contain a "knobs" list
    """
    def __init__(self, dataset_dir: str, crop_len: int = 48000, repeats: int = 1,
                 target_dbfs: float = -18.0, mmap: bool = False):
        self.dir = Path(dataset_dir)
        with open(self.dir / "config.json") as f:
            self.config = json.load(f)
        self.param_names = self.config["knobs"]
        self.config["param_names"] = self.param_names  # alias for export_nam
        self.num_params = len(self.param_names)
        self.crop_len = crop_len
        self.repeats = repeats

        # Load input sweep and normalize to target_dbfs RMS
        inp_raw, _ = sf.read(str(self.dir / "sweep.wav"))
        if inp_raw.ndim > 1:
            inp_raw = inp_raw.mean(axis=1)
        inp_raw = inp_raw.astype(np.float32)
        rms = float(np.sqrt(np.mean(inp_raw ** 2)))
        target_rms = 10 ** (target_dbfs / 20.0)
        self._scale = target_rms / (rms + 1e-8)
        self.inp = inp_raw * self._scale

        # Load combined outputs — fully into RAM (fast, high memory) or mmap (low memory, SSD-speed)
        out_path = self.dir / "outputs.npy"
        if not out_path.exists():
            raise FileNotFoundError(
                f"outputs.npy not found. Run first:\n"
                f"  python batch_harness.py --combine {self.dir}"
            )
        if mmap:
            print(f"  Memory-mapping outputs.npy (mmap mode) ...", file=sys.stderr, flush=True)
            self.outputs = np.load(str(out_path), mmap_mode="r")
        else:
            print(f"  Loading outputs.npy into RAM ...", file=sys.stderr, flush=True)
            self.outputs = np.load(str(out_path))

        # Load params.csv — successful rows only, ordered by permutation idx.
        # outputs.npy (built by batch_harness.py --combine) is COMPACTED: row i
        # is the i-th surviving .npy in sorted-filename order, not row `idx`. If
        # any permutation failed, the raw CSV `idx` has gaps and is no longer a
        # valid row index -- use the post-sort enumeration position instead.
        rows = []
        with open(self.dir / "params.csv", newline="") as f:
            for row in csv.DictReader(f):
                if int(row["ok"]) == 1:
                    rows.append(row)
        rows.sort(key=lambda r: int(r["idx"]))
        self.samples = [
            (i, {n: float(r[n]) for n in self.param_names})
            for i, r in enumerate(rows)
        ]

        self._check_output_diversity()

    def _check_output_diversity(self) -> None:
        """Spot-check a sample of output pairs for accidental duplicates."""
        if len(self.samples) < 2:
            return
        rng = np.random.default_rng(42)
        check = rng.choice(len(self.samples), size=min(20, len(self.samples)), replace=False)
        for i in range(len(check)):
            for j in range(i + 1, len(check)):
                idx_i = self.samples[int(check[i])][0]
                idx_j = self.samples[int(check[j])][0]
                if np.abs(self.outputs[idx_i] - self.outputs[idx_j]).max() < 1e-7:
                    warnings.warn(
                        f"Outputs {idx_i} and {idx_j} are identical "
                        f"— dataset may be invalid. Check param_map names "
                        "in batch_harness.py match circuit component names."
                    )

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        perm_idx, params = self.samples[real_idx]

        inp = self.inp
        out_row = self.outputs[perm_idx]          # mmap row view — not materialized yet

        sig_len = min(len(inp), out_row.shape[0])
        if sig_len > self.crop_len:
            start = np.random.randint(0, sig_len - self.crop_len)
            inp = inp[start:start + self.crop_len]
            # Scale (scalar) only the crop window, not the whole 5.76M-sample
            # row — avoids reading/processing 120x the data we actually use.
            out = (out_row[start:start + self.crop_len] * self._scale).astype(np.float32)
        else:
            pad = self.crop_len - sig_len
            inp = np.pad(inp[:sig_len], (0, pad))
            out = np.pad((out_row[:sig_len] * self._scale).astype(np.float32), (0, pad))

        inp_t = torch.from_numpy(inp.copy()).float().unsqueeze(0)
        out_t = torch.from_numpy(out.copy()).float().unsqueeze(0)
        params_t = torch.tensor([params[n] for n in self.param_names]).float()
        return inp_t, out_t, params_t


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mrstft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss (L1 on magnitudes)."""
    fft_sizes = [512, 1024, 2048]
    hop_sizes = [128, 256, 512]
    win_lengths = [512, 1024, 2048]
    total = torch.tensor(0.0, device=pred.device)
    for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        s_pred = torch.stft(pred.squeeze(1), fft, hop, win,
                            torch.hann_window(win, device=pred.device),
                            return_complex=True)
        s_target = torch.stft(target.squeeze(1), fft, hop, win,
                              torch.hann_window(win, device=pred.device),
                              return_complex=True)
        mag_pred = torch.abs(s_pred)
        mag_target = torch.abs(s_target)
        total += F.l1_loss(mag_pred, mag_target)
    return total / len(fft_sizes)


def pre_emphasis(x: torch.Tensor, coef: float) -> torch.Tensor:
    """First-order high-pass.

    NAM's trainer offers this (`pre_emph_coef` + `pre_emph_weight`) but it is OFF BY DEFAULT, and
    there it is an extra *MSE* term. Do not cite it as precedent for applying it to ESR.

    Distortion character lives in the harmonics. Un-emphasised, the loss is dominated by
    low-frequency fundamental energy, which is the easy part."""
    if coef <= 0.0:
        return x
    return x[..., 1:] - coef * x[..., :-1]


def esr_per_example(pred: torch.Tensor, target: torch.Tensor, floor: float) -> torch.Tensor:
    """Error-to-signal ratio, normalised PER EXAMPLE, then averaged.

    NOTE, and heed it -- NAM's own trainer carries this warning:

        "Be careful when computing ESR on minibatches! The average ESR over a minibatch of data
         is not the same as the ESR of all of the same data calculated at once (because of the
         denominator). (Hint: think about what happens if one item in the minibatch is all
         zeroes...)"

    That is a direct warning about this function. And it bites us *precisely where we care*: the
    crops we are trying to upweight -- the fading tails -- ARE the near-silent ones. A 0.5 s crop
    at RMS 0.09 has energy ~194, but a crop deep in a decay tail has energy ~0.02, and a silent
    one has ~0. An absolute epsilon does not save you: it is negligible for the loud crops and
    useless for the quiet ones.

    So the denominator is floored RELATIVE TO THE BATCH: no example may be normalised by less than
    `floor` times the batch's mean energy. A silent crop then contributes a bounded amount instead
    of an unbounded one, while a genuinely quiet-but-real crop is still counted at full weight.

    Why depart from NAM's plain MSE at all -- see below. Short version: NAM is a STATIC modeller.
    One device, one knob setting, one input level. It has no permutations, so the cross-setting
    energy bias that motivates this simply does not exist for it, and MSE is a perfectly good
    objective there. Our problem is parametric: 126 permutations spanning a 70x ENERGY range. That
    is a different optimisation problem and it needs a different objective. The justification is
    our own measurement (docs/loss-energy-bias.md), NOT an appeal to NAM.

    This is the whole point. Plain MSE is an ABSOLUTE error, so a crop's influence on the
    gradient is proportional to its ENERGY — which has two consequences we measured on the
    Big Muff and both of them are bad:

      * ACROSS the knob grid, output RMS spans 0.090 .. 0.755 — a 70x ENERGY ratio. Under MSE
        the loudest 8% of permutations take 28% of the gradient and the quietest half get 17%.
        The parametric model neglects the quiet settings, which drags its average ESR above a
        static model's. It was never only a capacity problem.

      * WITHIN a permutation, a decaying note's tail is worth nothing. On the most distorted
        Big Muff setting, windows below -40 dB of peak are 8.2% of the DURATION and 0.00% of
        the GRADIENT. That is the Boss DS-1 bug: as a note fades, the model drops the
        distortion, because staying dirty down there earns it nothing.

    Dividing by each example's own energy makes every permutation, and every crop, count
    equally -- by construction."""
    num = ((pred - target) ** 2).sum(dim=(1, 2))
    den = (target ** 2).sum(dim=(1, 2))
    den = torch.clamp(den, min=floor * den.mean().detach() + 1e-12)
    return (num / den).mean()


class ParamLoss(nn.Module):
    """ESR (per-example) + MRSTFT, with optional pre-emphasis.

    `--loss mse` restores the old absolute-error behaviour, for A/B only. It is not a good
    objective: we REPORT ESR but used to TRAIN on MSE, so the metric and the objective
    disagreed about what mattered."""

    def __init__(self, mrstft_weight: float = 0.1, kind: str = "esr",
                 pre_emph: float = 0.85, floor: float = 0.05):
        super().__init__()
        self.mrstft_weight = mrstft_weight
        self.kind = kind
        self.pre_emph = pre_emph
        self.floor = floor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "esr":
            p = pre_emphasis(pred, self.pre_emph)
            t = pre_emphasis(target, self.pre_emph)
            main = esr_per_example(p, t, self.floor)
        else:
            main = F.mse_loss(pred, target)
        mrstft = mrstft_loss(pred, target)
        return main + self.mrstft_weight * mrstft


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0,
                epoch: int = 0, total_epochs: int = 0, log_interval: int = 10):
    model.train()
    slimmable = isinstance(model, SlimmableParametricA2)
    total_loss = 0
    for step, (inp, out, params) in enumerate(loader):
        _watchdog_arm(f"epoch {epoch} step {step}")
        inp, out, params = inp.to(device), out.to(device), params.to(device)
        optimizer.zero_grad()
        if slimmable:
            preds = model(inp, params)                       # list, one per width
            loss = sum(criterion(p, out) for p in preds)     # joint over all tiers
        else:
            pred = model(inp, params)
            loss = criterion(pred, out)
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()
        _watchdog_disarm()
        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"  [{epoch:3d}/{total_epochs}] step {step+1}/{len(loader)}  "
                  f"loss={total_loss/(step+1):.6f}", file=sys.stderr, flush=True)
    return total_loss / len(loader)


def validate(model, loader, criterion, device, val_passes: int = 1):
    """Returns (mean_val_loss, esr_per_tier) where esr_per_tier is a list of ESRs
    in ascending-width order (len = #tiers; a single-element list for a plain,
    non-slimmable model).

    ESR here is per-example — each example normalised by its OWN target energy,
    then averaged across the batch — with the same floor `criterion` trains
    against (falls back to ParamLoss's own default, 0.05, for callers that pass a
    plain criterion with no `.floor`, e.g. export_checkpoint.py's verification
    step). This matches both NAM's own official esr() (nam/models/losses.py:
    per-example mean over the sample axis, THEN mean over the batch axis — NOT a
    single ratio over the whole batch at once) and this codebase's own training
    objective (esr_per_example). No pre-emphasis here, unlike the training loss:
    pre-emphasis is a loss-SHAPING choice specific to this codebase, not part of
    NAM's esr() definition, and every other circuit's published ESR_RECORD.md
    already reports plain (non-pre-emphasised) ESR -- changing that would make
    this circuit's numbers incomparable to its own history and to every other one.

    Previously this used a plain batch-aggregate ratio (sum of squared error over
    the WHOLE batch / sum of target energy over the whole batch), which is neither
    of the above: a batch's ratio is dominated by whichever examples in it happen
    to be loudest, so the reported number (and therefore best-checkpoint
    selection) could vary by tens of percent from one random crop draw to the
    next, independent of actual model quality -- see param_train.py's own
    esr_per_example() docstring ("the Boss DS-1 bug") for why training's loss
    already avoided this; validate() just never got the same fix.

    val_passes repeats the ENTIRE val_loader this many times, averaging every pass
    into the same running total. ParamDataset re-crops randomly on every single
    __getitem__ call (see its own docstring), so iterating the same loader again
    draws genuinely different random windows, not the same ones twice -- this is
    real additional coverage, not a wasted repeat. NAM's own trainer never needed
    this: its Dataset uses fixed, non-random sample windows (checked directly,
    nam/data.py), so its validation is already fully deterministic epoch to epoch.
    This codebase's parametric extension (many knob permutations, each covered via
    randomly-cropped repeats) is what introduces the noise val_passes averages
    down -- increasing it trades epoch wall-clock time for a less noisy reading
    (variance falls roughly as 1/val_passes, i.e. std roughly as 1/sqrt(val_passes)
    for independent draws), which matters most for best-checkpoint selection
    (see the plain "if new < best: save" rule below in the training loop, which
    has no other noise-smoothing of its own -- NAM's own PackedBestCheckpoint uses
    the identical rule, checked directly, so there was no existing technique to
    borrow; making the input to that rule less noisy is the mitigation available
    without changing the selection rule itself).
    """
    model.eval()
    slimmable = isinstance(model, SlimmableParametricA2)
    floor = getattr(criterion, "floor", 0.05)
    total_loss = 0.0
    esr_sums = None
    n = 0
    with torch.no_grad():
        for _ in range(max(1, val_passes)):
            for inp, out, params in loader:
                inp, out, params = inp.to(device), out.to(device), params.to(device)
                if slimmable:
                    preds = model(inp, params)                        # list, ascending width
                    total_loss += sum(criterion(p, out) for p in preds).item()
                    e = [esr_per_example(p, out, floor).item() for p in preds]
                else:
                    pred = model(inp, params)
                    total_loss += criterion(pred, out).item()
                    e = [esr_per_example(pred, out, floor).item()]
                esr_sums = e if esr_sums is None else [a + b for a, b in zip(esr_sums, e)]
                n += 1
    return total_loss / n, [s / n for s in esr_sums]


# ---------------------------------------------------------------------------
# Parameter sensitivity check
# ---------------------------------------------------------------------------

def param_sensitivity(model: ParametricA2, device: str,
                      sweep: np.ndarray, n_steps: int = 11) -> dict:
    """Sweep each knob from 0→1, measure output change.

    Returns dict of {param_name: max_output_diff}.
    If any diff is below threshold, the model may not have learned
    parametric behavior.
    """
    model.eval()
    inp = torch.from_numpy(sweep).float().unsqueeze(0).unsqueeze(0).to(device)
    results: dict[str, float] = {}
    with torch.no_grad():
        ref_params = torch.full((1, model.num_params), 0.5, device=device)
        ref_out = model(inp, ref_params)
        for pi in range(model.num_params):
            max_diff = 0.0
            for step in range(n_steps):
                p = step / (n_steps - 1)
                params = torch.full((1, model.num_params), 0.5, device=device)
                params[0, pi] = p
                out = model(inp, params)
                diff = (out - ref_out).abs().max().item()
                max_diff = max(max_diff, diff)
            results[f"param_{pi}"] = max_diff
    return results


# ---------------------------------------------------------------------------
# Graceful-stop signalling (for open-ended --epochs 0 training)
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[stop] signal received — finishing current epoch, then saving and exiting.",
          file=sys.stderr, flush=True)


def should_stop(ckpt_dir) -> bool:
    """True if a stop was requested via SIGINT/SIGTERM or a STOP sentinel file
    in the checkpoint dir. Consumes the STOP file so a later resume won't
    immediately re-trigger."""
    if _STOP_REQUESTED:
        return True
    if ckpt_dir is not None and (ckpt_dir / "STOP").exists():
        try:
            (ckpt_dir / "STOP").unlink()
        except OSError:
            pass
        print("[stop] STOP file found — finishing up and exiting.", file=sys.stderr, flush=True)
        return True
    return False


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def nam_variant(output: Path, tag: str) -> Path:
    """Sibling .param.nam path carrying a tag.

    model.param.nam + "best_lite" -> model.best_lite.param.nam
    """
    name = output.name
    if name.endswith(".param.nam"):
        base = name[: -len(".param.nam")]
        return output.with_name(f"{base}.{tag}.param.nam")
    return output.with_name(f"{output.stem}_{tag}{output.suffix}")


def export_nam_state(model, state, dataset, path: Path, device):
    """Export `state`'s weights to a .param.nam at `path`, then restore the
    model's current weights (so training can continue untouched)."""
    current = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.to(device)
    nam = model.export_nam(dataset.config, {"version": "0.7.0"}, sample_rate=48000,
                           input_audio=dataset.inp)
    path.write_text(json.dumps(nam, separators=(",", ":")))
    model.load_state_dict(current)
    model.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Train parametric A2 + FiLM NAM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, type=Path,
                    help="Dataset directory from batch_harness.py")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output .param.nam file path")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs, or 0 = open-ended: run until "
                         "stopped (touch <ckpt>/STOP, or SIGINT/SIGTERM). Best models "
                         "export live, so you can stop whenever they're good enough. "
                         "(default: %(default)s)")
    ap.add_argument("--restart-period", type=int, default=50,
                    help="Open-ended mode: SGDR cosine-warm-restart period in epochs "
                         "(default: %(default)s). Each low-LR trough tends to mint a new best.")
    ap.add_argument("--restart-mult", type=int, default=1,
                    help="Open-ended mode: SGDR period multiplier per restart "
                         "(1 = equal cycles; 2 = doubling). (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Batch size (default: %(default)s)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: %(default)s)")
    ap.add_argument("--film-lr-mult", type=float, default=5.0,
                    help="LR multiplier for FiLM knob-conditioning params — they get a "
                         "small gradient signal for subtle knob effects, so boost it "
                         "(default: %(default)s; 1.0 = off)")
    ap.add_argument("--knob-boost", type=str, default=None,
                    help="Per-knob gradient boost on FiLM conditioning weights: "
                         "NAME=mult,NAME2=mult2,... (e.g. 'Drive=3.0'). Unlike "
                         "--film-lr-mult (which boosts ALL knobs' conditioning equally), "
                         "this scales the backward-pass gradient into just the named "
                         "knob's column of every FiLM layer's weight matrix -- for a knob "
                         "whose true audible effect is small relative to others (e.g. "
                         "Drive next to Tone), so a capacity-limited tier doesn't learn "
                         "to ignore it in favor of the knob with the bigger loss payoff. "
                         "Device-agnostic: works for any knob name present in the "
                         "dataset's config.json, not specific to any one circuit.")
    ap.add_argument("--crop-len", type=int, default=44100,
                    help="Random crop length in samples (default: %(default)s)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Virtual dataset multiplier — increases steps/epoch without "
                         "changing the audio data (default: %(default)s)")
    ap.add_argument("--val-split", type=float, default=0.1,
                    help="Fraction of samples for validation (default: %(default)s)")
    ap.add_argument("--val-passes", type=int, default=4,
                    help="Repeat the full validation pass this many times per epoch and "
                         "average -- ParamDataset re-crops randomly on every call, so this is "
                         "real additional coverage, not a repeated identical reading. Reduces "
                         "noise in the reported val_esr (variance falls ~1/val_passes), which "
                         "matters most for best-checkpoint selection: that's a plain "
                         "'new best if lower' rule with no smoothing of its own, so a less "
                         "noisy input to it is the mitigation available without changing the "
                         "rule itself. Costs roughly (val_passes-1) extra forward-only passes "
                         "over the (much smaller than train) val set per epoch. "
                         "(default: %(default)s)")
    ap.add_argument("--loss", choices=["esr", "mse"], default="esr",
                    help="Training objective. 'esr' normalises each example by its own energy, so "
                         "quiet knob settings and fading notes count as much as loud ones. 'mse' is "
                         "the old absolute-error loss -- kept only for A/B; it lets the loudest 8%% "
                         "of permutations take 28%% of the gradient and gives a note's fade-out "
                         "0.00%% of it. (default: esr)")
    ap.add_argument("--pre-emph", type=float, default=0.85,
                    help="Pre-emphasis coefficient for the ESR loss (0 = off). Distortion lives in "
                         "the harmonics; un-emphasised, the loss is dominated by the fundamental.")
    ap.add_argument("--esr-floor", type=float, default=0.05,
                    help="Floor on the per-example ESR denominator, as a FRACTION OF THE BATCH MEAN "
                         "energy. An absolute epsilon does not work: it is negligible for loud crops "
                         "and useless for the near-silent fading tails we are specifically trying to "
                         "upweight. NAM's trainer warns about exactly this (a batch item of all "
                         "zeroes). 0 disables the floor -- don't.")
    ap.add_argument("--mrstft-weight", type=float, default=0.1,
                    help="MRSTFT loss weight (default: %(default)s)")
    ap.add_argument("--device", default="auto",
                    help="Device: auto, cpu, cuda, mps (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: %(default)s)")
    ap.add_argument("--widths", type=str, default=None,
                    help="Comma-separated slimmable channel widths, e.g. '3,4,8' "
                         "(default: 3,8). Narrowest = lite tier, widest = full tier; "
                         "middle tiers logged/checkpointed as w<N>.")
    ap.add_argument("--skip-param-sensitivity", action="store_true",
                    help="Skip the parameter sensitivity check after training (runs by default)")
    ap.add_argument("--skip-per-perm-esr", action="store_true",
                    help="Skip the per-permutation ESR check after training (runs by default). "
                         "Runs full-length inference over every permutation per tier -- can be "
                         "slow for large grids (e.g. a 1,944-permutation config).")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Directory to save epoch checkpoints and metrics CSV")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Checkpoint .pt to resume from")
    ap.add_argument("--log-csv", type=Path, default=None,
                    help="Path for metrics CSV (default: --checkpoint-dir/metrics.csv)")
    ap.add_argument("--mmap", action="store_true", default=True,
                    help="Memory-map outputs.npy instead of loading into RAM (~3.5 GB freed; "
                         "default on -- every recipe in this repo targets SSD storage and "
                         "memory pressure is a real constraint here. Use --no-mmap to force a "
                         "full RAM load instead (marginally faster on abundant-RAM machines).")
    ap.add_argument("--no-mmap", action="store_false", dest="mmap",
                    help="Load outputs.npy fully into RAM instead of memory-mapping it")
    ap.add_argument("--modeled-by", default=None,
                    help="Credit for who trained/captured this model, written into the exported "
                         ".nam's metadata (e.g. --modeled-by 'Gene Ko'). A training run is who's "
                         "doing this specific capture, not a property of the reusable dataset, so "
                         "this is a param_train.py flag rather than something baked into the "
                         "dataset's config.json by batch_harness.py -- overrides anything already "
                         "there if both are set.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset from {args.dataset} ...", file=sys.stderr)
    dataset = ParamDataset(str(args.dataset), crop_len=args.crop_len, repeats=args.repeats,
                           mmap=args.mmap)
    if args.modeled_by:
        # Mutates dataset.config in place, so every export_nam()/export_nam_state() call below
        # (including mid-training "best" checkpoint exports) picks this up automatically --
        # they all read this same dict, not a copy taken at dataset-load time.
        dataset.config["modeled_by"] = args.modeled_by
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    # drop_last=True on BOTH loaders: 2026-07-21, King of Tone (train 3554, val 394 --
    # neither divides evenly by batch_size=64) hung on MPS within 1-2 epochs, fresh or
    # resumed, while OD-3 (train 3456, val 384 -- BOTH exact multiples of 64) never did,
    # in any test tonight. The real difference: a dataset whose sizes don't divide evenly
    # forces a smaller last batch every epoch (KoT: 34 then 64 then 10), so MPS has to
    # keep re-specializing MRSTFT's stft() kernels for multiple shapes instead of one --
    # a documented class of MPSGraph deadlock (github.com/jamiepine/voicebox#905). Losing
    # up to batch_size-1 samples/epoch (here: 34 train, 10 val, ~1%/2.5%) is a small,
    # standard price -- val_passes already averages multiple passes for exactly this kind
    # of noise, and a shape-stable run beats an exhaustive one that hangs.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    print(f"  {n_total} total samples ({n_train} train, {n_val} val)",
          file=sys.stderr)
    print(f"  Params: {dataset.param_names}", file=sys.stderr)

    ckpt_dir = args.checkpoint_dir
    log_csv = args.log_csv
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if log_csv is None:
            log_csv = ckpt_dir / "metrics.csv"

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    num_params = dataset.num_params
    model = SlimmableParametricA2(num_params, widths=parse_widths(args.widths))
    desc = ", ".join(f"w{w}({m.weight_count()}w)"
                     for w, m in zip(model.widths, model.submodels))
    print(f"\nModel: SlimmableParametricA2  [{desc}], {num_params} params", file=sys.stderr)
    model.to(device)

    # FiLM (knob-conditioning) params get a higher LR — their gradient signal is
    # small (knob effects are subtle vs the overall signal), so at the shared LR
    # they under-learn and the knob goes near-dead.
    film_params = [p for n, p in model.named_parameters() if ".film." in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if ".film." not in n and p.requires_grad]
    groups = [{"params": other_params, "lr": args.lr}]
    if film_params:   # empty for a non-parametric (0-knob) static capture
        groups.append({"params": film_params, "lr": args.lr * args.film_lr_mult})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    print(f"FiLM fix: {len(film_params)} FiLM tensors at {args.film_lr_mult}× LR "
          f"({args.lr * args.film_lr_mult:.1e}), non-zero init (std 0.1)", file=sys.stderr)

    if args.knob_boost:
        boosts = parse_knob_boost(args.knob_boost)
        n_hooked = register_knob_boost_hooks(model, dataset.param_names, boosts)
        desc = ", ".join(f"{name}×{mult}" for name, mult in boosts.items())
        print(f"Knob boost: {desc} on {n_hooked} FiLM tensors", file=sys.stderr)

    start_epoch = 1
    # Per-tier best tracking. `labels` is ascending-width order; the widest tier
    # ("full") is primary — it drives best.pt, args.output, and resume.
    labels = model.tier_labels()
    # DISPLAY labels are the actual channel widths (w3..w8), so the log/metrics are unambiguous --
    # "lite"/"full" are positional and depend on the widths list, which is confusing. The lite/full
    # keys still index the dicts + name the checkpoint files + drive resume/release for back-compat;
    # only what's printed/columned changes.
    wlabel = {lbl: f"w{w}" for lbl, w in zip(labels, model.widths)}
    best_esr = {lbl: float("inf") for lbl in labels}    # label -> best val ESR
    best_state = {lbl: None for lbl in labels}          # label -> weights snapshot

    if args.resume is not None:
        print(f"Resuming from {args.resume} ...", file=sys.stderr)
        # Load to CPU, then move to `device` ourselves in small, explicit steps -- NOT
        # map_location=device, which bulk-deserializes the whole checkpoint (model weights
        # AND the full Adam state, exp_avg/exp_avg_sq/step per parameter -- 2x the param
        # count) directly onto MPS in one shot. 2026-07-21: every resumed run hung on MPS
        # tonight; a fresh (no-resume) run never did. This bulk one-shot transfer -- so
        # different from a fresh run's gradual, incremental device population (params
        # moved via model.to(device), optimizer state lazily created one .step() at a
        # time) -- is the leading suspect. model.load_state_dict() below already copies
        # per-parameter into the (already on-device) model, so only optimizer state needs
        # an explicit, separate move.
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = ckpt["epoch"] + 1
        # new per-tier format ...
        bt = ckpt.get("best_esr_by_tier")
        if bt is not None:
            best_esr.update({k: v for k, v in bt.items() if k in best_esr})
            best_state.update({k: v for k, v in ckpt.get("best_state_by_tier", {}).items()
                               if k in best_state})
        else:
            # ... or fall back to the old 2-tier scalar fields
            if "full" in best_esr and ckpt.get("best_esr") is not None:
                best_esr["full"] = ckpt["best_esr"]; best_state["full"] = ckpt.get("best_state")
            if "lite" in best_esr and ckpt.get("best_lite_esr") is not None:
                best_esr["lite"] = ckpt["best_lite_esr"]; best_state["lite"] = ckpt.get("best_lite_state")
        print(f"  Resumed at epoch {ckpt['epoch']}, best ESR (full) {best_esr['full']:.6f}",
              file=sys.stderr)

    open_ended = (args.epochs == 0)

    def make_scheduler(last_epoch):
        if open_ended:
            # Horizon-free: cosine warm restarts (SGDR) — LR cycles down and
            # restarts, so late training stays productive indefinitely.
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, args.restart_period),
                T_mult=max(1, args.restart_mult), last_epoch=last_epoch)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, last_epoch=last_epoch)

    scheduler = make_scheduler(start_epoch - 2)
    if args.resume and "scheduler_last_epoch" in ckpt:
        scheduler = make_scheduler(ckpt["scheduler_last_epoch"])
    criterion = ParamLoss(mrstft_weight=args.mrstft_weight, kind=args.loss,
                          pre_emph=args.pre_emph, floor=args.esr_floor)
    print(f"  Loss: {args.loss}" + (f" (pre-emph {args.pre_emph})" if args.loss == 'esr' else
          "  <-- ABSOLUTE error: loud permutations dominate the gradient"), file=sys.stderr)

    if open_ended:
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if open_ended:
        print(f"\nTraining open-ended (SGDR restarts every {args.restart_period} epochs). "
              f"Stop with: touch {ckpt_dir / 'STOP' if ckpt_dir else 'STOP'}  (or SIGINT/SIGTERM). "
              f"Best models export live.", file=sys.stderr)
    else:
        print(f"\nTraining {args.epochs} epochs ...", file=sys.stderr)
    if log_csv is not None:
        log_f = open(log_csv, "a" if args.resume else "w", newline="")
        log_w = csv.writer(log_f)
        if not args.resume:
            log_w.writerow(["epoch", "train_loss", "val_loss",
                            *[f"val_esr_{wlabel[lbl]}" for lbl in labels], "lr", "elapsed_s"])

    _watchdog_open(ckpt_dir)
    t0 = time.time()
    epoch = start_epoch - 1
    completed = 0
    while True:
        epoch += 1
        if not open_ended and epoch > args.epochs:
            break
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 epoch=epoch, total_epochs=args.epochs, log_interval=10)
        # Drain the MPS async queue between train (batch=args.batch_size) and val
        # (batch=len(val_ds)%batch_size or smaller) -- two distinct batch shapes back
        # to back, every epoch, with no sync between them. Matches a documented class
        # of MPS deadlock (MPSGraph kernel re-specialization under shape-switching,
        # e.g. github.com/jamiepine/voicebox#905) closely enough to be worth the
        # near-zero cost: synchronize() drains the queue WITHOUT evicting the kernel
        # cache (unlike empty_cache(), which would force full re-specialization).
        if device == "mps":
            torch.mps.synchronize()
        _watchdog_arm(f"epoch {epoch} validate()", timeout=60.0)
        val_loss, esr_list = validate(model, val_loader, criterion, device,
                                     val_passes=args.val_passes)
        if device == "mps":
            torch.mps.synchronize()
        _watchdog_disarm()
        scheduler.step()
        esr_by = dict(zip(labels, esr_list))     # label -> this-epoch val ESR

        # --- per-tier best-checkpointing: each width's optimum lands at a
        #     different epoch, so snapshot each independently. The widest tier
        #     ("full") writes best.pt and drives args.output + resume; the others
        #     write best_<label>.pt (e.g. best_lite.pt, best_w4.pt). ---
        new_best = {}
        for lbl in labels:
            e = esr_by[lbl]
            if e < best_esr[lbl]:
                _watchdog_arm(f"epoch {epoch} best-checkpoint save ({lbl})", timeout=60.0)
                best_esr[lbl] = e
                best_state[lbl] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                new_best[lbl] = True
                if ckpt_dir is not None:
                    fname = "best.pt" if lbl == "full" else f"best_{lbl}.pt"
                    torch.save({
                        "epoch": epoch,
                        "model": best_state[lbl],
                        "optimizer": optimizer.state_dict(),
                        "scheduler_last_epoch": scheduler.last_epoch,
                        "best_esr": best_esr["full"],          # back-compat scalar (full)
                        "best_esr_by_tier": dict(best_esr),
                        "args_dict": dict(vars(args)),
                    }, ckpt_dir / fname)
                export_nam_state(model, best_state[lbl], dataset,
                                 nam_variant(args.output, f"best_{lbl}"), device)
                _watchdog_disarm()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # Console: one ESR per tier, '*' marks tiers that improved this epoch
        esr_str = "  ".join(f"{wlabel[lbl]}={esr_by[lbl]:.6f}{'*' if new_best.get(lbl) else ''}"
                            for lbl in labels)
        if open_ended:
            prog, tail = f"[{epoch:4d}/inf]", f"({elapsed/3600:.2f}h)"
        else:
            eta = (elapsed / (epoch - start_epoch + 1)) * (args.epochs - epoch)
            prog, tail = f"[{epoch:3d}/{args.epochs}]", f"({elapsed/3600:.2f}h, ETA {eta/3600:.2f}h)"
        print(f"  {prog}  train={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"ESR[{esr_str}]  lr={lr_now:.2e}  {tail}",
              file=sys.stderr, flush=True)

        # Log CSV: one val_esr_<tier> column per width
        if log_csv is not None:
            log_w.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}",
                            *[f"{esr_by[lbl]:.8f}" for lbl in labels],
                            f"{lr_now:.2e}", f"{elapsed:.1f}"])
            log_f.flush()

        # Save checkpoint every epoch (overwrite previous to save disk space)
        if ckpt_dir is not None:
            _watchdog_arm(f"epoch {epoch} latest.pt save", timeout=60.0)
            ckpt_path = ckpt_dir / "latest.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler_last_epoch": scheduler.last_epoch,
                # per-tier best tracking (new format)
                "best_esr_by_tier": dict(best_esr),
                "best_state_by_tier": dict(best_state),
                # back-compat scalar fields (full tier)
                "best_esr": best_esr["full"],
                "best_state": best_state["full"],
                "best_lite_esr": best_esr.get("lite", float("inf")),
                "best_lite_state": best_state.get("lite"),
                "args_dict": dict(vars(args)),
            }, ckpt_path)
            _watchdog_disarm()

        completed += 1
        # Open-ended: stop gracefully on STOP file or SIGINT/SIGTERM.
        if open_ended and should_stop(ckpt_dir):
            break

    elapsed = time.time() - t0
    if log_csv is not None:
        log_f.close()

    print(f"\nTraining finished ({elapsed:.0f}s, {elapsed/max(1, completed):.1f}s/epoch, "
          f"{completed} epochs)",
          file=sys.stderr)
    esr_summary = ", ".join(f"{wlabel[lbl]} {best_esr[lbl]:.6f}"
                            for lbl in labels if best_state[lbl] is not None)
    print(f"Best validation ESR by tier: {esr_summary}", file=sys.stderr)

    # Finalize each non-primary tier's .param.nam from its own best weights (they
    # differ from the full-best epoch, so must be written from that tier's state).
    for lbl in labels:
        if lbl != "full" and best_state[lbl] is not None:
            export_nam_state(model, best_state[lbl], dataset,
                             nam_variant(args.output, f"best_{lbl}"), device)

    # Compose the single best container exported to args.output: each tier's own
    # submodel weights taken from ITS best epoch, spliced together. Valid because
    # the tiers are independent (no shared weights) — the container just selects
    # one at inference — so args.output ends up optimal on EVERY tier, not just
    # the widest. (The per-tier .best_<tier>.param.nam above stay for reference.)
    if best_state["full"] is not None:
        composite = {k: v.clone() for k, v in best_state["full"].items()}
        for i, lbl in enumerate(labels):
            if lbl == "full" or best_state[lbl] is None:
                continue
            pref = model.tier_state_prefix(i)
            for k, v in best_state[lbl].items():
                if k.startswith(pref):
                    composite[k] = v.clone()
        model.load_state_dict(composite)
        model.to(device)
        print("  Composed best-of-every-tier container for "
              f"{args.output.name}", file=sys.stderr)

    # Save best checkpoint(s): best.pt (full) + best_<label>.pt per other tier.
    if ckpt_dir is not None:
        for lbl in labels:
            if best_state[lbl] is None:
                continue
            fname = "best.pt" if lbl == "full" else f"best_{lbl}.pt"
            torch.save({
                "epoch": epoch,
                "model": best_state[lbl],
                "optimizer": optimizer.state_dict(),
                "scheduler_last_epoch": scheduler.last_epoch,
                "best_esr": best_esr["full"],
                "best_esr_by_tier": dict(best_esr),
                "args_dict": dict(vars(args)),
            }, ckpt_dir / fname)
            print(f"  Best {lbl} model saved to {ckpt_dir / fname}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Parameter sensitivity check -- runs by default; --skip-param-sensitivity to skip.
    # ------------------------------------------------------------------
    if not args.skip_param_sensitivity:
        print(f"\nParameter sensitivity check ...", file=sys.stderr)
        sweep_audio, _ = sf.read(str(dataset.dir / "sweep.wav"))
        if sweep_audio.ndim > 1:
            sweep_audio = sweep_audio.mean(axis=1)
        sweep_audio = sweep_audio[:48000]
        # All tiers, not just the two endpoints -- model.full/model.lite alone
        # silently skipped any middle tier (e.g. a w4 in widths [3,4,8]).
        for m, lbl in zip(model.submodels, model.tier_labels()):
            prefix = f"[{lbl}] "
            sens = param_sensitivity(m, device, sweep_audio)
            for k, v in sens.items():
                pname = dataset.param_names[int(k.split("_")[-1])] if "_" in k else k
                print(f"  {prefix}{pname}: max_diff = {v:.6f}", file=sys.stderr)
                if v < 1e-6:
                    warnings.warn(f"  {prefix}{pname}: output doesn't change with this param "
                                  f"(max_diff={v:.2e}) — model may be ignoring knobs")

    # ------------------------------------------------------------------
    # Per-permutation ESR -- runs by default; --skip-per-perm-esr to skip.
    # Full-length inference per permutation (not a training crop), per tier --
    # shows WHERE in the knob grid the model is weak, not just an average.
    # Reuses the already-loaded dataset in memory; no re-read from disk.
    # ------------------------------------------------------------------
    if not args.skip_per_perm_esr:
        print(f"\nPer-permutation ESR check ...", file=sys.stderr)
        for m, lbl in zip(model.submodels, model.tier_labels()):
            results = compute_per_perm_esr(m, dataset.inp, dataset.outputs, dataset.samples,
                                           dataset.param_names, device, dataset._scale)
            print(f"\n{summarize_per_perm_esr(results, lbl)}", file=sys.stderr)
            if ckpt_dir is not None:
                csv_path = ckpt_dir / f"per_perm_esr_{lbl}.csv"
                write_per_perm_esr_csv(results, dataset.param_names, csv_path)
                print(f"  Wrote {csv_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Export .param.nam
    # ------------------------------------------------------------------
    print(f"\nExporting to {args.output} ...", file=sys.stderr)
    nam_data = model.export_nam(dataset.config, {"version": "0.7.0"}, sample_rate=48000,
                                input_audio=dataset.inp)
    args.output.write_text(json.dumps(nam_data, separators=(",", ":")))
    print(f"  Architecture: {nam_data['architecture']}", file=sys.stderr)
    for sm in nam_data["config"]["submodels"]:
        w = len(sm["model"]["weights"])
        ch = sm["model"]["config"]["layers"]
        print(f"    max_value={sm['max_value']}  {ch}ch  {w} weights", file=sys.stderr)

    # Verify round-trip
    print(f"  Verifying round-trip ...", file=sys.stderr)
    test_inp = torch.randn(1, 1, 8192, device=device)
    test_params = torch.rand(1, num_params, device=device)
    # Pair each exported submodel with its SOURCE tier by ascending width. (Both
    # nam_data["config"]["submodels"] and model.submodels are width-ascending, and
    # tier_labels() matches that order.) Earlier this hardcoded ["lite","full"],
    # which for 3+ tiers mislabeled and compared the wrong pair (e.g. the 4ch export
    # vs the 8ch model) — a false WARN that also never tested the widest tier.
    for sm_data, lbl, src in zip(nam_data["config"]["submodels"],
                                 model.tier_labels(), model.submodels):
        ch = sm_data["model"]["config"]["layers"]
        m2 = ParametricA2(ch, num_params)
        m2.load_weights(sm_data["model"]["weights"])
        m2.to(device)
        with torch.no_grad():
            o1 = src(test_inp, test_params)
            o2 = m2(test_inp, test_params)
        md = (o1 - o2).abs().max().item()
        status = f"OK (max_diff={md:.2e})" if md <= 1e-6 else f"WARN max_diff={md:.2e}"
        print(f"    [{lbl}] round-trip {status}", file=sys.stderr)

    print(f"\nDone. Model saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
