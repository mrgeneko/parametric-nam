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
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as apply_spectral_norm

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
#   * The legacy "residual" head (head reads the final residual) was a host-app-only
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
    """Feature-wise Linear Modulation: gamma * x + beta.

    gamma_beta() is the ONE place this formula is computed -- forward() (training/live
    inference) and nam_standard.fold_film() (baking) both call it. fold_film() used to
    reimplement this affine step inline, independently of FiLM; that silent duplication
    is exactly the kind of two-place divergence worth avoiding, so both now share this
    method instead.
    """
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

    def gamma_beta(self, cond: torch.Tensor):
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.gamma_beta(cond)
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
    """Single A2 dilated conv layer with optional FiLM.

    spectral_norm: constrains conv/mixin/l1x1's spectral norm (Lipschitz bound) via
    PyTorch's parametrization system -- see docs/film_runaway_investigation.md ("A2"),
    the fix that actually addresses the traced blow-up mechanism (unbounded gain
    compounding through these three stages across the 23-layer residual stack).
    Default False: existing checkpoints/training runs are unaffected until opted in.
    Applying it to ALL THREE stages (not just conv/l1x1) matters -- omitting `mixin`
    (the direct 1x1 injection of the raw audio input) leaves an unbounded parallel
    path into the residual stream that the other two bounds don't cover.

    NOTE: while spectral_norm is applied (see enable_spectral_norm() below), `.weight` on
    these three submodules is a COMPUTED property (from the parametrization's `weight_orig`/
    power-iteration buffers), not a plain leaf Parameter -- code that does `.weight.data.
    copy_(...)` or in-place `.weight.mul_(...)` on them (state-dict loading, nam_standard.
    fold_film()) will silently target the wrong thing. ParametricA2.export_nam() strips this
    back to a plain already-bounded weight via _bake_spectral_norm() before either ever runs.

    Construction ALWAYS builds plain conv/mixin/l1x1 -- spectral_norm wrapping, if wanted, is
    applied via a SEPARATE, later call to enable_spectral_norm(), not here. Why: spectral_norm's
    power-iteration buffers (`_u`/`_v`) are randomly initialized, consuming extra draws from the
    shared global RNG stream. Wrapping inline here would shift every LATER layer's (and, for
    SlimmableParametricA2, every later TIER's) weight initialization relative to an otherwise-
    identical unconstrained run with the same seed -- confirmed empirically (2026-08-02):
    with spectral_norm applied inline, `head.weight` differed between two SlimmableParametricA2
    instances built with identical torch.manual_seed() calls, purely because of this. Deferring
    the wrap to after ALL of a model's raw modules exist (see ParametricA2.enable_spectral_norm()
    and SlimmableParametricA2.__init__) makes every real parameter's initial value bit-identical
    to an unconstrained run with the same seed, isolating A2's actual training-time effect from
    an accidental initialization-lottery difference in side-by-side comparisons.
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, cond_dim: int):
        super().__init__()
        # EXPLICIT causal left-pad, not Conv1d(padding=...) + crop. The old form padded
        # BOTH sides (Conv1d has no one-sided padding), computed `pad` extra trailing
        # samples, cropped them off with a slice that produced a NON-CONTIGUOUS tensor,
        # and then needed a .contiguous() copy of the entire [B, C, T] activation per
        # layer per direction -- gigabytes of pure memcpy per step at batch 64, and the
        # non-contiguous intermediate was the suspected trigger for the 2026-07-21 MPS
        # backward-pass hangs. F.pad(x, (pad, 0)) + an unpadded conv yields exactly T
        # already-contiguous samples, numerically identical (left-pad + crop-to-T IS the
        # causal left-pad), and removes the hang trigger at the source. (Verified
        # bit-equal on CPU against the pad-then-crop form; state-dict layout unchanged --
        # padding is not a parameter.)
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.mixin = nn.Conv1d(1, channels, 1, bias=False)
        self.l1x1 = nn.Conv1d(channels, channels, 1)
        self.film = FiLM(channels, cond_dim) if cond_dim > 0 else None

    def enable_spectral_norm(self):
        """Wrap conv/mixin/l1x1 with spectral_norm. Call ONLY after every raw module this
        model will ever have already exists -- see the class docstring."""
        self.conv = apply_spectral_norm(self.conv)
        self.mixin = apply_spectral_norm(self.mixin)
        self.l1x1 = apply_spectral_norm(self.l1x1)

    def forward(self, x: torch.Tensor, inp_audio: torch.Tensor, cond: torch.Tensor):
        """Returns (residual_out, post_activation). post_activation is the pre-layer1x1
        LeakyReLU output = the 'skip' term NAM/a2_fast accumulate into the head."""
        h = self.conv(F.pad(x, (self.pad, 0)))
        h = h + self.mixin(inp_audio)
        if self.film is not None:
            h = self.film(h, cond)
        post_act = F.leaky_relu(h, K_LEAKY_SLOPE)
        residual_out = x + self.l1x1(post_act)
        return residual_out, post_act


_DBU_0_RMS_VOLTS = 0.7746  # 0dBu reference (sqrt(0.6W * 600ohm), per NAM's own calibration docs)

# input_level_dbu we EXPORT is DERIVED from the schx Input V0dBFS (dBu of a full-scale sine:
# 20*log10((V0dBFS/sqrt(2))/0.7746)). The derivation itself is correct -- what was wrong was
# V0dBFS: our pedals shipped at 0.1V (amps at 0.03-0.05V), ~20 dB below the LiveSPICE ecosystem
# convention of 1V (verified across upstream + 4 community repos, ~270/282 schx). That is why
# input_level_dbu came out ~-20.8 dBu and a calibrating host over-drove every model ~13-33 dB
# (the host app: inputCalibrationGain_dB = userInterfaceCal_dBu - input_level_dbu). With the schx
# corrected to V0dBFS=1V the derivation yields ~-0.8 dBu, which matches the empirically-tuned
# value and gives a Scarlett 2i2 (+12.5 dBu) the correct ~+13 dB makeup. So the fix is: keep
# deriving from V0dBFS, and set V0dBFS to the 1V convention per device. A device captured on
# real gear may still override with a measured level via config["input_level_dbu"].
# See docs/input_calibration.md and LESSONS #20.


def _schx_input_v0dbfs(schx_path) -> "float | None":
    """Read a .schx's Circuit.Input V0dBFS (volts a digital sample of 1.0 maps to -- see
    schx_to_ngspice.py's Input handling), for computing input_level_dbu at export time.
    Returns None if the file is missing, unparseable, or has no Input component -- the
    metadata field is simply omitted rather than exported wrong."""
    if not schx_path:
        return None
    try:
        root = ET.parse(str(schx_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    for el in root.iter("Element"):
        comp = el.find("Component")
        if comp is None:
            continue
        if "Circuit.Input" not in (comp.get("_Type") or ""):
            continue
        raw = comp.get("V0dBFS")
        if not raw:
            continue
        try:
            return float(raw.split()[0])
        except (ValueError, IndexError):
            return None
    return None


def _input_level_dbu(v0dbfs_volts: float) -> float:
    """NAM's input_level_dbu: dBu RMS of a 1kHz sine at 0dBFS peak (see NAM's calibration
    docs, which describe MEASURING this on a real interface: disconnect from the gear,
    play a 0dBFS 1kHz sine, and read the analog output voltage with a multimeter). V0dBFS
    is a peak-referenced volts-per-digital-sample scale (schx_to_ngspice.py), so a 1kHz
    sine at digital peak 1.0 has RMS = V0dBFS/sqrt(2) -- the arithmetic here is identical
    to NAM's own formula, but the INPUT to it is not a measurement. There is no physical
    interface anywhere in this (ngspice/livespice) pipeline -- V0dBFS is the schx author's
    ASSUMED nominal input level for the circuit, not anything measured. It's a plausible
    stand-in (the DS-1's 0.1V is a commonly-cited "typical guitar pickup output" figure,
    not an arbitrary constant) but its provenance has not been verified per-schx against a
    documented source -- treat the exported input_level_dbu as an assumption a host's
    calibration UI can act on, not a measured calibration reference."""
    rms = v0dbfs_volts / math.sqrt(2)
    return 20.0 * math.log10(rms / _DBU_0_RMS_VOLTS)


class ParametricA2(nn.Module):
    """A2 (LeakyReLU, mixed kernels) with FiLM conditioning on knob params.

    Architecture matches the C++ A2FastModel with additional FiLM layers
    for parametric knob control.
    """
    def __init__(self, channels: int, num_params: int, spectral_norm: bool = False):
        super().__init__()
        self.channels = channels
        self.num_params = num_params
        # Recorded so export_nam()'s _bake_spectral_norm() knows whether there's
        # anything to strip -- see docs/film_runaway_investigation.md ("A2").
        self.spectral_norm = False   # set True by enable_spectral_norm() below, if requested
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
        # Applied LAST, after every raw module above already exists -- see A2Layer's docstring
        # for why (spectral_norm's extra RNG draws must not land between other parameters'
        # initialization). Standalone-construction convenience only: SlimmableParametricA2
        # does NOT rely on this (it always passes spectral_norm=False here and calls
        # enable_spectral_norm() itself afterward, once, across ALL tiers -- see its __init__).
        if spectral_norm:
            self.enable_spectral_norm()

    def enable_spectral_norm(self):
        """Wrap every layer's conv/mixin/l1x1 with spectral_norm. Call only after every raw
        module this instance will ever have already exists (see A2Layer.enable_spectral_norm())."""
        for layer in self.layers:
            layer.enable_spectral_norm()
        self.spectral_norm = True

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
        # The unpadded conv over T + (K-1) samples yields exactly T outputs, so the old
        # trailing crop + .contiguous() were a no-op slice plus a full-tensor copy.
        head_in = F.pad(head_in, (K_HEAD_KERNEL - 1, 0))
        x = self.head(head_in)
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

    def _bake_spectral_norm(self) -> "ParametricA2":
        """Build a fresh, plain (non-parametrized) copy of this model with conv/mixin/
        l1x1's CURRENT computed (already spectral-norm-bounded) weights baked in as
        ordinary leaf Parameters -- see docs/film_runaway_investigation.md ("A2"). Must
        run before anything reads `.weight` as a plain leaf tensor: `_export_weight_block()`
        (below), `nam_standard.fold_film()`'s in-place `.mul_()`, and `load_weights()`'s
        reconstruction from exported floats all assume that shape.

        NOT `copy.deepcopy(self)` + `remove_parametrizations(..., leave_parametrized=True)`:
        confirmed empirically that stripping a parametrization off a deep copy of a
        spectral_norm-wrapped Conv1d corrupts the ORIGINAL module too (`is_parametrized`
        still reports True for it, but its next forward() raises `AttributeError: ...
        has no attribute 'weight'` anyway) -- a torch.nn.utils.parametrize internals
        quirk, not something we can rely on. Building fresh and copying computed VALUES
        only ever *reads* the live model (confirmed side-effect-free), so a model
        mid-training keeps working normally (and can still be resumed/optimized) after
        an intermediate export.
        """
        if not self.spectral_norm:
            return self
        baked = ParametricA2(self.channels, self.num_params, spectral_norm=False)
        with torch.no_grad():
            baked.rechannel.weight.copy_(self.rechannel.weight)
            baked.head.weight.copy_(self.head.weight)
            baked.head.bias.copy_(self.head.bias)
            baked.head_scale.copy_(self.head_scale)
            for src, dst in zip(self.layers, baked.layers):
                dst.conv.weight.copy_(src.conv.weight)
                dst.conv.bias.copy_(src.conv.bias)
                dst.mixin.weight.copy_(src.mixin.weight)
                dst.l1x1.weight.copy_(src.l1x1.weight)
                dst.l1x1.bias.copy_(src.l1x1.bias)
                if src.film is not None:
                    dst.film.net.weight.copy_(src.film.net.weight)
                    dst.film.net.bias.copy_(src.film.net.bias)
        return baked

    def export_nam(self, config: dict, metadata: dict, sample_rate: int,
                   input_audio: "np.ndarray | None" = None) -> dict:
        # Rebinding the local `self` (not mutating the instance) so every existing
        # line below transparently reads/exports from the baked copy when this model
        # was trained with spectral_norm=True; a no-op (returns self) otherwise.
        self = self._bake_spectral_norm()
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

        # input_level_dbu: derived from the schx Input V0dBFS (see the module comment near
        # _DBU_0_RMS_VOLTS). Correct only when V0dBFS follows the 1V ecosystem convention
        # (then ~-0.8 dBu). A device captured on real gear may override with a measured value
        # via config["input_level_dbu"]. Omitted if the schx has no readable Input V0dBFS.
        override = config.get("input_level_dbu")
        if override is not None:
            input_level_dbu = float(override)
        else:
            v0dbfs = _schx_input_v0dbfs(config.get("schx"))
            input_level_dbu = _input_level_dbu(v0dbfs) if v0dbfs else None

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
            "input_level_dbu": input_level_dbu,
            "name": config.get("name", config.get("circuit", "")),
            "modeled_by": config.get("modeled_by", ""),
            "gear_make": config.get("gear_make", ""),
            "gear_model": config.get("gear_model", config.get("circuit", "")),
            "gear_type": config.get("gear_type", "amp"),
            "tone_type": config.get("tone_type", ""),
        }
        # Strip empty strings so absent fields don't clutter the file
        nam_metadata = {k: v for k, v in nam_metadata.items() if v != "" and v is not None}
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
    ascending (narrowest first = the 'lite' tier, widest last = the 'full' tier).
    A single width ('5' -> [5]) is allowed: it trains ONE tier, exported as a
    1-submodel container (see SlimmableParametricA2 for how the single tier maps
    to the `full` role). Useful for training an extra width to later splice into
    an existing container with tools/merge_tiers.py."""
    if not spec:
        return list(K_DEFAULT_WIDTHS)
    if isinstance(spec, (list, tuple)):
        vals = list(spec)
    else:
        vals = [int(x) for x in str(spec).split(",") if x.strip()]
    ws = sorted(set(int(w) for w in vals))
    if not ws:
        raise ValueError(f"--widths needs at least one channel count, got {ws}")
    return ws


class SlimmableParametricA2(nn.Module):
    """N independent ParametricA2 models (one per channel width) trained jointly,
    exported as one SlimmableContainer whose submodels are selected at runtime by
    ascending `max_value` breakpoints.

    Tiers are ordered ascending by width: index 0 = narrowest (the 'lite' tier),
    index -1 = widest (the 'full' tier). Default widths [3, 8] reproduce the prior
    2-tier lite/full behavior exactly (max_value 0.5 / 1.0).
    """
    def __init__(self, num_params: int, widths=None, spectral_norm: bool = False):
        super().__init__()
        self.widths = parse_widths(widths)
        self.num_params = num_params
        # spectral_norm is DELIBERATELY NOT passed here (always False per-tier) -- see below.
        kw = dict(spectral_norm=False)
        # Endpoints keep the attribute names `lite`/`full` so state-dict keys stay
        # `lite.*` / `full.*` — a default [3, 8] model is byte-compatible with old
        # 2-tier checkpoints (resume + export_checkpoint keep working). Any middle
        # tiers live in `mid` (empty ModuleList when widths has just 2 entries).
        if len(self.widths) == 1:
            # A single width is not really "slimmable": build ONE submodel under the
            # `full` role (widest == only tier) so it saves as best.pt / `full.` state
            # keys and exports as a 1-submodel container. `lite` is absent.
            self.lite = None
            self.full = ParametricA2(self.widths[0], num_params, **kw)
            self.mid = nn.ModuleList()
        else:
            self.lite = ParametricA2(self.widths[0], num_params, **kw)
            self.full = ParametricA2(self.widths[-1], num_params, **kw)
            self.mid = nn.ModuleList(ParametricA2(w, num_params, **kw) for w in self.widths[1:-1])
        # spectral_norm applied HERE, in one pass across every tier, only after every tier's raw
        # modules already exist -- see A2Layer's docstring (docs/film_runaway_investigation.md,
        # "A2"). Each ParametricA2 above was built with spectral_norm=False specifically so this
        # is the ONLY point any spectral_norm-related RNG draw happens; doing it per-tier during
        # construction (the old behavior) would shift the LATER tiers' raw weight initialization
        # relative to an otherwise-identical unconstrained run with the same seed -- confirmed
        # empirically (2026-08-02): `full.head.weight` differed between two SlimmableParametricA2
        # instances built with identical torch.manual_seed() calls, purely from this ordering.
        if spectral_norm:
            self.enable_spectral_norm()

    def enable_spectral_norm(self):
        """Wrap every tier's conv/mixin/l1x1 with spectral_norm. Call only after every raw
        module across every tier already exists -- see A2Layer.enable_spectral_norm(). Also
        the entry point for the --init-from clip-then-fine-tune retrofit path: apply this
        AFTER loading an old (unconstrained) checkpoint's weights, not before, so the initial
        clip uses the trained weights rather than requiring load_state_dict() to match an
        already-wrapped (and therefore differently-keyed) state dict."""
        for m in self.submodels:
            m.enable_spectral_norm()

    @property
    def submodels(self):
        """All tiers in ascending-width order."""
        if self.lite is None:                 # single-width: only the `full` tier exists
            return [self.full]
        return [self.lite, *self.mid, self.full]

    def forward(self, audio: torch.Tensor, params: torch.Tensor):
        return [m(audio, params) for m in self.submodels]   # ascending by width

    # --- tier addressing -----------------------------------------------------
    def tier_labels(self) -> list[str]:
        """Endpoints keep role names (lite/full) so existing tooling — metrics
        columns, best-checkpoint filenames, release naming — keeps working;
        middle tiers are labeled by width (e.g. 'w4')."""
        n = len(self.widths)
        if n == 1:
            return ["full"]          # single width lives under the `full` role
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
        if n == 1:
            return "full."           # single width lives under the `full` role
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
                 mmap: bool = False):
        self.dir = Path(dataset_dir)
        with open(self.dir / "config.json") as f:
            self.config = json.load(f)
        self.param_names = self.config["knobs"]
        self.config["param_names"] = self.param_names  # alias for export_nam
        self.num_params = len(self.param_names)
        self.crop_len = crop_len
        self.repeats = repeats

        # Load input sweep AT ITS NATIVE LEVEL -- no RMS rescaling.
        #
        # This used to rescale the whole clip (and, in __getitem__, the target by the SAME
        # scalar) so the file's overall RMS hit a fixed -18dBFS. That is only a valid operation
        # for a LINEAR system: circuit_output(k*x) == k*circuit_output(x) is generally FALSE for
        # a nonlinear distortion/clipping circuit, so any k != 1 taught the network a
        # mathematically wrong input/output relationship at the rescaled level -- outputs.npy
        # was rendered from the ORIGINAL unscaled file, not a k-times-louder one.
        #
        # It also silently broke on real-playing input. sweepv5/v6 (synthetic, sweep-tone-
        # dominated, crest factor ~11-12dB, peak-normalized to 0.9 by their own build process)
        # happen to scale DOWN under -18dBFS RMS and stay safe. [redacted]-sweep-v3.wav (real playing,
        # pick-transient crest factor ~21.8dB, native peak already 0.967) scales UP 1.65x under
        # the same formula, pushing its peak to 1.55 -- past 0dBFS in float terms, and (combined
        # with the linear-rescale error above) a plausible root cause of the "unexpected noise,
        # including low frequencies, at loud input levels" reported on both the DS-1 and JCM800
        # retrains: the network trained overwhelmingly around whatever level -18dBFS RMS happened
        # to rescale each file to, with no host-side input calibration to match it back at
        # inference (see the host app's updateInputCalibrationGain -- inert unless the
        # .nam declares an input_level, which export_nam below has never written).
        #
        # k=1 has no such failure mode, for any file, by construction: the input the network
        # trains on is EXACTLY the input that produced the stored target.
        inp_raw, _ = sf.read(str(self.dir / "sweep.wav"))
        if inp_raw.ndim > 1:
            inp_raw = inp_raw.mean(axis=1)
        self.inp = inp_raw.astype(np.float32)

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
            out = out_row[start:start + self.crop_len].astype(np.float32)
        else:
            pad = self.crop_len - sig_len
            inp = np.pad(inp[:sig_len], (0, pad))
            out = np.pad(out_row[:sig_len].astype(np.float32), (0, pad))

        inp_t = torch.from_numpy(inp.copy()).float().unsqueeze(0)
        out_t = torch.from_numpy(out.copy()).float().unsqueeze(0)
        params_t = torch.tensor([params[n] for n in self.param_names]).float()
        return inp_t, out_t, params_t


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

# (fft, hop, win) per resolution — shared by mrstft_loss and ParamLoss._stft_mags.
_MRSTFT_RESOLUTIONS = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]

# Hann windows cached per (length, device). torch.hann_window used to be allocated
# INSIDE every torch.stft call — 12-24 fresh GPU tensors per training step.
_HANN_CACHE: dict = {}


def _hann(win: int, device) -> torch.Tensor:
    key = (win, str(device))
    w = _HANN_CACHE.get(key)
    if w is None:
        w = _HANN_CACHE[key] = torch.hann_window(win, device=device)
    return w


_STFT_MAG_EPS = 1e-8  # matches auraloss's STFTLoss default `eps`


def _stft_mags(x: torch.Tensor) -> list:
    """The three MRSTFT magnitude spectra of [B, 1, T] audio. Clamped to sqrt(eps) at the
    source (matching auraloss's `x_mag = sqrt(clamp(real**2+imag**2, min=eps))`) so a
    log-magnitude loss downstream never sees log(0)."""
    mags = []
    for fft, hop, win in _MRSTFT_RESOLUTIONS:
        s = torch.stft(x.squeeze(1), fft, hop, win, _hann(win, x.device), return_complex=True)
        mags.append(torch.sqrt(torch.clamp(s.real**2 + s.imag**2, min=_STFT_MAG_EPS)))
    return mags


def _mrstft_combine(p_mags: list, t_mags: list) -> torch.Tensor:
    """Spectral convergence + log-magnitude L1 per resolution, averaged across
    resolutions -- matches auraloss's MultiResolutionSTFTLoss DEFAULT weights (w_sc=1.0,
    w_log_mag=1.0, w_lin_mag=0.0 -- the official NAM trainer's own default, vendored
    from auraloss). We used to compute ONLY the raw-linear-magnitude L1 term, i.e.
    exactly the one component auraloss disables by default.

    Both components are scale-invariant, unlike a raw-linear-magnitude L1: spectral
    convergence is a Frobenius-norm RATIO (scale cancels top and bottom); log-magnitude
    L1 is invariant to a uniform gain k on both signals because log(k*x) - log(k*y) =
    log(x) - log(y) (the log(k) terms cancel). That matters here specifically because
    ParamDataset no longer forces every dataset to a shared reference RMS (see
    docs/LESSONS.md #18) -- each device now trains at its own native level, and a
    scale-DEPENDENT loss term would silently carry a different effective weight per
    device depending on how loud that device's own data happens to be. ESR (the other
    term in ParamLoss) was already scale-invariant by construction; this closes the gap
    on MRSTFT so it doesn't undermine that."""
    total = torch.tensor(0.0, device=p_mags[0].device)
    for pm, tm in zip(p_mags, t_mags):
        sc = torch.norm(tm - pm, p="fro") / torch.norm(tm, p="fro")
        log_mag = F.l1_loss(torch.log(pm), torch.log(tm))
        total = total + sc + log_mag
    return total / len(p_mags)


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT loss -- see _mrstft_combine for the actual formula."""
    return _mrstft_combine(_stft_mags(pred), _stft_mags(target))


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

    def precompute(self, target: torch.Tensor) -> dict:
        """Target-side loss terms, computed ONCE per batch and shared across tiers.

        A slimmable step calls the criterion once per tier against the SAME target;
        without this, the target's pre-emphasis and its three STFTs are recomputed
        per tier — up to ~37% of the MRSTFT work duplicated on a 4-tier run. The
        target carries no autograd graph, so the cached tensors are plain constants
        and reusing them across sequential per-tier backward passes is safe."""
        return {"t_pre": pre_emphasis(target, self.pre_emph) if self.kind == "esr" else None,
                "t_mags": _stft_mags(target)}

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                cache: dict = None) -> torch.Tensor:
        if self.kind == "esr":
            p = pre_emphasis(pred, self.pre_emph)
            t = cache["t_pre"] if cache is not None else pre_emphasis(target, self.pre_emph)
            main = esr_per_example(p, t, self.floor)
        else:
            main = F.mse_loss(pred, target)
        t_mags = cache["t_mags"] if cache is not None else _stft_mags(target)
        p_mags = _stft_mags(pred)
        mrstft = sum(F.l1_loss(pm, tm) for pm, tm in zip(p_mags, t_mags)) / len(p_mags)
        return main + self.mrstft_weight * mrstft


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0,
                epoch: int = 0, total_epochs: int = 0, log_interval: int = 10,
                amp_dtype=None, scaler=None, per_tier_clip=False):
    """One epoch. Two deliberate structural choices, both numerically identical to
    the naive form:

    NO PER-STEP HOST SYNC. `total_loss += loss.item()` forced a full MPS queue
    drain every step, making the loop strictly alternate CPU batch-prep / GPU
    compute (with num_workers=0 they are the same thread's turns). The loss is
    accumulated as an on-device tensor and .item() happens only at the
    log_interval print — which sits INSIDE the armed watchdog region, so a wedged
    GPU queue still gets caught: the blocking sync happens under a 45 s timer at
    most log_interval steps after the wedge.

    PER-TIER FORWARD+BACKWARD for slimmable models. Tiers share no weights and
    the joint loss is an unweighted sum, so sum-then-backward and sequential
    per-tier backward produce identical gradients (modulo fp reduction order).
    Sequential frees each tier's activation graph before the next tier's forward
    runs, so peak memory is ~the LARGEST tier instead of the SUM of tiers — the
    sum is what forced 4-tier runs down to batch 32 on a 48 GB M4 Pro (Jetsam,
    docs/multi_width_slimmable_plan.md §12).

    GRAD CLIPPING: --per-tier-clip (default off) switches between one joint
    clip_grad_norm_ call over every tier's parameters combined, vs a separate call
    per tier over just that tier's own parameters. Joint clipping means a tier
    with much larger gradients can dominate the shared norm and set the *other*
    tier's effective step size almost entirely — measured directly on a live
    [5,6]-width TS-9 run: the wider tier's mean grad norm was ~5x the narrower
    tier's (96% of the combined squared norm from 58% of the combined parameter
    count), and the joint clip triggered on every sampled step. NAM's own
    PackedLightningModule has the identical joint-clip shape (no per-submodel
    override found in its source), so this isn't a bug relative to upstream —
    just an untested lever. optimizer.step() still happens once per step either
    way, over whichever clip result each tier's grads end up with."""
    import contextlib
    model.train()
    slimmable = isinstance(model, SlimmableParametricA2)
    can_cache = isinstance(criterion, ParamLoss)

    # --amp: autocast wraps ONLY the model forward. The prediction is cast back to
    # fp32 before the criterion, so ESR/MRSTFT — including the near-silent tails
    # the loss is specifically designed to weight — are always scored in fp32.
    def fwd(m, x, p):
        if amp_dtype is None:
            return m(x, p)
        with torch.autocast(device_type=device, dtype=amp_dtype):
            pred = m(x, p)
        return pred.float()

    def bwd(loss):
        (scaler.scale(loss) if scaler is not None else loss).backward()

    total_loss = torch.zeros((), device=device)
    for step, (inp, out, params) in enumerate(loader):
        _watchdog_arm(f"epoch {epoch} step {step}")
        inp, out, params = inp.to(device), out.to(device), params.to(device)
        optimizer.zero_grad()
        if slimmable:
            cache = criterion.precompute(out) if can_cache else None
            step_loss = torch.zeros((), device=device)
            for m in model.submodels:
                loss = (criterion(fwd(m, inp, params), out, cache) if can_cache
                        else criterion(fwd(m, inp, params), out))
                bwd(loss)
                step_loss += loss.detach()
        else:
            loss = criterion(fwd(model, inp, params), out)
            bwd(loss)
            step_loss = loss.detach()
        if scaler is not None:
            scaler.unscale_(optimizer)   # so clip_grad_norm_ sees true gradients
        if clip_norm > 0:
            if per_tier_clip and slimmable:
                for m in model.submodels:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        total_loss += step_loss
        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"  [{epoch:3d}/{total_epochs}] step {step+1}/{len(loader)}  "
                  f"loss={total_loss.item()/(step+1):.6f}", file=sys.stderr, flush=True)
        _watchdog_disarm()
    return total_loss.item() / len(loader)


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
    can_cache = isinstance(criterion, ParamLoss)
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
                    cache = criterion.precompute(out) if can_cache else None
                    total_loss += sum(criterion(p, out, cache) if can_cache
                                      else criterion(p, out) for p in preds).item()
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
    ap.add_argument("--stale-cycles", type=int, default=3,
                    help="Open-ended mode: stop automatically after this many consecutive "
                         "SGDR cycles in which NO tier minted a new best val ESR — the "
                         "stopping rule docs/training-budget.md specifies, which until now "
                         "was executed by a human watching the log (the 5150 run burned "
                         "~2.5h past its plateau waiting for one). Compared at CYCLE "
                         "granularity, i.e. at matched LR phase, so the cosine-tail "
                         "artifact that killed per-epoch patience (a best always lands "
                         "near each trough) cannot fire. Default 3, not the doc's 2: "
                         "replaying the OD-3 run's metrics.csv showed improvements arrive "
                         "in bursts with 100+-epoch droughts — 2 would have stopped at "
                         "ep 1150 and forfeited a further 16-23%% ESR that landed by 1689. "
                         "Sparse-capture datasets (whose val metric is noisiest) may want "
                         "4+, or 0 to disable (the old manual behavior). The counter "
                         "resets on --resume. (default: %(default)s)")
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
    ap.add_argument("--val-split", type=float, default=0.05,
                    help="Fraction of samples for validation (default: %(default)s). Was 0.1; "
                         "val here does not measure interpolation anyway (the same knob settings "
                         "land in train and val -- docs/architecture.md), so a big split buys "
                         "only noise-reduction on the checkpoint-selection signal, which "
                         "--val-passes already provides: 0.05 x 4 passes scores twice the crops "
                         "of the old 0.1 x 1. The Dumble production runs used 0.02 without "
                         "issue; halving the split also returns ~5%% of the step budget to "
                         "actual training.")
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
    ap.add_argument("--amp", choices=["off", "fp16", "bf16"], default="fp16",
                    help="Mixed-precision TRAINING forward (default: fp16). The model forward "
                         "runs under autocast in half precision; the LOSS is always computed "
                         "in fp32 (the MRSTFT magnitudes of near-silent fading tails are "
                         "exactly what the ESR loss exists to protect), and validate() always "
                         "runs fp32 so val ESR stays comparable across runs and matches the "
                         "exported (fp32) model. fp16 uses GradScaler loss scaling; bf16 "
                         "needs none (fp32 exponent range) but has fewer mantissa bits. "
                         "Made default 2026-07-28 for the throughput win (production-shape "
                         "training measured COMPUTE-bound on MPS, KoT: ~1.7 s/step of conv "
                         "work, TS-9 w4+w8 measured ~2.9x s/step faster than fp32) -- the "
                         "per-device quality A/B (judge on level_band_esr.py bands and "
                         "per_perm_esr.py spread per docs/RETRAINING.md, never the headline "
                         "val ESR) is still worth running per-device, but no longer gates "
                         "opt-in. Pass --amp off to disable.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: %(default)s)")
    ap.add_argument("--widths", type=str, default=None,
                    help="Comma-separated slimmable channel widths, e.g. '3,4,8' "
                         "(default: 3,8). Narrowest = lite tier, widest = full tier; "
                         "middle tiers logged/checkpointed as w<N>.")
    ap.add_argument("--spectral-norm", action="store_true",
                    help="Constrain every A2Layer's conv/mixin/l1x1 to spectral norm <=1 "
                         "(Lipschitz-bounded), closing the unbounded gain-compounding "
                         "mechanism behind the FiLM/LeakyReLU runaway instability -- see "
                         "docs/film_runaway_investigation.md ('A2'). Default off: existing "
                         "training runs are unaffected until opted in and validated "
                         "(measure aggregate + per-perm ESR before/after on a per-model "
                         "basis; this is a real capacity constraint, not free).")
    ap.add_argument("--per-tier-clip", action="store_true",
                    help="Slimmable only: clip_grad_norm_ EACH tier's own parameters "
                         "separately (each to --clip-norm) instead of one joint call over "
                         "every tier's parameters combined. Default (off) matches NAM's own "
                         "PackedLightningModule, which has no per-submodel clip override "
                         "either. Motivation: measured on a live [5,6] TS-9 run, the wider "
                         "tier's gradients had ~5x the norm of the narrower tier's (96%% of "
                         "the combined squared norm from 58%% of the combined parameters) and "
                         "the joint clip triggered on every sampled step -- the narrower "
                         "tier's effective step size was being set almost entirely by the "
                         "wider tier's gradient scale, not its own. Untested whether "
                         "per-tier clipping changes final ESR; this flag exists to test it.")
    ap.add_argument("--clip-norm", type=float, default=1.0,
                    help="Gradient clip norm, joint or per-tier depending on --per-tier-clip "
                         "(default: %(default)s)")
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
    ap.add_argument("--init-from", type=Path, default=None,
                    help="WEIGHTS-ONLY warm start from a checkpoint .pt (best.pt or "
                         "latest.pt): loads the model weights and NOTHING else -- fresh "
                         "optimizer, fresh LR schedule from epoch 1, best-ESR history reset "
                         "so new bests checkpoint immediately. This is the retrain-after-"
                         "dataset-change mode --resume cannot serve: --resume restores the "
                         "old best ESRs, which a re-rendered (different) target would never "
                         "beat, silently suppressing all checkpointing. Use when the circuit "
                         "and knobs are unchanged but the render improved (e.g. the "
                         "oversample=8 fleet re-render): the old solution is a close starting "
                         "point, plausibly cutting steps-to-plateau severalfold. Widths and "
                         "knob count must match the checkpoint. Mutually exclusive with "
                         "--resume. CAVEAT (docs/RETRAINING.md): checkpoints trained under "
                         "the old MSE loss carry the loud-perm bias the ESR loss removed -- "
                         "gate acceptance on level_band_esr.py bands, not headline ESR.")
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

    # --spectral-norm's power-iteration (nn.utils.parametrizations.spectral_norm's
    # _power_method -> F.normalize -> torch.div) hits a real MPS kernel gap under fp16
    # autocast: RuntimeError: Failed to create function state object for:
    # div_true_strided_float_half. Confirmed 2026-08-04 (JCM800 gain-only pipeline run,
    # crashed at the very first training step). This is NOT a general MPS+fp16 problem --
    # --amp fp16 was made the default 2026-07-28 specifically FOR measured MPS throughput
    # gains on ordinary (non-spectral_norm) training, so blanket-forcing --amp off on every
    # MPS run would throw that away for the common case. Auto-correct only the specific
    # combination that actually crashes; --device/--amp explicitly passed by the user still
    # take priority everywhere else.
    if device == "mps" and args.spectral_norm and args.amp == "fp16":
        print("  --spectral-norm + --amp fp16 hits a known MPS kernel gap in spectral_norm's "
              "power iteration (div_true_strided_float_half) -- auto-switching to --amp off "
              "for this run. Pass --amp off explicitly to silence this message.", file=sys.stderr)
        args.amp = "off"

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
    # --init-from + --spectral-norm (the "clip-then-fine-tune" retrofit path, docs/
    # film_runaway_investigation.md "A2"): the checkpoint being warm-started from was trained
    # WITHOUT spectral_norm, so its conv/mixin/l1x1 state_dict keys are plain (`.weight`), not
    # the parametrized form (`.parametrizations.weight.original`/`._u`/`._v`). Building the model
    # already-wrapped here would make the --init-from load below hard-fail on a key mismatch --
    # see the same bug class already fixed in export_checkpoint.py/param_infer.py. Deferred to
    # right after the load succeeds instead (enable_spectral_norm() there), which immediately
    # clips any layer whose spectral norm exceeds 1 using the just-loaded trained weights as the
    # starting point -- not a fresh random init, so fine-tuning under the now-active constraint
    # only needs to ADAPT, not relearn the whole function.
    model = SlimmableParametricA2(num_params, widths=parse_widths(args.widths),
                                  spectral_norm=(args.spectral_norm and args.init_from is None))
    desc = ", ".join(f"w{w}({m.weight_count()}w)"
                     for w, m in zip(model.widths, model.submodels))
    print(f"\nModel: SlimmableParametricA2  [{desc}], {num_params} params", file=sys.stderr)
    model.to(device)

    # --init-from MUST run (including its enable_spectral_norm() call) BEFORE film_params/
    # other_params/optimizer are built below -- not just before the load, which was the
    # original bug's near-miss fix. Reason: nn.Module.named_parameters()'s ENUMERATION ORDER
    # differs between a plain and a spectral_norm-wrapped conv -- plain yields weight-then-bias
    # (nn.Conv1d.__init__'s own registration order, both direct params); wrapped yields
    # bias-then-weight_orig (bias stays a direct param, weight moves into a `parametrizations`
    # submodule, and named_parameters() yields a module's own direct params before recursing
    # into submodules). The optimizer's state dict is POSITION-indexed, not name-indexed, so if
    # the optimizer's param list were captured in PLAIN order (built before wrapping, the
    # original code's order) and a LATER --resume of this same run builds ITS optimizer in
    # WRAPPED order (spectral_norm=True from construction, since --init-from is None on
    # resume), positions silently swap. Confirmed empirically (2026-08-03): the first
    # --resume after a clip-then-fine-tune run crashed inside Adam's exp_avg.lerp_() with a
    # shape mismatch, traced to exactly this. Doing the load+wrap here, before optimizer
    # construction, means the optimizer's param order is captured in its FINAL (wrapped, if
    # requested) form from the very first step -- consistent with what any future --resume of
    # this exact run will also produce.
    if args.init_from is not None:
        if args.resume is not None:
            sys.exit("--init-from and --resume are mutually exclusive: one starts a NEW run "
                     "from old weights, the other continues an old run. Pick one.")
        print(f"Warm start (weights only) from {args.init_from} ...", file=sys.stderr)
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(init_ckpt["model"])
        except RuntimeError as e:
            sys.exit(f"--init-from checkpoint does not fit this model (widths/knob-count "
                     f"mismatch?): {e}")
        src_best = init_ckpt.get("best_esr_by_tier") or {"full": init_ckpt.get("best_esr")}
        print(f"  Loaded weights (source run's best ESR: "
              f"{ {k: round(v, 6) for k, v in src_best.items() if v is not None} }). "
              f"Optimizer, schedule, and best-ESR tracking start FRESH.", file=sys.stderr)
        if args.spectral_norm:
            model.enable_spectral_norm()
            print(f"  Applied --spectral-norm AFTER loading (clip-then-fine-tune retrofit, "
                  f"docs/film_runaway_investigation.md 'A2') -- any conv/mixin/l1x1 layer whose "
                  f"spectral norm exceeded 1 in the loaded weights is now clipped to 1; training "
                  f"from here fine-tunes the whole model to adapt to the constraint, not just "
                  f"the clipped layers.", file=sys.stderr)
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

    amp_dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    scaler = None
    if amp_dtype is torch.float16:
        # fp16 gradients underflow without loss scaling; bf16 does not need it.
        scaler = torch.amp.GradScaler(device)
    if amp_dtype is not None:
        print(f"  AMP: {args.amp} model forward (loss + validation stay fp32"
              f"{', GradScaler on' if scaler else ''})", file=sys.stderr)

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
        # "a" if args.resume assumed --resume always continues the SAME metrics.csv a prior
        # run of THIS checkpoint-dir already wrote a header into -- false when --resume points
        # at a checkpoint from a DIFFERENT run/checkpoint-dir (e.g. warm-starting a fresh
        # open-ended continuation without touching the source run's own checkpoint dir).
        # Confirmed real (2026-08-04): such a run wrote 208 headerless data rows, which
        # release_run.sh's facts.py couldn't match any val_esr_w<N> column against at all. First
        # fix gated needs_header on "does the file already have content" alone -- WRONG, missed
        # that a non-resumed run always opens in "w" (truncate) mode regardless: a crashed
        # first attempt (e.g. the MPS/spectral_norm fp16 kernel gap, elsewhere in this file) can
        # write the header before dying mid-epoch-1, leaving a header-only file; on relaunch
        # (still no --resume) the old check saw "already has content" and skipped rewriting the
        # header, but the "w" open then truncated the file anyway, wiping the header with
        # nothing to replace it -- same missing-header failure, different trigger. A non-resumed
        # run ALWAYS needs a fresh header (mode="w" destroys whatever was there); only a resumed
        # run's pre-existing-content check matters at all.
        needs_header = (not args.resume) or not (log_csv.exists() and log_csv.stat().st_size > 0)
        log_f = open(log_csv, "a" if args.resume else "w", newline="")
        log_w = csv.writer(log_f)
        if needs_header:
            log_w.writerow(["epoch", "train_loss", "val_loss",
                            *[f"val_esr_{wlabel[lbl]}" for lbl in labels], "lr", "elapsed_s"])

    _watchdog_open(ckpt_dir)
    t0 = time.time()
    epoch = start_epoch - 1
    completed = 0
    # SGDR cycle-aware auto-stop bookkeeping (open-ended mode, --stale-cycles).
    cycle_improved = False
    stale_cycles = 0
    auto_stop = False
    while True:
        epoch += 1
        if not open_ended and epoch > args.epochs:
            break
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 clip_norm=args.clip_norm,
                                 epoch=epoch, total_epochs=args.epochs, log_interval=10,
                                 amp_dtype=amp_dtype, scaler=scaler,
                                 per_tier_clip=args.per_tier_clip)
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
        if new_best:
            cycle_improved = True

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

        # SGDR cycle-aware auto-stop: the documented budget rule (docs/training-budget.md:
        # "two consecutive cycles with no new best = the budget"), executed by the loop
        # instead of a human watching the log. CosineAnnealingWarmRestarts wraps T_cur to
        # 0 on the scheduler.step() that completes a cycle, so this fires exactly at each
        # trough — cycle-to-cycle comparisons happen at matched LR phase, which is what
        # makes this rule immune to the cosine-tail artifact that broke per-epoch patience.
        if open_ended and args.stale_cycles > 0 and getattr(scheduler, "T_cur", None) == 0:
            if cycle_improved:
                stale_cycles = 0
            else:
                stale_cycles += 1
            print(f"  [cycle end @ epoch {epoch}] "
                  + ("new best(s) this cycle" if cycle_improved
                     else f"no improvement — {stale_cycles}/{args.stale_cycles} stale cycles"),
                  file=sys.stderr, flush=True)
            cycle_improved = False
            if stale_cycles >= args.stale_cycles:
                print(f"[stop] {args.stale_cycles} consecutive SGDR cycles without a new best "
                      f"on any tier — plateau reached, stopping (disable with --stale-cycles 0).",
                      file=sys.stderr, flush=True)
                auto_stop = True

        # Open-ended: stop gracefully on plateau, STOP file, or SIGINT/SIGTERM.
        if open_ended and (auto_stop or should_stop(ckpt_dir)):
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
                                           dataset.param_names, device)
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
