#!/usr/bin/env python3
"""Serialize A2 models to the OFFICIAL Neural Amp Modeler .nam format.

Our "A2" is NAM's standard config (kernel sizes / dilations / LeakyReLU-non-gated /
condition_size=1 / widths all match `nam/train/_resources/config_model_packed.json`).
A **static** (no-FiLM) A2 is therefore literally a standard NAM WaveNet and exports as
a stock-loadable ``"WaveNet"`` .nam.

A **parametric** A2 (A2 + FiLM knob-conditioning) is host-app-internal — standard NAM
plugins have no runtime knob input. To deliver a specific tone to a standard plugin we
**bake** the knob setting: FiLM is affine (``gamma*h + beta``) over the layer's linear
ops, so freezing the knobs folds gamma/beta *exactly* into ``conv.weight/bias`` and
``mixin.weight`` — yielding an identical static A2. No retrain; works offline on an
already-trained model. See docs/spice_static_plan.md.
"""
from __future__ import annotations
import copy
import datetime

import torch

from param_train import (ParametricA2, SlimmableParametricA2,
                         K_KERNEL_SIZES, K_DILATIONS, K_HEAD_KERNEL,
                         K_NUM_LAYERS, K_LEAKY_SLOPE)

# .nam FILE-FORMAT version (NAM's namespace, not ours — our parametric block versions itself
# separately via config.parametric.schema_version).
#
# Must be "0.7.0", not an older value:
#   - the host app gates imports on it (its own version-compatibility check): anything
#     below 0.7.0 is classified A1 and REJECTED ("A1 NAM models are not supported"). A baked
#     file stamped 0.5.4 was bounced before its embedded parametric payload was even read.
#   - 0.7.0 is LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION in every core we checked (our own
#     fork, upstream, and the one NeuralAmpModelerPlugin ships), whose supported range is
#     0.5.0–0.7.0. So 0.7.0 is fully supported by stock plugins too — it is the one value that
#     satisfies both ends.
# param_train's parametric export already emits 0.7.0; this keeps the baked export consistent.
NAM_VERSION = "0.7.0"


def fold_film(model: ParametricA2, params) -> ParametricA2:
    """Deep-copy `model`, fold its FiLM at `params`, and remove it → a static A2 whose
    output equals the parametric model at that knob setting.

    `params` is an ordered vector/list of length model.num_params (caller maps knob
    names → order). A model that is already static (num_params==0) is returned as-is.
    """
    m = copy.deepcopy(model).eval()
    if m.num_params == 0:
        return m
    cond = torch.as_tensor(params, dtype=torch.float32).view(1, -1)
    if cond.shape[1] != m.num_params:
        raise ValueError(f"expected {m.num_params} params, got {cond.shape[1]}")
    with torch.no_grad():
        for layer in m.layers:
            if layer.film is None:
                continue
            # Via FiLM.gamma_beta(), NOT a reimplemented affine step -- this used to
            # independently recompute gamma/beta inline here, duplicating FiLM.forward()'s
            # formula. That divergence is exactly what let an unbounded-gamma investigation
            # (docs/film_runaway_investigation.md) find gamma bounded in training but NOT
            # in what actually got baked and shipped. One shared implementation now.
            gamma, beta = layer.film.gamma_beta(cond)
            gamma, beta = gamma.squeeze(0), beta.squeeze(0)   # [C], [C]
            g = gamma.view(-1, 1, 1)
            # h = conv(x) + mixin(inp);  film(h) = gamma*h + beta
            layer.conv.weight.mul_(g)
            layer.conv.bias.mul_(gamma).add_(beta)
            layer.mixin.weight.mul_(g)
            layer.film = None
    m.num_params = 0
    return m


def _wavenet_config(channels: int, head_scale: float) -> dict:
    """Official NAM WaveNet `config` for our A2 at a given width — mirrors exactly
    what NAM's `WaveNet.export_config()` emits for the A2 config (verified in-venv)."""
    n = K_NUM_LAYERS
    film_off = lambda: {"active": False, "shift": True, "groups": 1}
    layer = {
        "input_size": 1,
        "condition_size": 1,
        "head": {"out_channels": 1, "kernel_size": K_HEAD_KERNEL, "bias": True},
        "channels": channels,
        "kernel_sizes": list(K_KERNEL_SIZES),
        "dilations": list(K_DILATIONS),
        "activation": [{"type": "LeakyReLU", "negative_slope": K_LEAKY_SLOPE}
                       for _ in range(n)],
        "bottleneck": channels,
        "head1x1": {"active": False, "out_channels": 1, "groups": 1},
        "layer1x1": {"active": True, "groups": 1},
        "groups_input": 1,
        "groups_input_mixin": 1,
        "conv_pre_film": film_off(), "conv_post_film": film_off(),
        "input_mixin_pre_film": film_off(), "input_mixin_post_film": film_off(),
        "activation_pre_film": film_off(), "activation_post_film": film_off(),
        "layer1x1_post_film": film_off(), "head1x1_post_film": film_off(),
        "gating_mode": ["none"] * n,
        "secondary_activation": [None] * n,
        "slimmable": None,
    }
    return {"layers": [layer], "head": None, "head_scale": head_scale}


def _wavenet_dict(a2: ParametricA2) -> dict:
    """architecture/config/weights for a STATIC A2 (num_params must be 0)."""
    if a2.num_params != 0:
        raise ValueError("fold FiLM first (use fold_film / pass params=)")
    return {
        "architecture": "WaveNet",
        "config": _wavenet_config(a2.channels, a2.head_scale.item()),
        "weights": a2.export_weights(),   # NAM weight order; FiLM-free since static
    }


def _metadata(meta: dict | None) -> dict:
    n = datetime.datetime.now()
    meta = dict(meta or {})
    return {
        "date": {"year": n.year, "month": n.month, "day": n.day,
                 "hour": n.hour, "minute": n.minute, "second": n.second},
        "name": meta.get("name", ""),
        "modeled_by": meta.get("modeled_by", ""),
        "gear_make": meta.get("gear_make", ""),
        "gear_model": meta.get("gear_model", ""),
        "gear_type": meta.get("gear_type", "amp"),
        "tone_type": meta.get("tone_type", ""),
    }


def _resolve(a2: ParametricA2, params) -> ParametricA2:
    if a2.num_params == 0:
        return a2
    if params is None:
        raise ValueError("model is parametric — pass params=<knob vector> to bake a setting")
    return fold_film(a2, params)


def export_nam_standard(model, params=None, *, width: int | None = None,
                        sample_rate: int = 48000, metadata: dict | None = None) -> dict:
    """Official NAM .nam dict for `model` (ParametricA2 or SlimmableParametricA2).

    params : None if already static; else an ordered knob vector to bake FiLM at.
    width  : for a slimmable model, pick one tier → single "WaveNet" (default: widest).
             Pass width=0 / a sentinel to emit the full multi-tier "SlimmableContainer".
    """
    md = _metadata(metadata)

    if isinstance(model, SlimmableParametricA2):
        if width != "all":
            # default: deliver ONE tier as a plain WaveNet (the tone-delivery case)
            target = model.widths[-1] if width is None else width
            a2 = next((m for m in model.submodels if m.channels == target), None)
            if a2 is None:
                raise ValueError(f"width {target} not in {model.widths}")
            wn = _wavenet_dict(_resolve(a2, params))
            return {"version": NAM_VERSION, **wn, "sample_rate": sample_rate, "metadata": md}
        # multi-tier SlimmableContainer (each tier baked at the same setting)
        subs, n = [], len(model.submodels)
        for i, a2 in enumerate(model.submodels):
            subs.append({"max_value": 1.0 if i == n - 1 else round((i + 1) / n, 6),
                         "model": _wavenet_dict(_resolve(a2, params))})
        return {"version": NAM_VERSION, "architecture": "SlimmableContainer",
                "config": {"submodels": subs}, "weights": [],
                "sample_rate": sample_rate, "metadata": md}

    wn = _wavenet_dict(_resolve(model, params))
    return {"version": NAM_VERSION, **wn, "sample_rate": sample_rate, "metadata": md}
