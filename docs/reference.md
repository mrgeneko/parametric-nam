[← back to README](../README.md)

`.param.nam` file format, architecture details, and NAM ecosystem compatibility for parametric-nam.

# Reference

## `.param.nam` format

A standard NAM JSON with `"SlimmableContainer"` as the outer architecture (already
registered in NeuralAmpModelerCore). Each submodel uses `"ParametricWaveNet"` — a
custom architecture [our fork of NeuralAmpModelerCore](https://github.com/mrgeneko/NeuralAmpModelerCore)'s
factory registers. [`NAMix`](https://github.com/mrgeneko/NAMix) (see "Related Repos") is a
working sample host built against it.

```json
{
  "version": "0.7.0",
  "architecture": "SlimmableContainer",
  "config": {
    "submodels": [
      {
        "max_value": 0.5,
        "model": {
          "architecture": "ParametricWaveNet",
          "config": {
            "layers": 3,
            "head_scale": 0.448,
            "parametric": {
              "type": "film",
              "condition_size": 6,
              "film_layers": ["layer_0", "layer_1", ...],
              "parameters": [
                {"name": "volume",       "min": 0.0, "max": 1.0, "default": 0.5},
                {"name": "treble",       "min": 0.0, "max": 1.0, "default": 0.5},
                {"name": "middle",       "min": 0.0, "max": 1.0, "default": 0.5},
                {"name": "bass",         "min": 0.0, "max": 1.0, "default": 0.5},
                {"name": "mid",          "min": 0.0, "max": 1.0, "default": 0.5},
                {"name": "clean_master", "min": 0.0, "max": 1.0, "default": 0.5}
              ]
            }
          },
          "weights": [...]
        }
      },
      {
        "max_value": 1.0,
        "model": { "architecture": "ParametricWaveNet", "config": {"layers": 8, ...}, "weights": [...] }
      }
    ]
  },
  "sample_rate": 48000
}
```

`config.parametric.parameters` is the authoritative knob list. Array order = UI
presentation order. All values normalized 0.0–1.0. `"layers"` is the **channel count**
(3 or 8), not a layer count — there are always 23 layers. `min`/`max` per knob come from
the dataset `--bounds`, so the host maps the control to the real trained domain.

**Key design point**: `condition_size` is the number of knobs fed to FiLM. Audio flows
through `input_mixin` separately (always 1-dim) — knobs never mix with audio.

## Architecture

A2 is a WaveNet variant: 23-layer dilated causal convnet with LeakyReLU and a Conv1D
head. `ParametricWaveNet` adds FiLM (Feature-wise Linear Modulation) on every layer,
conditioned on the knob vector. FiLM is initialized to identity (gamma=1, beta=0) so
training starts from a well-behaved unconditional baseline.

Key constants (must match NeuralAmpModelerCore's A2 fast path exactly):
- 23 layers: 14 × kernel=6, then 2 × kernel=15, then 7 × kernel=6
- Dilations: repeating [1,3,7,17,41,101,239] with {1,13} degridding insert
- LeakyReLU(0.01) on every layer
- Head: Conv1D kernel=16

Weight layout per layer (for C++ `set_weights_` compatibility):
```
conv.weight → conv.bias → mixin.weight → l1x1.weight → l1x1.bias → film.weight → film.bias
```
Then after all 23 layers: `head.weight → head.bias → head_scale`.

## LoRA-style knob conditioning (removed)

> **STATUS: REMOVED (2026-08-27).** Training was disabled first (`--lora-rank > 0` errors
> unless `PARAMETRIC_NAM_ALLOW_LORA=1` is set), then the C++ that consumed LoRA-tagged models
> (`NAM/lora.h`, both the generic and hand-optimized fast paths, plus their tests) was deleted
> entirely from NeuralAmpModelerCore — nothing dead was left behind. **A `"film+lora"`
> (`schema_version` 2) model can no longer be loaded by any current build of the fork**:
> `require_supported_parametric_model()` rejects it loudly rather than silently misreading it,
> the same fail-loud contract every schema bump here follows. This toolchain's own Python
> tooling (`checkpoint_infer.py`, `export_checkpoint.py`, `nam_standard.fold_lora()`) still fully
> supports loading, exporting, and folding archived LoRA checkpoints — **`fold_lora()` bakes
> the LoRA delta into an ordinary static model at one fixed knob setting**, which then loads
> and plays in any plugin, LoRA support or not. What's gone is *real-time, knob-live* LoRA
> inference in a plugin host (NAMix, Chainsmith) — not the ability to read old checkpoints.

**Why it was tried.** FiLM only ever applies `gamma(cond)*x + beta(cond)` — a diagonal
per-channel affine rescale of one fixed backbone computation. As swept knob count grows
past ~3, the space of tonal behaviors the backbone must represent as "one shared
computation, rescaled per knob setting" outgrows what fixed weights can encode — measured
concretely on one 5-knob amp run, where the lite (4ch) tier plateaued at median ESR 0.126
on the full grid (not shippable) while the architecturally-identical full (8ch) tier
reached 0.025. `--lora-rank` added a genuine per-knob weight *delta* —
`W_effective = W_l1x1 + A(cond)@B(cond)` — more expressive than FiLM's fixed-computation
rescale, without a full hypernetwork's unconstrained-weights problem.

**Why it was dropped.** On the one device (a different, 2-knob amp) where both were trained
to convergence, same knobs, same grid, same excitation, FiLM-only won:

| | tiers | final ESR lite / full |
|---|---|---|
| FiLM+LoRA | 4ch + 8ch | 0.06186 / 0.02403 |
| FiLM only | 5ch + 9ch | **0.06016 / 0.02191** |

Widening a tier lifts the same capacity ceiling LoRA targets, and costs no schema bump, no
`rank` to choose, and no minimum core version. LoRA also made the FiLM/LeakyReLU runaway
instability worse when it occurred (~7x excitation elevation vs ~1.1x for FiLM-only), a
known bad interaction that was never fully bounded. The earlier 5-knob-amp testbed numbers
(FiLM+LoRA cutting ESR ~65-72% vs FiLM-only) were real but never re-run as a clean,
width-matched, to-convergence ablation, and are not archived for re-checking — the 2-knob
comparison above is the only real head-to-head that exists.

**Archived models**: `parametric-nam-models` has 7 LoRA `.nam` files from two runs on two
devices (both rank 4), neither marked `CURRENT` for its device. Both remain
readable/exportable/foldable via this toolchain's own Python tools as noted above, but are
no longer usable live in a plugin.

## NAM ecosystem compatibility

Verified against `sdatkinson/neural-amp-modeler` + `NeuralAmpModelerCore` (2026-07):

- **Our A2 *is* NAM's official "A2" config** — `K_KERNEL_SIZES`/`K_DILATIONS`,
  LeakyReLU/non-gated, `condition_size=1`, widths `[3,8]` all match NAM's
  `config_model_packed.json`. A **static** (no-FiLM) A2 is literally a standard NAM
  model → exports as `"WaveNet"` / `"SlimmableContainer"` and loads in the stock
  plugin.
- **`"ParametricWaveNet"` is our custom extension** (A2 + FiLM for knobs) and needs our
  fork of NeuralAmpModelerCore, which registers it. **Standard NAM plugins cannot drive user knobs**: the core's
  public API is `process(input, output, num_frames)` — audio in/out, *no* parameter
  argument, and no `SetParam`/`SetCondition` anywhere. NAM's own FiLM/`condition_dsp`
  is *input-derived* architecture, not a knob interface. So re-serializing parametric
  models "to match NAM" would **not** make them plugin-controllable.
- **Delivering parametric tones to standard plugins = snapshot baking.** FiLM is
  affine over the layer's linear ops, so freezing a knob setting folds `γ,β` exactly
  into `conv.weight/bias` + `mixin.weight`, yielding an identical *static* A2 with no
  FiLM. A parametric-aware host (`NAMix`) keeps live knobs;
  `bake_nam.py` "exports this tone" into a stock-standard `.nam` for hosts that don't.
  Works on **already-trained** parametric `.nam`s offline (no retrain) — pure weight
  transform.

---
