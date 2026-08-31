[← back to README](../README.md)

`export_checkpoint.py`, `bake_nam.py`, and `merge_tiers.py` sound like they could overlap —
they don't. Each does one distinct, non-optional transform on the way from a training
checkpoint to a published `.nam`. This maps the flow. See [`docs/scripts.md`](scripts.md) for
each tool's full flag reference.

# NAM lifecycle

## The main path: parametric training → published bundle

```
param_train.py
      │  writes checkpoints/{latest,best,best_<tier>}.pt + metrics.csv
      ▼
export_checkpoint.py                    (checkpoint(s) → .param.nam, no retrain)
      │  single mode:   one checkpoint, one .param.nam
      │  --compose:     splice each tier's OWN best checkpoint into one container
      │                 (tiers share no weights, so this is pure JSON surgery)
      ▼
merge_tiers.py  [optional]              (.param.nam + .param.nam → wider .param.nam)
      │  add a separately-trained tier (e.g. a later w5 run) to an existing
      │  container -- same "no shared weights" fact export_checkpoint.py's
      │  --compose relies on, just across files instead of checkpoints
      ▼
bake_nam.py  [optional, terminal]       (.param.nam → standard NAM .nam)
      │  freezes ONE knob setting into an ordinary WaveNet .nam for stock/
      │  upstream NAM plugins, which have no runtime knob input at all --
      │  this is a dead end, not a step back into the parametric path
      ▼
release_run.sh                          (verify + stage + package)
      validates NAM version / head_mode, composes the release container,
      runs plot_tone_response.py for the fidelity chart, stages the bundle
```

| Tool | Input | Output | When |
|---|---|---|---|
| `export_checkpoint.py` | one or more `.pt` checkpoints | `.param.nam` | Always, first — nothing downstream reads a checkpoint directly |
| `merge_tiers.py` | 2+ existing `.param.nam` files | one wider `.param.nam` | Only if a tier was trained in a separate run after the fact |
| `bake_nam.py` | one `.param.nam` + a fixed `--params` setting | standard (non-parametric) `.nam` | Only when targeting a plugin that can't drive knobs live — an alternate output format, not a pipeline stage everything passes through |
| `release_run.sh` | a finished run's checkpoints/dataset | staged, verified bundle | Before publishing anywhere (`parametric-nam-models`-style archive or your own) |

`bake_nam.py` is a **leaf**: nothing in this repo re-parametrizes a baked `.nam`. If you need
both a live-knob model and fixed presets, keep the `.param.nam` as the source of truth and bake
presets from it on demand — don't try to go the other direction.

## Reading a model or checkpoint without transforming it

These don't produce a new artifact — they inspect or validate one that already exists.

| Tool | Reads | Answers |
|---|---|---|
| `checkpoint_infer.py` | a `.pt` checkpoint | Run inference straight from a checkpoint, no export step — multiple `--params` sets per invocation |
| `nam_infer.py` | an exported `.param.nam` | Run inference (or `--sweep`) from a published file, no checkpoint needed |
| `per_combo_esr.py` | a checkpoint + dataset | Which knob-grid regions fit well vs. poorly (also called automatically by `param_train.py` itself — see [`docs/checklist.md`](checklist.md)) |
| `level_band_esr.py` | a checkpoint + dataset | Fade-out/quiet-tail accuracy, split by output level band |
| `scan_film_runaway.py` | a published `.param.nam` | FiLM/LeakyReLU blow-up at rare knob corners — see [`docs/checklist.md`](checklist.md) |
| `plot_tone_response.py` | a published `.param.nam` | Frequency-response chart, optionally against the real circuit |
| `ab_realtime_playback.py` | a published `.param.nam` | A/B the Python forward pass against the host app's C++ playback (FiLM parity) |

## The other path entirely: static captures

`capture_static.py` is **not** a variant of the flow above — it trains a completely different
kind of model, via the official upstream `neural-amp-modeler` trainer (`nam-full`), not this
repo's own FiLM-conditioned `param_train.py`. The output is a standard static NAM tied to one
fixed knob/switch setting from the start, not something `export_checkpoint.py`/`bake_nam.py`
ever touches. Reach for it when you want a single-setting model trained the same way the rest
of the NAM ecosystem trains one, not a knob-driven `.param.nam`.

Don't confuse it with `capture_common.py`, despite the similar name: `capture_common.py` is a
shared filename-parsing helper library used only by `gen_dataset_from_captures.py` (building a
**parametric** training dataset from a set of real per-setting captures, still headed for
`param_train.py` like a `.schx`-rendered dataset would be). `capture_static.py` does not import
it and shares no code with it — the naming collision is coincidental, not a relationship.
