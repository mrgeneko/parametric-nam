# spice-to-nam

Toolchain for converting SPICE circuit schematics (`.schx`) into **parametric**
Neural Amp Modeler (NAM) files. The resulting `.param.nam` captures a circuit's
behavior across its **full knob range** — enabling real-time parametric inference
in [redacted] at a fraction of the CPU cost of live simulation.

Given a `.schx` (e.g. an amp or pedal from `LiveSPICE-Amp-Collection`), it simulates
the circuit across a sweep of knob settings, trains a FiLM-conditioned WaveNet on the
paired audio, and exports a single `.param.nam` whose knobs match the real controls.

---

## Quick Start

```bash
# 1. Clone with submodules (LiveSPICE nests ComputerAlgebra — recursive is required)
git clone --recurse-submodules <this-repo> && cd spice-to-nam
git submodule update --init --recursive extern/LiveSPICE   # if you forgot --recurse

# 2. Build the simulator CLI (Linux shown; macOS: -r osx-arm64)
cd livespice_cli && dotnet publish -c Release -r linux-x64 --self-contained -o publish/ && cd ..

# 3. Python deps
python -m venv .venv && . .venv/bin/activate
pip install torch numpy scipy soundfile auraloss

# 4. One command: generate dataset -> combine -> train -> release folder
python run_pipeline.py \
    --dataset-dir    /tmp/ts9_ds \
    --nam-output     /tmp/ts9.param.nam \
    --checkpoint-dir /tmp/ts9_ckpt \
    --backend livespice \
    --schx "/path/to/Ibanez TS-9.schx" \
    --knobs drive,tone --range "drive=0.0,0.25,0.5,0.75,1.0" --range "tone=0.0,0.25,0.5,0.75,1.0" \
    --fixed-params "Level=1.0" \
    --input /path/to/guitar.wav \
    --slimmable --mmap --epochs 200 --crop-len 24000 --batch-size 64

# 5. Listen (Python inference, no C++ needed)
python param_infer.py --checkpoint /tmp/ts9_ckpt/best.pt \
    --input /path/to/dry.wav --output-dir /tmp/out/ --params "drive=0.7,tone=0.4"
```

The trained model lands at `/tmp/ts9.param.nam`; a self-contained **release folder**
(`/tmp/ts9_release/`) with the model, schematic copy, metrics, manifest, and a
`reproduce.sh` is built automatically. See [full build details](#build--dependencies)
for prerequisites (.NET SDK, recursive submodules).

## How it works

```
.schx schematic
    ↓  batch_harness.py        (simulate: input sweep × N knob permutations)
Paired audio dataset (config.json, sweep.wav, outputs.npy, params.csv)
    ↓  param_train.py          (train A2 Lite 3ch + Full 8ch jointly, FiLM knob conditioning)
.param.nam  (SlimmableContainer)
    ↓  param_infer.py          (PyTorch inference at arbitrary knob positions — no C++)
    ↓  Phase 3                 (NeuralAmpModelerCore ParametricWaveNet factory)
Real-time inference in [redacted]
```

`run_pipeline.py` orchestrates the first three steps in one command; you can also run
each script directly (below).

---

## `run_pipeline.py` — the primary entry point

Runs **generate → combine → train** as one command, with per-step timing, a durable
**release folder**, and macOS/Linux **sleep-inhibition** for long runs. Steps auto-skip
when their outputs exist (`--skip-generate/-combine/-train`, `--force-generate`).

```bash
python run_pipeline.py \
    --dataset-dir <ds> --nam-output <model.param.nam> --checkpoint-dir <ckpt> \
    --backend livespice --schx "<amp>.schx" \
    --knobs gain,tone,volume --random 200 --bounds tone=0.15,0.85 \
    --input guitar.wav --slimmable --mmap \
    --crop-len 24000 --epochs 0 --restart-period 50 --batch-size 64
```

- `--epochs 0` = **open-ended training** (see param_train) — runs until you
  `touch <ckpt>/STOP`, exporting the best model continuously.
- On completion it builds `<nam-output>_release/`: the best-full & best-lite
  `.param.nam`, the `.schx`, `metrics.csv`, a provenance `MANIFEST.md` (params +
  timing + hardware + git revs), and a runnable `reproduce.sh`. `--no-release` skips it.
- Generation flags (`--random`, `--bounds`, `--oversample`, ngspice `--koren` etc.)
  are forwarded to `batch_harness`; training flags to `param_train`.

## Scripts

### `batch_harness.py` — generate the dataset

Simulates the circuit across knob permutations, one WAV per permutation, then combines
them into `outputs.npy`.

```bash
# List the pots/switches in a schx:
python batch_harness.py --backend livespice --schx "<circuit>.schx"

# Grid sweep (factorial — fine for 1–3 knobs):
python batch_harness.py --backend livespice --schx "<circuit>.schx" \
    --knobs drive,tone --range "drive=0,0.5,1" --range "tone=0.2,0.5,0.8" \
    --input guitar.wav --output <ds> --dry-run   # check perm count + disk first

# Random sampling (for high knob counts — grids explode past ~3 knobs):
python batch_harness.py --backend livespice --schx "<amp>.schx" \
    --knobs gain,bass,mid,treble,master --random 200 \
    --bounds bass=0.15,0.85 --bounds mid=0.15,0.85 --bounds treble=0.15,0.85 \
    --input guitar.wav --output <ds>

# Combine per-permutation WAVs into outputs.npy:
python batch_harness.py --combine <ds>
```

- **`--random N`** samples N points in the knob hypercube (+ deterministic boundary
  **anchors** so extremes are covered; `--no-anchors` to skip). FiLM interpolates
  between them — far better than a coarse grid for >3 knobs. Seeded (`--seed`) and
  extensible.
- **`--bounds KNOB=lo,hi`** restricts a knob's sampled range (e.g. keep a tone pot in
  its usable band, or a hot gain pot out of a divergent regime). Recorded in the
  `.param.nam` so the host maps the knob to the real trained domain.
- **`--max-crest`** (default 50) fails any permutation whose output crest factor
  (peak/RMS) exceeds it — a scale-invariant detector for numerical divergence that
  RMS/length checks miss.
- **`--fixed-params k=v,...`** pins controls not being swept; **`--speaker`** selects a
  speaker tap. Knob order in `--knobs` = knob order in the `.param.nam`.

**Ganged (multi-section) pots:** `livespice_cli` drives **every** pot/switch whose
`Name` matches the swept knob. To model a mechanically-ganged control (e.g. a dual-gang
Gain), give both `Circuit.Potentiometer` components the **same `Name`** in the `.schx`;
`--knobs Gain` then moves both wipers in lockstep, and the knob appears once downstream.

### `param_train.py` — train a parametric NAM

```bash
python param_train.py --dataset <ds> --output <model.param.nam> --checkpoint-dir <ckpt> \
    --slimmable --mmap --crop-len 24000 --batch-size 64 --repeats 8 \
    --lr 3e-4 --epochs 200
```

- **`--slimmable`** (always use it) trains an A2 **Lite 3ch + Full 8ch** jointly and
  exports both tiers in one SlimmableContainer.
- **Dual best-checkpointing**: the best-full and best-lite epochs (which differ) are
  each saved and exported live to `<output>.best_full.param.nam` /
  `<output>.best_lite.param.nam`, plus `best.pt` / `best_lite.pt` and per-epoch
  `latest.pt` + `metrics.csv`.
- **`--epochs 0`** = open-ended: trains with an SGDR (cosine-warm-restart) schedule
  until you `touch <ckpt>/STOP` (or SIGINT/SIGTERM). Best models export continuously,
  so you can stop whenever ESR is good enough. `--restart-period` sets the cycle length.
- **`--resume <ckpt>/latest.pt`** continues a run. **`--mmap`** memory-maps
  `outputs.npy` (low RAM). **`--crop-len`** is the training window; 24000 is ≫ the
  model's receptive field and ~2× faster than 48000.

### `param_infer.py` — inference (Python, no C++)

```bash
python param_infer.py --checkpoint <ckpt>/best.pt \
    --input dry.wav --output-dir out/ \
    --params "drive=0.5,tone=0.7" --params "drive=0.9,tone=0.3"
```
Loads a checkpoint and renders WAVs at arbitrary knob positions. Repeat `--params` for
multiple outputs. Knob names/order come from the dataset `config.json`.

### `coverage_report.py` — find holes in a sampled dataset

```bash
python coverage_report.py --dataset <ds> --suggest 5
```
Read-only analysis of `params.csv`: success/failure summary, **per-knob boundary
coverage**, 1-D histograms, pairwise 2-D occupancy, and nearest-neighbour **gap
analysis** (largest empty regions, optionally emitted as fill points). Pairs with the
seed-extensible sampler to grow a dataset toward its gaps.

### `export_checkpoint.py` — checkpoint → `.param.nam`

```bash
python export_checkpoint.py --checkpoint <ckpt>/latest.pt --dataset <ds> \
    --output model_final.param.nam --state model
```
Re-exports a NAM from any `.pt` (e.g. the final `latest.pt` weights) without retraining.

---

## Backends

**LiveSPICE** (`--backend livespice`, default) — real-time-capable, `.schx`-native,
reference tube models, uniform audio-rate output, never aborts. Use it for essentially
everything.

**ngspice** (`--backend ngspice`, experimental) — an offline real-SPICE backend with
adaptive timestepping, for the handful of **stiff / very-high-gain** circuits (e.g. the
EVH 5150 Lead full) where LiveSPICE's fixed-step solver diverges or needs extreme
oversampling. It translates any `.schx` to an ngspice netlist (exact Dempwolf–Zölzer /
Koren tube models, E+F ideal transformer) and feeds the input via an XSPICE filesource.
Tuning knobs for stiff amps: `--koren`, `--ot-damp`, `--ot-snub`, `--nfb-comp`.
See **[`ngspice/README.md`](ngspice/README.md)** for usage, the convergence findings,
and the important fidelity caveats.

## What a run produces

A **dataset** directory (`batch_harness --combine` format):
```
config.json   # {"knobs": [...], "bounds": {...}, "param_map": {...}, ...}
sweep.wav     # the dry input used
outputs.npy   # [N_perms, N_samples] float32
params.csv    # idx, knob1, ..., rms, peak, ok, error
```

A **release folder** (`<nam-output>_release/`, built by `run_pipeline`): the shipped
`.best_full.param.nam` + `.best_lite.param.nam`, the `.schx`, `metrics.csv`,
`MANIFEST.md` (provenance: params, per-step timing, hardware, git revisions), and a
runnable `reproduce.sh`.

---

## Reference

### `.param.nam` format

A standard NAM JSON with `"SlimmableContainer"` as the outer architecture (already
registered in NeuralAmpModelerCore). Each submodel uses `"ParametricWaveNet"` — a
custom architecture Phase 3 registers.

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

### Architecture

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

### NAM ecosystem compatibility

- **`"SlimmableContainer"`** — standard, already registered in NeuralAmpModelerCore ✓
- **`"ParametricWaveNet"`** — custom; requires Phase 3 factory registration
- Official NAM tools deliberately removed user-controllable knob parameters in
  2023–2024; parametric inference requires custom C++ work
- Old parametric C++ code recoverable from NAM git history at commit `ae86979`
  (before PR #367 removed it)

---

## Build & Dependencies

### Python packages
```bash
pip install torch numpy scipy soundfile auraloss
```

### .NET SDK + LiveSPICE (for dataset generation)

`livespice_cli` builds against LiveSPICE, vendored as the `extern/LiveSPICE` submodule.
**Recursive submodule init is required** — LiveSPICE nests a `ComputerAlgebra` submodule
that `Circuit.csproj` references; without it the build fails with `CS0246` errors
(`Expression`, `Arrow`, `Variable`, `SolutionSet` not found).

```bash
git submodule update --init --recursive extern/LiveSPICE   # recursive is mandatory

cd livespice_cli
# Linux:
dotnet publish -c Release -r linux-x64  --self-contained -o publish/
# macOS (Apple Silicon):
dotnet publish -c Release -r osx-arm64  --self-contained -o publish/
```
(.NET 10 SDK: `apt install dotnet-sdk-10.0` on Linux, `brew install dotnet` on macOS.)
The binary lands at `livespice_cli/publish/livespice_cli`, where `batch_harness.py`
expects it.

### ngspice (only for `--backend ngspice`)
```bash
apt install ngspice        # Linux;  macOS: brew install ngspice
```

### NeuralAmpModelerCore (Phase 3 only)
Not required for training or Python inference — only for C++ inference validation and
[redacted] integration.

> **Input formats**: any WAV (16/24/32-bit) and any content read correctly — an earlier
> O(N²) bug in `livespice_cli`'s WAV reader (fixed in `fa90d57`) once made non-24-bit
> inputs stall. Real guitar DI is recommended over synthetic sweeps.

## Related Repos

- `LiveSPICE-Amp-Collection` — `.schx` circuit library
- `[redacted]` — target host for the trained `.param.nam` models

## Credits & Attribution

This toolchain builds on several open-source projects and published models:

- **[LiveSPICE](https://github.com/dsharlet/LiveSPICE)** (Dillon Sharlet, MIT) — the
  circuit simulator `livespice_cli` builds against (vendored as `extern/LiveSPICE`),
  and the reference for the tube-model equations. The ngspice backend's tube
  subcircuits are **ported from LiveSPICE's `Triode.cs` / `Pentode.cs`**.
- **[ngspice](https://ngspice.sourceforge.io/)** (BSD) — the adaptive-timestep SPICE
  engine used by the experimental `--backend ngspice`, including the **XSPICE** code
  models (originally Georgia Tech Research Institute) whose `filesource` feeds the input.
- **Tube models** (as implemented by LiveSPICE and ported here):
  - **Norman Koren**'s SPICE triode/pentode model — the `--koren` mode and all pentodes.
  - **Dempwolf–Zölzer** triode model — K. Dempwolf & U. Zölzer, *"A physically-motivated
    triode model for circuit simulations,"* Proc. DAFx-11 (2011) — the default triode model.
- **[Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)** /
  **[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)**
  (Steven Atkinson, MIT) — the `.nam` format, the WaveNet ("A2") architecture, and the
  `SlimmableContainer` that this project targets.
- **[auraloss](https://github.com/csteinmetz1/auraloss)** (Christian Steinmetz, Apache-2.0)
  — the multi-resolution STFT loss in `param_train.py`.
- **Ideal-transformer technique** — the ngspice output transformer uses the standard
  controlled-source (E+F) ideal-transformer method; the specific center-tapped
  equations are ported from LiveSPICE (above).

Underlying scientific-Python stack: PyTorch, NumPy, SciPy, soundfile.
