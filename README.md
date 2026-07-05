# spice-to-nam

Toolchain for converting SPICE circuit schematics (`.schx`) into parametric Neural Amp Modeler (NAM) files. The resulting `.param.nam` file captures a circuit's behavior across its full knob range — enabling real-time parametric inference in [redacted] at a fraction of the CPU cost of live simulation.

## Overview

```
.schx schematic
    ↓  batch_harness.py
Paired audio dataset: input sweep × N knob permutations
    ↓  param_train.py
.param.nam  (SlimmableContainer: A2 Lite 3ch + Full 8ch, FiLM knob conditioning)
    ↓  param_infer.py
Output WAVs at arbitrary knob positions (Python/PyTorch, no C++ required)
    ↓  Phase 3: NeuralAmpModelerCore ParametricWaveNet factory
Real-time inference in [redacted]
```

## Scripts

### `param_train.py` — Train a parametric NAM

```bash
# Slimmable (trains A2 Lite 3ch + Full 8ch jointly — always use this):
caffeinate -i python param_train.py \
    --dataset /path/to/dataset \
    --output /path/to/model.param.nam \
    --slimmable \
    --epochs 500 --batch-size 64 --repeats 50 \
    --lr 3e-4 --crop-len 48000 \
    --checkpoint-dir /path/to/checkpoints
```

Resume from a checkpoint:
```bash
caffeinate -i python param_train.py ... --resume /path/to/checkpoints/latest.pt
```

The dataset directory must contain the format written by `batch_harness.py --combine`:
```
dataset/
    config.json      # {"knobs": ["knob1", "knob2", ...], ...}
    sweep.wav        # dry input audio (limelight.wav or similar real guitar)
    outputs.npy      # [N_perms, N_samples] float32 array
    params.csv       # idx, knob1, knob2, ..., ok, error
```

**Note on input formats**: Earlier guidance claimed LiveSPICE "hangs" on sine
sweeps / broadband signals. That was a misdiagnosis — the real cause was an
O(N²) bug in `livespice_cli`'s WAV reader that stalled on 16-bit and 32-bit-float
inputs (24-bit PCM took a different code path and was unaffected). Fixed in
commit `fa90d57`. Any WAV format (16/24/32-bit) and any content — including sine
sweeps — now reads correctly. `limelight.wav` is still a fine choice, but it is
no longer a constraint.

Checkpoints written every epoch to `latest.pt`; best ESR checkpoint to `best.pt`; best weights exported immediately to `<output_stem>_best.param.nam`. Metrics logged to `metrics.csv`.

### `param_infer.py` — Run inference (Python, no C++ required)

Loads `best.pt` directly and runs PyTorch inference. Use this to preview models before Phase 3 C++ integration.

```bash
python param_infer.py \
    --checkpoint /path/to/checkpoints/best.pt \
    --input /path/to/dry.wav \
    --output-dir /path/to/output/ \
    --params "knob1=0.5,knob2=0.7,knob3=0.3,..."
```

Multiple `--params` flags produce multiple output files. Knob names and order come from the dataset `config.json`.

### `batch_harness.py`

Generates the training dataset by sweeping the circuit simulation across knob permutations.

```bash
# List pots in a schx:
python batch_harness.py --backend livespice \
    --schx "/path/to/circuit.schx"

# Generate a dataset:
python batch_harness.py --backend livespice \
    --schx "/path/to/circuit.schx" \
    --knobs knob1,knob2,knob3 \
    --values 0.2,0.4,0.6,0.8 \
    --fixed-params "Rock=1,Mid=0" \
    --speaker S2 \
    --input /path/to/limelight.wav \
    --output /path/to/dataset \
    --dry-run   # check permutation count and disk estimate first

# Combine per-permutation WAVs into outputs.npy for training:
python batch_harness.py --combine /path/to/dataset
```

Knob order in `--knobs` becomes the knob order in the `.param.nam` file and in all downstream UIs.

### Ganged (multi-section) pots

`livespice_cli` sets **every** pot (or switch) whose `Name` matches the swept knob, not
just the first one. To model a mechanically-ganged control — e.g. the Klon Centaur's
dual-gang Gain (a 100 kΩ section and a 1 kΩ section that turn together) — give **both**
`Circuit.Potentiometer` components the **same `Name`** (e.g. `Gain`) in the `.schx`. A
single `--knobs Gain` / `Gain=0.5` then drives both wipers in lockstep, and the knob still
appears once in the dataset and `.param.nam`. (Before this behavior, only the first
matching section moved, so ganged pots tracked incorrectly.)

## .param.nam Format

The file is a standard NAM JSON with `"SlimmableContainer"` as the outer architecture (already registered in NeuralAmpModelerCore). Each submodel uses `"ParametricWaveNet"` — a new architecture that Phase 3 registers.

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

`config.parametric.parameters` is the authoritative knob list. Array order = UI presentation order. All values normalized 0.0–1.0. `"layers"` is the **channel count** (3 or 8), not a layer count — there are always 23 layers.

**Key design point**: `condition_size` in the JSON is the number of knobs fed to FiLM. Audio flows through `input_mixin` separately (always 1-dim). These are two independent conditioning paths — knobs never mix with audio.

## Architecture

A2 is a WaveNet variant: 23-layer dilated causal convnet with LeakyReLU and a Conv1D head. `ParametricWaveNet` adds FiLM (Feature-wise Linear Modulation) on every layer, conditioned on the knob vector. FiLM is initialized to identity (gamma=1, beta=0) so training starts from a well-behaved unconditional baseline.

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

## NAM Ecosystem Compatibility

- **`"SlimmableContainer"`** — standard, already registered in NeuralAmpModelerCore ✓
- **`"ParametricWaveNet"`** — custom; requires Phase 3 factory registration
- Official NAM tools deliberately removed user-controllable knob parameters in 2023–2024; parametric inference requires custom C++ work
- Old parametric C++ code recoverable from NAM git history at commit `ae86979` (before PR #367 removed it)

## Dependencies

### Python packages

```bash
pip install torch numpy scipy soundfile auraloss
```

### .NET SDK (for dataset generation via livespice backend)

`livespice_cli` builds against LiveSPICE, which is vendored as the `extern/LiveSPICE`
submodule. **You must initialize submodules recursively** — LiveSPICE itself nests a
`ComputerAlgebra` submodule, and `Circuit.csproj` has a `ProjectReference` to
`extern/LiveSPICE/ComputerAlgebra/ComputerAlgebra/ComputerAlgebra.csproj`. Without the
recursive init that path is empty and the build fails with dozens of `CS0246` errors
(`Expression`, `Arrow`, `Variable`, `SolutionSet` not found — those types live in
ComputerAlgebra).

```bash
brew install dotnet          # csproj targets net10.0

# Recursive is required — plain `--init` leaves ComputerAlgebra empty:
git submodule update --init --recursive extern/LiveSPICE

cd livespice_cli
dotnet publish -c Release -r osx-arm64 -o publish/
```

The binary lands at `livespice_cli/publish/livespice_cli`, which is where
`batch_harness.py` expects it.

### NeuralAmpModelerCore (Phase 3 only)

Not required for training or Python inference. Needed for C++ inference validation and [redacted] integration. Already present in `~/work/[redacted]/NeuralAmpModelerCore_v050ko/`.

## Related Repos

- `LiveSPICE-Amp-Collection` — `.schx` circuit library
- `[redacted]` — Target host for the trained `.param.nam` models
