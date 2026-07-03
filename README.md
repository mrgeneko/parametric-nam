# spice-to-nam

Toolchain for converting SPICE circuit schematics (`.schx`) into parametric Neural Amp Modeler (NAM) files. The resulting `.param.nam` file captures a circuit's behavior across its full knob range — enabling real-time parametric inference in [redacted] at a fraction of the CPU cost of live simulation.

## Overview

```
.schx schematic
    ↓  batch_harness.py (in livespice-emitter repo)
Paired audio dataset: input sweep × N knob permutations
    ↓  param_train.py
.param.nam  (ParametricWaveNet: A2 architecture + FiLM conditioning)
    ↓  infer.py
Output WAVs at arbitrary knob positions
    ↓  sweep_report.py
Self-contained HTML analysis report
```

## Scripts

### `param_train.py` — Train a parametric NAM

```bash
python param_train.py \
    --data-dir /path/to/dataset \
    --checkpoint-dir /tmp/checkpoints \
    --epochs 500 \
    --channels 8 \
    --lr 1e-3 \
    --batch-size 16 \
    --repeats 1000

# Slimmable (trains A2 Lite 3ch + Full 8ch jointly, exports both):
python param_train.py --data-dir /path/to/dataset --slimmable ...
```

The dataset directory must contain the format written by `batch_harness.py`:
```
dataset/
    config.json          # circuit name, param_names, sample_rate
    input_-12dBFS.wav    # dry input sweep
    samples/
        000000/
            params.json  # {"gain_2": 0.1}
            output_-12dBFS.wav
        000001/
            ...
```

Exports `<circuit>.param.nam` on completion. Best checkpoint is saved as `best.pt`; training metrics are in `metrics.csv`.

### `infer.py` — Run inference

```bash
# Single pass
python infer.py --model my.param.nam --input dry.wav --output wet.wav --gain2 0.7

# Sweep multiple values
python infer.py --model my.param.nam --input dry.wav \
    --sweep 0.1,0.3,0.5,0.7,0.9 --out-dir /tmp/sweep_out

# Slimmable model: pick quality tier
python infer.py --model my.param.nam --input dry.wav --sweep 0.5 --quality lite
```

### `sweep_report.py` — Generate HTML analysis report

```bash
python sweep_report.py /tmp/sweep_out/limelight_gain2_*.wav \
    --model "ParametricA2 8ch ESR=0.00307" \
    --param-name "gain_2" \
    -o report.html
```

Produces a self-contained HTML file with overlaid waveforms, frequency spectra, and per-file stats (RMS, peak, DC, spectral centroid, phase correlation). No external dependencies at render time.

### `batch_harness.py` (in `livespice-emitter` repo)

Generates the training dataset by sweeping the circuit simulation across knob permutations.

```bash
# List pots in a schx:
python batch_harness.py --backend livespice \
    --schx "/path/to/Overdrive Special Preamp.schx"

# Generate a dataset (5 values × 3 knobs = 125 permutations):
python batch_harness.py --backend livespice \
    --schx "/path/to/Overdrive Special Preamp.schx" \
    --knobs od_master,level,treble \
    --values 0.0,0.25,0.5,0.75,1.0 \
    --input /path/to/sweep120s.wav \
    --output ./dumble_dataset
```

Knob order in `--knobs` becomes the knob order in the `.param.nam` file and in all downstream UIs.

## .param.nam Format

```json
{
  "version": "0.7.0",
  "architecture": "ParametricWaveNet",
  "config": {
    "layers": 8,
    "head_scale": 0.448,
    "parametric": {
      "type": "film",
      "condition_size": 1,
      "film_layers": ["layer_0", "layer_1", ...],
      "parameters": [
        {"name": "gain_2", "min": 0.0, "max": 1.0, "default": 0.5}
      ]
    }
  },
  "weights": [...],
  "sample_rate": 48000
}
```

`config.parametric.parameters` is the authoritative knob list. Array order = UI presentation order. All values are normalized 0.0–1.0.

Slimmable models use `architecture: "SlimmableParametricContainer"` with two submodels at `max_value` thresholds (0.5 = lite, 1.0 = full).

## Architecture

The model is A2 (23-layer WaveNet with LeakyReLU, mixed kernel sizes, and a 6.3k-sample receptive field) with FiLM conditioning injected on every layer. FiLM is initialized to identity (gamma=1, beta=0) so training starts from a well-behaved unconditional baseline.

Key constants (must match NeuralAmpModelerCore's A2 fast path exactly):
- 23 layers: 14 × kernel=6, then 2 × kernel=15, then 7 × kernel=6
- Dilations: repeating [1,3,7,17,41,101,239] with {1,13} degridding insert
- LeakyReLU(0.01) on every layer
- Head: Conv1D kernel=16

## Dependencies

### Python packages

```bash
pip install torch numpy scipy soundfile auraloss
```

`auraloss` provides Multi-Resolution STFT loss. `soundfile` handles WAV I/O. Everything else is standard.

### .NET SDK (for dataset generation via livespice backend)

`batch_harness.py` calls `livespice_cli`, a .NET 8 application that runs the SPICE simulation. Install .NET 8 and build the CLI from the `livespice-emitter` repo:

```bash
brew install dotnet@8   # or download from dotnet.microsoft.com/download/dotnet/8.0

cd livespice-emitter/livespice_cli
dotnet publish -c Release -r osx-arm64 -o publish/
# Use osx-x64 for Intel Mac
```

### NeuralAmpModelerCore (for C++ inference validation — Phase 3 only)

Not required for training or inference via Python. Needed when validating the exported `.param.nam` against the C++ inference engine. Clone from:
`https://github.com/sdatkinson/NeuralAmpModelerCore`

Use the same version already in `~/work/[redacted]/NeuralAmpModelerCore_v050/`.

### NAM Python trainer (reference only)

Not used directly — this repo has its own `param_train.py`. Install to reference the sweep file format:

```bash
pip install neural-amp-modeler
```

## Related Repos

- `livespice-emitter` — SPICE circuit simulation backend + `batch_harness.py`
- `LiveSPICE-Amp-Collection` — `.schx` circuit library (cloned from Yahiake/LiveSPICE-Amp-Collection)
- `[redacted]` — Target host for the trained `.param.nam` models
