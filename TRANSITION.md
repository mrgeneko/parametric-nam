# Transition Notes — spice-to-nam

Picking this up on a new machine. This doc covers where things stand, what to transfer, and exactly what to run next.

## What This Project Is

A toolchain that converts SPICE schematics (`.schx`) into parametric NAM models. A `.param.nam` file captures a circuit's full knob range so a host ([redacted]) can do real-time inference instead of live SPICE simulation.

The two active repos:
- `spice-to-nam` — Python training scripts (`param_train.py`, `infer.py`, `sweep_report.py`)
- `livespice-emitter` — C# SPICE simulation backend + `batch_harness.py` dataset generator

## Current State

**Proof of concept is done.** We trained a parametric A2 (8ch, FiLM conditioning) on the Marshall JCM800 modded preamp with a single parameter (`gain_2`). ESR reached 0.00307 at epoch 262. The exported model (`poc_a2.param.nam`) produces well-behaved output across the sweep — monotonic RMS, correct spectral centroid shift, no phase flips, no clipping.

**Next planned run: Dumble Overdrive Special Preamp** — a more complex circuit with more knobs, using the `livespice` backend (runs `.schx` directly, no code generation step).

## Setting Up the New Machine

### Python environment

```bash
pip install torch numpy scipy soundfile auraloss
```

PyTorch uses MPS (Metal Performance Shaders) automatically on Apple Silicon — both `param_train.py` and `infer.py` detect `torch.backends.mps.is_available()` at runtime and use it. No extra configuration needed.

`auraloss` provides the Multi-Resolution STFT loss used in training.

### .NET SDK (required for livespice-emitter)

The `livespice_cli` binary is a .NET 8 application that runs the SPICE simulation. Install the .NET 8 SDK:

```bash
# macOS via Homebrew:
brew install dotnet@8
# Or download from https://dotnet.microsoft.com/download/dotnet/8.0
```

Then rebuild `livespice_cli` from the `livespice-emitter` repo:

```bash
cd livespice-emitter/livespice_cli
dotnet publish -c Release -r osx-arm64 -o publish/
```

The `batch_harness.py` expects the binary at `livespice-emitter/livespice_cli/publish/livespice_cli`.

### NeuralAmpModelerCore (for C++ inference validation — Phase 3)

Not needed for dataset generation or Python training, but required when validating the exported model against the C++ inference engine. Already present in `~/work/[redacted]/NeuralAmpModelerCore_v050/`. Clone or copy to the same path on the new machine.

### NAM Python trainer (reference only)

Not used directly — we have our own `param_train.py`. Install as a reference for sweep format conventions:

```bash
pip install neural-amp-modeler
```

### Input sweep audio

The standard NAM training sweep (`sweep120s.wav`, 120s @ 48kHz) needs to be copied to the new machine. It's at:

```
/Users/USER/work/parametric_nam/data/sweep120s.wav
```

Copy it wherever makes sense on the new machine and pass its path to `batch_harness.py --input`.

## Repos to Clone

```bash
git clone <spice-to-nam-remote> spice-to-nam
git clone <livespice-emitter-remote> livespice-emitter
git clone https://github.com/Yahiake/LiveSPICE-Amp-Collection LiveSPICE-Amp-Collection
```

Note: `livespice-emitter` and `LiveSPICE-Amp-Collection` should be sibling directories alongside `spice-to-nam`. The `batch_harness.py` path constants assume this layout.

## The Dumble ODS Circuit

**File**: `LiveSPICE-Amp-Collection/amps/Dumble/Overdrive Special Preamp.schx`

### Circuit topology (important)

This is a **two-output design** with separate clean and OD signal paths:

**Clean path**: V1 → V2 (12AX7) → Tonestack (Treble/Middle/Bass) → Volume → Bright switch → V3 (12AX7) → Rock SPDT → Clean Master → **Speaker S1** (120V ref)

**OD path**: V3 → Rock SPDT → OD Trim → V5 (12AX7) → Level → V6 (12AX7) → Ratio1 → OD Master → **Speaker S2** (20V ref)

The **Rock SPDT switch** (default `Position=0` in the schx) routes V3's output to one of the two paths. The clean and OD masters are on completely separate signal chains.

### Continuous pots (discovered by batch_harness.py)

| Normalized key | Exact schx Name | Type | Taper | Value |
|---|---|---|---|---|
| `volume` | `"Volume  "` | Potentiometer | Log | 1MΩ |
| `treble` | `"Treble"` | Potentiometer | Lin | 250kΩ |
| `level` | `"Level  "` | Potentiometer | Log | 250kΩ — between V5 and V6 in OD path |
| `ratio_1` | `"Ratio  1"` | Potentiometer | Lin | 100kΩ — OD output, before OD Master |
| `clean_master` | `"Clean Master"` | Potentiometer | Log | 1MΩ — clean path only |
| `od_master` | `"OD Master"` | Potentiometer | Log | 1MΩ — OD path only |
| `od_trim` | `"OD Trim  "` | Potentiometer | Lin | 100kΩ — OD path, between V3 and V5 |
| `middle` | `"Middle"` | VariableResistor | Log | 250kΩ — tonestack |
| `bass` | `"Bass"` | VariableResistor | Log | 500kΩ — tonestack |

Switches (not continuously variable, fixed at schx defaults):
- `Bright` (SPDT, Position=0)
- `Mid` (SPDT, Position=0)
- `Rock` (SPDT, Position=0) — **determines which output path is active**

### Critical open question before running the full dataset

**We don't yet know which output LiveSPICE captures when the schx has two Speaker elements (S1 and S2).** This determines which knobs are relevant:

- If S1 (clean): `clean_master` matters, `od_master`/`level`/`ratio_1`/`od_trim` don't affect output
- If S2 (OD): `od_master` matters, `clean_master` doesn't

**Run this sanity check first** — sweep `clean_master` and `od_master` independently on a short clip:

```bash
# 3-sample sweep of clean_master only (hold everything else at default 0.5)
python batch_harness.py --backend livespice \
    --schx "/path/to/LiveSPICE-Amp-Collection/amps/Dumble/Overdrive Special Preamp.schx" \
    --knobs clean_master \
    --values 0.1,0.5,0.9 \
    --input /path/to/sweep120s.wav \
    --output /tmp/dumble_clean_check

# 3-sample sweep of od_master only
python batch_harness.py --backend livespice \
    --schx "/path/to/LiveSPICE-Amp-Collection/amps/Dumble/Overdrive Special Preamp.schx" \
    --knobs od_master \
    --values 0.1,0.5,0.9 \
    --input /path/to/sweep120s.wav \
    --output /tmp/dumble_od_check

# Then run the sweep report on whichever set you want to compare
# and check which master actually changes the output
```

Combine the outputs from both checks and pass to `sweep_report.py`, or simply listen and compare RMS.

### Recommended knob set (after sanity check confirms which path is active)

**If OD path is captured (expected)**:

```bash
python batch_harness.py --backend livespice \
    --schx "/path/to/Overdrive Special Preamp.schx" \
    --knobs od_master,level,treble,middle,bass \
    --values 0.0,0.25,0.5,0.75,1.0 \
    --input /path/to/sweep120s.wav \
    --output ./dumble_dataset
```

That's 5^5 = 3,125 permutations. `volume` is omitted (pre-tonestack gain stage that mostly scales amplitude). `ratio_1` and `od_trim` are internal OD path controls that can be added later if the 5-knob model shows gaps.

**Knob order matters** — it becomes the UI presentation order in [redacted] and other NAM hosts. The order above (`od_master → level → treble → middle → bass`) puts the most important controls first.

## Training Command (After Dataset Generation)

```bash
python param_train.py \
    --data-dir ./dumble_dataset \
    --checkpoint-dir /tmp/dumble_checkpoints \
    --epochs 500 \
    --channels 8 \
    --lr 1e-3 \
    --batch-size 16 \
    --repeats 1000
```

Or with slimmable (trains 3ch lite + 8ch full jointly):

```bash
python param_train.py \
    --data-dir ./dumble_dataset \
    --checkpoint-dir /tmp/dumble_checkpoints \
    --slimmable \
    --epochs 500 --lr 1e-3 --batch-size 16 --repeats 1000
```

Target ESR: < 0.01. The JCM800 single-param POC hit 0.003 — expect this to be harder with 5 params but the dataset will be much larger (3,125 vs 3 samples).

## Inference and Reporting

```bash
# Sweep the trained model across od_master values
python infer.py --model dumble.param.nam --input dry.wav \
    --sweep 0.0,0.25,0.5,0.75,1.0 \
    --out-dir /tmp/dumble_sweep

# Generate HTML report
python sweep_report.py /tmp/dumble_sweep/*.wav \
    --model "Dumble ODS 5-knob" \
    --param-name "od_master" \
    -o dumble_report.html
```

## What's on the Old Machine (~/work/)

| Path | What it is | Transfer needed? |
|---|---|---|
| `spice-to-nam/` | This repo | Clone fresh |
| `livespice-emitter/` | Simulation backend repo | Clone fresh + rebuild livespice_cli |
| `LiveSPICE-Amp-Collection/` | Circuit library | Clone fresh |
| `parametric_nam/data/sweep120s.wav` | 120s training sweep | **Copy this** |
| `parametric_nam/poc_a2.param.nam` | JCM800 POC model | Copy if wanted |
| `parametric_nam/sweep_out/limelight_gain2_*.wav` | POC inference sweep | Optional |

## What Phases 3 and 4 Still Require

These are not blockers for the Dumble dataset run or training, but are needed to ship to [redacted]:

- **Phase 3**: `ParametricWaveNet` C++ subclass (WaveNet + condition injection) registered via `factory::Helper` in NeuralAmpModelerCore. Validates that the `.param.nam` produces identical output to the Python model.
- **Phase 4**: [redacted] integration — `NAMProcessorWrapper.mm` reads the `parametric` block, `NAMAudioUnit.swift` builds dynamic `AUParameter` instances at model load time.

See `PLAN.md` for the detailed implementation plan for both phases.
