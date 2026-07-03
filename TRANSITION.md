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

Requires Python 3.13 — the macOS system Python (3.9) is too old to build `neural-amp-modeler` 0.13.0. Install via `python.org` or Homebrew if not present (`/usr/local/bin/python3.13`).

Create a venv inside the repo:

```bash
python3.13 -m venv spice-to-nam/.venv
source spice-to-nam/.venv/bin/activate
```

Install dependencies. `neural-amp-modeler` 0.13.0 (required for A2 support) is not yet on PyPI, so install from GitHub:

```bash
pip install torch numpy scipy soundfile auraloss \
    "neural-amp-modeler @ git+https://github.com/sdatkinson/neural-amp-modeler.git@v0.13.0"
```

PyTorch uses MPS (Metal Performance Shaders) automatically on Apple Silicon — both `param_train.py` and `infer.py` detect `torch.backends.mps.is_available()` at runtime and use it. No extra configuration needed.

`auraloss` provides the Multi-Resolution STFT loss used in training.

Activate the venv before running any project scripts:

```bash
source spice-to-nam/.venv/bin/activate
```

### .NET SDK (required for livespice-emitter)

The `livespice_cli` binary is a .NET application. Any recent SDK (tested with .NET 10) works. Install via `dotnet.microsoft.com` if not present. On this machine it installed to `/usr/local/share/dotnet/dotnet` and is **not on PATH by default** — use the full path.

`livespice_cli.csproj` references `../../LiveSPICE_upstream/` (Circuit, ComputerAlgebra, Util), so `LiveSPICE_upstream` must be a sibling of `livespice-emitter` under `~/work/`.

Build:

```bash
/usr/local/share/dotnet/dotnet publish livespice-emitter/livespice_cli \
    -c Release -r osx-arm64 -o livespice-emitter/livespice_cli/publish/
```

The `batch_harness.py` expects the binary at `livespice-emitter/livespice_cli/publish/livespice_cli`.

### NeuralAmpModelerCore (for C++ inference validation — Phase 3)

Not needed for dataset generation or Python training, but required when validating the exported model against the C++ inference engine. Already present in `~/work/[redacted]/NeuralAmpModelerCore_v050/`. Clone or copy to the same path on the new machine.

### NAM Python trainer (reference only)

Not used directly — we have our own `param_train.py`. Already installed as part of the venv setup above (v0.13.0 from GitHub). No separate install step needed.

### Input sweep audio

Already present on this machine at:

```
~/work/parametric_nam/data/sweep120s.wav
```

(120s @ 48kHz. Note: `sweep.wav` in the same directory is 190s — `sweep120s.wav` is the shorter one despite the name being less obvious.)

## Repos to Clone

```bash
git clone https://github.com/mrgeneko/spice-to-nam.git spice-to-nam
git clone https://github.com/mrgeneko/livespice-emitter.git livespice-emitter
git clone https://github.com/Yahiake/LiveSPICE-Amp-Collection LiveSPICE-Amp-Collection
```

`livespice-emitter`, `LiveSPICE-Amp-Collection`, and `LiveSPICE_upstream` must all be sibling directories under `~/work/`. `livespice_cli.csproj` references `LiveSPICE_upstream` by relative path — the build will fail if it's missing.

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

### Speaker and channel routing (fully resolved)

**`livespice_cli` originally summed all Speaker elements** — S1 (clean, 120V ref) + S2 (OD, 20V ref). A `--speaker NAME` flag was added to `Program.cs` to select a single speaker by name.

**`Rock` is the channel selector switch** (not a tonestack switch). Position=0 routes V3 to the clean path → S1. Position=1 routes V3 to the OD path → S2. With Rock=0, S2 is completely silent (SPDT disconnects Throw1 — confirmed: S2 RMS = 0.000000). With Rock=1, S1 goes silent and S2 carries the OD signal (confirmed: S2 RMS = 0.009946).

**`Mid` is the Rock/Jazz tonestack switch** (labeled "Rock/Jazz" on the front panel, named "Mid" in the schx). Position=0 = Jazz/flat (James-style), Position=1 = Rock/scooped (Fender/Marshall/Vox style). Confirmed to change output (RMS 0.012 → 0.022, max sample diff 0.156).

**`Bright`** has no measurable effect (outputs sample-identical at Position=0 and 1). Exclude from dataset.

**Bug fixed in `Program.cs`**: component names with trailing spaces (e.g. `"Rock  "`, `"OD Trim  "`) were silently not being set because the params parser called `.Trim()` on the user key but compared against the untrimmed component name. Fixed to use `.Name.Trim() == name` for both pot and switch lookups.

**Tonestack is shared** between clean and OD channels (confirmed via forum thread: same V1→tonestack→volume path feeds both channel branches). This means `volume`, `treble`, `middle`, `bass`, and the `mid` switch all affect OD channel output too.

### Capturing the OD channel

Use `--speaker S2` and pass `Rock=1` as a fixed param to `livespice_cli`. In `batch_harness.py` this means adding `Rock=1` to the `--params` string alongside the swept knob values — currently requires a small addition to `batch_harness.py` to support fixed (non-swept) params passed through to every permutation.

OD-specific knobs (the shared tonestack knobs also apply):
- `od_master` — output level of OD path
- `od_trim` — internal gain control (between V3 and V5)
- `level` — gain staging between V5 and V6
- `ratio_1` — OD output attenuator before OD Master

### Recommended knob set

Clean path is captured, so:

```bash
python ~/work/livespice-emitter/batch_harness.py --backend livespice \
    --schx "$HOME/work/LiveSPICE-Amp-Collection/amps/Dumble/Overdrive Special Preamp.schx" \
    --knobs clean_master,treble,middle,bass,volume \
    --values 0.0,0.25,0.5,0.75,1.0 \
    --input "$HOME/work/parametric_nam/data/sweep120s.wav" \
    --output /Volumes/DATA2/SSD2/work/dumble_dataset
```

That's 5^5 = 3,125 permutations — **~100 GB** on disk (`batch_harness.py` will print an exact estimate before starting).

Use `--range KNOB=v1,v2,...` to override values for a specific knob (e.g. if tonestack sweep resolution needs to differ from the master sweep).

**Knob order matters** — it becomes the UI presentation order in [redacted] and other NAM hosts. The order above (`clean_master → treble → middle → bass → volume`) puts the most important controls first.

## Training Command (After Dataset Generation)

```bash
python ~/work/spice-to-nam/param_train.py \
    --data-dir /Volumes/DATA2/SSD2/work/dumble_dataset \
    --checkpoint-dir /Volumes/DATA2/SSD2/work/dumble_checkpoints \
    --epochs 500 \
    --channels 8 \
    --lr 1e-3 \
    --batch-size 16 \
    --repeats 1000
```

Or with slimmable (trains 3ch lite + 8ch full jointly):

```bash
python ~/work/spice-to-nam/param_train.py \
    --data-dir /Volumes/DATA2/SSD2/work/dumble_dataset \
    --checkpoint-dir /Volumes/DATA2/SSD2/work/dumble_checkpoints \
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

## What's on This Machine (~/work/)

| Path | What it is | Status |
|---|---|---|
| `spice-to-nam/` | This repo | ✓ cloned, venv at `.venv/` |
| `livespice-emitter/` | Simulation backend repo | ✓ cloned, binary built |
| `LiveSPICE-Amp-Collection/` | Circuit library | ✓ cloned |
| `LiveSPICE_upstream/` | LiveSPICE C# source (livespice_cli dep) | ✓ present |
| `parametric_nam/data/sweep120s.wav` | 120s training sweep | ✓ already here |
| `parametric_nam/poc_a2.param.nam` | JCM800 POC model | ✓ already here |
| `parametric_nam/sweep_out/limelight_gain2_*.wav` | POC inference sweep | ✓ already here |

## What Phases 3 and 4 Still Require

These are not blockers for the Dumble dataset run or training, but are needed to ship to [redacted]:

- **Phase 3**: `ParametricWaveNet` C++ subclass (WaveNet + condition injection) registered via `factory::Helper` in NeuralAmpModelerCore. Validates that the `.param.nam` produces identical output to the Python model.
- **Phase 4**: [redacted] integration — `NAMProcessorWrapper.mm` reads the `parametric` block, `NAMAudioUnit.swift` builds dynamic `AUParameter` instances at model load time.

See `PLAN.md` for the detailed implementation plan for both phases.
