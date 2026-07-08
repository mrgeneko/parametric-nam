# Transition Notes — spice-to-nam

This doc covers where things stand, what to transfer, and exactly what to run next.

## What This Project Is

A toolchain that converts SPICE schematics (`.schx`) into parametric NAM models. A `.param.nam` file captures a circuit's full knob range so a host ([redacted]) can do real-time inference instead of live SPICE simulation.

The two active repos:
- `spice-to-nam` — Python training scripts (`param_train.py`, `param_infer.py`)
- `livespice-emitter` — C# SPICE simulation backend + `batch_harness.py` dataset generator

## Current State (July 2026)

### Dumble clean channel — actively training

- **Dataset**: `/Volumes/DATA/work/dumble_clean/` (SSD, 864 permutations, limelight.wav input)
- **Checkpoints**: `/Volumes/DATA/work/dumble_clean_ckpt/`
  - `latest.pt` — updated every epoch
  - `best.pt` — updated whenever ESR improves
- **Best model**: `/Volumes/DATA/work/dumble_clean.param_best.nam`
- **Knobs (6)**: `volume, mid, treble, middle, bass, clean_master`
- **Speaker**: S1, Rock=0 (fixed)
- **ESR at epoch 12**: 0.0077 (below 0.01 threshold — "good")
- **Epoch time**: ~55-107 min/epoch (thermal throttling from earlier livespice runs; normalizing)

Relaunch command (if killed):
```bash
caffeinate -i ~/work/spice-to-nam/.venv/bin/python ~/work/spice-to-nam/param_train.py \
    --dataset /Volumes/DATA/work/dumble_clean \
    --output /Volumes/DATA/work/dumble_clean.param.nam \
    --slimmable \
    --epochs 500 --batch-size 64 --repeats 50 \
    --lr 3e-4 --crop-len 48000 \
    --checkpoint-dir /Volumes/DATA/work/dumble_clean_ckpt \
    --resume /Volumes/DATA/work/dumble_clean_ckpt/latest.pt \
    > /tmp/dumble_clean_train.log 2>&1
```

Monitor: `tail -f /tmp/dumble_clean_train.log`

### Dumble OD channel — dataset not yet generated

- **Target dataset**: `/Volumes/DATA/work/dumble_od_v2/`
- **Knobs (6)**: `treble, middle, bass, level, od_trim, od_master`
- **Speaker**: S2, Rock=1, Mid=0 (fixed)
- **Ranges**: od_master=0.1,0.3,0.5,0.7,0.9 / od_trim=0.3,0.5,0.7 / level=0.2,0.4,0.6,0.8 / bass/middle/treble=0.2,0.4,0.6,0.8
- **Permutations**: 3,840 (~23 min to generate)

Dataset generation command (run after clean training is stable):
```bash
~/work/spice-to-nam/.venv/bin/python ~/work/livespice-emitter/batch_harness.py \
    --backend livespice \
    --schx "$HOME/work/LiveSPICE-Amp-Collection/amps/Dumble/Overdrive Special Preamp.schx" \
    --knobs od_master,od_trim,level,bass,middle,treble \
    --range od_master=0.1,0.3,0.5,0.7,0.9 --range od_trim=0.3,0.5,0.7 --range level=0.2,0.4,0.6,0.8 \
    --range bass=0.2,0.4,0.6,0.8 --range middle=0.2,0.4,0.6,0.8 --range treble=0.2,0.4,0.6,0.8 \
    --fixed-params "Rock=1,Mid=0" --speaker S2 \
    --input "/Users/USER2/Downloads/limelight.wav" \
    --output /Volumes/DATA/work/dumble_od_v2 \
    > /tmp/dumble_od_v2.log 2>&1
```

Then combine: `python ~/work/livespice-emitter/batch_harness.py --combine /Volumes/DATA/work/dumble_od_v2`

### EVH 5150 Lead Full — trained via ngspice backend (blackbox / Linux+ROCm)

First **ngspice-backend** parametric capture — the 5150 full amp diverges in
LiveSPICE, so it's the proving ground for `--backend ngspice`. Machine: blackbox
(RX 9070 XT, ROCm), not the macOS box the Dumble runs use.

- **Dataset**: `/home/USER/work/tmp/evh5150_ds/` (200 perms, 5 knobs, 30 s sweep slice)
- **Release**: `/home/USER/work/tmp/evh5150_release/` (best-full/lite `.param.nam`, `MANIFEST.md`, `reproduce.sh`)
- **Knobs (5)**: `leadpre, leadpost, high, low, mid`; `presence` fixed 0.5
- **Result**: converged 200/200; **validation ESR ~0.19**, but **~0.43 on a new DI**
  (limelight) — "sounds like the amp, not production-close."
- **Status**: paused pending an improvement pass (see below). Unlike the Dumble
  (low-gain, ESR 0.008), the 5150 is high-gain + had two backend/method issues.

**Lessons (full write-up: [`docs/evh5150_training_notes.md`](docs/evh5150_training_notes.md)):**
1. **Aliasing** — the ngspice 48 kHz resample lacks an anti-alias filter (~20% RMS);
   fix = oversample→LPF→decimate in `_run_ngspice`.
2. **Training input** — use the **full** standard sweep, not a 30 s slice (a 30 s
   subset under-covers dark/dynamic real playing → false breakup).
3. **Fewer knobs** for high-gain amps — capacity per knob matters (dimensionality,
   not coverage holes — those were ruled out).
4. Convergence recipe: `--koren --ot-damp 3k --ot-snub 220n --nfb-comp nNFB=1n`,
   and **bound `leadpost>=0.05`** (master at 0 hangs the solver).

Next planned step: a 1-knob (`leadpre`) pilot with the full sweep + anti-aliasing to
validate the fixes cheaply before a reduced-knob overnight regen.

## Setting Up on a New Machine

### Python environment

Requires Python 3.13.

```bash
python3.13 -m venv spice-to-nam/.venv
source spice-to-nam/.venv/bin/activate
pip install torch numpy scipy soundfile auraloss
```

PyTorch uses MPS automatically on Apple Silicon — no extra config.

### .NET SDK (for dataset generation)

```bash
brew install dotnet          # csproj targets net10.0

# Recursive init is REQUIRED (see note below):
git submodule update --init --recursive extern/LiveSPICE

cd livespice_cli
dotnet publish -c Release -r osx-arm64 -o publish/
```

`batch_harness.py` expects the binary at `livespice_cli/publish/livespice_cli`.

**ComputerAlgebra.csproj gotcha**: `livespice_cli.csproj` references LiveSPICE's
`Circuit.csproj`, which in turn `ProjectReference`s
`extern/LiveSPICE/ComputerAlgebra/ComputerAlgebra/ComputerAlgebra.csproj`.
`ComputerAlgebra` is a submodule *of the LiveSPICE submodule*, so a plain
`git submodule update --init extern/LiveSPICE` leaves it empty and the build fails with
dozens of `CS0246` errors (missing `Expression`, `Arrow`, `Variable`, `SolutionSet` —
all ComputerAlgebra types). Always init with `--recursive`.

### Repos to clone

```bash
git clone https://github.com/mrgeneko/spice-to-nam.git
git clone https://github.com/mrgeneko/livespice-emitter.git
git clone https://github.com/Yahiake/LiveSPICE-Amp-Collection
# LiveSPICE_upstream — required for livespice_cli build, must be sibling of livespice-emitter
```

### Training input audio

`limelight.wav` (real guitar, 21.6s, 48kHz mono, ~-25.7 dBFS RMS) works well, but
is **not** a hard requirement.

**Correction (previously believed a solver limitation):** the earlier claim that
"LiveSPICE's nonlinear solver hangs indefinitely on sine sweeps / sustained /
broadband signals (confirmed with sweep120s.wav and a NAM studio sample file)"
was a **misdiagnosis**. The real cause was an **O(N²) bug in `livespice_cli`'s
`ReadWav`**: it called `data.ToArray()` inside the per-sample loop for 16-bit and
32-bit-float inputs, copying the whole data chunk on every sample. Large
16/32-bit-float files therefore stalled *inside the reader, before the simulation
even started*. 24-bit PCM used a direct-indexing path and was unaffected — which
is exactly why `limelight.wav` and the PCM_24 [redacted] sweep worked while the 32-bit
float `sweep120s.wav` "hung." Fixed in commit `fa90d57`; `sweep120s.wav`
(bit-identical to the first 120s of the [redacted] sweep) went from an 8-minute hang to
23.5s, matching its PCM_24 twin. **Any format/content is fine now** — sine sweeps
included.

`limelight.wav` is at `/Users/USER2/Downloads/limelight.wav` on the current machine.

## The Dumble ODS Circuit

**File**: `LiveSPICE-Amp-Collection/amps/Dumble/Overdrive Special Preamp.schx`

### Circuit topology

Two separate output paths sharing V1→tonestack:

**Clean path**: V1 → V2 (12AX7) → Tonestack (Treble/Middle/Bass) → Volume → Bright → V3 → Rock SPDT → Clean Master → **Speaker S1** (120V ref)

**OD path**: V3 → Rock SPDT → OD Trim → V5 (12AX7) → Level → V6 (12AX7) → Ratio1 → OD Master → **Speaker S2** (20V ref)

**Rock** (SPDT): Position=0 → clean (S1); Position=1 → OD (S2). Use `--speaker S1/S2` + `--fixed-params "Rock=0/1"`.

**Mid** (SPDT): Rock/Jazz tonestack switch. Position=0 = Jazz/flat. Fixed at 0 for both clean and OD datasets.

**Bright** (SPDT): No measurable effect. Excluded.

**Tonestack** (treble/middle/bass) is shared between both paths.

### Knob sets

**Clean** (`--speaker S1`, Rock=0 fixed):
`volume, mid, treble, middle, bass, clean_master`

**OD** (`--speaker S2`, Rock=1, Mid=0 fixed):
`treble, middle, bass, level, od_trim, od_master`

### Data on disk (current machine)

| Path | Contents |
|---|---|
| `/Volumes/DATA/work/dumble_clean/` | Clean dataset (864 perms, limelight.wav) |
| `/Volumes/DATA/work/dumble_clean_ckpt/` | Training checkpoints + metrics.csv |
| `/Volumes/DATA/work/dumble_clean.param_best.nam` | Best clean model so far |
| `/Users/USER2/Downloads/dumble_clean_previews/` | Inference preview WAVs |
| `/Volumes/DATA2/SSD2/work/dumble_clean/` | Original clean dataset (HDD — slow, use SSD copy) |
| `/Volumes/DATA2/SSD2/work/dumble_od/` | Original OD dataset (HDD, limelight, 1296 perms, old ranges) |

## Phase 3: C++ Inference (NeuralAmpModelerCore)

Register `"ParametricWaveNet"` factory in `NeuralAmpModelerCore_v050ko`. The outer `"SlimmableContainer"` is already registered and works.

Key design decisions:
- Audio flows through `input_mixin` (1-dim), knobs flow through FiLM (6-dim) — separate paths, never concatenated
- `"layers"` in the JSON is the **channel count** (3 or 8), not layer array count
- FiLM weights are interleaved per layer: after `l1x1` bias, before next layer's conv weights
- `film_condition_size` must be separate from `condition_size` in `LayerParams`/`LayerArrayParams` — `_input_mixin` uses `condition_size=1`, FiLM modules use `film_condition_size=6`
- Old parametric C++ code recoverable from NAM git history at commit `ae86979`

See `PLAN.md` Phase 3 section for the full implementation plan.

## ESR Targets

| ESR | Quality |
|---|---|
| < 0.01 | Good |
| < 0.005 | Excellent |
| < 0.003 | POC-level (JCM800 single-param achieved 0.003) |
