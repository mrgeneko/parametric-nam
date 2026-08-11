# parametric-nam

Toolchain for converting SPICE circuit schematics (`.schx`) into **parametric**
Neural Amp Modeler (NAM) files. The resulting `.param.nam` captures a circuit's
behavior across its **full knob range** — enabling real-time parametric inference
in a compatible real-time host at a fraction of the CPU cost of live simulation.

Given a `.schx` (e.g. an amp or pedal from `LiveSPICE-Amp-Collection`), it simulates
the circuit across a sweep of knob settings, trains a FiLM-conditioned WaveNet on the
paired audio, and exports a single `.param.nam` whose knobs match the real controls.

---

## Quick Start

### Try it now — no sibling repos needed

`examples/muff/` bundles a real, complete recipe: the `.schx` circuit, the
`sweepv5.wav` excitation (ours outright, freely licensed), and an annotated `config.toml`. Feel
free to substitute your own favorite sweep or DI recording — just point `input` in the config at
it. A good input covers the top of the band (real playing alone rarely does) and includes real
transient attacks, not just steady tones. The only external piece is the oracle, which is public:

```bash
git clone https://github.com/mrgeneko/parametric-nam
git clone --recurse-submodules https://github.com/mrgeneko/livespice-cli
cd parametric-nam && ./setup.sh && . .venv/bin/activate

python run_pipeline.py --config examples/muff/config.toml \
    --dataset-dir /tmp/muff_ds --nam-output /tmp/muff.param.nam \
    --checkpoint-dir /tmp/muff_ckpt
```

This trains an actual Big Muff Pi V1 (66#5) model end to end. See
`examples/muff/Big Muff Pi V1 (66#5).md` for the circuit notes.

### Bring your own circuit

The bundled example above needs nothing beyond what it already cloned. To train your own
circuit, all you need on top of that is your own `.schx` (e.g. from
`LiveSPICE-Amp-Collection`, or one you built yourself) and a `config.toml` recipe for it.
`tools/scaffold_config.py --schx yours.schx` discovers its real controls and measures a
starting `oversample` for you (Scripts, below) — or copy `examples/template.config.toml`
by hand (see **Per-circuit configs** below for the format). Either way, finish with
`tools/grid_adequacy.py --config ... --apply` to turn the placeholder knob grid into a
measured one. No other repos are required.

`setup.sh` builds the oracle from `../livespice-cli` (needs the .NET SDK; `--no-cli` to
skip). If your checkout isn't a sibling of this repo, point at it explicitly:

```bash
export LIVESPICE_CLI=/path/to/livespice-cli/publish/livespice_cli
```

### Train a device

```bash
python run_pipeline.py --config path/to/your-device/config.toml \
    --dataset-dir    /tmp/ds \
    --nam-output     /tmp/model.param.nam \
    --checkpoint-dir /tmp/ckpt
```

One command runs **four** steps:

| step | what it does | why it exists |
|---|---|---|
| **0 — Grid adequacy** | renders each knob cell's midpoint and checks the grid can represent the target ESR | a too-coarse cell puts a floor under the model that **no training can lift**. Cheap either way — render count scales with knob-axis count, not the full permutation count (~15–200 across the real device fleet) — but not a fixed time: cost per render follows `--oversample` and backend (`ngspice` is markedly slower than `livespice`). |
| 1 — Generate | renders the dataset across the knob grid | |
| 2 — Combine | per-permutation WAVs → `outputs.npy` | |
| 3 — Train | FiLM-conditioned WaveNet → `.param.nam` + release folder | prints the **training budget** in gradient steps |

**Adding a *new* device to the private fleet?** That workflow (and the toolchain's other
methodology docs — grid adequacy, convergence, training budget, retraining, architecture,
lessons learned) lives in a private internal notes repo, since it's specific to maintaining
that fleet rather than to using this toolchain standalone.

### Listen

```bash
python param_infer.py --checkpoint /tmp/ckpt/best.pt \
    --input dry.wav --output-dir /tmp/out/ --params "Sustain=0.7,Tone=0.4"
```

## How it works

```
.schx schematic                             fixed-setting captures (.nam export or raw .wav)
    ↓  gen_dataset_from_schx.py                  ↓  gen_dataset_from_captures.py
    (simulate: sweep × N permutations)           (align each capture to a shared sweep, 1/file)
                        ╲                        ╱
                         Paired audio dataset (config.json, sweep.wav, outputs.npy, params.csv)
                             ↓  param_train.py  (train A2 Lite 4ch + Full 8ch jointly, FiLM knob conditioning)
                         .param.nam  (SlimmableContainer)
                             ↓  param_infer.py  (PyTorch inference at arbitrary knob positions — no C++)
                             ↓  NeuralAmpModelerCore  (ParametricWaveNet factory, C++ inference)
                         Real-time inference in a compatible host app
```

Two ways to reach the same dataset contract: simulate a circuit, or point at a folder of
real captures — see `gen_dataset_from_captures.py` below. `run_pipeline.py` orchestrates the
`.schx` path's three steps in one command; you can also run each script directly (below).

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
    --input guitar.wav --mmap \
    --crop-len 24000 --epochs 0 --restart-period 50 --batch-size 64
```

- `--epochs 0` = **open-ended training** (see param_train) — runs until you
  `touch <ckpt>/STOP`, exporting the best model continuously.
- **Monitor progress**: every step's output streams, timestamped, to both the terminal
  and `<dataset-dir>/pipeline.log` (`--log` to put it elsewhere) — `tail -f
  <dataset-dir>/pipeline.log` from another terminal to follow a long run, including
  live training epochs once it reaches that step. `<checkpoint-dir>/metrics.csv` has
  the same training progress in structured, per-epoch form (val ESR per tier).
- On completion it builds `<nam-output>_release/`: the best-full & best-lite
  `.param.nam`, the `.schx`, `metrics.csv`, a provenance `MANIFEST.md` (params +
  timing + hardware + git revs), and a runnable `reproduce.sh`. `--no-release` skips it.
- Generation flags (`--random`, `--bounds`, `--oversample`, ngspice `--koren` etc.)
  are forwarded to `gen_dataset_from_schx`; training flags to `param_train`.

### Per-circuit configs (`--config`)

A circuit's *recipe* — schematic, input, knob grid, fixed params, widths, and
hyperparameters — lives in a declarative TOML, so the pipeline stays generic (one
`run_pipeline.py`, one small reviewable file per circuit) instead of a script per
`.schx`. Only the per-run output paths stay on the CLI:

```bash
python run_pipeline.py --config /path/to/device-config.toml \
    --dataset-dir /tmp/ds --nam-output /tmp/model.param.nam \
    --checkpoint-dir /tmp/ckpt
```

Any CLI flag overrides the config (config is loaded as argparse defaults). The `[knobs]`
table (`NAME = [v1, v2, …]`) expands to `--knobs`/`--range`, `[fixed]` to
`--fixed-params`, and `widths = [3,4,8]` to `--widths`.

**This repo carries no per-device configs of its own** beyond the bundled example — a
`config.toml` is a *living* recipe you maintain alongside your own circuits, distinct from
a specific run's *frozen* `reproduce.sh` output. `examples/muff/config.toml` is a fully
worked, annotated example of the format; `examples/template.config.toml` is a minimal
blank one to copy for a new device.

## Scripts

### `tools/scaffold_config.py` — generate a starting `config.toml` for a new circuit

```bash
python tools/scaffold_config.py --schx path/to/circuit.schx
```

Discovers the circuit's real controls (ganged pots collapsed to one, same logic
`gen_dataset_from_schx.py --schx X` with no `--knobs` uses to list them) and writes a
`config.toml` with a `[knobs]` entry per control, copying the rest of
`examples/template.config.toml` verbatim. For the `livespice` backend (default) it also
*measures* a starting `oversample` — runs `measure_truncation.py`'s own candidate sweep
and prints the same kind of table, but doesn't pick a value out of it automatically: a
"stalled" candidate in that table can mean a render problem, not convergence, and the
docs treat it as something to investigate by hand, not a green light. So it writes the
largest tested candidate as a starting point and leaves the judgment call to you. The
`[knobs]` grid it writes is a **placeholder** (`--grid-points` evenly-spaced values per
knob, default 3) — run `tools/grid_adequacy.py --config <output> --apply` next to turn it
into a measured one.

### `tools/grid_adequacy.py` — measure whether a knob grid is dense enough

```bash
./tools/grid_adequacy.py --config path/to/device-config.toml --target 0.009
./tools/grid_adequacy.py --config ... --suggest       # propose a regrid, print it, stop
./tools/grid_adequacy.py --config ... --apply         # iterate suggest+reverify, write the result
```

Renders the circuit's true output at each grid cell's midpoint and compares it against
interpolating the two neighboring points — the residual is the interpolation error **the
grid imposes**, independent of any model. Run this before training, not after: a cell
whose residual exceeds `--target` is a floor that no training capacity or time can lift,
since the information was never sampled. `--suggest` proposes a denser/sparser grid and
prints it as TOML for you to paste into `[knobs]` yourself. `--apply` goes further: it
bisects the failing cells, re-probes the new grid, and repeats (reusing the same render
cache across rounds) until every cell clears `--target` or `--max-iterations` runs out,
writing the converged grid straight into `--config`'s `[knobs]` table — one command
instead of suggest → hand-copy → rerun → repeat. It doesn't touch anything else in the
file, comments included.

### `measure_truncation.py` — measure BDF2 truncation error, pick `oversample`

```bash
./measure_truncation.py --input examples/sweepv5.wav --config path/to/device-config.toml
```

LiveSPICE integrates with BDF2 (O(h²)), so the simulated circuit is never quite the real
one — the gap is **truncation error**, and it shrinks as `--oversample` rises. Measures it
against a high-oversample reference (not the next rung up, which understates the error
~3x) so you can pick an `oversample` whose truncation sits comfortably below the model ESR
you're chasing — training past that point just teaches the model the integrator's own
mistakes.

### `gen_dataset_from_schx.py` — generate the dataset

Simulates the circuit across knob permutations, one WAV per permutation, then combines
them into `outputs.npy`.

```bash
# List the pots/switches in a schx:
python gen_dataset_from_schx.py --backend livespice --schx "<circuit>.schx"

# Grid sweep (factorial — fine for 1–3 knobs):
python gen_dataset_from_schx.py --backend livespice --schx "<circuit>.schx" \
    --knobs drive,tone --range "drive=0,0.5,1" --range "tone=0.2,0.5,0.8" \
    --input guitar.wav --output <ds> --dry-run   # check perm count + disk first

# Random sampling (for high knob counts — grids explode past ~3 knobs):
python gen_dataset_from_schx.py --backend livespice --schx "<amp>.schx" \
    --knobs gain,bass,mid,treble,master --random 200 \
    --bounds bass=0.15,0.85 --bounds mid=0.15,0.85 --bounds treble=0.15,0.85 \
    --input guitar.wav --output <ds>

# Combine per-permutation WAVs into outputs.npy:
python gen_dataset_from_schx.py --combine <ds>
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

### `gen_dataset_from_captures.py` — build a dataset from real hardware captures, no `.schx` needed

An alternative to `gen_dataset_from_schx.py` for devices with **no SPICE model** — a set of
already-captured, fixed-setting files (one per knob setting) instead of a circuit to simulate.
Two source kinds, auto-detected per file by extension and freely mixable in one run:

- **`.nam`** — an already-trained/exported fixed-setting model (real hardware or amp-modeler
  export). Run once against the shared sweep to produce its wet render (digital inference,
  exactly phase-aligned to the input by construction).
- **`.wav` / `.aif` / `.aiff`** — an already-recorded wet capture: the shared sweep played
  through the real device and captured directly (e.g. via an audio interface). Any bit depth,
  sample rate, or channel count libsndfile can read works (stereo is downmixed to mono; a
  sample-rate mismatch against the shared sweep is resampled automatically). A real analog
  signal chain has its own latency the `.nam` path's pure inference doesn't, so each capture is
  time-aligned via best-effort NAM blip-based calibration before use — real calibration only
  for a NAM-recognized standard sweep (e.g. `sweep-v3.wav`); capturing against this repo's own
  bundled `sweepv5.wav` always falls back to a disclosed `delay=0`, same as
  `capture_static.py`'s identical fallback for a non-standard excitation. `neural-amp-modeler`
  is optional — without it every raw capture just uses `delay=0` too.

```bash
python gen_dataset_from_captures.py \
    --captures "~/Downloads/5150 DST *.nam" \
    --output /tmp/5150_ds --gear-make "EVH" --gear-model "5150 Iconic EL34 15w"

python gen_dataset_from_captures.py --captures "~/Downloads/Klon *.wav" --output /tmp/klon_ds

python gen_dataset_from_captures.py --combine /tmp/5150_ds   # same --combine flag as gen_dataset_from_schx.py
```

- **Filenames encode the knob settings**, same convention for either source kind.
  `"5150 DST G2, B5, M5, T5.nam"` → `Gain=0.2, Bass=0.5, Mids=0.5, Treble=0.5`
  (comma-separated `PrefixDigit` tokens, digit → value/10). Override the prefix→knob-name map
  and the digit→value scale per batch with `--knob-map` (e.g. `--knob-map D=Drive`) /
  `--knob-scale`, or skip filename parsing entirely with `--mapping-csv` for conventions too
  irregular to tokenize.
- **A knob that never changes across the batch is auto-fixed**, not swept — real capture
  batches are a scattered handful of points, not a dense grid, and most tokens in any given
  set of files won't vary at all.
- Every file is aligned against a **shared sweep input** (`--input`, same convention as the
  `.schx` pipeline — defaults to `sweepv5.wav`), producing exactly one permutation per file.
  This is a **scattered point-sample dataset**, not a Cartesian grid — `grid_adequacy.py`'s
  interpolation reasoning doesn't apply; there's nothing systematic to interpolate between a
  handful of arbitrary points.
- Output has the **same directory contract** `gen_dataset_from_schx.py` produces (`config.json`,
  `sweep.wav`, `params.csv`, `outputs.npy` after `--combine`), so `run_pipeline.py --config`
  and `param_train.py` read it completely unmodified — from here on it's the identical
  training path as a SPICE-rendered dataset.
- **`--restricted-input`** marks the sweep as licence-restricted-for-training-only (not
  redistributable) if that's what you're feeding it — writes a `NOTICE` file and an
  `input_restricted` flag into `config.json` so the dataset directory is self-documenting.

### `param_train.py` — train a parametric NAM

```bash
python param_train.py --dataset <ds> --output <model.param.nam> --checkpoint-dir <ckpt> \
    --mmap --crop-len 24000 --batch-size 64 --repeats 8 \
    --lr 3e-4 --epochs 200
```

- Training is **always slimmable**: it jointly trains an A2 **Lite 4ch + Full 8ch**
  and exports both tiers in one SlimmableContainer. `--widths` overrides the tier
  channel counts (default `4,8`).
- **Per-tier best-checkpointing**: each tier's own best epoch (they differ) is saved
  and exported live to `<output>.best_full.param.nam` / `<output>.best_lite.param.nam`
  (and `.best_w<N>.param.nam` for any middle tier), plus `best.pt` / `best_<tier>.pt`
  and per-epoch `latest.pt` + `metrics.csv`.
- **Composite export**: whenever *any* tier improves, its current per-tier bests are
  also spliced into one combined container and exported live to
  `<output>.optimal.param.nam` — the same multi-tier artifact `release_run.sh`
  otherwise builds from checkpoints after the fact, kept up to date automatically so
  you can spot-test it mid-run without a separate manual export step. Cheap: reuses
  the per-tier weights already held in memory, no extra disk read and no forward pass.
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

### `sweep_report.py` — visualize a set of WAVs at different knob settings

```bash
python sweep_report.py /tmp/ds/*_gain_*.wav --param-name gain -o report.html
```

Self-contained HTML report from any set of WAVs, one per parameter setting (sorted by
name): overlaid waveforms, frequency spectra, and per-file stats (peak, RMS, RMS
monotonicity across the sweep, spectral centroid trend). Doesn't care where the WAVs came
from — point it at a generated dataset's per-permutation renders to eyeball the sweep
before training (`coverage_report.py` checks the *sampling*, this checks the *audio*), or
at a batch of `param_infer.py` outputs to eyeball a trained model's knob response.

### `export_checkpoint.py` — checkpoint → `.param.nam`

```bash
python export_checkpoint.py --checkpoint <ckpt>/latest.pt --dataset <ds> \
    --output model_final.param.nam --state model
```
Re-exports a NAM from any `.pt` (e.g. the final `latest.pt` weights) without retraining.

### `bake_nam.py` — freeze a knob setting for stock/upstream NAM plugins

```bash
python bake_nam.py --in model.param.nam --params "Gain=0.7,Tone=0.5" -o tone.nam
```

Standard NAM plugins have no runtime knob input, so a raw `.param.nam` handed to one just
throws (`"No config parser registered for architecture: ParametricWaveNet"`). This bakes
one specific knob setting into an ordinary static `.nam` instead — FiLM is affine over the
layer's linear ops, so freezing a setting folds exactly into `conv`/`mixin` weights, no
retraining, pure offline weight transform. Omitted knobs fall back to their own declared
default, not a blanket 0.5 (wrong for circuits like the Timmy, whose controls don't center
at noon).

By default the output is **dual-payload**: the baked tone at the top level (so any stock
plugin plays it) plus the full original parametric model under an `embedded_parametric`
key that stock loaders ignore but a parametric-aware host (`NAMix`,
`NeuralAmpModelerPlugin`/"Anti-Static") reads for live knobs — one file, safe to hand
anyone. `--no-embed-parametric` drops that for a static-only file, e.g. for a capture pack
where embedding the full master in every tone just multiplies size.

### `tools/plot_tone_response.py` — frequency-response chart for an exported `.nam`

```bash
python tools/plot_tone_response.py \
    --model bundle.optimal.param.nam --config dataset/config.json \
    --out tone_response.svg --summary tone_response.md \
    --schx circuit.schx
```

Renders a low-level sweep through every tier of a `SlimmableContainer` `.param.nam`, at each
tone knob's min/max (the drive knob held low, so the curves reflect frequency shaping, not
distortion — `--include-drive` charts it anyway), via the real C++ render path
(`render_parametric`). Draws a grid of magnitude-response curves: rows = tiers, columns =
swept knobs — self-consistency across tiers by default. Pass `--schx` to also render the
schematic through the LiveSPICE oracle and overlay it as ground truth, turning the chart
into a per-knob accuracy check (does the model's Bass/Mid/Treble response actually match the
circuit?) instead of just tier agreement. Best-effort throughout: falls back to a
model-only chart if the oracle is unavailable, and exits 0 with a warning (never blocks a
release) if `render_parametric` itself isn't built.

### `release_run.sh` — verify + stage a finished run

Calls `plot_tone_response.py` above automatically to produce the release bundle's
`tone_response.svg`/`.md`. Its own defaults assume a private archive layout you don't
have; override `RUN`/`DS`/`SCHX`/`CONFIG` to point at any recipe instead, including the
bundled `examples/muff/` one — continuing straight from the "Try it now" run above:

```bash
RUN=/tmp/muff DS=/tmp/muff_ds \
SCHX=examples/muff/"Big Muff Pi V1 (66#5).schx" \
CONFIG=examples/muff/config.toml \
    ./release_run.sh
```
Packages a finished (or killed) `run_pipeline.py` run into a verified release bundle:
validates the `.nam` payloads, composes the tiers into one "optimal" container, measures
what the composite buys, and writes `MANIFEST.md` + `reproduce.sh`. Nothing about the
model shape is hardcoded — tiers/widths/config are all derived from the run itself.

No git or network dependency — it never touches another repo. Publishing the resulting
bundle somewhere (e.g. your own model archive) is a separate, optional step; the script
prints the command for that at the end rather than running it.

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

## Known issue: rare knob-corner blowup (FiLM/LeakyReLU runaway)

Trained `.schx` models can develop a **narrow, catastrophic instability** at a specific
knob-grid corner — the model's prediction spikes to tens or hundreds of times its normal
peak level on real transient content, while every aggregate metric (val ESR, per-tier
loss curves) looks fine, because the corner is a single cell out of hundreds-to-thousands
and contributes almost nothing to the training loss. It has recurred independently on at
least two shipped models (the pre-fix Boss DS-1, and Tweed 5F6-A Full sag) at different
knob combinations, so treat it as a real, recurring failure mode of this pipeline, not a
one-off.

What it looks like, concretely (Tweed 5F6-A Full sag, lite tier): at
`NormalVol=BrightVol=0.025` (the swept grid's own minimum) combined with `Treble=Bass=
Middle=0.8`, the model predicted a peak of **100.2 V** against a ground-truth peak of
**0.64 V** (156×) — RMS stayed normal, so it's a brief spike, not sustained distortion,
and it's invisible to ESR unless you check per-permutation, not just aggregate. A second,
nearby corner (`Bass=0.5` instead of `0.8`) showed the same signature at 41×. Both
cleared the same distinctive test: moving `NormalVol`/`BrightVol` off the exact trained
minimum by as little as **0.025** (to 0.05) — not even to the next grid point — dropped
the peak straight back to ~0.7 V. That knife-edge sensitivity (fires exactly *at* a
trained grid value, not in a neighborhood around it) is the fingerprint of this failure:
it isn't a smooth under-generalization gradient, it's closer to a discontinuity the
network found room to plant right on a specific training point.

**Why `.schx` training is exposed to this in particular**: the swept knob grid is a
finite set of discrete points, and *mixed* corners — several knobs simultaneously at
their own min/max, not just one knob varied in isolation — are combinatorially numerous
(2ⁿ for n swept knobs) and easy to under-sample relative to how densely the "safe" middle
of the grid gets covered. A corner like this can go completely unnoticed by grid
adequacy (which checks interpolation error, not model behavior) and by aggregate
validation ESR (which averages over everything else).

**Mitigations, both already in this repo**:
- **`tools/check_transient_coverage.py`** (pre-training gate, run automatically by
  `gen_dataset_from_schx.py` unless `--skip-transient-check`) — checks that the
  excitation's transient content actually reaches each corner's own saturation onset,
  across the **full min/max hypercube** (every swept knob independently at its own grid
  min or max, not just one knob varied from center) plus the traditional solo-knob
  corners. A corner whose real-playing content never crosses into saturation during
  training is a corner the network has to extrapolate at inference time.
- **`tools/scan_film_runaway.py`** (post-training, run by hand against a finished
  `.param.nam`) — replays a real reference clip through every tier at every corner
  (`--config` for the exact trained grid, not just the reduced set) and flags any window
  whose peak is anomalous relative to that model's own typical output. Scanning the
  Tweed fp32 model with a **generic** guitar reference came back clean even including
  this exact corner — the spike only showed up against the **actual training excitation**
  at that permutation, so when investigating a suspected corner, prefer `--reference
  <the training sweep.wav>` over an arbitrary clip; a scan that doesn't reproduce the
  triggering content can give a false sense of safety.

If either tool flags a corner, don't just retrain longer at the same grid — the blowup
in the Tweed case was a spike lasting well under a second inside a single permutation's
~27 s clip, so it barely moves that permutation's own loss, let alone the average across
hundreds of permutations; more of the same data is unlikely to fix it. Narrowing the
grid's swept range, adding an explicit loss weight on transient/peak error, or excluding
that exact corner combination and documenting it as an unsupported setting are the
levers that actually address it.

---

## What a run produces

A **dataset** directory (`gen_dataset_from_schx --combine` format):
```
config.json   # {"knobs": [...], "bounds": {...}, "param_map": {...}, ...}
sweep.wav     # the dry input used
outputs.npy   # [N_perms, N_samples] float32
params.csv    # idx, knob1, ..., rms, peak, ok, error
```

A **release folder** (`<nam-output>_release/`, built by `run_pipeline`): every
per-tier `.best_<tier>.param.nam` plus the composite `.optimal.param.nam`, the
`.schx`, `metrics.csv`, `MANIFEST.md` (provenance: params, per-step timing, hardware,
git revisions), and a runnable `reproduce.sh`.

---

## Reference

### `.param.nam` format

A standard NAM JSON with `"SlimmableContainer"` as the outer architecture (already
registered in NeuralAmpModelerCore). Each submodel uses `"ParametricWaveNet"` — a
custom architecture [our fork of NeuralAmpModelerCore](https://github.com/mrgeneko/NeuralAmpModelerCore)'s
factory registers. [`NAMix`](https://github.com/mrgeneko/NAMix) and
[`NeuralAmpModelerPlugin`](https://github.com/mrgeneko/NeuralAmpModelerPlugin) (see
"Related Repos") are working sample hosts built against it.

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
  FiLM. A parametric-aware host (`NAMix`, `NeuralAmpModelerPlugin`) keeps live knobs;
  `bake_nam.py` "exports this tone" into a stock-standard `.nam` for hosts that don't.
  Works on **already-trained** parametric `.nam`s offline (no retrain) — pure weight
  transform.

---

## System Requirements

Training parametric NAM models is GPU- and memory-intensive; a GPU is required in
practice. Real device grids run into hundreds or thousands of permutations (e.g. the
JCM800 hot-rod's 1944), and real training budgets run into the tens of thousands of
gradient steps (see `--target-steps` above) — CPU-only training isn't realistically
feasible for a real run, not just slow. `--device auto` (the default) refuses to start
if no GPU is detected, rather than silently queuing a run that will never finish —
pass `--device cpu` explicitly if you genuinely want to (e.g. a short debugging run).

- **GPU VRAM**: 16 GB+ — needed for comfortable training of slimmable models
  with 4 widths at batch-size 16+.
- **System RAM**: 24 GB+.

### Notes for AMD GPU users

PyTorch's ROCm build may lack compiled kernels for certain GPU architectures.
If you encounter `rocBLAS` errors or `illegal memory access` on an AMD GPU
(e.g. Radeon RX 6400, gfx1034), two workarounds are needed:

1. **Missing `TensileLibrary.dat`** — create a symlink to the fallback database:
   ```bash
   ln -s TensileLibrary_Type_4xi8I_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_fallback.dat \
         /path/to/torch/lib/rocblas/library/TensileLibrary.dat
   ```

2. **Override GPU architecture** — set this environment variable before running
   training (maps gfx1034 → gfx1030, which has precompiled kernels):
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0
   ```

3. **Symlink missing architecture kernels** — create gfx1034 → gfx1030 symlinks
   so rocBLAS and MIOpen find suitable code objects for the actual GPU:
   ```bash
   LIBDIR="/path/to/torch/lib/rocblas/library"
   for f in "$LIBDIR"/*gfx1030*; do
     base=$(basename "$f")
     target="${base/gfx1030/gfx1034}"
     [ ! -e "$LIBDIR/$target" ] && ln -s "$base" "$LIBDIR/$target"
   done
   ```

Even with these workarounds, GPUs with only 4 GB VRAM (e.g. RX 6400) are
**not sufficient** for training slimmable WaveNet models — they consistently
fail with `illegal memory access` regardless of batch size or crop length, and
CPU training isn't a realistic fallback (see System Requirements above). You
need a GPU with more VRAM.

---

## Build & Dependencies

**The easy path is `./setup.sh`** (see Quick Start) — it does everything below
idempotently. The manual steps are documented here for reference / troubleshooting.

### Python packages
Pinned in [`requirements.txt`](requirements.txt) (torch, numpy, scipy, soundfile):
```bash
pip install -r requirements.txt
```
For NVIDIA CUDA, install the matching `torch` build from the PyTorch index first
(see the note at the top of `requirements.txt`), then `pip install -r requirements.txt`.

### .NET SDK + the oracle (for dataset generation)

`livespice_cli` — **the oracle** — lives in
[**`mrgeneko/livespice-cli`**](https://github.com/mrgeneko/livespice-cli), a small standalone
public repo, not here. This repo used to carry its own near-copy, and the two drifted, as
duplicates do: a fix making an unknown `--params` name a hard error (instead of silently
rendering at the defaults, so every "swept" permutation comes out **identical** and the trainer
learns the knob does nothing) landed in one copy and not the other. Two tools built from one
schematic, disagreeing about what the device's knobs *are*. There is now exactly one. (It was
originally extracted from `hotspice/oracle/` — same reasoning, promoted to its own repo since it
has no other functional connection to this project or to hotspice's emitter.)

```bash
git clone --recurse-submodules https://github.com/mrgeneko/livespice-cli   # as a SIBLING of this repo
cd livespice-cli && ./build.sh
```
(.NET 10 SDK: `apt install dotnet-sdk-10.0` on Linux, `brew install dotnet` on macOS. `--recurse-submodules`
matters — LiveSPICE has its own nested submodule; a plain clone leaves it empty and the build fails.)

`gen_dataset_from_schx.py` resolves the binary in this order: **`$LIVESPICE_CLI`** → a sibling
`../livespice-cli/publish/livespice_cli`. Set the env var if your checkout lives
elsewhere.

The oracle builds against **pristine upstream LiveSPICE**, never a patched fork — an oracle built
from the thing under test is not an oracle — and `build.sh` warns if it finds fork markers in the
submodule (a sign the pin points somewhere other than pristine upstream).

> **The micro-sign patch is gone, and deliberately.** LiveSPICE's `Quantity` parser knows
> `U+03BC GREEK MU` (and ASCII `u`) but *not* `U+00B5 MICRO SIGN`, so upstream reads `4.7µF`
> written with U+00B5 as **4.7 farads** — no throw, no warning, every capacitor a dead short, and
> the simulation confidently wrong. We used to patch a vendored LiveSPICE fork, which is
> incompatible with keeping the oracle pristine. `livespice_cli` now **normalises U+00B5 → U+03BC
> when it loads the schematic**, so both encodings render bit-identically and no fork is needed.
> (Our library currently uses U+03BC in 6 of 23 files and U+00B5 in none — but U+00B5 is what most
> editors emit, so this was one save-from-the-wrong-tool away from a corrupted dataset.)

### ngspice (only for `--backend ngspice`)
```bash
apt install ngspice        # Linux;  macOS: brew install ngspice
```

### NeuralAmpModelerCore (C++ inference only)
Not required for training or Python inference — only for C++ inference validation and
host-app integration.

### `neural-amp-modeler` (optional — `.nam`-sourced captures and real `.wav` delay calibration)
Not required for the `.schx` path, or for a `.wav`-sourced dataset without real calibration
(it just falls back to a disclosed `delay=0` — see `gen_dataset_from_captures.py`'s
`detect_delay`). Needed only for **loading an existing `.nam` file as a capture source**
and for **real NAM blip-based latency calibration** on a `.wav` capture. Build as a sibling
of `livespice-cli`, not inside this repo:

```bash
git clone https://github.com/sdatkinson/neural-amp-modeler ../neural-amp-modeler
cd ../neural-amp-modeler && python3 -m venv venv && ./venv/bin/pip install -e .
```

Auto-discovered from `~/work/neural-amp-modeler/venv` (same convention
`capture_static.py` already uses); override with `$NEURAL_AMP_MODELER_HOME` if yours lives
elsewhere. The delay-calibration half runs in a **subprocess** under that venv's own
interpreter, not a lazy in-process import — `nam.train.core` depends on `numba`, which
checks `numpy`'s version at import time, and importing it in-process would hand it
whatever `numpy` *this* repo's own venv already loaded, not the sibling venv's compatible
one. A subprocess avoids that entirely; see `tools/nam_delay_helper.py`.

> **Input formats**: any WAV (16/24/32-bit) and any content read correctly — an earlier
> O(N²) bug in `livespice_cli`'s WAV reader (fixed in `fa90d57`) once made non-24-bit
> inputs stall.

## Related Repos

- `LiveSPICE-Amp-Collection` — `.schx` circuit library
- [`livespice-cli`](https://github.com/mrgeneko/livespice-cli) — the oracle (`livespice_cli`), public
- [`NeuralAmpModelerCore`](https://github.com/mrgeneko/NeuralAmpModelerCore) — our fork,
  public. Adds the `"ParametricWaveNet"` architecture + live knob support on top of Steven
  Atkinson's original DSP core.
- Sample host apps with working parametric-knob support, both public, both built against
  that fork: [`NAMix`](https://github.com/mrgeneko/NAMix) (Linux VST3/standalone) and
  [`NeuralAmpModelerPlugin`](https://github.com/mrgeneko/NeuralAmpModelerPlugin) — rebranded
  "Anti-Static" — (macOS/Windows VST3/AU/standalone)

## Credits & Attribution

This toolchain builds on several open-source projects and published models:

- **[LiveSPICE](https://github.com/dsharlet/LiveSPICE)** (Dillon Sharlet, MIT) — the
  circuit simulator `livespice_cli` builds against (pristine, pinned in
  [`livespice-cli`](https://github.com/mrgeneko/livespice-cli)'s `extern/LiveSPICE` submodule),
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
  — not a dependency, but `param_train.py`'s multi-resolution STFT loss is a from-scratch
  reimplementation ported to match its `MultiResolutionSTFTLoss` formulas and default
  weights (themselves vendored into the official NAM trainer).
- **Ideal-transformer technique** — the ngspice output transformer uses the standard
  controlled-source (E+F) ideal-transformer method; the specific center-tapped
  equations are ported from LiveSPICE (above).

Underlying scientific-Python stack: PyTorch, NumPy, SciPy, soundfile.
