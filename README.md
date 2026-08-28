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

### Try it now

`examples/muff/` holds a real, complete recipe: the `.schx` circuit and an annotated
`config.toml`. One thing is not in the repo — **the excitation**. Download `T3K-sweep-v3.wav`
from **<https://www.tone3000.com/capture>** into `examples/` before your first run (see
[The sweep file](#the-sweep-file) below). Feel free to substitute your own favorite sweep or DI
recording — just point `input` in the config at it. A good input covers the top of the band
(real playing alone rarely does) and includes real transient attacks, not just steady tones.
The only external piece is the oracle, which is public:

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

### The sweep file

**This repo does not ship an excitation.** Download **`T3K-sweep-v3.wav`** from
**<https://www.tone3000.com/capture>** and put it in `examples/` — that is the default `input`
for `scaffold_config.py`, `gen_dataset_from_captures.py`, and the bundled example configs.
`examples/*.wav` is gitignored, so it will not be committed back.

```bash
# after downloading:
ls examples/T3K-sweep-v3.wav
```

It is a 190 s real-playing clip at 48 kHz. Two properties matter downstream:

- **It is a NAM-recognized standard sweep**, so `gen_dataset_from_captures.py`'s blip-based
  time alignment does real calibration against it rather than falling back to `delay=0`.
- **The two full-scale single-sample impulses are NAM's calibration blips — do not remove
  them.** In `T3K-sweep-v3.wav` they sit at exactly t=10.5 s and t=11.5 s, value 0.99,
  surrounded by true zero. `nam.train.core._calibrate_latency_v_all` searches around that
  expected blip location, so stripping them forces `gen_dataset_from_captures.py` /
  `capture_static.py` to fall back to a disclosed `delay=0` — losing the real calibration that
  is one of the reasons to use this sweep at all. They are not recording artifacts, despite
  looking exactly like slate clicks. (A declicked variant is harmless for a RENDERED dataset,
  where dry and wet are aligned by construction, but it is the wrong input for any real-capture
  device.) They also double as the hardest transient in the file — a single-sample impulse is
  maximally broadband with an instant attack, which is useful content for a circuit whose
  failure mode is runaway on sharp attacks.

  One consequence to be aware of rather than to fix: `build_excitation.py` normalises the real
  segment by its own max, so the blip (0.99) sets the scale and actual playing lands at ~90.8%
  of `--realistic-peak` rather than 100% — a −0.8 dB offset, not a defect.

- **On its own it does NOT cover the loud region** — 22 dB crest factor, and only **0.075 %** of
  its samples sit within 6 dB of peak. A model trained on it alone never sees the device
  saturating, and goes out-of-distribution the moment a hot input arrives. **This is the
  excitation defect behind this pipeline's FiLM-runaway blowups** (see the Known issue below,
  and the Boss DS-1, whose shipped model spiked to 12.39 peak on a real pick attack where the
  reference circuit produced 0.46).

So for a real training run, do not feed it in raw — build an excitation from it:

```bash
python tools/build_excitation.py --input examples/T3K-sweep-v3.wav \
    --output ~/work/tmp/DEVICE_excitation.wav \
    --realistic-peak <below the device's measured saturation onset> \
    --sweep-peaks <levels up to the device's max output + headroom>
```

That concatenates the real clip with amplitude-stepped log sweeps that **do** reach maximum
output, plus a leading silence so the render is not sampling a cold-start transient. Then gate
it with `tools/check_transient_coverage.py`, which fails if any knob corner's transient content
never reaches that corner's own saturation onset.

**Why not the old bundled sweep?** Until 2026-08-28 this repo shipped its own `sweepv5.wav`
(original Logic Pro material plus synthesised chirps — ours outright, and freely
redistributable). **Licensing is not why it went.** It was replaced because the TONE3000 sweep
is simply the better and more standard choice:

- **It is the ecosystem standard.** Captures made against it are directly comparable with
  everyone else's, and NAM recognises it, so blip-based time alignment does real calibration
  instead of falling back to a disclosed `delay=0`.
- **Its transients are real.** `sweepv5.wav`'s attacks had been thinned because they caused
  ngspice divergence during dataset generation — and that is the root cause of the Boss DS-1
  shipping a model that spiked to **12.39** peak on a real pick attack where the reference
  circuit produced **0.46**. The model had never been shown a stable response to a sharp attack.
  The DS-1 was moved onto `sweep-v3.wav` for exactly this reason; the rest of the fleet has now
  followed.

The trade-off is that `sweepv5.wav` came with the synthesised loud-region coverage already
attached, whereas `sweep-v3.wav` is real playing only — hence `build_excitation.py` above,
which adds that coverage back explicitly and at levels tied to the device's *measured*
saturation point rather than to whatever the source clip happened to contain.

Comments throughout the codebase that cite measurements "on sweepv5" refer to the old file.
Those numbers are properties of the circuit **and that excitation**, so they do not
automatically carry over — `measure_truncation.py` and `grid_adequacy.py` both measure through
whatever `input` is configured.

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
largest tested candidate as a starting point and leaves the judgment call to you.

It also writes a **`[knob-kind]` table**, guessing each knob's role (`hi`/`lo`/`mid`
tone-shaping, `drive` gain/distortion, `rms` volume/level) from its name via
`tools/knob_classify.py` — the same heuristic `preflight.py` uses for its own direction and
EQ-swamp checks (see [that Known issue](#known-issue-preflights-eq-knob-checks-can-be-swamped-by-the-circuits-own-gain-control)
below). A name that matches nothing is written commented-out as `UNCONFIRMED` rather than
silently defaulted or guessed wrong — **always review this table by hand**: a misclassified
knob doesn't just get the wrong sensitivity metric in `gen_dataset_from_schx.py`, it also
silently skips `preflight.py`'s role-aware EQ-check protection for every OTHER knob's
direction check. Override a bad guess with `preflight.py --knob-kind NAME=kind,...`, or just
edit the table before your first real run.

The `[knobs]` grid it writes is a **placeholder** (run `tools/grid_adequacy.py --config
<output> --apply` next to turn it into a measured one) but a role-aware one, not a naive
`linspace(0,1)` for every knob: a tone/EQ knob (`hi`/`lo`/`mid`) stays evenly spaced but
narrowed to `[0.2, 0.8]` (`--grid-points` points, default 3) — the fully-CCW/CW extremes
rarely hold a tone stack's interesting behavior. A gain/volume knob (`drive`/`rms`) is NOT
evenly spaced: fixed anchors at `min`/`min+0.05`/`min+0.15`/`max-0.05`/`max` (`0.1, 0.15,
0.25, 0.95, 1.0`) are always included — density where a gain knob's character changes
fastest near the bottom, plus a point just below max to stabilize that grid cell — merged
with `--grid-points`' own evenly-spaced baseline across the full range, matching hand-tuned
grids already used on this fleet (e.g. `Gain=[0.1,0.15,0.25,...,0.9-0.95,1.0]`). An
unclassified knob keeps the old naive `[0, 1]` evenly-spaced behavior.

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

`backend = "ngspice-deck"` (or `"ltspice-deck"`) in the config renders through a
hand-written ngspice/LTspice deck (`pedal-dir`/`module`/`probe-node` in the TOML, same
convention as `preflight.py`/`prepare_excitation.py` — see Backends, below) instead of a
`.schx`, for a device whose clipping needs a real component `.schx` has no model for (a
MOSFET, a real BJT) — or, for `ltspice-deck` specifically, a circuit whose ngspice deck
can't converge on real playing content at all (see Backends).

**Exit status — an over-target cell is not a failure.** It exits **0** even with cells over
target, because refining one costs renders and permutations and a config may rationally accept a
coarse cell, coarsen an axis, or fix a knob outright instead of paying. That is a priced
trade-off, not a defect, and gating on it would turn a judgement call into an error.

What *does* fail the run (**exit 2**) is a **failed render**, because that is the case that
produces a WRONG table rather than a weaker one: every cell is computed from whichever probes
survived, and nothing else in the output says so. Measured on the Joyo American Sound, 38 of 48
probes timed out and the printed table put one cell at 0.6578 (18.8× over) where the clean
re-run measured 0.0475 (1.4×) — a 14× error, indistinguishable from a real result. If the cause
is a timeout rather than true non-convergence, raise `--ltspice-timeout` (new) or lower
`--workers`; the same plumbing gap existed in `preflight.py`, now `--render-timeout` there.

### `measure_truncation.py` — measure BDF2 truncation error, pick `oversample`

```bash
./measure_truncation.py --input examples/T3K-sweep-v3.wav --config path/to/device-config.toml
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

**Typical generation time** scales with circuit complexity, `--oversample`, and backend —
there's no universal number, but two real measured points: a simple pedal-scale circuit
renders a permutation in low single-digit seconds under the default `livespice` backend,
while the EVH 5150 Lead Full (sag) — a 6-stage preamp cascade + 4-tube power amp with
whole-amp sag — took **~75 min for a full permutation-batch** on the same backend. The
`--timeout-mult` flag's per-permutation timeout ceiling is built around `oversample 8`
costing up to **40x realtime** (100s of audio → 4000s) for the hardest corners, since some
real operating points (e.g. a very low-gain setting) are a genuine 20-40x slower than that
circuit's own typical corner without ever actually diverging.

**`--backend ngspice` can be dramatically slower than the above, or fail to converge at
all** — it's the adaptive-timestep tradeoff; see **Backends**, below, for the measured
numbers (a real 2.7x-slower case and a real total-non-convergence case).

### `tools/render_ngspice_deck.py` — render a hand-written ngspice deck's knob sweep

```bash
python tools/render_ngspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \
    --module gen_ocd_ngspice --probe-node OUT \
    in.wav --grid Gain=0,0.5,1 Tone=0,1 Volume=1.0 --outdir caps/
```

For a device whose circuit is a hand-written ngspice netlist (`gen_<device>_ngspice.py`, kept
in a private devices repo, exposing `build_deck()`/`KNOB_NAMES`) rather than a `.schx` — the
same situation `--backend ngspice-deck` addresses on `preflight.py`/`prepare_excitation.py`
below, and for the same reason: the `.schx` format itself has no component for whatever the
circuit needs (a real MOSFET, most often) or a topology LiveSPICE's fixed-timestep solver
can't hold at all. `--pedal-dir`/`--module` point at the private module the same way those two
tools do; `--probe-node` and `--ok-max-peak` (a render whose peak exceeds this is treated as
diverged, default 50V) are device-specific and worth passing explicitly rather than trusting a
default that happened to fit one device.

Writes `caps/cap_0000.wav ...` + `caps/manifest.jsonl` + `caps/mapping.csv` (an exact
filename→knob-value table for `gen_dataset_from_captures.py --mapping-csv`, sidestepping that
tool's filename-token convention entirely, since these knob values are already known exactly,
not encoded in the filename). **`--absolute`** renders a file already built at the
`V0dBFS=1V` convention (e.g. `tools/build_excitation.py`/`tools/prepare_excitation.py
--backend ngspice-deck` output) using its own sample values directly as volts, with no
peak-rescaling — necessary because those files intentionally have different segments at
different absolute drive levels, which a normal `--vin` rescale would flatten.

Uses `ngspice_spicelib.py` directly, not `render_backends.py`'s `NgspiceBackend` adapter:
that adapter returns raw audio arrays for `preflight.py`/`find_saturation_point.py`'s own
metric computation — a different contract than the capture-files-plus-manifest this tool
produces. Adding a new device means writing `gen_<device>_ngspice.py`, not another copy of
this render harness too — it used to be one per device (`render_ocd.py`, `render_bd2.py`,
etc., each a near-identical ~100-line copy differing only in the hardcoded module and a
couple of device-specific defaults) until this consolidated them.

### `tools/render_ltspice_deck.py` — render an LTspice deck's knob sweep

```bash
python tools/render_ltspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \
    --module gen_ocd_ltspice --tap spk \
    in.wav --grid Gain=0,0.5,1 Tone=0,1 Volume=1.0 --outdir caps/
```

LTspice counterpart of `render_ngspice_deck.py` above, for a device whose ngspice-deck
version can't converge on real playing content **at all**, independent of `maxstep` — found
on the Fulltone OCD, whose ideal tanh-bounded op-amp B-source is a genuine Newton-solver
dead end in ngspice (70/70 renders across the full knob grid timed out, at every `maxstep`
from 3e-6 down to 3e-8). LTspice gets past it with a real op-amp macromodel plus explicit
`.ic`/`uic` initial-condition hints on the `.tran` line — neither is available through
ngspice's B-source style. `--pedal-dir`/`--module` point at the same `build_deck()`/
`KNOB_NAMES` module convention as the ngspice-deck tools; `--tap` (not `--probe-node`) names
the node to render, matching `render_backends.py`'s `LtspiceBackend`. `--out-scale` scales
the tap down before LTspice's `.wave` writes it (LTspice's wav output is
+/-1V-PCM-bounded) and `render_grid`/`render_one` divide it back out — raise it toward 1.0
only if the tap is known to never exceed `1/out_scale` volts. `--timeout` defaults to
scaling with the input's own duration rather than a flat number (see
`tools/ltspice_spicelib.py`'s `render_grid` docstring) — a flat timeout tuned for a short
probe silently kills every job partway through a full excitation capture, which looks
exactly like a genuine convergence failure.

Same `manifest.jsonl`/`mapping.csv`/`--absolute` output contract as `render_ngspice_deck.py`.

#### Two ways to build a deck

A `gen_<device>_ltspice.py` module exposes `build_deck()` + `KNOB_NAMES`; where its netlist comes
from is up to it, and there are two established patterns:

- **Hand-written** (`gen_ocd_ltspice.py`) — necessary when the circuit has no faithful `.schx` at
  all, e.g. the OCD's real 2N7000 MOSFET clipping, which LiveSPICE has no component for. The
  netlist is authored directly and is the only description of that circuit.
- **Derived from the `.schx`** (`gen_joyo_ltspice.py`) — for a device that *does* have an audited
  `.schx` which LiveSPICE simply cannot solve correctly. It imports the component list from the
  `.schx`'s own generator (`from gen_joyo_american import NET`) and translates only the syntax,
  so values, nets and topology have exactly one source. **Prefer this whenever a `.schx` exists.**
  A hand-copied netlist is precisely the drift `parametric-devices/tools/check_generator_drift.py`
  exists to catch, and it would go unnoticed here because nothing compares two backends
  automatically. Two requirements: the `.schx` generator must guard its file write behind
  `if __name__ == '__main__'` so importing it has no side effects, and any pot taper must mirror
  `schx_to_ngspice.adjust_wipe` so a taper change cannot silently mean two different circuits on
  two backends.

**macOS: the LTspice you install decides whether any of this works** — the Homebrew cask and
the current `LTspice_26.pkg` are Wine-wrapped and their batch mode silently produces nothing.
See [LTspice on macOS](#ltspice-on-macos-only-for---backend-ltspice-deck) under Build &
Dependencies before debugging a deck that "won't converge".

Not only for ngspice-can't-converge cases: the **Joyo American Sound**
(`gen_joyo_ltspice.py`) is here because *LiveSPICE* is the backend that cannot render it.
`IdealOpAmp` cannot saturate — it has no supply rails at all — and on this pedal rail saturation
is the dominant nonlinearity at hot settings, so the output reaches **228x full scale** at
ordinary knob settings — while LiveSPICE's non-ideal
`Circuit.OpAmp` makes the circuit too stiff for its fixed-step solver (diverges even at
`oversample=128`, and `measure_truncation.py` reports STALLS). See
`parametric-devices/pedals/Joyo American Sound.md`.

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
  for a NAM-recognized standard sweep — `T3K-sweep-v3.wav` IS one, so calibration is real on
  the default input; a non-standard excitation falls back to a disclosed `delay=0`, same as
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
  `.schx` pipeline — defaults to `examples/T3K-sweep-v3.wav`), producing exactly one
  permutation per file.
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
- **`--lora-rank N`** — **DISABLED (2026-08-27).** Requesting a non-zero rank exits with an
  error; set `PARAMETRIC_NAM_ALLOW_LORA=1` to override for an ablation. See the status note in
  [LoRA-style knob conditioning](#lora-style-knob-conditioning). Adds a low-rank ("LoRA-style") knob-conditioned
  weight update to every layer's `l1x1`, alongside FiLM rather than instead of it — see
  that section for the design rationale, config format, and cross-repo requirement. Rank is recoverable from the
  checkpoint's own state-dict shape, so `export_checkpoint.py`/`param_infer.py` never
  need it re-specified.
- **`--cycle-checkpoints`** (default on; `--no-cycle-checkpoints` to disable) saves a
  full resumable snapshot (`<ckpt-dir>/cycle_<epoch>.pt`) at every SGDR cycle boundary
  during open-ended (`--epochs 0`) training, alongside the usual `best*.pt`/`latest.pt`
  (which are overwritten in place and don't preserve history). Lets you go back and
  compare a model's behavior at an earlier point in training — e.g. to narrow down when
  an instability (see [Known issue](#known-issue-rare-knob-corner-blowup-filmleakyrelu-runaway)
  below) first appeared, which `best.pt` alone can't answer since it only ever holds the
  single best-so-far snapshot.
- **Typical training time** depends on dataset size, `--crop-len`/`--batch-size`, and
  hardware — no universal number, but one real measured example: a two-tier (4ch/8ch)
  model at `--crop-len 48000 --batch-size 64 --repeats 32` on Apple Silicon (MPS backend)
  ran **~100-120s/epoch** (median ~104s across 295 epochs of one real run). Open-ended
  (`--epochs 0`) SGDR runs commonly take **several hours across many restart cycles**
  before `--stale-cycles` triggers auto-stop — that same run improved its all-time-best
  ESR on both tiers as late as its 6th 50-epoch cycle (~9 hours of wall time in), so don't
  read an early plateau as convergence.

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

### Choosing one

| symptom | backend |
|---|---|
| nothing wrong — it converges and the output is physically plausible | **livespice** |
| diverges, or needs extreme `oversample` | **ngspice** (`.schx`-native, no deck needed) |
| converges but the output is **impossible** (bigger than the supply rails allow) | **ltspice-deck** / **ngspice** — see below |
| ngspice can't converge on real playing content at any `maxstep` | **ltspice-deck** |

The third row is the one that costs you a training run, because it is **silent**: divergence
announces itself, wrongness does not. Two questions catch it, and neither is a knob sweep —
knob sweeps are *relative* (dB vs centre) and cannot see a circuit that is uniformly too loud:

1. **Is the absolute output physically possible?** A 9 V pedal cannot output 200 V. Probe absolute
   node voltages at the hot corners, not transfer ratios.
2. **Does the model include the nonlinearity that actually dominates there?** If the real circuit
   spends its time clipping against its rails, a model that cannot saturate is not approximately
   right — it is unbounded.

**LiveSPICE** (`--backend livespice`, default) — real-time-capable, `.schx`-native,
reference tube models, uniform audio-rate output, never aborts. The right default for
essentially everything. But read "never aborts" as a *risk*, not only a feature: its
`IdealOpAmp` has no supply rails and cannot saturate, so a circuit whose dominant nonlinearity
is op-amp clipping renders happily and wrongly. Measured on the Joyo American Sound: **228x full
scale** at ordinary knob settings, converging cleanly, passing preflight, with every knob
responding in the correct direction. `measure_truncation.py` reported *textbook* convergence for
it — truncation error tells you how well you solved the equations you wrote, never whether they
were the right equations. See `parametric-devices/backends.toml`, which exists to record exactly
these cases.

**ngspice** (`--backend ngspice`, experimental) — an offline real-SPICE backend with
adaptive timestepping, for the handful of **stiff / very-high-gain** circuits (e.g. the
EVH 5150 Lead full) where LiveSPICE's fixed-step solver diverges or needs extreme
oversampling. It translates any `.schx` to an ngspice netlist (exact Dempwolf–Zölzer /
Koren tube models, E+F ideal transformer) and feeds the input via an XSPICE filesource.
Tuning knobs for stiff amps: `--koren`, `--ot-damp`, `--ot-snub`, `--nfb-comp`.
See **[`ngspice/README.md`](ngspice/README.md)** for usage, the convergence findings,
and the important fidelity caveats.

**This safety comes at a real, sometimes severe, speed cost** — adaptive timestepping
means ngspice's solve time grows with the circuit's actual stiffness rather than staying
fixed, and on a genuinely stiff circuit that growth can be dramatic: measured ~2.7x slower
than LTspice on the Fender 5E3 at hard drive (step count exploding to 45492 vs LTspice's
12543 for the same clip), and on the EVH 5150 specifically, ngspice **failed to converge
on a hard-drive render at all** — aborting within microseconds regardless of timestep,
integration method, or input upsampling — while a hand-converted LTspice netlist of the
same circuit (in the separate `ltspice-batch` repo) rendered the same drive/pot position in
~32s. (There is no *generic* schx-to-LTspice translator here, but a deck can still be derived
from a `.schx` rather than hand-written — see "Two ways to build a deck" below.) Don't assume a
slow or stuck ngspice render will eventually finish; time-box it and compare against
`livespice` (or, for a hand-written-deck device, `ltspice-deck` below) before spending a
long timeout budget on it.

**Don't confuse this with `preflight.py`/`prepare_excitation.py --backend ngspice-deck`** — a
different flag on different tools, for a different situation. The `--backend ngspice` above
still describes the circuit as a `.schx` file, just solves it with ngspice instead of
LiveSPICE. `ngspice-deck` is for a device that has **no `.schx` at all** — a hand-written
ngspice netlist (`gen_ocd_ngspice.py` and siblings, kept in a private devices repo), typically
because the circuit needs something `.schx` has no component for (a real MOSFET) or a feedback
loop LiveSPICE's fixed-timestep solver can't hold at all, not even with `--backend ngspice`'s
translation.

**LTspice** (`--backend ltspice-deck` on `preflight.py`/`prepare_excitation.py`/
`grid_adequacy.py`/`check_transient_coverage.py`, or `tools/render_ltspice_deck.py`
directly) — the same deck-module situation as `ngspice-deck`. Two circumstances call for it:
a device whose ngspice deck can't converge on real playing content **at all**, independent of
timestep; and a device where every backend converges but LTspice is the one whose answer is
*stable* (see below).
Found on the Fulltone OCD: its ideal tanh-bounded op-amp B-source is a genuine
Newton-solver dead end in ngspice (70/70 renders across the full knob grid timed out, at
every `maxstep` from 3e-6 down to 3e-8), while LTspice gets past it with a real op-amp
macromodel and explicit `.ic`/`uic` initial-condition hints unavailable through ngspice's
B-source style — see `tools/ltspice_spicelib.py`'s module docstring for the full
investigation.

**Also worth reaching for when ngspice *does* converge.** On the Joyo American Sound both SPICE
backends bound the output correctly and agree to 1.3% at the clipping corners, and LTspice was
still chosen: ngspice's answers kept moving with settle time where LTspice's did not (centre
0.858 -> 1.137 V going from 0.8 s to 2.5 s of settle, versus 0.9597 -> 0.9590), and it was 2-3x
slower at the hard corners (22.5 s vs 7.1 s at all-max). *A result that changes when you lengthen
the settle has not converged*, whatever the solver reports.

Needs the native LTspice XVII app (`~/Applications/LTspice.app` or `/Applications/LTspice.app`),
not a SPICE binary — and on macOS **which build you install decides whether batch mode works at
all**: see [LTspice on macOS](#ltspice-on-macos-only-for---backend-ltspice-deck).

`render_backends.py` is the adapter layer `preflight.py` and `prepare_excitation.py` share
across all three hand-deck/schx splits (`NgspiceBackend`, `LtspiceBackend`) — see its module
docstring for the two-method contract a backend implements.

## Known issue: rare knob-corner blowup (FiLM/LeakyReLU runaway)

Trained `.schx` models can develop a **narrow, catastrophic instability** at a specific
knob-grid corner — the model's prediction spikes to tens or hundreds of times its normal
peak level on real transient content, while every aggregate metric (val ESR, per-tier
loss curves) looks fine, because the corner is a single cell out of hundreds-to-thousands
and contributes almost nothing to the training loss. It has recurred independently on at
least three shipped/prototype models (the pre-fix Boss DS-1, Tweed 5F6-A Full sag's
FiLM-only 5-knob release, and Tweed 5F6-A Full sag's FiLM+LoRA 2-knob prototype) at
different knob combinations, so treat it as a real, recurring failure mode of this
pipeline, not a one-off — and not specific to either conditioning mechanism. Whether
LoRA's extra per-layer capacity makes the failure *worse* once it occurs (more room for
an unconstrained input region to produce an extreme wrong answer) versus FiLM alone is a
real, plausible hypothesis, not yet confirmed: a same-corner probe against a FiLM-only
model found only ~1.1× elevation where the FiLM+LoRA case showed ~7×, but on a different
grid/config, so it's suggestive rather than a controlled comparison.

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

**Mitigations, all already in this repo**:
- **`tools/check_transient_coverage.py`** (pre-training gate, run automatically by
  `gen_dataset_from_schx.py` unless `--skip-transient-check`) — checks that the
  excitation's transient content actually reaches each corner's own saturation onset,
  across the **full min/max hypercube** (every swept knob independently at its own grid
  min or max, not just one knob varied from center) plus the traditional solo-knob
  corners. A corner whose real-playing content never crosses into saturation during
  training is a corner the network has to extrapolate at inference time. Supports
  `backend = "ngspice-deck"` or `"ltspice-deck"` in the config (same
  `pedal-dir`/`module`/`probe-node` convention as
  `preflight.py`/`prepare_excitation.py`/`grid_adequacy.py`) for a device
  with no `.schx` at all — `gen_dataset_from_schx.py`'s own automatic call is
  livespice-only, since that's the only path it drives; for a hand-deck device, call
  `check_transient_coverage.py` directly as its own gate before generating.
- **`tools/scan_film_runaway.py`** (post-training, run by hand against a finished
  `.param.nam`) — replays a real reference clip through every tier at every corner
  (`--config` for the exact trained grid, not just the reduced set) and flags any window
  whose peak is anomalous relative to that model's own typical output. Scanning the
  Tweed fp32 model with a **generic** guitar reference came back clean even including
  this exact corner — the spike only showed up against the **actual training excitation**
  at that permutation, so when investigating a suspected corner, prefer `--reference
  <the training sweep.wav>` over an arbitrary clip; a scan that doesn't reproduce the
  triggering content can give a false sense of safety.
- **`tools/build_excitation.py --synth-burst-peaks`** (excitation-design fix, added after
  diagnosing the Tweed FiLM+LoRA case) — `check_transient_coverage.py` only checks that
  the excitation's peak *level* reaches saturation onset per corner; it says nothing
  about *shape*. The Tweed FiLM+LoRA blowup happened at a trained grid point (not a gap
  needing extrapolation) whose excitation had moderate-level content and separately had
  high-crest-factor (sharp-transient) content, but never both together at the same
  time — that exact (level, shape) combination was simply never in the loss, so nothing
  constrained the model's behavior there. `--synth-burst-peaks` inserts one synthesized,
  deterministic, license-free broadband transient burst (`_transient_burst()`, crest≈8.5,
  instant attack + exponential decay) at every `--sweep-peaks` level, closing that gap
  directly. Verified end-to-end on the Tweed 2-knob retrain: `scan_film_runaway.py` came
  back clean across the full grid **at full training convergence** (not just an early
  checkpoint — the original instability itself only emerged well into training, so a
  clean early scan alone doesn't prove a fix held).

If a corner gets flagged, first check whether it's a *level* problem
(`check_transient_coverage.py`, fixed by raising `--realistic-peak`/`--sweep-peaks`) or a
*shape* problem (a real reference clip flags it but the excitation's own peak clears
onset fine — fixed by `--synth-burst-peaks`, not by more level). Don't just retrain
longer at the same excitation — the blowup in the Tweed case was a spike lasting well
under a second inside a single permutation's clip, so it barely moves that permutation's
own loss, let alone the average across hundreds of permutations; more of the same data
without closing the actual coverage gap is unlikely to fix it. If neither excitation fix
applies, narrowing the grid's swept range, adding an explicit loss weight on
transient/peak error, or excluding that exact corner combination and documenting it as an
unsupported setting remain the fallback levers.

---

## Known issue: excitation needs a silent lead-in (cold-start settling transient)

Every render starts a `.tran` from a cold, all-capacitors-at-0V initial condition, not the
already-biased-up state a real, already-powered-on device is always in. For a circuit with a
slow-charging DC-blocking network — a large output-coupling capacitor into a high-value pot,
for instance — real content starting at t=0 captures a genuine but non-representative
multi-second "circuit powering on" transient instead of the device's true steady behavior.

**Concrete evidence (Fulltone OCD, ngspice backend — `gen_ocd_ngspice.py`/`render_ocd.py` in
parametric-devices)**: `C10` (10µF) into the Volume pot's low leg (`RVOL2`, up to 500kΩ) gives
an RC time constant of roughly **5 seconds**. A sustained 200Hz test tone with no lead-in,
rendered from t=0, showed output RMS *slowly drifting* for about 15 seconds and then making an
**abrupt jump to a different, higher steady value at ~16 seconds** — neither of which a real
pedal, always already running, would ever do. This wasn't a settling-window artifact of a short
probe: a direct 20-second render showed the same two-phase drift-then-jump behavior end to end.
Prepending 3 seconds of silence before any real content resolved it completely: every corner
tested (across both `Gain` and `Tone` extremes) showed the tone snapping to a single stable,
unchanging RMS/peak within about a second of starting, with zero drift for the remainder of a
20+ second render. A quick way to rule this out on a new circuit: render a sustained tone with
no lead-in for at least 20-30 seconds (not just 1-2) and check whether RMS/peak in 1-second
windows is still changing well after the excitation's actual attack transient should have
settled — a circuit without a slow DC-blocking network won't show this, but there's no way to
know that in advance without checking, since the RC time constant is set by both output-stage
values, not something knob positions expose directly.

**Fix, applied by default**: `tools/build_excitation.py --lead-silence-s` (default `3.0`)
prepends that many seconds of true silence before the `--input`-derived real-playing segment.
The silent segment is written into the excitation file itself (not stripped after generation),
so it also gives the DC-blocking network real settling time before the file's `real` segment
dynamics are what training actually samples. Set to `0` to disable for a circuit already
verified not to need it. This is a cheap, non-device-specific fix — prefer it over adding
explicit `.ic` initial-condition statements to a specific device's deck, which would need to be
hand-derived and re-verified per circuit.

The same fix is also needed in `tools/preflight.py --backend ngspice-deck` and
`tools/find_saturation_point.py`'s own amplitude sweep (both take `--lead-silence-s`, default
`3.0`) — a short knob-direction probe with no lead-in hits the identical cold-start transient,
and it isn't just noisy, it can flip the answer: OCD's Tone knob probed REVERSED at a 5-second
probe with no lead-in and correctly OK once probed with a lead-in (or a long-enough probe to
outlast the transient on its own). See the next "Known issue" entry for the other, independent
fix this same investigation needed.

**Caveat**: 3 seconds was sufficient for the one circuit (OCD) this was diagnosed and verified
against, not derived from the RC time constant analytically for every possible circuit. A
circuit with a slower network (larger coupling cap and/or higher-value downstream resistance)
could need longer — when in doubt, run the sustained-tone check above with the circuit's own
component values before trusting the default.

---

## Known issue: preflight's EQ-knob checks can be swamped by the circuit's own gain control

`tools/preflight.py`'s dead/reversed-knob check holds every non-tested knob at a flat `0.5`
center. For a low-to-moderate-gain circuit that's fine, but for a genuinely high-gain device
(Fulltone OCD's stages run up to ~50dB combined), `0.5` on the Gain/Drive knob can already be
deep into saturation — and hard clipping SWAMPS a passive tone stack's real effect, since the
extra content an EQ knob lets through just becomes more clipping mush, not more clean signal at
that band. This reads as a **false REVERSED (or DEAD) direction on a knob that's actually
correct**, and it isn't just about probe input level — it's the circuit's OWN gain control
self-saturating regardless of how quiet the input is.

**Concrete evidence (Fulltone OCD, `--backend ngspice-deck`)**: with Gain held at the default
`0.5` center, Tone probed REVERSED (`-11.6%`) at a 10-second native-level probe — this is the
SAME Tone pot already verified correct on the LiveSPICE side (`+627%` rising) and independently
re-verified correct here once the fix below was applied. Confirmed the mechanism directly: a
manual amplitude sweep at fixed Gain=0.5 showed Tone's measured direction flipping between
REVERSED, DEAD, and OK depending purely on probe input level (`1.0V` → reversed, `0.3V` → OK
`+875%`, `0.1V` → dead, `0.05V` → OK `+80%`) — a clean signature of clipping-swamp, not a real
circuit fault.

**Fix, applied by default**: `--eq-check-drive-level` (default `0.1`). While checking an
EQ/tone knob's direction, every OTHER knob `classify()` tags as `"drive"` (Gain, Distortion,
Overdrive, etc. — the same name-based classification the direction check already used) is held
at this LOW value instead of the usual `0.5`, so the circuit isn't self-saturating from its own
gain control while the tone stack's real effect is being measured. This alone fixed OCD's Tone
reading at every probe duration tested, with no need for a reduced input level at all.
Volume/level-classified knobs are untouched — they're normally a post-clipping-stage
attenuator, so they don't change whether the nonlinear stage itself saturates.

**Complementary, not a replacement**: `--find-peak`/`--clean-probe-peak` (input-level scaling,
probing an EQ knob at both a driven and a linear-region level, pass-if-either) stays on by
default too — it catches the same failure mode via a different knob than the one under test,
and via a knob `classify()` can't identify as `"drive"` by name. Both mitigations were verified
independently on OCD; neither alone was assumed sufficient going forward.

## Known issue: audible knob-move transient noise (inference-side, not a training/dataset bug)

**This is a C++ inference-runtime issue (`NeuralAmpModelerCore`'s `ParametricA2FastModel`/
`ParametricWaveNet`, consumed downstream by e.g. Chainsmith FX), not something in this repo's
own training or dataset-generation pipeline** — documented here since anyone shipping a model
trained by `parametric-nam` can hit it. User report: with a silent input (no instrument
plugged in), moving a parametric model's knob visibly lights up the output peak meters — not
loud, but real and reproducible.

**Root cause: dilated-convolution ring-buffer state inconsistency across a FiLM conditioning
change, not DC offset.** The fast inference path's causal convolutions keep a history/ring
buffer of past layer activations, computed under whatever FiLM gamma/beta was active *when
they were computed* (FiLM is applied inside each layer, not before/after the stack). When a
knob changes, new incoming samples get the new conditioning while the ring buffer still holds
activations computed under the old conditioning — every layer's convolution then mixes old-
and new-conditioned taps in the same kernel window, a computation that matches neither steady
state, until the receptive field's worth of new samples flushes the stale history out (measured
at 6346 samples ≈ 12.4 buffers @ 512/48kHz, identical across channel counts since it's a
property of the shared kernel/dilation schedule, not any one model's weights). Two earlier,
more obvious hypotheses were ruled out first: a knob-position-dependent static DC offset
(measured 30-100x too small to explain the transient) and undersmoothed knob interpolation (a
single discrete knob move responds well to slower smoothing, ~4x quieter at 100ms vs. 20ms;
a fast continuous drag barely improves even at a full second of smoothing, since it re-targets
the smoother every buffer instead of letting it lag).

**Status: root-caused, not fixed.** The 100ms knob-smoothing time constant is kept (helps a
single deliberate knob move, ~zero cost). The theoretically correct fix — keep a rolling window
of raw input audio and re-run it through the full stack under the new conditioning when the
knob changes (throttled, since a full receptive-field forward pass every buffer during a fast
drag would blow a real-time audio callback's budget) — is identified but not implemented. Fast
continuous-drag noise remains audible. See internal engineering notes
("Knob-move transient noise investigation") for the full investigation, including the ruled-out
hypotheses and the measurements behind each.

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

### LoRA-style knob conditioning

> **STATUS: DISABLED (2026-08-27) — has not paid off on shipped ESR.** `--lora-rank > 0` now
> exits with an error in both `param_train.py` and `run_pipeline.py`; set
> `PARAMETRIC_NAM_ALLOW_LORA=1` to override for an ablation. **Only new training is gated —
> loading, exporting, folding and inferring existing LoRA models is untouched**, so the two
> archived LoRA bundles keep working (verified: the JCM800 LoRA bundle still renders and
> responds to knobs through `render_parametric`).
>
> The mechanism works and the Tweed numbers below are real, but on the one device where both
> were tried the FiLM-only bundle is the better shipped model, and no width-matched ablation has
> ever been run. Treat `--lora-rank > 0` as a research setting, not a default, and
> do not ship a release run on it without evaluating that run on its own terms.
>
> **What the archive actually contains** (`parametric-nam-models`, as of 2026-08-27): 7 LoRA
> `.nam` files, but only **two runs on two devices**, both `rank 4` —
> `tweed-5f6-a-full-sag/2026-08-17_e1cb0e0` (2-knob) and
> `marshall-jcm800-2203-preamp-and-power-amp-sag/2026-08-21_e1cb0e0` (2-knob).
>
> * **No clean head-to-head is archived.** The −72 %/−65 % figures below come from a FiLM-only
>   2-knob Tweed *testbed* run that was never published, so they cannot be re-checked from the
>   models repo. The only published FiLM-only Tweed runs are **5-knob** — a different and much
>   harder problem, not comparable.
> * **On shipped ESR — what an end user actually gets — FiLM-only is ahead.** JCM800, same 2
>   knobs, same 143-perm grid, same excitation:
>
>   | | tiers | final ESR lite / full |
>   |---|---|---|
>   | `2026-08-21` FiLM+LoRA | 4ch + 8ch | 0.06186 / 0.02403 |
>   | `2026-08-26` FiLM only | 5ch + 9ch | **0.06016 / 0.02191** |
>
>   The FiLM-only bundle is the better model and is the one marked `CURRENT`. That is the
>   comparison that matters for choosing what to ship, and it does not favour LoRA.
>
>   *Method caveat, for anyone using this to judge LoRA itself rather than to pick a model:* the
>   LoRA arm stopped at epoch 3980 (best 3945) while still descending steeply — 0.06717 @3400 →
>   0.06294 @3800 → 0.06186 @3945 — where the FiLM arm was flat across that span and only
>   dropped at ~4600, 868 epochs later. At matched epoch 3980 the LoRA arm was marginally
>   *ahead* on both tiers (0.06186/0.02403 vs 0.06455/0.02440), with narrower tiers. So this
>   ranks two **runs**, not two **methods** — there is no converged LoRA model here to ship even
>   if you wanted one.
>
>   For the same reason, don't read the older `2026-07-30` run's better-looking 0.0421 / 0.0153
>   as a third data point: it is a 90-permutation grid on a different excitation, so it covers
>   less of the knob space and is an easier fit, not a better product.
> * **A known bad interaction, not yet bounded.** LoRA's extra per-layer capacity makes FiLM
>   runaway *worse* once it occurs — ~7× excitation elevation on the FiLM+LoRA case against
>   ~1.1× for a FiLM-only model (see the FiLM-runaway discussion above). The mitigations there
>   (`--spectral-norm`, etc.) have not been re-validated with LoRA enabled.
> * **Fast-path support is recent.** LoRA was excluded from the C++ fast path until
>   NeuralAmpModelerCore `cc0a4a8` (2026-08-17) added it, verified bit-for-bit against the
>   generic reference across channel widths 3–8; `35456c9` then folded the correction into
>   `l1x1`'s weight matrix rather than a per-frame GEMV. Real, measured, and profiled in-app —
>   but only weeks old, and it requires a core at or past those commits.
> * **Thin exercise of the release plumbing.** The schema_version 1→2 bump, `fold_lora()`, and
>   the export path have been exercised on very few real checkpoints — a state-dict loading bug
>   in `release_run.sh` survived until an actual LoRA checkpoint first hit it on 2026-08-21.
>
> **Practical default: prefer widening a tier over enabling LoRA.** It is the remedy for the
> same capacity ceiling, it is what the current best JCM800 bundle actually used, and it needs
> no schema bump, no `rank` to choose, and no minimum core version. Revisit if the
> width-matched ablation (above, run to convergence on both arms) says otherwise.

**Why, beyond FiLM.** FiLM only ever applies `gamma(cond)*x + beta(cond)` — a diagonal
per-channel affine rescale of one fixed backbone computation. As swept knob count grows
past ~3, the space of tonal behaviors the backbone must represent as "one shared
computation, rescaled per knob setting" outgrows what fixed weights can encode — measured
concretely on the Tweed 5F6-A Full (sag) 5-knob run, where the lite (4ch) tier plateaued
at median ESR 0.126 on the full grid (not shippable) while the architecturally-identical
full (8ch) tier reached 0.025. `--lora-rank` adds a genuine per-knob weight *delta* —
`W_effective = W_l1x1 + A(cond)@B(cond)` — computed as two small matmuls
(`A @ (B @ x)`, never materializing the full `channels × channels` delta) rather than a
scale on a fixed computation. `A`/`B` are produced by a linear map of `cond`, mirroring
FiLM's own `nn.Linear`, so the result is smooth in the knob value by construction — more
expressive than FiLM, without the unconstrained-arbitrary-weights problem a full
hypernetwork would have. Zero-init on `net_A` means LoRA starts as a true no-op (delta ≡
0), so enabling it never perturbs an otherwise-converged FiLM baseline at step 0.

**Additive, not a replacement.** FiLM and LoRA are independently toggleable
(`--lora-rank 0` = FiLM only, unchanged from before LoRA existed) and, when both are on,
compose — LoRA is not a mode that turns FiLM off. Measured on the Tweed 5F6-A Full (sag)
2-knob testbed (81→121-perm grid across two runs): FiLM+LoRA cut lite-tier ESR from
FiLM-only's ~0.146 to ~0.040 (-72%) and full-tier from ~0.034 to ~0.012 (-65%), matching
what the capacity-ceiling diagnosis predicted.

**Config format** — `config.parametric` gains an optional `lora` object, and
`schema_version` bumps from 1 to 2 (only for LoRA-enabled exports; a non-LoRA export
still declares 1, byte-for-byte unchanged from before LoRA existed):
```json
"parametric": {
  "type": "film+lora",
  "schema_version": 2,
  "condition_size": 2,
  "lora": {"rank": 4},
  "film_layers": ["layer_0", "layer_1", ...],
  "parameters": [...]
}
```
`"type"` is `"film"` (or absent, for every export before this field existed) when
`--lora-rank 0`, `"film+lora"` otherwise. Weight layout per layer gains an optional LoRA
block immediately after `film.bias` when active, in lockstep with `weight_count()`/
`_export_weight_block()`/`_load_weight_block()` (`param_train.py`) — all three of which
must stay in sync exactly like the existing `film` branch does.

**Requires the LoRA-aware NeuralAmpModelerCore fork** — same requirement `"ParametricWaveNet"`
itself already has. A LoRA-tagged model is safe to hand to an *older* build of this fork
that predates LoRA: `require_supported_parametric_model()` rejects `schema_version 2`
loudly (`"...is newer than this build supports..."`) rather than silently misreading it,
the same fail-loud contract every schema bump in this project follows. **`ParametricA2FastModel` (the hand-optimized C++ fast path) supports LoRA as of
NeuralAmpModelerCore `cc0a4a8` (2026-08-17)** — per-layer additive low-rank correction on
`layer1x1`'s output, matching `NAM/lora.h`'s `Process()`, with the dispatch detector widened to
accept `type=="film+lora"` (`schema_version` 2, `lora.rank > 0`) and gated so a malformed config
still falls back to generic. Verified bit-for-bit against the generic reference across all
supported channel widths (3–8), including the shipped `rank==channels` edge case. `35456c9`
then folded the correction into `l1x1`'s weight matrix instead of a per-frame GEMV.

*This paragraph previously said the fast path had no LoRA support "and never will" — that was
true when written (the exclusion is `f9d4701`) and is now stale; the core's own README was
corrected in `b3a08e7`.* **A `"film+lora"` model on a core older than `cc0a4a8` still falls
through to the generic path**, so the performance you get depends on which core you link.
`is_parametric_a2_shape()` gates on `"type"`: `"film"` (or absent) takes the fast path
regardless, so a FiLM-only model built today runs identically, at the same performance, whether
or not the LoRA code exists in the binary at all.

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

### LTspice on macOS (only for `--backend ltspice-deck`)

**Install the native build from analog.com — NOT `brew install --cask ltspice`, and NOT the
current `LTspice_26.pkg`.** Both of those are the *Windows* binary wrapped in CrossOver/Wine.
The GUI launches and looks fine, which is what makes this worth writing down: `-b` batch mode
then produces **no `.raw`, no `.log`, no output of any kind, and exit status 0** — even on a
five-line RC netlist. There is nothing to debug against, and it looks exactly like a circuit
that failed to converge. Verified 2026-08-27 against LTspice 26.0.2.1 (`NSPrincipalClass =
CXApplication` in its `Info.plist` is the giveaway; a Wine-backed bottle also shows up as
`~/Library/Application Support/LTspice/Bottles/`).

The working install is **LTspice XVII 17.2.4**, ADI's last native macOS release ("Download for
MacOS 10.15 and forward"), a universal arm64/x86_64 binary. It runs correctly on macOS 26.

Three requirements, each of which silently breaks it if missed:

1. **The bundle must be named exactly `LTspice.app`.** It hardcodes its own name internally.
   Renamed — even to something as close as `LTspice17.app` — every launch dies at startup with
   `NSInvalidArgumentException ... 'data parameter is nil'` from `newJSONValue`, before it
   reads your netlist. The crash names JSON, so it reads like a corrupt install or a failed
   update check; it is neither.
2. **`~/Library/Application Support/LTspice/lib/sub/` must exist**, holding
   `UniversalOpAmp2.lib` and friends. The installer's `postinstall` unpacks `lib.zip` there.
   `gen_ocd_ltspice.py` and `gen_joyo_ltspice.py` reference that path directly.
3. **If a newer `LTspice.app` is already in `/Applications`, the 17.2.4 installer will not
   replace it** — macOS Installer refuses to overwrite a bundle carrying a higher
   `CFBundleVersion`, so 26.0.2.1 blocks it. It fails *silently*: the receipt registers
   (`pkgutil --pkg-info com.analog.LTspice` reports 17.2.4) and `lib.zip` lands, but the app
   is untouched. Check what you actually have before trusting the receipt:

```bash
plutil -p /Applications/LTspice.app/Contents/Info.plist | grep -E "ShortVersion|PrincipalClass"
#   want: "17.2.4"     and  NSPrincipalClass => "LTapplication"
#   wrong: "26.0.2.1"  and  NSPrincipalClass => "CXApplication"   <- Wine, batch mode is dead
```

If 26 is in the way and you want to keep it, install 17.2.4 to `~/Applications/LTspice.app`
instead — the name is what matters, not the directory, and
`ltspice_spicelib._find_ltspice_bin()` searches `~/Applications` **first**, then
`/Applications`, then honours an explicit `LTSPICE_BIN` env var override. Extract the app from
the pkg without running the installer:

```bash
pkgutil --expand-full ~/Downloads/LTspice.pkg /tmp/ltpkg17
ditto /tmp/ltpkg17/LTspice.pkg/Payload/Applications/LTspice.app ~/Applications/LTspice.app
xattr -dr com.apple.quarantine ~/Applications/LTspice.app
```

Smoke-test batch mode before blaming a circuit — this must write `rc.raw` and `rc.log`:

```bash
printf '* rc\nV1 in 0 SINE(0 1 1k)\nR1 in out 1k\nC1 out 0 100n\n.tran 0 5m 0 10u\n.end\n' > /tmp/rc.net
~/Applications/LTspice.app/Contents/MacOS/LTspice -b /tmp/rc.net && ls /tmp/rc.raw /tmp/rc.log
```

Then render a real device end-to-end (`gen_joyo_ltspice` is the cheapest known-good check):

```bash
python tools/render_ltspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \
    --module gen_joyo_ltspice --tap spk --absolute --out-scale 0.1 tone.wav out.wav \
    --knob Voice=1 --knob Drive=1 --knob Bass=1 --knob Treble=1 --knob Mids=1 --knob Level=1
# -> "peak=3.99V OK".  A Joyo peak in the hundreds means LiveSPICE, not LTspice (see below).
```

One gotcha that is not LTspice's fault: `.net` (or `.cir`) files must be plain SPICE netlists,
and LTspice **requires braces** around B-source expressions (`B1 a b V={min(...)}`), unlike
ngspice where they are optional.

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
