[← back to README](../README.md)

Full per-script reference for parametric-nam. See the main README's Quick Start and `run_pipeline.py` sections to get oriented first.

# Scripts

## `scaffold_config.py` — generate a starting `config.toml` for a new circuit

```bash
python scaffold_config.py --schx path/to/circuit.schx
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
`knob_classify.py` — the same heuristic `preflight.py` uses for its own direction and
EQ-swamp checks (see [that Known issue](known-issues.md#known-issue-preflights-eq-knob-checks-can-be-swamped-by-the-circuits-own-gain-control)).
A name that matches nothing is written commented-out as `UNCONFIRMED` rather than
silently defaulted or guessed wrong — **always review this table by hand**: a misclassified
knob doesn't just get the wrong sensitivity metric in `gen_dataset_from_schx.py`, it also
silently skips `preflight.py`'s role-aware EQ-check protection for every OTHER knob's
direction check. Override a bad guess with `preflight.py --knob-kind NAME=kind,...`, or just
edit the table before your first real run.

The `[knobs]` grid it writes is a **placeholder** (run `grid_adequacy.py --config
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

## `grid_adequacy.py` — measure whether a knob grid is dense enough

```bash
./grid_adequacy.py --config path/to/device-config.toml --target 0.009
./grid_adequacy.py --config ... --suggest       # propose a regrid, print it, stop
./grid_adequacy.py --config ... --apply         # iterate suggest+reverify, write the result
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
convention as `preflight.py`/`prepare_excitation.py` — see [Backends](backends.md)) instead of a
`.schx`, for a device whose clipping needs a real component `.schx` has no model for (a
MOSFET, a real BJT) — or, for `ltspice-deck` specifically, a circuit whose ngspice deck
can't converge on real playing content at all (see [Backends](backends.md)).

**Exit status — an over-target cell is not a failure.** It exits **0** even with cells over
target, because refining one costs renders and permutations and a config may rationally accept a
coarse cell, coarsen an axis, or fix a knob outright instead of paying. That is a priced
trade-off, not a defect, and gating on it would turn a judgement call into an error.

What *does* fail the run (**exit 2**) is a **failed render**, because that is the case that
produces a WRONG table rather than a weaker one: every cell is computed from whichever probes
survived, and nothing else in the output says so. Measured on one real pedal config, 38 of 48
probes timed out and the printed table put one cell at 0.6578 (18.8× over) where the clean
re-run measured 0.0475 (1.4×) — a 14× error, indistinguishable from a real result. If the cause
is a timeout rather than true non-convergence, raise `--ltspice-timeout` (new) or lower
`--workers`; the same plumbing gap existed in `preflight.py`, now `--render-timeout` there.

## `measure_truncation.py` — measure BDF2 truncation error, pick `oversample`

```bash
./measure_truncation.py --input examples/T3K-sweep-v3.wav --config path/to/device-config.toml
```

LiveSPICE integrates with BDF2 (O(h²)), so the simulated circuit is never quite the real
one — the gap is **truncation error**, and it shrinks as `--oversample` rises. Measures it
against a high-oversample reference (not the next rung up, which understates the error
~3x) so you can pick an `oversample` whose truncation sits comfortably below the model ESR
you're chasing — training past that point just teaches the model the integrator's own
mistakes.

## `build_excitation.py` — build a training excitation that covers the full input range

A sweep/DI file on its own (e.g. `T3K-sweep-v3.wav`) samples its own loud region essentially
never — a model trained on it alone never sees the device saturating, and goes
out-of-distribution the moment a hot input arrives. Build a proper training excitation from it:

```bash
python build_excitation.py --input examples/T3K-sweep-v3.wav \
    --output ~/work/tmp/DEVICE_excitation.wav \
    --realistic-peak <below the device's measured saturation onset> \
    --sweep-peaks <levels up to the device's max output + headroom>
```

That concatenates the `--input` clip with amplitude-stepped log sweeps that **do** reach
maximum output, plus a leading silence so the render is not sampling a cold-start transient.
Then gate it with `check_transient_coverage.py`, which fails if any knob corner's
transient content never reaches that corner's own saturation onset.

**Not called by `run_pipeline.py`** — it's a manual step you run once per device to produce
the excitation file that `--dataset-dir`'s generation step (`gen_dataset_from_schx.py`, via
`run_pipeline.py`'s `input` config setting) then consumes.

**Keep transients intact, whatever you use as `--input`.** Sharp-attack content — whether a
real-playing recording's pick attacks or a standard capture sweep's own built-in noise-staircase/
calibration-blip transients — is the excitation's source of real transient dynamics; thinning or
filtering it (e.g. to work around a solver convergence issue during dataset generation) removes
exactly what a model needs to learn a stable response to a sharp attack — a real shipped pedal
model once spiked to **12.39** peak on a real pick attack where the reference circuit produced
**0.46**, traced back to exactly this. If an excitation's transients are causing solver
divergence, fix the render (oversample, timestep, backend) rather than the input.

Note that measurements like truncation error or grid adequacy are properties of the circuit
**and the specific excitation used**, so they do not automatically carry over if you change
excitations — `measure_truncation.py` and `grid_adequacy.py` both measure through whatever
`input` is configured.

## `gen_dataset_from_schx.py` — generate the dataset

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
while a high-gain amp head — a 6-stage preamp cascade + 4-tube power amp with whole-amp
sag — took **~75 min for a full permutation-batch** on the same backend. The
`--timeout-mult` flag's per-permutation timeout ceiling is built around `oversample 8`
costing up to **40x realtime** (100s of audio → 4000s) for the hardest corners, since some
real operating points (e.g. a very low-gain setting) are a genuine 20-40x slower than that
circuit's own typical corner without ever actually diverging.

**`--backend ngspice` can be dramatically slower than the above, or fail to converge at
all** — it's the adaptive-timestep tradeoff; see [Backends](backends.md) for the measured
numbers (a real 2.7x-slower case and a real total-non-convergence case).

## `render_ngspice_deck.py` — render a hand-written ngspice deck's knob sweep

```bash
python render_ngspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \
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
`V0dBFS=1V` convention (e.g. `build_excitation.py`/`prepare_excitation.py
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

## `render_ltspice_deck.py` — render an LTspice deck's knob sweep

```bash
python render_ltspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \
    --module gen_ocd_ltspice --tap spk \
    in.wav --grid Gain=0,0.5,1 Tone=0,1 Volume=1.0 --outdir caps/
```

LTspice counterpart of `render_ngspice_deck.py` above, for a device whose ngspice-deck
version can't converge on real playing content **at all**, independent of `maxstep` — found
on one pedal, whose ideal tanh-bounded op-amp B-source is a genuine Newton-solver dead end in
ngspice (70/70 renders across the full knob grid timed out, at every `maxstep` from 3e-6 down
to 3e-8). LTspice gets past it with a real op-amp macromodel plus explicit
`.ic`/`uic` initial-condition hints on the `.tran` line — neither is available through
ngspice's B-source style. `--pedal-dir`/`--module` point at the same `build_deck()`/
`KNOB_NAMES` module convention as the ngspice-deck tools; `--tap` (not `--probe-node`) names
the node to render, matching `render_backends.py`'s `LtspiceBackend`. `--out-scale` scales
the tap down before LTspice's `.wave` writes it (LTspice's wav output is
+/-1V-PCM-bounded) and `render_grid`/`render_one` divide it back out — raise it toward 1.0
only if the tap is known to never exceed `1/out_scale` volts. `--timeout` defaults to
scaling with the input's own duration rather than a flat number (see
`ltspice_spicelib.py`'s `render_grid` docstring) — a flat timeout tuned for a short
probe silently kills every job partway through a full excitation capture, which looks
exactly like a genuine convergence failure.

Same `manifest.jsonl`/`mapping.csv`/`--absolute` output contract as `render_ngspice_deck.py`.

## Two ways to build a deck

A `gen_<device>_ltspice.py` module exposes `build_deck()` + `KNOB_NAMES`; where its netlist comes
from is up to it, and there are two established patterns:

- **Hand-written** — necessary when the circuit has no faithful `.schx` at all, e.g. a real
  MOSFET clipping stage which LiveSPICE has no component for. The netlist is authored directly
  and is the only description of that circuit.
- **Derived from the `.schx`** — for a device that *does* have an audited `.schx` which
  LiveSPICE simply cannot solve correctly. It imports the component list from the `.schx`'s own
  generator (e.g. `from gen_mydevice import NET`) and translates only the syntax, so values,
  nets and topology have exactly one source. **Prefer this whenever a `.schx` exists.**
  A hand-copied netlist is precisely the drift `parametric-devices/tools/check_generator_drift.py`
  exists to catch, and it would go unnoticed here because nothing compares two backends
  automatically. Two requirements: the `.schx` generator must guard its file write behind
  `if __name__ == '__main__'` so importing it has no side effects, and any pot taper must mirror
  `schx_to_ngspice.adjust_wipe` so a taper change cannot silently mean two different circuits on
  two backends.

**macOS: the LTspice you install decides whether any of this works** — the Homebrew cask and
the current `LTspice_26.pkg` are Wine-wrapped and their batch mode silently produces nothing.
See [LTspice on macOS](../README.md#ltspice-on-macos-only-for---backend-ltspice-deck) under Build &
Dependencies before debugging a deck that "won't converge".

Not only for ngspice-can't-converge cases: one real pedal is here because *LiveSPICE* is the
backend that cannot render it. `IdealOpAmp` cannot saturate — it has no supply rails at all —
and on this pedal rail saturation is the dominant nonlinearity at hot settings, so the output
reaches **228x full scale** at ordinary knob settings — while LiveSPICE's non-ideal
`Circuit.OpAmp` makes the circuit too stiff for its fixed-step solver (diverges even at
`oversample=128`, and `measure_truncation.py` reports STALLS).

## `gen_dataset_from_captures.py` — build a dataset from real hardware captures, no `.schx` needed

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
    --captures "~/Downloads/MyAmp DST *.nam" \
    --output /tmp/myamp_ds --gear-make "Manufacturer" --gear-model "Amp Model 15w"

python gen_dataset_from_captures.py --captures "~/Downloads/MyPedal *.wav" --output /tmp/mypedal_ds

python gen_dataset_from_captures.py --combine /tmp/myamp_ds   # same --combine flag as gen_dataset_from_schx.py
```

- **Filenames encode the knob settings**, same convention for either source kind.
  `"MyAmp DST G2, B5, M5, T5.nam"` → `Gain=0.2, Bass=0.5, Mids=0.5, Treble=0.5`, and identically
  for a `.wav` capture: `"MyPedal G7, B3.wav"` → `Gain=0.7, Bass=0.3` (comma-separated
  `PrefixDigit` tokens, digit → value/10, using the default prefix map's `G`/`B`/`M`/`T`/`Rvb`/
  `Rsn`/`Prsn` → `Gain`/`Bass`/`Mids`/`Treble`/`Reverb`/`Resonance`/`Presence`; unrecognized
  tokens like `"DST"` or a device name are silently ignored). Override the prefix→knob-name map
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

## `param_train.py` — train a parametric NAM

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
- **`--lora-rank N`** — **REMOVED (2026-08-27), Python side kept for reading old checkpoints
  only.** Requesting a non-zero rank exits with an error; set `PARAMETRIC_NAM_ALLOW_LORA=1` to
  override for an ablation. See the status note in
  [LoRA-style knob conditioning](reference.md#lora-style-knob-conditioning-removed) — the C++ that consumed
  LoRA-tagged models in a real-time plugin host has been deleted entirely; this flag only
  matters for training a new ablation or for `export_checkpoint.py`/`param_infer.py` reading an
  old checkpoint (rank is recoverable from the checkpoint's own state-dict shape, so those never
  need it re-specified).
- **`--cycle-checkpoints`** (default on; `--no-cycle-checkpoints` to disable) saves a
  full resumable snapshot (`<ckpt-dir>/cycle_<epoch>.pt`) at every SGDR cycle boundary
  during open-ended (`--epochs 0`) training, alongside the usual `best*.pt`/`latest.pt`
  (which are overwritten in place and don't preserve history). Lets you go back and
  compare a model's behavior at an earlier point in training — e.g. to narrow down when
  an instability (see [Known issue](known-issues.md#known-issue-rare-knob-corner-blowup-filmleakyrelu-runaway))
  first appeared, which `best.pt` alone can't answer since it only ever holds the
  single best-so-far snapshot.
- **Typical training time** depends on dataset size, `--crop-len`/`--batch-size`, and
  hardware — no universal number, but one real measured example: a two-tier (4ch/8ch)
  model at `--crop-len 48000 --batch-size 64 --repeats 32` on Apple Silicon (MPS backend)
  ran **~100-120s/epoch** (median ~104s across 295 epochs of one real run). Open-ended
  (`--epochs 0`) SGDR runs commonly take **several hours across many restart cycles**
  before `--stale-cycles` triggers auto-stop — that same run improved its all-time-best
  ESR on both tiers as late as its 6th 50-epoch cycle (~9 hours of wall time in), so don't
  read an early plateau as convergence.

## `param_infer.py` — inference (Python, no C++)

```bash
python param_infer.py --checkpoint <ckpt>/best.pt \
    --input dry.wav --output-dir out/ \
    --params "drive=0.5,tone=0.7" --params "drive=0.9,tone=0.3"
```
Loads a checkpoint and renders WAVs at arbitrary knob positions. Repeat `--params` for
multiple outputs. Knob names/order come from the dataset `config.json`.

## `coverage_report.py` — find holes in a sampled dataset

```bash
python coverage_report.py --dataset <ds> --suggest 5
```
Read-only analysis of `params.csv`: success/failure summary, **per-knob boundary
coverage**, 1-D histograms, pairwise 2-D occupancy, and nearest-neighbour **gap
analysis** (largest empty regions, optionally emitted as fill points). Pairs with the
seed-extensible sampler to grow a dataset toward its gaps.

## `sweep_report.py` — visualize a set of WAVs at different knob settings

```bash
python sweep_report.py /tmp/ds/*_gain_*.wav --param-name gain -o report.html
```

Self-contained HTML report from any set of WAVs, one per parameter setting (sorted by
name): overlaid waveforms, frequency spectra, and per-file stats (peak, RMS, RMS
monotonicity across the sweep, spectral centroid trend). Doesn't care where the WAVs came
from — point it at a generated dataset's per-permutation renders to eyeball the sweep
before training (`coverage_report.py` checks the *sampling*, this checks the *audio*), or
at a batch of `param_infer.py` outputs to eyeball a trained model's knob response.

## `export_checkpoint.py` — checkpoint → `.param.nam`

```bash
python export_checkpoint.py --checkpoint <ckpt>/latest.pt --dataset <ds> \
    --output model_final.param.nam --state model
```
Re-exports a NAM from any `.pt` (e.g. the final `latest.pt` weights) without retraining.

## `bake_nam.py` — freeze a knob setting for stock/upstream NAM plugins

```bash
python bake_nam.py --in model.param.nam --params "Gain=0.7,Tone=0.5" -o tone.nam
```

Standard NAM plugins have no runtime knob input, so a raw `.param.nam` handed to one just
throws (`"No config parser registered for architecture: ParametricWaveNet"`). This bakes
one specific knob setting into an ordinary static `.nam` instead — FiLM is affine over the
layer's linear ops, so freezing a setting folds exactly into `conv`/`mixin` weights, no
retraining, pure offline weight transform. Omitted knobs fall back to their own declared
default, not a blanket 0.5 (wrong for a circuit whose controls don't center at noon).

By default the output is **dual-payload**: the baked tone at the top level (so any stock
plugin plays it) plus the full original parametric model under an `embedded_parametric`
key that stock loaders ignore but a parametric-aware host (`NAMix`,
`NeuralAmpModelerPlugin`/"Anti-Static") reads for live knobs — one file, safe to hand
anyone. `--no-embed-parametric` drops that for a static-only file, e.g. for a capture pack
where embedding the full master in every tone just multiplies size.

## `plot_tone_response.py` — frequency-response chart for an exported `.nam`

```bash
python plot_tone_response.py \
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

## `release_run.sh` — verify + stage a finished run

Calls `plot_tone_response.py` above automatically to produce the release bundle's
`tone_response.svg`/`.md`. Its own defaults assume a private archive layout you don't
have; override `RUN`/`DS`/`SCHX`/`CONFIG` to point at any recipe instead, including the
bundled `examples/muff/` one — continuing straight from the
[Try it now](../README.md#try-it-now) run:

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
