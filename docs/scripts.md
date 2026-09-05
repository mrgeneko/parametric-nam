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

> For the end-to-end path from a new `.schx` to a trained model, see
> [new-circuit-walkthrough.md](new-circuit-walkthrough.md). This page is the per-script
> reference.

## `--workspace` — one directory per run

```bash
./prepare_excitation.py --config <device>.config.toml --workspace ~/runs/duke_run1 ...
./run_pipeline.py       --config <device>.config.toml --workspace ~/runs/duke_run1
```

A single training run produces files from four scripts, and before `--workspace` each needed
its own path flag with nothing checking that the five agreed:

```
~/runs/duke_run1/
├── config.toml                 the config this run was built from (stamped on first use)
├── pipeline.log                --log
├── excitation/
│   ├── excitation.wav          --output of prepare_excitation.py
│   └── excitation.recipe.json  sidecar; consumers derive it from the wav's own path
├── dataset/                    --dataset-dir
│   ├── outputs.npy  params.csv  config.json  sweep.wav
│   └── NOTICE_RESTRICTED_INPUT.md
├── checkpoints/                --checkpoint-dir
│   ├── best.pt  best_lite.pt  latest.pt  cycle_*.pt
│   └── metrics.csv  watchdog.log  per_combo_esr_w*.csv
├── duke_run1.param.nam         --nam-output (named after the workspace, not "model")
└── release/                    --release-dir
    ├── *.param.nam  <device>.schx  metrics.csv
    ├── dataset_config.json  dataset_params.csv
    └── ESR_RECORD.md  MANIFEST.md  reproduce.sh
```

**Any individual flag still wins.** `--workspace` only fills paths left unset, so
`--dataset-dir /Volumes/SSD1/duke_ds` still parks the big file elsewhere (Duke's `outputs.npy`
is 7.99 GB). A path set in the config counts as explicit too.

**A workspace is per RUN, not per circuit** — so a second run goes in a second directory and
the first survives intact. To keep that true, `run_pipeline.py` stamps `config.toml` into the
workspace on first use and **refuses** to run a *different* config into it, pointing you at a
new `--workspace` (`--force-workspace` overwrites in place, and re-stamps, so the guard stays
armed). Re-running the *same* config is not blocked: that is a resume after a crash, and
generation already skips when `outputs.npy` exists.

**Not in the workspace, deliberately:**

| | where | why |
|---|---|---|
| content-keyed caches | `~/.cache/parametric-nam/{findpeak,gridadq}` | keyed on circuit bytes and clip audio, not paths — so they stay valid across runs and workspaces. Per-workspace caches would re-render every probe for a re-run of the same circuit. Safe to delete at any time. |
| scratch | `$TMPDIR/parametric-nam-*` | deleted on exit; `--keep-scratch` keeps it |
| the device config + recipe | `parametric-devices/…` | the committed record — input to a run, not output of one |

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

**Probe renders are cached on disk** (`~/.cache/parametric-nam/gridadq`), so the second run
of this tool on a device is near-instant instead of re-rendering everything the first one
already rendered. That second run is not hypothetical: `--apply` refines the grid, and
`run_pipeline.py`'s STEP 1 then re-verifies it, so the normal path runs `grid_adequacy`
twice on the same circuit (see [checklist](checklist.md)).

The cache is keyed on everything that changes a probe's **answer** — the `.schx`'s own
bytes, the knob values, `[fixed]`, oversample/iterations, and the **audio** of the probe
clips. Editing a resistor or rebuilding the excitation therefore invalidates it even though
both keep the same file path, which is the case a path-keyed or mtime-keyed cache would get
wrong (`prepare_excitation.py` overwrites the wav in place). Copying a circuit to a new path
correctly *hits*.

The knob grid is deliberately **not** in the key, and needs no change detection: the cache
is per probe point, so editing an axis simply asks for different keys — surviving points
hit, newly-inserted ones render. Adding one value to a 4-knob pedal's Gain axis costs ~7 s
rather than the full ~28 s.

Failed renders are never cached, so one transient non-convergence can't become permanent.
The directory is LRU-pruned to 8 GB (`GRIDADQ_CACHE_MAX_BYTES`); `--no-disk-cache` bypasses
it entirely.

`backend = "ngspice-deck"` (or `"ltspice-deck"`) in the config renders through a
hand-written ngspice/LTspice deck (`pedal-dir`/`module`/`probe-node` in the TOML, same
convention as `preflight.py`/`prepare_excitation.py` — see [Backends](backends.md)) instead of a
`.schx`, for a device whose clipping needs a real component `.schx` has no model for (a
MOSFET, a real BJT) — or, for `ltspice-deck` specifically, a circuit whose ngspice deck
can't converge on real playing content at all (see [Backends](backends.md)).

**Exit status — an over-target cell is not a failure.** It exits **0** even with cells over
target, because refining one costs renders and combinations and a config may rationally accept a
coarse cell, coarsen an axis, or fix a knob outright instead of paying. That is a priced
trade-off, not a defect, and gating on it would turn a judgement call into an error.

What *does* fail the run (**exit 2**) is a **failed render**, because that is the case that
produces a WRONG table rather than a weaker one: every cell is computed from whichever probes
survived, and nothing else in the output says so. Measured on one real pedal config, 38 of 48
probes timed out and the printed table put one cell at 0.6578 (18.8× over) where the clean
re-run measured 0.0475 (1.4×) — a 14× error, indistinguishable from a real result. If the cause
is a timeout rather than true non-convergence, raise `--ltspice-timeout` (new) or lower
`--workers`; the same plumbing gap existed in `preflight.py`, now `--render-timeout` there.

## `stability_sweep.py` — which knob combinations does the SOLVER fail on?

```bash
./stability_sweep.py --config device.toml --probe-s 3 --oversample 8,16,32
./stability_sweep.py --config device.toml --shard 0-11/48      # one worker's slice
```

Four tools already gate a dataset and none checks solver stability across the grid:
`preflight.py` (knob dead/reversed/level), `check_transient_coverage.py` (does the excitation
reach saturation), [`grid_adequacy.py`](#grid_adequacypy--measure-whether-a-knob-grid-is-dense-enough)
(per-cell interpolation error), `measure_truncation.py` (BDF2 truncation vs oversample). Newton
stability was only ever discovered DURING a render, hours in, by the solver-spike detector.

Always uses `--no-retry`: the question is whether the CONFIGURED oversample is stable, not
whether the ladder can rescue it — a rescued render still cost you the failed attempt. Reports
the unstable COMBINATIONS, not just a count, so a region is visible; exit status is nonzero when
nothing is stable.

**A short probe is enough, which is what makes it affordable.** On the Mesa Dual Rectifier the
observed spikes clustered at 0.93–1.05 s against a 120.5 s clip — the first transient after the
excitation's 1 s lead silence, where the solver settles to quiescence and is then hit with
signal. A 3 s probe provokes the same failure at 1/40th the cost:

| | 448 combinations |
|---|---|
| 3 s probe, 4 machines | ~0.2 h |
| 8 s probe, 4 machines | ~0.4 h |
| the real render, 120.5 s | ~7 h |

Worked result — 36 of 448 combinations tripped the retry ladder mid-render, 12 all the way to
oversample 64. Every one had the same signature: `RD Gain <= 0.25` with `Red Master = 1.0`. The
device's own note said "converges at --oversample 8", true of the 4 corners it had tested. See
[`docs/backend-comparison-mesa.md`](backend-comparison-mesa.md) for what that turned into.

## `holdout_esr.py` — did the model LEARN the knob space, or memorise it?

```bash
# preferred: derive the trained cells from the training dataset itself
./holdout_esr.py --checkpoint ckpt/best.pt --dataset full_27cell_ds \
    --trained-dataset corners_8cell_ds

# or classify by value: a cell is "trained" when EVERY knob sits on one of these
./holdout_esr.py --checkpoint ckpt/best.pt --dataset full_27cell_ds \
    --trained-values 0.25,0.75  -o holdout.csv
```

Training's own val ESR is a random slice of the **same** combinations the model trained on, so a
model that fits its training settings and interpolates badly still scores well.
`per_combo_esr.py` shares the blind spot — per-cell, but only over trained cells. The only way to
measure interpolation is to score cells the model has never seen: render a denser grid than you
train on, train on a subset, score the remainder.

Complements [`grid_adequacy.py`](#grid_adequacypy--measure-whether-a-knob-grid-is-dense-enough),
which asks the same question **before** training with no model in the loop (circuit truth at a
cell midpoint vs interpolating its neighbours). This asks it **after**, with the model, so the
answer includes the network's own capacity limits and not just the grid's. Use `grid_adequacy` to
choose a grid; use this to check the choice was right.

Measured on the Joyo American Sound (2026-08-30) — full 3³ EQ grid rendered, trained on the 8
corners of `[0.25, 0.75]³`, scored on the 19 interior cells:

| w9 | mean ESR |
|---|---:|
| TRAINED corners | 0.00817 |
| HELD-OUT interior | 0.02547 (3.1× worse) |
| dead-centre (0.5,0.5,0.5) | 0.04002 (4.9× worse) |

Two points per axis gave accurate end stops and a mushy middle — and the middle is where tone
controls get set. That run's training val ESR was **0.00299**, which shows none of it.

**A large ratio is not automatically audible.** Decompose before acting: on that run 54% of the
dead-centre error was a static level/tone offset (+0.79 dB), the kind a player dials out with a
small knob nudge. ESR scores a benign 0.8 dB tilt the same as 0.8 dB of wrong harmonics.


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

## `prepare_excitation.py` — size an excitation from measured saturation onset, automatically

Closes the manual gap `build_excitation.py` above leaves: someone has to read a saturation-onset
number by hand and pick `--realistic-peak`/`--sweep-peaks` themselves — literally how every
existing device's excitation was sized before this tool existed (e.g. the non-midpoint-default pedal's excitation,
peak-sized from a direct output-V-vs-input-V sweep at one hand-picked knob setting).

Runs `find_saturation_point.py` at every corner of the knob grid — all-min/all-max/center/
solo-extreme/full-hypercube, the same set `check_transient_coverage.py` below checks against —
**plus `--sample-grid N` points drawn from the full grid product**, takes the **worst-case
(highest) onset** across them, and derives both levels from it.

Corners alone are not sufficient *in principle*: onset is **not monotonic** in the knobs, so its
maximum over the grid need not sit at a vertex. Measured on a five-knob amp channel — its true
worst onset (23.177 V, at Bass=min with every *other* knob at its **centre** grid value) is
**1.27x** the highest of all 32 hypercube vertices. Probing the whole grid would guarantee it and
is not worth it: 648 points at a measured ~1.2/min is ~9 h per channel, before every render, to
choose one scalar. `scaffold_config.py` passes a knob-count-scaled budget (~1.5x the corner
count, capped at 64) so the default path measures the interior rather than assuming headroom
covers it.

- `--sweep-peaks` max = `--margin × worst-case onset` (default margin **2.0x** — past the onset,
  not just at it), staged as fractions of that (`--sweep-peak-fracs`, default `0.25,0.5,0.75,1.0`).
- `--realistic-peak` = `--realistic-peak-frac × worst-case onset` (default frac **1.3**). Never
  go *below* 1.0: `check_transient_coverage.py`'s default margin requires the transient content
  to reach the worst corner's own onset, and the worst corner's onset *is* worst-case onset by
  definition, so any fraction below 1.0 guarantees that check fails there. It is 1.3 rather than
  a bare 1.0 because the hazard is not sitting *on* the boundary but the **measured** worst being
  below the **true** worst (see the non-monotonicity note above); 1.3 covers the observed
  vertex-to-true-worst ratio from vertex data alone, at no extra probing cost. It is insurance,
  not a substitute for `--sample-grid` — a hotter excitation drives every already-covered corner
  further into saturation, so do not inflate it past what the non-monotonicity demands.

Then invokes `build_excitation.py` with the derived levels. Refuses to build if any corner's
onset can't be determined, rather than silently building against a partial result (same
"refuse to guess" convention as `preflight.py`/`check_transient_coverage.py`).

```bash
# livespice:
python prepare_excitation.py --backend livespice \
    --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml \
    --real-clip examples/T3K-sweep-v3.wav --output ~/work/tmp/DEVICE_excitation.wav

# ngspice-deck:
python prepare_excitation.py --backend ngspice-deck \
    --pedal-dir ~/work/parametric-devices/pedals --module gen_device_ngspice \
    --range "Gain=0.1,0.5,0.9" --range "Tone=0.2,0.5,0.8" --fixed-params "Volume=1.0" \
    --real-clip ~/work/parametric-devices/pedals/device_realistic_clip.wav \
    --output ~/work/tmp/device_excitation.wav
```

**`--peak-max-v`** (default 40, the sweep ceiling `find_saturation_point` hunts within) needs to
sit well above the true onset or you read a false onset off your own probe's ceiling instead of
the real plateau — caught directly on the MOSFET-clipping pedal: `--peak-max-v 10` reported a suspicious
9.947V onset (right at its own ceiling); re-measuring at `--peak-max-v 25` gave the real,
comfortably-interior 7.836V. 40V suits an amp; use something like 3–5V for a small pedal circuit.

Running `check_transient_coverage.py` afterward is still worth doing as an independent gate — a
clean result is *expected* (this tool derives its levels from the same onset numbers that check
verifies against) but **not guaranteed**: both are sampling a non-monotonic function, so the
grid's true worst corner need not be one either of them probed. Two real amp channels failed such
a check (6/43 and 4/43 corners) against an excitation whose peak had been hand-picked rather than
measured at all.

## `check_transient_coverage.py` — gate: does the excitation reach saturation everywhere?

Pre-generation gate answering a narrower, more dangerous question than "does the excitation have
enough peak somewhere": does the excitation's **transient-bearing** content — the `--input`
segment placed at `--realistic-peak`, not just the sweep tail — actually reach saturation at
**every** knob-grid corner?

> That segment is **not** necessarily real playing. The standard capture sweep normally passed
> as `--input` is itself synthesized (frequency sweep + noise-staircase + calibration blips);
> its ~22 dB crest factor comes from that structure, not from musical dynamics. A genuine
> playing recording works equally well but is not required. See `build_excitation.py`'s
> docstring. What matters to this check is only that it is the crest-bearing part.

This is not hypothetical — it's the exact failure this tool was built to catch. The tweed-style amp's
`--realistic-peak` had been chosen for input-signal realism, not cross-checked against the
measured onset, so that `--input` content stayed in the *linear* region at every corner tested
while only the sweep (a smooth tone, no attack shape) crossed into saturation there. The network
never saw a transient and saturation together at that corner, and ran open-loop when a real one
eventually arrived.

For every corner (same set `prepare_excitation.py` uses above) **and the same `--sample-grid`
interior points**, finds that corner's own saturation onset (`find_saturation_point.py`) and
compares it against the excitation's transient peak. Exit status is nonzero if any corner fails,
so a generation script can gate on it.

The interior budget defaults to the one `scaffold_config.py` hands `prepare_excitation.py`, from
a single shared definition, so **the gate is never weaker than the sizing that produced the
excitation it is checking**. It probed corners only until 2026-09-04, which meant an excitation
sized against 104 points was re-checked against 40 — a verifier weaker than its own input reads
as confirmation while checking less. `gen_dataset_from_schx.py` runs this same gate before
generating (`--transient-sample-grid` overrides the budget, `0` restores corners-only).

```bash
python check_transient_coverage.py --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml \
    [--transient-peak 0.2] [--margin 1.0] [--oversample 8] [--iterations 256] \
    [--json report.json] [--no-cache] \
    && python gen_dataset_from_schx.py ...
```

`--transient-peak` (the excitation's transient-bearing `--input`-segment peak, in volts at
V0dBFS=1) is
auto-read from the excitation's `<stem>.recipe.json` sidecar (`build_excitation.py`'s
`args.realistic_peak`) if present; otherwise it's **required** — this tool refuses to guess it
from the raw audio rather than silently mis-slicing the file's realistic/sweep boundary.
`--margin` (default 1.0) is how far past each corner's own onset the transient peak must reach.
For a hand-written ngspice deck with no `.schx` at all, `--backend ngspice-deck` takes the same
`[knobs]`/`[fixed]`/pedal-dir/module/probe-node TOML convention as `preflight.py`/
`prepare_excitation.py`.

## `check_input_headroom.py` — warn if the excitation doesn't get loud enough (runs automatically)

Runs automatically as `run_pipeline.py`'s **Step 2**, right after grid adequacy — the only one
of the excitation-coverage scripts on this page that's wired into the pipeline by default.
Answers a narrower, cheaper question than `check_transient_coverage.py` above: at *default*
(0.5) knob settings, does the excitation's peak reach this device's own measured saturation
onset (`preflight.py --find-peak`)?

A dense knob grid (`grid_adequacy.py`) is a completely different question from an excitation
that gets loud enough — a model can only learn what's in the data, and a perfectly-dense grid
rendered entirely in the device's linear region still teaches it nothing about saturation. This
gap went unnoticed on TAD Blackface 85 Reverb (2026-07-31) until someone asked "was this
checked?" after the config was otherwise fully measured — nothing else in the pipeline would
have caught it.

```bash
python check_input_headroom.py --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml
python check_input_headroom.py --config ~/work/parametric-nam-models/pedals/DEVICE/config.toml --margin 0.9
```

**WARN, not a gate** — unlike grid adequacy (a provable floor, and `run_pipeline.py` aborts on
it), a low ratio here can be a real gap *or* a genuine high-headroom device: `--find-peak`
probes at default knob settings, not the grid's own hottest corner, so a knob that changes how
hard the excitation drives later stages can have a much lower onset at the grid's extreme than
at default. TAD Blackface itself is the false-positive case — it turned out to have real, dense
nonlinear behavior concentrated at **low** Volume rather than at high input level. Treat a WARN
as "go check the grid's hot corner directly," not an automatic verdict. `run_pipeline.py
--skip-headroom-check` skips it; `--headroom-margin` (default 0.8) controls the threshold.

For the full, every-corner, hard-gate version of this same question, use
`prepare_excitation.py`/`check_transient_coverage.py` above instead — this script is a cheap
automatic tripwire, not a substitute for them.

## `distribute_pull.py` — hand rendering chunks out as workers free up

**Takes the same `--config` as `run_pipeline.py`.** One description of a device, whether it
renders on one machine or four:

```bash
# single machine
python run_pipeline.py     --config device.config.toml --workspace ~/runs/device_run1

# sharded across a fleet — same config, same device
python distribute_pull.py  --config device.config.toml \
    --worker host-a:~/work/parametric-nam:10 \
    --worker host-b:~/work/parametric-nam:10 \
    --chunks 41 --output '~/device_ds' --collect ~/runs/device_run1/dataset
```

It expands the config into the renderer's `--backend`/`--schx`/`--input`/`--knobs`/`--range`/
`--fixed-params`/`--oversample` using `run_pipeline.py`'s own loader, so the two paths cannot
drift. Anything after `--` is appended and wins, so `-- --oversample 4` still overrides the
config without editing it.

`schx` and `input` are rewritten **relative to the repo**, because each worker runs from its
own checkout and homes differ across a fleet (`/Users/gene`, `/Users/chewie`, `/home/gene`) —
an absolute path from the controller can be a *different user's* home on a worker. Keep the
device files in a sibling directory of the repo and this is automatic; the tool warns when a
path is too far outside to travel.

> Before this existed, the sharded path took eleven hand-written renderer flags. The Mesa RED
> launch omitted `--backend` — whose default is `cpp` — and all four workers quarantined in
> under a second.

`distribute_gen.sh` splits the grid **once**, up front, by core count or `--weights`, and each
worker keeps its slice. That only works when every worker's throughput is known in advance *and
stays constant*. On a 648-combination run neither held: one Linux box managed 43.8 combos/hr and
finished early then idled, while a laptop sharing the machine with an unrelated training run
managed 7.6 and still had 8.9 h left — a thing no core count could have predicted. Static
sharding turned a 1.6 h job into an 8.9 h one.

This cuts the grid into chunks expressed as ordinary `--shard i-i/N` specs and dispatches one at
a time to whichever worker is free, so a fast worker simply takes more. No throughput estimate is
needed anywhere: the schedule *is* the measurement.

```bash
python distribute_pull.py \
    --worker host0:/path/to/parametric-nam:8 \
    --worker host1:/path/to/parametric-nam:10:DOTNET_ROOT=$HOME/.dotnet \
    --chunks 31 --collect ~/my_dataset \
    -- --backend livespice --schx "..." --input "..." --range "Gain=..." ...
```

Four things that are easy to get wrong:

- **A *slow* worker is invisible unless you watch combinations, not chunks.** The scheduler
  streams each worker's output and parses the renderer's own per-combination progress line, so
  it knows every worker's **seconds per combination** — the one unit comparable across
  machines. A worker that goes `--slow-mult` (default 3) times the *fleet median* with nothing
  completed has its chunk abandoned and requeued elsewhere. The baseline is a **median** over
  the whole fleet, so a single pathological host cannot raise the bar it is judged against, and
  it does not exist at all until `--slow-min-samples` (4) combinations have finished anywhere —
  a cold fleet is not evidence about any host. `--slow-floor-min` (10) keeps a fast fleet from
  killing a machine over ordinary variance.

  This is not the same guard as the renderer's own. `gen_dataset_from_schx.py` has a stall
  detector that deliberately tolerates a **slow but progressing** render up to
  `TOTAL_CEILING_MULT` (20×) its per-rung budget — on a full amp at oversample 8 that is
  **36.7 hours for one combination**. Measured on a Mesa Dual Rectifier run: a worker rendering
  at 54× real-time needed ~148 min per combination against a 110 min budget, so every render
  progressed, nothing stalled, nothing failed, and nothing printed. It held a chunk for **7.5
  hours and produced zero combinations** while its neighbours finished a 16-combination chunk
  every 40 minutes. Per-combination pacing catches that in about an hour.

- **`--collect LOCAL_DIR` — use it.** The scheduler renders; gathering is the other half. `sig/`
  merges cleanly because its filenames are the **global** grid index, but `params.csv` is one
  file *per worker* holding only that worker's rows, so rsyncing each output dir onto one path
  leaves you the last worker's metadata describing the whole grid. `--collect` copies every
  worker's `params.csv` to a private name *before* any `sig/` transfer and writes the merged file
  *last*, which is correct even when a worker is localhost and its output dir **is** the merge
  target. (`gen_dataset_from_schx.py --combine` hard-fails on a row/`.npy` mismatch, so a botched
  merge fails loudly rather than training on a hole — but it is a backstop, not a plan.)
- **Pick `--chunks` coprime with your knob-axis sizes.** Modulo sharding steps by `chunks`, so an
  axis of cardinality `c` is *constant* inside every shard whenever `c` divides `chunks`. The
  dataset is unaffected — the union of shards is still the whole grid — but
  `gen_dataset_from_schx.py`'s own per-shard knob-sensitivity check goes blind and reports the
  frozen knob as `RMS varies only 0.00% — knob may have no effect`, which reads exactly like a
  dead knob or a `param_map` typo. A prime is the easy answer; the tool warns and suggests one.
- **A fast-failing worker is worse than a slow one.** It drains the queue faster than healthy
  machines can take work — one that could not import a dependency killed 27 of 31 chunks in ~70 s.
  `--quarantine-after N` (default 3) benches a worker after N consecutive failures with no
  successes, and a chunk is not handed back to a host that already failed it.

## `gen_dataset_from_schx.py` — generate the dataset

Simulates the circuit across knob combinations, one WAV per combination, then combines
them into `outputs.npy`.

```bash
# List the pots/switches in a schx:
python gen_dataset_from_schx.py --backend livespice --schx "<circuit>.schx"

# Grid sweep (factorial — fine for 1–3 knobs):
python gen_dataset_from_schx.py --backend livespice --schx "<circuit>.schx" \
    --knobs drive,tone --range "drive=0,0.5,1" --range "tone=0.2,0.5,0.8" \
    --input guitar.wav --output <ds> --dry-run   # check combo count + disk first

# Random sampling (for high knob counts — grids explode past ~3 knobs):
python gen_dataset_from_schx.py --backend livespice --schx "<amp>.schx" \
    --knobs gain,bass,mid,treble,master --random 200 \
    --bounds bass=0.15,0.85 --bounds mid=0.15,0.85 --bounds treble=0.15,0.85 \
    --input guitar.wav --output <ds>

# Combine per-combination WAVs into outputs.npy:
python gen_dataset_from_schx.py --combine <ds>
```

- **`--random N`** samples N points in the knob hypercube (+ deterministic boundary
  **anchors** so extremes are covered; `--no-anchors` to skip). FiLM interpolates
  between them — far better than a coarse grid for >3 knobs. Seeded (`--seed`) and
  extensible.
- **`--bounds KNOB=lo,hi`** restricts a knob's sampled range (e.g. keep a tone pot in
  its usable band, or a hot gain pot out of a divergent regime). Recorded in the
  `.param.nam` so the host maps the knob to the real trained domain.
- **`--max-crest`** (default 50) fails any combination whose output crest factor
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
renders a combination in low single-digit seconds under the default `livespice` backend,
while a high-gain amp head — a 6-stage preamp cascade + 4-tube power amp with whole-amp
sag — took **~75 min for a full combination-batch** on the same backend. The
`--timeout-mult` flag's per-combination timeout ceiling is built around `oversample 8`
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

`manifest.jsonl`'s per-line `v0dbfs` field records the reference voltage this batch actually
used (`1.0` under `--absolute`, else `--vin`) — copy it into `gen_dataset_from_captures.py
--v0dbfs` when assembling the dataset, so the exported `.nam`'s `input_level_dbu` gets
populated the same way a `.schx`'s own `V0dBFS` attribute already does. Omitted from the
dataset (not guessed) if you don't pass it.

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
  combination per file.
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
  matters for training a new ablation or for `export_checkpoint.py`/`checkpoint_infer.py` reading
  an old checkpoint (rank is recoverable from the checkpoint's own state-dict shape, so those never
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

## `checkpoint_infer.py` — inference from a training checkpoint (Python, no C++)

```bash
python checkpoint_infer.py --checkpoint <ckpt>/best.pt \
    --input dry.wav --output-dir out/ \
    --params "drive=0.5,tone=0.7" --params "drive=0.9,tone=0.3"
```
Loads a `.pt` training checkpoint and renders WAVs at arbitrary knob positions. Repeat
`--params` for multiple outputs. Knob names/order come from the dataset `config.json`.

See `nam_infer.py` for the equivalent that loads directly from an exported `.param.nam`
file instead — no checkpoint needed, e.g. for a released model you don't have the
training run for.

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
from — point it at a generated dataset's per-combination renders to eyeball the sweep
before training (`coverage_report.py` checks the *sampling*, this checks the *audio*), or
at a batch of `checkpoint_infer.py`/`nam_infer.py` outputs to eyeball a trained model's knob
response.

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
key that stock loaders ignore but a parametric-aware host (`NAMix`) reads for live
knobs — one file, safe to hand anyone. `--no-embed-parametric` drops that for a static-only file, e.g. for a capture pack
where embedding the full master in every tone just multiplies size.

## `plot_tone_response.py` — frequency-response chart for an exported `.nam`

```bash
python plot_tone_response.py \
    --model bundle.optimal.param.nam --dataset-config dataset/config.json \
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
have; override `RUN`/`DS`/`SCHX`/`CONFIG` to point at any recipe instead — continuing
straight from a run made by the
[new-circuit walkthrough](new-circuit-walkthrough.md), whose `--workspace` layout the
paths below assume:

```bash
RUN=~/runs/device_run1 DS=~/runs/device_run1/dataset \
SCHX="path/to/My Pedal.schx" \
CONFIG=my_pedal.config.toml \
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
