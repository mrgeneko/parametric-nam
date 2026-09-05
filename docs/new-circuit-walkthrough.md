# From a new `.schx` to a trained `.param.nam`

The end-to-end path for a circuit this repo has never seen: **six steps, five commands**.
Step 2 is pure review — no command, and the one step whose omission the pipeline's own
gates cannot catch for you.

For what each script does in detail see [scripts.md](scripts.md); for the gates the pipeline
applies and what each one aborts on, [checklist.md](checklist.md).

Throughout, `~/runs/device_run1` is a **workspace** — one directory holding everything this
run produces. It is per *run*, not per circuit: a second run goes in a second directory, and
`run_pipeline.py` refuses to run a different config into an existing one. See
[`--workspace`](scripts.md#--workspace--one-directory-per-run).

---

## 1. Scaffold the config

**First, get an input file — the repo does not ship one.** `T3K-sweep-v3.wav` is the standard
TONE3000 capture sweep; download it from
**<https://www.tone3000.com/create/capture>** and put it in `examples/`:

```
parametric-nam/examples/T3K-sweep-v3.wav
```

You can substitute your own sweep or DI recording; a good one covers the top of the band and
contains sharp attacks rather than only steady tones. Whatever you use, pass its path as
`--input` below.

```bash
./scaffold_config.py --schx "My Pedal.schx" \
                     --input examples/T3K-sweep-v3.wav \
                     --output my_pedal.config.toml \
                     --write-backend-sidecar
```

Does four things you would otherwise do by hand:

- **Discovers every control** from the `.schx` (ganged pots collapsed to one knob).
- **Measures oversample** against a reference, rather than taking argparse's default of 2,
  which is too low for most real circuits.
- **Guesses each knob's kind** (`drive`/`hi`/`lo`/`mid`/`rms`) from its name.
- **Builds a first excitation** via `prepare_excitation.py` and points `input` at it.
  This one is **provisional** — it is sized against the placeholder grid written moments
  earlier, which step 3 is about to change. Step 4 replaces it. The scaffold says so when
  it finishes.

`--write-backend-sidecar` writes `<stem>.backends.toml` if the backend *diverges* while
measuring oversample. That verdict is a measurement — the scaffold just watched the circuit
fail — so it should not be hand-written. A circuit whose backend cannot converge is better
discovered here than four hours into a render.

## 2. Review the config — the step you cannot skip

This is the one step with no command and no automatic gate. Open the `config.toml` the
scaffold wrote and fix two things it cannot know.

Here is what it produces for a four-knob pedal, and what you change:

```toml
schx       = "/path/to/My Pedal.schx"
input      = "/path/to/my_pedal_excitation.wav"   # sized against this config's grid AS IT WAS ...
backend    = "livespice"
oversample = 8

[knobs]
Drive    = [0.0, 0.25, 0.5, 0.75, 1.0]
Level    = [0.0, 0.5, 1.0]
Tone     = [0.2, 0.5, 0.8]
Wobble   = [0.0, 0.5, 1.0]

[knob-kind]
Drive    = "drive"   # guessed from name (gain/distortion) -- verify
Tone     = "hi"      # guessed from name (treble/high-freq) -- verify
Level    = "rms"     # guessed from name (output level) -- verify
# Wobble = "UNCONFIRMED"   # name matched no known keyword -- classify by hand

[fixed]
Presence = 0.5       # pinned, not swept -- see "Pinning a control" below
```

**Fix 1 — knob order.** The scaffold writes `[knobs]` alphabetically. You want **signal
flow**: the order a signal actually passes through the circuit. Above, if `Tone` sits before
`Level` in the circuit, reorder the `[knobs]` lines to `Drive, Tone, Level, Wobble`. The
order is recorded in the dataset and baked into the model, so changing it later means
re-rendering everything.

**Fix 2 — knob kinds.** Every guess is name-based and marked `-- verify`; anything the
scaffold could not match is **commented out** as `UNCONFIRMED`, which means it is not
classified at all until you uncomment and set it. The valid values:

| kind | means | effect |
|---|---|---|
| `"drive"` | a gain/distortion control | held **low** while another knob's EQ effect is checked |
| `"hi"` | treble / high-frequency | checked by per-band spectral spread; held low during another EQ knob's check |
| `"lo"` | bass / low-frequency | same as `hi`, different band |
| `"mid"` | midrange | same as `hi`, different band |
| `"rms"` | a volume/level control | checked by output-level spread |

Omitting a knob entirely is legal — it falls back to a plain RMS-spread check with no
special preflight treatment. That is the right choice for a control that is genuinely none
of the above (`Wobble`, a modulation depth); it is the *wrong* choice for one you simply did
not get around to.

Getting this wrong is quiet, not loud: a misclassified knob gets the wrong sensitivity
metric in `grid_adequacy.py` **and** skips `preflight.py`'s role-aware EQ safeguard, so a
genuinely dead control can pass every downstream gate.

**Also check the `[knobs]` ranges.** EQ knobs are narrowed to `[0.2, 0.8]` rather than the
full `[0.0, 1.0]` — that is the scaffold's own role-aware default (a tone control at a hard
endpoint is rarely a useful training point), not something `grid_adequacy` chose. Widen it
if your circuit is meant to be played there.

### Pinning a control at one value

Not every control has to become a model knob. Each control lands in exactly one of three
states:

| where you put it | what happens | in the trained model? |
|---|---|---|
| `[knobs]` | swept across the listed values | **yes** — a knob you can turn |
| `[fixed]` | held at the value you state, every render | no |
| neither | held at whatever the `.schx` itself defaults to | no |

To pin one, add a `[fixed]` table naming the control and its value in the same 0–1 units the
`[knobs]` lists use:

```toml
[fixed]
Presence = 0.5
Rock     = 1.0
```

Put a control in `[fixed]` **or** `[knobs]`, never both — the two are concatenated into one
parameter string per render, so listing a name twice is ambiguous rather than clever.

Two reasons to pin. The first is **cost**: combinations multiply, so dropping a 3-point axis
from a 5-knob grid cuts the render count — and the dataset size, and the training time — by
two thirds. The second is **scope**: a control that is really a mode switch, or one you have
no intention of modelling, adds a dimension the network has to spend capacity on for no
benefit. Devices with more than three swept knobs tend to hit a capacity ceiling rather than
a training-time one, so pinning is a real lever, not a shortcut.

The cost of pinning is that it makes the config a **variant** of the device rather than the
device — the resulting model cannot move that control at all. Say so in a comment, and
ideally in the config's filename, so nobody later mistakes it for the full device.

> There is also a `[defaults]` table, which is a different thing and easy to confuse with
> `[fixed]`. It applies to knobs that **are** swept, and records each one's default position
> in the `.nam`'s `parameters[].default`, so that baking a static model with no explicit
> parameter values uses the circuit's real default position rather than the midpoint of the
> range — for a pedal whose controls do not center at noon. It does not remove a knob from
> the grid.

## 3. Refine the knob grid

```bash
./grid_adequacy.py --config my_pedal.config.toml --apply
```

Renders each cell's midpoint and compares it against interpolating its neighbours. A cell
whose residual exceeds the target is a floor no amount of training can lift — the
information was never sampled. `--apply` bisects failing cells, re-probes, and writes the
converged grid back, leaving all your comments intact.

Probe renders are cached on disk, so step 6's re-verification of this grid is free.

> **`--apply` can also *coarsen* an axis**, down to a 2-point floor, when a knob's midpoints
> interpolate well. Think before accepting that on an interacting tone network. Measured
> error says the endpoints interpolate well *over one probe window*, which is not the same
> as the knob being linear everywhere in use — and a 2-point axis leaves nothing to catch it
> if it isn't. Restoring a midpoint by hand is cheap insurance; if you do, say so in a
> comment, because a later `--apply` will otherwise re-cut it.

## 4. Re-size the excitation against the refined grid

```bash
./prepare_excitation.py --backend livespice \
                        --config my_pedal.config.toml \
                        --real-clip examples/T3K-sweep-v3.wav \
                        --workspace ~/runs/device_run1
```

**Do not skip this because step 1 already built one.** Step 1's excitation was sized against
the *placeholder* grid. Step 3 changed the grid underneath it, which changes the corner set,
which changes the worst-case saturation onset the excitation has to reach.

**Why the excitation has to be scaled to the circuit.** A model can only learn behaviour the
training data contains. If the input never drives the circuit hard enough to clip at some
knob setting, that setting's clipping is simply absent from the dataset — and the model will
confidently produce a clean output where the real circuit distorts. So the excitation's
transient peak has to exceed the *worst-case* saturation onset across the whole grid, not
the typical one.

Two things make that worst case impossible to guess:

- **Onset is measured at the output.** A setting with the output level turned down needs far
  *more* input drive to reach saturation. The knob combinations that demand the hottest
  excitation are often the quiet-sounding ones.
- **Onset is not monotonic in the knobs**, so the worst corner need not be a hypercube
  vertex at all — it can sit in the interior, above every corner you would have thought to
  check. This is why `prepare_excitation.py` probes interior grid points as well as corners,
  and why a peak chosen by ear or by eye is not good enough.

A hand-picked peak that has never been measured against a corner set will look fine, train
without error, and quietly fail at whatever fraction of the grid it never reached.

The excitation lands at `~/runs/device_run1/excitation/excitation.wav`, its
`.recipe.json` sidecar beside it, and `input` in the config is updated to match. Keep the
pair together: every consumer finds the sidecar by deriving it from the wav's own path.

## 5. Verify coverage independently

```bash
./check_transient_coverage.py --config my_pedal.config.toml
```

**What it compares.** The excitation `build_excitation.py` produces is not one uniform
signal. It is a concatenation:

```
[ your --input clip, placed at --realistic-peak ]   crest-bearing, mostly low level
[ amplitude-stepped log sine sweeps ]               dense level x frequency coverage, loud
[ short fade-out to zero ]
```

The first segment carries the sharp attacks; the sweeps carry the loud, smooth content. This
tool takes the peak of that **first segment alone** (read automatically from the recipe
sidecar) and, for every knob setting it tests, measures that setting's own saturation onset
and checks the segment's peak clears it.

**Why that is not the same as "is the excitation loud enough".** The file's overall peak
comes from the sweeps, so it can clear saturation comfortably while the crest-bearing segment
never does. When that happens the training data contains sharp attacks — but only in the
clean, linear region — and saturation — but only from smooth swept tones. The two never
co-occur, so the model has no example of what this circuit does to *a hard attack that
clips*, and improvises when one arrives. Nothing about the overall peak, the file's RMS, or
the dataset's size reveals this.

**Which settings it tests.** Every structural corner of the knob grid, plus interior sample
points — because saturation onset is not monotonic in the knobs, so the worst case can sit
between corners rather than at one.

It exits nonzero if any setting fails, so it chains as a gate:

```bash
./check_transient_coverage.py --config my_pedal.config.toml && ./run_pipeline.py ...
```

**Per-combination ESR cannot substitute for this.** It is tempting to assume a coverage gap
would show up as bad ESR at those knob settings. It does not: validation data is drawn from
the same under-covered renders, so both the model and its scoring are missing the same
content, and the gap reads as *good* performance. In a measured control on a real amp,
corners that failed coverage scored **better** than passing corners in the same region.
Coverage is a property of the input signal and has to be checked against the input signal.

## 6. Train

```bash
./run_pipeline.py --config my_pedal.config.toml --workspace ~/runs/device_run1
```

Everything from here is automatic. This one command runs six stages of its own, and
prints each as a banner — these are `run_pipeline.py`'s internal steps, not the six
walkthrough steps on this page:

| banner it prints | what it does | on failure |
|---|---|---|
| `STEP 1 / 6 — Grid Adequacy` | re-verifies the grid from step 3 (cached, so ~instant) | **aborts** |
| `STEP 2 / 6 — Input Headroom` | does the excitation reach saturation at default settings | warns |
| `STEP 3 / 6 — Preflight` | dead/reversed knobs, input calibration | **aborts** |
| `STEP 4 / 6 — Dataset Generation` | renders every combination (coverage re-checked inside) | **aborts** |
| `STEP 5 / 6 — Combine` | assembles `outputs.npy` | |
| `STEP 6 / 6 — Training` | SGDR warm restarts, open-ended | |
| `RELEASE` | models, `.schx`, `MANIFEST.md`, `reproduce.sh` | |

Training with `epochs = 0` runs until you `touch <workspace>/checkpoints/STOP`, exporting
the best model continuously. Use the first run to find the budget, then set a real schedule.

**On SGDR restarts.** The defaults are `--restart-mult 1` (equal-length cycles) and
`--restart-decay 0.97`. Every full-LR restart pays a roughly *fixed* recovery cost — several
epochs spent re-descending to where the previous cycle already was — so with equal-length
cycles that cost stays a constant fraction of the run no matter how long you train
(measured: ~54% of total epochs spent re-climbing).

`--restart-mult 2` grows each cycle geometrically while the per-restart cost stays fixed, so
the wasted fraction shrinks toward zero (~9% on the same budget). It is **not** a safe
always-on default, which is why it isn't one: `--stale-cycles` counts *cycles*, not epochs,
so geometrically growing cycles make that auto-stop rule geometrically slower to fire. If
you use `mult 2`, lower `--stale-cycles` to 2–3, add `--stale-epochs` as an epoch-counted
backstop, or set an explicit epoch/step budget instead of relying on `--stale-cycles`.

---

## What ends up where

```
~/runs/device_run1/
├── config.toml                 stamped on first use; guards against a different run
├── pipeline.log
├── excitation/
│   ├── excitation.wav
│   └── excitation.recipe.json  the sizing measurement — the file worth keeping
├── dataset/
│   ├── outputs.npy  params.csv  config.json  sweep.wav
│   └── NOTICE_RESTRICTED_INPUT.md
├── checkpoints/
│   ├── best.pt  best_lite.pt  latest.pt  cycle_*.pt
│   └── metrics.csv  watchdog.log  per_combo_esr_w*.csv
├── device_run1.param.nam
└── release/
    ├── *.param.nam  <device>.schx  metrics.csv
    ├── dataset_config.json  dataset_params.csv
    └── ESR_RECORD.md  MANIFEST.md  reproduce.sh
```

Deliberately outside it: content-keyed caches in `~/.cache/parametric-nam/`
(`findpeak`, `gridadq` — shared across runs, safe to delete at any time), scratch in
`$TMPDIR` (deleted on exit), and the device's committed config + recipe sidecar, which are
*input* to a run rather than output of one.

## If something goes wrong

| symptom | look at |
|---|---|
| Model plateaus above target no matter how long you train | `grid_adequacy.py` — the grid, not the training |
| A knob does nothing in the trained model | `preflight.py` — likely a misclassified or genuinely dead control |
| Corners fail transient coverage | step 4 — the excitation was sized against a different grid |
| Backend diverges mid-render | the `.backends.toml` sidecar; try another backend |
| Renders are unexpectedly slow to re-verify | check the cache is not being invalidated — an edited `.schx` or rebuilt excitation invalidates it *correctly* |

More devices hit a **capacity** ceiling than a training-time one once more than three knobs
are swept. If the widest tier plateaus, widen it before training longer.
