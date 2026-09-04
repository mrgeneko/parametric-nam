# From a new `.schx` to a trained `.param.nam`

The end-to-end path for a circuit this repo has never seen. Six commands, three of which
you review the output of rather than just run.

For what each script does in detail see [scripts.md](scripts.md); for the gates the pipeline
applies and what each one aborts on, [checklist.md](checklist.md).

Throughout, `~/runs/device_run1` is a **workspace** — one directory holding everything this
run produces. It is per *run*, not per circuit: a second run goes in a second directory, and
`run_pipeline.py` refuses to run a different config into an existing one. See
[`--workspace`](scripts.md#--workspace--one-directory-per-run).

---

## 1. Scaffold the config

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

`--write-backend-sidecar` writes `<stem>.backends.toml` if the backend *diverges* while
measuring oversample. That verdict is a measurement — the scaffold just watched the circuit
fail — so it should not be hand-written. A circuit whose backend cannot converge is better
discovered here than four hours into a render.

## 2. Review the config — the step you cannot skip

The scaffold cannot know two things:

**Knob order.** It writes them alphabetically; you want **signal flow**. The order is
recorded in the dataset and the model, so getting it right later means re-rendering.

**Knob kinds.** Each guess is marked `UNCONFIRMED`. A misclassified knob silently gets the
wrong sensitivity metric in `grid_adequacy.py` *and* skips `preflight.py`'s role-aware EQ
safeguard — so a dead knob can pass every gate.

Also check the `[knobs]` ranges. EQ knobs are narrowed to `[0.2, 0.8]` by the scaffold's own
role-aware default; that is deliberate, not something `grid_adequacy` chose.

## 3. Refine the knob grid

```bash
./grid_adequacy.py --config my_pedal.config.toml --apply
```

Renders each cell's midpoint and compares it against interpolating its neighbours. A cell
whose residual exceeds the target is a floor no amount of training can lift — the
information was never sampled. `--apply` bisects failing cells, re-probes, and writes the
converged grid back, leaving all your comments intact.

Probe renders are cached on disk, so step 6's re-verification of this grid is free.

> **`--apply` can also *coarsen* an axis.** It cut Duke of Tone's Tone and Presence to their
> 2-point floor on measured error alone. Those were restored by hand: endpoints
> interpolating well over one probe window is not the same as a knob being linear
> everywhere in use, and a 2-point axis leaves nothing to catch it if it isn't. If you
> override a suggestion, say so in a comment — a later `--apply` will otherwise re-cut it.

## 4. Re-size the excitation against the refined grid

```bash
./prepare_excitation.py --backend livespice \
                        --config my_pedal.config.toml \
                        --real-clip <a real playing clip> \
                        --workspace ~/runs/device_run1
```

**Do not skip this because step 1 already built one.** Step 1's excitation was sized against
the *placeholder* grid. Step 3 changed the grid underneath it, which changes the corner set,
which changes the worst-case saturation onset the excitation has to reach.

This is not hypothetical. Mesa Orange and RED both trained against an excitation whose
15.4369 V transient peak had been hand-picked and never measured against any corner set.
They failed 6 of 43 and 4 of 43 corners respectively. RED's re-measured worst case was
20.8276 V — at `RD Gain=lo-solo`, **1.27× the highest of all 32 hypercube vertices**, because
saturation onset is not monotonic in the knobs and the worst corner need not be a vertex.

The excitation lands at `~/runs/device_run1/excitation/excitation.wav`, its
`.recipe.json` sidecar beside it, and `input` in the config is updated to match. Keep the
pair together: every consumer finds the sidecar by deriving it from the wav's own path.

## 5. Verify coverage independently

```bash
./check_transient_coverage.py --config my_pedal.config.toml
```

Step 4 sized the excitation; this checks the result from the outside, across structural
corners *and* interior sample points. It reads the required peak from the recipe sidecar
automatically.

Per-combination ESR cannot substitute for this: validation is drawn from the same
under-covered data, so a gap looks like *good* performance. Measured on Orange — failing
corners scored **better** (0.1140) than passing neighbours (0.1405) in the same region.

## 6. Train

```bash
./run_pipeline.py --config my_pedal.config.toml --workspace ~/runs/device_run1
```

Everything from here is automatic:

| | | on failure |
|---|---|---|
| STEP 0 | `grid_adequacy` re-verify (cached, so ~instant) | **aborts** |
| STEP 0b | `check_input_headroom` | warns |
| STEP 0c | `preflight` — dead/reversed knobs, input calibration | **aborts** |
| STEP 1 | generate (transient coverage re-checked inside) | **aborts** |
| STEP 2 | combine → `outputs.npy` | |
| STEP 3 | train — SGDR warm restarts, open-ended | |
| — | release folder: models, `.schx`, `MANIFEST.md`, `reproduce.sh` | |

Training with `epochs = 0` runs until you `touch <workspace>/checkpoints/STOP`, exporting
the best model continuously. Use the first run to find the budget, then set a real schedule.

Always pass `--restart-mult 2 --restart-decay 0.85`: equal-length full-reset SGDR cycles
waste roughly half the steps (measured).

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
