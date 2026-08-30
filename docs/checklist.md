[← back to README](../README.md)

The [Scripts](../README.md#scripts) table and [`docs/scripts.md`](scripts.md) say what each
validation tool checks. This says *when* to run one, and whether `run_pipeline.py` already runs
it for you — the map from "which pipeline stage am I at" to "which tool(s) apply" that used to
live only in the author's head.

# Checklist

## Stage 1 — Authoring a new circuit (once per device, not per training run)

These pick solver settings for a *new* circuit's `config.toml`. Re-run only when the circuit,
excitation, or backend changes — not on every training invocation of an already-tuned device.

| Tool | Answers | Run when |
|---|---|---|
| `measure_truncation.py` | Is `oversample` high enough (BDF2/livespice truncation error)? | Picking `oversample` for a new device, or auditing the fleet after a solver change |
| `measure_ngspice_timestep.py` | Is `maxstep` fine enough (ngspice-deck backend only)? | Same, for a hand-written ngspice-deck circuit — `render_ngspice_deck.py`/`render_backends.py`/`ngspice_spicelib.py` all default `maxstep` for convergence, never measured for accuracy |

Not run by `run_pipeline.py` — there is no reasonable "auto" trigger for a one-time tuning
decision. Run by hand, write the result into `config.toml`, move on.

## Stage 2 — Before generating a dataset

Run automatically by `run_pipeline.py` for every generation unless skipped. All are gated on
`--config` being set (they read the circuit's recipe TOML directly).

| Step | Tool | Backend scope | On failure |
|---|---|---|---|
| STEP 0 | `grid_adequacy.py` | any | **Aborts.** A cell whose interpolation error exceeds the target ESR puts a floor under the model no training can lift. |
| STEP 0b | `check_input_headroom.py` | any | **Warns, continues.** A low ratio can be a real gap or a genuine high-headroom device — this is a prompt to check the grid's own hottest corner, not a verdict. |
| STEP 0c | `preflight.py` | **livespice only** — no mode exists for the schx-translated `ngspice` backend or `cpp` | **Aborts.** A dead or reversed knob renders and trains "successfully" and produces a plausible but wrong model. |
| *(inside STEP 1)* | `check_transient_coverage.py` | livespice only | **Aborts.** Called directly by `gen_dataset_from_schx.py` itself, not a separate `run_pipeline.py` step — see its own transient-check block. Forwarded flags: `--skip-transient-check`/`--transient-peak`/`--transient-margin`. |

Skip flags: `--skip-grid-check`, `--skip-headroom-check`, `--skip-preflight-check`,
`--skip-transient-check` (forwarded to `gen_dataset_from_schx.py`).

## Stage 3 — After generating a dataset

Manual — investigate a specific concern, not a blanket gate.

| Tool | Answers | Applies to |
|---|---|---|
| `coverage_report.py` | Are there holes in the sampled knob space (boundary coverage, pairwise gaps)? | **Random-sampled datasets only** (`--random`) — a full Cartesian grid has no sampling holes to find; `grid_adequacy.py` is the right tool there instead |

## Stage 4 — After training

| Tool | Answers | Auto? |
|---|---|---|
| `per_perm_esr.py` | Which knob-grid regions fit well vs. poorly? | `compute_per_perm_esr()` is called automatically by `param_train.py` at the end of training. The standalone CLI re-runs it against an already-saved checkpoint later. |
| `level_band_esr.py` | Is the model dropping distortion/detail as a note decays (headline ESR can't see this — it's energy-weighted, and a quiet tail is nearly all duration, none of the energy)? | Manual. Run when a model sounds wrong in a way the headline ESR doesn't reflect, or as a matter of course before shipping anything with a decay/sustain-sensitive circuit. |
| `scan_film_runaway.py` | Does the model blow up 10s-100s of times normal peak at a narrow (knob-corner × real-transient) combination the training excitation under-covered? | Runs automatically as a post-training WARN step in `run_pipeline.py` **if `--film-reference` is set** (a real-playing clip — not derivable from the training excitation, which is exactly the gap that let two published models ship with this bug unnoticed). Skipped with a note if unconfigured. Skip flag: `--skip-film-runaway-check`. |

This is also "the post-training check" referenced elsewhere as a fixed pair: `scan_film_runaway.py`
plus `release_run.sh`'s own validation, before publishing anywhere.

## Quick reference — symptom to tool

| Symptom | Tool |
|---|---|
| Model plateaus above the ESR target no matter how long you train | `grid_adequacy.py` (Stage 1/2) |
| A knob does nothing, or moves the wrong direction | `preflight.py` |
| Model never learned saturation/breakup character | `check_input_headroom.py`, `check_transient_coverage.py` |
| Random-sampled dataset feels sparse in some region | `coverage_report.py` |
| Model sounds fine on sustained notes but wrong as they decay | `level_band_esr.py` |
| Model spikes/blows up only at specific, rare knob settings | `scan_film_runaway.py` |
| Model is good on average but bad in one part of the knob grid | `per_perm_esr.py` |
| ngspice/livespice render is slow or won't converge | `measure_truncation.py`, `measure_ngspice_timestep.py`, [`docs/backends.md`](backends.md) |
