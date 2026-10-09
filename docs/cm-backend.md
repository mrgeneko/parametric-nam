[← back to README](../README.md)

# `--backend cm`: a `cm_run`-compatible renderer

An offline, fixed-timestep renderer for `.schx` circuits that have been converted to a `.cm.json`
circuit file. It sits next to `livespice` (same schematic, same knob names, same default speaker) and
differs in two ways that matter for dataset generation: it is several times faster on tube amps, and
it uses a **physical** numerics profile instead of LiveSPICE's.

> **Availability.** The renderer this backend drives is not yet publicly released, so `--backend cm` is only usable where
> a `cm_run`-compatible executable and the matching `.cm.json` circuit files are available. Every other backend is
> unaffected, and the tests use stub renderers.

## Setup

* The renderer is any executable that follows the contract below. Point `--cm-run PATH` or `$CM_RUN`
  at it (default: `cm_run` on `PATH`).
* The circuit is `<stem>.cm.json` **beside the `.schx`**. It records the SHA-256 of the schematic it
  was converted from; the backend refuses a file that does not match, at start-up, before any render.
  Convert the schematic again if you edit it.
* The transient/saturation coverage gate runs for this backend too, with the saturation onsets measured
  by the same renderer (`--skip-transient-check` to bypass it).

```
python gen_dataset_from_schx.py --backend cm --schx amp.schx --knobs Gain,Tone \
    --input sweep.wav --output ./training_data --oversample auto
```

## In the pipeline

The backend is selectable wherever a backend is: `run_pipeline.py --backend cm` (or `backend = "cm"` in a device config),
`scaffold_config.py --backend cm` (writes `oversample = "auto"`, sizes the excitation with cm probes), `prepare_excitation.py`,
`preflight.py`, `check_transient_coverage.py`, `check_input_headroom.py`, `grid_adequacy.py` and `gate_config.py`. The probe
renders of the sizing, coverage, preflight and grid steps go through the same renderer (the saturation onsets they measure
agree with livespice's to the sweep's resolution). Config keys: `cm_run`, `cm_lead_in`, `cm_tables`, `trunc_target`
(`oversample = "auto"` is resolved at render time; the probes use 2x, where the onsets do not move). Fleet mode forwards
`cm_lead_in`, `cm_tables` and `trunc_target` to the workers, each of which finds its own renderer via `$CM_RUN` or `PATH`.

## The renderer contract

```
cm_run CIRCUIT.cm.json in.wav out.wav --prepared off --os N --tol-rel 1e-4 --tables off
       --resampler fir-linear --iterations N [--knob Name=V ...] [--output NAME]
       [--lead-in SECONDS] [--metrics FILE.json] [--progress]
```

It writes a mono float wav, prints `PROGRESS done/total` lines to stderr (the stall detector reads
them, as it does for `livespice`), and writes a metrics JSON with `solves`, `unconverged`, `severe`,
`divergences`, `rescues`, `rescued`, `max_iterations`, `first_bad_sample` and `dc_converged`.
Knobs are `--knob Name=V` with V in 0..1 (the taper is in the circuit file); a switch group is one
knob whose value is the position index.

## What fails a render, and the ladder

* **Divergence** (a solve that blew up) or a **start-up DC point that did not converge**: the render
  fails and the ladder retries.
* **Unconverged solves**: the renderer rescues and limits its own hard steps, so a few remain. A
  render fails if any is *severe* (last Newton step over 1 V) or if `unconverged / solves` exceeds
  `--newton-max-fraction` (default **1e-5** for this backend; 1e-6 for `livespice`). Otherwise the
  counts are recorded in the `warnings` column. A fleet-wide count at oversample 4 and 8 over about
  130 circuits and three knob settings found no divergences, 9 of 381 runs with any unconverged solve
  at oversample 4 (worst fraction 1.1e-6) and 2 with a severe one; doubling the oversample did not
  remove the persistent ones.
* **Ladder:** oversample doubles from `--oversample` (default 2) to 32, with 4x the iterations on the
  last rung. Higher oversample is not always cleaner for convergence, so the ladder is a retry, not a
  quality knob; pick the rate with `--oversample auto`.

## `--oversample auto`

Same measurement as for `livespice`: renders a stratified sample of windows at each candidate rate and at a 32x
reference, and chooses the lowest rate whose truncation ESR is under `--trunc-target`. For this backend the candidates
are the factors the renderer's tuning measures, cheapest first, **1, 2, 3, 4, 6, 8, 16**, and the default target is
**1e-3**, the same as `livespice` (the backend used 6e-3, an audibility line, on 2026-10-08/09; the faster renderer makes the
stricter target affordable). The fall per doubling is checked as before, whatever the step between two candidates. What the
circuit files' quality tables (worst case over the typical-knob plan and the stress signal, tolerance 1e-4) allow at 1e-3 over
the 121 files with a table: 1x for 13, 2x for 27, 3x for 31, 4x for 24, 6x for 19, 8x for 5 (at 6e-3 it was 1x 36, 2x 50,
3x 28, 4x 5, 6x 1); the TS-9 stays at 1x, the Deluxe Reverb goes from 2x to 3x, the SVT Full from 2x to 4x, the JCM800
preamp from 3x to 4x, the ENGL from 4x to 8x. Two files have no robust cell at 1e-3 (the Dumble, the EVH 5150), so their ladder
starts at the top measured rate. The retry ladder after a failed render still doubles from the chosen rate (a start of 1, 3
or 6 gives 1/2/4/8/16/32, 3/6/12/24/32 or 6/12/24/32).

For this backend the probe uses at least 16 windows of 2 s (32 s in all): a calibrated excitation's error sits in its loud, swept
passages, and 4 windows missed it (on a combo amp with its sized excitation the probe read 5e-4 at 1x from 4 windows and 1.5e-2 from 8 or more,
and the full dataset showed 1.9e-2: the pick went from 1x to 2x).

The probe estimates the whole-signal ESR of the sampled windows, which is what a model is fitted against; the renderer's
tuning tables report the worst case over a set of knob settings, so they may ask for a rate or two more.

**The file's quality table is a floor (2026-10-09).** The probe samples a few windows at a few knob settings; the file's
`quality.measured.cells` are worst case over `cm_tune`'s knob plan and the stress signal. The ladder therefore starts at the
cheapest robust cell (tolerance 1e-4, the default filter, this backend's tables setting) whose ESR is within `--trunc-target`,
and never below it. Found on the HM-2: the probe picked 1x, where the table reads ESR 1.59.

## Tube tables

`--cm-tables on|off` (default **on**): the renderer's tabulated tube characteristics, 1.12-1.20x faster on the tube amps
measured (Deluxe, Tweed, Mesa, ENGL, Bogner, Hiwatt, SVT, JCM800), with the output within ESR 4e-6 of the exact
equations; outside a table's range the exact equations are used. `off` renders with the exact equations throughout.

## Start-up

A cold start has a settling transient (coupling capacitors, a supply that sags). On a sag/ac amp the
ESR against the settled render was 0.37 in the first second, 8e-3 in the second and 2e-7 from 2 s.
The backend runs `--cm-lead-in` seconds (default **6**) of silence through the circuit first and
discards them, so the output stays sample-aligned with the input and starts settled. `0` starts cold
like `livespice`. The renderer aligns the mains phase: it adds just enough silent lead-in that the first kept sample sees the
circuit's declared mains phase, whichever way the render starts (this is why a render can start a fraction of
a cycle "late"; the output stays aligned with the input).

**Prepared state (2026-10-09).** When the circuit file carries a prepared (settled) state the renderer accepts (its
`cm_prepare` output for this netlist and measurement version), the render starts from it with a **0.5 s** lead-in
instead of the 6 s. Measured on four sag/ac amps (Deluxe, JCM800 power amp, AC30, Twin; three knob settings each; 10 s of
T3K at 4x) against a 20 s lead-in: the 6 s lead-in is within 4e-9 ESR, 2 s within 1.5e-4, the prepared state alone within
4e-7 (the first second 1.4e-6), the prepared state plus 0.5 s within 3e-10 everywhere. Whether a file's state is valid is
known only to the renderer, so the backend probes it once per circuit (a render of a few milliseconds of silence, reading
`prepared_state_used` from the metrics) and falls back to the 6 s lead-in for files without one. The saving is 5.5 s of
rendering per combination: 3 % with the 190 s T3K sweep, about 20 % with a 20 s excitation. `--cm-lead-in 0` still means
a cold start.

## What is recorded

`params.csv` carries each combination's render cost from `cm_run`'s metrics: `proc_time` is the wall seconds of the render
and `dsp_load` is 100 / the real-time factor (they were -1 before 2026-10-09; the `livespice` path parses the same two from
its oracle's stdout).


`config.json` carries a `renderer` block: `name`, `version` (from `cm_run --build-info`, one line), `profile`
(`physical`) and `esr_vs_oracle`. Compare datasets only when name, version and profile agree. `esr_vs_oracle` is null until
`oracle_check.py --dataset DIR` fills it (or `run_pipeline.py --oracle-check N` does, after combining): it re-renders N
combinations (the first, the last and the middle of the grid) with an independent renderer (livespice-cli) over the first 40 s of
the same input, through the same capture chain, and records the oracle, its version, the median and maximum ESR, the ESR after the
best single gain, the gain, and the ESR after a cabinet-like low-pass (4th-order, 5 kHz): a hot swept excitation puts most of the
full-band difference above 6 kHz, which a guitar cabinet removes, so the cabinet figure is the one to read against an audibility threshold.
On the TS-9 it reads about 1-2e-3 full-band (cm against livespice); on a combo amp (Deluxe, sag ac) with its sized excitation 3-4e-2
full-band and 4-8e-3 through the cabinet low-pass.

## Not comparable sample-for-sample with `livespice` data

On the same circuit, knobs and input the two renders differ, mostly above 6 kHz, and the cause is the **resampler**, not the circuit
numerics. `livespice` interpolates the input linearly up to the oversampled rate and takes the **plain average** of each output
sample's oversampled values back down: a boxcar over one output period. The linear interpolation and the average together droop the path by about 1.9 dB at 10 kHz,
4.2 dB at 15 kHz and 7.8 dB at 20 kHz (4x; the average alone is 0.6, 1.4 and 2.5 dB, and it does not narrow as the oversample
rises), and the average rejects aliases poorly (about 6 dB at 30 kHz, which folds to 18 kHz). This backend uses a linear-phase FIR resampler, flat to near
Nyquist with strong alias rejection. Measured on a combo amp (Deluxe, sag ac) with its sized excitation, one combination, oversample 8
on both sides: this backend's default against `livespice` ESR 3.9e-2 (84 % of the difference above 6.4 kHz); the same circuit with the
numerics profile left at "physical" but `livespice`'s resampler, 1e-7; `livespice`'s numerics with this backend's FIR resampler, 3.9e-2
again. The physical-versus-LiveSPICE numerics profile (thermal voltage, op-amp output resistance) is a negligible part of the
difference on that circuit. On the TS-9 the gap is ESR 9e-4 to 2e-3, on the combo amp 3e-3 to 4e-2 depending on how hot the
excitation is, and it does not shrink with oversample (the boxcar is one output period wide at any rate). Through a cabinet-like 5 kHz
low-pass the combo amp's figure is 4e-3 to 8e-3.

Which is closer to a real capture: an audio interface's converter filter is flat to about 20 kHz and rejects what lies above, which the FIR
resembles and the boxcar does not; the cm render therefore leaves out the aliasing and the droop `livespice` adds. That is an expectation from the
filters, not a comparison with hardware, which has not been done. Either way, compare a model trained on `livespice` data with data from this
backend by ESR on the same input, not by identity. Run in the renderer's oracle mode for LiveSPICE (`cm_run --oracle-livespice`, for equivalence checks only; the standalone `--resampler livespice` was removed) the renderer
reproduces `livespice` to 1e-6 or better at the same oversample, which confirms that units, pot tapers, knob mapping and the default speaker agree.

## Running on a fleet

`distribute_pull.py` renders `--backend cm` chunks on workers the same way as any backend, with these conditions:

* **`cm_run` on each worker.** A worker finds its own renderer: `$CM_RUN`, else `cm_run` on the PATH of the shell that runs the chunk (the controller's path means nothing there). Either put a `cm_run` where a non-interactive ssh command finds it, or give the worker `HOST:DIR:PARALLEL:CM_RUN=/path/to/cm_run`; the dispatch-time version check exports that field too. `fleet_inventory.py` lists `cm` for a host where `sh -lc "cm_run --build-info"` succeeds.
* **The same libcm build everywhere.** The version check compares the worker's `cm_run --build-info` with the controller's and refuses a worker that differs. A source tree copied to a worker has no git, so write the commit beside the sources before building (`git rev-parse --short=12 HEAD > COMMIT`, plus `-dirty` if the tree was); otherwise the build reports `unknown` and the worker is refused.
* **The excitation's recipe on each worker.** See "The transient check needs the excitation's recipe on every worker" in [`scripts.md`](scripts.md#distribute_pullpy--hand-rendering-chunks-out-as-workers-free-up): `--sync-file` the wav and its `.recipe.json`, or pass `-- --transient-peak V` / `-- --skip-transient-check`.

Output does not depend on the worker: the same grid rendered on Apple silicon, an AMD Ryzen and two Intel cores agreed to ESR 1e-15 or better (TS-9 at oversample 2; Deluxe Full at oversample 4 with tables).

## Not covered

* `measure_truncation.py` still drives `livespice` and stays until livespice is deprecated; use `--oversample auto` for this backend.
* Hand-written decks (`ngspice-deck`, `ltspice-deck`) stay on their own backends.
