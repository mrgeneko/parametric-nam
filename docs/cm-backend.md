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
**6e-3** (an ESR difference below about 0.006 is taken as inaudible; `livespice` keeps 1e-3). The fall per doubling is
checked as before, whatever the step between two candidates. Typical picks on the T3K sweep: a pedal such as the TS-9
1x, a combo amp 1-2x, a high-gain amp or the Metal Zone 2x. The retry ladder after a failed render still doubles
from the chosen rate (a start of 1, 3 or 6 gives 1/2/4/8/16/32, 3/6/12/24/32 or 6/12/24/32).

For this backend the probe uses at least 16 windows of 2 s (32 s in all): a calibrated excitation's error sits in its loud, swept
passages, and 4 windows missed it (on a combo amp with its sized excitation the probe read 5e-4 at 1x from 4 windows and 1.5e-2 from 8 or more,
and the full dataset showed 1.9e-2: the pick went from 1x to 2x).

The probe estimates the whole-signal ESR of the sampled windows, which is what a model is fitted against; the renderer's
tuning tables report the worst case over a set of knob settings, so they may ask for a rate or two more.

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
a cycle "late"; the output stays aligned with the input). With that, the circuit file's "prepared state" and a
lead-in are equivalent (ESR below 1e-8 on the test amp at 44.1 and 48 kHz); `--cm-lead-in` is kept because it
needs nothing from the file.

## What is recorded

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

The renderer's default profile is physical, LiveSPICE's is not, and the two differ systematically:
on four test circuits (a pedal, a bass preamp, a tube combo amp, a cabinet amp) the ESR between a
`livespice` render and the `cm` render was 2e-3 to 5e-3 at the same oversample, and it did **not**
shrink at oversample 32 (3-5e-3), so it is a model difference, not truncation. Run in its LiveSPICE
compatibility mode the renderer reproduces `livespice` to about 1e-6 at the same oversample and
to 1e-14 at 32x, which confirms that units, pot tapers, knob mapping and the default speaker agree;
the physical profile is the one used here. Compare a model trained on `livespice` data with data from
this backend by ESR on the same input, not by identity.

## Not covered

* `measure_truncation.py` still drives `livespice`; use `--oversample auto` for this backend.
* Hand-written decks (`ngspice-deck`, `ltspice-deck`) stay on their own backends.
