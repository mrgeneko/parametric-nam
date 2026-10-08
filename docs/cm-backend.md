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

Same measurement as for `livespice`: renders a stratified sample of windows at each power of two and
at a 32x reference, and chooses the lowest rate whose truncation ESR is under `--trunc-target`
(default 1e-3). With this backend the truncation is usually well below that at 2x or 4x, which is why
datasets rendered here need lower rates than `livespice` ones.

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
(`physical`) and `esr_vs_oracle` (null until a validation step fills it in with an independent renderer's ESR on the same
input). Compare datasets only when name, version and profile agree.

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
