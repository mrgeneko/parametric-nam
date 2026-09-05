[← back to README](../README.md)

# Known Issues

Known issues and their investigations/fixes for parametric-nam.

## Known issue: rare knob-corner blowup (FiLM/LeakyReLU runaway)

Trained `.schx` models can develop a **narrow, catastrophic instability** at a specific
knob-grid corner — the model's prediction spikes to tens or hundreds of times its normal
peak level on real transient content, while every aggregate metric (val ESR, per-tier
loss curves) looks fine, because the corner is a single cell out of hundreds-to-thousands
and contributes almost nothing to the training loss. It has recurred independently on at
least three shipped/prototype models (a pre-fix pedal model, one amp's FiLM-only 5-knob
release, and that same amp's FiLM+LoRA 2-knob prototype) at different knob combinations, so
treat it as a real, recurring failure mode of this pipeline, not a one-off — and not specific
to either conditioning mechanism. Whether LoRA's extra per-layer capacity makes the failure
*worse* once it occurs (more room for an unconstrained input region to produce an extreme
wrong answer) versus FiLM alone is a real, plausible hypothesis, not yet confirmed: a
same-corner probe against a FiLM-only model found only ~1.1× elevation where the FiLM+LoRA
case showed ~7×, but on a different grid/config, so it's suggestive rather than a controlled
comparison.

What it looks like, concretely (one 5-knob amp model, lite tier): at
`NormalVol=BrightVol=0.025` (the swept grid's own minimum) combined with `Treble=Bass=
Middle=0.8`, the model predicted a peak of **100.2 V** against a ground-truth peak of
**0.64 V** (156×) — RMS stayed normal, so it's a brief spike, not sustained distortion,
and it's invisible to ESR unless you check per-combination, not just aggregate. A second,
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
- **`check_transient_coverage.py`** (pre-training gate, run automatically by
  `gen_dataset_from_schx.py` unless `--skip-transient-check`) — checks that the
  excitation's transient content actually reaches each corner's own saturation onset,
  across the **full min/max hypercube** (every swept knob independently at its own grid
  min or max, not just one knob varied from center) plus the traditional solo-knob
  corners. A corner whose transient-bearing `--input` content never crosses into saturation during
  training is a corner the network has to extrapolate at inference time. Supports
  `backend = "ngspice-deck"` or `"ltspice-deck"` in the config (same
  `pedal-dir`/`module`/`probe-node` convention as
  `preflight.py`/`prepare_excitation.py`/`grid_adequacy.py`) for a device
  with no `.schx` at all — `gen_dataset_from_schx.py`'s own automatic call is
  livespice-only, since that's the only path it drives; for a hand-deck device, call
  `check_transient_coverage.py` directly as its own gate before generating.
- **`scan_film_runaway.py`** (post-training, run by hand against a finished
  `.param.nam`) — replays a real reference clip through every tier at every corner
  (`--config` for the exact trained grid, not just the reduced set) and flags any window
  whose peak is anomalous relative to that model's own typical output. Scanning the
  affected amp's fp32 model with a **generic** guitar reference came back clean even
  including this exact corner — the spike only showed up against the **actual training
  excitation** at that combination, so when investigating a suspected corner, prefer
  `--reference <the training sweep.wav>` over an arbitrary clip; a scan that doesn't
  reproduce the triggering content can give a false sense of safety.
- **`build_excitation.py --synth-burst-peaks`** (excitation-design fix, added after
  diagnosing the FiLM+LoRA case above) — `check_transient_coverage.py` only checks that
  the excitation's peak *level* reaches saturation onset per corner; it says nothing
  about *shape*. That FiLM+LoRA blowup happened at a trained grid point (not a gap
  needing extrapolation) whose excitation had moderate-level content and separately had
  high-crest-factor (sharp-transient) content, but never both together at the same
  time — that exact (level, shape) combination was simply never in the loss, so nothing
  constrained the model's behavior there. `--synth-burst-peaks` inserts one synthesized,
  deterministic, license-free broadband transient burst (`_transient_burst()`, crest≈8.5,
  instant attack + exponential decay) at every `--sweep-peaks` level, closing that gap
  directly. Verified end-to-end on a 2-knob retrain of that amp: `scan_film_runaway.py`
  came back clean across the full grid **at full training convergence** (not just an early
  checkpoint — the original instability itself only emerged well into training, so a
  clean early scan alone doesn't prove a fix held).

If a corner gets flagged, first check whether it's a *level* problem
(`check_transient_coverage.py`, fixed by raising `--realistic-peak`/`--sweep-peaks`) or a
*shape* problem (a real reference clip flags it but the excitation's own peak clears
onset fine — fixed by `--synth-burst-peaks`, not by more level). Don't just retrain
longer at the same excitation — the blowup in the Tweed case was a spike lasting well
under a second inside a single combination's clip, so it barely moves that combination's
own loss, let alone the average across hundreds of combinations; more of the same data
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

**Concrete evidence (one real pedal, ngspice backend)**: `C10` (10µF) into the Volume pot's
low leg (`RVOL2`, up to 500kΩ) gives an RC time constant of roughly **5 seconds**. A sustained
200Hz test tone with no lead-in,
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

**Fix, applied by default**: `build_excitation.py --lead-silence-s` (default `3.0`)
prepends that many seconds of true silence before the `--input`-derived segment.
The silent segment is written into the excitation file itself (not stripped after generation),
so it also gives the DC-blocking network real settling time before the file's `real` segment
dynamics are what training actually samples. Set to `0` to disable for a circuit already
verified not to need it. This is a cheap, non-device-specific fix — prefer it over adding
explicit `.ic` initial-condition statements to a specific device's deck, which would need to be
hand-derived and re-verified per circuit.

The same fix is also needed in `preflight.py --backend ngspice-deck` and
`find_saturation_point.py`'s own amplitude sweep (both take `--lead-silence-s`, default
`3.0`) — a short knob-direction probe with no lead-in hits the identical cold-start transient,
and it isn't just noisy, it can flip the answer: that pedal's Tone knob probed REVERSED at a
5-second probe with no lead-in and correctly OK once probed with a lead-in (or a long-enough
probe to outlast the transient on its own). See the next "Known issue" entry for the other,
independent fix this same investigation needed.

**Caveat**: 3 seconds was sufficient for the one circuit this was diagnosed and verified
against, not derived from the RC time constant analytically for every possible circuit. A
circuit with a slower network (larger coupling cap and/or higher-value downstream resistance)
could need longer — when in doubt, run the sustained-tone check above with the circuit's own
component values before trusting the default.

---

## Known issue: preflight's EQ-knob checks can be swamped by the circuit's own gain control

`preflight.py`'s dead/reversed-knob check holds every non-tested knob at a flat `0.5`
center. For a low-to-moderate-gain circuit that's fine, but for a genuinely high-gain device
(one real pedal's stages run up to ~50dB combined), `0.5` on the Gain/Drive knob can already be
deep into saturation — and hard clipping SWAMPS a passive tone stack's real effect, since the
extra content an EQ knob lets through just becomes more clipping mush, not more clean signal at
that band. This reads as a **false REVERSED (or DEAD) direction on a knob that's actually
correct**, and it isn't just about probe input level — it's the circuit's OWN gain control
self-saturating regardless of how quiet the input is.

**Concrete evidence (same pedal, `--backend ngspice-deck`)**: with Gain held at the default
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
gain control while the tone stack's real effect is being measured. This alone fixed the
MOSFET-clipping pedal's Tone reading at every probe duration tested, with no need for a reduced
input level at all.
Volume/level-classified knobs are untouched — they're normally a post-clipping-stage
attenuator, so they don't change whether the nonlinear stage itself saturates.

**Complementary, not a replacement**: `--find-peak`/`--clean-probe-peak` (input-level scaling,
probing an EQ knob at both a driven and a linear-region level, pass-if-either) stays on by
default too — it catches the same failure mode via a different knob than the one under test,
and via a knob `classify()` can't identify as `"drive"` by name. Both mitigations were verified
independently on the MOSFET-clipping pedal; neither alone was assumed sufficient going forward.

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
