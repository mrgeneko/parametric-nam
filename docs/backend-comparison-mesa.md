[← back to README](../README.md)

# When a circuit defeats the solver: LiveSPICE vs LTspice on the Dual Rectifier RED

Everything here was measured between 2026-09-01 and 2026-09-02 on the Mesa Boogie Dual Rectifier
Solo Head RED Full (sag v30). It is written down because the conclusion is **not** "use the other
backend for hard circuits", which is what we set out expecting to confirm.

## What started it

A 448-combination grid render completed 393 combinations. All 55 failures shared one knob value:

    Red Master = 1.0     (RD Gain = 0.1 / 0.15 / 0.25, every EQ corner)

i.e. minimum preamp gain into a fully-open master — a small signal driving the power and sag
stages wide open. Not scattered: a precise rectangle in knob space, and **100% of that region
failed**.

The failure is `solver spike: N isolated single-sample overshoot(s)` — 9 to 17 bad samples out of
5,784,000, i.e. 0.0002% of the signal, which the detector rejects the whole render for. It is
right to: a single-sample glitch is a poison training target.

**More oversample does not fix it.** The retry ladder escalated 8 → 16 → 32 → 64 → 128 and the
spikes persisted identically at every rung. The same corner on the same amp's ORANGE channel is
already recorded in `gen_dataset_from_schx.py`'s escalation-ceiling comment (2026-08-30) as
"exhausted os=8/16/32 identically"; this adds 64 and 128 to that list.

A full-grid sweep (`stability_sweep.py`, 448 combinations at a 3 s probe) confirmed raising the
baseline is a losing trade:

    oversample=8    431/448 converged, 17 spikes
    oversample=16   437/448 converged, 11 spikes   -- fixes only 6 of 17

    A. os=8 + retry ladder   568 relative units   <- what was already running
    B. os=16 baseline        909 units  (+60%)    <- taxes all 448 to rescue 6
    C. os=8, drop Master=1.0 424 units  (-25%)

## Was it really the solver? Building the LTspice comparison

To answer that, the same circuit had to run on a different solver. There was no Mesa deck, so
`parametric-devices/amps/gen_mesa_red_ltspice.py` generates one **from LiveSPICE's own resolved
netlist** (`livespice_cli --circuit ... --netlist`, 369 components, every terminal already
resolved). Two rules made the comparison meaningful rather than decorative:

* topology from the netlist, never re-transcribed from the schematic — a hand-copy introduces
  differences that look like solver differences;
* the tubes emit **the same equations LiveSPICE solves** (Dempwolf-Zölzer triodes from
  `Triode.cs:162`, its Koren-form pentode from `Pentode.cs:103`) as B-sources, rather than a
  stock LTspice tube model. Swapping in Koren triodes would confound "different solver" with
  "different device physics".

### Validation caught five real transcription bugs

The deck was checked against a LiveSPICE capture of a knob setting both backends handle, and the
first attempt correlated **0.32**. Each bug below was found by reading LiveSPICE's source, not by
guessing:

| bug | source | effect |
|---|---|---|
| Transformer as coupled inductors | `CenterTapTransformer.cs:44` — it is IDEAL, pure constraints, no magnetising inductance | false LF roll-off and plate loading |
| `out_scale` too small | peak pinned at exactly 50.0 = 1/0.02 | `.wave` clipped the output |
| Push-pull sensing inverted | `AddTerminal(sa,-Isa)` vs `(sc,+Isc)` — the halves carry OPPOSITE currents | primary saw `Isa-Isc`, not `Isa+Isc` |
| Pot taper guessed as `x²` | `VariableResistor.cs:88` — it is `(e^2x-1)/(e²-1)`, and reverse forms flip x BEFORE the curve | wrong knob positions (at x=0.5 the guess gave 0.75 where LiveSPICE gives 0.269) |
| **Turns ratio inverted** | `Ratio.cs:33` — `implicit operator` is `n/d`, so `"1:23"` = **0.0435**, not 23 | step-UP instead of step-down output transformer |

Also confirmed as NOT bugs, each by reading the source: `Speaker` really is
`Resistor.Analyze(..., Impedance)` so `Infinity` genuinely means open circuit; `NamedWire`
entries are already resolved to node names; and the `BIAS` rail is orphaned in the schematic
itself (the pentode grids reference ground via `RgsV6`/`RlPIB`), so LiveSPICE runs unbiased too.

Correlation moved −0.46 → +0.60 across these fixes. `AddTerminal(node, i)` means i flows OUT of
the node — derived from `Analysis.cs:150`, where `AddPassiveComponent(a,c,i)` is
`AddTerminal(a,i) + AddTerminal(c,-i)`. Getting that backwards inverts the whole amp.

## The result: LTspice does not rescue this circuit

With the transformer correctly ideal, LTspice fails on its own terms:

    Fatal Error: Analysis: Time step too small; time = 1.264, timestep = 1.25e-15:
                 trouble with node "nspk"

Two mitigations were tried and neither helped: a microhm series resistance to damp the E/F
algebraic loop (moved the failure from the sense branch to the primary node), and replacing the
open-circuit speaker with a finite 8 Ω load (failed at the same time, t=1.264 vs 1.263).

**The timing is the interesting part.** The excitation opens with 1 s of lead silence, so LTspice
dies 0.26 s after signal onset — and LiveSPICE's spikes cluster at 0.93–1.05 s, i.e. the same
silence→signal transition. Both solvers struggle in the same place, which points at the CIRCUIT
being genuinely stiff there rather than at either implementation.

### What this means in practice

* Neither backend is simply "more robust". They have different fragilities: LiveSPICE emits
  isolated Newton overshoots; LTspice collapses its timestep.
* **LiveSPICE's ideal-component idioms do not translate to conventional SPICE.** An ideal
  transformer and an `Impedance = Infinity` speaker are solvable symbolically in its MNA and are
  numerically hostile to a discretising solver. Any future schx→deck port of an amp with an
  output transformer will hit this.
* Switching backends is not a fix for the `Master = 1.0` corner, so the dataset options are the
  ones weighed above: keep 393 with the ladder, or drop `Red Master = 1.0` for a complete 392.

## Reusable outcomes

* `gen_mesa_red_ltspice.py` — netlist-driven schx→LTspice generator; the mapping table
  (ideal transformer, tapers, tube B-sources, MNA sign convention) applies to any LiveSPICE circuit.
* `stability_sweep.py` — finds unstable knob REGIONS before a render, from a short probe, and
  reports the cheapest stable oversample. The 3 s probe reproduces what the 120 s clip does
  because the spikes cluster at the silence→signal onset.
* `livespice_cli --progress` + adaptive stall detection — removed the guessed wall-clock timeout
  that had made 55 combinations unrenderable purely by running out of clock.
