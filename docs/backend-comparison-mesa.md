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
entries are already resolved to node names; and one supply rail is orphaned in the schematic (a circuit finding, tracked in
`parametric-devices`)
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

**The timing.** LTspice dies at t = 1.2635 s — the **exact first nonzero sample** of the
excitation — at the power-amp node (`nspk`). That is an adaptive-timestep collapse when the
solver must shrink dt after coasting through 1.26 s of digital silence.

For LiveSPICE the honest answer is that **the failing renders' spike positions were never
recorded**, for a reason worth documenting (below). Positions are known only for combos
re-rendered with the WAV retained. On combo 56 (`RD Gain=0.1, Red Master=1.0`, EQ 0.2),
os=8/iterations=256, the full 120 s render produces 14 isolated overshoots at **15.4–109.4 s**,
every one of them in ordinary, smooth, band-limited signal.

### A logging bug that produced phantom evidence

An earlier draft of this section claimed LiveSPICE's spikes clustered at 1.30–11.57 s, with a
"late cluster" sitting inside a 2.26 s silent gap. **That was wrong, and the numbers behind it
were never real.** `gen_dataset_from_schx.py` logged escalations as `r.error[:80]`, and 80
characters lands mid-number in the sample index:

    solver spike: 14 isolated single-sample overshoot(s); worst |42| at sample 5072724 (neig...
                                                                                    ^ cut here
    solver spike: 14 isolated single-sample overshoot(s); worst |42| at sample 50727

True index **5072724** logged as **50727**. Every logged index was a 5-digit prefix of a larger
number, which is why they all appeared to fall in the first 11.6 s of a 120 s render — a
suspiciously tidy pattern that should have been the tell. A whole line of analysis was built on
those phantom indices before the truncation was spotted.

Fixed by keeping the limit generous and making any cut visible with an ellipsis, so a mangled
value can never again masquerade as a plausible one. **The lesson generalises: a silent truncation
that yields a well-formed value is worse than one that yields an obvious mess.**

### Ruled out: the T3K sweep's single-sample impulses

The T3K sweep contains sharp one-sample amplitude jumps, and single-sample discontinuities are a
plausible way to stress a Newton solver. Tested directly against the re-rendered combo 56 — using
spike positions computed from the waveform, not from logs — the hypothesis **is not the cause.**

The impulses are real. At t = 11.5 s and 12.5 s the file holds a lone sample of +15.4369 V
surrounded by exact zeros — latency-calibration impulses:

    x[552000-2 .. +2] = [0.  0.  15.4369  0.  0.]

But no spike sits near one. At all 14 overshoots the input is ordinary smooth signal:

    out-spike t      y       input x   local RMS   max|step| +-64   nearest input glitch
      15.4023 s   -37.66    -2.5716      5.4834         0.17989          -0.5979 s
      96.0675 s   -39.58    -1.2889      0.8660         0.08952         -22.6233 s
     106.6817 s   -41.57    -2.3511      5.5134         0.12205         -12.0090 s

Maximum sample-to-sample step anywhere near a spike is 0.07–0.18 V against a local RMS of
0.6–11 V — perfectly band-limited. The nearest isolated input discontinuity is 0.6–22.6 s away.

The signature is also informative: every overshoot is **negative**, clustered at −37 to −41.6 V,
and the sample immediately preceding each is consistently ≈ −11 V. That looks like Newton landing
on a spurious solution branch, not like a transient being tracked badly.

### Ruled out: near-singular pivots

The Gauss-Jordan solve divides the pivot row — including the augmented RHS column — by the pivot
(`Abj[ij] *= 1.0/p`), and upstream only skips `p == 0` exactly, so a tiny-but-nonzero pivot would
scale the extracted Newton delta enormously. A prototype small-pivot guard was built to test this
(a `Simulation.MinPivot` threshold plus a `--min-pivot X` flag, default 0.0 bit-identical to
upstream, so the default was provably inert and the measurement trustworthy).

It is a cliff, not a curve:

    --min-pivot   overshoots   rms      % of baseline rms
    (pristine)        17       8.3183      100.0%
    1e-9              17       8.3183      100.0%   bit-identical to pristine
    1e-8              17       8.3183      100.0%   bit-identical to pristine
    1e-7               0       0.1024        1.2%   amp destroyed
    1e-6               0       0.2290        2.8%   amp destroyed

No pivot falls in (0, 1e-8], so the spikes are not near-zero-pivot divisions; and the pivots
between 1e-8 and 1e-7 are load-bearing, so freezing them cuts the signal path. **There is no
threshold that removes the spikes and keeps the circuit.** Hypothesis refuted.

**The prototype was removed once it had answered the question** — `extern/LiveSPICE` is back on
its pinned upstream commit and `livespice_cli` carries no `--min-pivot` flag. Keeping a dead
lever in a reference implementation costs more than re-deriving it: the patch is small, and this
table is the reason nobody needs to. If it is ever revisited, note that the guard must go in
BOTH `Solve` and `SolveVector` (the solver is chosen at codegen time by
`Vector.IsHardwareAccelerated`), and that `MinPivot = 0.0` reproducing upstream bit-for-bit is
the check that makes any result from it believable.

### Ruled out: the DC-initialisation warning (it is real, and benign)

`TransientSolution.cs:112-120` solves the steady state per partition with `NSolve` from an
**all-zeros initial guess**, and on any exception does this:

    catch (Exception)
    {
        Log.WriteLine(MessageType.Warning, "Failed to find partition initial conditions, simulation may be unstable.");
    }

The partition's initial conditions are dropped and those states fall through to `Simulation.cs`'s
`init is Constant ? (double)init : 0.0` fallback. This circuit emits that warning on **every**
render, on stdout, where `gen_dataset_from_schx.py` (which parses stdout only for `DSP load:` and
`Processing time`) discards it.

**Cause found, and it is not the spikes.** A DC-path analysis of the netlist (capacitors treated
as open, tube grids as non-conducting) identified the offending partitions -- series-capacitor
midpoints with no DC path to ground -- and adding 1 MOhm bleeders removes the warning entirely.
It changes nothing that matters: overshoots went 17 -> 18 and rms moved 0.5 %. Those midpoints'
DC potential genuinely *is* zero, so the `0.0` fallback is already the right answer and the
warning is cosmetic. *(Which nodes, and whether to add the bleeders, is a circuit question --
see the device's build doc in `parametric-devices`.)*

**The pipeline lesson is the swallowed warning, not the circuit.** LiveSPICE reports a real
diagnostic on stdout and `gen_dataset_from_schx.py` parses stdout only for `DSP load:` and
`Processing time`, discarding everything else on success. A backend telling us the DC solve
failed should not be invisible. Worth surfacing warnings rather than dropping them.

## RESOLVED: the cause was the circuit, not the backend

Everything above chased the solver and the backends. **The cause was a mis-transcribed tone
stack.** Six wiring defects in the Red stack, mirrored in Orange, emitted by
`parametric-devices/amps/gen_mesa_dualrec_solo_full.py` -- component VALUES were right
throughout, only the wiring was wrong.

**The circuit findings, the generator fix and the before/after preflight numbers live in
`parametric-devices/amps/Mesa Boogie Dual Rectifier Solo Head RED Full.md`**, which is the
per-circuit build doc and the right home for them. They are deliberately NOT duplicated here.
Fixed and pushed by a parallel session as `59002ac`; all five builds regenerated.

What that resolution means for THIS document, which is about the backends and the pipeline:

* The `Master = 1.0` corner was never a solver-robustness question. Correcting the wiring took
  it from 17 isolated overshoots to **0** on a 20 s render, with the amp intact
  (rms 8.32 -> 7.49). Preflight went from 1 hard FAIL to PASSED on both channels.
* Every solver-side hypothesis below was correctly refuted, and the refutations are the durable
  part: iteration count, timestep resolution, pivot conditioning, determinism and platform
  arithmetic were all measured and eliminated. A topology defect is invisible to all of them.
* The one solver-side lever that appeared to work -- supply source impedance -- was damping the
  symptom of the wiring defect, not the cause. Recorded below because the dose-response is a
  useful diagnostic technique, NOT as a fix.
* Two open findings elsewhere closed as symptoms of the same defect: `Presence` reading REVERSED
  on the RED build (`-12.6%` preflight, since `+28.7%`; independently corroborated here at
  `-2.18 dB -> +1.36 dB` treble:bass), and the reduced 9x8 Orange knob grid, which can go back to
  the full 9x9 now the corner renders.

### Diagnostic technique: dose-response on supply source impedance

*(Superseded as an explanation -- see RESOLVED above. Kept because the dose-response
METHOD is reusable: sweeping one component across orders of magnitude to see whether a
symptom tracks it monotonically is how the solver-side hypotheses were eliminated.)*

Three experiments on the same 20 s render of the `Master = 1.0` corner (combo 56, os=8), each
changing ONE variable, converge on one story.

**Newton is not under-iterating.** Renders at `--iterations` 256, 1024 and 4096 are **bit-identical**
(SHA-256). The loop breaks on its own convergence test long before 256, so the ceiling is never
reached. Raising it cannot help, and the retry ladder pinning iterations at 256 was never the
limitation.

**Timestep resolution is not the axis either.** The fleet exhausted oversample 8 -> 64 on these
combos and an earlier run went to 128, with the spikes intact throughout.

**Supply source impedance controls it completely, and monotonically:**

    Rsrc        overshoots   peak     rms      note
    0.01 Ohm        97       42.334   8.6055   sag removed
    39 Ohm (stock)  17       39.885   8.3183
    200 Ohm          0       30.430   6.7559   spikes GONE, amp intact (-1.8 dB)

Note what makes this different from every other intervention that reached zero overshoots: the
min-pivot guard hit zero by flattening the output to 1.2 % of baseline RMS. This reaches zero at
81 % of baseline RMS. The amp still works.

**What this showed at the time.** Newton *converges* -- and the converged answer still contains an isolated
-40 V sample. So it is converging to a **different root**. The system is nonlinear with multiple
solutions in this region; started from the previous sample's state, the solve occasionally lands on
a spurious branch for exactly one timestep, then returns. That explains every negative result at
once: iteration count cannot help (it converges either way), oversample cannot help (each timestep
still has multiple roots), but damping *can*, because it removes the region where the extra root
exists. The signature agrees -- every overshoot is negative, clustered at -37..-41.6 V, each
preceded by a sample at ~-11 V. That is a branch, not a tracking error.

`RailC` is the coupling node: it ties all four output plates (via the transformer centre tap), all
four screen grids through 1 kOhm each, and both phase-inverter plate loads. With a stiff rail that
is a zero-impedance common node between the output stage and its own drive. Series resistance damps
it. This is also why the failures are exclusively at `Red Master = 1.0`, where current draw through
that shared node is greatest.

**What this does NOT establish.** A single-sample width is not physical -- a real oscillation spans
many samples -- so the circuit is genuinely marginal AND the solver reports that marginality badly.
Raising `Rsrc` moves the operating point out of the marginal region; it does not show the solver is
blameless.

**Both component-value leads this technique produced were WRONG**, and that is the point worth
keeping. Each produced a clean, monotonic dose-response; neither was the cause. The specific
values, the measurements behind them, and what was ultimately applied are circuit questions and
live in the device's build doc in `parametric-devices`.

**The transferable lesson: a lever that moves a symptom monotonically is not thereby a cause.**
Both leads looked compelling on exactly the evidence this document was gathering, while the
actual defect -- a mis-wired tone stack -- was untouched by either. Netlist inference produced
four wrong "fixes" this session against one right answer, and the right answer came from
node-by-node comparison against an authoritative schematic, not from sweeping the model.

### Ruled out: non-determinism

Renders are bit-identical (SHA-256) across: repeated runs, 10 concurrent renders, parameter
ordering, presence of `--sr`, and input length (a 20 s render equals the first 20 s of the 120 s
render exactly). This Mac and the fleet's worker3 report identical spike counts, magnitudes and
indices for combo 56. Earlier speculation about cross-platform arithmetic — `Vector.IsHardwareAccelerated` selecting scalar vs SIMD solvers, FMA contraction, libm differences — is a real
mechanism but is **not** what is happening here, and the apparent machine divergence was an
artifact of the truncation bug above.

Note the fleet data cannot rank platforms in any case: of the four workers, **only worker3 ever
received a `Red Master = 1.0` combo**. Workers 0/1/2 rendered zero of them, so their clean records
are a sharding artifact, not evidence of stability.

### What this means in practice

* Neither backend is simply "more robust". They have different fragilities: LiveSPICE emits
  isolated Newton overshoots; LTspice collapses its timestep.
* **LiveSPICE's ideal-component idioms do not translate to conventional SPICE.** An ideal
  transformer and an `Impedance = Infinity` speaker are solvable symbolically in its MNA and are
  numerically hostile to a discretising solver. Any future schx→deck port of an amp with an
  output transformer will hit this.
* Switching backends was never going to fix the `Master = 1.0` corner -- the defect was in the
  circuit. All renders of the pre-fix topology were discarded; the dataset needs a full
  re-render of all 448 combinations, not a patch of the 55.

## Reusable outcomes

* `gen_mesa_red_ltspice.py` — netlist-driven schx→LTspice generator; the mapping table
  (ideal transformer, tapers, tube B-sources, MNA sign convention) applies to any LiveSPICE circuit.
* `stability_sweep.py` — finds unstable knob REGIONS before a render, from a short probe, and
  reports the cheapest stable oversample. Note the probe length must be chosen against measured
  spike positions, not assumed: a 13 s probe missed this circuit's first real spike at 15.4 s.
* `livespice_cli --progress` + adaptive stall detection — removed the guessed wall-clock timeout
  that had made 55 combinations unrenderable purely by running out of clock.
