# Proposal: damped Newton in our LiveSPICE fork

> **STATUS: PROPOSAL.** Nothing here is implemented. The diagnosis below IS measured — every
> row of the elimination table was run, not reasoned about.

## The failure

A high-gain amp at `Gain=0.95, Master=0.10` renders output containing isolated single-sample
excursions that cannot be circuit behaviour:

```
index 320137:   -1: +0.740  →  +10.707  →  +1: +1.195     (8.9x the larger neighbour)
index 339933:   -1: +3.378  →  +13.893  →  +1: +3.291     (4.1x)
```

The driving signal at those instants is **20–150 Hz** (~2400 samples/cycle). Output cannot go
1 V → 11 V → 1 V in 41 µs on that input. `gen_dataset_from_schx.py`'s spike detector correctly
rejects the render; the combination then walks the escalation ladder and consumes hours.

## What it is not

Each row was measured on a 12 s reproduction clip, not inferred:

| hypothesis | result |
|---|---|
| Newton **iteration exhaustion** | eliminated — iterations 8→4096 (512x) give BYTE-IDENTICAL output and FLAT runtime |
| **timestep resolution** | eliminated — oversample 8/16/32 identical; ladder to 256 cannot help |
| **topology defect** | eliminated — `preflight.py` PASSES all five knobs, no generator drift, `.schx` is the post-fix build |
| **near-Nyquist content** | eliminated — the documented false-positive class needs ~12 kHz; this is 20–150 Hz |
| **capture-sweep single-sample impulses** | eliminated — only 21% of flagged samples coincide, and not the ones that trip the gate |
| **missing grid capacitance** | **NOT eliminated — this FIXES it.** See "Capacitance" below (2026-09-14) |
| **a previously-resolved wiring defect** at the opposite Master extreme | eliminated — that corner now renders clean |

The flat runtime across a 512x iteration range is the decisive one: the loop is **breaking out
early at every timestep**, including the bad ones. The solver converges in under 8 iterations,
confidently, to a wrong answer.

## Why

`Circuit/Simulation/Simulation.cs`, in the emitted Newton loop:

```
v += dv                          // full step, always — no damping, no clamp
done &= (|dv| < |v|*1e-4)        // converged when the STEP is small
if (done) break;
```

Two properties combine badly. The step is applied **undamped**, so one poorly-conditioned
Jacobian can carry the iterate into a different root of a nonlinear device curve. And the
convergence test measures the **step**, not the answer — a wrong root has a tiny final `dv`
and is declared converged. Nothing in `Circuit/` or `ComputerAlgebra/` limits step size; the
grep is empty.

## Tested: a global damping clamp is NOT the fix

Implemented and measured (2026-09-06) before proposing further work. The patch adds a
`DampingFactor` property mirroring `Iterations`, clamping each update to
`factor * max(|v|, 1)` before it is applied, exposed as `livespice_cli --damping`. Built in
an isolated tree so the live binary was never touched; `--damping 0` reproduces the unpatched
binary byte-for-byte, so the control is exact.

On the 12 s reproduction fixture:

| damping | time | result |
|---|---|---|
| 0 (control) | 86 s | 2 spikes, worst \|14\|, peak 14.683 |
| **0.01** | **296 s** | **0 spikes** — but peak 17.169, and see below |
| 0.1 | 21 s | byte-identical to undamped — clamp never binds |
| 0.5 | 99 s | **diverges**, 41,022 non-finite samples |
| 2.0 | 111 s | differs from undamped |
| 10.0 | 18 s | byte-identical to undamped |

Two reasons this is rejected:

**It does not repair the spikes, it replaces the answer.** The `damping 0.01` render differs
from the undamped one in **19.3% of samples by more than 10% of peak** (rms of the difference
1.41 against a signal rms of 1.97). A global clamp tight enough to stop the one bad step also
throttles every good step, so convergence is incomplete everywhere. The spikes are gone
because the whole waveform is different, not because the pathology was fixed.

**Its behaviour is non-monotonic in its own parameter.** 0.1 and 10.0 are no-ops while 0.01,
0.5 and 2.0 each change the solve and 0.5 diverges outright -- deterministically, reproduced
across a rebuild with the clamp reordered relative to the convergence test. A fix whose effect
does not vary monotonically with its own knob is not understood well enough to ship, and the
reordering experiment ruled out the obvious explanation (that clamping before `done &=` was
corrupting the convergence test -- it makes no difference).

The useful residue: the whole loop is testable in ~90 s per configuration against an exact
control, and a wrongly-damped solve fails LOUDLY (`inf`), which `gen_dataset_from_schx.py:550`
already rejects via `np.isfinite`. Getting this wrong is safe, not silent.

## Capacitance: a circuit-side fix that works (2026-09-14)

Reproduced on a second circuit and a different corner: Mesa RED (sag v30) at
`RD Gain=0.1 / Red Master=0.2` — the grid's QUIETEST cell, where the 2026-08-30 Orange
incident was its LOUDEST. Same signature:

```
oversample   8   worst |12| at sample 890319, neighbours <= 3.0, p99 5.7
oversample  64   worst |12| at sample 890319, neighbours <= 3.3, p99 5.7
oversample 128   worst |12| at sample 890319, neighbours <= 3.4, p99 5.7
```

Identical sample, identical magnitude, 16x range of timestep. (`gen_dataset_from_schx.py`'s
escalation ceiling has since been dropped from 256 to 128 on this evidence.)

**Enabling interelectrode capacitance on the triodes fixes it.** `SimulateCapacitances="true"`
with `Cgp=1.7pF Cgk=2.3pF Cpk=0.5pF` — the 12AX7 values the Soldano and Tweed builds already
carry — renders the same cell clean at oversample 8: rms 1.179 / peak 14.355, sitting
monotonically between its neighbours (0.965/10.670 and 2.150/20.393), so it is the physically
correct value and not a render that merely dodged the detector.

This is consistent with the wrong-root diagnosis rather than contrary to it. Grid capacitance
adds state at exactly the nodes carrying the nonlinear device curves, bounding dV/dt per
timestep, which keeps the Newton iterate inside the correct basin. It is a **conditioning fix
at the circuit level**, which is why it succeeds where both timestep refinement and global
damping fail.

It costs about **5.5x render time**, and it is not free of side effects: Tweed, the one circuit
here that has always had it, escalates rungs on 97.8% of combinations against 0-3% for the
Mesa builds. Capacitance buys correctness with time.

**Preferred over damping for accuracy reasons.** Damping alters the numerics everywhere to
rescue one timestep (measured: 19.3% of samples changed by >10% of peak). Capacitance adds a
real physical effect the model was missing. One changes the answer; the other changes the
circuit to be the circuit.

## The solver revision is not a nice-to-have (2026-09-14)

The Prerequisite section below was written as a precaution. It is now a measured incident.

The same capacitance-enabled `.schx`, the same excitation, the same Python, rendered on two
machines: clean on one, 500 x SIGABRT on the other. The cause was the **solver binary**:

```
blackbox   livespice_cli built 2026-09-09   -> renders clean
this Mac   livespice_cli built 2026-09-02   -> hard crash on every render
ed5613f "Carry upstream 5398a63 (capacitor current unknown)"   committed 2026-09-06
```

`ed5613f` adds capacitor currents as system variables — "we don't need to solve for
differentials before discretization" — which removes the exact elimination step that throws
`Failed to eliminate differentials from system of equations`. The Mac's binary was four days
older than its own repo HEAD, so it was still running the pre-patch solver.

Three consequences worth stating:

* A circuit-level finding recorded in `gen_mesa_dualrec_solo_full.py` on 2026-07-30 — *"enabling
  it makes this circuit's system of equations unsolvable ... do not re-attempt"* — was correct
  when written and correct on any machine that has not rebuilt. **It did not expire; the fleet
  diverged.** Nothing connected the note to the dependency that invalidated it.
* Had the render been sharded across both machines, half the dataset would have come from one
  solver and half from another, with nothing in the artifact recording it. `findpeak_cache_key`
  covers circuit bytes, params, oversample and iterations — not the solver.
* Rebuilding invalidates every cached onset on that machine, silently, because the cache cannot
  tell that the thing which produced its entries has changed.

## Enhancement: a continuity detector inside the solver

The detect-and-retry design below is worth building even without the conditional damping, and
is best sequenced in that order.

**A retry alone does nothing.** The solve is deterministic: re-running the same timestep from
the same starting point with the same solver reproduces the identical wrong root. A retry is
only meaningful if something changes, and there are exactly two candidates.

**1. A different initial guess — try this first.** Newton converges to whichever root's basin
it starts in. Linear extrapolation from the last two accepted timesteps starts near the true
solution by construction, because the true solution is continuous with them. When this
succeeds it does not alter the solve at all: the answer is ordinary undamped Newton, reached
from a better start. Whether it is *sufficient* depends on whether the bad step is caused by
where the iterate starts or by how far the first step throws it. Untested.

**2. Damp that re-solve — the fallback.** Bounds how far the iterate can move, which is the
mechanism that keeps it in-basin. Known to work in the crude global form (damping 0.01 gave
0 spikes); conditional application is what removes the collateral damage.

**The detector pays for itself before either repair exists.** Today the failure is caught
*after* rendering ~9.8M samples, the whole render is discarded, and the ladder re-renders it up
to four more times at rising cost — over 2.5 h on a single cell before it was killed. Detecting
at the offending timestep turns that into seconds, even if the response is only to fail
honestly. That alone justifies the work.

**Where it goes.** `Circuit/Simulation/Simulation.cs`, in the emitted Newton loop — the same
three lines quoted above. Note this is expression-compiler emit, built once per circuit and run
for every timestep, so "re-solve this timestep" means emitting a retry path rather than editing
a loop. It is on the path every circuit takes, which is why the cost is validation rather than
code.

## What to change

The measurement above redirects the design: damping must be **conditional**, not global.
Damp the timestep that misbehaves, leave every other timestep alone.



**1. Conditional damping.** Detect the bad timestep first, then re-solve it damped. Global
clamping is measured above and rejected -- it alters 19.3% of samples. The detector is the
continuity test in (2); damping becomes the *response* to it, not a always-on setting.

**2. A continuity detector, not a residual one.** The instinct is to re-enable the disabled
residual check (`|JxF| > eps` → `ThrowSimulationDiverged`, commented out in the same file).
It would not catch this: a converged iterate satisfies `F(v) ≈ 0` **at the wrong root too**.
Residual catches exhaustion, and exhaustion is measurably not happening here. The property
that separates a wrong root from a right one is **continuity** — whether the state moved
further in one timestep than the circuit can physically move. That test already exists in
spirit as the post-render spike detector; moving it inside the solver turns a whole-render
rejection into a one-timestep retry.

**3. Retry on detection** — re-solve that timestep from an extrapolated guess, accepting only a
solution that passes the continuity bound.

## Why this is affordable

The fork already exists: `livespice-cli`'s submodule points at our own LiveSPICE, currently
**0 commits ahead of upstream**. This is the first commit to a fork already in use, not the
adoption of a maintenance burden.

## What it costs

The change touches the expression-compiler path used by **every** circuit, so the real work is
validation, not code:

* the reproduction clip must go 2 spikes → 0 (deterministic, 45 s);
* `measure_truncation.py` and a full ESR comparison on an already-trained device, to prove the
  damped solve did not shift results everywhere else;
* flags default OFF, so existing datasets stay byte-reproducible.

## Prerequisite

**The solver revision must enter the cache key and the dataset provenance first.**
`findpeak_cache_key` covers circuit bytes, params, oversample and iterations — but not which
solver produced them. Change the solver and every cached saturation onset, and every existing
dataset, silently becomes incomparable with new ones. `config.json` should likewise record the
oracle's git SHA so a model can be traced to the solver that built it.

## The alternative worth stating

`ngspice-deck` and `ltspice-deck` already exist for circuits LiveSPICE cannot handle, and a
prior controlled comparison on this same amp found **LTspice did not rescue it** either — that
investigation ended in a circuit defect, not a solver one. What LiveSPICE uniquely provides is
`.schx` in, audio out, with no hand-written deck. That is why it is the default, and why
patching it is likely cheaper than migrating away from it.
