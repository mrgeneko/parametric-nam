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
