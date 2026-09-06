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

## What to change

**1. Damped Newton.** Clamp `|dv|` to a fraction of `max(|v|, floor)` before `v += dv`. Standard
SPICE practice. A local change at the emission site, exposed as a `DampingFactor` property
following the existing `Iterations` pattern, surfaced through `livespice_cli --damping`.

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
