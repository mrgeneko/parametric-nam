[← back to README](../README.md)

# Scaling training across machines

Rendering already distributes: `distribute_gen.sh` splits one `--grid` across machines over
SSH ([`docs/scripts.md`](scripts.md)). Training does not, and it is the larger cost — this
repo's sibling model registry records **1,874 h across 45 runs, a median of 35.1 h each, with
24 runs over a day and one at 143 h**. This documents what was measured about making that
faster, and what turned out to be false.

Everything below was measured on 2026-08-31 against real datasets, not estimated.

## Measurement 1 — a training step is GPU compute, not data loading

Profiled with `--widths 5,9`, batch 38, a 200-combination dataset, `num_workers=0`
(the trainer's setting), MPS, 60 steps after 10 warmup, every phase bracketed with
`torch.mps.synchronize()` — MPS is asynchronous, so without that every phase but the last
reads as ~0. Measured independently on two machines:

| phase | M5 Air ms | share | M4 Pro ms | share |
|---|---:|---:|---:|---:|
| data (crop fetch from mmap'd `outputs.npy`) | 7.9 | **0.4%** | 7.0 | **0.6%** |
| forward | 594.5 | 28.4% | 357.4 | 30.5% |
| loss | 49.6 | 2.4% | 31.9 | 2.7% |
| backward | 1425.9 | **68.2%** | 760.9 | **64.9%** |
| optimizer | 13.2 | 0.6% | 15.6 | 1.3% |
| **total** | **2091.2** | | **1172.8** | |

`data` is near-identical in absolute terms across the two (7.9 vs 7.0 ms) while compute differs
1.8x — exactly what you expect if loading is CPU/IO-bound and independent of the GPU.

**`num_workers > 0` is not worth doing.** Loading is 0.4% of a step. This refuted a plausible
theory: step cost *appeared* to track dataset size across three runs (1.17 s/step at 4.9 GB,
0.905 at 1.5 GB, 0.215 at 0.2 GB), which looks like an I/O bottleneck. It is not — those runs
used batch 64 / 38 / 8, and compute scales with batch. A batch-size effect was misread as a
dataset-size effect because the two co-varied.

Corollary: the 15.6 GB 675-cell dataset will **not** have slower steps for being large. Step
cost is set by batch size, which is a free parameter.

## Measurement 2 — machine throughput differs, and differently per workload

| machine | CPU cores | GPU cores | render (caps/h) | train (s/step) | train vs M3 Max |
|---|---:|---:|---:|---:|
| M4 Pro mini | 12 | 16 | 21.7 | 1.326 | 1.47x |
| M5 MacBook Air | 10 | 10 | 13.0 (fanned) / 9.0 (unfanned) | 2.390 | 2.64x |
| M3 Max MacBook Pro | 14 | 30 | 22.0 | 0.905 | 1.00x |

**Core count predicts the wrong thing depending on workload.** For *rendering* (pure CPU,
single-threaded per LTspice sim) the Air ran at half what its 10 CPU cores implied — it is
fanless and was thermally throttled; adding a desk fan raised it from 9.0 to 13.0 caps/h. For
*training* (GPU) it is 2.6x slower than the M3 Max, close to what its 10-vs-30 GPU cores
implied.

Do not carry a ratio from one workload to the other. Measure per workload.

Both newer chips also **beat what their GPU core count predicted** against the M3 Max (M4 Pro
1.47x measured vs 1.88x predicted; M5 Air 2.64x vs 3.00x), so per-core GPU throughput has risen
across generations. Core count is a starting guess, not a substitute for a benchmark —
`profile_step.py`-style timing takes minutes.

## Measurement 3 — a Linux/ROCm desktop is a render arm, not a training arm

Measured on 2026-09-16 on `blackbox` (Ryzen 5 9600X, 6c/12t, + AMD RX 9070 XT), against the real
`klon_ds` dataset (36 combinations), `--widths 4,8`, batch 32, `--crop-len 48000`, 50 steps —
close to but not identical to the Mac fleet's widths 5,9 / batch 38 above, so treat the
cross-machine ratios as directional, not exact:

| device | s/step |
|---|---:|
| RX 9070 XT (ROCm 7.2, `--amp fp16`) | 7.69 |
| Ryzen 5 9600X (CPU, `--amp off`) | 7.15 |

**The GPU is slower than its own CPU on this box.** This repo's models are tiny (the
SlimmableContainer trained here has 3.7k–13k weights) — too little compute per step to amortize
ROCm/HIP's per-kernel dispatch overhead, so a fast desktop CPU keeps up with or beats the GPU.
This matches a real production run (the Klon Centaur release, same dataset/widths/batch,
`--device auto` picked the GPU): 414.6 s/epoch, i.e. 7.4 s/step — consistent with the measurement
above, not a fluke.

Against the Mac fleet from Measurement 2, blackbox's best mode (CPU, 7.15 s/step) is still
**3.0x slower than the M5 Air** (2.390), 5.4x slower than the M4 Pro mini, and 7.9x slower than
the M3 Max. Even the fleet's slowest, fanless, thermally-throttled machine beats blackbox at
training by 3x.

**Conclusion: blackbox is a render worker, not a training candidate.** Rendering (SPICE
simulation) is CPU-bound and its 12 threads help there (see `distribute_gen.sh`); training does
not benefit from adding it to the fleet's training rotation.

## Option A — parallel schedule search (no code changes)

Run the same dataset on several machines with **different schedules**, same seed and same step
budget, and compare. The runs are independent — no communication at all — so this is
embarrassingly parallel in a way data parallelism is not.

Why it is the higher-value lever: it attacks **how many steps convergence needs**, which is
what makes these runs long. The 143 h run had **9 combinations** — it was not slow because the
data was big, it was slow because it took 4,800 epochs to settle.

The one comparison already run (`--restart-mult 2 --restart-decay 0.85` vs equal cycles)
gave **3.7x better ESR in 35% fewer steps**. But it changed four variables at once
(`restart-period`, `restart-mult`, `restart-decay`, `lr`) *and* warm-started, so **it cannot
be attributed**. Disambiguating it is the point of the search.

What makes a comparison valid:

* **Fixed step budget, not wall time** — machines differ ~2.6x, and an arm given more steps
  wins trivially.
* **Same `--seed`** — the val split uses its own `torch.Generator().manual_seed(args.seed)`,
  independent of global RNG, so the split is identical across arms regardless of what else
  changes.
* **Judge on `per_combo_esr_wN.csv`, not the val split** — val is 1-34 combinations here and
  noisy enough to pick noise.

Suggested arms, chosen to separate the confound:

| arm | schedule | question |
|---|---|---|
| control | `--restart-period 50 --restart-mult 1 --restart-decay 0.97` | pre-2026-09-20 defaults (`--restart-mult` now defaults to 2 -- see below) |
| geometric | `--restart-period 150 --restart-mult 2 --restart-decay 0.85 --stale-cycles 2` | the unattributed winner |
| long-equal | `--restart-period 400 --restart-mult 1 --restart-decay 0.9` | is it geometric growth, or just cycles long enough to amortise? |

That third arm matters. The fixed post-restart recovery cost (~80 of 150 epochs, measured over
23 cycles) is amortised by *long* cycles just as it is by *growing* ones — and long equal cycles
keep a working auto-stop, which `mult=2` does not (see below).

**Layout matters more than machine count.** One arm per machine is paced by the slowest, so
three arms on three machines costs the same as three arms serially on the M3 Max — no saving.
Two arms sequentially on the M3 Max plus one on the M4 Pro is ~20 h against ~30 h serial.

### Known defect in `--restart-mult 2` — fixed by `--restart-max-period` (2026-09-15)

`--stale-cycles` counts **cycles**, and `mult=2` grows them geometrically (150, 300, 600, 1200,
2400, 4800). By cycle 6 the auto-stop needs ~9,600 epochs of no improvement to fire. The EQ-test
run plateaued 1,665 epochs before it would have triggered and had to be stopped by hand via the
`STOP` file; left alone it would have ground ~8 further hours.

**`--restart-max-period N` caps cycle growth at N epochs**; every later cycle stays that length
(mult effectively reverts to 1 from there). **It defaults to 1200 — on.** Pass
`--restart-max-period 0` to opt out and restore the historical uncapped behavior exactly.

The cap also made it possible to fix the plateau-stop rule outright — see
[The plateau rule](#the-plateau-rule-stale-epochs-replaced-stale-cycles) below.

Two deliberate properties of the default, both so it can't surprise an existing config:

- It is **inert at `--restart-mult 1`**, which never grows cycles. **UPDATE 2026-09-20:**
  `--restart-mult` itself now defaults to `2` (see "Making `--restart-mult 2` the default"
  below), so this cap is now load-bearing on a *default* run too — only an explicit
  `--restart-mult 1` makes it inert again.
- If `--restart-period` is itself ≥ 1200, the default **disables itself with a notice** rather
  than pinning every cycle to the period you asked for, which would silently turn your
  `--restart-mult` into a no-op. An explicit `--restart-max-period` still wins there.

It applies at the next cycle **boundary**, never mid-cycle, so enabling it on a `--resume`
cannot disturb an in-flight cycle.

A fractional `--restart-mult` (e.g. 1.5) was the previously proposed fix for the same problem,
but **PyTorch rejects it**: `CosineAnnealingWarmRestarts` raises `Expected integer T_mult >= 1`,
so it would need a custom scheduler whose resume arithmetic stays reproducible. The cap gets the
same bounded-cycle property with no custom scheduler — it clamps `T_i` at each cycle boundary,
which is stable because the bare `step()` branch is purely incremental and never re-derives `T_i`.

#### Why 1200, and why not tighter

Replaying every `mult=2` run's `metrics.csv` (7 runs / 29 cycles) says **doubling earns its keep
far longer than expected** — do not cap tighter on intuition:

- The **second half** of each cycle delivers a near-constant ~1.2x ESR gain regardless of how
  long that half is. One dual-rectifier run's second halves: 1.21 / 1.24 / 1.22 / 1.19 / 1.13x
  across cycles of 100 / 200 / 400 / 800 / 1347 epochs. If cycles were too long this figure
  would decay toward 1.00x. It doesn't.
- The final new global best lands at **≥90% of cycle length in 19 of 23** amp/pedal cycles;
  tail waste after the last improvement is typically 2–10%.

What does fail is the far end. The one run that reached cycles of 2400 and 4800 (a 4-knob
EQ-ish pedal already at its knob-count ESR ceiling) spent 2400 epochs for 1.10x, then wasted
the last **30%** of its 4800-epoch cycle — last new best at 70% of it. 1200 is the largest
cycle length in the dataset that still showed a live tail.

Caveat on the evidence: "improvements arrive late in a cycle" is partly intrinsic to cosine
annealing (a best always tends to land near a trough), so the within-cycle profile alone can't
prove a length is optimal. The second-half-gain-vs-length figures are the part that actually
discriminates.

### The plateau rule: `--stale-epochs` replaced `--stale-cycles` (2026-09-16)

`--stale-cycles 3` was the default stopping rule. Simulating both rules against **41 distinct
real runs**' own cycle structures and improvement timelines showed it is not merely wasteful but
**unsafe**:

| rule | fired early | worst ESR forfeited |
|---|---|---|
| `--stale-cycles 3` | **14 of 41 runs** | **2.615x** |
| `--stale-epochs 1500` | 0 of 41 | 1.000x (nothing) |

Worst case: a distortion-pedal run where the cycle rule fires at epoch 3198, but the run kept
minting new bests until **9321**. Others forfeited 1.69x, 1.59x, 1.52x, 1.45x.

**Defaults as shipped 2026-09-16** (both gated on the cap being active): `--stale-epochs` =
`max(1500, 1.25 × cap)` = 1500, and `--stale-cycles` = 0. With `--restart-max-period 0` the old
pairing (`--stale-cycles 3`, `--stale-epochs 0`) is kept instead.

Two things made this work, neither of which was true before:

1. **The cap defuses the cosine-tail objection.** `--stale-epochs` was documented as a blunt
   instrument because a best tends to land near each LR trough, so an epoch counter can fire
   mid-cycle during a high-LR stretch. With every cycle ≤ the cap and patience > the cap, any
   window of that many epochs necessarily spans a complete cycle, trough included. Hence the
   `1.25 ×` coupling rather than a bare constant.
2. **1500 is measured, not guessed.** The longest drought ever *followed by* further improvement
   was 1164 epochs (the 4-knob Joyo run at its knob-count ESR ceiling); next worst 664,
   everything else ≤303. 1500 clears the worst case with 1.29x margin. The `max(1500, …)` floor
   stops a lowered cap from dropping patience under that.

Note the rules are OR'd, so enabling the epoch rule *without* disabling the cycle rule would
change nothing in precisely the 14 dangerous cases — the cycle rule fires first. It had to be a
replacement. For the same reason `run_pipeline.py` no longer forwards either flag unless
explicitly passed; it used to forward `--stale-cycles 3` unconditionally, which would have
overridden the new per-run choice.

#### Lowered to a flat 750, decoupled from the cap (2026-09-20)

`--stale-epochs` now defaults to a **flat 750**, no longer coupled to `--restart-max-period` at
all — `DEFAULT_STALE_EPOCHS` in `param_train.py` on request, to shorten the default plateau wait
below what the `1.25 ×` coupling forced (1500 at the standard 1200 cap).

This is **not a free win** — it gives up real safety margin the 1500 default had, and the
tradeoff was made without re-running the 41-run simulation at the new value:

- **It reopens exactly the case 1500 was sized to cover.** 750 sits strictly between the 41-run
  dataset's two longest real droughts (664, comfortably cleared; 1164, the 4-knob Joyo run at
  its knob-count ESR ceiling, now missed) — so it forfeits that one run early, same as the old
  `--stale-cycles 3` default did to 14 different runs. Unlike those 14 cases, **the magnitude of
  what 750 forfeits on the Joyo run was never measured** — no value between 665 and 1163 was
  simulated against the real data, so there is no equivalent "1.69x, 1.59x, …" figure for this
  decision the way there was for the old cycle rule's failures.
- **It drops the trough-coverage guarantee entirely.** 750 < 1200 (the cap default), so patience
  is no longer `> max_period` — the "any stopping window spans a complete cycle" property from
  point 1 above no longer holds. A run whose cycle has grown (via `--restart-mult > 1`) past 750
  epochs can now be stopped mid-cycle, on a high-LR stretch, before that cycle's own trough — the
  exact cosine-tail failure mode the cap was built to defuse. **No longer inert by default**
  as of the same day: `--restart-mult` itself now defaults to `2` (see below), so a *default*
  run's cycles will grow past 750 epochs (the 800-epoch cycle, the 5th restart) before the
  1200 cap flattens them — only an explicit `--restart-mult 1` keeps cycle length reliably
  under 750 now.

If a future run's plateau stop looks premature and its cycle reached or exceeded 750 epochs
(true for any sufficiently long run at the current default, not just an explicit `mult>1`
one), this coupling loss is the first thing to check.

### Making `--restart-mult 2` the default (2026-09-20)

`--restart-mult` changed from `1` to `2` as the fleet-wide default. The unconfounded part of
the case for this: every full-LR restart pays a roughly fixed recovery cost regardless of
cycle length (~7-epoch avg, ~30-epoch worst-case), so at equal-length cycles that cost is a
constant *fraction* of the whole run (measured ~54% at 150-epoch cycles) — growing cycles
geometrically shrinks that fraction toward zero instead (~9% on the same budget). That much
holds regardless of whether growth is the right SHAPE for the amortization win.

What is **not** settled: the one real before/after comparison ("Option A" above) that showed
3.7x better ESR in 35% fewer steps changed four variables at once and warm-started, so it
cannot be attributed to geometric growth specifically — the `long-equal` arm proposed above
(same amortization via a longer *flat* `--restart-period`, no growth at all) has never
actually been run. The default was still changed, on the strength of the unconfounded
fixed-cost argument alone — but if that 3-arm experiment is ever run and `long-equal` matches
`geometric`, this default should revert to `1` with a longer `--restart-period` instead of
staying pinned to an unproven growth-shape claim.

#### Trend-based rules were tried and lost

A rate-based rule is the obvious improvement — any patience rule wastes exactly X epochs by
construction. Measured against the same 41 runs, every variant was **worse than flat patience**:

| rule | fired early | worst forfeited |
|---|---|---|
| per-cycle gain < 1.02, 2 consecutive | 30 of 41 | 6.878x |
| normalised per-100-epoch gain, best of 16 threshold/K/warmup combos | 15 of 41 | 2.615x |
| adaptive patience (2.5 × longest rewarded drought, floor 500) | **0 of 41** | 1.000x |
| `--stale-epochs 1500` | **0 of 41** | 1.000x |

The adaptive variant matches flat patience on safety but not on cost — total waste across the
fleet was 6,929 epochs against 6,752 for the flat rule, i.e. slightly worse for real added
complexity.

The reason is structural: **improvement in these runs is bursty, and long droughts are routinely
rewarded.** A flat stretch of no progress looks identical whether it precedes a 2x gain or
nothing at all, so a rule that infers "the trend has gone flat" is reading exactly the signal a
rewarded drought also produces. There is no separating statistic in the data. Dumb patience wins
because it does not try to predict — it just waits long enough that a rewarded drought cannot be
mistaken for a plateau. A trend rule would need a feature that actually distinguishes the two
(per-combo ESR structure, or gradient/loss-landscape signal), not a smarter function of the same
val-ESR series.

## Option B — data-parallel training (~1 day of work, ~1.5x)

The profile says 96.6% of a step is forward+backward, so data parallelism attacks the real
cost. Gradient sync is negligible: the model is ~17k weights (68 KB) against ~2 s of compute.

**`DistributedDataParallel` cannot be used as-is.** Verified on torch 2.12.1:

```
gloo available: True
cpu allreduce    OK
MPS allreduce    FAILS: 'c10d::allreduce_' is not currently implemented for the MPS device
DDP on MPS       FAILS: 'c10d::allgather_' is not currently implemented for the MPS device
```

The collectives are not implemented for MPS. Gradients must be staged manually:

```python
for p in model.parameters():
    g = p.grad.cpu()                       # 68 KB total
    dist.all_reduce(g)                     # gloo on CPU tensors -- works
    p.grad.copy_(g.to("mps").div_(world_size))
```

That mechanic is ~10 lines. **The work is making `param_train.py` rank-aware**: `best_state`
tracking, checkpoint writes, the SGDR plateau detector, per-combination ESR and export all
assume one process and must run on rank 0 with the stop decision broadcast, plus
`DistributedSampler` in place of `shuffle=True`.

Expect **~1.7x, not 2x**, for the M3 Max + M4 Pro pair (measured 0.905 and 1.326 s/step; a
throughput-proportional batch split of roughly 23/15 puts both at ~0.54 s against 0.905 solo):

* All-reduce is synchronous, so every step is paced by the **slowest rank**. An even 19/19
  batch split between the M3 Max and M4 Pro would gain almost nothing — the slower machine
  doing half the work still takes nearly as long as one machine doing all of it. The split must
  be proportional to measured throughput (~23/15 for this pair), with weighted gradient
  averaging.
* **The M5 Air cannot participate** at 2.6x slower; it would pace every step.
* Halving a batch does not halve GPU time — fixed per-kernel overheads make small batches less
  efficient, so real speedup lands under the arithmetic.
* Smaller per-rank batches change the **optimisation**, not just its speed, unless the global
  batch is raised to compensate — which makes prior runs non-comparable.

## Recommended order

1. **Schedule search.** No code, attacks epoch count, and disambiguates a result we already
   have but cannot explain. Applied across 45+ future runs.
2. **Data parallelism**, if training is still the bottleneck afterwards. Roughly a day's work
   for ~1.5x on two machines.
3. **Not `num_workers`.** Measured at 0.4% of a step.

They compose: one shortens each step, the other reduces how many steps are needed.
