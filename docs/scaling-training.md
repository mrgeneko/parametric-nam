[← back to README](../README.md)

# Scaling training across machines

Rendering already distributes: `distribute_gen.sh` splits one `--grid` across machines over
SSH ([`docs/scripts.md`](scripts.md)). Training does not, and it is the larger cost — this
repo's sibling model registry records **1,874 h across 45 runs, a median of 35.1 h each, with
24 runs over a day and one at 143 h**. This documents what was measured about making that
faster, and what turned out to be false.

Everything below was measured on 2026-08-31 against real datasets, not estimated.

## Measurement 1 — a training step is GPU compute, not data loading

Profiled with `--widths 5,9`, batch 38, a 200-permutation dataset, `num_workers=0`
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

## Option A — parallel schedule search (no code changes)

Run the same dataset on several machines with **different schedules**, same seed and same step
budget, and compare. The runs are independent — no communication at all — so this is
embarrassingly parallel in a way data parallelism is not.

Why it is the higher-value lever: it attacks **how many steps convergence needs**, which is
what makes these runs long. The 143 h run had **9 permutations** — it was not slow because the
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
* **Judge on `per_perm_esr_wN.csv`, not the val split** — val is 1-34 permutations here and
  noisy enough to pick noise.

Suggested arms, chosen to separate the confound:

| arm | schedule | question |
|---|---|---|
| control | `--restart-period 50 --restart-mult 1 --restart-decay 0.97` | current defaults |
| geometric | `--restart-period 150 --restart-mult 2 --restart-decay 0.85 --stale-cycles 2` | the unattributed winner |
| long-equal | `--restart-period 400 --restart-mult 1 --restart-decay 0.9` | is it geometric growth, or just cycles long enough to amortise? |

That third arm matters. The fixed post-restart recovery cost (~80 of 150 epochs, measured over
23 cycles) is amortised by *long* cycles just as it is by *growing* ones — and long equal cycles
keep a working auto-stop, which `mult=2` does not (see below).

**Layout matters more than machine count.** One arm per machine is paced by the slowest, so
three arms on three machines costs the same as three arms serially on the M3 Max — no saving.
Two arms sequentially on the M3 Max plus one on the M4 Pro is ~20 h against ~30 h serial.

### Known defect in `--restart-mult 2`

`--stale-cycles` counts **cycles**, and `mult=2` grows them geometrically (150, 300, 600, 1200,
2400, 4800). By cycle 6 the auto-stop needs ~9,600 epochs of no improvement to fire. The EQ-test
run plateaued 1,665 epochs before it would have triggered and had to be stopped by hand via the
`STOP` file; left alone it would have ground ~8 further hours. Pair `mult>1` with a low
`--stale-cycles` (2-3) or an explicit epoch budget.

A fractional `--restart-mult` (e.g. 1.5) would give the amortisation without unbounded cycles,
but **PyTorch rejects it**: `CosineAnnealingWarmRestarts` raises
`Expected integer T_mult >= 1`. It would need a custom scheduler, and `--resume` reconstructs
schedule position from `scheduler_last_epoch`, so that arithmetic has to stay reproducible.
Test the long-equal arm first — if it matches geometric, fractional `T_mult` is unnecessary.

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
tracking, checkpoint writes, the SGDR plateau detector, per-permutation ESR and export all
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
