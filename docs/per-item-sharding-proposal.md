# Per-item sharding: decoupling load balance from worker parallelism

**Status:** proposal, not implemented. Written 2026-09-12 after the Mesa Orange
gain/master render, which was sharded by hand across three machines and needed three
restarts.

## Summary

Dispatch **one combination at a time** to each of K concurrent slots per host, where each
slot owns its own output directory, instead of dispatching a multi-combination chunk to a
host that renders it with K-way internal parallelism.

This removes `--chunks`, which `distribute_pull.py`'s own docstring calls "the one tuning
knob", shrinks the straggler tail from one chunk to one render, and generalises to every
other shardable task in the pipeline.

## The problem: two knobs welded together

`distribute_pull.py` takes `--worker HOST:DIR:PARALLEL` and passes PARALLEL straight through
as the renderer's `--workers`:

```python
rc, dt, out = w.run_chunk(chunk, f"{gen_args_str} --workers {w.parallel}", ...)
```

`worker_loop` holds **one chunk per host**. So a chunk must contain at least PARALLEL
combinations to keep the box busy:

- chunks too small — the worker starves (`--chunks 64` over 72 combinations gives blackbox
  1-2 items to spread across 12 cores)
- chunks too large — the tail is lumpy, because the last chunk still has to finish

Granularity of load balancing and degree of intra-worker parallelism are the same number,
and they want opposite things.

### The stated reason for chunking does not hold on the livespice path

The docstring justifies chunk size by per-invocation startup: "each dispatch pays an ssh
round-trip plus the renderer's own startup (schx parse and symbolic solve, ~30-60 s for a
full amp)". Measured on this repo, Mesa Orange (sag v30):

```
schx XML parse            0.003 s
python + module import    0.079 s
livespice_cli invocation  --circuit <schx>    <- the solve is PER RENDER already
```

`livespice_cli` is spawned once per render and parses the circuit itself, so the symbolic
solve is paid per combination no matter how work is grouped. Python-side startup is 80 ms
against a ~19-minute render: **~0.1%**, plus one ssh round-trip.

Measured for livespice only. The ngspice-deck and ltspice paths build their decks in
process; re-measure before assuming this generalises to them.

## Prerequisite (Phase 0): make existence mean completeness

`gen_dataset_from_schx.py` resume-skips on existence alone:

```python
path = sig_path(out_dir, idx)
if path.exists():
    return Result(idx, ok=True)
```

but writes non-atomically:

```python
np.save(str(path), sig)
```

A render killed or a machine lost mid-write leaves a truncated `.npy` that every later run
treats as complete, and that flows into `--combine` as training data. Nothing downstream
looks at it again.

The same file already uses the correct pattern for its ngspice `.fs` write -- write a
`.tmp`, then `os.replace()` -- it is simply not applied to the `.npy` path.

**Everything below depends on this.** With atomic writes, a file's presence means it is
finished, `.tmp` files are unambiguous garbage, and every cleanup question has an obvious
answer. Without it, "clean up incomplete files" is undecidable from disk.

Add a one-time size check on resume to catch partials already written by older builds.

## Design

Each host runs K dispatch slots. Slot *k* renders into `<output>/slot-k` and is dispatched
one combination at a time:

```
--shard i-i/N --workers 1        # N = combination count, so chunk i selects exactly index i
```

Per-slot directories are what make this legal: `acquire_generation_lock` is exclusive per
output directory, and K concurrent generations into one directory is precisely the silent
`params.csv`/`.npy` desync it exists to prevent. Merging is already safe because filenames
are the **global** combination index and `merge_params()` is already keyed on it.

### Why per-slot directories rather than a persistent pulling worker

A long-lived worker process holding one lock, one directory and an internal pool is the
cleaner model, and it is what you would design from scratch. It needs a queue protocol over
ssh and a pull-loop mode written into every tool that wants to participate. Per-slot
directories get the same scheduling behaviour with no new protocol, and the lowest common
denominator across tools:

| task | per-item output | can concurrent units share a sink? |
|---|---|---|
| `gen_dataset` | large `.npy` by global index, one `params.csv` | **no** -- exclusive lock, shared csv |
| `grid_adequacy` | one `--shard-out` JSON per shard | yes, already separate |
| `stability_sweep` | per-job measurements | yes |
| corner probing | one content-keyed cache JSON per corner | yes, naturally |

Only `gen_dataset` needs the change; `grid_adequacy` already works this way.

## Phases

**Phase 0 — atomic `.npy` writes.** Prerequisite, above. Independently worth doing.

**Phase 1 — split the knobs in `distribute_pull`.** Add `--slots K` (concurrent dispatches
per host, defaulting to `w.parallel` so existing callers are unchanged) and `--chunk-size`
(1 = per-item). Spawn K `worker_loop` threads per host, each with its own slot directory.
The queue itself is unchanged: a deque of specs handed to whoever is free.

**Phase 2 — derive the item count.** `--shard i-i/N` needs the real combination count.
Parse the `--range KNOB=v1,v2,...` specs already in the passthrough and take the product;
`--items N` overrides for tools whose grid is not expressed that way. Too high gives empty
dispatches (80 ms, harmless); **too low silently drops work, so this must fail loudly rather
than guess.**

**Phase 3 — collect across hosts x slots.** `--collect` walks `hosts x slots` instead of one
directory per host. `merge_params()` already does the right thing. Add hard assertions:
merged row count == N, `.npy` count == N, `params.csv` 1:1 with `sig/`, and every `.npy` the
same byte size (all 24 were exactly 39,379,328 in the reference run, so truncation shows up
immediately).

**Phase 4 — tests.** Slot directories distinct per host and slot; per-item dispatch really
emits `--workers 1`; merge across many slots yields exactly N rows with no duplicates; a
failed item retried elsewhere does not double-count.

**Phase 5 — validate against known-good data.** Re-run the Orange 2-knob generation per-item
on one machine and compare `.npy` files index by index against the static-shard output.
Renders are deterministic (see below), so they should be byte-identical -- a real check, not
a smoke test.

## Lifecycle

### Stopping

Each dispatch launches through `shard_ctl start` on the worker, so every slot has a runfile
recording its pgid, and teardown is `shard_ctl stop` per slot -- signalling the process
**group**, not a pid. The controller keeps a manifest of `(host, slot, output dir)` and
`--stop` walks it. Ctrl-C on the controller runs the same teardown instead of orphaning
every in-flight remote process.

This matters because it is what went wrong by hand: a kill aimed at a bash wrapper
re-parented the renderer to init, which kept running with stale code and wrote into a log
the replacement run had truncated.

### Cleaning up incomplete files

With Phase 0 there are no partial `.npy`, only stray `.tmp` files, which are garbage by
definition. `--reap` sweeps them **only for slots whose lock is free**. That is decidable:
the lock lives outside the output directory and `flock` auto-releases on death, so "can I
acquire it?" answers "is anyone still working here?" A live slot is never reaped, and a dead
worker's lock needs no manual clearing.

### A worker going away

Per-item dispatch bounds the loss: an unreachable host costs the renders in flight on it --
up to K, not a whole chunk. SSH failure re-queues those indices elsewhere (`--retries`), and
repeated failures quarantine the host (`--quarantine-after`).

If the worker was only **partitioned**, it keeps rendering and finishes the index into its
own slot directory while another worker renders the same index. Two files, same global-index
name, different directories. Because renders are deterministic this is a no-op collision at
merge -- and better than harmless: hash the duplicates and it becomes a free cross-machine
consistency check. Two independent renders of the same index that **differ** is an alarm
worth failing on, not a conflict to resolve by picking one.

On rejoin, the stale `shard_ctl` runfile names a pid that is gone or recycled by an
unrelated process; the cmdline-marker guard in `_is_ours()` answers correctly. Reboot is
where pid recycling is most likely, which is exactly what that guard is for.

### The controller dying

Workers are detached, so renders continue. Restart with `--resume`: scan every host x slot
directory, collect the indices present, re-queue only what is missing. Global-index
filenames mean there is no controller-side state to recover.

## Failure walkthrough: worker dies mid-render, no response

1. **Detection, ~180 s.** Dispatch runs `ssh -o ServerAliveInterval=60`; with the default
   `ServerAliveCountMax=3` the client gives up after ~3 minutes and exits non-zero. That
   default is inherited rather than chosen -- pin it explicitly.
2. **Requeue** on a different host; **quarantine** after repeated failures.
3. **On disk there:** at most a `.tmp` (Phase 0). The `flock` is released by the kernel when
   the machine dies, so no stale lock.
4. **Weaker path: hung but alive.** It keeps answering keepalives, so ssh never errors and
   detection falls to the stall detector, whose floors (`--slow-startup-floor-min 90`,
   `--slow-steady-floor-min 30`) are sized for chunks. Under per-item dispatch these want
   retuning to roughly 2-3x the expected single-render time, or a hang costs 90 minutes
   before anything notices.

## Determinism, and what it licenses

Five renders of the same cell at the same input level, Mesa Orange sag v30, oversample 8:

```
input  0.05322 V   n=5   rms 3.704067 .. 3.704067   spread 0.0000%
input     40.0 V   n=5   rms 3.700245 .. 3.700245   spread 0.0000%
```

Bit-identical. This is what licenses treating duplicate renders as benign and comparing
copies across machines as a consistency check.

**Measured on one machine only.** This fleet is mixed ARM and x86, and cross-machine
reproducibility is unverified -- different vectorisation is plausible. Circumstantial
evidence is good: three machines independently reported `all-min 0.909 V` and
`OR Gain=lo-solo 0.137 V`, agreeing to the printed precision. A full-precision comparison of
one cached curve across machines would settle it and is nearly free.

## Costs and risks

- **Disk duplication.** Each output directory gets its own copy of the excitation wav
  (39 MB for the Orange). Thirty slots is ~1.2 GB. Tolerable across this fleet; the fix, if
  it bites, is `--no-input-copy` or hardlinking when slots share a filesystem.
- **~30 concurrent ssh sessions.** Wants `ControlMaster=auto` with `ControlPersist`, or you
  pay a handshake per item.
- **Stall-detector floors are wrong** until retuned (above). A mis-tuned floor quarantines a
  healthy worker.
- **Retry loses resume-skip.** A re-dispatched chunk currently skips what is already
  rendered; per-slot directories break that. Per-item dispatch makes it moot -- a retry
  re-renders exactly one combination.

## What it enables

The dispatcher is already tool-agnostic: `JOBS = {gen_dataset, grid_adequacy}` with `--tool`
selecting and `--collect` running each tool's own merge. Per-item dispatch serves those two
unchanged, and opens the way for the biggest piece of duplicated work in the pipeline today.

**Corner probing is not sharded at all.** `prepare_excitation.py`,
`check_transient_coverage.py` and `preflight.py` each sweep every corner through
`find_saturation_point`. In the Orange run all three workers independently ran the same
25-corner coverage gate before rendering -- roughly 25 minutes each, ~50 minutes spent
computing identical numbers.

It is an ideal registry entry: results are tiny and content-keyed on
`sha256(ONSET_METHOD + schx bytes + params + capture tag)`, so merging is just copying JSONs
into each worker's `~/.cache/parametric-nam/findpeak` -- collision-free by construction, no
merge logic. Probe once across the fleet, distribute the cache, and every downstream gate
hits warm. This depends on the cross-machine determinism question above.

## Effort

Phases 1-3 with tests: a few hours. Phase 5: about an hour of wall-clock. Phase 0 is
small and should land regardless of whether the rest is built -- as should the `.npy`
size assertion in Phase 3, which would have caught the Phase 0 bug on its own.
