# Proposal: a gate script between scaffolding and generation

> **STATUS (2026-09-28): sequencing step 1 is implemented as `gate_config.py`** (single-machine
> only; see [scripts.md](scripts.md#gate_configpy--run-the-pre-generation-gate-record-that-it-passed)).
> **Step 2 is implemented, but WARN-only, not "require"**: `run_pipeline.py` and
> `distribute_pull.py --tool gen_dataset` both call `--verify` and warn (not refuse) on a
> missing/stale sidecar; `--require-gate` opts into aborting, `--skip-gate-check` silences it.
> Refusing by default, as originally proposed, is deferred — see
> [implementation-roadmap.md](implementation-roadmap.md) item 3. **Step 3 (fleet mode) is
> implemented** — `--workers`/`--inventory`, sharding grid/transient-coverage via
> `distribute_pull.py`, the excitation `.wav` sync — see roadmap item 8. Step 4 (folding
> sharding into `run_pipeline.py`) is still a proposal. Written 2026-09-23 after a live AC30
> Top Boost session hit three variants of the same problem in one afternoon. See
> [fleet-deployment-proposal.md](fleet-deployment-proposal.md) and
> [per-item-sharding-proposal.md](per-item-sharding-proposal.md) for the existing, larger
> plan for fleet/worker concerns — this proposal is deliberately narrow and defers to those
> for anything about machines rather than circuits. Where they overlap, this says so.

## What prompted it

`scaffold_config.py` writes a device config. `run_pipeline.py` trains from one. Between
them, a device needs its excitation sized (`prepare_excitation.py`), its knob grid checked
for adequacy (`grid_adequacy.py`), its excitation checked for saturation coverage
(`check_transient_coverage.py`), and its knobs checked for dead/reversed response
(`preflight.py`) — four scripts, run by hand, in an order that matters, each with its own
flags.

`run_pipeline.py` already runs two of the four automatically when used single-machine:
grid adequacy *was* its Step 1 (default-on; **removed 2026-09-28** along with
`--skip-grid-check` and `--grid-target` — `grid_adequacy.py` is now opt-in), and the transient
gate runs implicitly inside `gen_dataset_from_schx.py`'s step 3, gated by
`--skip-transient-check`. preflight is step 2. So on the single-machine path, this mostly
already works.

**It does not work on the sharded path.** `distribute_pull.py --tool gen_dataset` has no
equivalent of any of Steps 1–3 — it is purely a render scheduler for whichever tool it's
pointed at. All the pre-generation judgment lives in `run_pipeline.py`, which the sharded
path doesn't go through. One afternoon sharding the AC30 Top Boost's excitation/grid/dataset
pipeline by hand across three machines hit three separate variants of that gap:

| what happened | root cause | which script would have caught it |
|---|---|---|
| `gen_dataset` silently re-ran the full transient-coverage gate once per chunk (64×), ~100 min cold each | `--skip-transient-check` has to be remembered and forwarded manually on the sharded path; nothing defaults it | this proposal — or `sync_findpeak_cache.sh`, see below |
| `check_transient_coverage.py` shards were re-launched twice because the circuit changed mid-flight and the excitation had to be re-sized both times | nothing records which commit/config a shard's result belongs to | this proposal, if it writes a fingerprinted sidecar |
| the sized excitation `.wav` (gitignored, ~40 MB) wasn't on the workers when their coverage-check shards launched the first time | nothing verifies worker state before dispatch | fleet-deployment-proposal.md §2/§3 (inventory + dispatch-time version check), not this proposal |
| Bass/Treble direction was reversed and nearly shipped, caught only because the user asked "are you sure" and prompted a manual preflight run | preflight has no place at all in the sharded path | this proposal |

**A correction found while writing this**: the first row above already has a purpose-built
fix that shipped 2026-09-21 and that I didn't know about — `sync_findpeak_cache.sh`
(`docs/scripts.md`). Its own docs describe exactly this scenario: a sharded
`check_transient_coverage.py` run splits corners across machines, so each machine's onset
cache only holds a third of the answers; a later `gen_dataset` run on a different machine
re-derives what another machine already measured. Running `sync_findpeak_cache.sh --workers
...` after a sharded sizing/coverage pass and before `gen_dataset` would have made the
internal gate cheap (cache hits) instead of needing to be skipped (safety check bypassed).
I used `--skip-transient-check` instead, because I didn't find the existing tool. That's a
real gap this proposal should not repeat: a gate script has to compose the tools that
already exist, not add a parallel way to get the same result.

## What this proposes

One script — working name `gate_config.py` — run after `scaffold_config.py` and before
either `run_pipeline.py` or `distribute_pull.py`. It owns the full pre-generation sequence
in the order it actually has to happen (sizing depends on nothing; the transient gate depends
on the sized excitation; preflight depends on nothing but benefits from running last since
it's cheapest to re-run after a knob-grid change). **Revised 2026-09-28: `grid_adequacy.py` is
no longer a default step.** It is opt-in (`--check-grid`, check-only, never `--apply`), and if
run it must come *before* sizing, because `--apply` rewrites the grid and would stale a
sizing that already ran. The fingerprint hashes the knob ranges, so a later `--apply` by hand
invalidates the sidecar and forces a re-gate on its own:

1. `prepare_excitation.py` (if the excitation is missing or the recipe sidecar's fingerprint
   doesn't match the current schx/knob-range/oversample)
2. `check_transient_coverage.py`
3. `preflight.py`
   (optional, before 1: `grid_adequacy.py` under `--check-grid`)

It exits 0 with a written, fingerprinted sidecar (e.g. `<config>.gate.json`) on pass, and
exits non-zero with a specific reason on failure — never silently partial. `run_pipeline.py`
and `distribute_pull.py` both **require** a current sidecar to proceed by default (escape
hatch: `--skip-gate`, loud about what it's skipping), instead of each independently deciding
whether and how to re-check.

**The fingerprint is the actual point.** Without it, moving these calls out of
`run_pipeline.py` just relocates the trust problem: something has to know the last gate run
was against *this* schx, *this* knob range, *this* oversample — not a stale one from before
the tone-stack rebuild three commits ago. Hash `(schx file contents, excitation recipe
fingerprint, knob ranges, oversample, backend)` into the sidecar. `run_pipeline.py` /
`distribute_pull.py` recompute the same hash from the current config and refuse — not skip —
on a mismatch. This is the same shape of guard `--merge-onsets` already uses (refuses on a
solver-build mismatch); extending that convention rather than inventing a new one.

## Where fleet/worker concerns fit — answering the three questions directly

**Should worker/sharding config be available *before* the gate script, so it can verify
workers are reachable, on the right commit, and have the excitation synced?**

Yes to the goal, but that machinery should not be invented inside this proposal — it's
already scoped, in more depth than this document would give it, in
fleet-deployment-proposal.md §2 (a generated host inventory: address, repo path, cores,
backends) and §3 ("the scheduler verify a commit SHA and a simulator version *at dispatch*
and refuse mismatched workers" — sequenced there as step 3, not yet built). Building a
second, narrower worker-verification mechanism inside `gate_config.py` would create exactly
the two-parallel-implementations problem that caused today's bugs in the first place.

What `gate_config.py` *should* do, once that inventory exists: accept the same `--inventory`
/ `--worker` selection `distribute_pull.py` and (eventually) `run_pipeline.py` use, and when
a fleet is named:
- shard its own grid-adequacy and transient-coverage probing across it (both already have
  `--shard`; **both now wired 2026-09-28**, `distribute_pull.py --tool grid_adequacy` and
  `--tool check_transient_coverage` -- closing per-item-sharding-proposal.md's open item on
  this. Found and fixed a real bug doing it: `check_transient_coverage.py --emit-onsets`
  didn't `mkdir -p` its target's parent the way `grid_adequacy.py`'s own `--shard-out` write
  already did, so a sharded run into a brand-new `--output` directory failed outright),
- run `sync_findpeak_cache.sh` across the same worker set immediately before and after,
  so the union is warm for whatever runs next,
- and — this is the one piece that's genuinely new, not just "call the existing thing" —
  sync the excitation `.wav` itself to every named worker, since it's gitignored and
  `git pull` alone (unlike code and the fingerprinted sidecar) will never carry it. This
  needs the same "discover each worker's real `$HOME`" treatment `sync_findpeak_cache.sh`
  already has, not a hardcoded path.

Until the inventory file exists, `gate_config.py` can take a plain `--workers host1,host2`
list (mirroring `sync_findpeak_cache.sh`'s own flag) scoped to *this* run, and do reachability
+ commit-SHA comparison itself as a small, later-deletable piece — small enough to throw away
once the real inventory lands, not an investment worth designing carefully today.

**Should that be a separate step, or part of the same script?**

Same invocation, composed from separately-runnable pieces — not a monolith. Concretely:
`gate_config.py` should be a thin sequencer that shells out to `prepare_excitation.py`,
`grid_adequacy.py`, `check_transient_coverage.py`, `preflight.py`, and (fleet mode)
`sync_findpeak_cache.sh`, the same way `run_pipeline.py` already shells out to its steps
today. Each piece stays independently runnable and testable, matching how this codebase is
already built (small composable scripts, not a framework) — `gate_config.py`'s only new
contribution is the ordering, the fingerprinted sidecar, and (fleet mode) the excitation
sync. This also means it's low-risk to build: it doesn't touch the logic of any of the four
checks, only their sequencing and record-keeping.

**Should `run_pipeline.py` decide shard-vs-local from config, instead of the user picking
between `run_pipeline.py` and `distribute_pull.py`?**

Agree with the destination, disagree with doing it now. The two genuinely shardable steps
are grid adequacy and generation (Steps 1 and 4); preflight and the transient gate are cheap
enough single-machine that sharding them mostly isn't worth the complexity, and training/
export are inherently single-machine. So "restructure run_pipeline" really means: give Steps
1 and 4 a fleet-aware code path that delegates to the same scheduling logic
`distribute_pull.py` already has, driven by whether a fleet inventory is present — not
duplicate that logic.

That's real, useful work, but it should come **after**, not alongside, this proposal:
1. it wants the same inventory file fleet-deployment-proposal.md §2 describes, which doesn't
   exist yet;
2. it wants `gate_config.py`'s fleet-verification plumbing (reachability, commit check,
   artifact sync) to already exist and be exercised, so `run_pipeline.py` can reuse it rather
   than grow a third copy;
3. today's session is a live demonstration of how much subtle behavior drifts when there are
   two parallel orchestration paths that both wrap the same underlying tools — consolidating
   `run_pipeline.py` and `distribute_pull.py`'s decision logic is exactly the right fix for
   that, but doing it in the same change as introducing the gate script raises the chance of
   a new drift bug while fixing the old one. Land the gate script, prove it out on a real
   device end to end, then fold sharding into `run_pipeline.py` on top of a mechanism that's
   already been exercised for real.

## What this proposal does *not* cover

- Fleet inventory format, dispatch-time version verification, pull-based agents — all
  fleet-deployment-proposal.md's, unchanged.
- Sharding `check_transient_coverage.py` inside `distribute_pull.py --tool` — that's
  per-item-sharding-proposal.md's open item, orthogonal to this.
- The `gen_dataset` non-atomic-write resume-skip bug per-item-sharding-proposal.md flags as
  a prerequisite for lease-based dispatch. Unrelated to gating, but worth the same kind of
  attention.

## Sequencing

1. ~~`gate_config.py`, single-machine only.~~ **Implemented 2026-09-28**: sequences
   `prepare_excitation` → `check_transient_coverage` → `preflight`, writes the fingerprinted
   sidecar. `grid_adequacy` is opt-in (`--check-grid`, before `prepare_excitation`, check-only),
   not a default step — see "What this proposes" above.
2. ~~`run_pipeline.py` / `distribute_pull.py` require a current sidecar by default.~~
   **Implemented 2026-09-28, but WARN-only**: both call `--verify` and warn, not refuse, on a
   missing/stale sidecar (`--require-gate` to abort, `--skip-gate-check` to silence). Refusing
   by default is deferred to [implementation-roadmap.md](implementation-roadmap.md) item 3.
3. ~~`gate_config.py` fleet mode: `--workers`/`--inventory`, calls `sync_findpeak_cache.sh`,
   syncs the excitation `.wav`, shards grid-adequacy and transient-coverage probing.~~
   **Implemented 2026-09-28** -- proven on a real device (`duke_of_tone_distortion`,
   `--workers localhost`, full pass end to end). See
   [implementation-roadmap.md](implementation-roadmap.md) item 8 for exactly what shipped,
   including the verdict-check design and the two real bugs it surfaced.
4. Fleet mode is now proven on a real device (previous step) -- fold shard-vs-local selection into
   `run_pipeline.py` itself, reading the same inventory, retiring the manual choice between
   the two entry points for Steps 1 and 4.
