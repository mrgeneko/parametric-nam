[← back to README](../README.md)

# Implementation roadmap: fleet, gating, and sharding

The fleet, gating and sharding work is described across four documents, each with its own
sequencing. None of them says how the sequences fit together. This is that merge: one ordered
checklist, with the source doc for each item and the dependency that fixes its position.

**Status: this ordering is a synthesis, not a decision.** Where the source docs disagree or
are silent, the placement below is a judgment call and says so. Each item's design lives in
its source doc; this file only orders them.

Sources:
[`fleet-deployment-proposal.md`](fleet-deployment-proposal.md) (**fleet**),
[`per-item-sharding-proposal.md`](per-item-sharding-proposal.md) (**sharding**),
[`config-gate-proposal.md`](config-gate-proposal.md) (**gate**),
[`scaling-training.md`](scaling-training.md) (**scaling**).

## Already shipped

- [x] Atomic `.npy` writes in `gen_dataset_from_schx.py` (sharding, Phase 0) — temp file plus
      `os.replace`, so a file's existence means it is complete
- [x] `sync_findpeak_cache.sh` — unions the per-machine onset caches before a sharded job
- [x] `--shard` on `grid_adequacy.py`, `prepare_excitation.py`, `check_transient_coverage.py`
      and `measure_truncation.py`
- [x] Per-combination pacing, and `distribute_pull.py --config` (fleet, "what went wrong")

## Render / fleet track, in order

- [x] **1. Mesh SSH** (fleet §1) — **resolved with a substitute, not as proposed.** Tailscale
      SSH (`tailscale up --ssh`) was tried and not adopted: sandboxed macOS GUI builds cannot
      run the server ("The Tailscale SSH server does not run in sandboxed Tailscale GUI
      builds."), and a fleet-wide attempt on the non-sandboxed build still had some failing
      connections whose pairs, errors and ACL rule were never recorded. The fleet runs plain
      OpenSSH with a dedicated key over the Tailscale network (`ssh -o BatchMode=yes <alias>`).
      Remaining gap: no shared DNS names (aliases are per-machine) and manual key
      distribution — both fold into item 4, the generated inventory. See the status note in
      fleet §1 before retrying.
- [x] **2. `gate_config.py`, single-machine** (gate step 1) — **implemented 2026-09-28**, with
      tests; `--verify` is the hook item 3 will use. No dependencies. Thin sequencer:
      `prepare_excitation` → `check_transient_coverage` → `preflight`. `grid_adequacy.py` is
      opt-in (`--check-grid`, check-only, run before sizing), not a default step.
      Writes a fingerprinted `<config>.gate.json` (hash of schx contents, excitation recipe
      fingerprint, knob ranges, oversample, backend). Exits non-zero with a specific reason,
      never silently partial.
- [ ] **3. Require the sidecar** (gate step 2) — **partly done 2026-09-28, warn-only.**
      `run_pipeline.py` and `distribute_pull.py --tool gen_dataset` both call `gate_config.py
      --verify` before generating and log a WARNING (not a refusal) when it's missing or stale.
      `--require-gate` opts a run into aborting (exit 2) now; `--skip-gate-check` silences the
      check. **Remaining:** flip the default from warn to refuse once the gate has been adopted
      for a while — at that point `--require-gate` becomes the (only) default behavior and
      `--skip-gate-check` remains as the escape hatch, matching this item's original design.
      Needs 2 (done).
- [x] **4. Host inventory with `--probe-hosts`** (fleet §2) — **implemented 2026-09-28** as
      `fleet_inventory.py`. Generated, reviewable TOML; physical core count (reuses
      `cpu_topology.physical_cpu_count`, not re-derived); backends per host (filesystem probes
      mirroring `check_oracle`); accelerator/`gpus`/`vram_gb` via the worker's own venv+torch.
      `train` is always written `false` (a GPU does not imply training eligibility -- human
      call, every time); `max_render_s` is not probed. **The per-host SSH address/key/login-user
      piece item 1 left open is closed as of 2026-09-28**: for a `--worker` target,
      `user`/`identity_file` are resolved from THIS machine's own `ssh -G <alias>` (no
      connection made -- it's what your existing `~/.ssh/config` already says), so a fleet
      that already leans on named aliases (as the mini's own session does) gets those recorded
      rather than re-typed. `identity_file` is written only when `ssh -G` resolves to exactly
      one -- more than one means nothing is pinned for that alias, which isn't a fact worth
      writing down. Finding this also surfaced and fixed two real bugs in the remote-probe
      plumbing itself: SSH re-joins argv into one string for the remote shell to re-parse, so
      unquoted probes silently mis-ran (`ngspice-deck` went undetected remotely) or were parsed
      as shell syntax outright (the accelerator probe); and naive uniform quoting then broke
      `~`-expansion for `repo` detection. Both fixed in `default_ssh`/`_ssh_quote` -- see their
      docstrings. Verified against a real target (`--worker localhost`), not just mocks: the
      remote probe now matches the local self-probe of the identical machine exactly.
- [x] **5. Dispatch-time version verification** (fleet step 3) — **implemented 2026-09-28**
      in `distribute_pull.py`. Before any chunk is dispatched (once per worker, not per chunk),
      each worker's `git rev-parse HEAD` and `prepare_excitation.solver_identity()` (self-
      invoked on the worker's own checkout -- stays in sync with that function automatically,
      rather than re-implementing its logic remotely) are compared against the controller's
      own. A worker that fails is **excluded from the run**, not an abort of the whole thing;
      if every worker fails, the run refuses to start (`ap.error`, exit 2).
      `--skip-version-check` opts out entirely. A solver-identity mismatch only refuses when
      BOTH sides are determinate and differ -- `"livespice:UNKNOWN"` means "couldn't tell", not
      "different", and a non-livespice backend's `"<backend>:unidentified"` on both sides
      compares equal with no special-casing. Verified against real ssh targets, not just
      mocks: `--worker localhost` (this machine, dispatch-checked over a real ssh round trip)
      matched the controller exactly; pointed at a genuinely different checkout
      (`parametric-devices`, no `prepare_excitation.py`/`.venv` there), it was correctly
      excluded and, being the only worker, the run refused to start. Does not consult item 4's
      inventory file -- `--worker HOST:DIR` already carries the repo path it needs.
- [x] **6. Per-item sharding, Phases 1–3** (sharding) — **implemented 2026-09-28** in
      `distribute_pull.py`. `--chunk-size 1` dispatches ONE combination at a time per slot
      (`--slots K`, default each worker's own PARALLEL/core count) into its own
      `<output>/slot-K`, so K concurrent generations never collide on
      `gen_dataset_from_schx.py`'s exclusive per-directory lock -- proven for real (not just
      unit tests): two concurrent dispatches into ONE shared dir collided exactly as the
      proposal predicts, the identical dispatch into per-slot dirs did not, both over a real
      ssh round trip to a real target. `derive_item_count` (Phase 2) reads the real combination
      count from `--range` (fails loudly, never guesses; `--items` overrides). `--collect`
      walks hosts × slots (Phase 3) with `labels` (a slot's own scratch-csv name, since two
      slots share one `.host`) and `expected_count` (merged rows/`.npy` count must equal N,
      every `.npy` the same byte size) -- reuses the EXISTING chunk/queue/retry/quarantine
      machinery unchanged, since a per-item "chunk" is just a shard spec with `TOTAL` = the
      real item count.
      **Not done:** the stall-detector floors (`--slow-startup-floor-min`/
      `--slow-steady-floor-min`) are NOT auto-retuned for per-item dispatch -- the proposal's
      own "~2-3x the expected single-render time" has no generic default this tool can derive,
      so pass them explicitly when adopting `--chunk-size 1`, or a genuine hang still costs the
      legacy floors' full wait. The Lifecycle section's `shard_ctl`-per-slot stop/resume/reap
      machinery is not built either -- Ctrl-C behaves exactly as it did before (daemon threads,
      no per-slot runfile/manifest), unchanged from the legacy path, not a new regression.
- [x] **7. Validate per-item output** (sharding, Phase 5) — **done 2026-09-28**, as literally
      specified: `arbiter_fuzz_face` (7-combination pedal, real oracle, real `ngspice` backend)
      rendered twice through `distribute_pull.py`'s real CLI on this machine -- once legacy
      (`--chunks 1`, `--workers 7` internal) as the static-shard reference, once per-item
      (`--chunk-size 1 --slots 7`) -- and the two collected/combined `outputs.npy` compared:
      identical sha256, every row bit-identical (`np.array_equal`), identical `output_scale`,
      identical `params.csv`. Confirms Phase 1-3's determinism claim for real, not by
      assumption.
      **Found and fixed a real bug doing this**, not a hypothetical: per-item mode never
      created `<output>/slot-K` before dispatching to it, and
      `gen_dataset_from_schx.py`'s disk-space check
      (`shutil.disk_usage(args.output.parent if not args.output.exists() else args.output)`)
      only falls back ONE level, so a brand-new `--output` path (the common case for a
      device's first-ever render) left both the slot dir AND its parent missing and raised
      `FileNotFoundError` outright. Fixed with an `ssh ... mkdir -p` per slot before any
      thread starts; regression-tested and mutation-checked.
- [ ] **8. Gate fleet mode** (gate step 3). `--workers` or `--inventory`; calls
      `sync_findpeak_cache.sh` before and after; syncs the gitignored excitation `.wav` to
      every named worker; shards grid-adequacy and transient-coverage probing. Needs 4 and 5.
- [ ] **9. Fold shard-vs-local into `run_pipeline.py`** (gate step 4). Steps 1 and 4 of the
      pipeline get a fleet-aware path that delegates to `distribute_pull.py`'s scheduling,
      driven by whether an inventory is present. Do only after 8 is proven on a real device,
      to avoid drift between two orchestration paths.
- [ ] **10. Destination-aware results** (fleet §5). Queue carries a destination so a dataset is
      collected once, where training will run.
- [ ] **11. Pull agents, queue, dashboard** (fleet §4). Largest piece. Justified by
      observability and removing the controller as a single point of failure, not by
      onboarding.
- [ ] **12. Linux worker image** (fleet §3). Only if Linux machines are added often enough to
      pay for it. Native on macOS regardless.

## Training track (independent of the above)

- [ ] **A. Schedule search** (scaling, Option A). Same dataset, same seed, same **step**
      budget, judged on `per_combo_esr_wN.csv`. Arms: control, geometric, long-equal. The
      long-equal arm has never been run; if it matches geometric, revert the `--restart-mult 2`
      default to 1 with a longer `--restart-period`.
- [ ] **B. Data-parallel training** (scaling, Option B), only if training is still the
      bottleneck after A. Roughly a day for ~1.5–1.7x on two machines; needs manual gradient
      staging because collectives are not implemented on MPS, and a throughput-proportional
      batch split.
- [x] **Not `num_workers`.** Data loading is 0.4% of a step. Decided in scaling; listed so it
      is not re-proposed.

## Open questions

- Sharding (6) versus inventory and version checks (4, 5): fleet ranks inventory first;
  sharding calls itself nearer-term. Either order is defensible.
- Gate work (2, 3) is ordered only against itself in its own doc. Placing it first here is
  because it has no fleet dependency, not because a doc says so.
