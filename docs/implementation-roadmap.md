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
- [ ] **3. Require the sidecar** (gate step 2). `run_pipeline.py` and `distribute_pull.py`
      refuse — not skip — on a missing or mismatched fingerprint. Escape hatch: `--skip-gate`,
      loud about what it skips. Needs 2.
- [ ] **4. Host inventory with `--probe-hosts`** (fleet §2). Generated, reviewable file;
      physical (not logical) core count; backends per host; accelerator, `vram_gb`, and a
      `train` flag separate from having a GPU. Also the natural home for what item 1 left
      open: per-host address, key path and login user, so aliases and key setup are recorded
      once rather than hand-maintained in each machine's `~/.ssh/config`. Optional, so explicit `--worker` flags keep
      working. Recorded throughput is for reporting only, never a scheduler input.
- [ ] **5. Dispatch-time version verification** (fleet step 3). Scheduler checks commit SHA and
      simulator version and refuses mismatched workers. Would have caught the 472-commit-stale
      checkout. Uses 4.
- [ ] **6. Per-item sharding, Phases 1–3** (sharding). `--slots` and `--chunk-size` in
      `distribute_pull.py`, item count derived from the grid (must fail loudly, not guess), and
      `--collect` across hosts × slots with hard assertions (merged rows == N, `.npy` count ==
      N, same byte size). Retune the stall-detector floors for per-item dispatch.
      *Placement is a judgment call:* sharding calls itself the nearer-term step and does not
      depend on 4 and 5, so it can be pulled earlier.
- [ ] **7. Validate per-item output** (sharding, Phase 5). Re-run one known-good device
      per-item and compare `.npy` files index by index against the static-shard output.
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
