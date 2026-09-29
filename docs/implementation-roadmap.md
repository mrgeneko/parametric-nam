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
      rather than re-typed. **They were recorded but not yet applied** -- every dispatch was a bare
      `ssh <name>` -- until `ssh_target.py` (2026-09-28) rendered them into the ssh config every
      fleet ssh/rsync call now uses (per-host login/address/port/key; docs/scripts.md, "How each
      host is reached"). `identity_file` is written only when `ssh -G` resolves to exactly
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
- [x] **8. Gate fleet mode** (gate step 3) — **implemented 2026-09-28** in `gate_config.py`.
      `--workers host1,host2` (mirrors `sync_findpeak_cache.sh`'s own flag) or `--inventory
      [PATH]` (every host `fleet_inventory.py --probe-hosts` recorded, with each host's own
      `repo` field used for its `distribute_pull.py --worker` spec instead of a hardcoded
      guess) opts a run in. When a fleet is named: `sync_findpeak_cache.sh` runs once before
      and once after; `--check-grid` and transient-coverage both dispatch via
      `distribute_pull.py --tool grid_adequacy`/`--tool check_transient_coverage` instead of
      running single-machine; the excitation `.wav` (gitignored, never carried by `git pull`)
      is rsynced to every host's own repo-relative path right after sizing. Excitation sizing
      and preflight stay single-machine, matching the proposal's own scoping.
      **The verdict problem, solved without touching `distribute_pull.py`'s exit-code
      contract:** that scheduler's own exit code only reflects whether every shard
      *dispatched*, not the merged grid/transient *verdict* (`_collect_grid_adequacy`/
      `_collect_check_transient_coverage` never returned anything for `main()` to fold in) --
      so fleet mode re-runs the same merge command those collectors already run internally,
      against the shard files they collected, and gates on THAT exit code instead. A dispatch
      failure and a "dispatched fine but the gate itself failed" are recorded as distinct
      steps (`grid-dispatch` vs `grid`, `transient-dispatch` vs `transient`) in the sidecar.
      **Two real bugs found and fixed getting this prerequisite in place**, both via actually
      running it, not just mocking: `check_transient_coverage.py` wasn't wired into
      `distribute_pull.py --tool` at all yet (added, mirroring `grid_adequacy`'s own shape),
      and its `--emit-onsets` write didn't `mkdir -p` its target's parent the way
      `grid_adequacy.py`'s own `--shard-out` write already did -- a sharded run into a
      brand-new `--output` directory failed outright before either fix. Verified end to end
      for real (not just mocks): a full fleet-mode gate run against `localhost` as a 1-node
      fleet (`duke_of_tone_distortion`, 3 knobs, `--check-grid`) passed every step --
      cache-sync, grid dispatch+merge, wav sync, transient dispatch+merge (a real
      corner-by-corner PASSED report), cache-sync-after, preflight -- and a dispatch/verdict
      failure was separately confirmed to stop the gate at the right step via mutation
      testing.
- [x] **9. Fold shard-vs-local into `run_pipeline.py`** (gate step 4) — **implemented
      2026-09-28**. Rescoped for the current codebase: this item's original "Steps 1 and 4"
      meant grid-adequacy and generation, but grid-adequacy is no longer a `run_pipeline.py`
      step at all (item 2 made it opt-in via `gate_config.py --check-grid`, itself
      fleet-capable via item 8). Only generation (STEP 3, formerly "Step 4") remained to fold.
      `--fleet-workers`/`--inventory` switch STEP 3 to dispatch via `distribute_pull.py --tool
      gen_dataset` -- **explicitly opt-in**, not automatic on an inventory file's mere
      presence (this is the most-used, most production-critical script in the repo; a silent
      behavior change based on a file existing would be a much bigger surprise than anything
      else on this list). Host resolution reuses `gate_config.py`'s own
      `resolve_fleet_hosts`/`repo_dir_for_host` (item 8's functions, not a third copy);
      `--config` expansion reuses `distribute_pull.py`'s own `gen_args_from_config` the same
      way, rather than re-deriving STEP 3's ~40 forwarded flags a second time. Requires
      `--config` -- no equivalent exists for a hand-typed-flags invocation.
      **A real, general bug found building this**, not specific to `run_pipeline.py`:
      `distribute_pull.py`'s legacy (non-per-item) dispatch never `mkdir -p`'d `--output` on
      the worker before dispatching to it, which only ever worked because every prior
      invocation's `--output` (or its parent) happened to already exist -- this fleet mode's
      own fresh `~/.cache/parametric-nam/pipeline-fleet/` namespace never had, and
      `gen_dataset_from_schx.py`'s disk-space check raised `FileNotFoundError` outright before
      any render started. Fixed in `distribute_pull.py` itself (the same fix per-item mode's
      own slot directories already had, one level shallower), so every future caller of the
      legacy dispatch path benefits, not just this one. Verified against a real device on
      `localhost` as a 1-node fleet (`duke_of_tone_distortion`): a real chunk rendered
      end-to-end through the fleet path into the correctly-created scratch directory, with a
      valid `params.csv` row and `.npy` file to show for it (stopped there deliberately -- the
      full 63-combination grid wasn't needed to prove the wiring). Mutation-tested the
      `--no-combine` forwarding and the new `mkdir -p` call (2 mutations, both caught).
- [x] **10. Destination-aware results** (fleet §5) — **implemented 2026-09-28** in
      `distribute_pull.py` as `--collect HOST:DIR` (the "per-job sink host" option; there is no
      queue yet, so the destination rides on the job, and item 11 can carry it in the queue).
      Merge, consistency checks and `--combine` all run on the sink; per shard: local copy if the
      worker is the sink, else worker-to-sink rsync, else a tar relay through the controller.
      Not done: `run_pipeline.py` fleet mode still collects locally; no `train = true`-based
      sink autodiscovery; direct worker-to-sink transfer between two *different* machines is
      unit-tested (commands, fallback) but was only exercised for real via localhost aliases.
      See docs/scripts.md.
- [x] **11. Pull agents, queue, dashboard** (fleet §4) — **implemented 2026-09-28** as
      `fleet_queue.py` (SQLite state and every scheduling rule), `fleet_coordinator.py` (stdlib
      HTTP service + dashboard), `fleet_agent.py` (worker daemon), `fleet_client.py` and
      `fleet_ctl.py` (operator CLI). The rules are ported from `distribute_pull.py`, not
      redesigned: retries go to a different worker, N failures with no success quarantines,
      version pin per job, whole-chunk jobs only on slot 0. `distribute_pull.py`'s collect
      block was extracted to `run_collect` and is shared, so `--collect HOST:DIR` (item 10) works
      unchanged. Tested against real HTTP, SQLite, subprocess renderers and localhost ssh
      (coordinator killed mid-render, lease loss, cancel, slow-abandon, stop-release, version
      refusal, real collect to a sink); 12 mutations of the scheduling/auth/shutdown rules, the
      two that survived (`release(block=...)`) got their own tests. See docs/scripts.md.
      Not done: no ssh reverse-tunnel helper (plain HTTP: use loopback + tunnel, or a trusted
      LAN); fleet-wide combo-rate pacing lives in coordinator memory, so a restart makes the
      fleet "cold" until it re-learns; no `run_pipeline.py` fleet mode over the queue (it still
      uses `distribute_pull.py`); the remote-train leg; no agent auto-update; the dashboard is
      read-only (cancel/unquarantine are CLI). Worker-to-sink transfer between two genuinely
      different machines is still only exercised via localhost aliases (carried from item 10).
- [ ] **12. Linux worker image** (fleet §3). **Decided not to build (2026-09-28).** The Linux
      workers are `blackbox` and `optiplex7010`; both already run natively with a repo checkout
      and venv, and neither has Docker/Podman, so an image would mean installing a container
      runtime on two working hosts. Revisit if Linux machines are added often. Native on macOS
      regardless.
- [x] **11b. Multi-machine fleet run** (follow-up to 10/11) -- **implemented 2026-09-29** on
      `thinkcentre-m920q` (i5-8400, new addition to the Tailscale fleet, replacing a 35W
      i5-8700T that was slow at this), the first time item 11's agent/coordinator and item
      10's collect ran between genuinely different machines rather than localhost aliases.
      Coordinator on the controller bound to its Tailscale IP; agent on thinkcentre-m920q
      over Tailscale; a real 2-combination ngspice render (Arbiter Fuzz Face); collected back
      over rsync. ~108s/combination wall time on the i5-8400 (2 parallel slots, oversample=8);
      no comparable same-session baseline from the old i5-8700T to compare against.
      `blackbox`/`optiplex7010` were mid-render throughout and never touched.
      Two real bugs found and fixed along the way (both pushed, both covered by new tests):
      `_parse_range_axes` was multiplying a repeated `--range KNOB=...` instead of treating
      it as an override, inflating the derived chunk count for the documented
      `--config ... -- --range KNOB=...` override pattern; and `_collect`/`_collect_to_sink`
      discarded rsync's stderr on failure, so a real connectivity/auth failure read exactly
      like an ordinary empty shard. Also: `ngspice` and `livespice` backends both need
      `livespice_cli` built (parses the `.schx` format; ngspice only simulates), and its
      apphost needs `DOTNET_ROOT` set for a framework-dependent build -- `PATH` alone isn't
      enough, worth remembering for future Linux workers.

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
