#!/usr/bin/env python3
"""Pre-generation gate: excitation sizing -> transient coverage -> preflight, with a
fingerprinted sidecar recording that the gate passed for THIS circuit, grid and excitation.

Run after scaffold_config.py and before run_pipeline.py / distribute_pull.py:

    python gate_config.py --config device.config.toml [--sweep-file T3K-sweep-v3.wav]

Design: docs/config-gate-proposal.md. This is a thin sequencer. It shells out to the four
tools the same way run_pipeline.py does and adds only ordering, the sidecar, and the
staleness rules below; it does not change what any check does.

STEPS, in the order they have to happen
  0. grid_adequacy.py   ONLY with --check-grid (opt-in). Check-only, never --apply: --apply
                        rewrites the knob grid and would stale a sizing that already ran.
  1. excitation         prepare_excitation.py, only when it is needed (see below).
  2. transient          check_transient_coverage.py  -- does the excitation reach saturation
                        at every knob-grid corner?
  3. preflight          preflight.py -- dead / reversed knobs, input-level calibration.
  Stops at the first failure. Steps with no mode for the config's backend are recorded as
  SKIPPED with the reason, never silently dropped.

WHEN THE EXCITATION IS RE-SIZED (--resize auto, the default)
  Re-sizing is slow (an onset measurement at every corner), so it is not done speculatively:
    * the excitation wav is missing                         -> size it
    * a PRIOR gate sidecar exists and its sizing fingerprint no longer matches the circuit /
      grid / oversample / backend / solver                  -> the sizing is known-stale, redo it
    * otherwise (including no prior sidecar at all)         -> trust it and let step 2 decide
  A recipe.json records the wav's own hashes and its sizing results but NOT the circuit or grid
  it was sized against, so "is this excitation stale?" cannot be read off the recipe; only a
  previous gate run can answer it. --resize always / never override.

THE FINGERPRINT
  sha256 of (schx or deck-module file CONTENT, the excitation wav CONTENT, knob ranges, fixed
  params, knob kinds, and every config key that changes what the checks measure -- oversample,
  backend, conv, capture chain, ...). Content hashes, not paths, so it survives the cross-machine
  layouts these configs are written for. Training hyper-parameters (epochs, lr, widths, ...) are
  deliberately NOT included: retuning them must not invalidate a gate that measures the circuit.
  The sidecar stores the components as well as the hash, so a mismatch names what changed.
  It is recomputed at the end and must equal the value after sizing, which catches the circuit
  being edited while a multi-hour gate was in flight.

  --verify recomputes it and exits 0 only if a passing sidecar matches. That is the entry point
  run_pipeline.py / distribute_pull.py will use to REFUSE (not skip) on a stale gate.

Exit status: 0 gate passed (sidecar written) | 1 a check failed (sidecar written, status "fail")
| 2 could not run (bad config, missing file, no sweep file for a needed re-size).

Relative paths in a config resolve against the current directory, exactly as in every other
script here.
"""
import argparse
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_pipeline import load_config, git_rev  # noqa: E402  (same precedent as prepare_excitation.py)
import sizing_inputs  # noqa: E402

SCHEMA = 1
PYTHON = sys.executable

# Config keys that change what the gated tools MEASURE. Anything not listed (training
# hyper-parameters, output paths, patience, ...) can change without invalidating a gate.
SIZING_KEYS = ("schx", "backend", "oversample", "iterations", "acm_tables", "acm_lead_in", "trust_region", "newton_check", "conv", "method", "maxstep",
               "pedal_dir", "module", "probe_node", "capture_hp_hz", "capture_order",
               "capture_chain", "no_capture_chain", "out_scale", "lead_silence_s", "circuit",
               "knobs", "ranges", "fixed_params")
# Affect a check's verdict but not how the excitation should be sized.
CHECK_KEYS = ("knob_kind", "input_level_dbu")
FILE_KEYS = ("schx",)      # hashed by content rather than by path

TRANSIENT_BACKENDS = ("livespice", "acm", "ngspice-deck")
PREFLIGHT_BACKENDS = ("livespice", "acm", "ngspice-deck")


class GateError(Exception):
    """The gate could not run (exit 2) -- as opposed to a check failing (exit 1)."""


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(components: dict) -> str:
    return hashlib.sha256(json.dumps(components, sort_keys=True, default=str).encode()).hexdigest()


def _solver_identity(backend: str) -> str:
    try:
        from prepare_excitation import solver_identity
        return solver_identity(backend)
    except Exception as e:   # never let identity lookup crash a gate
        return f"{backend}:unavailable({type(e).__name__})"


def deck_module_path(cfg: dict) -> "Path | None":
    if cfg.get("pedal_dir") and cfg.get("module"):
        return Path(cfg["pedal_dir"]) / f"{cfg['module']}.py"
    return None


def compute_components(cfg: dict, with_solver: bool = True) -> "tuple[dict, dict]":
    """Returns (sizing_components, check_components). Raises GateError if a file the circuit
    is defined by (schx / deck module) or the excitation is missing where one is required."""
    sizing: dict = {}
    for k in SIZING_KEYS:
        if k in cfg and cfg[k] is not None and k not in FILE_KEYS:
            sizing[k] = cfg[k] if not isinstance(cfg[k], Path) else str(cfg[k])
    if cfg.get("schx"):
        p = Path(cfg["schx"])
        if not p.is_file():
            raise GateError(f"schx not found: {p}")
        sizing["schx_sha256"] = _sha256_file(p)
    mod = deck_module_path(cfg)
    if mod is not None:
        if not mod.is_file():
            raise GateError(f"deck module not found: {mod}")
        sizing["module_sha256"] = _sha256_file(mod)
    if with_solver:
        sizing["solver"] = _solver_identity(str(cfg.get("backend", "")))
    check = {k: cfg[k] for k in CHECK_KEYS if k in cfg and cfg[k] is not None}
    return sizing, check


def excitation_sha256(cfg: dict) -> "str | None":
    p = cfg.get("input")
    return _sha256_file(Path(p)) if p and Path(p).is_file() else None


def fingerprints(cfg: dict, with_solver: bool = True) -> dict:
    sizing, check = compute_components(cfg, with_solver)
    exc = excitation_sha256(cfg)
    components = {**{f"sizing.{k}": v for k, v in sizing.items()},
                  **{f"check.{k}": v for k, v in check.items()},
                  "excitation_sha256": exc}
    return {"components": components,
            "sizing_fingerprint": _canon(sizing),
            "fingerprint": _canon(components)}


def sidecar_path(config: Path) -> Path:
    return Path(config).with_suffix(".gate.json")


def diff_components(old: dict, new: dict) -> "list[str]":
    keys = sorted(set(old) | set(new))
    return [k for k in keys if old.get(k) != new.get(k)]


# ---------------------------------------------------------------------------
# verify (the entry point run_pipeline.py / distribute_pull.py will call)
# ---------------------------------------------------------------------------
def verify_gate(config: Path, cfg: "dict | None" = None, with_solver: bool = True) -> "tuple[bool, str]":
    """True only if a PASSING sidecar exists whose fingerprint equals the current one."""
    sp = sidecar_path(config)
    if not sp.is_file():
        return False, f"no gate sidecar ({sp.name}); run gate_config.py --config {config}"
    try:
        side = json.loads(sp.read_text())
    except Exception as e:
        return False, f"{sp.name} is unreadable ({e})"
    if side.get("schema") != SCHEMA:
        return False, f"{sp.name} has schema {side.get('schema')!r}, expected {SCHEMA}; re-run the gate"
    if side.get("status") != "pass":
        return False, (f"last gate run FAILED at step {side.get('failed_step')!r}: "
                       f"{side.get('reason', 'no reason recorded')}")
    cfg = cfg if cfg is not None else load_config(Path(config))
    try:
        now = fingerprints(cfg, with_solver)
    except GateError as e:
        return False, str(e)
    if now["fingerprint"] != side.get("fingerprint"):
        changed = diff_components(side.get("components", {}), now["components"])
        return False, ("gate is STALE -- changed since it passed: "
                       + (", ".join(changed) if changed else "(fingerprint differs)")
                       + f"; re-run gate_config.py --config {config}")
    return True, f"gate passed {side.get('finished_at', '?')} and still matches"


# ---------------------------------------------------------------------------
# decisions and commands (pure, so they are unit-testable without running a renderer)
# ---------------------------------------------------------------------------
def recipe_status(cfg: dict) -> "dict | None":
    """How the excitation's own recipe.json sits against the current circuit and grid, or None if
    there is no excitation path. See sizing_inputs.compare for the verdicts."""
    inp = cfg.get("input")
    if not inp:
        return None
    rp = Path(inp).with_suffix(".recipe.json")
    if not rp.is_file():
        return {"status": "no-recipe", "reasons": []}
    try:
        recorded = json.loads(rp.read_text()).get("sizing", {}).get("inputs")
    except Exception as e:
        return {"status": "unreadable", "reasons": [f"{rp.name}: {e}"]}
    sizing, _ = compute_components(cfg, with_solver=False)
    return sizing_inputs.compare(
        recorded, circuit_sha256=sizing.get("schx_sha256") or sizing.get("module_sha256"),
        knob_ranges=sizing_inputs.parse_ranges(cfg.get("ranges", [])),
        fixed=sizing_inputs.parse_fixed(cfg.get("fixed_params")))


def decide_excitation(mode: str, input_exists: bool, prior: "dict | None",
                      sizing_fp: str, recipe: "dict | None" = None) -> "tuple[bool, str]":
    """Returns (resize?, reason). `recipe` is recipe_status(); it is only consulted when there is
    no prior gate to compare against -- a prior gate is the stronger evidence."""
    if mode == "always":
        return True, "--resize always"
    if not input_exists:
        if mode == "never":
            raise GateError("excitation wav is missing and --resize never was given")
        return True, "excitation wav is missing"
    if mode == "never":
        return False, "--resize never"
    if prior is not None and prior.get("sizing_fingerprint") not in (None, sizing_fp):
        return True, "circuit/grid/solver changed since the last gate -- the sizing is stale"
    if prior is None:
        st = (recipe or {}).get("status")
        why = "; ".join((recipe or {}).get("reasons", []))
        if st == "stale":
            return True, f"its recipe.json was sized against a different grid ({why})"
        if st == "circuit-differs":
            return False, ("no prior gate; keeping the existing excitation, but its recipe was sized "
                           "against a different circuit revision (advisory -- step 2 decides)")
        if st == "match":
            return False, "no prior gate; the excitation's recipe matches the current circuit and grid"
        return False, ("no prior gate to compare against and the recipe does not record what it was "
                       "sized against; trusting the existing excitation (step 2 decides)")
    return False, "sizing fingerprint unchanged since the last gate"


def plan_excitation(mode: str, cfg: dict, prior: "dict | None",
                    sizing_fp: str) -> "tuple[bool, str, dict | None]":
    exists = bool(cfg.get("input")) and Path(cfg["input"]).is_file()
    recipe = recipe_status(cfg) if exists else None
    resize, why = decide_excitation(mode, exists, prior, sizing_fp, recipe)
    return resize, why, recipe


def resolve_sweep_file(arg: "str | None", cfg: dict) -> str:
    if arg:
        return arg
    inp = cfg.get("input")
    if inp:
        recipe = Path(inp).with_suffix(".recipe.json")
        if recipe.is_file():
            try:
                src = json.loads(recipe.read_text()).get("source", {}).get("path")
                if src and Path(src).is_file():
                    return src
            except Exception:
                pass
    raise GateError("a re-size is needed but no sweep file is known: pass --sweep-file (the "
                    "existing recipe.json's source path is missing or unreadable)")


def prepare_command(cfg: dict, config: Path, sweep_file: str, extra: "list[str]") -> "list[str]":
    inp = cfg.get("input")
    if not inp:
        raise GateError("config has no `input`; cannot tell prepare_excitation where to write")
    cmd = [PYTHON, str(HERE / "prepare_excitation.py"), "--backend", str(cfg.get("backend", "livespice")),
           "--config", str(config), "--sweep-file", sweep_file, "--output", str(inp)]
    if Path(inp).is_file():
        cmd.append("--no-update-config")   # same path being rewritten: leave the config text alone
    return cmd + extra


def transient_command(cfg: dict, config: Path) -> "tuple[list[str] | None, str]":
    b = cfg.get("backend")
    if b not in TRANSIENT_BACKENDS:
        return None, f"check_transient_coverage.py has no mode for backend {b!r}"
    return [PYTHON, str(HERE / "check_transient_coverage.py"), "--config", str(config)], ""


def preflight_command(cfg: dict) -> "tuple[list[str] | None, str]":
    """Mirrors run_pipeline.py's STEP 2 exactly (the same arguments, the same skip rules)."""
    b, inp = cfg.get("backend"), cfg.get("input")
    if b not in PREFLIGHT_BACKENDS:
        return None, f"preflight.py has no mode for backend {b!r}"
    if not inp:
        return None, "config has no `input`"
    if b in ("livespice", "acm"):
        if not cfg.get("schx"):
            return None, f"{b} preflight needs `schx`"
        cmd = [PYTHON, str(HERE / "preflight.py"), "--backend", b,
               "--schx", str(cfg["schx"]), "--input", str(inp)]
    else:
        if not (cfg.get("pedal_dir") and cfg.get("module")):
            return None, "ngspice-deck preflight needs `pedal-dir` and `module`"
        cmd = [PYTHON, str(HERE / "preflight.py"), "--backend", "ngspice-deck",
               "--pedal-dir", str(cfg["pedal_dir"]), "--module", str(cfg["module"]),
               "--probe-node", str(cfg.get("probe_node", "OUT")),
               "--maxstep", str(cfg.get("maxstep", 3e-6)), "--input", str(inp)]
    for flag, key in (("--knobs", "knobs"), ("--knob-kind", "knob_kind"),
                      ("--fixed-params", "fixed_params")):
        if cfg.get(key):
            cmd += [flag, str(cfg[key])]
    return cmd, ""


def grid_command(config: Path, target: float) -> "list[str]":
    # check-only on purpose: never --apply here (it rewrites the grid)
    return [PYTHON, str(HERE / "grid_adequacy.py"), "--config", str(config), "--target", str(target)]


# ---------------------------------------------------------------------------
# fleet mode (config-gate-proposal.md's "Where fleet/worker concerns fit",
# docs/implementation-roadmap.md item 8). Opt-in: --workers or --inventory names a fleet.
#
# Shards ONLY the two steps that already have --shard support via distribute_pull.py (grid,
# opt-in via --check-grid same as single-machine; transient coverage). Excitation sizing and
# preflight stay single-machine -- the proposal never asks to shard either, and sizing in
# particular is sequential/cache-adaptive in a way that doesn't parallelize the same way.
#
# THE VERDICT PROBLEM. distribute_pull.py's own exit code only reflects whether every shard
# DISPATCHED successfully (--tool grid_adequacy/check_transient_coverage's --collect just logs
# the merged report; neither _collect_grid_adequacy nor _collect_check_transient_coverage
# returns anything distribute_pull.py's main() folds into ITS exit code). So "distribute_pull
# exited 0" does not mean "the gate passed" -- a device whose grid is genuinely inadequate
# would still dispatch every shard successfully and NOT reflect that in the exit code. Rather
# than change distribute_pull.py's exit-code contract (a change with its own, wider blast
# radius -- every existing caller of --tool grid_adequacy/check_transient_coverage, not just
# this one), fleet mode re-runs the SAME merge command locally against the collected shard
# files after dispatch, exactly what _collect_grid_adequacy/_collect_check_transient_coverage
# already run internally for their own human-readable report -- a second, cheap subprocess
# call, not duplicated logic, and it gives gate_config.py its own real exit code to gate on.
# ---------------------------------------------------------------------------
import fleet_inventory  # noqa: E402
import ssh_target  # noqa: E402
from distribute_pull import _relpath_or_warn as repo_relpath  # noqa: E402  reuse, don't reimplement
from distribute_pull import sync_path_to_worker  # noqa: E402

FLEET_WORK_ROOT = "~/.cache/parametric-nam/gate-fleet"   # ON EACH WORKER -- same
                                                          # "your machine's own state"
                                                          # convention as ~/.cache/parametric-nam/findpeak


def _normalize_inventory_arg(inventory_arg) -> "Path | None":
    """argparse's --inventory nargs='?' gives the literal string "__DEFAULT__" for a BARE
    --inventory (no path) -- translate that (and a plain string from any other caller) into
    what fleet_inventory.load_inventory() actually expects: None means its own default path,
    anything else must be a Path."""
    if inventory_arg is None or inventory_arg == "__DEFAULT__":
        return None
    return Path(inventory_arg)


def resolve_fleet_hosts(workers_csv: "str | None", inventory_arg) -> "list[str]":
    """Host names for fleet mode. --workers (comma-separated, mirroring
    sync_findpeak_cache.sh's own flag) wins if given; otherwise every host in the loaded
    inventory. Raises if fleet mode was requested (this function is only called when it was)
    but resolves to zero hosts -- that is a configuration mistake, not "run locally", which
    --workers/--inventory being ABSENT already means."""
    if workers_csv:
        hosts = [h.strip() for h in workers_csv.split(",") if h.strip()]
        if not hosts:
            raise GateError("--workers given but empty")
        return hosts
    inv = fleet_inventory.load_inventory(_normalize_inventory_arg(inventory_arg))
    if not inv:
        raise GateError("--inventory has no hosts (or the file doesn't exist) -- pass "
                        "--workers host1,host2 instead, or run fleet_inventory.py --probe-hosts")
    return sorted(inv)


def repo_dir_for_host(host: str, inventory_arg, default: str = "~/work/parametric-nam") -> str:
    """The repo checkout path to use for `host` in a distribute_pull.py --worker spec: the
    inventory's own recorded `repo` field when available (measured, not guessed -- see
    fleet_inventory.py), else the same default candidate fleet_inventory.py itself tries
    first."""
    inv = fleet_inventory.load_inventory(_normalize_inventory_arg(inventory_arg))
    return (inv.get(host) or {}).get("repo") or default


def fleet_context(hosts: "list[str]", inventory_arg, *, configure: bool = True) -> dict:
    """Everything fleet mode needs about `hosts`: their repo dirs, plus the inventory FILE (None
    when it doesn't exist) that distribute_pull.py --inventory and this module's own ssh/rsync
    calls use to reach each host with the inventory's own user/address/port/identity_file
    (ssh_target.py). `configure=False` (--dry-run) computes the same answer without writing
    the generated ssh config."""
    inv_path = _normalize_inventory_arg(inventory_arg) or fleet_inventory.default_inventory_path()
    inv_path = inv_path if inv_path.is_file() else None
    if configure:
        ssh_target.configure(inv_path) if inv_path else ssh_target.reset()
    return {"hosts": hosts,
            "repo_dirs": {h: repo_dir_for_host(h, inventory_arg) for h in hosts},
            "inventory": inv_path}


def sync_findpeak_cache(hosts: "list[str]", timeout: float = 600.0,
                        ssh_config: "Path | None" = None) -> "tuple[int, str]":
    """Runs sync_findpeak_cache.sh --workers h1,h2,... . Never raises -- this is an
    optimization (warms the shared onset cache so a sharded probe hits it instead of
    re-measuring), not a correctness requirement; a failure here should not fail the gate."""
    cmd = [str(HERE / "sync_findpeak_cache.sh"), "--workers", ",".join(hosts)]
    if ssh_config:
        cmd += ["--ssh-config", str(ssh_config)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, f"{type(e).__name__}: {e}"


def sync_excitation_wav(cfg: dict, hosts: "list[str]", repo_dirs: "dict[str, str]",
                        timeout: float = 300.0) -> "list[tuple[str, bool, str]]":
    """rsync the gitignored excitation wav to every named worker's OWN repo-relative path --
    config `input` paths are already repo-relative by convention (distribute_pull.py's own
    repo_relpath/_relpath_or_warn is what makes that convention work at all), so the
    destination is `{that worker's repo dir}/{same relative path}`, not a $HOME-relative one
    the way sync_findpeak_cache.sh's cache sync is. This is the one piece config-gate-
    proposal.md calls genuinely new, not just "call the existing thing": git pull carries the
    fingerprinted sidecar and the code, but never this gitignored file.

    Returns [(host, ok, detail), ...]; never raises for a single host's failure -- one
    unreachable worker should not stop the sync to the rest.
    """
    inp = cfg.get("input")
    if not inp or not Path(inp).is_file():
        raise GateError(f"excitation wav not found locally: {inp!r} -- run the gate (sizing) first")
    rel = repo_relpath("input", inp, HERE)
    results = []
    for host in hosts:
        ok, detail = sync_path_to_worker(host, repo_dirs[host], inp, rel, timeout=timeout)
        results.append((host, ok, detail))
    return results


def _dispatch_command(tool: str, config: Path, hosts: "list[str]", repo_dirs: "dict[str, str]",
                      work_subdir: str, chunks: int,
                      inventory: "Path | None" = None) -> "tuple[list[str], Path]":
    """One distribute_pull.py --tool TOOL invocation across `hosts`. Returns (cmd, local_collect_dir)."""
    local_collect = (Path.home() / ".cache" / "parametric-nam" / "gate-fleet" /
                     config.stem / work_subdir)
    worker_flags = []
    for h in hosts:
        worker_flags += ["--worker", f"{h}:{repo_dirs[h]}"]
    cmd = [PYTHON, str(HERE / "distribute_pull.py"), "--tool", tool, "--config", str(config),
           "--chunks", str(chunks), "--output", f"{FLEET_WORK_ROOT}/{config.stem}/{work_subdir}",
           "--collect", str(local_collect), "--skip-gate-check", *worker_flags]
    if inventory:
        cmd += ["--inventory", str(inventory)]
    return cmd, local_collect


def _merge_verdict_command(tool_script: str, merge_flag: str, shard_glob: str,
                           collect_dir: Path, config: Path,
                           extra: "list[str] | None" = None) -> "list[str] | None":
    """Re-runs the same merge command _collect_grid_adequacy/_collect_check_transient_coverage
    already ran internally, against the shard files it collected -- see the module-level
    comment above for why this is the exit code fleet mode actually gates on, not
    distribute_pull.py's own (which only reflects dispatch, not verdict)."""
    shards = sorted(collect_dir.glob(shard_glob))
    if not shards:
        return None
    return [PYTHON, str(HERE / tool_script), merge_flag, *[str(s) for s in shards],
           "--config", str(config), *(extra or [])]


def grid_command_fleet(config: Path, hosts: "list[str]", repo_dirs: "dict[str, str]",
                       chunks: int = 16, inventory: "Path | None" = None) -> "tuple[list[str], Path]":
    return _dispatch_command("grid_adequacy", config, hosts, repo_dirs, "grid", chunks, inventory)


def transient_command_fleet(config: Path, hosts: "list[str]", repo_dirs: "dict[str, str]",
                            chunks: int = 16, inventory: "Path | None" = None) -> "tuple[list[str], Path]":
    return _dispatch_command("check_transient_coverage", config, hosts, repo_dirs, "transient",
                             chunks, inventory)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def _run(cmd: "list[str]") -> int:
    print(f"\n$ {' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    return subprocess.call([str(c) for c in cmd], env={**os.environ, "PYTHONUNBUFFERED": "1"})


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def run_gate(args, config: Path, cfg: dict, prior: "dict | None") -> dict:
    """Runs the steps, returns the sidecar dict (status pass|fail). Raises GateError for
    setup problems. Stops at the first failing step."""
    started = _now()
    steps: "list[dict]" = []
    side: dict = {"schema": SCHEMA, "config": str(config), "started_at": started,
                  "tool": "gate_config.py", "tool_git_rev": git_rev(HERE), "steps": steps}

    def record(name, status, rc=None, secs=None, detail="", cmd=None, **extra):
        steps.append({"step": name, "status": status, "rc": rc,
                      "seconds": None if secs is None else round(secs, 1),
                      "detail": detail, **({"cmd": " ".join(map(str, cmd))} if cmd else {}),
                      **{k: v for k, v in extra.items() if v is not None}})

    def fail(name, reason):
        side.update(status="fail", failed_step=name, reason=reason, finished_at=_now())
        return side

    def timed(name, cmd):
        t = time.time()
        rc = _run(cmd)
        return rc, time.time() - t

    # Fleet mode (docs/implementation-roadmap.md item 8): --workers/--inventory opt in. See
    # the fleet-mode comment above grid_command_fleet for what is and is not sharded, and why
    # the merge verdict is re-checked locally rather than trusted from distribute_pull.py's
    # own exit code.
    fleet = None
    if getattr(args, "workers", None) or getattr(args, "inventory", None):
        hosts = resolve_fleet_hosts(args.workers, args.inventory)
        fleet = fleet_context(hosts, args.inventory)
        rc, out = sync_findpeak_cache(hosts, ssh_config=ssh_target.active_config())
        record("fleet-cache-sync", "pass" if rc == 0 else "warn", rc, detail=out[:300])
        # non-fatal by design -- see sync_findpeak_cache's own docstring: an optimization,
        # not a correctness requirement, so a failure here does not fail the gate.

    if args.check_grid:
        if fleet:
            cmd, collect_dir = grid_command_fleet(config, fleet["hosts"], fleet["repo_dirs"],
                                                  inventory=fleet["inventory"])
            rc, secs = timed("grid-dispatch", cmd)
            record("grid-dispatch", "pass" if rc == 0 else "fail", rc, secs, cmd=cmd)
            if rc != 0:
                return fail("grid-dispatch", f"distribute_pull.py exit {rc} -- a shard "
                                             f"failed to dispatch/render; see its own log above")
            merge_cmd = _merge_verdict_command("grid_adequacy.py", "--merge", "shard_*.json",
                                               collect_dir, config,
                                               extra=["--target", str(args.grid_target)])
            if merge_cmd is None:
                return fail("grid", "fleet dispatch produced no shard_*.json to merge -- "
                                    "nothing was actually probed")
            cmd = merge_cmd
        else:
            cmd = grid_command(config, args.grid_target)
        rc, secs = timed("grid", cmd)
        record("grid", "pass" if rc == 0 else "fail", rc, secs, cmd=cmd)
        if rc != 0:
            return fail("grid", f"grid_adequacy.py exit {rc} (a cell is over target {args.grid_target}); "
                                    f"regrid with grid_adequacy.py --apply, then re-run the gate")

    # -- excitation ---------------------------------------------------------
    fp0 = fingerprints(cfg, with_solver=not args.no_solver_id)
    resize, why, recipe = plan_excitation(args.resize, cfg, prior, fp0["sizing_fingerprint"])
    if resize:
        sweep = resolve_sweep_file(args.sweep_file, cfg)
        cmd = prepare_command(cfg, config, sweep, shlex.split(args.prepare_args or ""))
        rc, secs = timed("excitation", cmd)
        record("excitation", "pass" if rc == 0 else "fail", rc, secs, why, cmd, recipe_inputs=recipe)
        if rc != 0:
            return fail("excitation", f"prepare_excitation.py exit {rc}")
        cfg = load_config(config)          # it may have written the config's `input` line
    else:
        record("excitation", "not-needed", detail=why, recipe_inputs=recipe)

    if fleet:
        # AFTER sizing (whether it ran or not, so the wav is definitely current), and BEFORE
        # the transient step that needs it present on every worker -- the one piece config-
        # gate-proposal.md calls genuinely new: git pull never carries a gitignored file.
        sync_results = sync_excitation_wav(cfg, fleet["hosts"], fleet["repo_dirs"])
        failed_hosts = [(h, d) for h, ok, d in sync_results if not ok]
        record("fleet-wav-sync", "fail" if failed_hosts else "pass",
              detail="; ".join(f"{h}: {d}" for h, d in failed_hosts) or "ok")
        if failed_hosts:
            return fail("fleet-wav-sync", f"excitation wav failed to reach "
                                          f"{len(failed_hosts)} worker(s): "
                                          + "; ".join(h for h, _ in failed_hosts))

    # -- transient coverage -------------------------------------------------
    if fleet:
        cmd, collect_dir = transient_command_fleet(config, fleet["hosts"], fleet["repo_dirs"],
                                                   inventory=fleet["inventory"])
        rc, secs = timed("transient-dispatch", cmd)
        record("transient-dispatch", "pass" if rc == 0 else "fail", rc, secs, cmd=cmd)
        if rc != 0:
            return fail("transient-dispatch", f"distribute_pull.py exit {rc} -- a shard "
                                              f"failed to dispatch/render; see its own log above")
        merge_cmd = _merge_verdict_command("check_transient_coverage.py", "--merge-onsets",
                                           "tcov_shard_*.json", collect_dir, config)
        skip = None if merge_cmd else "fleet dispatch produced no tcov_shard_*.json to merge"
    else:
        merge_cmd, skip = transient_command(cfg, config)
    if merge_cmd is None:
        record("transient", "skipped", detail=skip)
    else:
        rc, secs = timed("transient", merge_cmd)
        record("transient", "pass" if rc == 0 else "fail", rc, secs, cmd=merge_cmd)
        if rc != 0:
            return fail("transient", f"check_transient_coverage.py exit {rc}: the excitation does not "
                                         f"reach saturation at every corner; re-run with --resize always")

    if fleet:
        rc, out = sync_findpeak_cache(fleet["hosts"])
        record("fleet-cache-sync-after", "pass" if rc == 0 else "warn", rc, detail=out[:300])

    # -- preflight ------------------------------------------------------------
    cmd, skip = preflight_command(cfg)
    if cmd is None:
        record("preflight", "skipped", detail=skip)
    else:
        rc, secs = timed("preflight", cmd)
        record("preflight", "pass" if rc == 0 else "fail", rc, secs, cmd=cmd)
        if rc != 0:
            return fail("preflight", f"preflight.py exit {rc}: a knob is dead or reversed, or the "
                                         f"input level is implausible")

    # -- the inputs must not have moved while we were checking them -------------
    # Re-read everything from disk: a stale in-memory cfg would hide an edit to the config itself.
    fp1 = fingerprints(load_config(config), with_solver=not args.no_solver_id)
    # fp0 was taken BEFORE a possible re-size, which legitimately changes the excitation hash, so
    # only the sizing half (circuit, grid, solver -- not the excitation) is comparable.
    if fp1["sizing_fingerprint"] != fp0["sizing_fingerprint"]:
        changed = diff_components(fp0["components"], fp1["components"])
        return fail("consistency", "the circuit/grid changed while the gate was running ("
                                      + ", ".join(changed) + "); nothing here is trustworthy -- re-run")
    side.update(status="pass", finished_at=_now(), fingerprint=fp1["fingerprint"],
                sizing_fingerprint=fp1["sizing_fingerprint"], components=fp1["components"])
    return side


def write_sidecar(path: Path, side: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(side, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the device config.toml")
    ap.add_argument("--sweep-file", help="clip for prepare_excitation.py --sweep-file. Only needed "
                    "when a re-size happens and the existing recipe.json's source is gone.")
    ap.add_argument("--resize", choices=["auto", "always", "never"], default="auto",
                    help="when to re-size the excitation (default auto; see the module docstring)")
    ap.add_argument("--prepare-args", default="", help="extra args for prepare_excitation.py, one "
                    "quoted string (e.g. '--corner-workers 4')")
    ap.add_argument("--check-grid", action="store_true",
                    help="also run grid_adequacy.py, check-only, before sizing (opt-in)")
    ap.add_argument("--grid-target", type=float, default=0.03,
                    help="interpolation ESR --check-grid must meet (default 0.03)")
    ap.add_argument("--workers", metavar="host1,host2,...",
                    help="fleet mode (docs/implementation-roadmap.md item 8): shard grid "
                        "(--check-grid) and transient-coverage probing across these hosts via "
                        "distribute_pull.py, sync the shared onset cache before and after, and "
                        "sync the gitignored excitation wav to every host. Mirrors "
                        "sync_findpeak_cache.sh's own --workers flag. Either this or "
                        "--inventory turns fleet mode on; --workers wins if both are given.")
    ap.add_argument("--inventory", nargs="?", const="__DEFAULT__", default=None, metavar="PATH",
                    help="fleet mode: use every host fleet_inventory.py's --probe-hosts wrote "
                        "here, instead of an explicit --workers list, and look up each host's "
                        "own recorded repo path from it rather than assuming "
                        "~/work/parametric-nam. PATH is optional -- a bare --inventory means "
                        "fleet_inventory.py's own default (~/.config/parametric-nam/fleet.toml).")
    ap.add_argument("--verify", action="store_true",
                    help="do not run anything: exit 0 only if a passing sidecar still matches")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run and why, write nothing, exit 0")
    ap.add_argument("--no-solver-id", action="store_true",
                    help="leave the solver source revision out of the fingerprint (tests / offline)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = Path(args.config)
    try:
        if not config.is_file():
            raise GateError(f"config not found: {config}")
        cfg = load_config(config)

        if args.verify:
            ok, msg = verify_gate(config, cfg, with_solver=not args.no_solver_id)
            print(("OK: " if ok else "REFUSE: ") + msg)
            return 0 if ok else 1

        sp = sidecar_path(config)
        prior = None
        if sp.is_file():
            try:
                prior = json.loads(sp.read_text())
            except Exception:
                prior = None
        fp = fingerprints(cfg, with_solver=not args.no_solver_id)
        resize, why, _ = plan_excitation(args.resize, cfg, prior, fp["sizing_fingerprint"])
        if args.dry_run:
            print(f"config:      {config}\nsidecar:     {sp} ({'present' if prior else 'absent'})")
            print(f"fingerprint: {fp['fingerprint'][:16]}...  sizing: {fp['sizing_fingerprint'][:16]}...")
            print(f"excitation:  {'RE-SIZE' if resize else 'keep'} -- {why}")
            fleet = None
            if args.workers or args.inventory:
                hosts = resolve_fleet_hosts(args.workers, args.inventory)
                fleet = fleet_context(hosts, args.inventory, configure=False)
                print(f"fleet:       {', '.join(hosts)}")
            if args.check_grid:
                if fleet:
                    cmd, _ = grid_command_fleet(config, fleet["hosts"], fleet["repo_dirs"],
                                                inventory=fleet["inventory"])
                    print(f"grid:        {shlex.join(map(str, cmd))}  (then merge locally)")
                else:
                    print(f"grid:        {shlex.join(map(str, grid_command(config, args.grid_target)))}")
            if fleet:
                print(f"fleet-wav-sync: rsync excitation to {', '.join(fleet['hosts'])}")
                cmd, _ = transient_command_fleet(config, fleet["hosts"], fleet["repo_dirs"],
                                                 inventory=fleet["inventory"])
                print(f"{'transient:':12s}{shlex.join(map(str, cmd))}  (then merge locally)")
            else:
                c, s = transient_command(cfg, config)
                print(f"{'transient:':12s}{'SKIP -- ' + s if c is None else shlex.join(map(str, c))}")
            c, s = preflight_command(cfg)
            print(f"{'preflight:':12s}{'SKIP -- ' + s if c is None else shlex.join(map(str, c))}")
            return 0

        side = run_gate(args, config, cfg, prior)
    except GateError as e:
        print(f"gate_config: {e}", file=sys.stderr)
        return 2

    write_sidecar(sp, side)
    print()
    for s in side["steps"]:
        print(f"  {s['step']:11s} {s['status']:11s} {s['detail']}")
    if side["status"] == "pass":
        print(f"\nGATE PASSED -> {sp}")
        return 0
    print(f"\nGATE FAILED at {side['failed_step']}: {side['reason']}\n(recorded in {sp})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
