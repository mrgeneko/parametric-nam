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
SIZING_KEYS = ("schx", "backend", "oversample", "iterations", "conv", "method", "maxstep",
               "pedal_dir", "module", "probe_node", "capture_hp_hz", "capture_order",
               "capture_chain", "no_capture_chain", "out_scale", "lead_silence_s", "circuit",
               "knobs", "ranges", "fixed_params")
# Affect a check's verdict but not how the excitation should be sized.
CHECK_KEYS = ("knob_kind", "input_level_dbu")
FILE_KEYS = ("schx",)      # hashed by content rather than by path

TRANSIENT_BACKENDS = ("livespice", "ngspice-deck")
PREFLIGHT_BACKENDS = ("livespice", "ngspice-deck")


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
    if b == "livespice":
        if not cfg.get("schx"):
            return None, "livespice preflight needs `schx`"
        cmd = [PYTHON, str(HERE / "preflight.py"), "--backend", "livespice",
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

    if args.check_grid:
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

    # -- transient coverage -------------------------------------------------
    cmd, skip = transient_command(cfg, config)
    if cmd is None:
        record("transient", "skipped", detail=skip)
    else:
        rc, secs = timed("transient", cmd)
        record("transient", "pass" if rc == 0 else "fail", rc, secs, cmd=cmd)
        if rc != 0:
            return fail("transient", f"check_transient_coverage.py exit {rc}: the excitation does not "
                                         f"reach saturation at every corner; re-run with --resize always")

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
            for label, (c, s) in (("transient", transient_command(cfg, config)),
                                  ("preflight", preflight_command(cfg))):
                print(f"{label + ':':12s}{'SKIP -- ' + s if c is None else shlex.join(map(str, c))}")
            if args.check_grid:
                print(f"grid:        {shlex.join(map(str, grid_command(config, args.grid_target)))}")
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
