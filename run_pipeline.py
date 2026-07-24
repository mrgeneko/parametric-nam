#!/usr/bin/env python3
"""
Dataset generation → combine → training pipeline.

Streams real-time timestamped output from each step and logs everything
to <dataset-dir>/pipeline.log (or --log).

Steps auto-skip when their outputs already exist; use --force-generate
to wipe and regenerate, or --skip-*/--only-* for manual control.

Example — full Dumble clean pipeline:
    python run_pipeline.py \\
        --dataset-dir ~/work/tmp/dumble_clean \\
        --nam-output   ~/work/tmp/dumble_clean.param.nam \\
        --checkpoint-dir ~/work/tmp/dumble_clean_ckpt \\
        --backend livespice \\
        --schx "$HOME/work/parametric-devices/amps/Dumble Overdrive Special Preamp.schx" \\
        --knobs volume,mid,treble,middle,bass,clean_master \\
        --range volume=0.1,0.3,0.5,0.7,0.9 \\
        --range mid=0.0,1.0 \\
        --range treble=0.2,0.4,0.6,0.8 \\
        --range middle=0.2,0.4,0.6,0.8 \\
        --range bass=0.2,0.4,0.6,0.8 \\
        --range clean_master=0.1,0.3,0.5,0.7,0.9 \\
        --fixed-params "Rock=0" \\
        --speaker S1 \\
        --input ~/work/sweep-files/sweepv5.wav \\
        --slimmable --mmap --epochs 200
"""

import argparse, csv, json, math, os, platform, shutil, subprocess, sys, time, tomllib
from datetime import datetime
from pathlib import Path

HERE    = Path(__file__).resolve().parent
PYTHON  = sys.executable
BATCH   = HERE / "batch_harness.py"
TRAIN   = HERE / "param_train.py"

# argparse dests whose config values are filesystem paths (argparse's type=Path is
# only applied to CLI strings, not to set_defaults values, so we convert here).
_CONFIG_PATH_DESTS = {"dataset_dir", "nam_output", "checkpoint_dir", "release_dir",
                      "log", "schx", "input", "resume"}


def load_config(path: Path) -> dict:
    """Load a per-circuit TOML config into a dict keyed by argparse `dest`.

    A config captures a circuit's reusable *recipe* declaratively so the pipeline
    stays generic (one run_pipeline.py, one config file per circuit). CLI flags
    override anything here (config is loaded via set_defaults).

    Scalars map by name (hyphens or underscores both accepted: `crop-len` or
    `crop_len`). Structured tables expand to the flat flags:
      [knobs]      NAME = [v1, v2, ...] -> --knobs NAME,...  + one --range per knob
      [fixed]      NAME = value         -> --fixed-params NAME=value,...
      [defaults]   NAME = value         -> --defaults NAME=value,... (baked-tone default)
      [knob-boost] NAME = mult          -> --knob-boost NAME=mult,... (FiLM gradient
                                           boost for a knob whose audible effect is
                                           small relative to others -- generic over
                                           any knob name, not circuit-specific)
    `widths = [3, 4, 8]` becomes the "3,4,8" string --widths expects.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    knobs = raw.pop("knobs", None)
    fixed = raw.pop("fixed", None)
    defaults = raw.pop("defaults", None)
    knob_boost = raw.pop("knob-boost", None)
    out: dict = {}
    for k, v in raw.items():
        dest = k.replace("-", "_")
        if dest == "widths" and isinstance(v, list):
            v = ",".join(str(x) for x in v)
        elif dest in _CONFIG_PATH_DESTS and v is not None:
            v = Path(str(v)).expanduser()
        out[dest] = v
    if knobs:
        out["knobs"]  = ",".join(knobs.keys())
        out["ranges"] = [f"{name}=" + ",".join(str(x) for x in vals)
                         for name, vals in knobs.items()]
    if fixed:
        out["fixed_params"] = ",".join(f"{name}={val}" for name, val in fixed.items())
    if defaults:
        out["defaults"] = ",".join(f"{name}={val}" for name, val in defaults.items())
    if knob_boost:
        out["knob_boost"] = ",".join(f"{name}={mult}" for name, mult in knob_boost.items())
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ts():
    return datetime.now().strftime("%H:%M:%S")


def fmt_dur(secs):
    """Human-readable duration, e.g. 2h 38m 1s / 3m 12s / 19s."""
    secs = int(round(secs))
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def log(msg, fh):
    line = f"[{ts()}] {msg}\n"
    print(line, end="", flush=True)
    fh.write(line)
    fh.flush()


def section(title, fh):
    bar = "=" * 64
    msg = f"\n{bar}\n[{ts()}] {title}\n{bar}\n"
    print(msg, flush=True)
    fh.write(msg)
    fh.flush()


def stream_run(cmd, fh, label):
    """Run cmd, stream each output line with a timestamp to stdout and log.
    Returns the wall-clock seconds the command took."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log(f"CMD: {' '.join(str(c) for c in cmd)}", fh)
    fh.write("\n")
    fh.flush()

    _t = time.time()
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        out = f"[{ts()}] {stripped}\n"
        print(out, end="", flush=True)
        fh.write(out)
        fh.flush()
    proc.wait()
    dur = time.time() - _t

    if proc.returncode != 0:
        log(f"ERROR: {label} exited with code {proc.returncode} — aborting pipeline.", fh)
        sys.exit(proc.returncode)

    log(f"{label} finished OK ({fmt_dur(dur)}).", fh)
    return dur


def inhibitor_prefix():
    """Command prefix that keeps the machine awake for the whole pipeline.

    macOS → `caffeinate -i`; Linux → `systemd-inhibit` (when available).
    Returns [] when no inhibitor is available for this platform.
    """
    system = platform.system()
    if system == "Darwin":
        return ["caffeinate", "-i"]
    if system == "Linux":
        exe = shutil.which("systemd-inhibit")
        if exe:
            return [exe,
                    "--what=sleep:idle",
                    "--who=run_pipeline.py",
                    "--why=NAM dataset generation / training",
                    "--mode=block"]
    return []


def self_inhibit(no_inhibit):
    """Re-exec this script once under a sleep inhibitor so a single lock
    covers the entire run — including the idle gap between generation and
    training. A guard env var prevents an infinite re-exec loop; if the
    inhibitor is unavailable or exec fails, we fall through and run normally.
    """
    if no_inhibit or os.environ.get("PIPELINE_INHIBITED"):
        return
    prefix = inhibitor_prefix()
    if not prefix:
        return
    os.environ["PIPELINE_INHIBITED"] = "1"
    argv = prefix + [PYTHON, str(Path(__file__).resolve())] + sys.argv[1:]
    try:
        os.execvp(prefix[0], argv)
    except OSError as e:
        print(f"[warn] could not start sleep inhibitor ({e}); "
              f"running without it.", file=sys.stderr)


# ---------------------------------------------------------------------------
# release folder
# ---------------------------------------------------------------------------

def nam_variant(output, tag):
    """Sibling .param.nam path carrying a tag (mirrors param_train.nam_variant)."""
    name = output.name
    if name.endswith(".param.nam"):
        return output.with_name(f"{name[:-len('.param.nam')]}.{tag}.param.nam")
    return output.with_name(f"{output.stem}_{tag}{output.suffix}")


def git_rev(path):
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def env_info():
    code = ("import sys,torch;print(sys.version.split()[0]);"
            "print(torch.__version__);print(getattr(torch.version,'hip',None))")
    try:
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return (r.stdout.strip().split("\n") + ["?", "?", "?"])[:3]
    except Exception:
        pass
    return ["?", "?", "?"]


def hw_info():
    """(cpu_model, logical_cores, accelerator) — best-effort, cross-platform
    (Linux /proc/cpuinfo, macOS sysctl; GPU/MPS via the venv's torch)."""
    system = platform.system()
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        if system == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        elif system == "Darwin":
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                cpu = r.stdout.strip()
    except Exception:
        pass

    # Accelerator actually available to the training interpreter.
    code = ("import torch\n"
            "if torch.cuda.is_available():\n"
            "    print(torch.cuda.get_device_name(0))\n"
            "elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():\n"
            "    print('Apple GPU (MPS/Metal)')\n"
            "else:\n"
            "    print('CPU only (no GPU)')\n")
    gpu = "unknown"
    try:
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and r.stdout.strip():
            gpu = r.stdout.strip()
    except Exception:
        pass
    return cpu, os.cpu_count(), gpu


def portable(p) -> str:
    """Rewrite an absolute path into an env-anchored one, so a generated reproduce.sh RUNS.

    They did not. Every archived reproduce.sh baked in the absolute paths of whatever machine made
    it -- `REPO="/home/USER/work/..."`, `INPUT="/Volumes/DATA/work/sweep120s.wav"` -- so a script
    whose entire purpose is reproducibility could not be run anywhere but the one box that wrote it,
    and in one case pointed at an input file that has since been PURGED for licensing.

    Anchor to the sibling repos and $HOME instead, each overridable, so the script works on any
    machine with the standard layout and on machines without it too.
    """
    s = str(p)
    for base, var in (
        (HERE,                              "${PARAMETRIC_NAM:-$HOME/work/parametric-nam}"),
        (HERE.parent / "parametric-devices", "${PARAMETRIC_DEVICES:-$HOME/work/parametric-devices}"),
        (HERE.parent / "sweep-files",        "${SWEEP_FILES:-$HOME/work/sweep-files}"),
        (HERE.parent / "hotspice",           "${HOTSPICE:-$HOME/work/hotspice}"),
        (Path.home(),                        "$HOME"),
    ):
        b = str(base)
        if s == b:
            return var
        if s.startswith(b + os.sep):
            return var + s[len(b):]
    return s


def reproduce_command(args):
    """Reconstruct a one-shot run_pipeline.py command from parsed args."""
    c = ['HIP_VISIBLE_DEVICES=0 "$PY" run_pipeline.py \\',
         f'    --dataset-dir    "{portable(args.dataset_dir)}" \\',
         f'    --nam-output     "{portable(args.nam_output)}" \\',
         f'    --checkpoint-dir "{portable(args.checkpoint_dir)}" \\',
         f'    --backend {args.backend} \\']
    if args.schx:          c.append(f'    --schx "{portable(args.schx)}" \\')
    if args.circuit:       c.append(f'    --circuit "{args.circuit}" \\')
    if args.knobs:         c.append(f'    --knobs {args.knobs} \\')
    if args.oversample != "2": c.append(f'    --oversample {args.oversample} \\')
    if args.trunc_target != 1e-3: c.append(f'    --trunc-target {args.trunc_target} \\')
    if args.random:        c.append(f'    --random {args.random} \\')
    if args.no_anchors:    c.append('    --no-anchors \\')
    if getattr(args, "koren", False): c.append('    --koren \\')
    if getattr(args, "ot_damp", "47k") != "47k": c.append(f'    --ot-damp {args.ot_damp} \\')
    if getattr(args, "ot_snub", "10n") != "10n": c.append(f'    --ot-snub {args.ot_snub} \\')
    if getattr(args, "nfb_comp", None): c.append(f'    --nfb-comp {args.nfb_comp} \\')
    if getattr(args, "method", ""): c.append(f'    --method {args.method} \\')
    if getattr(args, "input_upsample", 0): c.append(f'    --input-upsample {args.input_upsample} \\')
    if args.max_crest != 50.0: c.append(f'    --max-crest {args.max_crest:g} \\')
    if args.values:        c.append(f'    --values {args.values} \\')
    for r in (args.ranges or []):  c.append(f'    --range "{r}" \\')
    for b in (args.bounds or []):  c.append(f'    --bounds "{b}" \\')
    for g in (args.gang or []):    c.append(f'    --gang "{g}" \\')
    for s in (args.steps or []):   c.append(f'    --steps "{s}" \\')
    if args.fixed_params:  c.append(f'    --fixed-params "{args.fixed_params}" \\')
    if getattr(args, "defaults", None): c.append(f'    --defaults "{args.defaults}" \\')
    if args.speaker:       c.append(f'    --speaker {args.speaker} \\')
    if args.input:         c.append(f'    --input "{portable(args.input)}" \\')
    if args.widths:        c.append(f'    --widths {args.widths} \\')
    if not args.mmap:      c.append('    --no-mmap \\')
    if getattr(args, "knob_boost", None): c.append(f'    --knob-boost "{args.knob_boost}" \\')
    if getattr(args, "per_tier_clip", False): c.append('    --per-tier-clip \\')
    if getattr(args, "clip_norm", 1.0) != 1.0: c.append(f'    --clip-norm {args.clip_norm} \\')
    epochs_part = f'--epochs {args.epochs}'
    if args.epochs == 0:
        epochs_part += f' --restart-period {args.restart_period} --restart-mult {args.restart_mult}'
    c.append(f'    --repeats {args.repeats} {epochs_part} '
             f'--crop-len {args.crop_len} --batch-size {args.batch_size} --lr {args.lr}')
    return "\n".join(c)


def build_release(args, fh, timings=None):
    """Assemble a durable release folder next to the model: the best-full and
    best-lite .param.nam files, the schematic, the full ESR history, dataset
    params, a provenance manifest (with per-step timing + hardware), and a
    replication script."""
    nam_out = args.nam_output
    stem = nam_out.name[:-len(".param.nam")] if nam_out.name.endswith(".param.nam") else nam_out.stem
    release_dir = args.release_dir or (nam_out.parent / f"{stem}_release")
    release_dir.mkdir(parents=True, exist_ok=True)
    section("RELEASE — assembling durable folder", fh)
    log(f"Release folder: {release_dir}", fh)

    # --- copy model files (every per-tier export, primary) ---
    # Glob rather than a hardcoded ("best_full", "best_lite") tuple -- that
    # silently dropped any middle tier (e.g. "best_w5") for 3+-width slimmable
    # models, even though param_train.py exports one .param.nam per tier
    # regardless of tier count. nam_out.stem strips only the LAST suffix
    # (".nam"), so match by prefix instead of relying on Path.stem twice.
    prefix = nam_out.name[:-len(".param.nam")] if nam_out.name.endswith(".param.nam") else nam_out.stem
    for src in sorted(nam_out.parent.glob(f"{prefix}.best_*.param.nam")):
        shutil.copy2(src, release_dir / src.name)
        log(f"  + {src.name}", fh)
    if nam_out.exists():
        shutil.copy2(nam_out, release_dir / nam_out.name)

    # --- schematic, metrics, dataset params ---
    if args.schx and args.schx.exists():
        shutil.copy2(args.schx, release_dir / args.schx.name)
        log(f"  + {args.schx.name}", fh)
    metrics = args.checkpoint_dir / "metrics.csv"
    if metrics.exists():
        shutil.copy2(metrics, release_dir / "metrics.csv")
    for fn in ("config.json", "params.csv"):
        src = args.dataset_dir / fn
        if src.exists():
            shutil.copy2(src, release_dir / f"dataset_{fn}")

    # --- parse ESRs from metrics.csv ---
    # Columns are val_esr_<wlabel> per tier, in ascending-width order (see param_train.py's
    # tier_labels()/wlabel) -- e.g. val_esr_lite,val_esr_w5,val_esr_full for widths [3,5,8], or
    # just val_esr_w5,val_esr_w8 for widths [5,8] (no "lite"/"full" alias when a tier isn't the
    # historical 3ch/8ch endpoint). Checkpoint files/dict keys still use the positional lite/full
    # role names for back-compat regardless of what the column is called -- first tier is always
    # "lite", last is always "full", middles keep their literal width label -- so the CSV column
    # (display) and the file-role suffix (nam_variant tag) can differ and must be matched by
    # POSITION, not by hardcoding "val_esr_full"/"val_esr_lite" (that only ever matched a [3,*,8]
    # run and KeyError'd on anything else, e.g. the EVH 5150 [5,8] run this was found on).
    tiers = []  # list of (role_suffix, display_label, best_row)
    final = None
    if metrics.exists():
        rows = list(csv.DictReader(open(metrics)))
        if rows:
            final = rows[-1]
            esr_cols = [c for c in rows[0].keys() if c.startswith("val_esr_")]
            for i, col in enumerate(esr_cols):
                disp = col[len("val_esr_"):]
                if len(esr_cols) == 1:
                    role = "full"
                elif i == 0:
                    role = "lite"
                elif i == len(esr_cols) - 1:
                    role = "full"
                else:
                    role = disp
                best = min(rows, key=lambda r: float(r[col]))
                tiers.append((role, disp, best))

    def esr_row(label, fname, disp, row):
        if row is None:
            return f"| `{fname}` | — | — | not produced |"
        return f"| `{fname}` | {row['epoch']} | {float(row[f'val_esr_{disp}']):.4f} | {label} |"

    esr_lines = [
        "| File | Epoch | ESR | Notes |",
        "|------|------:|----:|-------|",
    ]
    for role, disp, best in tiers:
        note = "ship this" if role == "full" else f"{disp}-optimal"
        esr_lines.append(esr_row(note, nam_variant(nam_out, f"best_{role}").name, disp, best))
    esr_table = "\n".join(esr_lines) if tiers else "(metrics.csv not found)"

    # --- timing + hardware ---
    def _tline(label, key):
        if timings and key in timings:
            return f"- {label}: {fmt_dur(timings[key])}"
        return f"- {label}: (skipped or cached — not re-run this pass)"
    timing_md = "\n".join([
        _tline("Generate dataset", "generate"),
        _tline("Combine", "combine"),
        _tline("Train", "train"),
    ])
    if timings:
        timing_md += f"\n- **Total (timed steps)**: {fmt_dur(sum(timings.values()))}"
    cpu, cores, gpu = hw_info()

    # --- provenance ---
    py, tv, hip = env_info()
    repo_rev = git_rev(HERE)
    schx_repo_rev = git_rev(args.schx.parent) if args.schx else None
    knob_spec = args.knobs or "(none)"
    ranges = "; ".join(args.ranges or []) or (args.values or "(default)")
    if not tiers:
        best_line = "(metrics.csv not found)"
    else:
        parts = [f"best-{disp} ESR {float(row[f'val_esr_{disp}']):.4f} @ ep{row['epoch']}"
                 for _role, disp, row in tiers]
        best_line = "; ".join(parts)

    widths_desc = args.widths or "3,5,8"  # matches param_train.py's own default

    manifest = f"""# {(args.circuit or stem)} — parametric NAM run manifest

Auto-generated by run_pipeline.py. Provenance + parameters to replicate this model.

## Models (one SlimmableContainer, widths [{widths_desc}])
{esr_table}

Full per-epoch ESR history: `metrics.csv`.

## Timing (this run)
{timing_md}

## Hardware
- CPU: {cpu} ({cores} logical cores)
- Accelerator: {gpu}

## Environment
- Python {py}, PyTorch {tv} (HIP {hip})
- Repo `parametric-nam` @ {repo_rev}
- Schematic repo @ {schx_repo_rev}
- Schematic: `{args.schx.name if args.schx else '(cpp backend)'}`
- Input: `{args.input}`

## Dataset
- Backend: {args.backend}   Circuit: {args.circuit or (args.schx.stem if args.schx else '?')}
- Knobs: {knob_spec}
- Values/ranges: {ranges}
- Fixed params: {args.fixed_params or '(none)'}
- Speaker: {args.speaker or '(none)'}

## Training
- slimmable ({args.widths or '3,8'}ch), FiLM conditioning
- epochs {args.epochs}, repeats {args.repeats}, crop_len {args.crop_len}, batch {args.batch_size}, lr {args.lr}, seed {args.seed}
- {best_line}

## Replicate
Run `./reproduce.sh`.
"""
    (release_dir / "MANIFEST.md").write_text(manifest)

    repro = f"""#!/usr/bin/env bash
# Replicate this model (generate dataset + train). Auto-generated.
set -euo pipefail
REPO="${{PARAMETRIC_NAM:-$HOME/work/parametric-nam}}"
PY="$REPO/.venv/bin/python"
cd "$REPO"

{reproduce_command(args)}
"""
    repro_path = release_dir / "reproduce.sh"
    repro_path.write_text(repro)
    repro_path.chmod(0o755)

    log(f"Release folder ready: {release_dir}", fh)
    return release_dir


# ---------------------------------------------------------------------------
# missing-permutation gate
# ---------------------------------------------------------------------------

def check_missing_permutations(dataset_dir: Path, fh, allow_missing: bool) -> None:
    """Stop the pipeline if any requested permutation failed to generate.

    A SPICE convergence failure isn't just "one fewer training example" -- it can
    indicate a real problem with the circuit or with how well the solver's
    settings cover the parameter space (see the Boss DS-1 case: two failures at
    Dist=0.6, a non-boundary value, turned out to be a fixable convergence gap in
    the ngspice BJT model, not incidental flakiness). It's also load-bearing for
    correctness: ParamDataset indexes outputs.npy by each sample's position in the
    sorted list of successes, so a gap changes what every later index refers to.
    Default is to stop and require a human to look, via --allow-missing-perms.
    """
    config_path, params_path = dataset_dir / "config.json", dataset_dir / "params.csv"
    if not config_path.exists() or not params_path.exists():
        return  # nothing generated yet under this path (e.g. fresh --skip-generate)
    expected = json.loads(config_path.read_text()).get("permutation_count")
    if expected is None:
        return
    with open(params_path, newline="") as f:
        rows = list(csv.DictReader(f))
    failed = [r for r in rows if r.get("ok") != "1"]
    if len(rows) - len(failed) >= expected and not failed:
        return
    skip_cols = {"idx", "dsp_load", "proc_time", "rms", "peak", "ok", "error"}
    lines = [f"{len(failed)} of {expected} permutations failed to generate "
             f"(only {len(rows) - len(failed)} succeeded):"]
    for r in failed:
        knobs = ", ".join(f"{k}={v}" for k, v in r.items() if k not in skip_cols)
        lines.append(f"  idx={r['idx']}  {knobs}  -> {r.get('error', '?')}")
    body = "\n".join(lines)
    if allow_missing:
        log(f"WARNING: proceeding with missing permutations (--allow-missing-perms):\n{body}", fh)
    else:
        log(f"ERROR: {body}\n"
            "A SPICE convergence failure can indicate a real circuit/solver issue -- "
            "investigate before training on an incomplete grid. Pass --allow-missing-perms "
            "to proceed anyway once you've confirmed it's expected.", fh)
        sys.exit(1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate dataset, combine, and train a parametric NAM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- pipeline control ---
    g = ap.add_argument_group("pipeline")
    g.add_argument("--config",         type=Path, default=None,
                   help="TOML per-circuit config (knobs/ranges/fixed/schx/input/widths/"
                        "hyperparams). CLI flags override it. See configs/.")
    # These three may come from --config OR the CLI; validated after parsing.
    g.add_argument("--dataset-dir",    type=Path, default=None,
                   help="Dataset directory (generation output / training input)")
    g.add_argument("--nam-output",     type=Path, default=None,
                   help="Output .param.nam path")
    g.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Directory for epoch checkpoints and metrics.csv")
    g.add_argument("--log",            type=Path, default=None,
                   help="Log file (default: <dataset-dir>/pipeline.log)")
    g.add_argument("--force-generate", action="store_true",
                   help="Re-run generation even if outputs.npy already exists")
    g.add_argument("--skip-generate",  action="store_true")
    g.add_argument("--skip-combine",   action="store_true")
    g.add_argument("--skip-train",     action="store_true")
    g.add_argument("--release-only",   action="store_true",
                   help="Skip generate/combine/train entirely and just (re)build the release "
                        "folder from what's already at --checkpoint-dir/--nam-output. For "
                        "rebuilding after a release-step-only bug fix, without re-training.")
    g.add_argument("--release-dir",    type=Path, default=None,
                   help="Durable release folder (default: <nam-output>_release)")
    g.add_argument("--no-release",     action="store_true",
                   help="Skip building the release folder after training")
    g.add_argument("--no-inhibit", "--no-caffeinate", dest="no_inhibit",
                   action="store_true",
                   help="Don't hold a sleep inhibitor for the run "
                        "(caffeinate on macOS, systemd-inhibit on Linux; "
                        "inhibiting is on by default)")

    # --- generation (batch_harness.py) ---
    g = ap.add_argument_group("generation")
    g.add_argument("--backend",      choices=["cpp", "livespice", "ngspice"], default="livespice")
    g.add_argument("--koren",        action="store_true",
                   help="ngspice: Koren triode model (softer, for stiff amps)")
    g.add_argument("--ot-damp",      default="47k", help="ngspice: OT plate-to-plate damper R")
    g.add_argument("--ot-snub",      default="10n", help="ngspice: OT snubber C")
    g.add_argument("--nfb-comp",     default=None, help="ngspice: NFB compensation cap NODE=value")
    g.add_argument("--method",       default="", choices=["", "trap", "gear"],
                   help="ngspice: integration method. Auto-set per-circuit if omitted.")
    g.add_argument("--input-upsample", type=int, default=0,
                   help="ngspice: upsample input audio N-fold before the filesource write "
                        "(fixes near-Nyquist solver non-convergence). Auto-set per-circuit if omitted.")
    g.add_argument("--schx",         type=Path, help="Path to .schx (livespice)")
    g.add_argument("--circuit",      help="Circuit name (cpp)")
    g.add_argument("--knobs",        help="Comma-separated knob names")
    g.add_argument("--input",        type=Path, help="Input sweep WAV")
    g.add_argument("--workers",      type=int,  default=os.cpu_count())
    g.add_argument("--values",       help="Default sweep values for all knobs")
    g.add_argument("--range",        action="append", metavar="KNOB=v1,v2,...",
                   dest="ranges",    help="Per-knob value list (repeatable)")
    g.add_argument("--oversample",   type=str, default="2",
                   help="oversampling factor, or 'auto' to measure the truncation-error-minimizing "
                        "value per circuit (ngspice and livespice both supported -- see "
                        "batch_harness.py --oversample auto). Default 2; high-gain amps need more "
                        "for stability, e.g. EVH 5150 Lead full = 32.")
    g.add_argument("--trunc-target", type=float, default=1e-3,
                   help="with --oversample auto: the truncation ESR to get under (default 1e-3)")
    g.add_argument("--random",       type=int, metavar="N",
                   help="N random permutations instead of a grid (for high knob counts)")
    g.add_argument("--bounds",       action="append", metavar="KNOB=lo,hi",
                   dest="bounds", help="Per-knob sample range under --random (repeatable)")
    g.add_argument("--no-anchors",   action="store_true",
                   help="Under --random, skip the boundary/corner anchor points")
    g.add_argument("--max-crest",    type=float, default=50.0,
                   help="Fail perms whose output crest factor exceeds this "
                        "(catches numerical divergence; 0 disables)")
    g.add_argument("--allow-missing-perms", action="store_true",
                   help="Proceed to combine/train even if some permutations failed to "
                        "generate. Default is to stop -- a convergence failure can "
                        "indicate a real circuit/solver issue, not just bad luck.")
    g.add_argument("--gang",         action="append", metavar="KNOB=Name1,Name2,...")
    g.add_argument("--steps",        action="append", metavar="KNOB=N")
    g.add_argument("--fixed-params", help="Fixed k=v,... for every permutation")
    g.add_argument("--defaults", help="Per-knob baked-tone default k=v,... (in trained units); "
                   "recorded in the .nam so a no-args bake uses the circuit's real default")
    g.add_argument("--speaker",      help="Speaker name (e.g. S1)")

    # --- training (param_train.py) ---
    g = ap.add_argument_group("training")
    g.add_argument("--epochs",         type=int,   default=100,
                   help="Training epochs, or 0 = open-ended (run until touch <ckpt>/STOP)")
    g.add_argument("--restart-period", type=int,   default=50,
                   help="Open-ended SGDR restart period in epochs")
    g.add_argument("--restart-mult",   type=int,   default=1,
                   help="Open-ended SGDR period multiplier per restart")
    g.add_argument("--stale-cycles",   type=int,   default=3,
                   help="Open-ended auto-stop: stop after this many consecutive SGDR cycles "
                        "with no new best on any tier (0 = manual STOP-file only; see "
                        "param_train.py --help for why the default is 3, not the doc's 2). "
                        "Forwarded to param_train.py.")
    g.add_argument("--batch-size",     type=int,   default=16)
    g.add_argument("--lr",             type=float, default=3e-4)
    g.add_argument("--crop-len",       type=int,   default=44100)
    g.add_argument("--repeats",        type=int,   default=1)
    g.add_argument("--mrstft-weight",  type=float, default=0.1)
    g.add_argument("--widths",         type=str,   default=None,
                   help="Slimmable channel widths, e.g. '3,4,8' (default 3,8)")
    g.add_argument("--mmap",           action="store_true", default=True,
                   help="Memory-map the training dataset (default on -- see param_train.py --help). "
                        "Use --no-mmap to force a full RAM load instead.")
    g.add_argument("--no-mmap",        action="store_false", dest="mmap")
    g.add_argument("--resume",         type=Path,  default=None)
    g.add_argument("--device",         default="auto")
    g.add_argument("--target-steps",   type=int, default=0,
                   help="Ask for a TRAINING BUDGET directly and let repeats be derived from it, "
                        "instead of backing into one. total steps = epochs × n_perms × repeats × "
                        "(1-val_split) / batch. Because that depends on n_perms, changing the knob "
                        "grid otherwise changes how long the model trains — regridding the Big Muff "
                        "126→60 perms silently HALVED its budget. 0 = off (use --repeats).")
    g.add_argument("--skip-grid-check", action="store_true",
                   help="skip the STEP 0 grid-adequacy measurement. It renders each knob cell's "
                        "midpoint and checks whether the grid can even represent the target ESR — a "
                        "cell that fails puts a floor under the model that NO training can lift. "
                        "Takes a couple of minutes. See docs/grid-adequacy.md.")
    g.add_argument("--grid-target",    type=float, default=0.03,
                   help="the interpolation ESR the knob grid must support (default 0.03, the "
                        "audibility floor -- NOT the training-fidelity ESR; see tools/grid_adequacy.py)")
    g.add_argument("--seed",           type=int,   default=42)
    g.add_argument("--param-sensitivity", action="store_true")
    g.add_argument("--val-split",      type=float, default=0.05,
                   help="Val fraction (was 0.1 -- see param_train.py --help for why 0.05)")
    g.add_argument("--amp",            choices=["off", "fp16", "bf16"], default="off",
                   help="Mixed-precision training forward (opt-in, needs a per-device A/B "
                        "-- see param_train.py --help). Forwarded to param_train.py.")
    g.add_argument("--per-tier-clip",  action="store_true",
                   help="Slimmable only: clip_grad_norm_ each tier's own parameters "
                        "separately instead of one joint call over every tier combined "
                        "(default off, matches NAM's own PackedLightningModule -- see "
                        "param_train.py --help). Forwarded to param_train.py.")
    g.add_argument("--clip-norm",      type=float, default=1.0,
                   help="Gradient clip norm, joint or per-tier depending on "
                        "--per-tier-clip (default: %(default)s). Forwarded to param_train.py.")
    g.add_argument("--init-from",      type=Path,  default=None,
                   help="Weights-only warm start from a previous run's best.pt (fresh "
                        "optimizer/schedule/bests). Forwarded to param_train.py.")
    g.add_argument("--val-passes",     type=int,   default=4,
                   help="Repeat the full validation pass this many times per epoch and average "
                        "-- reduces noise in reported val_esr and best-checkpoint selection. "
                        "Forwarded to param_train.py.")
    g.add_argument("--knob-boost",     type=str,   default=None,
                   help="Per-knob FiLM gradient boost: NAME=mult,NAME2=mult2,... (e.g. "
                        "'Drive=3.0'). For a knob whose audible effect is small relative "
                        "to others, so a capacity-limited tier doesn't learn to ignore "
                        "it. Forwarded to param_train.py; also settable per-circuit via "
                        "a [knob-boost] table in --config. Generic over any knob name.")

    # Load --config (if any) into defaults BEFORE parsing, so CLI flags override it.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=Path)
    _cfg_ns, _ = _pre.parse_known_args()
    if _cfg_ns.config is not None:
        if not _cfg_ns.config.exists():
            ap.error(f"--config not found: {_cfg_ns.config}")
        ap.set_defaults(**load_config(_cfg_ns.config))

    args = ap.parse_args()

    if args.release_only:
        args.skip_generate = args.skip_combine = args.skip_train = True

    # Did the USER pass --epochs / --repeats on the command line, or did they merely come from the
    # config's defaults? target-steps derives both, but an explicit flag must always win over a
    # derivation -- otherwise you cannot override it, and a silently-ignored flag is worse than none.
    _cli = set(sys.argv[1:])
    args.epochs_explicit  = any(a == "--epochs"  or a.startswith("--epochs=")  for a in _cli)
    args.repeats_explicit = any(a == "--repeats" or a.startswith("--repeats=") for a in _cli)


    # dataset-dir/nam-output/checkpoint-dir are required but may arrive via --config.
    _missing = [n for n in ("dataset_dir", "nam_output", "checkpoint_dir")
                if getattr(args, n) is None]
    if _missing:
        ap.error("missing required " + ", ".join("--" + m.replace("_", "-") for m in _missing)
                 + " (set on the CLI or in --config)")

    # Keep the machine awake for the whole pipeline (may run for hours).
    # Re-execs under caffeinate / systemd-inhibit before any work begins.
    self_inhibit(args.no_inhibit)

    dataset_dir  = args.dataset_dir
    outputs_npy  = dataset_dir / "outputs.npy"
    sig_dir      = dataset_dir / "sig"
    log_path     = args.log or (dataset_dir / "pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as fh:
        t0 = time.time()
        header = (f"\n{'#'*64}\n"
                  f"Pipeline started {datetime.now().isoformat()}\n"
                  f"Dataset : {dataset_dir}\n"
                  f"NAM out : {args.nam_output}\n"
                  f"Ckpt dir: {args.checkpoint_dir}\n"
                  f"{'#'*64}\n")
        print(header, flush=True)
        fh.write(header)
        fh.flush()

        timings = {}  # step -> wall seconds (absent = skipped)

        # ------------------------------------------------------------------
        # Step 1: Generate
        # ------------------------------------------------------------------
        run_generate = not args.skip_generate
        if run_generate and not args.force_generate:
            if outputs_npy.exists() and not sig_dir.exists():
                log("SKIP generate: outputs.npy exists and no sig/ dir found. "
                    "Use --force-generate to redo.", fh)
                run_generate = False

        # ------------------------------------------------------------------
        # Step 0: is the knob grid dense enough to be worth rendering?
        #
        # A grid too coarse in even one cell puts a FLOOR under the model that no amount of training
        # can lift -- the information is simply not in the data. That is not hypothetical: the Big
        # Muff's shipped 14x9 grid had an interpolation error of 0.0899 in Sustain 0.85-1.0, TEN
        # TIMES the 0.0090 ESR the model was chasing, while 19 of its 21 cells were oversampled by
        # 20-100x. It was simultaneously too dense almost everywhere and too coarse in the one place
        # the pedal actually lives, and no training run could ever have told us.
        #
        # So measure it BEFORE spending the renders and the epochs. It costs a couple of minutes.
        # See docs/grid-adequacy.md.
        # ------------------------------------------------------------------
        if run_generate and args.config and not args.skip_grid_check:
            section("STEP 0 / 3 — Grid Adequacy", fh)
            log("Measuring whether the knob grid can even represent the target ESR. A cell whose "
                "interpolation error exceeds it cannot be fixed by training — the data does not "
                "contain what the model would need. Re-run tools/grid_adequacy.py --suggest for a "
                "proposed regrid, or pass --skip-grid-check to render anyway.", fh)
            # stream_run aborts the pipeline on a non-zero exit, which is exactly what a too-coarse
            # grid deserves: rendering and training on it would burn hours to hit a floor we already
            # know about.
            stream_run([PYTHON, str(HERE / "tools" / "grid_adequacy.py"),
                        "--config", str(args.config),
                        "--target", str(args.grid_target)],
                       fh, "grid-adequacy")

        if run_generate:
            section("STEP 1 / 3 — Dataset Generation", fh)
            gen_cmd = [
                PYTHON, BATCH,
                "--backend", args.backend,
                "--output",  dataset_dir,
                "--workers", args.workers,
            ]
            if args.schx:          gen_cmd += ["--schx",         args.schx]
            if args.circuit:       gen_cmd += ["--circuit",      args.circuit]
            if args.knobs:         gen_cmd += ["--knobs",        args.knobs]
            if args.input:         gen_cmd += ["--input",        args.input]
            if args.values:        gen_cmd += ["--values",       args.values]
            if args.fixed_params:  gen_cmd += ["--fixed-params", args.fixed_params]
            if args.speaker:       gen_cmd += ["--speaker",      args.speaker]
            if args.koren:         gen_cmd += ["--koren"]
            if args.ot_damp != "47k": gen_cmd += ["--ot-damp", args.ot_damp]
            if args.ot_snub != "10n": gen_cmd += ["--ot-snub", args.ot_snub]
            if args.nfb_comp:      gen_cmd += ["--nfb-comp",    args.nfb_comp]
            if args.method:        gen_cmd += ["--method",      args.method]
            if args.input_upsample: gen_cmd += ["--input-upsample", args.input_upsample]
            if args.oversample != "2": gen_cmd += ["--oversample", args.oversample]
            if args.trunc_target != 1e-3: gen_cmd += ["--trunc-target", args.trunc_target]
            if args.random:        gen_cmd += ["--random",       args.random]
            if args.no_anchors:    gen_cmd += ["--no-anchors"]
            if args.max_crest != 50.0: gen_cmd += ["--max-crest", args.max_crest]
            for r in (args.ranges or []):
                gen_cmd += ["--range", r]
            for b in (args.bounds or []):
                gen_cmd += ["--bounds", b]
            for g in (args.gang or []):
                gen_cmd += ["--gang", g]
            for s in (args.steps or []):
                gen_cmd += ["--steps", s]
            timings["generate"] = stream_run(gen_cmd, fh, "Generation")

        check_missing_permutations(dataset_dir, fh, args.allow_missing_perms)

        # ------------------------------------------------------------------
        # Step 2: Combine
        # ------------------------------------------------------------------
        run_combine = not args.skip_combine
        if run_combine and not sig_dir.exists():
            if outputs_npy.exists():
                log("SKIP combine: sig/ dir gone and outputs.npy exists — already combined.", fh)
            else:
                log("ERROR: sig/ dir not found and no outputs.npy — "
                    "did generation complete?", fh)
                sys.exit(1)
            run_combine = False

        if run_combine:
            section("STEP 2 / 3 — Combine", fh)
            timings["combine"] = stream_run([PYTHON, BATCH, "--combine", dataset_dir], fh, "Combine")

        # ------------------------------------------------------------------
        # Step 3: Train
        # ------------------------------------------------------------------
        if not args.skip_train:
            section("STEP 3 / 3 — Training", fh)

            # ------------------------------------------------------------------
            # THE TRAINING BUDGET IS A DERIVED QUANTITY, AND IT WAS INVISIBLE.
            #
            # ParamDataset has len() = n_perms * repeats, and every item is a fresh random crop. So:
            #
            #     total gradient steps = epochs * n_perms * repeats * (1 - val_split) / batch_size
            #
            # It depends on n_perms. Which means CHANGING THE KNOB GRID SILENTLY CHANGES HOW LONG THE
            # MODEL TRAINS, and nobody was watching. Two ways that already bit:
            #
            #   - Regridding the Big Muff from 14x9=126 to 12x5=60 HALVED its budget (24,300 steps ->
            #     11,700). The resulting model would have looked worse and the regrid would have been
            #     blamed, inverting the truth.
            #   - repeats=30 was copied into every config, so the budget is a side-effect of how many
            #     knobs a circuit happens to have: the Big Muff got ~11.7k steps and the JCM800
            #     hot-rod (9x6x6x6 = 1944 perms) got ~369k. A 30x spread, chosen by nobody.
            #
            # So: derive it, print it, and let a config ask for a budget directly (target-steps)
            # rather than back into one via repeats.
            # ------------------------------------------------------------------
            n_perms, audio_s, sr = 0, 0.0, 48000
            try:
                with open(Path(dataset_dir) / "params.csv") as f:
                    n_perms = sum(1 for _ in csv.DictReader(f))
                dcfg = json.loads((Path(dataset_dir) / "config.json").read_text())
                audio_s = float(dcfg.get("input", {}).get("duration_s") or 0.0)
                sr = int(dcfg.get("input", {}).get("samplerate") or 48000)
            except Exception:
                pass

            repeats, epochs = int(args.repeats), int(args.epochs)

            # OPEN-ENDED (epochs=0) has NO horizon, so a step budget is meaningless: SGDR cycles the
            # LR down and restarts, and you stop when the best-per-cycle curve flattens. That is the
            # ONLY honest way to find the budget, because a fixed cosine-to-zero run always ends at
            # its best BY CONSTRUCTION and so can never tell you whether more would have helped.
            #
            # `repeats` still has to come from somewhere. Derive it as the fixed-mode rule would with
            # a nominal 450-epoch schedule, which keeps steps/epoch (and therefore the length of an
            # SGDR cycle) the same across grids of wildly different sizes.
            nominal_epochs = epochs if epochs > 0 else 450
            if epochs == 0 and args.target_steps and n_perms and not args.repeats_explicit:
                repeats = max(1, round(args.target_steps * args.batch_size
                                       / max(1, nominal_epochs * n_perms * (1 - args.val_split))))
                spe = math.ceil(n_perms * repeats * (1 - args.val_split) / args.batch_size)
                log(f"OPEN-ENDED (epochs=0): no step budget — SGDR runs until you stop it. "
                    f"repeats {repeats} → {spe} steps/epoch, restart every "
                    f"{args.restart_period} epochs ({spe * args.restart_period:,} steps/cycle).", fh)
                log(f"  Watch the BEST val ESR PER CYCLE. When two consecutive cycles fail to "
                    f"improve it, that is the budget — measured under the loss you are ACTUALLY "
                    f"training with. Stop with: touch {args.checkpoint_dir}/STOP", fh)

            if args.target_steps and n_perms and epochs > 0:
                # WHICH VARIABLE IS DERIVED MATTERS, and the obvious choice is wrong.
                #
                # `epochs` is NOT a budget quantity -- it is the LR SCHEDULE LENGTH. CosineAnnealingLR
                # uses T_max=epochs and steps once per epoch, so epochs decides how finely the LR
                # anneals and how many validation/checkpoint opportunities there are. Derive it from
                # the budget and a big grid destroys it: the JCM800 hot-rod has 1944 permutations, so
                # a data-derived repeats=94 gives 2569 steps/epoch and epochs = 24300/2569 = NINE.
                # Nine points on the cosine curve, nine chances to checkpoint a best model.
                #
                # So `epochs` stays a knob (schedule granularity) and `repeats` is derived. repeats is
                # the right free variable anyway: it is a pure virtual multiplier with no meaning of
                # its own, whereas epochs and batch_size both mean something.
                repeats_f = (args.target_steps * args.batch_size
                             / max(1, epochs * n_perms * (1 - args.val_split)))
                if not args.repeats_explicit:
                    repeats = max(1, round(repeats_f))

                if repeats_f < 1 and not args.epochs_explicit:
                    # The grid alone already exceeds the budget at repeats=1. Shorten the schedule
                    # rather than silently overspend -- and say so.
                    steps_ep = math.ceil(n_perms * 1 * (1 - args.val_split) / args.batch_size)
                    epochs = max(1, math.ceil(args.target_steps / max(1, steps_ep)))
                    log(f"  NOTE: {n_perms} perms exceed the budget at repeats=1 — holding repeats=1 "
                        f"and shortening the schedule to {epochs} epochs.", fh)

                log(f"--target-steps {args.target_steps:,} → repeats {repeats} "
                    f"(epochs {epochs} = the LR schedule length, not a budget knob)", fh)

                if audio_s:
                    crop_s = args.crop_len / sr
                    per_ep = 1.0 - (1.0 - min(1.0, crop_s / audio_s)) ** repeats
                    total_crops = epochs * repeats
                    log(f"  audio coverage: {repeats} crops/perm/epoch → ~{per_ep:.0%} of each "
                        f"permutation's {audio_s:.0f}s per epoch; {total_crops:,} crops/perm over the "
                        f"run ({total_crops / max(1, audio_s / crop_s):.0f}× its length)", fh)

            if n_perms and epochs > 0:
                items = n_perms * repeats
                steps_ep = math.ceil(items * (1 - args.val_split) / args.batch_size)
                total = steps_ep * epochs
                log(f"TRAINING BUDGET: {n_perms} perms × {repeats} repeats = {items:,} items/epoch  →  "
                    f"{steps_ep} steps/epoch × {epochs} epochs = {total:,} GRADIENT STEPS", fh)
                # RF = sum((k-1)*d over the 23 layers) + (head 16-tap - 1) + 1 = 6,347 samples.
                # The long-quoted 2,483 figure matched no version of the kernel/dilation lists.
                log(f"  crop {args.crop_len} samples ({args.crop_len/sr:.1f}s) vs a 6347-sample "
                    f"(132 ms) receptive field — {args.crop_len/6347:.1f}× it, so the model can see "
                    f"its own context (the crop's first RF samples have zero left-context)", fh)
                if not args.target_steps:
                    log(f"  NOTE: the budget DEPENDS ON THE GRID (steps ∝ n_perms). Change the knob "
                        f"grid and this number moves silently. Prefer target-steps.", fh)

            train_cmd = [
                PYTHON, TRAIN,
                "--dataset",         dataset_dir,
                "--output",          args.nam_output,
                "--checkpoint-dir",  args.checkpoint_dir,
                "--epochs",          epochs,    # derived from target-steps unless explicit
                "--restart-period",  args.restart_period,
                "--restart-mult",    args.restart_mult,
                "--stale-cycles",    args.stale_cycles,
                "--batch-size",      args.batch_size,
                "--lr",              args.lr,
                "--crop-len",        args.crop_len,
                "--repeats",         repeats,   # derived above — may differ from args.repeats
                "--mrstft-weight",   args.mrstft_weight,
                "--val-split",       args.val_split,
                "--val-passes",      args.val_passes,
                "--device",          args.device,
                "--seed",            args.seed,
            ]
            if args.widths:              train_cmd += ["--widths", args.widths]
            if not args.mmap:             train_cmd.append("--no-mmap")
            if args.resume:              train_cmd += ["--resume", args.resume]
            if args.amp != "off":        train_cmd += ["--amp", args.amp]
            if args.init_from:           train_cmd += ["--init-from", args.init_from]
            if args.param_sensitivity:   train_cmd.append("--param-sensitivity")
            if args.knob_boost:          train_cmd += ["--knob-boost", args.knob_boost]
            if args.per_tier_clip:       train_cmd.append("--per-tier-clip")
            if args.clip_norm != 1.0:    train_cmd += ["--clip-norm", args.clip_norm]
            timings["train"] = stream_run(train_cmd, fh, "Training")

        # ------------------------------------------------------------------
        # Release folder (durable copy of models + provenance)
        # ------------------------------------------------------------------
        # Outside the `skip_train` gate so --release-only (or a plain re-run after training
        # already finished) can rebuild the release folder without re-training — e.g. after
        # fixing a bug in build_release() itself, which used to require burning another full
        # training run just to re-reach this step.
        if not args.no_release:
            try:
                build_release(args, fh, timings)
            except Exception as e:
                log(f"WARNING: release folder build failed ({e}); "
                    f"models are still in place.", fh)

        elapsed = time.time() - t0
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        footer = f"\n{'#'*64}\nPipeline complete in {h}h {m}m {s}s\n{'#'*64}\n"
        print(footer, flush=True)
        fh.write(footer)


if __name__ == "__main__":
    main()
