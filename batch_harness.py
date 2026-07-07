#!/usr/bin/env python3
"""
Parallel batch processor for LiveSPICE harness.

livespice backend (no code changes needed for new circuits):
    python batch_harness.py --backend livespice \\
        --schx "/path/to/Overdrive Special Preamp.schx" \\
        --knobs volume \\
        --input sweep.wav --output ./training_data

    # List all pots in a schx (omit --knobs):
    python batch_harness.py --backend livespice --schx circuit.schx

cpp backend:
    python batch_harness.py --backend cpp --circuit marshall_jcm800_2203_preamp_modded \\
        --input sweep.wav --output ./training_data

Output (sharded by 2-digit prefix — ~100 files per dir):
    training_data/
        config.json
        sweep.wav
        params.csv
        sig/00/000000.npy
        ...

Post-processing:
    python batch_harness.py --combine training_data/
        → training_data/outputs.npy   (float32, shape [N_perms, N_samples])
"""

import atexit, argparse, csv, json, os, re, shutil, signal, subprocess, sys, threading, time
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "harness/build_o3/harness"
LIVESPICE_CLI = HERE / "livespice_cli/publish/livespice_cli"

# CPP backend only — the C++ harness has its own schematic registry
CPP_CIRCUIT_KNOBS = {
    "marshall_jcm800_2203_preamp_modded": ["gain_1", "gain_2", "bass", "middle", "treble", "volume"],
    "marshall_jcm800_2203_preamp":        ["gain", "middle", "treble", "bass1", "volume"],
}

# ---------------------------------------------------------------------------
# Orphan-process safeguard
# Track all live Popen objects so an atexit / SIGTERM handler can kill them.
# Without this, killing the Python process leaves livespice_cli workers
# running as orphans indefinitely.
# ---------------------------------------------------------------------------

_procs_lock = threading.Lock()
_active_procs: set = set()


def _kill_active_procs():
    with _procs_lock:
        for p in list(_active_procs):
            try:
                p.kill()
            except Exception:
                pass


def _sigterm_handler(*_):
    sys.exit(1)  # triggers atexit


atexit.register(_kill_active_procs)
signal.signal(signal.SIGTERM, _sigterm_handler)


# ---------------------------------------------------------------------------
# schx discovery (livespice backend)
# ---------------------------------------------------------------------------

def parse_schx_controls(schx_path: str) -> dict:
    """Return {normalized_key: exact_Name} for every Potentiometer,
    VariableResistor, and SPDT switch in the schx."""
    tree = ET.parse(schx_path)
    controls = {}
    for comp in tree.getroot().iter("Component"):
        t = comp.get("_Type", "")
        if not any(x in t for x in ("Potentiometer", "VariableResistor", "SPDT")):
            continue
        name = comp.get("Name", "")
        key = re.sub(r'_+', '_', name.strip().lower().replace(" ", "_"))
        controls[key] = name
    return controls


def parse_schx_pots(schx_path: str) -> dict:
    """Backwards-compat wrapper — returns only pots/variable resistors."""
    return {k: v for k, v in parse_schx_controls(schx_path).items()
            if not any(x in v for x in ("SPDT",))}


def resolve_knobs(knob_keys: List[str], control_map: dict) -> dict:
    """Map user-supplied knob names → exact schx component Names.
    Matching is case-insensitive and ignores extra spaces/hyphens."""
    resolved = {}
    for k in knob_keys:
        normalized = re.sub(r'_+', '_', k.strip().lower().replace(" ", "_").replace("-", "_"))
        if normalized not in control_map:
            available = ", ".join(sorted(control_map.keys()))
            raise SystemExit(f"Knob '{k}' not found in schx.\nAvailable: {available}")
        resolved[k] = control_map[normalized]
    return resolved


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def list_circuits():
    subprocess.run([str(HARNESS), "--list"])


def fmt_params(params: dict, param_map: dict = None) -> str:
    """Serialize params dict to 'Name=val,...' string.
    param_map translates user-friendly keys → exact circuit component names. A
    mapped value may be a list of Names (a ganged control, e.g. a dual-gang pot):
    the single knob value is written to every Name in the list."""
    if param_map:
        parts = []
        for k, v in params.items():
            target = param_map.get(k, k)
            if isinstance(target, (list, tuple)):
                parts.extend(f"{n}={v}" for n in target)
            else:
                parts.append(f"{target}={v}")
        return ",".join(parts)
    return ",".join(f"{k}={v}" for k, v in params.items())


# ---------------------------------------------------------------------------
# permutation generators
# ---------------------------------------------------------------------------

def grid_permutations(knobs: List[str], values_per_knob: dict) -> List[dict]:
    return [dict(zip(knobs, vs)) for vs in product(*[values_per_knob[k] for k in knobs])]


def random_permutations(knobs: List[str], n: int, seed: int = 0,
                        bounds: dict = None, anchors: bool = True) -> List[dict]:
    """N points in the knob hypercube.

    bounds:  {knob: (lo, hi)} per-knob sample range (default (0.0, 1.0)). Lets
             you keep a knob out of a divergent regime (e.g. cap a high-gain
             pot) without dropping it as a variable.
    anchors: if True, deterministically PREPEND boundary points so the extremes
             players actually use are covered (uniform random essentially never
             lands on them): the all-min and all-max corners, plus each knob
             solo at its lo/hi with the others centered. Uniform-random points
             then fill the remainder up to n. Anchors respect `bounds`.
    """
    import random as rng
    rng.seed(seed)
    bounds = bounds or {}
    lo = {k: float(bounds.get(k, (0.0, 1.0))[0]) for k in knobs}
    hi = {k: float(bounds.get(k, (0.0, 1.0))[1]) for k in knobs}

    perms: List[dict] = []
    if anchors:
        perms.append({k: lo[k] for k in knobs})            # all-min corner
        perms.append({k: hi[k] for k in knobs})            # all-max corner
        for k in knobs:                                    # each knob solo at its extremes
            mid = {kk: (lo[kk] + hi[kk]) / 2 for kk in knobs}
            p_lo = dict(mid); p_lo[k] = lo[k]; perms.append(p_lo)
            p_hi = dict(mid); p_hi[k] = hi[k]; perms.append(p_hi)

    while len(perms) < n:                                  # uniform-random fill within bounds
        perms.append({k: lo[k] + (hi[k] - lo[k]) * rng.random() for k in knobs})
    return perms[:n]


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

@dataclass
class Result:
    idx: int
    dsp_load: float = -1.0
    proc_time: float = -1.0
    ok: bool = False
    error: str = ""
    rms: float = 0.0
    peak: float = 0.0


def sig_path(out_dir: Path, idx: int) -> Path:
    return out_dir / "sig" / f"{idx // 100:02d}" / f"{idx:06d}.npy"


def process_one(idx: int, params: dict, out_dir: Path, input_wav: Path,
                backend: str, circuit: str = None, schx: str = None,
                param_map: dict = None, fixed_params: str = None,
                speaker: str = None, expected_frames: int = 0,
                timeout_s: int = 1200, oversample: int = 2) -> Result:
    path = sig_path(out_dir, idx)
    if path.exists():
        return Result(idx, ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "cpp":
        args = [
            str(HARNESS),
            "--input", str(input_wav),
            "--output", str(path.with_suffix(".wav")),
            "--circuit", circuit,
            "--params", fmt_params(params),
        ]
    elif backend == "livespice":
        swept = fmt_params(params, param_map)
        all_params = f"{fixed_params},{swept}" if fixed_params else swept
        args = [
            str(LIVESPICE_CLI),
            "--input", str(input_wav),
            "--output", str(path.with_suffix(".wav")),
            "--circuit", schx,
            "--params", all_params,
        ]
        if speaker:
            args += ["--speaker", speaker]
        if oversample and oversample != 2:
            args += ["--oversample", str(oversample)]
    else:
        return Result(idx, error=f"unknown backend: {backend}")

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with _procs_lock:
        _active_procs.add(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return Result(idx, error=f"timeout after {timeout_s}s")

        if proc.returncode != 0:
            return Result(idx, error=stderr[:500])

        dsp = proc_t = -1.0
        for line in stdout.splitlines():
            if "DSP load:" in line:
                dsp = float(line.split()[2].rstrip("%"))
            if "Processing time" in line:
                proc_t = float(line.split()[2])

        out_wav = path.with_suffix(".wav")
        sig, _sr = sf.read(str(out_wav))
        sig = sig.astype(np.float32)

        # --- WAV integrity checks (delete bad file so --resume re-runs it) ---
        if not np.isfinite(sig).all():
            out_wav.unlink(missing_ok=True)
            return Result(idx, error="NaN/Inf in output — solver may have diverged")

        if expected_frames and len(sig) < expected_frames * 0.99:
            out_wav.unlink(missing_ok=True)
            return Result(idx, error=f"truncated: {len(sig)} of {expected_frames} samples")

        rms = float(np.sqrt(np.mean(sig ** 2)))
        peak = float(np.max(np.abs(sig)))

        if rms < 1e-6:
            out_wav.unlink(missing_ok=True)
            return Result(idx, error=f"silent WAV: RMS={rms:.2e}")

        clip_frac = float(np.mean(np.abs(sig) > 0.999))
        warning = f"clipping:{clip_frac*100:.1f}%" if clip_frac > 0.01 else ""

        np.save(str(path), sig)
        out_wav.unlink(missing_ok=True)
        return Result(idx, dsp, proc_t, True, warning, rms, peak)

    except Exception as e:
        return Result(idx, error=str(e)[:500])
    finally:
        with _procs_lock:
            _active_procs.discard(proc)


# ---------------------------------------------------------------------------
# combine → single outputs.npy
# ---------------------------------------------------------------------------

def combine(out_dir: Path):
    csv_path = out_dir / "params.csv"
    if not csv_path.exists():
        print("params.csv not found", file=sys.stderr)
        sys.exit(1)

    npy_paths = sorted((out_dir / "sig").rglob("*.npy"))
    if not npy_paths:
        print("No .npy files found under sig/", file=sys.stderr)
        sys.exit(1)

    sample = np.load(str(npy_paths[0]))
    n_perms, n_samples = len(npy_paths), len(sample)

    # Cross-check .npy count against params.csv OK rows
    with open(csv_path) as f:
        ok_rows = sum(1 for r in csv.DictReader(f) if r.get("ok") == "1")
    if n_perms != ok_rows:
        print(f"WARNING: {n_perms} .npy files but {ok_rows} OK rows in params.csv — "
              f"some permutations may have failed", file=sys.stderr)

    out_path = out_dir / "outputs.npy"
    arr = np.lib.format.open_memmap(str(out_path), mode="w+",
                                    dtype=np.float32,
                                    shape=(n_perms, n_samples))
    bad_lengths = []
    for i, p in enumerate(npy_paths):
        sig = np.load(str(p))
        if len(sig) != n_samples:
            bad_lengths.append((str(p), len(sig)))
        arr[i] = sig
        p.unlink()
    arr.flush()

    if bad_lengths:
        print(f"WARNING: {len(bad_lengths)} files had unexpected length:", file=sys.stderr)
        for path, count in bad_lengths[:5]:
            print(f"  {path}: {count} samples (expected {n_samples})", file=sys.stderr)

    for d in sorted((out_dir / "sig").iterdir()):
        try: d.rmdir()
        except OSError: pass
    try: os.rmdir(out_dir / "sig")
    except OSError: pass

    print(f"Wrote {n_perms} × {n_samples} = {n_perms * n_samples // 1_000_000:.0f}M samples → {out_path}")
    print(f"Shape: ({n_perms}, {n_samples}) — {'OK' if not bad_lengths else 'WARNING: length mismatches above'}")


# ---------------------------------------------------------------------------
# post-generation dataset quality checks
# ---------------------------------------------------------------------------

def run_post_generation_checks(out_dir: Path, knobs: List[str],
                                perms: List[dict], values_per_knob: dict):
    csv_path = out_dir / "params.csv"
    if not csv_path.exists():
        return

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("ok") == "1" and row.get("rms"):
                try:
                    rows.append({**row, "_rms": float(row["rms"])})
                except ValueError:
                    pass

    if not rows:
        print("\nPost-generation checks: no OK rows with RMS data — skipping")
        return

    print("\nPost-generation checks:")
    any_warn = False

    # 1. Extreme-value audio check: each knob at its min and max must produce audio.
    print("  Extreme-value audio (RMS at min/max of each knob range):")
    for k in knobs:
        if k not in values_per_knob:
            continue
        vals = sorted(values_per_knob[k])
        if len(vals) < 2:
            continue
        for v, label in [(vals[0], "min"), (vals[-1], "max")]:
            group_rms = [r["_rms"] for r in rows if abs(float(r[k]) - v) < 1e-6]
            if not group_rms:
                continue
            median_rms = sorted(group_rms)[len(group_rms) // 2]
            if median_rms < 1e-3:
                print(f"    WARNING {k}={v} ({label}): median RMS={median_rms:.2e} — silent or near-silent")
                any_warn = True
            else:
                print(f"    {k}={v} ({label}): median RMS={median_rms:.4f}  OK")

    # 2. Knob sensitivity check: varying each knob should change the output RMS.
    print("  Knob sensitivity (RMS spread across knob range):")
    for k in knobs:
        if k not in values_per_knob or len(values_per_knob[k]) < 2:
            continue
        by_val: dict = {}
        for r in rows:
            v = round(float(r[k]), 4)
            by_val.setdefault(v, []).append(r["_rms"])
        mean_by_val = {v: sum(rms_list) / len(rms_list) for v, rms_list in by_val.items()}
        overall_mean = sum(mean_by_val.values()) / len(mean_by_val)
        rms_range = max(mean_by_val.values()) - min(mean_by_val.values())
        rel_range = rms_range / overall_mean if overall_mean > 1e-9 else 0.0
        if rel_range < 0.01:
            print(f"    WARNING {k}: RMS varies only {rel_range*100:.2f}% — knob may have no effect "
                  f"(check param_map name)")
            any_warn = True
        else:
            print(f"    {k}: RMS spread {rel_range*100:.1f}%  OK")

    if not any_warn:
        print("  All checks passed.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--list",    action="store_true", help="list cpp circuits")
    ap.add_argument("--combine", type=Path,           help="combine sharded .npy files")
    ap.add_argument("--backend", choices=["cpp", "livespice"], default="cpp")

    # livespice backend
    ap.add_argument("--schx",  type=Path, help="path to .schx file (livespice)")
    ap.add_argument("--knobs", help="comma-separated knob names to vary (livespice); "
                               "omit to list all pots in the schx")

    # cpp backend
    ap.add_argument("--circuit", help="circuit name (cpp)")

    ap.add_argument("--input",   type=Path)
    ap.add_argument("--output",  type=Path, default=HERE / "training_data")
    ap.add_argument("--workers", type=int,  default=os.cpu_count())
    ap.add_argument("--values",  help="comma-separated sweep values applied to all knobs (default: 0.1,0.3,0.5,0.7)")
    ap.add_argument("--range",        action="append", metavar="KNOB=v1,v2,...",
                                      help="per-knob value override; repeatable (e.g. --range od_master=0.0,0.5,1.0)")
    ap.add_argument("--gang", action="append", metavar="KNOB=Name1,Name2,...",
                    help="map one swept knob to several ganged pot Names (e.g. a dual-gang "
                         "pot): --gang gain=Gain_A,Gain_B . The knob is one column in the "
                         "dataset; its value is written to every listed pot. Repeatable.")
    ap.add_argument("--steps", action="append", metavar="KNOB=N",
                    help="mark a knob as a discrete N-position switch (e.g. --steps a_sym=2). "
                         "Recorded in config.json and exported as a 'steps' hint in the .nam so "
                         "the host renders a stepped control and quantizes to N positions. Set "
                         "--range to the matching values (2 -> 0,1 ; 3 -> 0,0.5,1). Repeatable.")
    ap.add_argument("--fixed-params", help="fixed k=v,... passed to every livespice_cli call (e.g. Rock=1)")
    ap.add_argument("--oversample", type=int, default=2,
                    help="livespice_cli oversampling (default 2). High-gain end-to-end amps "
                         "need more to stay numerically stable (e.g. EVH 5150 Lead full = 32); "
                         "preamps are fine at 2. Higher = proportionally slower.")
    ap.add_argument("--speaker",      help="speaker name to capture (e.g. S1, S2); default: sum all speakers")
    ap.add_argument("--random",  type=int,  help="N random permutations instead of grid")
    ap.add_argument("--bounds",  action="append", metavar="KNOB=lo,hi",
                    help="per-knob sample range under --random (repeatable; default 0,1). "
                         "e.g. --bounds LeadPre=0,0.9 to keep a hot gain pot out of a "
                         "divergent regime.")
    ap.add_argument("--no-anchors", action="store_true",
                    help="under --random, skip the deterministic boundary/corner anchor "
                         "points (by default they are prepended so knob extremes are covered)")
    ap.add_argument("--seed",    type=int,  default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        return list_circuits()
    if args.combine:
        return combine(args.combine)

    # ------------------------------------------------------------------
    # Resolve knobs and circuit identity
    # ------------------------------------------------------------------
    param_map = None
    schx = None

    if args.backend == "livespice":
        if not args.schx:
            ap.error("--schx is required for --backend livespice")
        if not args.schx.exists():
            ap.error(f"schx not found: {args.schx}")
        schx = str(args.schx)
        control_map = parse_schx_controls(schx)

        if not args.knobs:
            print(f"Controls in {args.schx.name}:")
            for k, v in sorted(control_map.items()):
                print(f"  {k:<30s}  (exact name: {v!r})")
            sys.exit(0)

        knob_keys = [k.strip() for k in args.knobs.split(",")]

        # Parse --gang: map a swept knob -> list of exact ganged pot Names.
        def _norm(s):
            return re.sub(r'_+', '_', s.strip().lower().replace(" ", "_").replace("-", "_"))
        exact_names = set(control_map.values())
        gang_map = {}
        for spec in (args.gang or []):
            key, _, names = spec.partition("=")
            key = key.strip()
            targets = []
            for t in (n.strip() for n in names.split(",") if n.strip()):
                if t in exact_names:
                    targets.append(t)
                elif _norm(t) in control_map:
                    targets.append(control_map[_norm(t)])
                else:
                    ap.error(f"--gang target '{t}' not found in schx {args.schx.name}")
            if not targets:
                ap.error(f"--gang '{spec}' lists no valid pot Names")
            gang_map[key] = targets

        # Build param_map: ganged knobs -> list of Names; others -> single exact Name.
        param_map = {}
        for k in knob_keys:
            if k in gang_map:
                param_map[k] = gang_map[k]
            else:
                param_map[k] = resolve_knobs([k], control_map)[k]
        knobs = knob_keys
        circuit_label = args.schx.stem

    else:  # cpp
        if not args.circuit:
            ap.error("--circuit is required for --backend cpp")
        if args.circuit not in CPP_CIRCUIT_KNOBS:
            print(f"Unknown circuit '{args.circuit}'. Known: {list(CPP_CIRCUIT_KNOBS)}", file=sys.stderr)
            sys.exit(1)
        knobs = CPP_CIRCUIT_KNOBS[args.circuit]
        if args.knobs:
            knobs = [k.strip() for k in args.knobs.split(",")]
        circuit_label = args.circuit

    if not args.input:
        ap.error("--input <wav> is required")

    in_wav = Path(args.input)
    if not in_wav.exists():
        ap.error(f"input WAV not found: {in_wav}")

    # ------------------------------------------------------------------
    # Generate permutations
    # ------------------------------------------------------------------
    if args.random:
        bounds = {}
        for b in (args.bounds or []):
            if "=" not in b:
                ap.error(f"--bounds must be KNOB=lo,hi (got: {b!r})")
            kname, vstr = b.split("=", 1)
            kname = kname.strip()
            if kname not in knobs:
                ap.error(f"--bounds knob '{kname}' not in --knobs list: {knobs}")
            parts = vstr.split(",")
            if len(parts) != 2:
                ap.error(f"--bounds must be KNOB=lo,hi (got: {b!r})")
            bounds[kname] = (float(parts[0]), float(parts[1]))
        perms = random_permutations(knobs, args.random, args.seed,
                                    bounds=bounds, anchors=not args.no_anchors)
    else:
        global_vals = [float(v) for v in args.values.split(",")] if args.values else [0.1, 0.3, 0.5, 0.7]
        values_per_knob = {k: global_vals for k in knobs}
        if args.range:
            for r in args.range:
                if "=" not in r:
                    ap.error(f"--range must be KNOB=v1,v2,... (got: {r!r})")
                kname, vstr = r.split("=", 1)
                kname = kname.strip()
                if kname not in values_per_knob:
                    ap.error(f"--range knob '{kname}' not in --knobs list: {knobs}")
                values_per_knob[kname] = [float(v) for v in vstr.split(",")]
        perms = grid_permutations(knobs, values_per_knob)

    # Parse --steps: mark knobs as discrete N-position switches (metadata for exporter/host).
    steps_map = {}
    for spec in (args.steps or []):
        if "=" not in spec:
            ap.error(f"--steps must be KNOB=N (got: {spec!r})")
        kname, nstr = spec.split("=", 1)
        kname = kname.strip()
        if kname not in knobs:
            ap.error(f"--steps knob '{kname}' not in knobs list: {knobs}")
        try:
            n = int(nstr)
        except ValueError:
            ap.error(f"--steps count must be an integer (got: {nstr!r})")
        if n < 2:
            ap.error(f"--steps count must be >= 2 (got: {n})")
        steps_map[kname] = n

    wav_info = sf.info(str(in_wav))
    audio_frames = wav_info.frames
    sr = wav_info.samplerate
    # 10x realtime at the default 2x oversample; scale up with oversample since
    # sim cost is ~proportional to it (e.g. 32x -> ~160x audio length).
    timeout_s = max(120, int(audio_frames / sr * 10 * max(1, args.oversample / 2)))

    bytes_per_perm = audio_frames * 4  # float32
    total_bytes = bytes_per_perm * len(perms) + in_wav.stat().st_size  # npy + sweep.wav copy
    avail_bytes = shutil.disk_usage(args.output.parent if not args.output.exists() else args.output).free

    def fmt_bytes(n):
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or unit == "TB":
                return f"{n:.1f} {unit}"
            n /= 1024

    print(f"Backend:      {args.backend}")
    print(f"Circuit:      {circuit_label}")
    print(f"Knobs:        {', '.join(knobs)}")
    if steps_map:
        print(f"Stepped:      {steps_map}")
    if param_map:
        for k, v in param_map.items():
            print(f"  {k} → {v!r}")
    print(f"Input:        {in_wav}")
    if args.fixed_params:
        print(f"Fixed params: {args.fixed_params}")
    if args.speaker:
        print(f"Speaker:      {args.speaker}")
    print(f"Permutations: {len(perms)}")
    if not args.random:
        for k in knobs:
            vals = values_per_knob[k]
            print(f"  {k}: {vals}")
    print(f"Workers:      {args.workers}")
    print(f"Timeout:      {timeout_s}s per permutation ({audio_frames/sr:.0f}s audio × 10 × os/2, oversample={args.oversample})")
    print(f"Disk est:     {fmt_bytes(total_bytes)} needed  ({fmt_bytes(avail_bytes)} free on target)")
    print()

    if args.dry_run:
        for i, p in enumerate(perms[:5]):
            print(f"  [{i:06d}] {p}")
        if len(perms) > 5:
            print(f"  ... and {len(perms) - 5} more")
        return

    if args.backend == "cpp" and not HARNESS.exists():
        print(f"Harness not found at {HARNESS}. Build it first.", file=sys.stderr)
        sys.exit(1)
    if args.backend == "livespice" and not LIVESPICE_CLI.exists():
        print(f"livespice_cli not found at {LIVESPICE_CLI}. Build it first.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Effective per-knob sampled range (min/max actually seen across perms).
    # Recorded so the exported .nam declares the true trained domain per knob
    # — e.g. a tone pot swept only 0.15..0.85 must not advertise 0..1 to the
    # host, or it would send out-of-domain values the model never saw.
    knob_bounds = {k: [min(p[k] for p in perms), max(p[k] for p in perms)]
                   for k in knobs}

    (out_dir / "config.json").write_text(json.dumps({
        "backend": args.backend,
        "circuit": circuit_label,
        "schx": schx,
        "knobs": knobs,
        "steps": steps_map,
        "bounds": knob_bounds,
        "oversample": args.oversample,
        "param_map": param_map,
        "fixed_params": args.fixed_params,
        "speaker": args.speaker,
        "input_wav": str(in_wav),
        "permutation_count": len(perms),
        "values": args.values or "0.1,0.3,0.5,0.7",
        "workers": args.workers,
    }, indent=2))

    (out_dir / "sweep.wav").write_bytes(in_wav.read_bytes())

    csv_path = out_dir / "params.csv"
    fresh = not csv_path.exists()
    csv_fh = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_fh)
    if fresh:
        csv_w.writerow(["idx"] + knobs + ["dsp_load", "proc_time", "rms", "peak", "ok", "error"])

    existing = {int(p.stem) for p in (out_dir / "sig").rglob("*.npy")} if (out_dir / "sig").exists() else set()
    to_run = [(i, p) for i, p in enumerate(perms) if i not in existing]
    print(f"Resume: {len(existing)} done, {len(to_run)} remaining\n")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_one, i, p, out_dir, in_wav,
                        args.backend, args.circuit, schx, param_map,
                        args.fixed_params, args.speaker, audio_frames, timeout_s,
                        args.oversample): i
            for i, p in to_run
        }
        for f in as_completed(futs):
            idx = futs[f]
            r = f.result()
            done += 1
            elapsed = time.time() - t0
            eta = (len(to_run) - done) * elapsed / done
            finish = time.strftime("%H:%M", time.localtime(time.time() + eta))
            p = perms[idx]
            row = ([idx] + [p[k] for k in knobs] +
                   [r.dsp_load, r.proc_time, f"{r.rms:.6f}", f"{r.peak:.6f}", int(r.ok), r.error])
            csv_w.writerow(row)
            csv_fh.flush()
            status = "OK" if r.ok else "FAIL"
            extra = f"  [{r.error}]" if r.error else ""
            print(f"[{done:>4}/{len(to_run)}] {done/len(to_run)*100:>5.1f}%  "
                  f"perm_{idx:06d}  {status}  "
                  f"DSP={r.dsp_load:.1f}%  elapsed={elapsed:.0f}s  ETA ~{finish}{extra}")

    csv_fh.close()
    total = time.time() - t0
    ok = len(list((out_dir / "sig").rglob("*.npy"))) if (out_dir / "sig").exists() else 0
    print(f"\nDone: {ok}/{len(perms)} files in {total:.0f}s "
          f"({total / max(len(perms), 1) * 1000:.0f} ms/perm)")

    run_post_generation_checks(out_dir, knobs, perms, values_per_knob if not args.random else {})

    print(f"\nPost-process with:  python batch_harness.py --combine {out_dir}")


if __name__ == "__main__":
    main()
