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

import argparse, csv, json, os, re, shutil, subprocess, sys, time
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


def random_permutations(knobs: List[str], n: int, seed: int = 0) -> List[dict]:
    import random as rng
    rng.seed(seed)
    return [dict(zip(knobs, [rng.random() for _ in knobs])) for _ in range(n)]


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


def sig_path(out_dir: Path, idx: int) -> Path:
    return out_dir / "sig" / f"{idx // 100:02d}" / f"{idx:06d}.npy"


def process_one(idx: int, params: dict, out_dir: Path, input_wav: Path,
                backend: str, circuit: str = None, schx: str = None,
                param_map: dict = None, fixed_params: str = None,
                speaker: str = None) -> Result:
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
    else:
        return Result(idx, error=f"unknown backend: {backend}")

    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            return Result(idx, error=r.stderr[:500])

        dsp = proc = -1.0
        for line in r.stdout.splitlines():
            if "DSP load:" in line:
                dsp = float(line.split()[2].rstrip("%"))
            if "Processing time" in line:
                proc = float(line.split()[2])

        sig, _sr = sf.read(str(path.with_suffix(".wav")))
        np.save(str(path), sig)
        Path(str(path.with_suffix(".wav"))).unlink(missing_ok=True)
        return Result(idx, dsp, proc, True)
    except Exception as e:
        return Result(idx, error=str(e)[:500])


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

    out_path = out_dir / "outputs.npy"
    arr = np.lib.format.open_memmap(str(out_path), mode="w+",
                                    dtype=np.float32,
                                    shape=(n_perms, n_samples))
    for i, p in enumerate(npy_paths):
        arr[i] = np.load(str(p))
        p.unlink()
    arr.flush()

    for d in sorted((out_dir / "sig").iterdir()):
        try: d.rmdir()
        except OSError: pass
    try: os.rmdir(out_dir / "sig")
    except OSError: pass

    print(f"Wrote {n_perms} × {n_samples} = {n_perms * n_samples // 1_000_000:.0f}M samples → {out_path}")


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
    ap.add_argument("--fixed-params", help="fixed k=v,... passed to every livespice_cli call (e.g. Rock=1)")
    ap.add_argument("--speaker",      help="speaker name to capture (e.g. S1, S2); default: sum all speakers")
    ap.add_argument("--random",  type=int,  help="N random permutations instead of grid")
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
        perms = random_permutations(knobs, args.random, args.seed)
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

    print(f"Backend:      {args.backend}")
    print(f"Circuit:      {circuit_label}")
    print(f"Knobs:        {', '.join(knobs)}")
    if param_map:
        for k, v in param_map.items():
            print(f"  {k} → {v!r}")
    audio_frames = sf.info(str(in_wav)).frames
    bytes_per_perm = audio_frames * 4  # float32
    total_bytes = bytes_per_perm * len(perms) + in_wav.stat().st_size  # npy + sweep.wav copy
    avail_bytes = shutil.disk_usage(args.output.parent if not args.output.exists() else args.output).free

    def fmt_bytes(n):
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or unit == "TB":
                return f"{n:.1f} {unit}"
            n /= 1024

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

    (out_dir / "config.json").write_text(json.dumps({
        "backend": args.backend,
        "circuit": circuit_label,
        "schx": schx,
        "knobs": knobs,
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
        csv_w.writerow(["idx"] + knobs + ["dsp_load", "proc_time", "ok", "error"])

    existing = {int(p.stem) for p in (out_dir / "sig").rglob("*.npy")} if (out_dir / "sig").exists() else set()
    to_run = [(i, p) for i, p in enumerate(perms) if i not in existing]
    print(f"Resume: {len(existing)} done, {len(to_run)} remaining\n")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(process_one, i, p, out_dir, in_wav,
                        args.backend, args.circuit, schx, param_map,
                        args.fixed_params, args.speaker): i
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
            row = [idx] + [p[k] for k in knobs] + [r.dsp_load, r.proc_time, int(r.ok), r.error]
            csv_w.writerow(row)
            csv_fh.flush()
            print(f"[{done:>4}/{len(to_run)}] {done/len(to_run)*100:>5.1f}%  "
                  f"perm_{idx:06d}  {'OK' if r.ok else 'FAIL'}  "
                  f"DSP={r.dsp_load:.1f}%  elapsed={elapsed:.0f}s  ETA ~{finish}")

    csv_fh.close()
    total = time.time() - t0
    ok = len(list((out_dir / "sig").rglob("*.npy"))) if (out_dir / "sig").exists() else 0
    print(f"\nDone: {ok}/{len(perms)} files in {total:.0f}s "
          f"({total / max(len(perms), 1) * 1000:.0f} ms/perm)")
    print(f"\nPost-process with:  python batch_harness.py --combine {out_dir}")


if __name__ == "__main__":
    main()
