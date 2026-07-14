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

import atexit, argparse, csv, json, os, re, shutil, signal, subprocess, sys, threading, time, tomllib
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

# Per-circuit ngspice defaults, keyed by a substring of the .schx filename. Applied
# automatically when the matching CLI flag is unset, so a circuit's known-good
# convergence/damping config isn't re-typed each run (CLI flags win). Analogous to
# how the 5150 ot-damp/nfb-comp are hand-set today, but auto-derived per circuit.
# Data lives in ngspice/circuit_defaults.toml, not here -- add a new circuit's
# tuning there (a data file), not by editing this shared script.
def _load_ngspice_circuit_defaults() -> dict:
    path = HERE / "ngspice" / "circuit_defaults.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


NGSPICE_CIRCUIT_DEFAULTS = _load_ngspice_circuit_defaults()

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
    # What it actually took to make this permutation converge. Recorded per row, because
    # "this dataset needed 4x oversampling at high gain" is a property OF THE DATA and the
    # next person deserves to know it without re-deriving it.
    rung: int = 0
    settings: str = ""


def sig_path(out_dir: Path, idx: int) -> Path:
    return out_dir / "sig" / f"{idx // 100:02d}" / f"{idx:06d}.npy"


def _finalize_wav(idx, path, out_wav, expected_frames, max_crest, dsp=-1.0, proc_t=-1.0,
                   warmup_s=1.0):
    """Read out_wav, run integrity + crest checks, save .npy. Shared by all backends."""
    sig, sr = sf.read(str(out_wav))
    sig = sig.astype(np.float32)
    if not np.isfinite(sig).all():
        out_wav.unlink(missing_ok=True)
        return Result(idx, error="NaN/Inf in output — solver may have diverged")
    if expected_frames and len(sig) < expected_frames * 0.99:
        out_wav.unlink(missing_ok=True)
        return Result(idx, error=f"truncated: {len(sig)} of {expected_frames} samples")
    # Skip the DC-solver startup transient (~0.5-1s ring before settling, see circuit
    # docs' "Known artifacts") when computing divergence-detection stats, so it doesn't
    # trip a false positive on otherwise-clean, quiet (e.g. low-gain) permutations. The
    # full untrimmed signal is still saved — sample position must stay aligned with
    # sweep.wav, which param_train.py's ParamDataset crops using the same offset.
    warmup_n = int(warmup_s * sr)
    stats_sig = sig[warmup_n:] if len(sig) > warmup_n else sig
    rms = float(np.sqrt(np.mean(stats_sig ** 2)))
    peak = float(np.max(np.abs(stats_sig)))
    if rms < 1e-6:
        out_wav.unlink(missing_ok=True)
        return Result(idx, error=f"silent WAV: RMS={rms:.2e}")
    # Crest factor (peak/RMS) is a scale-invariant divergence detector: clean audio
    # sits <~10, numerical runaway spikes to tens–thousands. Fail those.
    crest = peak / (rms + 1e-9)
    if max_crest > 0 and crest > max_crest:
        out_wav.unlink(missing_ok=True)
        return Result(idx, error=f"unstable: crest={crest:.0f} > --max-crest {max_crest:g} "
                                 f"(peak={peak:.1f} rms={rms:.2f}) — likely numerical divergence")
    warn = []
    if crest > 15:
        warn.append(f"crest:{crest:.0f}")
    if peak <= 2.0:
        clip_frac = float(np.mean(np.abs(stats_sig) > 0.999))
        if clip_frac > 0.01:
            warn.append(f"clipping:{clip_frac*100:.1f}%")
    np.save(str(path), sig)
    out_wav.unlink(missing_ok=True)
    return Result(idx, dsp, proc_t, True, " ".join(warn), rms, peak)


def _run_ngspice(idx, params, path, out_wav, expected_frames, timeout_s,
                 param_map, fixed_params, ng):
    """Translate netlist (baked pots) -> ngspice -> resample to 48k -> out_wav.
    Returns a failure Result, or None on success."""
    import sys as _sys
    _ngdir = str(Path(__file__).resolve().parent / "ngspice")
    if _ngdir not in _sys.path:
        _sys.path.insert(0, _ngdir)
    import schx_to_ngspice as X

    nl = json.load(open(ng["netlist"]))
    pmap = param_map or {}
    pots = {pmap.get(k, k): float(v) for k, v in params.items()}     # knob -> pot Name
    if fixed_params:
        for kv in fixed_params.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                pots[k.strip()] = float(v)
    dur = expected_frames / 48000.0
    csv_path, cir_path = path.with_suffix(".csv"), path.with_suffix(".cir")
    over = int(ng.get("oversample", 1) or 1)
    cir_path.write_text(X.translate(nl, pots=pots, input_pwl=ng["input"], dur=dur,
                                    csv=str(csv_path), koren=ng.get("koren", False),
                                    ot_damp=ng.get("ot_damp", "47k"), ot_snub=ng.get("ot_snub", "10n"),
                                    nfb_comp=ng.get("nfb_comp"), input_mode="filesource",
                                    oversample=over, conv=ng.get("conv"),
                                    method=ng.get("method", "trap")))
    proc = subprocess.Popen(["ngspice", "-b", str(cir_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with _procs_lock:
        _active_procs.add(proc)
    try:
        try:
            proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate()
            return Result(idx, error=f"ngspice timeout after {timeout_s}s")
    finally:
        with _procs_lock:
            _active_procs.discard(proc)
    cir_path.unlink(missing_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return Result(idx, error="ngspice produced no output (diverged at t~0?)")
    d = np.loadtxt(str(csv_path))
    csv_path.unlink(missing_ok=True)
    if d.ndim < 2 or len(d) < 2:
        return Result(idx, error="ngspice: too few rows (diverged at t~0)")
    ng_t, ng_v = d[:, 0], d[:, 1]
    if ng_t[-1] < dur * 0.99:
        return Result(idx, error=f"ngspice aborted at {ng_t[-1]:.3f}s of {dur:.3f}s (diverged)")
    if over > 1:
        # anti-aliased resample: interp to over*48k uniform, FIR-decimate to 48k.
        # This removes the >24kHz content the amp emits before it folds into audband.
        from scipy.signal import decimate
        grid_hi = np.arange(expected_frames * over) / (48000.0 * over)
        sig_hi = np.interp(grid_hi, ng_t, ng_v)
        sig = decimate(sig_hi, over, ftype="fir", zero_phase=True).astype(np.float32)
        if len(sig) < expected_frames:
            sig = np.pad(sig, (0, expected_frames - len(sig)))
        sig = sig[:expected_frames]
    else:
        grid = np.arange(expected_frames) / 48000.0    # naive interp (aliases HF)
        sig = np.interp(grid, ng_t, ng_v).astype(np.float32)
    sf.write(str(out_wav), sig, 48000, subtype="FLOAT")
    return None


# ---------------------------------------------------------------------------
# convergence escalation
# ---------------------------------------------------------------------------

# Failures a stiffer solve can actually fix. Anything else -- a bad knob name, a missing file,
# an unknown backend -- will fail identically on every rung, so retrying just burns an hour.
_CONVERGENCE_FAILURE = re.compile(
    r"diverg|NaN|Inf|timestep too small|singular|convergence|no convergence|"
    r"unstable|crest|truncated|timeout|iteration",
    re.IGNORECASE)


def _is_convergence_failure(err: str) -> bool:
    return bool(err) and bool(_CONVERGENCE_FAILURE.search(err))


def _rungs(backend: str, oversample: int, ng: dict) -> list:
    """Escalating convergence settings, cheapest first.

    We ALREADY KNEW how to fix these -- the Boss DS-1 needed input_upsample=4, the MT-2 needed
    diode_cjo damping -- but that knowledge was hand-entered per circuit in circuit_defaults.toml
    and NEVER REACHED FOR when a render actually failed. A failed permutation just recorded its
    error and the run carried on with a hole in the dataset.

    So: escalate on failure, and record which rung won.

    The livespice ladder raises --iterations as well as --oversample, which matters more than it
    looks: livespice_cli DEFAULTS TO 8 NEWTON ITERATIONS and 8 is not enough for stiff circuits.
    The Ibanez TS-9 needs 45 -- at the default the SOLVER SILENTLY STOPS SHORT and hands back a
    converged-looking answer that is 5.8e-03 wrong. That is not a crash; it is worse, because
    nothing reports it. (See livespice-emitter, which now measures the cap per circuit.)
    """
    if backend == "livespice":
        os_ = oversample or 2
        # RUNG 0 IS ALREADY 256 ITERATIONS, and that is not paranoia -- it is a bug fix.
        #
        # livespice_cli defaults to 8 Newton iterations and 8 IS NOT ENOUGH for a stiff circuit.
        # The Ibanez TS-9 at Drive=0.5 needs 45; below that THE SOLVER SILENTLY STOPS SHORT and
        # hands back a converged-LOOKING answer. Measured against the same render at 256, on the
        # ACTUAL dataset input (sweep60_composite.wav): ESR 7.38e-03. That is -21 dB of error,
        # baked into the training data, with nothing to catch it. It is not a crash, so the retry
        # ladder below cannot save us -- there is no failure to retry.
        #
        # And it is FREE: LiveSPICE breaks out of the Newton loop the moment it converges, so a
        # high ceiling costs nothing where it is not needed. Measured render time, TS-9 and Big
        # Muff, --iterations 8 vs 128: identical to 0.01 s.
        #
        # An under-converged dataset is worse than a failed one. A failed permutation is a hole
        # you can see; an under-converged one is a lie you cannot.
        return [
            dict(oversample=os_,     iterations=256),
            dict(oversample=os_ * 2, iterations=256),    # halve the timestep
            dict(oversample=os_ * 4, iterations=256),    # last resort
        ]

    if backend == "ngspice":
        base = dict(ng or {})
        up = int(base.get("input_upsample", 1) or 1)
        out = [dict(base)]
        for u in (max(up, 2), max(up, 4)):
            r = dict(base); r["input_upsample"] = u
            out.append(r)
        r = dict(out[-1]); r["method"] = "gear"          # the DS-1's fix
        out.append(r)
        r = dict(r)                                      # + the MT-2's fix
        r["conv"] = ",".join(x for x in (base.get("conv"), "diode_cjo=100p") if x)
        out.append(r)
        return out

    if backend == "cpp":
        os_ = oversample or 2
        return [dict(oversample=os_), dict(oversample=os_ * 2), dict(oversample=os_ * 4)]

    return [dict()]


def _rung_str(rung: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(rung.items()) if v not in (None, "", 0))


def process_one(idx: int, params: dict, out_dir: Path, input_wav: Path,
                backend: str, circuit: str = None, schx: str = None,
                param_map: dict = None, fixed_params: str = None,
                speaker: str = None, expected_frames: int = 0,
                timeout_s: int = 1200, oversample: int = 2,
                max_crest: float = 0.0, ng: dict = None,
                warmup_s: float = 1.0, no_retry: bool = False) -> Result:
    """Render one permutation, ESCALATING THE SOLVER when it fails to converge.

    A failed permutation used to record its error and be forgotten -- leaving a hole in the
    dataset that only surfaced later as "WARNING: N .npy files but M OK rows". We already knew
    how to fix these (the DS-1 needed input_upsample=4, the MT-2 needed diode_cjo damping); that
    knowledge was just never reached for automatically.
    """
    path = sig_path(out_dir, idx)
    if path.exists():
        return Result(idx, ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    rungs = [dict()] if no_retry else _rungs(backend, oversample, ng)
    last = None
    for i, rung in enumerate(rungs):
        os_i = rung.get("oversample", oversample)
        ng_i = ng
        if backend == "ngspice":
            ng_i = rung or ng
        r = _render_once(idx, params, out_dir, input_wav, backend, circuit, schx,
                         param_map, fixed_params, speaker, expected_frames, timeout_s,
                         os_i, max_crest, ng_i, warmup_s,
                         iterations=rung.get("iterations"))
        if r.ok:
            r.rung, r.settings = i, (_rung_str(rung) if i else "")
            if i:
                print(f"  [{idx}] converged on rung {i}: {_rung_str(rung)}", file=sys.stderr)
            return r
        last = r
        # Only a CONVERGENCE failure is worth another attempt. A bad knob name or a missing file
        # fails identically on every rung; retrying it just burns the clock.
        if not _is_convergence_failure(r.error):
            return r
        if i + 1 < len(rungs):
            print(f"  [{idx}] {r.error[:80]} — escalating to rung {i+1}: "
                  f"{_rung_str(rungs[i+1])}", file=sys.stderr)

    if last is not None:
        last.error = f"{last.error} [exhausted {len(rungs)} convergence rungs]"
    return last or Result(idx, error="no rungs")


def _render_once(idx: int, params: dict, out_dir: Path, input_wav: Path,
                 backend: str, circuit: str = None, schx: str = None,
                 param_map: dict = None, fixed_params: str = None,
                 speaker: str = None, expected_frames: int = 0,
                 timeout_s: int = 1200, oversample: int = 2,
                 max_crest: float = 0.0, ng: dict = None,
                 warmup_s: float = 1.0, iterations: int = None) -> Result:
    path = sig_path(out_dir, idx)
    out_wav = path.with_suffix(".wav")

    try:
        if backend == "ngspice":
            err = _run_ngspice(idx, params, path, out_wav, expected_frames, timeout_s,
                               param_map, fixed_params, ng or {})
            return err or _finalize_wav(idx, path, out_wav, expected_frames, max_crest,
                                        warmup_s=warmup_s)

        if backend == "cpp":
            args = [str(HARNESS), "--input", str(input_wav), "--output", str(out_wav),
                    "--circuit", circuit, "--params", fmt_params(params)]
        elif backend == "livespice":
            swept = fmt_params(params, param_map)
            all_params = f"{fixed_params},{swept}" if fixed_params else swept
            args = [str(LIVESPICE_CLI), "--input", str(input_wav), "--output", str(out_wav),
                    "--circuit", schx, "--params", all_params]
            if speaker:
                args += ["--speaker", speaker]
            if oversample and oversample != 2:
                args += ["--oversample", str(oversample)]
            # livespice_cli DEFAULTS TO 8 NEWTON ITERATIONS, and 8 is not enough for stiff
            # circuits -- the Ibanez TS-9 needs 45. Below that the solver stops short and returns
            # a converged-LOOKING answer that is 5.8e-03 wrong, silently. Not a crash: worse.
            if iterations:
                args += ["--iterations", str(iterations)]
        else:
            return Result(idx, error=f"unknown backend: {backend}")

        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with _procs_lock:
            _active_procs.add(proc)
        try:
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.communicate()
                return Result(idx, error=f"timeout after {timeout_s}s")
            if proc.returncode != 0:
                return Result(idx, error=stderr[:500])
            dsp = proc_t = -1.0
            for line in stdout.splitlines():
                if "DSP load:" in line:
                    dsp = float(line.split()[2].rstrip("%"))
                if "Processing time" in line:
                    proc_t = float(line.split()[2])
        finally:
            with _procs_lock:
                _active_procs.discard(proc)
        return _finalize_wav(idx, path, out_wav, expected_frames, max_crest, dsp, proc_t,
                             warmup_s=warmup_s)

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


def audit_convergence(out_dir: Path, knobs: list, perms: list, backend: str,
                      input_wav: Path, schx: str, param_map: dict, fixed_params: str,
                      speaker: str, oversample: int, iterations: int,
                      warmup_s: float, n_probe: int = 8):
    """Is the DATASET converged? The only check that can see it.

    Every other check we run detects DIVERGENCE -- NaN, crest, truncation, timeout -- and those
    are the easy case: the solver blew up and said so. UNDER-CONVERGENCE is the dangerous one. The
    solver stops short of the answer and returns a perfectly plausible waveform: normal RMS, normal
    crest, no NaN. Every check passes. The data is simply WRONG.

    The Ibanez TS-9 at Drive=0.5 needs 45 Newton iterations. livespice_cli defaults to 8. On the
    real dataset input the resulting error is ESR 7.38e-03 -- -21 dB -- and it went straight into
    the training set of a shipped model with nothing to catch it.

    THE ONLY WAY TO SEE IT IS TO SOLVE IT HARDER AND COMPARE -- but you must solve the SAME
    EQUATIONS harder, i.e. hold the TIMESTEP FIXED and only raise the iteration count.

    This distinction is the whole design, and getting it wrong makes the audit useless:

      NEWTON UNDER-CONVERGENCE   the solver stopped short of the answer to the equations it was
                                 given. A BUG. Silently wrong. Detect by raising ITERATIONS at a
                                 FIXED timestep: if the answer moves, it had not converged.

      DISCRETIZATION ERROR       BDF2's inherent O(h^2) truncation. NOT a bug, and never zero.
                                 Halving h always changes the answer by ~4x less. Every circuit
                                 has it. Reported as INFO so you can judge whether --oversample is
                                 adequate, but it must NEVER be an error -- an audit that fires on
                                 every circuit is an audit nobody reads.

    (The first cut of this function doubled the oversample too, and duly reported the Big Muff --
    which is Newton-converged at 8 iterations, ESR 0.00e+00 -- as "UNDER-CONVERGED". It was
    measuring the method, not the mistake.)

    We probe the EXTREMES, because that is where stiffness lives -- and where the default knob
    position, which is what everything else checks, would never take us.
    """
    if backend != "livespice" or not perms:
        return
    import tempfile

    # BOTH ends of every knob, plus both corners. "All knobs at max" is NOT reliably the stiff
    # setting: the Boss DS-1's Dist pot is ReverseLinear, so all-max is MINIMUM drive -- and its
    # truncation error is 1.9e-02 at the DEFAULTS and 3.3e-04 at all-max, i.e. 57x worse in the
    # direction we would not have looked. Probe both ends and let the measurement decide.
    picks = []
    for k in knobs:
        lo, hi = min(p[k] for p in perms), max(p[k] for p in perms)
        for v in (lo, hi):
            picks.append(min((p for p in perms if p[k] == v),
                             key=lambda p: abs(sum(p.values()) - sum(perms[0].values()))))
    picks.append(max(perms, key=lambda p: sum(p.values())))   # all-max corner
    picks.append(min(perms, key=lambda p: sum(p.values())))   # all-min corner
    seen, uniq = set(), []
    for p in picks:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key); uniq.append(p)
    uniq = uniq[:n_probe]

    print("  Convergence audit (same timestep, 4x the iterations — did Newton actually converge?):")
    worst, worst_at = 0.0, None
    worst_disc, worst_disc_at = 0.0, None
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, p in enumerate(uniq):
            swept = fmt_params(p, param_map)
            allp = f"{fixed_params},{swept}" if fixed_params else swept
            outs = []
            # SAME timestep, 4x the iterations -> isolates NEWTON convergence.
            # Then same iterations, 2x the timestep -> the discretization error, for INFO only.
            for tag, os_, it_ in (("base",  oversample,     iterations),
                                  ("newton", oversample,    iterations * 4),
                                  ("finer",  oversample * 2, iterations)):
                w = td / f"{tag}.wav"
                a = [str(LIVESPICE_CLI), "--input", str(input_wav), "--output", str(w),
                     "--circuit", schx, "--params", allp,
                     "--oversample", str(os_), "--iterations", str(it_)]
                if speaker:
                    a += ["--speaker", speaker]
                r = subprocess.run(a, capture_output=True, text=True)
                if r.returncode != 0 or not w.exists():
                    print(f"    (probe render failed: rc={r.returncode} {r.stderr[:120]})")
                    outs = []
                    break
                sig, sr = sf.read(str(w))
                outs.append(np.asarray(sig, dtype=np.float64))
            if len(outs) != 3:
                continue
            n = min(len(o) for o in outs)
            # Clamp the warm-up skip. A skip longer than the clip leaves one sample to compare and
            # the ESR becomes noise -- which is how this audit first reported a nonsensical 2.08 on
            # a 1-second probe. Never let a window silently eat the signal you are measuring.
            sk = min(int(warmup_s * 48000), max(0, n // 10))
            base, newton, finer = (o[sk:n] for o in outs)
            if len(base) < 4800:
                continue

            def _esr(a_, b_):
                den = float(np.sum(b_ ** 2))
                return float(np.sum((a_ - b_) ** 2) / den) if den > 0 else 0.0

            esr = _esr(base, newton)          # THE BUG CHECK: same equations, solved harder
            disc = _esr(base, finer)          # informational: the method's own truncation error
            # `>=`, not `>`. A PERFECTLY converged circuit gives esr == 0.0 exactly, and `> 0.0`
            # is false -- so the audit reported "could not probe" on the very circuits that had
            # nothing wrong with them. A success that looks like a failure to run is worse than
            # either.
            if worst_at is None or esr > worst:
                worst, worst_at = esr, p
            if worst_disc_at is None or disc > worst_disc:
                worst_disc, worst_disc_at = disc, p

    if worst_at is None:
        print("    (could not probe)")
        return
    at = " ".join(f"{k}={worst_at[k]:g}" for k in knobs)
    if worst > 1e-5:
        print(f"    *** UNDER-CONVERGED: ESR {worst:.2e} at [{at}] ***")
        print(f"    Rendered at oversample={oversample}, iterations={iterations}. Solving the SAME")
        print(f"    equations 4x harder MOVES the answer, so Newton stopped short and this dataset")
        print(f"    is WRONG at that setting -- and nothing else would have told you: the RMS is")
        print(f"    normal, the crest is normal, there is no NaN.")
        print(f"    Raise the iteration count and regenerate. See docs/convergence.md.")
    else:
        print(f"    OK — worst ESR {worst:.2e} at [{at}] (solving harder does not move it)")
    if worst_disc_at is not None:
        dat = " ".join(f"{k}={worst_disc_at[k]:g}" for k in knobs)
        print(f"    (info) BDF2 truncation at oversample={oversample}: {worst_disc:.2e} at [{dat}]. "
              f"This is the METHOD, not a bug — it shrinks ~4x each time you double --oversample. "
              f"Raise it if you want a tighter model of the circuit.")


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
    ap.add_argument("--backend", choices=["cpp", "livespice", "ngspice"], default="cpp")
    ap.add_argument("--no-retry", action="store_true",
                    help="Do NOT escalate solver settings when a permutation fails to converge. "
                         "By default a failed render is retried with a stiffer solve (more Newton "
                         "iterations, finer timestep, ngspice damping) and the rung that won is "
                         "recorded per row in params.csv. Use this only to see the raw failure.")
    # ngspice backend (offline adaptive-timestep SPICE for stiff/high-gain amps)
    ap.add_argument("--koren", action="store_true",
                    help="ngspice: Koren triode model (softer, for stiff amps) vs exact DempwolfZolzer")
    ap.add_argument("--ot-damp", default="47k", help="ngspice: OT plate-to-plate damper R (heavier=more stable)")
    ap.add_argument("--ot-snub", default="10n", help="ngspice: OT snubber C")
    ap.add_argument("--nfb-comp", default=None, help="ngspice: NFB compensation cap NODE=value (e.g. nNFB=1n)")
    ap.add_argument("--conv", default="", help="ngspice: device convergence overrides key=val,... "
                    "(diode_cjo/diode_tt/bjt_*/jfet_*/tmax; e.g. diode_cjo=100p,tmax=20.8333u). "
                    "tmax caps the max internal solver step (slower, more robust through hard "
                    "transients). Auto-set per-circuit if omitted.")
    ap.add_argument("--method", default="", choices=["", "trap", "gear"],
                    help="ngspice: integration method (default trap). Auto-set per-circuit if omitted.")
    ap.add_argument("--input-upsample", type=int, default=0,
                    help="ngspice: upsample the input audio N-fold (bandlimited resample) before "
                         "writing the filesource file. Fixes solver non-convergence on near-Nyquist "
                         "sweep content by giving ngspice's linear interpolation a smoother waveform "
                         "-- unlike --oversample, which only affects the solver step hint and output "
                         "decimation, not the input's actual time resolution. Auto-set per-circuit "
                         "if omitted (0).")

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
    ap.add_argument("--defaults", help="per-knob DEFAULT value, k=v,... (in trained units). "
                    "Recorded in the .nam's parameters[].default so a bake with no --params "
                    "uses the circuit's real default position, not the range midpoint "
                    "(e.g. a Timmy's controls don't center at noon).")
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
    ap.add_argument("--max-crest", type=float, default=50.0,
                    help="fail (exclude) any permutation whose output crest factor "
                         "(peak/RMS) exceeds this — catches numerical divergence that "
                         "RMS/length checks miss. Clean audio is <~10; high-gain solver "
                         "runaway is tens–thousands. 0 disables. (default: %(default)s)")
    ap.add_argument("--skip-warmup-s", type=float, default=1.0,
                    help="seconds of DC-solver startup transient to exclude from the "
                         "rms/peak/crest divergence-detection stats (the transient can "
                         "otherwise false-positive --max-crest, especially on quiet/"
                         "low-gain permutations). The saved .npy is NOT trimmed — sample "
                         "position must stay aligned with sweep.wav. 0 disables. "
                         "(default: %(default)s)")
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

    if args.backend in ("livespice", "ngspice"):
        if not args.schx:
            ap.error(f"--schx is required for --backend {args.backend}")
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
    if args.backend == "ngspice":
        # ngspice adaptive sim can run ~5-50x realtime on stiff amps; be generous
        # (a divergent perm aborts near t=0 anyway, so this mainly guards hangs).
        timeout_s = max(600, int(audio_frames / sr * 120))

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
    if args.backend in ("livespice", "ngspice") and not LIVESPICE_CLI.exists():
        print(f"livespice_cli not found at {LIVESPICE_CLI}. Build it first.", file=sys.stderr)
        sys.exit(1)
    if args.backend == "ngspice" and not shutil.which("ngspice"):
        print("ngspice not found on PATH. Install it (e.g. apt install ngspice).", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ngspice backend: dump the authoritative netlist once + write the input as an
    # XSPICE-filesource file (time,raw-sample; V0dBFS applied by the translator).
    ng = None
    if args.backend == "ngspice":
        netlist_path = out_dir / "netlist.json"
        r = subprocess.run([str(LIVESPICE_CLI), "--circuit", schx, "--netlist", str(netlist_path)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not netlist_path.exists():
            print(f"netlist dump failed: {r.stderr[:300]}", file=sys.stderr)
            sys.exit(1)
        insig, _isr = sf.read(str(in_wav))
        if insig.ndim > 1:
            insig = insig.mean(axis=1)
        # convergence overrides: CLI --conv/--method/--input-upsample win, else per-circuit default (auto)
        _cdef = next((v for k, v in NGSPICE_CIRCUIT_DEFAULTS.items() if k in Path(schx).name), {})
        _conv_str = args.conv or _cdef.get("conv", "")
        _method = args.method or _cdef.get("method", "trap")
        _input_up = args.input_upsample or int(_cdef.get("input_upsample", 1) or 1)
        conv = dict(kv.split("=", 1) for kv in _conv_str.split(",") if "=" in kv)
        fsrc = out_dir / "input_fsrc.txt"
        fsrc_sr = sr  # NOTE: local to the filesource write -- `sr` itself (48kHz)
                      # drives output frame-count/timeout logic elsewhere and must
                      # not be mutated by upsampling the ngspice INPUT only.
        if _input_up > 1:
            # filesource has no explicit interpolation setting -> ngspice linearly
            # interpolates between our (t, v) points at whatever internal step it
            # needs. Near-Nyquist content in a 48kHz file makes that a genuinely bad
            # approximation (sharp slope-discontinuity "kinks" every sample), which
            # can stress the solver into non-convergence regardless of knob setting
            # or device model tuning -- verified: bumping device capacitances/tmax
            # only ever relocated the failure to a different knob combination,
            # never eliminated the underlying cause. Upsampling with a proper
            # bandlimited (polyphase FIR) resampler before writing the filesource
            # gives the solver a much smoother waveform to interpolate, fixing it
            # at the source. 2x was already sufficient in testing; use a modest
            # factor for margin, not the largest one that happens to work.
            from scipy.signal import resample_poly
            insig = resample_poly(insig, _input_up, 1).astype(np.float32)
            fsrc_sr = sr * _input_up
        _t = np.arange(len(insig)) / fsrc_sr
        np.savetxt(str(fsrc), np.column_stack([_t, insig]), fmt="%.8f %.6f")
        ng = {"netlist": str(netlist_path), "input": str(fsrc), "koren": args.koren,
              "ot_damp": args.ot_damp, "ot_snub": args.ot_snub, "nfb_comp": args.nfb_comp,
              "oversample": args.oversample, "conv": conv, "method": _method}
        print(f"ngspice: netlist + filesource input ready ({'Koren' if args.koren else 'DempwolfZolzer'} "
              f"tubes, method={_method}, ot_damp={args.ot_damp}, oversample={args.oversample}x + anti-alias"
              f"{', input_upsample='+str(_input_up)+'x' if _input_up > 1 else ''}"
              f"{', conv='+_conv_str if _conv_str else ''})", file=sys.stderr)

    # Effective per-knob sampled range (min/max actually seen across perms).
    # Recorded so the exported .nam declares the true trained domain per knob
    # — e.g. a tone pot swept only 0.15..0.85 must not advertise 0..1 to the
    # host, or it would send out-of-domain values the model never saw.
    knob_bounds = {k: [min(p[k] for p in perms), max(p[k] for p in perms)]
                   for k in knobs}

    # Per-knob declared default (optional). Keyed by knob name, in trained units;
    # validated against the swept range so a default can't fall outside it.
    knob_defaults = {}
    for kv in (args.defaults or "").split(","):
        if "=" not in kv:
            continue
        kname, vstr = kv.split("=", 1)
        kname = kname.strip()
        if kname not in knobs:
            ap.error(f"--defaults knob '{kname}' not in --knobs list: {knobs}")
        val = float(vstr)
        lo, hi = knob_bounds[kname]
        if not (lo - 1e-9 <= val <= hi + 1e-9):
            ap.error(f"--defaults {kname}={val} is outside its swept range [{lo}, {hi}]")
        knob_defaults[kname] = val

    (out_dir / "config.json").write_text(json.dumps({
        "backend": args.backend,
        "circuit": circuit_label,
        "schx": schx,
        "knobs": knobs,
        "steps": steps_map,
        "bounds": knob_bounds,
        "defaults": knob_defaults,
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
        # `rung` / `solver` say WHAT IT TOOK to converge this permutation. That is a property of
        # the DATA, not of the run: "high gain needed 4x oversampling" is exactly what the next
        # person needs and would otherwise have to rediscover.
        csv_w.writerow(["idx"] + knobs +
                       ["dsp_load", "proc_time", "rms", "peak", "ok", "error", "rung", "solver"])

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
                        args.oversample, args.max_crest, ng,
                        warmup_s=args.skip_warmup_s, no_retry=args.no_retry): i
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
                   [r.dsp_load, r.proc_time, f"{r.rms:.6f}", f"{r.peak:.6f}", int(r.ok), r.error,
                    r.rung, r.settings])
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

    # Is the DATASET converged? Every other check detects DIVERGENCE (the solver blew up and said
    # so). This is the only one that can see UNDER-convergence -- the solver stopping short and
    # returning a plausible, wrong waveform that passes every RMS/crest/NaN check we have.
    # Probe at the settings the DATASET WAS ACTUALLY RENDERED WITH -- rung 0 -- not at some
    # hardcoded ideal. An audit that checks different settings than the data was made with is
    # auditing nothing: it happily passed a dataset rendered at 8 iterations because it was
    # probing at 256.
    _rung0 = _rungs(args.backend, args.oversample, ng)[0]
    audit_convergence(out_dir, knobs, perms, args.backend, in_wav, schx, param_map,
                      args.fixed_params, args.speaker,
                      _rung0.get("oversample", args.oversample),
                      _rung0.get("iterations", 8), args.skip_warmup_s)

    print(f"\nPost-process with:  python batch_harness.py --combine {out_dir}")


if __name__ == "__main__":
    main()
