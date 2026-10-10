#!/usr/bin/env python3
"""oracle_check.py -- how far a finished dataset is from an INDEPENDENT renderer, recorded in its manifest.

  python oracle_check.py --dataset DIR [--oracle livespice] [--n 3] [--seconds 40] [--oracle-oversample 8] [--no-write]

A dataset's config.json carries a `renderer` block (name, version, profile, esr_vs_oracle). The first three are filled in at
generation; `esr_vs_oracle` is null until something compares the dataset with a renderer built on different numerics. This does
that: it picks `--n` combinations of the dataset (the first, the last and the middle of params.csv, so both ends of every knob
and the centre are in), re-renders the first `--seconds` of the same input through the oracle with the same knob values and the
same capture chain, undoes the dataset's output scaling, and compares from 1 s on (the same warm-up the dataset's checks skip).
ESR is reported per combination, after the best single gain (a pure level difference is a different finding from a shape
difference), and **after a cabinet-like low-pass** (`--cab-hz`, a 4th-order Butterworth, 5 kHz by default). The full-band figure
is dominated by content above 6 kHz on a hot, swept excitation (on the Deluxe 84 % of the difference was above 6.4 kHz) that a guitar
cabinet removes, so the figure to read against an audibility threshold is the cabinet one. The result is written into `renderer.esr_vs_oracle` of config.json:

  {"oracle": "livespice", "oracle_version": ..., "oracle_oversample": 8, "n": 3, "seconds": 40, "esr_median": ..., "esr_max": ...,
   "esr_gain_fit_median": ..., "esr_cabinet_median": ..., "esr_cabinet_max": ..., "cabinet_hz": 5000, "gain_median": ...,
   "combinations": [idx, ...], "date": ...}

Oracles: `livespice` (livespice-cli). A dataset rendered by livespice compared with livespice would measure nothing, so that is
refused. What the number means: the ESR between two renderers' answers for the same circuit, knobs and input -- for the acm
backend against livespice, mostly the difference between the two resamplers (livespice averages each output period, a boxcar with
high-frequency droop and weak alias rejection; the acm renderer uses a linear-phase FIR) plus the two truncation errors, which is
the figure to read against the audibility threshold the dataset is generated for.
"""
import argparse
import csv
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def pick_rows(rows: list, n: int) -> list:
    """Indices into `rows` of up to n combinations: first, last, then evenly between (centre first)."""
    m = len(rows)
    if m <= n:
        return list(range(m))
    want = [0, m - 1, m // 2] + [int(round(i * (m - 1) / (n - 1))) for i in range(1, n - 1)]
    out = []
    for i in want:
        if i not in out:
            out.append(i)
    return sorted(out[:n])


def esr_pair(a: np.ndarray, b: np.ndarray, skip: int) -> "tuple[float, float, float]":
    """(ESR, ESR after the best single gain, that gain) of a against reference b from sample `skip` on."""
    m = min(len(a), len(b))
    a, b = a[skip:m].astype(np.float64), b[skip:m].astype(np.float64)
    den = float((b ** 2).sum()) + 1e-30
    esr = float(((a - b) ** 2).sum()) / den
    g = float(a @ b) / (float(a @ a) + 1e-30)
    return esr, float(((g * a - b) ** 2).sum()) / den, g


def cabinet_lowpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """A cabinet-like roll-off: 4th-order Butterworth low-pass. Generic on purpose (no speaker IR): it only has to remove what a guitar cabinet removes."""
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, hz, "low", fs=sr, output="sos"), y)


def oracle_identity(oracle: str) -> str:
    from prepare_excitation import solver_identity
    return solver_identity(oracle)


def render_oracle(oracle: str, cfg: dict, in_wav: Path, params: dict, out_wav: Path, oversample: int) -> None:
    from gen_dataset_from_schx import LIVESPICE_CLI, fmt_params
    if oracle != "livespice":
        raise SystemExit(f"unknown oracle {oracle!r} (livespice)")
    swept = fmt_params(params, cfg.get("param_map"))
    fixed = cfg.get("fixed_params")
    allp = f"{fixed},{swept}" if fixed else swept
    cmd = [str(LIVESPICE_CLI), "--input", str(in_wav), "--output", str(out_wav), "--circuit", cfg["schx"], "--params", allp,
           "--oversample", str(oversample), "--iterations", "256"]
    if cfg.get("speaker"):
        cmd += ["--speaker", cfg["speaker"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_wav.exists():
        raise SystemExit(f"the oracle render failed: {(r.stderr or '').strip().splitlines()[-1:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--oracle", default="livespice", choices=["livespice"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--oracle-oversample", type=int, default=8)
    ap.add_argument("--cab-hz", type=float, default=5000.0, help="corner of the cabinet-like low-pass for the audible-band ESR (default 5000)")
    ap.add_argument("--oracle-lead-in", type=float, default=8.0, metavar="S",
                    help="seconds of silence run through the oracle before the input (then dropped): a circuit with a supply that sags or a "
                         "mains source needs several seconds to settle, and an oracle that starts cold would be scored on its own start-up "
                         "(default 8; the dataset's renders have their own lead-in). 0 = start cold")
    ap.add_argument("--no-write", action="store_true", help="print the result, leave config.json alone")
    args = ap.parse_args()

    cfg_path = args.dataset / "config.json"
    cfg = json.loads(cfg_path.read_text())
    backend = cfg.get("backend")
    if backend == args.oracle:
        raise SystemExit(f"this dataset was rendered by {backend}: comparing it with the same renderer measures nothing")
    if not cfg.get("schx") or not Path(cfg["schx"]).exists():
        raise SystemExit(f"the dataset's schematic is not available: {cfg.get('schx')!r}")
    rows = list(csv.DictReader(open(args.dataset / "params.csv")))
    ok_rows = [r for r in rows if str(r.get("ok", "1")) in ("1", "True", "true")]
    outputs = np.load(args.dataset / "outputs.npy", mmap_mode="r")
    in_wav = Path(cfg["input_wav"])
    if not in_wav.exists():
        in_wav = args.dataset / "sweep.wav"
        print(f"note: the recorded input {cfg['input_wav']} is gone; using the dataset's sweep.wav")
    x, sr = sf.read(str(in_wav), dtype="float32")
    x = x if x.ndim == 1 else x[:, 0]
    n_samples = min(len(x), int(args.seconds * sr))
    scale = float(cfg.get("output_scale") or 1.0)
    chain = cfg.get("capture_chain")
    results, idxs = [], []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clip = td / "clip.wav"
        lead = int(max(0.0, args.oracle_lead_in) * sr)
        lead -= lead % int(round(sr / 10))   # a whole number of tenths of a second: 50 and 60 Hz mains both come back to their starting phase
        sf.write(str(clip), np.concatenate([np.zeros(lead, dtype="float32"), x[:n_samples]]), sr, subtype="FLOAT")
        for k in pick_rows(ok_rows, args.n):
            r = ok_rows[k]
            params = {name: float(r[name]) for name in cfg["knobs"]}
            out = td / f"o{k}.wav"
            render_oracle(args.oracle, cfg, clip, params, out, args.oracle_oversample)
            y, _ = sf.read(str(out), dtype="float64")
            y = y if y.ndim == 1 else y[:, 0]
            y = y[lead:]
            if chain:
                from capture_chain import capture_chain as _chain
                y = _chain(y, sr, corner_hz=chain["corner_hz"], order=chain["order"])
            ds = np.asarray(outputs[int(r["idx"])][: len(y)], dtype=np.float64) / scale
            e, eg, g = esr_pair(ds, y, int(1.0 * sr))
            ec = esr_pair(cabinet_lowpass(ds, sr, args.cab_hz), cabinet_lowpass(y, sr, args.cab_hz), int(1.0 * sr))[0]
            results.append((e, eg, g, ec)); idxs.append(int(r["idx"]))
            label = ", ".join(f"{kname}={params[kname]:g}" for kname in cfg["knobs"])
            print(f"  idx {int(r['idx']):<4} {label:<40} ESR {e:.3e}   after best gain {eg:.3e} (gain {g:.4f})   after {args.cab_hz:g} Hz cabinet low-pass {ec:.3e}")
    es = [r[0] for r in results]
    rec = {"oracle": args.oracle, "oracle_version": oracle_identity(args.oracle), "oracle_oversample": args.oracle_oversample,
           "n": len(results), "seconds": round(n_samples / sr, 2), "oracle_lead_in": round(lead / sr, 2), "esr_median": float(np.median(es)), "esr_max": float(np.max(es)),
           "esr_gain_fit_median": float(np.median([r[1] for r in results])),
           "esr_cabinet_median": float(np.median([r[3] for r in results])), "esr_cabinet_max": float(np.max([r[3] for r in results])), "cabinet_hz": args.cab_hz, "gain_median": float(np.median([r[2] for r in results])),
           "combinations": idxs, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    print(f"\n{backend} vs {args.oracle}: ESR median {rec['esr_median']:.3e}, max {rec['esr_max']:.3e}; after the {args.cab_hz:g} Hz cabinet low-pass "
          f"median {rec['esr_cabinet_median']:.3e}, max {rec['esr_cabinet_max']:.3e}")
    if not args.no_write:
        cfg.setdefault("renderer", {})["esr_vs_oracle"] = rec
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"recorded in {cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
