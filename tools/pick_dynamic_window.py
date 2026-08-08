#!/usr/bin/env python3
"""Cut the most-dynamic N-second window out of a long real-playing clip.

Formalizes a pattern that had been hand-rolled ad hoc (a throwaway "python -" one-liner,
never committed) at least three times in this codebase's history -- the Tweed 5F6-A 17s
window, the JCM800 95s window, and the Dumble 95s window were each picked this way, with
the exact selection logic re-derived from scratch each time and left undocumented beyond a
config-comment sentence like "picked by sliding a 95s window and taking the highest
peak/rms".

WHY RMS, NOT PEAK OR CREST: build_excitation.py's own docstring notes that a high-crest
real-playing clip "samples its own loud region essentially never" -- a single loud transient
in an otherwise-quiet clip has high peak and high crest but is exactly the WRONG kind of
window (little sustained loud content). RMS (energy density) favors windows with genuinely
more loud/dynamic playing throughout, which is what "most dynamic" has meant in practice
across the prior ad hoc uses. Peak is still reported for the chosen window so it's auditable
either way; --metric lets you rank by peak or peak*rms instead if a future case wants that.

Writes a <output-stem>.recipe.json sidecar (same convention as build_excitation.py) so
gen_dataset_from_schx.py's input_provenance() picks it up automatically when this window is later
used (directly, or as build_excitation.py's own --input) as a training config's excitation.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000


def _audio_provenance(path, x=None):
    """Duplicated from build_excitation.py deliberately -- same convention, independently
    verifiable, no cross-tool import dependency. See that file's identical helper."""
    if x is None:
        x, _ = sf.read(str(path), dtype="float32")
    mono = x if x.ndim == 1 else x.mean(axis=1)
    mono = np.asarray(mono, dtype=np.float32)
    return {
        "name": Path(path).name,
        "path": str(path),
        "audio_sha1": hashlib.sha1(mono.tobytes()).hexdigest(),
        "samplerate": SR,
        "frames": int(len(mono)),
        "duration_s": round(len(mono) / SR, 3),
    }


def _tool_git_rev():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                            capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _fade(y, ms_in, ms_out):
    y = y.copy()
    ni, no = int(SR * ms_in / 1000), int(SR * ms_out / 1000)
    if ni > 0: y[:ni] *= np.sin(np.linspace(0, np.pi / 2, ni)) ** 2
    if no > 0: y[-no:] *= np.sin(np.linspace(np.pi / 2, 0, no)) ** 2
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dur", type=float, required=True, help="window length, seconds")
    ap.add_argument("--hop", type=float, default=0.5, help="candidate-window stride, seconds")
    ap.add_argument("--metric", choices=["rms", "peak", "peak_rms"], default="rms",
                    help="ranking metric for 'most dynamic' -- see module docstring for why "
                         "rms is the default")
    ap.add_argument("--fade-ms", type=float, default=10.0,
                    help="fade in/out at the cut boundaries (avoids a click; we're cutting "
                         "from mid-file, not from silence)")
    args = ap.parse_args()

    x, sr = sf.read(args.input, dtype="float32")
    if x.ndim > 1: x = x[:, 0]
    if sr != SR: raise SystemExit(f"input sr {sr} != {SR}")

    win = int(round(args.dur * SR))
    hop = max(1, int(round(args.hop * SR)))
    if win >= len(x):
        raise SystemExit(f"--dur {args.dur}s >= input length {len(x)/SR:.1f}s -- nothing to window")

    starts = list(range(0, len(x) - win + 1, hop))
    best_i, best_score, best_peak, best_rms = None, -1.0, 0.0, 0.0
    for s in starts:
        seg = x[s:s + win]
        peak = float(np.abs(seg).max())
        rms = float(np.sqrt((seg.astype(np.float64) ** 2).mean()))
        score = {"rms": rms, "peak": peak, "peak_rms": peak * rms}[args.metric]
        if score > best_score:
            best_i, best_score, best_peak, best_rms = s, score, peak, rms

    chosen = _fade(x[best_i:best_i + win].copy(), args.fade_ms, args.fade_ms)
    sf.write(args.output, chosen, SR, subtype="FLOAT")
    crest_db = 20 * np.log10(best_peak / (best_rms + 1e-12))
    print(f"wrote {args.output}  dur {win/SR:.1f}s  start {best_i/SR:.1f}s/{len(x)/SR:.1f}s"
          f"  peak {best_peak:.3f}  rms {best_rms:.4f}  crest {crest_db:.1f}dB"
          f"  (ranked by {args.metric}, {len(starts)} candidates @ {args.hop}s hop)")

    recipe = {
        "tool": "pick_dynamic_window.py",
        "tool_git_rev": _tool_git_rev(),
        "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "args": {
            "dur": args.dur, "hop": args.hop, "metric": args.metric, "fade_ms": args.fade_ms,
        },
        "selection": {
            "window_start_s": round(best_i / SR, 3),
            "candidates_evaluated": len(starts),
            "score": round(best_score, 6),
        },
        "source": _audio_provenance(args.input, x=x),
        "output": {**_audio_provenance(args.output, x=chosen),
                   "peak": round(best_peak, 6), "rms": round(best_rms, 6),
                   "crest_db": round(float(crest_db), 2)},
    }
    recipe_path = Path(args.output).with_suffix(".recipe.json")
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"wrote {recipe_path}  (build recipe -- picked up automatically by gen_dataset_from_schx.py)")


if __name__ == "__main__":
    main()
