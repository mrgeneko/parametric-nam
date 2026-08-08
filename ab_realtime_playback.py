#!/usr/bin/env python3
"""ab_realtime_playback.py — A/B our Python forward against the host app's C++ playback.

This is the FiLM-on parity check (internal engineering notes §4.2/4.3): the
last unverified link between how we TRAIN a parametric model and how the product PLAYS it.
The static round-trip test (tests/test_nam_standard.py) folds FiLM away, so it does NOT
cover conditioning; this does.

  # 1. render the C++ side (in the host app's core repo)
  ./build/tools/render_parametric model.param.nam in.wav cpp.wav --knobs 0.75,0.30

  # 2. compare against our Python forward at the same knobs
  python ab_realtime_playback.py --model model.param.nam --input in.wav --cpp cpp.wav \
      --params "Sustain=0.75,Tone=0.30"

Three things MUST be reconciled or the comparison falsely mismatches:

  * DC BLOCKER — the C++ parametric models apply a ~20 Hz IIR high-pass after the net
    (FiLM's shift terms inject a large DC offset). Our Python forward does not, so we apply
    the identical filter to the Python output here.
  * KNOB SMOOTHING — the C++ one-pole smooths knobs from their 0.5 default toward the
    target, so the first ~0.1 s is a ramp, not the steady-state tone. We discard a warmup
    region (also covering the ~6347-sample receptive field).
  * HEAD LATENCY — our head uses centered padding (k//2) while a2_fast/NAM are causal, a
    fixed ~7-sample offset. We lag-search and report the alignment we found.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from param_train import ParametricA2, check_parametric_schema


def load_parametric(nam_path: Path, width: int | None = None):
    """Rebuild the model from a .param.nam, honoring its declared head_mode."""
    d = json.loads(nam_path.read_text())
    entries = ([s["model"] for s in d["config"]["submodels"]]
               if d.get("architecture") == "SlimmableContainer" else [d])
    if width is not None:
        entries = [m for m in entries if int(m["config"]["layers"]) == width] or entries
    m = entries[-1]                       # widest tier by default
    cfg = m["config"]
    par = cfg["parametric"]
    check_parametric_schema(par, source=str(nam_path))
    head_mode = par.get("head_mode", "residual")
    names = [p["name"] for p in par["parameters"]]
    model = ParametricA2(int(cfg["layers"]), len(names), head_mode=head_mode)
    model.load_weights(m["weights"])
    model.eval()
    return model, names, head_mode


def dc_block(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """The C++ parametric DC blocker, replicated exactly:
       y[n] = x[n] - x[n-1] + alpha*y[n-1],  alpha = 1 - 2*pi*20/sr   (fc ~= 20 Hz)."""
    alpha = 1.0 - (2.0 * np.pi * 20.0 / sample_rate)
    y = np.zeros_like(x)
    xp = yp = 0.0
    for n in range(len(x)):
        yn = x[n] - xp + alpha * yp
        xp, yp = x[n], yn
        y[n] = yn
    return y


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 64):
    """(corr, lag) maximizing Pearson correlation over integer lags."""
    best = (-1.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        aa, bb = (a[: len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: len(b) + lag])
        n = min(len(aa), len(bb))
        if n < 1000:
            continue
        c = np.corrcoef(aa[:n], bb[:n])[0, 1]
        if c > best[0]:
            best = (c, lag)
    return best


def esr(ref: np.ndarray, est: np.ndarray) -> float:
    """Error-to-signal ratio, the metric we train against."""
    return float(np.sum((ref - est) ** 2) / max(np.sum(ref ** 2), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="the .param.nam both sides ran")
    ap.add_argument("--input", required=True, type=Path, help="input wav fed to both sides")
    ap.add_argument("--cpp", required=True, type=Path, help="wav rendered by render_parametric")
    ap.add_argument("--params", required=True, help="knob=value,... (must match the C++ render)")
    ap.add_argument("--width", type=int, default=None, help="tier width (default: widest)")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds to discard (knob smoothing + receptive field); default 1.0")
    ap.add_argument("--corr-min", type=float, default=0.999,
                    help="fail below this correlation (default 0.999)")
    ap.add_argument("--no-dc-block", action="store_true",
                    help="skip DC-blocking the Python side (diagnostic: shows the DC mismatch)")
    a = ap.parse_args()

    model, names, head_mode = load_parametric(a.model, a.width)
    kv = dict(x.split("=", 1) for x in a.params.split(",") if "=" in x)
    unknown = [k for k in kv if k not in names]
    if unknown:
        ap.error(f"unknown knob(s) {unknown}; model knobs are {names}")
    vec = [float(kv[n]) for n in names]

    x, sr = sf.read(str(a.input), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    cpp, sr_cpp = sf.read(str(a.cpp), dtype="float32")
    if cpp.ndim > 1:
        cpp = cpp[:, 0]
    if abs(sr - sr_cpp) > 0.5:
        ap.error(f"sample-rate mismatch: input {sr} vs cpp {sr_cpp}")

    print(f"model     : {a.model.name}  head_mode={head_mode}  knobs={dict(zip(names, vec))}")

    # Python forward (chunked to keep memory sane).
    with torch.no_grad():
        xt = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
        pt = torch.tensor([vec], dtype=torch.float32)
        py = model(xt, pt).squeeze().numpy()

    # Reconcile the C++ signal path: it DC-blocks after the net; we must too.
    if not a.no_dc_block:
        py = dc_block(py, sr)

    n = min(len(py), len(cpp))
    skip = int(a.warmup * sr)
    if skip >= n:
        ap.error("--warmup exceeds the signal length")
    py_t, cpp_t = py[skip:n], cpp[skip:n]

    corr, lag = best_lag(py_t, cpp_t)
    # Re-align and score at the winning lag.
    if lag >= 0:
        pa, ca = py_t[: len(py_t) - lag], cpp_t[lag:]
    else:
        pa, ca = py_t[-lag:], cpp_t[: len(cpp_t) + lag]
    m = min(len(pa), len(ca))
    pa, ca = pa[:m], ca[:m]

    e = esr(ca, pa)
    max_abs = float(np.max(np.abs(ca - pa)))
    rms_ratio = float(np.sqrt(np.mean(pa ** 2)) / max(np.sqrt(np.mean(ca ** 2)), 1e-12))

    print(f"warmup    : discarded {a.warmup:.2f}s (knob smoothing + receptive field)")
    print(f"alignment : lag={lag} samples (centered vs causal head)")
    print(f"corr      : {corr:.6f}")
    print(f"ESR       : {e:.3e}")
    print(f"max|diff| : {max_abs:.3e}   rms(py)/rms(cpp) = {rms_ratio:.4f}")

    ok = corr >= a.corr_min
    print("\n" + ("PASS — Python forward matches the host app's C++ playback with FiLM active."
                  if ok else
                  f"FAIL — correlation {corr:.6f} < {a.corr_min}. Training and playback DISAGREE; "
                  "do not retrain/ship until this is resolved."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
