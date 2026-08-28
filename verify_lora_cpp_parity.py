#!/usr/bin/env python3
"""Cross-repo numeric parity check (LoRA plan Phase 5).

Exports a tiny LoRA-enabled ParametricA2 model, renders it through the REAL C++ inference
path (NeuralAmpModelerCore's render_parametric tool, which -- since Phase 4's fast-path
exclusion gate -- takes the generic, LoRA-aware path for any "film+lora"-tagged model), and
diffs the result against this SAME model's own Python forward() at several knob settings.
This is the check that actually proves Python and C++ agree bit-for-bit on the LoRA math,
not just that each side's own test suite passes in isolation.

Two known C++-vs-Python behavioral differences (both documented in render_parametric.cpp's
own header comment) are compensated for here, not in the C++ tool:
  * KNOB SMOOTHING. C++ one-pole smooths knobs from a 0.5 default toward the target, over
    real audio. Worked around by rendering the ENTIRE input in a single --block call sized
    to the whole file: the one-pole update happens once per process() call, so a single
    call whose block size is many multiples of the ~20ms smoothing time constant converges
    to (numerically indistinguishable from) the target before any sample of that call is
    produced -- no ramp to discard.
  * DC BLOCKER. The parametric C++ model applies a ~20 Hz IIR high-pass after the net
    (FiLM's shift terms inject DC); Python's forward() does not. Applied here to Python's
    output with the identical formula/coefficient parametric_wavenet.cpp uses. Because C++
    prewarms (converges) that blocker at the 0.5 DEFAULT knob before SetKnobValues() sets
    the real target, there's still a brief resettling transient at the start of the real
    audio -- the head is discarded (--discard-head-sec) to avoid comparing across it.

Usage:
    .venv/bin/python verify_lora_cpp_parity.py

Requires NeuralAmpModelerCore's run_tests target already built (produces render_parametric):
    cd ~/work/chainsmith/NeuralAmpModelerCore/build && cmake --build . --target run_tests
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from param_train import ParametricA2

KNOB_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def dc_alpha(sample_rate: float) -> float:
    """Must match ParametricWaveNet::Reset()'s _dc_alpha formula exactly (parametric_wavenet.cpp)."""
    return 1.0 - (2.0 * np.pi * 20.0 / sample_rate)


def dc_highpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """y[n] = x[n] - x[n-1] + alpha*y[n-1] -- must match ParametricWaveNet::process()'s DC
    blocker exactly, since Python's own forward() applies no such filter at all."""
    y = np.empty_like(x)
    x_prev = 0.0
    y_prev = 0.0
    for i in range(len(x)):
        yi = x[i] - x_prev + alpha * y_prev
        y[i] = yi
        x_prev = x[i]
        y_prev = yi
    return y


def find_render_parametric(nam_core: Path):
    """Returns (render_bin, input_wav) if both exist, else (None, None) -- the shared
    availability check both main() and the pytest wrapper (tests/test_lora_cpp_parity.py)
    use to decide whether this cross-repo check can run at all in the current environment."""
    render_bin = nam_core / "build" / "tools" / "render_parametric"
    input_wav = nam_core / "example_audio" / "input.wav"
    if render_bin.exists() and input_wav.exists():
        return render_bin, input_wav
    return None, None


def run_parity_check(nam_core: Path, rank: int, channels: int = 3, tol: float = 5e-5,
                     discard_head_sec: float = 0.5, verbose: bool = False) -> float:
    """Exports a tiny (possibly LoRA-enabled) ParametricA2, renders it through the real C++
    render_parametric binary at each of KNOB_VALUES, diffs against this same model's own
    Python forward() (DC-highpassed to match the C++ side -- see module docstring), and
    returns the worst max_diff observed. Raises AssertionError if any knob setting exceeds
    `tol`, or if render_parametric/its fixture input aren't present at `nam_core`."""
    render_bin, input_wav = find_render_parametric(nam_core)
    if render_bin is None:
        raise AssertionError(
            f"render_parametric not found under {nam_core} -- build it first: "
            f"cd {nam_core}/build && cmake --build . --target run_tests")

    audio, sr = sf.read(str(input_wav), dtype="float32")
    assert sr == 48000, f"expected 48kHz input, got {sr}"

    torch.manual_seed(0)
    model = ParametricA2(channels=channels, num_params=1, lora_rank=rank).eval()
    # Randomize (init is near-identity FiLM / exactly-zero LoRA) so LoRA actually does
    # something distinguishable from FiLM-only -- mirrors tests/test_lora.py's own pattern.
    for p in model.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p.data)

    config = {"param_names": ["drive"]}
    nam = model.export_nam(config, {"version": "0.7.0"}, sample_rate=48000, input_audio=None)
    ptype = nam["config"]["parametric"]["type"]
    expected_type = "film+lora" if rank > 0 else "film"
    assert ptype == expected_type, f"expected type={expected_type!r}, got {ptype!r}"
    if verbose:
        lora_info = nam["config"]["parametric"].get("lora", {})
        print(f"exported: type={ptype} schema_version={nam['config']['parametric']['schema_version']} "
              f"lora_rank={lora_info.get('rank', 0)} channels={channels} "
              f"weights={len(nam['weights'])} "
              f"(expect {'GENERIC' if rank > 0 else 'FAST'} C++ path per Phase 4's gate)")

    alpha = dc_alpha(sr)
    discard = int(discard_head_sec * sr)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        nam_path = tmp / "model.param.nam"
        nam_path.write_text(json.dumps(nam))

        worst = 0.0
        for knob in KNOB_VALUES:
            out_wav = tmp / f"out_{knob}.wav"
            # --block sized to the WHOLE file: see module docstring on why this eliminates
            # the knob-smoothing ramp entirely rather than requiring a head discard for it.
            subprocess.run(
                [str(render_bin), str(nam_path), str(input_wav), str(out_wav),
                 "--knobs", str(knob), "--block", str(len(audio))],
                check=True, capture_output=True, text=True)
            cpp_out, cpp_sr = sf.read(str(out_wav), dtype="float32")
            assert cpp_sr == sr

            with torch.no_grad():
                inp = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
                params = torch.full((1, 1), float(knob))
                py_out = model(inp, params).squeeze().numpy()
            py_out_hp = dc_highpass(py_out, alpha)

            n = min(len(cpp_out), len(py_out_hp)) - discard
            a = cpp_out[discard:discard + n]
            b = py_out_hp[discard:discard + n]
            max_diff = float(np.max(np.abs(a - b)))
            worst = max(worst, max_diff)
            if verbose:
                status = "OK" if max_diff < tol else "FAIL"
                print(f"  knob={knob:.2f}  max_diff={max_diff:.3e}  [{status}]")

        assert worst < tol, f"worst max_diff {worst:.3e} exceeds tolerance {tol:.0e}"
        return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nam-core", type=Path, default=Path.home() / "work/chainsmith/NeuralAmpModelerCore")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--channels", type=int, default=3, help="A2 nano -- matches render_parametric's own fixture width")
    ap.add_argument("--tol", type=float, default=5e-5)
    ap.add_argument("--discard-head-sec", type=float, default=0.5)
    args = ap.parse_args()

    try:
        worst = run_parity_check(args.nam_core, args.rank, args.channels, args.tol,
                                 args.discard_head_sec, verbose=True)
    except AssertionError as e:
        sys.exit(str(e))
    print(f"worst max_diff across all knob settings: {worst:.3e} (tol {args.tol:.0e})")
    print("PASS -- Python and C++ agree within tolerance at every knob setting.")


if __name__ == "__main__":
    main()
