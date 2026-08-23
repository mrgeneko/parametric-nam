#!/usr/bin/env python3
"""Render-backend adapters for preflight.py / find_saturation_point.py.

WHY THIS EXISTS: preflight.py (LiveSPICE/livespice_cli) and the ngspice hand-deck family
(gen_ocd_ngspice.py and siblings, rendered via ngspice_spicelib.py) used to each carry their
OWN full copy of the knob-check/saturation-sweep logic (~300 lines apiece), differing only in
how a single render actually happens. That's the exact duplication that already caused one
divergence (amps/preflight_jc120.py in the private devices repo silently drifting from
preflight.py) and would only get worse with a third backend (e.g. LTspice). This module is the
one place that difference lives, so preflight.py and find_saturation_point.py can be
genuinely backend-agnostic instead of backend-specific-and-copy-pasted.

CONTRACT a backend implements (two methods, nothing else):

  backend.prepare_input(raw, sr, level_v, scratch, tag) -> input_handle
    Scale `raw` (a mono float array at whatever its own native peak is) so its peak hits
    `level_v` volts, and do whatever one-time setup the backend needs to render against it
    repeatedly (LiveSpice: just write a WAV file, return its path; ngspice: also build the
    XSPICE filesource, return the (sr, t, input_src) tuple ngspice_spicelib.py's render_grid
    needs). Called once per distinct level; the returned handle is reused across many
    knob-param renders at that level.

  backend.render_many(jobs, input_handle, scratch) -> {tag: np.ndarray | None}
    jobs: [{"params": {knob: val, ...}, "tag": str}, ...], all rendered against the SAME
    input_handle (parallelized however suits the backend). Returns raw voltage-scale mono
    audio per tag (at the backend's own samplerate), or None for a render that didn't
    converge.

Adding a third backend (e.g. LTspice) means writing one class implementing these two methods,
not another ~300-line copy of preflight.py's checks.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gen_dataset_from_schx import LIVESPICE_CLI  # noqa: E402

from tools.ngspice_spicelib import load_input, render_grid  # noqa: E402
from scipy.io import wavfile  # noqa: E402


class LiveSpiceBackend:
    """Renders via the compiled livespice_cli binary (one subprocess per render)."""

    def __init__(self, schx, oversample=8, iterations=256, workers=None):
        self.schx = schx
        self.oversample = oversample
        self.iterations = iterations
        self.workers = workers

    def prepare_input(self, raw, sr, level_v, scratch, tag):
        peak = float(np.abs(raw).max()) + 1e-12
        scaled = (raw / peak * level_v).astype(np.float32)
        path = f"{scratch}/input_{tag}.wav"
        sf.write(path, scaled, sr, subtype="FLOAT")  # FLOAT: values >1.0 (>1V drive) must survive
        return path

    def _render_one(self, params, in_wav, scratch, tag):
        out = f"{scratch}/pf_{tag}.wav"
        kv = ",".join(f"{k}={v}" for k, v in params.items())
        r = subprocess.run([str(LIVESPICE_CLI), "--input", in_wav, "--output", out,
                             "--circuit", self.schx, "--params", kv,
                             "--oversample", str(self.oversample), "--iterations", str(self.iterations)],
                            capture_output=True, text=True)
        try:
            y, _ = sf.read(out, dtype="float32")
            return y[:, 0] if y.ndim > 1 else y
        except Exception:
            sys.stderr.write(r.stderr[-400:] + "\n")
            return None

    def render_many(self, jobs, input_handle, scratch):
        if not jobs:
            return {}
        workers = max(1, min(self.workers or os.cpu_count() or 4, len(jobs)))
        out = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._render_one, j["params"], input_handle, scratch, j["tag"]): j["tag"]
                    for j in jobs}
            for f in futs:
                out[futs[f]] = f.result()
        return out


class NgspiceBackend:
    """Renders a hand-written ngspice deck via ngspice_spicelib.py's render_grid (spicelib
    driving ngspice with SimRunner/RawRead), for devices whose deck has no .schx counterpart
    at all (typically a real MOSFET, or a feedback loop LiveSPICE's fixed-timestep solver
    can't hold)."""

    def __init__(self, build_deck, probe_node="OUT", maxstep=3e-6, parallel_sims=8):
        self.build_deck = build_deck
        self.probe_node = probe_node
        self.maxstep = maxstep
        self.parallel_sims = parallel_sims

    def prepare_input(self, raw, sr, level_v, scratch, tag):
        wav_path = f"{scratch}/rawinput_{tag}.wav"
        wavfile.write(wav_path, sr, raw.astype(np.float32))
        return load_input(wav_path, level_v, scratch, src_name=f"input_{tag}.src")  # (sr, t, input_src)

    def render_many(self, jobs, input_handle, scratch):
        if not jobs:
            return {}
        sr_, t_, input_src_ = input_handle
        outfiles = {j["tag"]: f"{scratch}/pf_{j['tag']}.wav" for j in jobs}
        rg_jobs = [(j["params"], outfiles[j["tag"]]) for j in jobs]
        peaks = render_grid(self.build_deck, rg_jobs, self.probe_node, sr_, t_, input_src_, scratch,
                             maxstep=self.maxstep, parallel_sims=self.parallel_sims)
        out = {}
        for j in jobs:
            pk = peaks[outfiles[j["tag"]]]
            if pk is None:
                out[j["tag"]] = None
            else:
                _, y16 = wavfile.read(outfiles[j["tag"]])
                # undo render_grid's peak-normalized int16 write -> raw voltage-scale float
                out[j["tag"]] = y16.astype(np.float64) / (0.9 * 32767.0) * pk
        return out
