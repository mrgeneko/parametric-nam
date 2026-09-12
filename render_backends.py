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
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_dataset_from_schx import LIVESPICE_CLI, _run_ngspice  # noqa: E402

from ngspice_spicelib import load_input, render_grid  # noqa: E402
import ltspice_spicelib  # noqa: E402
from scipy.io import wavfile  # noqa: E402


def parse_conv(s):
    """"key=val,key2=val2,..." -> dict, the same string format gen_dataset_from_schx.py's
    own --conv has always used. Shared here (not duplicated per-caller) so every tool that
    can render through NgspiceSchxBackend (prepare_excitation.py, preflight.py,
    check_transient_coverage.py, grid_adequacy.py, gen_dataset_from_schx.py, run_pipeline.py)
    parses the same override string identically. Values stay strings -- schx_to_ngspice.py's
    own qty() does the unit conversion at render time (and its own module docstring is the
    unit-suffix authority: uppercase M is MEGA, lowercase m is milli -- the OPPOSITE of
    PSpice's own convention -- so a value copied verbatim from a PSpice .MODEL card must be
    converted to this convention, or written as plain scientific notation, before it reaches
    here)."""
    return dict(kv.split("=", 1) for kv in (s or "").split(",") if "=" in kv)


def conv_cache_tag(conv):
    """Cache-key fragment identifying a device-model override, the same role capture_chain.
    cache_tag() plays for the capture chain: without this, an onset/probe measured under one
    --conv (e.g. a corrected transistor fit) could be served back to a caller expecting the
    generic default, or vice versa."""
    if not conv:
        return ""
    return "|conv=" + ",".join(f"{k}={conv[k]}" for k in sorted(conv))


def describe_subprocess_failure(r: subprocess.CompletedProcess) -> str:
    """One-line, ACTIONABLE description of why a livespice_cli render failed.

    Two call sites (this module's own LiveSpiceBackend and grid_adequacy.py's separate
    inline livespice invocation) used to each just print/group by the LAST LINE of stderr.
    For a .NET unhandled exception that line is `at Foo.Bar() in .../Program.cs:line N` --
    the least informative part of the trace (a stack frame, not the exception itself, which
    is on an earlier line starting "Unhandled exception. System.XxxException: ..."). Worse,
    every crash of that shape shares the same last line regardless of root cause, so
    grid_adequacy's failure-grouping-by-message bucketed genuinely different problems as one.

    Also distinguishes a NEGATIVE returncode (Python's convention for "killed by signal N")
    from a normal nonzero exit. A signal kill -- SIGKILL/-9 especially -- on a healthy render
    command is the signature of the OS (or macOS's Jetsam) killing the process for memory
    pressure, not an application bug; conflating the two sends whoever's debugging looking
    for a code defect that isn't there. Found 2026-09-07: a 9-tube Soldano SLO-100 build
    failed 242/675 grid_adequacy probes at the default 8 parallel workers on a machine
    already swapping heavily (vm_stat: ~58 MB free, tens of millions of swap-ins/outs) from
    a concurrent training run -- dropping to --workers 2 made the failures disappear, which a
    stack-trace-tail message never would have pointed at.
    """
    # getattr, not r.returncode directly: some callers only have a partial mock/result object
    # (e.g. an ngspice failure path with no subprocess involved at all) that carries `.stderr`
    # but not `.returncode` -- that's still a real failure worth describing, just not one this
    # function can classify as a signal kill vs. a normal nonzero exit.
    rc = getattr(r, "returncode", None)
    if rc is not None and rc < 0:
        try:
            sig = signal.Signals(-rc).name
        except ValueError:
            sig = str(-rc)
        return (f"KILLED BY SIGNAL {sig} (rc={rc}) -- not an application error; "
                f"this is the OS terminating the process, almost always memory pressure "
                f"under parallel load (check `vm_stat` free pages / swap activity, and "
                f"whether something else is training/rendering concurrently) rather than a "
                f"circuit or solver bug. Lower --workers before assuming the circuit is broken.")
    lines = [ln for ln in (getattr(r, "stderr", "") or "").strip().splitlines() if ln.strip()]
    reason = "(no stderr)"
    if lines:
        exc_line = next((ln for ln in lines
                          if "Exception" in ln or ln.lower().startswith("unhandled")), None)
        reason = exc_line or lines[-1]
    if rc is None:
        return f"rc=unknown: {reason}"
    if rc == 0:
        return f"rc=0 (process exited cleanly) but produced no readable output: {reason}"
    return f"rc={rc}: {reason}"


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
            sys.stderr.write(f"[{tag}] {describe_subprocess_failure(r)}\n")
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


class NgspiceSchxBackend:
    """Renders a .schx circuit through the GENERIC schx-to-ngspice translation path
    (ngspice/schx_to_ngspice.py), for a circuit LiveSPICE's fixed-timestep solver cannot hold
    under real signal (typically a tight, DC-coupled feedback loop -- see e.g. the Arbiter Fuzz
    Face and BD-2/MT-2 docs) but that needs no hand-written deck at all, unlike NgspiceBackend/
    LtspiceBackend above (which exist for a device with NO .schx counterpart, e.g. a real
    MOSFET/BJT LiveSPICE has no model for).

    This is the exact machinery grid_adequacy.py's own --backend "ngspice" already used
    inline (one netlist dump via LIVESPICE_CLI at construction, then gen_dataset_from_schx.
    _run_ngspice per render) -- factored out here so prepare_excitation.py/preflight.py/
    check_transient_coverage.py can share it too, instead of each needing their own copy or,
    worse, silently falling back to LiveSpiceBackend for a circuit LiveSpice cannot render at
    all (see check_transient_coverage.py's check_coverage(), which was exactly that gap).
    """

    def __init__(self, schx, oversample=2, fixed_params=None, param_map=None, conv=None):
        self.schx = schx
        self.oversample = oversample
        self.fixed_params = fixed_params   # "Name=val,..." string, or None -- see _run_ngspice
        self.param_map = param_map         # knob-name -> netlist pot Name, or None (identity)
        # Device-model convergence/fidelity overrides (key=val,... parsed to a dict by
        # parse_conv() below) -- e.g. bjt_vaf/bjt_rb/... for a real datasheet-fitted transistor
        # (see schx_to_ngspice.bjt_model's _BJT_OPTIONAL). MUST reach every tool that renders
        # or measures this circuit identically, or one tool sizes/checks against a different
        # transistor response than what the dataset actually trains on -- the same class of
        # bug capture_chain.py's cache_tag/resolve() exist to prevent for the capture chain.
        self.conv = conv or {}
        self.ng_base = None                # filled lazily, once, on first prepare_input

    def _ensure_netlist(self, scratch):
        if self.ng_base is not None:
            return
        netlist_path = Path(scratch) / "netlist.json"
        r = subprocess.run([str(LIVESPICE_CLI), "--circuit", str(self.schx),
                            "--netlist", str(netlist_path)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not netlist_path.exists():
            sys.exit(f"netlist dump failed for {self.schx}: {r.stderr[:300]}")
        self.ng_base = {
            "netlist": str(netlist_path), "koren": False,
            "ot_damp": "47k", "ot_snub": "10n", "nfb_comp": None,
            "conv": self.conv, "method": "trap", "input_upsample": 1,
            "oversample": self.oversample,
        }

    def prepare_input(self, raw, sr, level_v, scratch, tag):
        self._ensure_netlist(scratch)
        peak = float(np.abs(raw).max()) + 1e-12
        scaled = (raw / peak * level_v).astype(np.float32)
        path = f"{scratch}/rawinput_{tag}.wav"
        sf.write(path, scaled, sr, subtype="FLOAT")  # FLOAT: values >1.0 (>1V drive) must survive
        return {"input_raw": path, "n": len(scaled)}

    def _render_one(self, params, handle, scratch, tag):
        ngp = dict(self.ng_base)
        # keyed per-tag, like grid_adequacy's own choose_oversample precedent -- _filesource
        # caches by upsample factor only, so a shared dir would serve one job's filesource
        # back for a different one.
        ngp.update({"input_raw": handle["input_raw"], "fsrc_dir": f"{scratch}/fs_{tag}"})
        out_wav = Path(scratch) / f"ns_{tag}.wav"
        fail = _run_ngspice(0, params, Path(scratch) / f"ng_{tag}", out_wav, handle["n"],
                            120, self.param_map, self.fixed_params, ngp)
        if fail is not None or not out_wav.exists():
            return None
        y, _ = sf.read(str(out_wav), dtype="float64")
        return y

    def render_many(self, jobs, input_handle, scratch):
        if not jobs:
            return {}
        out = {}
        workers = max(1, min(os.cpu_count() or 4, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._render_one, j["params"], input_handle, scratch, j["tag"]): j["tag"]
                    for j in jobs}
            for f in futs:
                out[futs[f]] = f.result()
        return out


class LtspiceBackend:
    """Renders a hand-written LTspice deck via ltspice_spicelib.py's render_grid (one LTspice
    subprocess per render), for a device whose ngspice-deck counterpart can't converge on real
    playing content at all -- see ltspice_spicelib.py's own docstring for why (a razor-steep
    tanh-bounded op-amp B-source is a genuine Newton-solver dead end in ngspice, independent of
    timestep; LTspice needs a real op-amp macromodel + .ic/uic hints instead, neither of which
    ngspice's B-source style has room for)."""

    def __init__(self, build_deck, tap="spk", maxstep=3e-6, parallel_sims=8, out_scale=0.05,
                 timeout=None):
        self.build_deck = build_deck
        self.tap = tap
        self.maxstep = maxstep
        self.parallel_sims = parallel_sims
        self.out_scale = out_scale
        # Per-render ceiling, forwarded to render_grid. None keeps its duration-scaled default
        # (DEFAULT_TIMEOUT_S_PER_AUDIO_S = 20 s per audio second), which assumes a render runs
        # at better than ~20x realtime. That is a per-render WALL figure, so it does not account
        # for parallel_sims renders competing for the same cores: on a slow circuit the default
        # can kill every job and report it as "RENDER FAILED", indistinguishable from a genuine
        # convergence failure. Measured on the budget clone pedal, a 10 s probe takes ~129 s
        # alone (inside the 200 s default) but well past it with 8 running concurrently.
        self.timeout = timeout

    def prepare_input(self, raw, sr, level_v, scratch, tag):
        wav_path = f"{scratch}/rawinput_{tag}.wav"
        sf.write(wav_path, raw.astype(np.float32), sr, subtype="FLOAT")
        return ltspice_spicelib.load_input(wav_path, level_v, scratch, src_name=f"input_{tag}.wav")

    def render_many(self, jobs, input_handle, scratch):
        if not jobs:
            return {}
        sr_, dur_s_, wav_path_, in_scale_ = input_handle
        outfiles = {j["tag"]: f"{scratch}/pf_{j['tag']}.wav" for j in jobs}
        rg_jobs = [(j["params"], outfiles[j["tag"]]) for j in jobs]
        peaks = ltspice_spicelib.render_grid(self.build_deck, rg_jobs, self.tap, sr_, dur_s_,
                                             wav_path_, in_scale_, scratch,
                                             maxstep=self.maxstep, parallel_sims=self.parallel_sims,
                                             out_scale=self.out_scale, timeout=self.timeout)
        out = {}
        for j in jobs:
            pk = peaks[outfiles[j["tag"]]]
            if pk is None:
                out[j["tag"]] = None
            else:
                y, _ = sf.read(outfiles[j["tag"]], dtype="float64")
                out[j["tag"]] = y
        return out
