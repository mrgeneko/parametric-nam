#!/usr/bin/env python3
"""Shared ngspice render backend for this repo's per-device render_*.py scripts, using
spicelib (https://github.com/nunobrum/spicelib) to drive ngspice in PARALLEL instead of the
sequential subprocess-per-render loop every render_*.py used to hand-roll independently
(several per-pedal render scripts and amps/render_ngspice.py were all near-identical copies of
the same pattern).

Tried and measured (the mid-boost overdrive pedal, 9-point Drive x Tone sweep, parallel_sims=8): 2.1x wall-clock
speedup over the sequential baseline (20.7s vs 43.5s), plus spicelib's RawRead parses ngspice's
binary .raw output directly -- no change needed to how results are read back, since the deck
strategy below still explicitly `write`s the raw file itself (see next paragraph for why).

INTEGRATION GOTCHA (worth knowing before touching this file): spicelib's SimRunner has its own
built-in raw/log file auto-detection, but it ASSUMES a traditional deck with a top-level
`.tran` analysis statement (SimRunner passes `-r <netlist>.raw` and expects ngspice to dump
there automatically when the deck finishes). This repo's decks all use an INTERACTIVE-STYLE
`.control ... tran ... .endc` block instead (needed to set solver options and the XSPICE
`filesource` input in the same file) -- and running `tran` from inside `.control` does NOT
trigger that automatic dump. Confirmed empirically: even with parallel_sims=1 (no concurrency
involved), SimRunner's own `task.raw_file` path never appears on disk, while ngspice's own
console log claims it wrote there. The fix used here: keep an explicit `write <path> v(...)`
command inside `.control` (same convention every render_*.py in this repo already used with
`wrdata`, just switched to binary `write` for RawRead), and track/read back results via
self-managed deterministic paths -- NOT via SimRunner's task.raw_file/task.log_file. This means
SimRunner is being used purely as a parallel process-pool scheduler here, not its full
"batteries included" pipeline (its own raw/log auto-tracking, wait_completion() etc. still
work fine for scheduling -- only the auto-raw-file-discovery part doesn't apply to this repo's
deck style).

Usage from a device's render_X.py:
    from ngspice_spicelib import load_input, render_grid
    sr, t, input_src = load_input(infile, vin, tmp, src_name='mydevice_input.src')
    jobs = [({'Drive': 0.2, 'Tone': 0.5}, 'out1.wav'), ...]
    results = render_grid(build_deck, jobs, probe_node='OUT', sr=sr, t=t,
                           input_src=input_src, tmp=tmp, parallel_sims=8)
    # results: {outfile: peak_or_None}
"""
import os
import sys
import numpy as np
from scipy.io import wavfile
from spicelib import SimRunner, RawRead
from spicelib.simulators.ngspice_simulator import NGspiceSimulator


def load_input(infile, vin, tmp, src_name='input.src'):
    """Read wav, scale to +/-vin, write an XSPICE filesource time/value file (padded with a
    held final value so a tran to any duration up to the file's own lands safely inside the
    data). Returns (sr, t, input_src) where input_src is the two filesource lines that drive
    node `sig`. Written ONCE and reused across a knob sweep -- same convention as every
    render_*.py in this repo already used, just factored out.

    `vin=None` selects ABSOLUTE-VOLTS mode: the file's own sample values are used directly as
    volts, with NO peak rescaling. Use this for a file already built at the parametric-nam
    V0dBFS=1V convention (e.g. build_excitation.py's output), where different SEGMENTS
    intentionally sit at different absolute levels (the --input clip at --realistic-peak,
    sweeps at each of --sweep-peaks) -- rescaling by the file's own peak would flatten that
    deliberate multi-level structure into one uniform scale, destroying the point of building
    it that way. Only meaningful for a float WAV (int PCM has no way to represent volts >1
    without clipping, so it's still peak-normalized to +/-1 in that case regardless of vin)."""
    sr, x = wavfile.read(infile)
    if x.dtype.kind == 'i':
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    if vin is not None:
        x = x / (np.max(np.abs(x)) + 1e-9) * vin
    t = np.arange(len(x)) / sr
    pad_t = t[-1] + np.arange(1, int(0.01 * sr)) / sr
    T = np.concatenate([t, pad_t])
    V = np.concatenate([x, np.full(len(pad_t), x[-1])])
    src = os.path.join(tmp, src_name)
    np.savetxt(src, np.c_[T, V], fmt='%.8e')
    input_src = ('a1 %vd([sig 0]) filesrc\n'
                 f'.model filesrc filesource (file="{src}" amploffset=[0] amplscale=[1] '
                 'timeoffset=0 timescale=1 timerelative=false amplstep=false)')
    return sr, t, input_src


def _read_result(raw_path, probe_node, t, sr):
    """Read one job's raw file, resample onto the input's own time base (matches every
    render_*.py's existing np.interp behaviour), return (yv, peak) or (None, None).

    Checks the sim actually REACHED the requested duration, not just that it produced
    "enough" samples -- an ngspice `tran` that aborts early ("Timestep too small...")
    still writes a valid, readable raw file for whatever partial time range it completed.
    Without this check, a partial trace with >= len(t)//2 samples was accepted as
    converged, and np.interp then silently flat-extrapolates the last simulated value
    for the remainder of `t` -- indistinguishable from a real signal until inspected
    sample-by-sample. Found via a real render (the MOSFET-clipping pedal) that aborted at t=5.045s (node na_o)
    out of a ~8.4s request but still passed the old length-only check, corrupting an
    LTspice-vs-ngspice comparison with >3s of held-flat fake reference data."""
    if not os.path.exists(raw_path):
        return None, None
    try:
        raw = RawRead(raw_path)
        trace = raw.get_trace(f'v({probe_node.lower()})')
    except Exception:
        return None, None
    v = np.asarray(trace.get_wave(), dtype=float)
    tv = np.asarray(raw.get_trace('time').get_wave(), dtype=float)
    if v.ndim < 1 or len(v) < len(t) // 2 or len(tv) == 0:
        return None, None
    if tv[-1] < 0.999 * t[-1]:
        return None, None
    yv = np.interp(t, tv, v)
    pk = float(np.max(np.abs(yv)))
    return yv, pk


def render_grid(build_deck, jobs, probe_node, sr, t, input_src, tmp,
                 maxstep=3e-6, parallel_sims=8, timeout=300,
                 opt_line="reltol=2e-3 abstol=1e-9 gmin=1e-10 method=gear", rungs=None):
    """Render many (knobs, outfile) jobs in parallel, with the SAME progressively-finer-
    timestep retry escalation every render_*.py used sequentially (maxstep, maxstep/3,
    maxstep/10) -- but each escalation ROUND runs its still-pending jobs in parallel, instead
    of retrying one file at a time. `build_deck(input_src=..., knobs=...)` must match the
    signature every gen_*_ngspice.py's build_deck() already has. Returns {outfile: peak} for
    every job (peak is None for jobs that never converged after all three rounds).

    `probe_node` is looked up as v(<probe_node>) in the raw file -- must match a node name in
    the deck (e.g. 'OUT', 'spk'), same as every render_*.py's existing `wrdata ... v(X)` arg.

    `rungs`: override the escalation ladder (default `(maxstep, maxstep/3, maxstep/10)`).
    Pass `rungs=(maxstep,)` for a genuine SINGLE-SHOT attempt with no silent escalation --
    needed by measure_ngspice_timestep.py, which measures maxstep's own effect on the
    render and would otherwise get a DIFFERENT actual timestep than requested for any job that
    fails to converge at the nominal one (found directly: a "maxstep=1e-7" run silently
    resolved to ~1e-8 for a hard corner via this same escalation, invalidating the very
    comparison the tool exists to make).
    """
    dur = len(t) / sr
    runner = SimRunner(simulator=NGspiceSimulator, parallel_sims=parallel_sims,
                        timeout=timeout, output_folder=tmp)
    results = {}
    pending = list(jobs)
    for round_i, step in enumerate(rungs if rungs is not None else (maxstep, maxstep / 3, maxstep / 10)):
        if not pending:
            break
        raw_paths, tasks = {}, {}
        for knobs, outfile in pending:
            tag = os.path.splitext(os.path.basename(outfile))[0]
            raw_out = os.path.join(tmp, f'{tag}_r{round_i}.raw')
            cir_path = os.path.join(tmp, f'{tag}_r{round_i}.cir')
            deck = build_deck(input_src=input_src, knobs=knobs)
            deck += (f"\n.options {opt_line}\n"
                     f".control\ntran {step:.1e} {dur:.6f}\n"
                     f"write {raw_out} v({probe_node})\n.endc\n.end\n")
            open(cir_path, 'w').write(deck)
            tasks[outfile] = runner.run(cir_path, exe_log=True)
            raw_paths[outfile] = raw_out
        runner.wait_completion()

        still_pending = []
        for knobs, outfile in pending:
            # A NEGATIVE retcode is a signal, not a solve: ngspice CRASHED (e.g. the documented
            # macOS large-filesource segfault -- see gen_dataset_from_schx.py's _run_ngspice,
            # which hit the identical failure via a direct subprocess.Popen and had to learn this
            # the hard way). spicelib's RunTask.retcode is subprocess.run()'s own returncode, same
            # sign convention. A crashed job writes no raw file, which _read_result already reads
            # as "not converged yet" -- so without this check it gets retried at a finer timestep
            # on every remaining round for nothing, since a finer timestep cannot fix a crash, and
            # it is reported back as a bare None with no way to tell it apart from a genuine
            # non-convergence.
            task = tasks.get(outfile)
            if task is not None and task.retcode is not None and task.retcode < 0:
                print(f"ngspice killed by signal {-task.retcode} on {os.path.basename(outfile)} "
                      f"(crash, not a solve failure) -- not retrying at a finer timestep.",
                      file=sys.stderr)
                results[outfile] = None
                continue
            yv, pk = _read_result(raw_paths[outfile], probe_node, t, sr)
            if yv is None:
                still_pending.append((knobs, outfile))
                continue
            wavfile.write(outfile, sr, (yv / (pk + 1e-9) * 0.9 * 32767).astype(np.int16))
            results[outfile] = pk
        pending = still_pending

    for _knobs, outfile in pending:
        results[outfile] = None
    return results


def render_one(build_deck, knobs, outfile, probe_node, sr, t, input_src, tmp, maxstep=3e-6):
    """Single-render convenience wrapper matching every render_*.py's existing render()
    signature/return value (peak, or None on failure) -- for the --knob (single render) CLI
    path, where spicelib's parallelism has nothing to parallelize."""
    results = render_grid(build_deck, [(knobs, outfile)], probe_node, sr, t, input_src, tmp,
                           maxstep=maxstep, parallel_sims=1)
    return results[outfile]
