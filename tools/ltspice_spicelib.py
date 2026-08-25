#!/usr/bin/env python3
"""Shared LTspice render backend for gen_*_ltspice.py device modules, mirroring
ngspice_spicelib.py's load_input/render_grid/render_one contract as closely as LTspice's own
constraints allow.

WHY THIS EXISTS. gen_ocd_ngspice.py's ideal tanh-bounded op-amp (`B{out} {out} 0
V=4.5+3.0*tanh(2e4*V(vp,vm))`) cannot converge on real playing content in ngspice: found
directly, re-rendering the full OCD excitation with ngspice_spicelib.render_grid's truncation
bug fixed (a render that ABORTS mid-way used to be silently accepted as "converged" if it had
produced >= 50% of the expected samples -- see that fix's own docstring) showed EVERY one of 70
renders across the full knob grid timing out at 300s, at every maxstep from 3e-6 down to 3e-8.
This isn't a step-size problem, it's a genuine Newton-solver dead end with this circuit's
razor-steep tanh nonlinearity. LTspice gets past it with two changes neither of which is
available through ngspice's B-source style: a real op-amp macromodel (UniversalOpAmp2.lib
instead of a raw tanh expression) and explicit `.ic` initial-condition hints + `uic` on the
`.tran` line (skips LTspice's own `.op` search, which was otherwise landing on a WRONG,
degenerate equilibrium for this circuit -- found by probing a bias node directly with a silent
input and seeing it settle to 0V instead of the correct ~4.5V VREF-centered bias).

Validated (Fulltone OCD, gen_ocd_ltspice.py): at a signal level low enough that ngspice CAN
converge (there's no ngspice ground truth at real playing levels -- see above), LTspice matches
it almost exactly in steady state (ESR=0.0004, corr=1.0000) after excluding a short (<60ms)
onset-settling transient from the .ic guess not being perfectly self-consistent. At hard drive,
internal clipping-node voltages match the circuit's own documented physical bounds (MOSFET/
diode clamp levels) almost exactly. See internal engineering notes on the LTspice port
investigation for the full comparison.

CONTRACT DIFFERENCES FROM ngspice_spicelib.py (both inherent to LTspice, not stylistic):

  - load_input returns (sr, dur_s, wav_path, in_scale), not (sr, t, input_src). LTspice's
    `wavefile=` input is PCM-only (a FLOAT32 WAV fails: "Fatal Error: Bad wavefile format
    found"), and PCM is hard-bounded to +/-1.0 -- confirmed directly: writing a WAV with
    soundfile's PCM_24 subtype silently CLIPS any sample outside [-1, 1] with no error, so an
    excitation with peaks well past 1V (routine in this pipeline -- OCD's real excitation peaks
    at 11.75V) would be silently corrupted at the input stage if written naively. `in_scale`
    is the divisor applied when writing (wav_value = raw_volts / in_scale, chosen so the file's
    own peak stays under 1.0); every gen_*_ltspice.py build_deck must apply the inverse gain
    internally via an E-source (`Ein sig 0 sig_raw 0 {in_scale}`) to restore real volts before
    the rest of the circuit sees the signal -- the same pattern this module's callers already
    use for OUTPUT (out_scale, below), just mirrored for input.

  - build_deck's signature is necessarily richer than ngspice's `build_deck(input_src, knobs)`:
    `build_deck(wav_path, dur_s, maxstep, out_wav, knobs=None, tap=..., out_scale=..., in_scale=...,
    method=None) -> str`. LTspice has no separate `.control` block to append a `tran`/`write`
    command to after the fact (unlike ngspice's interactive-style decks) -- the `.tran ...
    maxstep` line and `.wave "out_wav" ... V(tap)` line must be part of the deck text itself,
    so dur_s/maxstep/out_wav have to be first-class build_deck parameters, not appended
    externally by render_grid the way ngspice_spicelib does it.

  - LTspice's `.wave` OUTPUT is ALSO +/-1V-PCM-bounded (a genuine >1V signal at the tap gets
    silently hard-clipped at exactly 1.0) -- every gen_*_ltspice.py build_deck must scale its
    tap down before the `.wave` line (`Eoutscale spkout 0 {tap} 0 {out_scale}`, e.g. 0.05) and
    this module divides back out after reading.

Usage from a device's render_X.py (once one exists, mirroring render_ngspice_deck.py):
    from tools.ltspice_spicelib import load_input, render_grid
    sr, dur_s, wav_path, in_scale = load_input(infile, vin, tmp, src_name='mydevice_input.wav')
    jobs = [({'Drive': 0.2, 'Tone': 0.5}, 'out1.wav'), ...]
    results = render_grid(build_deck, jobs, tap='spk', sr=sr, dur_s=dur_s, wav_path=wav_path,
                           in_scale=in_scale, tmp=tmp, parallel_sims=8)
    # results: {outfile: peak_or_None} -- outfile is written as a FLOAT32 WAV in real volts.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

PAD_S = 0.01  # held-final-value tail pad, matches ngspice_spicelib.load_input's own convention


def _find_ltspice_bin() -> Path:
    env = os.environ.get("LTSPICE_BIN")
    if env:
        return Path(env)
    for candidate in (Path.home() / "Applications/LTspice.app/Contents/MacOS/LTspice",
                       Path("/Applications/LTspice.app/Contents/MacOS/LTspice")):
        if candidate.exists():
            return candidate
    return Path.home() / "Applications/LTspice.app/Contents/MacOS/LTspice"


LTSPICE_BIN = _find_ltspice_bin()


def load_input(infile, vin, tmp, src_name="input.wav"):
    """Read wav, scale to +/-vin (vin=None: absolute-volts mode, the file's own sample values
    used directly -- same convention/use-case as ngspice_spicelib.load_input, e.g. a
    tools/build_excitation.py file whose segments intentionally sit at different absolute
    levels), pad the tail with the held final sample (so a `.tran` requesting exactly the
    returned dur_s lands safely inside the data, matching ngspice_spicelib's own pad), and
    write a PCM_24 WAV pre-divided by `in_scale` so the file itself never exceeds +/-1.0
    regardless of how many volts the excitation actually represents.

    Returns (sr, dur_s, wav_path, in_scale)."""
    x, sr = sf.read(infile, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    if vin is not None:
        x = x / (np.max(np.abs(x)) + 1e-9) * vin
    pad_n = int(PAD_S * sr)
    x = np.concatenate([x, np.full(pad_n, x[-1])])
    peak = float(np.max(np.abs(x))) + 1e-9
    in_scale = max(1.0, peak * 1.0001)  # never scale UP -- only down enough to clear +/-1.0
    wav_path = os.path.join(tmp, src_name)
    sf.write(wav_path, x / in_scale, sr, subtype="PCM_24")
    dur_s = len(x) / sr
    return sr, dur_s, wav_path, in_scale


def _run_ltspice(net_path, timeout) -> bool:
    import subprocess
    try:
        subprocess.run([str(LTSPICE_BIN), "-b", str(net_path)], capture_output=True,
                        text=True, timeout=timeout)
        return True
    except (subprocess.TimeoutExpired, OSError):
        return False


def _read_result(raw_wav, dur_target_n, out_scale):
    """Read one job's .wave output, un-scale it, return (yv, peak) or (None, None).

    Checks the render actually REACHED the requested duration -- an LTspice run that crashes
    or aborts ("Time step too small...") mid-transient still leaves a valid, readable partial
    .wave file for whatever it completed. Same lesson as ngspice_spicelib._read_result's own
    truncation fix: a length-only "did it produce enough samples" check isn't enough, since a
    genuinely truncated file can still exceed that threshold and get silently accepted, then
    treated as though its tail were real steady-state data."""
    if not os.path.exists(raw_wav):
        return None, None
    try:
        y, _sr = sf.read(raw_wav, dtype="float64")
    except Exception:
        return None, None
    if y.ndim > 1:
        y = y[:, 0]
    if len(y) < 0.999 * dur_target_n:
        return None, None
    y = y / out_scale
    pk = float(np.max(np.abs(y)))
    return y, pk


DEFAULT_TIMEOUT_S_PER_AUDIO_S = 20.0  # see render_grid's timeout= docstring
MIN_TIMEOUT_S = 120.0


def render_grid(build_deck, jobs, tap, sr, dur_s, wav_path, in_scale, tmp,
                 maxstep=3e-6, parallel_sims=8, timeout=None, out_scale=0.05, rungs=None):
    """Render many (knobs, outfile) jobs, one LTspice subprocess per job, with the same
    progressively-finer-timestep retry escalation ngspice_spicelib.render_grid uses (maxstep,
    maxstep/3, maxstep/10) -- each round runs its still-pending jobs in parallel via
    ThreadPoolExecutor (LTspice has no built-in multi-job scheduler analogous to spicelib's
    SimRunner, so this manages concurrency itself; a thread per job is fine since each spends
    its time blocked in subprocess.run, not holding the GIL). `outfile` is written as a FLOAT32
    WAV in real volts (not peak-normalized int16 -- unlike ngspice_spicelib, there's no
    RawRead/resample step here, LTspice's own .wave output already IS a valid audio file at the
    right samplerate, so it's rescaled in place rather than re-derived).

    `rungs`: override the escalation ladder (default `(maxstep, maxstep/3, maxstep/10)`), same
    contract as ngspice_spicelib.render_grid's own `rungs` param -- pass `rungs=(maxstep,)` for
    a genuine single-shot attempt with no silent escalation.

    `timeout`: per-render subprocess ceiling, seconds. None (the default) SCALES WITH `dur_s`
    (max(MIN_TIMEOUT_S, dur_s * DEFAULT_TIMEOUT_S_PER_AUDIO_S)) rather than being one fixed
    number for every caller -- found the hard way: a flat 120s default worked fine for
    grid_adequacy's short 8s probe clips but silently killed every job partway through a real
    60s excitation capture (measured needing ~7-11 min/render at maxstep=3e-6, ~11s wall-clock
    per second of audio for OCD's stiffest corners), which LOOKS EXACTLY LIKE a genuine
    non-convergence (every job "fails" and escalates through the whole rungs ladder) rather
    than the timeout-too-short bug it actually was. DEFAULT_TIMEOUT_S_PER_AUDIO_S=20 is ~2x that
    single measured device's worst observed ratio -- a working default, not a validated
    per-device constant; a future device with a genuinely stiffer circuit or much longer
    excitation should re-measure rather than assume this holds, and can still pass an explicit
    `timeout=` to override the scaling entirely.

    Returns {outfile: peak} for every job (peak is None for a job that never converged/reached
    full duration after all rounds)."""
    if timeout is None:
        timeout = max(MIN_TIMEOUT_S, dur_s * DEFAULT_TIMEOUT_S_PER_AUDIO_S)
    dur_target_n = int(round(dur_s * sr))
    results = {}
    pending = list(jobs)
    for round_i, step in enumerate(rungs if rungs is not None else (maxstep, maxstep / 3, maxstep / 10)):
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=parallel_sims) as ex:
            futs = {}
            for knobs, outfile in pending:
                tag = os.path.splitext(os.path.basename(outfile))[0]
                net_path = os.path.join(tmp, f"{tag}_r{round_i}.net")
                raw_wav = os.path.join(tmp, f"{tag}_r{round_i}_raw.wav")
                deck = build_deck(wav_path, dur_s, step, raw_wav, knobs=knobs, tap=tap,
                                   out_scale=out_scale, in_scale=in_scale)
                with open(net_path, "w") as f:
                    f.write(deck)
                fut = ex.submit(_run_ltspice, net_path, timeout)
                futs[fut] = (knobs, outfile, raw_wav)

            still_pending = []
            for fut in as_completed(futs):
                knobs, outfile, raw_wav = futs[fut]
                fut.result()  # exceptions already swallowed by _run_ltspice -> False
                yv, pk = _read_result(raw_wav, dur_target_n, out_scale)
                if yv is None:
                    still_pending.append((knobs, outfile))
                    continue
                sf.write(outfile, yv.astype(np.float32), sr, subtype="FLOAT")
                results[outfile] = pk
        pending = still_pending

    for _knobs, outfile in pending:
        results[outfile] = None
    return results


def render_one(build_deck, knobs, outfile, tap, sr, dur_s, wav_path, in_scale, tmp,
                maxstep=3e-6, out_scale=0.05, timeout=None):
    """Single-render convenience wrapper matching ngspice_spicelib.render_one's role -- for a
    --knob (single render) CLI path, where render_grid's parallelism has nothing to
    parallelize. timeout=None scales with dur_s -- see render_grid's own docstring."""
    results = render_grid(build_deck, [(knobs, outfile)], tap, sr, dur_s, wav_path, in_scale,
                           tmp, maxstep=maxstep, parallel_sims=1, out_scale=out_scale,
                           timeout=timeout)
    return results[outfile]
