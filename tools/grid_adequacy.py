#!/usr/bin/env python3
"""Is the knob grid dense enough? Answer it BEFORE training, and without training anything.

THE QUESTION.
A parametric model is fitted to the circuit at a finite set of knob settings and is expected to fill
in everything between them. So the grid has to be dense enough that "everything between them" is
recoverable. Too coarse and the model cannot win no matter how long you train it -- the information
simply is not in the data. Too dense and you pay for renders and epochs that teach it nothing.

Nobody could say which we had. The obvious way to find out -- train on grid A, train on grid B,
compare -- is expensive, slow, and confounded by optimiser noise (we tried: nine overnight runs whose
ESR got monotonically WORSE as the grid got denser, which is physically impossible and told us only
that the runs were undertrained).

THE MEASUREMENT.
Render the circuit's TRUE output at the midpoint of a grid cell. Compare it against interpolating the
two neighbouring grid points. The residual is the interpolation error THE GRID IMPOSES: a property of
the circuit and the sampling, with no model in it at all. Compare that residual to the ESR you are
chasing:

    residual >> target ESR   the GRID is the limiting factor. No capacity, no training time, and no
                             architecture can recover what was never sampled. Refine the cell -- or
                             decide the refinement is not worth its render budget, which is a
                             legitimate answer (see EXIT STATUS).
    residual << target ESR   the cell is OVERSAMPLED. Those points cost render time and lengthen
                             every epoch (dataset items = perms x repeats) and teach the model
                             nothing it could not have interpolated.

It is cheap: 3 renders per cell probe, and cells are independent so they run concurrently.

WHAT IT FOUND ON THE BIG MUFF (14 x 9 = 126 perms, the shipped grid):

    Sustain 0.001-0.70   11 cells, all <= 0.0010     oversampled 10-100x
    Sustain 0.70-0.85                     0.0091     at the limit
    Sustain 0.85-1.00                     0.0813     9x OVER -- the model cannot win here
    Tone    every one of 8 cells          0.0000     all nine values interpolate perfectly

So the grid was simultaneously too dense (almost everywhere) and too coarse (in one place), and the
one coarse cell is where a Big Muff actually lives. The Sustain axis had been hand-tuned dense at the
BOTTOM, on the reasoning that a linear-taper pot puts most of its audible change in the bottom decade.
That is true of LEVEL, and backwards for WAVEFORM SHAPE, which changes fastest at the top where the
pedal is deep into clipping -- and waveform shape is what the model has to learn.

CAVEAT: GRID ADEQUACY IS NOT TRAINING BALANCE.
This tool only answers "is the circuit's response sampled densely enough to interpolate" -- a
property of the circuit and the sampling, with no model in it. It says nothing about whether a
TRAINED model actually learns to represent every knob proportionally.

WHAT IT MISSED ON THE TS-9.
Drive measured <= 0.0008 in every cell (trivially easy to interpolate) while Tone measured up to
0.1038 at the top of its travel -- by every signal this tool reports, Drive was the "easy" knob and
shipped with a sparse 4-point grid while Tone earned 8. But a post-training sensitivity sweep (sweep
each knob 0->1, measure output RMS change per slimmable tier) found the 3ch and 5ch tiers had all
but ignored Drive (0.6% / 3.8% RMS spread across its full range) while the 8ch tier learned it
properly (15.4%). Likely cause: raw dataset RMS spread across each knob's full range was Drive 8.9%
vs Tone 51.9% -- Tone dominates the audible variation in the training signal by ~6x, so a
capacity-limited tier spends its budget on the bigger loss-reduction opportunity (Tone) at Drive's
expense, even though Drive's underlying circuit response was easy to sample. A grid this tool calls
"adequate" can still starve a knob of gradient signal relative to a knob with more audible range, if
nothing else corrects for it (loss weighting, tier width, or -- the blunt instrument -- just giving
the starved knob more grid points so it has more to learn from per epoch). Grid density and training
balance are separate problems; this tool only measures the first one.

EXIT STATUS.
An over-target cell does NOT fail the run (exit 0). It is a priced trade-off, not a defect:
refining costs renders and permutations, and a config may rationally accept a coarse cell,
coarsen an axis, or fix a knob outright instead of paying. Gating on it would turn a judgement
call into an error.

FAILED RENDERS DO fail the run (exit 2), because that is the case that silently produces a WRONG
table rather than a weaker one -- every cell is computed from whatever probes survived, and
nothing else in the output says so. Measured on the Joyo American Sound: 38 of 48 probes timed
out, and the table put one cell at 0.6578 (18.8x over) where a clean re-run measured 0.0475
(1.4x). If the cause is a timeout rather than true non-convergence, raise --ltspice-timeout.

    ./tools/grid_adequacy.py --config path/to/device-config.toml --target 0.009
    ./tools/grid_adequacy.py --config ... --suggest       # propose a regrid, print it, stop
    ./tools/grid_adequacy.py --config ... --apply         # iterate suggest+reverify, write the result
"""
from __future__ import annotations

import argparse
import os as _os
import re
import subprocess
import sys
import tempfile
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gen_dataset_from_schx import (LIVESPICE_CLI, NGSPICE_CIRCUIT_DEFAULTS, _run_ngspice,
                                    check_oracle, write_probe_clip)

import importlib  # noqa: E402

from tools.ngspice_spicelib import load_input  # noqa: E402
from tools import ltspice_spicelib  # noqa: E402
from tools.prepare_excitation import _parse_fixed  # noqa: E402
from tools.render_backends import NgspiceBackend, LtspiceBackend  # noqa: E402


def esr(a: np.ndarray, b: np.ndarray) -> float:
    m = min(len(a), len(b))
    den = float(np.sum(b[:m] ** 2))
    return float(np.sum((a[:m] - b[:m]) ** 2) / den) if den > 0 else float("nan")


def load_config(p: Path) -> dict:
    with open(p, "rb") as f:
        return tomllib.load(f)


def write_knobs(config_path: Path, knobs: dict) -> None:
    """In-place text splice for --apply: replace only the [knobs] table's VALUE lines,
    leaving every comment (including ones explaining the PREVIOUS grid, which read fine
    as history) and every other section byte-for-byte untouched. Requires the standard
    `[knobs]` header + one `NAME = [...]` line per knob layout every config in this repo
    uses -- a full TOML round-trip (parse -> dict -> re-serialize) would blow away
    exactly the hand-written commentary these configs are full of.

    Only consumes the CONSECUTIVE `NAME = [...]` lines right after the header, stopping at
    the first blank line, comment, or new section -- some configs put an explanatory
    comment between [knobs]'s last line and the next section (e.g. muff's "Volume is a
    passive output divider..." paragraph ahead of [fixed]); stopping there instead of at
    the next `[section]` keeps that from being silently deleted as if it were still part
    of the knobs table."""
    text = config_path.read_text()
    m = re.search(r"^\[knobs\][ \t]*(?:#.*)?\n", text, re.MULTILINE)
    if not m:
        sys.exit(f"--apply: no [knobs] table found in {config_path} -- write it by hand.")
    body_start = m.end()
    assign_re = re.compile(r"^[A-Za-z0-9_.-]+[ \t]*=.*\n", re.MULTILINE)
    body_end = body_start
    while (am := assign_re.match(text, body_end)):
        body_end = am.end()

    width = max(len(k) for k in knobs)
    lines = [f"{name:<{width}} = [{', '.join(repr(float(v)) for v in vals)}]\n"
             for name, vals in knobs.items()]
    config_path.write_text(text[:body_start] + "".join(lines) + text[body_end:])


class Renderer:
    """Renders a knob setting on STRATIFIED WINDOWS spanning the input, not on one clip.

    WHICH PART OF THE INPUT YOU PROBE DECIDES THE ANSWER, and both obvious choices are wrong. The
    first probe_s seconds understates everything (our sweeps open quiet). The single highest-slew
    window OVERSTATES it: on sweepv5 that window lands inside the chirp section, and measured there
    the Big Muff's Sustain 0.03-0.12 cell reads 0.0079 against a whole-file truth of 0.0008 -- a 10x
    exaggeration that would have bought points the model does not need.

    The quantity that matters is the whole-file one, because that is what the model is fitted against
    and scored on. So: several short windows spanning the input, rendered SEPARATELY (never spliced --
    a concatenation manufactures discontinuities that are not in the signal and the probe then
    measures its own splice), with the ESR numerator and denominator POOLED across them. sum(err) /
    sum(sig) is exactly the whole-file ESR restricted to the sampled windows, and an unbiased estimate
    of it.

    This is the same trap, and the same fix, as choose_oversample in gen_dataset_from_schx.py.

    BACKEND: ngspice needed the same three fixes choose_oversample already required --
    (1) probe clips carry a quiet lead-in (write_probe_clip) so the solver has an
    operating point at t=0 instead of jumping straight into a large-signal mid-chirp
    state; (2) clips must be >= 3s -- ngspice SEGFAULTS (exit -11, no output, which
    _run_ngspice's caller reads as "diverged") on shorter ones, discovered on the DS-1
    at exactly the 2.00s the old 4-window/8s-probe default produced; (3) a per-circuit
    netlist dump + NGSPICE_CIRCUIT_DEFAULTS lookup (method/conv/input_upsample), done
    once up front, same as gen_dataset_from_schx.main() does for the real dataset render.

    BACKEND "ngspice-deck": for a device with no .schx at all (a real MOSFET/BJT .schx has no
    model for -- see render_backends.py's NgspiceBackend). Reuses that same backend directly
    (its render_many contract already handles the int16-peak-unnormalization and retry
    escalation) instead of re-deriving the schx-netlist-translation path's ng_base machinery,
    which doesn't apply here (there's no schx to dump a netlist from). Each probe window's
    clip is already a WAV on disk (write_probe_clip, below) -- load_input(vin=None) reads it
    in ABSOLUTE-VOLTS mode, preserving the clip's own real drive level rather than rescaling
    it to some fixed peak, same convention preflight.py/prepare_excitation.py use.
    """

    def __init__(self, schx, inp, oversample, iterations, fixed, td, probe_s, n_windows=4,
                backend="livespice", pedal_dir=None, module=None, probe_node="OUT",
                lead_silence_s=None, ngspice_deck_maxstep=3e-6, ltspice_deck_maxstep=3e-6,
                ltspice_out_scale=0.05, ltspice_timeout=None):
        self.schx, self.os_, self.it = schx, oversample, iterations
        self.fixed, self.td, self.backend = fixed, td, backend
        self.lead_silence_s = lead_silence_s
        self.cache: dict = {}
        self.ng_base = None
        self.ngd_backend = None
        self.ltd_backend = None
        # Concurrent jobs can legitimately compute the SAME full params dict --
        # e.g. axis=Dist's cell (0.2,0.4) at oth={Tone:0.0} and axis=Tone's cell
        # (0.0,0.2) at oth={Dist:0.4} both render {Dist:0.4, Tone:0.0} -- and the
        # temp file paths below are derived purely from hash(key), with no
        # per-call uniqueness. Without a lock, two ThreadPoolExecutor workers can
        # both pass the "not yet cached" check and race to write the SAME .csv/
        # .cir file simultaneously, corrupting it mid-write (surfaced as ngspice's
        # CSV having an inconsistent column count partway through). Per-key
        # locking serializes only genuine duplicate work; different keys still
        # render in parallel.
        self._cache_lock = threading.Lock()
        self._key_locks: dict = {}
        # Grouped by message, not printed per-render: a config-level failure (bad knob
        # name, wrong circuit) fails EVERY render with the identical text, and this probe
        # can fire hundreds of them -- printing the same line hundreds of times just
        # buries the one thing worth reading. See gen_dataset_from_schx.py's identical fix.
        self.fail_counts: dict = {}
        self._fail_lock = threading.Lock()

        if backend == "ngspice-deck":
            # Same segfault as choose_oversample: short clips crash ngspice outright.
            probe_s = max(probe_s, 8.0)
            n_windows = 2
            sys.path.insert(0, _os.path.abspath(pedal_dir))
            mod = importlib.import_module(module)
            self.ngd_backend = NgspiceBackend(mod.build_deck, probe_node=probe_node,
                                              maxstep=ngspice_deck_maxstep)
        elif backend == "ltspice-deck":
            probe_s = max(probe_s, 8.0)
            n_windows = 2
            sys.path.insert(0, _os.path.abspath(pedal_dir))
            mod = importlib.import_module(module)
            self.ltd_backend = LtspiceBackend(mod.build_deck, tap=probe_node,
                                              maxstep=ltspice_deck_maxstep,
                                              out_scale=ltspice_out_scale,
                                              timeout=ltspice_timeout)
        elif backend == "ngspice":
            # Same segfault as choose_oversample: short clips crash ngspice outright.
            probe_s = max(probe_s, 8.0)
            n_windows = 2

            netlist_path = td / "netlist.json"
            r = subprocess.run([str(LIVESPICE_CLI), "--circuit", str(schx),
                                "--netlist", str(netlist_path)],
                               capture_output=True, text=True)
            if r.returncode != 0 or not netlist_path.exists():
                sys.exit(f"netlist dump failed for {schx}: {r.stderr[:300]}")
            cdef = next((v for k, v in NGSPICE_CIRCUIT_DEFAULTS.items() if k in Path(schx).name), {})
            conv = dict(kv.split("=", 1) for kv in cdef.get("conv", "").split(",") if "=" in kv)
            self.ng_base = {
                "netlist": str(netlist_path), "koren": False,
                "ot_damp": "47k", "ot_snub": "10n", "nfb_comp": None,
                "conv": conv, "method": cdef.get("method", "trap"),
                "input_upsample": int(cdef.get("input_upsample", 1) or 1),
            }

        x, sr = sf.read(str(inp))
        self.sr = sr
        total = len(x)
        win = min(total, int(probe_s * sr) // n_windows or total)
        if total <= win * n_windows:
            starts, win = [0], total
        else:
            # evenly-spaced CENTRES, not endpoints -- endpoints land on the sweep's leading silence
            # and its near-silent decay tail, and the probe then measures nothing. Same bug, same fix,
            # as choose_oversample in gen_dataset_from_schx.
            span = total - win
            starts = [int((i + 0.5) * span / n_windows) for i in range(n_windows)]

        # Probe clips carry a QUIET LEAD-IN. A window cut from the middle of a sweep starts at
        # whatever amplitude the signal happened to be at -- often full scale, mid-chirp -- which asks
        # the simulator to jump to a large-signal solution from an uninitialised state at t=0. ngspice
        # simply diverges ("diverged at t~0", every rung, no output). See gen_dataset_from_schx.
        self.clips = []
        self.lead_n = 0
        for i, st in enumerate(starts):
            c = td / f"probe{i}.wav"
            self.lead_n = write_probe_clip(x[st:st + win], sr, c, lead_s=self.lead_silence_s)
            self.clips.append(c)
        self.win_s = win / sr

        if backend == "ngspice-deck":
            self.ngd_handles = [load_input(str(c), None, str(td), src_name=f"gridadq_{i}.src")
                                for i, c in enumerate(self.clips)]
        elif backend == "ltspice-deck":
            self.ltd_handles = [ltspice_spicelib.load_input(str(c), None, str(td),
                                                             src_name=f"gridadq_{i}.wav")
                                for i, c in enumerate(self.clips)]

    def _note_fail(self, msg: str):
        with self._fail_lock:
            self.fail_counts[msg] = self.fail_counts.get(msg, 0) + 1

    def print_fail_summary(self):
        for msg, n in sorted(self.fail_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n} render(s) failed: {msg}", file=sys.stderr)

    def __call__(self, params: dict):
        """-> list of rendered windows (one array per probe window)."""
        key = tuple(sorted(params.items()))
        with self._cache_lock:
            if key in self.cache:
                return self.cache[key]
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._cache_lock:
                if key in self.cache:      # another thread finished while we waited
                    return self.cache[key]
            return self._render(params, key)

    def _render(self, params: dict, key):
        if self.backend == "ngspice-deck":
            fixed_dict = _parse_fixed(self.fixed) if self.fixed else {}
            knobs_full = {**fixed_dict, **params}
            out = []
            for i, handle in enumerate(self.ngd_handles):
                tag = f"{abs(hash(key))}_{i}"
                ys = self.ngd_backend.render_many([{"params": knobs_full, "tag": tag}], handle, self.td)
                y = ys.get(tag)
                if y is None:
                    self._note_fail("ngspice-deck render did not converge")
                out.append(None if y is None else np.asarray(y, dtype=np.float64))
            with self._cache_lock:
                self.cache[key] = out
            return out

        if self.backend == "ltspice-deck":
            fixed_dict = _parse_fixed(self.fixed) if self.fixed else {}
            knobs_full = {**fixed_dict, **params}
            out = []
            for i, handle in enumerate(self.ltd_handles):
                tag = f"{abs(hash(key))}_{i}"
                ys = self.ltd_backend.render_many([{"params": knobs_full, "tag": tag}], handle, self.td)
                y = ys.get(tag)
                if y is None:
                    self._note_fail("ltspice-deck render did not converge")
                out.append(None if y is None else np.asarray(y, dtype=np.float64))
            with self._cache_lock:
                self.cache[key] = out
            return out

        allp = ",".join(f"{k}={v}" for k, v in params.items())
        if self.fixed:
            allp = f"{self.fixed},{allp}"
        out = []
        for i, clip in enumerate(self.clips):
            w = self.td / f"{abs(hash(key))}_{i}.wav"
            if self.backend == "ngspice":
                nlen = int(sf.info(str(clip)).frames)
                ngp = dict(self.ng_base)
                ngp.update({
                    "input_raw": str(clip),
                    # keyed per-clip (like choose_oversample) -- _filesource caches by
                    # upsample factor only, so a shared dir would serve clip A's
                    # filesource back for clip B.
                    "fsrc_dir": str(self.td / f"fs_{abs(hash(str(clip)))}"),
                    "oversample": self.os_,
                })
                fail = _run_ngspice(0, params, self.td / f"ng_{abs(hash(key))}_{i}", w,
                                    nlen, 120, None, self.fixed, ngp)
                if fail is None and w.exists():
                    d, _ = sf.read(str(w))
                    out.append(np.asarray(d, dtype=np.float64))
                else:
                    self._note_fail(str(getattr(fail, 'error', 'no output')))
                    out.append(None)
                continue
            r = subprocess.run(
                [str(LIVESPICE_CLI), "--input", str(clip), "--output", str(w),
                 "--circuit", str(self.schx), "--params", allp,
                 "--oversample", str(self.os_), "--iterations", str(self.it)],
                capture_output=True, text=True)
            if r.returncode == 0 and w.exists():
                d, _ = sf.read(str(w))
                out.append(np.asarray(d, dtype=np.float64))
            else:
                tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
                self._note_fail(tail)
                out.append(None)
        with self._cache_lock:
            self.cache[key] = out
        return out


def cell_error(render, knobs: dict, axis: str, lo: float, hi: float, others: dict) -> float:
    """Interpolation error the grid imposes on the cell [lo, hi] of `axis`.

    Pooled across the probe windows: sum of squared error over sum of squared signal, which IS the
    whole-file ESR restricted to those windows. Averaging per-window ESRs instead would let one quiet
    window dominate the mean.
    """
    mid = 0.5 * (lo + hi)
    A = render({**others, axis: lo})
    B = render({**others, axis: hi})
    M = render({**others, axis: mid})
    num = den = 0.0
    for a, b, m in zip(A, B, M):
        if a is None or b is None or m is None:
            continue
        n = min(len(a), len(b), len(m))
        # skip the LEAD-IN, not a blind 10% -- on a short window the lead is more than that,
        # and the probe would be scoring its own warm-up transient
        sk = max(n // 10, getattr(render, 'lead_n', 0) + int(0.02 * 48000))
        interp = 0.5 * (a[:n] + b[:n])                # the best a grid-limited learner can recover
        num += float(np.sum((interp[sk:] - m[sk:n]) ** 2))
        den += float(np.sum(m[sk:n] ** 2))
    return num / den if den > 0 else float("nan")


def measure_grid(render, knobs: dict, target: float, workers: int) -> tuple[dict, int, int]:
    """Probe every cell in `knobs` with `render`, print the per-axis report, and return
    (worst_by_axis, n_coarse, n_over) -- used standalone and by --apply's iteration loop."""
    # Probe each cell at several positions of the OTHER knobs -- the knobs interact, and a cell
    # that interpolates cleanly at one Tone can fail at another. (On the Big Muff the Sustain
    # 0.85-1.0 cell is 0.0044 at Tone=0 and 0.0813 at Tone=1: an 18x spread. Probing one slice
    # would have missed it.)
    jobs = []
    for axis, vals in knobs.items():
        other_axes = {k: v for k, v in knobs.items() if k != axis}
        # Three reference slices: the other knobs at their low / mid / high value. Index each
        # axis at ITS OWN position -- axes have different lengths (e.g. Gain 9, Middle 3), so the
        # old code's single shared index (the first axis's midpoint) ran off the end of shorter
        # axes with an IndexError.
        def _slice(pos):
            return {k: (v[0] if pos == "lo" else v[-1] if pos == "hi" else v[len(v) // 2])
                    for k, v in other_axes.items()}
        slices = [{}] if not other_axes else [_slice(p) for p in ("lo", "mid", "hi")]
        for lo, hi in zip(vals, vals[1:]):
            for oth in slices:
                jobs.append((axis, lo, hi, oth))

    # ex.map blocks silently until EVERY job finishes, with no per-completion feedback --
    # a real probe batch can run for minutes with nothing printed, easy to mistake for a
    # hang (confirmed directly: had to check `ps`/a profiler to tell the difference).
    # submit + as_completed gives a heartbeat as results actually land, in completion
    # order rather than submission order -- res doesn't need to preserve job order below,
    # only the (axis, lo, hi) values each row already carries.
    print(f"  probing {len(jobs)} cell x reference-slice combination(s) ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(cell_error, render, knobs, j[0], j[1], j[2], j[3]): j for j in jobs}
        res = []
        for i, fut in enumerate(as_completed(futures), 1):
            j = futures[fut]
            res.append((j[0], j[1], j[2], fut.result()))
            print(f"    {i}/{len(jobs)} probes done", flush=True)

    render.print_fail_summary()

    # EVERY probe failed. Do NOT fall through to the per-cell table below: every cell would
    # print "?" with no explanation of why, one row at a time, across every axis -- exactly
    # the buried-cryptic-failure pattern this guard exists to stop. See the analogous fix in
    # measure_truncation.py / gen_dataset_from_schx.py.
    if jobs and all(not np.isfinite(r[3]) for r in res):
        raise RuntimeError(
            f"EVERY probe render failed ({len(jobs)}/{len(jobs)}). Nothing was measured, so "
            "there is no grid-adequacy table to print. See the failure(s) above -- this "
            "usually means a config problem (wrong knob/fixed-param name, wrong --circuit, "
            "wrong --backend), not a per-cell convergence issue; fix that and re-run.")

    worst_by_axis: dict = {}
    n_coarse = n_over = 0
    for axis, vals in knobs.items():
        rows = [r for r in res if r[0] == axis]
        worst = {}
        for lo, hi in zip(vals, vals[1:]):
            es = [r[3] for r in rows if r[1] == lo and r[2] == hi and np.isfinite(r[3])]
            worst[(lo, hi)] = max(es) if es else float("nan")
        worst_by_axis[axis] = worst

        print(f"  {axis}:")
        for (lo, hi), e in worst.items():
            if not np.isfinite(e):
                verdict = "?"
            elif e > target:
                verdict = f"OVER TARGET ({e/target:.1f}x) — the grid limits this cell"
                n_coarse += 1
            elif e <= target / 10:
                verdict = "oversampled — these points teach the model nothing"
                n_over += 1
            else:
                verdict = "ok"
            print(f"    {lo:>7.4g} - {hi:<7.4g}  {e:>8.4f}   {verdict}")
        print()

    print(f"  {n_coarse} cell(s) over target, {n_over} oversampled.")
    if n_coarse:
        print("  Within such a cell the grid, not the model, sets the error floor: no capacity and")
        print("  no training time recovers what was not sampled. That is a COST, not a verdict --")
        print("  a cell may be over target because refining it was judged not worth the render")
        print("  budget, or because the axis was deliberately coarsened. Exit status stays 0.")

    return worst_by_axis, n_coarse, n_over


def suggest_axis(values: list, worst: dict, target: float) -> list:
    """Refine the cells that fail, drop points that are not earning their place.

    A cell whose error is FAR under target (<= target/10) is oversampled: its interior is recoverable
    by interpolation, so the point between two such cells is redundant. A cell over target gets
    bisected. Applied once -- run again to iterate.
    """
    out = [values[0]]
    i = 0
    while i < len(values) - 1:
        lo, hi = values[i], values[i + 1]
        e = worst.get((lo, hi), float("nan"))
        if np.isfinite(e) and e > target:
            out.append(round(0.5 * (lo + hi), 4))        # bisect: the grid is losing here
            out.append(hi)
            i += 1
        elif (i + 2 < len(values)
              and np.isfinite(e) and e <= target / 10
              and np.isfinite(worst.get((values[i + 1], values[i + 2]), float("nan")))
              and worst[(values[i + 1], values[i + 2])] <= target / 10):
            out.append(values[i + 2])                     # drop values[i+1]: both cells are trivial
            i += 2
        else:
            out.append(hi)
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="the same TOML the pipeline uses")
    ap.add_argument("--target", type=float, default=0.03,
                    help="the interpolation ESR the knob grid must support (default 0.03). A cell "
                         "whose interpolation error exceeds this is the LIMITING FACTOR -- no training "
                         "fixes it. NOTE this is an AUDIBILITY floor, not a training-fidelity target: "
                         "grid interpolation error is masked, signal-correlated distortion the user "
                         "has no reference for, so ~0.03 (-15 dB) is inaudible in use. Do NOT set it to "
                         "the model ESR you train to (0.003-0.007) -- that over-densifies the grid for "
                         "error nobody can hear. Plain ESR matches what the NAM trainer selects on.")
    ap.add_argument("--probe-s", type=float, default=8.0,
                    help="seconds of the input to probe with, taken from its highest-slew window")
    ap.add_argument("--iterations", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 4) - 2))
    ap.add_argument("--suggest", action="store_true", help="propose a regrid and print it as TOML")
    ap.add_argument("--apply", action="store_true",
                    help="iterate suggest+reverify (implies --suggest's logic) until the grid "
                         "clears --target or --max-iterations runs out, writing the result back "
                         "into --config's [knobs] table -- one command instead of suggest, "
                         "hand-copy, rerun, repeat.")
    ap.add_argument("--max-iterations", type=int, default=5,
                    help="cap on --apply's suggest/reverify loop (default 5)")
    ap.add_argument("--lead-silence-s", type=float, default=None,
                    help="override write_probe_clip's 1.0s default lead-in for a circuit whose "
                         "own settling time is longer (e.g. a slow RC network -- found directly "
                         "on the Fulltone OCD, whose ~5s C10/RVOL2 time constant left every "
                         "probe under-settled at the 1.0s default, producing a maxstep- and "
                         "grid-density-dependent 'stuck DC' artifact that looked like runaway "
                         "non-convergence). Default: use write_probe_clip's own 1.0s.")
    ap.add_argument("--ngspice-deck-maxstep", type=float, default=3e-6,
                    help="[ngspice-deck] ngspice timestep ceiling -- measure with "
                         "tools/measure_ngspice_timestep.py rather than guessing; the ecosystem "
                         "default is too coarse for at least one circuit found so far (the "
                         "Fulltone OCD's most extreme Gain/Tone corner needed ~1e-7)")
    ap.add_argument("--ltspice-deck-maxstep", type=float, default=3e-6, help="[ltspice-deck]")
    ap.add_argument("--ltspice-timeout", type=float, default=None,
                    help="[ltspice-deck] per-render wall ceiling in seconds. Default: "
                         "ltspice_spicelib's duration-scaled 20 s per audio second (so ~160 s at "
                         "--probe-s 8). That is a PER-RENDER figure and ignores the concurrent "
                         "probes competing for cores, so on a slow circuit the default times "
                         "renders out and they are reported as 'did not converge' -- which is "
                         "indistinguishable from a real convergence failure, and silently "
                         "computes the whole table from the surviving minority. If the run "
                         "reports failed renders, RAISE THIS before believing any cell.")
    ap.add_argument("--ltspice-out-scale", type=float, default=0.05,
                    help="[ltspice-deck] LTspice .wave output is +/-1V-PCM-bounded -- see "
                         "tools/ltspice_spicelib.py's docstring")
    args = ap.parse_args()

    cfg = load_config(args.config)
    knobs: dict = cfg["knobs"]
    inp = Path(_os.path.expanduser(cfg["input"]))
    fixed = ",".join(f"{k}={v}" for k, v in (cfg.get("fixed") or {}).items())
    backend = cfg.get("backend", "livespice")
    pedal_dir = module = probe_node = None
    if backend in ("ngspice-deck", "ltspice-deck"):
        schx = None
        pedal_dir = _os.path.expanduser(cfg["pedal-dir"])
        module = cfg["module"]
        probe_node = cfg.get("probe-node", "OUT")
        sys.path.insert(0, _os.path.abspath(pedal_dir))
        mod = importlib.import_module(module)
        unknown = set(knobs) - set(mod.KNOB_NAMES)
        if unknown:
            sys.exit(f"unknown knob(s) {sorted(unknown)} -- {module}'s real knobs are "
                     f"{mod.KNOB_NAMES}")
    else:
        schx = Path(_os.path.expanduser(cfg["schx"]))
    oversample = cfg.get("oversample", 2)
    if str(oversample).strip().lower() == "auto":
        # This tool measures INTERPOLATION error (grid density in knob-space), a different
        # question from truncation error (discretisation in time) that "auto" answers for
        # the real dataset render. It needs a fixed number to probe at, not a recursive
        # auto-measurement. 4 matches what auto has picked on the one ngspice circuit
        # measured so far (the DS-1) -- a reasonable default, not a derived one.
        print(f"  oversample auto (config) -> using 4 for this probe (interpolation error is "
             f"not what --oversample auto measures)")
        oversample = 4
    else:
        oversample = int(oversample)
    check_oracle(backend)

    n_perms = int(np.prod([len(v) for v in knobs.values()]))
    print(f"  config     {args.config}")
    print(f"  backend    {backend}")
    print(f"  grid       {' x '.join(str(len(v)) for v in knobs.values())} = {n_perms} permutations")
    print(f"  target ESR {args.target}   (a cell above this is the limiting factor)")
    probe_s = max(args.probe_s, 8.0) if backend in ("ngspice", "ngspice-deck", "ltspice-deck") else args.probe_s
    print(f"  probe      {probe_s:.0f}s @ oversample {oversample}, {args.iterations} iters"
         f"{' (bumped for ngspice -- short clips SIGSEGV, see Renderer)' if probe_s != args.probe_s else ''}\n")

    # --apply's suggest/reverify loop reuses this SAME Renderer (and its render cache) across
    # iterations -- a cell whose (axis, other-knobs-slice) params dict didn't change between
    # rounds (most of them; a bisection only touches the cells that failed) is a cache hit, not
    # a re-render. Cross-INVOCATION reuse (e.g. tuning one axis at a time across separate runs)
    # still re-renders from scratch -- that would need an on-disk cache keyed by (schx-rev,
    # params, oversample, iterations).
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        render = Renderer(schx, inp, oversample, args.iterations, fixed, td, args.probe_s,
                          backend=backend, pedal_dir=pedal_dir, module=module,
                          probe_node=probe_node, lead_silence_s=args.lead_silence_s,
                          ngspice_deck_maxstep=args.ngspice_deck_maxstep,
                          ltspice_deck_maxstep=args.ltspice_deck_maxstep,
                          ltspice_out_scale=args.ltspice_out_scale,
                          ltspice_timeout=args.ltspice_timeout)

        knobs_cur = {k: list(v) for k, v in knobs.items()}
        iteration = 0
        while True:
            iteration += 1
            if args.apply and iteration > 1:
                tot = int(np.prod([len(v) for v in knobs_cur.values()]))
                print(f"  --- iteration {iteration}/{args.max_iterations}: "
                      f"{' x '.join(str(len(v)) for v in knobs_cur.values())} = {tot} "
                      f"permutations ---\n")
            try:
                worst_by_axis, n_coarse, n_over = measure_grid(render, knobs_cur, args.target,
                                                                args.workers)
            except RuntimeError as e:
                sys.exit(f"\nERROR: {e}")

            if not args.apply:
                break
            if n_coarse == 0:
                print(f"  grid adequate after {iteration} iteration(s).")
                break
            if iteration >= args.max_iterations:
                print(f"  NOTE: {n_coarse} cell(s) still over target after "
                      f"{args.max_iterations} iteration(s) -- writing anyway. Rerun --apply to "
                      f"keep refining, or accept the cost: each further split multiplies the "
                      f"permutation count, and a coarse cell may be the cheaper trade.")
                break
            knobs_cur = {axis: suggest_axis(vals, worst_by_axis[axis], args.target)
                        for axis, vals in knobs_cur.items()}

    if args.suggest and not args.apply:
        print("\n  suggested regrid:\n\n  [knobs]")
        tot = 1
        for axis, vals in knobs.items():
            new = suggest_axis(list(vals), worst_by_axis[axis], args.target)
            tot *= len(new)
            print(f"  {axis:<8} = {new}")
        print(f"\n  -> {tot} permutations (was {n_perms})")

    if args.apply:
        write_knobs(args.config, knobs_cur)
        tot = int(np.prod([len(v) for v in knobs_cur.values()]))
        print(f"\n  wrote [knobs] ({tot} permutations, was {n_perms}) -> {args.config}")

    # EXIT POLICY, and it is deliberately the opposite of what it looks like it should be.
    #
    # Over-target cells do NOT fail the run. They are a priced trade-off: refining a cell costs
    # render time and permutations, and a config may rationally accept a coarse cell (or coarsen
    # an axis, or fix a knob outright) rather than pay. Gating on it turns a judgement call into
    # an error and trains people to pass --force at a measurement.
    #
    # FAILED RENDERS DO fail the run, which is the case that actually burned us. A run that could
    # not render every probe still prints a complete, plausible table -- computed from whatever
    # survived -- with nothing in the exit status to say so. Measured on the Joyo American Sound:
    # 38 of 48 probes timed out and the table it printed put one cell at 0.6578 (18.8x over) where
    # the clean re-run measured 0.0475 (1.4x). A 14x error, indistinguishable from a real result.
    # An incomplete measurement is not a weaker measurement, it is a wrong one.
    n_failed = sum(render.fail_counts.values())
    if n_failed:
        print(f"\n  MEASUREMENT INVALID: {n_failed} render(s) failed -- every number above was\n"
              f"  computed from the probes that survived. Do not act on this table. If the cause\n"
              f"  is a timeout rather than true non-convergence, raise --ltspice-timeout (or\n"
              f"  lower --workers) and re-run.", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
