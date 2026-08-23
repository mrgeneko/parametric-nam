#!/usr/bin/env python3
"""Render audio through a hand-written ngspice deck, with its knobs as SWEPT PARAMETERS --
generic, --pedal-dir/--module-parameterized replacement for what used to be a per-device copy
of this exact file (render_ocd.py, render_bd2.py, render_boss_od3.py, render_boss_hm2.py,
render_jc120.py, render_ngspice.py -- all near-identical, differing only in which
gen_*_ngspice.py they imported and a handful of device-specific defaults). Same consolidation
this repo already did for preflight.py/prepare_excitation.py (--backend ngspice-deck) --
adding a new device now means writing gen_<device>_ngspice.py, not another ~100-line copy of
this file too.

  Single render at a knob setting:
    python3 tools/render_ngspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \\
        --module gen_ocd_ngspice --probe-node OUT \\
        in.wav out.wav --knob Gain=0.8 --knob Tone=0.3

  Parametric sweep (the capture driver -- cartesian product of the grids):
    python3 tools/render_ngspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \\
        --module gen_ocd_ngspice --probe-node OUT \\
        in.wav --grid Gain=0,0.5,1 Tone=0,1 Volume=1.0 --outdir caps/
  -> caps/cap_0000.wav ... + caps/manifest.jsonl (one {file, knobs:{...}} per line) +
     caps/mapping.csv (filename,Knob1,Knob2,... for gen_dataset_from_captures.py's
     --mapping-csv -- sidesteps that pipeline's filename-token convention entirely, built for
     named real captures like "G5 T3.wav", not synthetic cap_NNNN.wav files, and its
     digit-scale dial-position-vs-percentage ambiguity, since our knob values are already
     known exactly. Only OK (converged) renders are included).

Uses ngspice_spicelib.py directly (load_input/render_grid/render_one -- spicelib driving
ngspice via SimRunner/RawRead), NOT render_backends.py's NgspiceBackend adapter: that adapter
returns raw audio arrays for preflight.py/find_saturation_point.py's own metric computation, a
different contract than what this file needs (capture WAV files on disk at deterministic
names, plus a manifest) -- forcing both needs through one abstraction would have been more
awkward than sharing the lower-level primitives both already use.

--absolute: use the input file's own sample values directly as volts (V0dBFS=1V convention),
with NO peak rescaling -- for a file already built by tools/build_excitation.py or
tools/prepare_excitation.py --backend ngspice-deck, whose different segments intentionally sit
at different absolute drive levels (a real-playing clip at --realistic-peak, sweeps at each
--sweep-peaks value). --vin is ignored when this is set (rescaling by the file's own single
peak would flatten that deliberate multi-level structure). See ngspice_spicelib.py's
load_input() docstring.

--probe-node and --ok-max-peak are device-specific (the per-device files this replaces used
different values: 'spk'/50 for OCD and BD-2, 'nspout'/200 for the JC-120, 'OUT'/50 for OD-3 and
HM-2) -- pass them explicitly rather than relying on a default that happened to fit one device.
"""
import argparse
import csv
import importlib
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ngspice_spicelib import load_input, render_grid, render_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pedal-dir', required=True, help='directory containing --module, added to sys.path')
    ap.add_argument('--module', required=True, help='module exposing build_deck/KNOB_NAMES, e.g. gen_ocd_ngspice')
    ap.add_argument('--probe-node', required=True, help="node/tap to render and measure (e.g. OUT, spk, nspout)")
    ap.add_argument('--ok-max-peak', type=float, default=50.0,
                     help='a render whose peak exceeds this is treated as diverged/FAILED, '
                          'not a real result (default 50V; raise for a device with a genuinely '
                          'high native swing, e.g. an amp\'s power-amp stage)')
    ap.add_argument('infile'); ap.add_argument('outfile', nargs='?')
    ap.add_argument('--knob', action='append', default=[], help='NAME=VAL (single render)')
    ap.add_argument('--grid', nargs='+', default=[], help='NAME=v1,v2,... (sweep)')
    ap.add_argument('--outdir')
    ap.add_argument('--vin', type=float, default=0.15)
    ap.add_argument('--absolute', action='store_true',
                     help="use the input file's own sample values as volts directly (V0dBFS=1V "
                          "convention, e.g. tools/build_excitation.py output) -- ignores --vin")
    ap.add_argument('--maxstep', type=float, default=3e-6)
    ap.add_argument('--tmp', default=None, help='default: /tmp/render_ngspice_deck_<module>')
    ap.add_argument('--parallel-sims', type=int, default=8, help='concurrent ngspice processes for --grid')
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.pedal_dir))
    mod = importlib.import_module(a.module)
    build_deck, KNOB_NAMES = mod.build_deck, mod.KNOB_NAMES

    def parse_knob(s):
        k, v = s.split('=', 1)
        if k not in KNOB_NAMES: sys.exit(f"unknown knob {k!r}; valid: {KNOB_NAMES}")
        return k, v

    tmp = a.tmp or f"/tmp/render_ngspice_deck_{a.module}"
    os.makedirs(tmp, exist_ok=True)
    sr, t, input_src = load_input(a.infile, None if a.absolute else a.vin, tmp, src_name='input.src')

    if a.grid:
        assert a.outdir, "--grid needs --outdir"
        os.makedirs(a.outdir, exist_ok=True)
        axes = [parse_knob(g) for g in a.grid]
        names = [k for k, _ in axes]
        grids = [[float(x) for x in v.split(',')] for _, v in axes]
        combos = list(itertools.product(*grids))
        print(f"sweep: {names} -> {len(combos)} captures (parallel_sims={a.parallel_sims})")
        jobs = []
        knobs_by_file = {}
        for i, combo in enumerate(combos):
            knobs = dict(zip(names, combo))
            outfile = os.path.join(a.outdir, f"cap_{i:04d}.wav")
            jobs.append((knobs, outfile))
            knobs_by_file[outfile] = knobs
        results = render_grid(build_deck, jobs, probe_node=a.probe_node, sr=sr, t=t, input_src=input_src,
                               tmp=tmp, maxstep=a.maxstep, parallel_sims=a.parallel_sims)
        man = open(os.path.join(a.outdir, 'manifest.jsonl'), 'w')
        for outfile, pk in results.items():
            knobs = knobs_by_file[outfile]
            f = os.path.basename(outfile)
            ok = bool(pk is not None and pk < a.ok_max_peak)
            man.write(json.dumps({"file": f, "knobs": knobs,
                                  "peak": None if pk is None else round(float(pk), 3),
                                  "ok": ok}) + "\n")
            print(f"  {f}: {knobs}  {'peak=%.2f' % pk if pk else 'FAILED'}  {'ok' if ok else 'CHECK'}")
        man.close()
        print(f"wrote {a.outdir}/manifest.jsonl")
        csv_path = os.path.join(a.outdir, 'mapping.csv')
        with open(csv_path, 'w', newline='') as cf:
            w = csv.DictWriter(cf, fieldnames=['filename'] + names)
            w.writeheader()
            for outfile, pk in results.items():
                if not (pk is not None and pk < a.ok_max_peak):
                    continue
                row = {'filename': os.path.basename(outfile)}
                row.update(knobs_by_file[outfile])
                w.writerow(row)
        print(f"wrote {csv_path}")
    else:
        assert a.outfile, "single render needs an outfile"
        knobs = {k: float(v) for k, v in (parse_knob(k) for k in a.knob)}
        pk = render_one(build_deck, knobs, a.outfile, a.probe_node, sr, t, input_src, tmp, maxstep=a.maxstep)
        pks = f"peak={pk:.3f}V" if pk is not None else "peak=--"
        print(f"  {a.outfile}: knobs={knobs or 'defaults'} "
              f"{pks} {'OK' if pk and pk < a.ok_max_peak else 'FAILED (did not converge)'}")


if __name__ == '__main__':
    main()
