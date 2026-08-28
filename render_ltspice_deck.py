#!/usr/bin/env python3
"""Render audio through a hand-written LTspice deck, with its knobs as SWEPT PARAMETERS --
LTspice counterpart of render_ngspice_deck.py, for a device whose ngspice-deck version can't
converge on real playing content at all (see ltspice_spicelib.py's own docstring: a
razor-steep tanh-bounded op-amp B-source is a genuine Newton-solver dead end in ngspice,
independent of timestep -- confirmed on the Fulltone OCD, 70/70 renders across the full knob
grid timing out at every maxstep from 3e-6 down to 3e-8).

  Single render at a knob setting:
    python3 render_ltspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \\
        --module gen_ocd_ltspice --tap spk \\
        in.wav out.wav --knob Gain=0.8 --knob Tone=0.3

  Parametric sweep (the capture driver -- cartesian product of the grids):
    python3 render_ltspice_deck.py --pedal-dir ~/work/parametric-devices/pedals \\
        --module gen_ocd_ltspice --tap spk \\
        in.wav --grid Gain=0,0.5,1 Tone=0,1 Volume=1.0 --outdir caps/
  -> caps/cap_0000.wav ... + caps/manifest.jsonl (one {file, knobs:{...}} per line) +
     caps/mapping.csv (filename,Knob1,Knob2,... for gen_dataset_from_captures.py's
     --mapping-csv). Only OK (converged) renders are included.

--absolute: use the input file's own sample values directly as volts (V0dBFS=1V convention),
with NO peak rescaling -- for a file already built by build_excitation.py or
prepare_excitation.py --backend ltspice-deck. --vin is ignored when this is set. See
ltspice_spicelib.py's load_input() docstring.

--tap and --ok-max-peak are device-specific, same convention as render_ngspice_deck.py's
--probe-node/--ok-max-peak -- pass them explicitly rather than relying on a default that
happened to fit one device.
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
from ltspice_spicelib import load_input, render_grid, render_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pedal-dir', required=True, help='directory containing --module, added to sys.path')
    ap.add_argument('--module', required=True, help='module exposing build_deck/KNOB_NAMES, e.g. gen_ocd_ltspice')
    ap.add_argument('--tap', required=True, help="node to render and measure (e.g. spk)")
    ap.add_argument('--ok-max-peak', type=float, default=50.0,
                     help='a render whose peak exceeds this is treated as diverged/FAILED, '
                          'not a real result (default 50V; raise for a device with a genuinely '
                          'high native swing)')
    ap.add_argument('infile'); ap.add_argument('outfile', nargs='?')
    ap.add_argument('--knob', action='append', default=[], help='NAME=VAL (single render)')
    ap.add_argument('--grid', nargs='+', default=[], help='NAME=v1,v2,... (sweep)')
    ap.add_argument('--outdir')
    ap.add_argument('--vin', type=float, default=0.15)
    ap.add_argument('--absolute', action='store_true',
                     help="use the input file's own sample values as volts directly (V0dBFS=1V "
                          "convention, e.g. build_excitation.py output) -- ignores --vin")
    ap.add_argument('--maxstep', type=float, default=3e-6)
    ap.add_argument('--out-scale', type=float, default=0.05,
                     help="LTspice .wave output is +/-1V-PCM-bounded -- the deck scales the tap "
                          "down by this factor before writing, and render_grid divides it back "
                          "out. Raise it (toward 1.0) only if you know the tap never exceeds "
                          "1/out_scale volts; lower it for a device with a higher native swing.")
    ap.add_argument('--tmp', default=None, help='default: /tmp/render_ltspice_deck_<module>')
    ap.add_argument('--parallel-sims', type=int, default=8, help='concurrent LTspice processes for --grid')
    ap.add_argument('--timeout', type=float, default=None,
                     help='per-render subprocess timeout, seconds (default: scales with the '
                          'input duration -- see ltspice_spicelib.py render_grid\'s own '
                          'docstring for the formula/reasoning). A flat number here is fragile: '
                          'a value tuned for grid_adequacy\'s short 8s probes silently kills '
                          'every job partway through a full 60s excitation capture, which LOOKS '
                          'exactly like a genuine convergence failure (every job escalates '
                          'through the whole maxstep ladder and fails again) rather than the '
                          'timeout-too-short bug it actually is. Only pass this explicitly to '
                          'override the duration-based default, e.g. for a device measured to '
                          'need a different ratio.')
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.pedal_dir))
    mod = importlib.import_module(a.module)
    build_deck, KNOB_NAMES = mod.build_deck, mod.KNOB_NAMES

    def parse_knob(s):
        k, v = s.split('=', 1)
        if k not in KNOB_NAMES: sys.exit(f"unknown knob {k!r}; valid: {KNOB_NAMES}")
        return k, v

    tmp = a.tmp or f"/tmp/render_ltspice_deck_{a.module}"
    os.makedirs(tmp, exist_ok=True)
    sr, dur_s, wav_path, in_scale = load_input(a.infile, None if a.absolute else a.vin, tmp,
                                                src_name='input.wav')

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
        results = render_grid(build_deck, jobs, tap=a.tap, sr=sr, dur_s=dur_s, wav_path=wav_path,
                               in_scale=in_scale, tmp=tmp, maxstep=a.maxstep,
                               parallel_sims=a.parallel_sims, out_scale=a.out_scale,
                               timeout=a.timeout)
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
        pk = render_one(build_deck, knobs, a.outfile, a.tap, sr, dur_s, wav_path, in_scale, tmp,
                         maxstep=a.maxstep, out_scale=a.out_scale, timeout=a.timeout)
        pks = f"peak={pk:.3f}V" if pk is not None else "peak=--"
        print(f"  {a.outfile}: knobs={knobs or 'defaults'} "
              f"{pks} {'OK' if pk and pk < a.ok_max_peak else 'FAILED (did not converge)'}")


if __name__ == '__main__':
    main()
