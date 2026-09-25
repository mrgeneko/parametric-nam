#!/usr/bin/env python3
"""General .schx -> LTspice netlist translator.

LTspice counterpart of ../ngspice/schx_to_ngspice.py, built for exactly one reason: the Ampeg
SVT Full combined preamp+power-amp (521 components, 13 triodes + 6 pentodes + 1 center-tapped
OT) renders so slowly through LiveSPICE that a full oversample-measurement sweep projected to
~40 hours for 11 settings on one machine. LTspice's adaptive solver is widely regarded as fast
on large circuits and might render this specific circuit meaningfully faster than either
LiveSPICE or ngspice -- but hand-porting all 521 components (including the fitted Koren/
DempwolfZolzer triode models and the custom Pentode model) into LTspice syntax by hand would be
its own multi-hour, error-prone undertaking. This module exists to do that port automatically
instead, by RETARGETING schx_to_ngspice.py's already-working parsing/translation logic at
LTspice's own netlist syntax, not by re-deriving the circuit math from scratch.

REUSE, NOT REINVENTION. Every piece of this module that is dialect-INDEPENDENT (unit/quantity
parsing, pot-taper math, node-name resolution, the DempwolfZolzer triode's B-source equations,
the op-amp macromodel, the CenterTapTransformer's E/F-controlled-source and mutual-inductance
math, the diode/BJT/JFET .model parameter strings) is imported directly from schx_to_ngspice.py
(see `NG` below) rather than copy-pasted -- so a future fix to that shared math (e.g. a triode
parameter bug) fixes both backends at once instead of silently diverging. Only the parts that
are GENUINELY LTspice-specific get their own code here:

  * Koren triode / Pentode: ngspice's `(cond)?a:b` ternary is not documented LTspice B-source
    syntax (LTspice's own B-source reference lists `if(x,y,z)` -- "y if x>0.5, else z" -- with no
    mention of a ternary operator); rewritten below using `if(...)` instead. The underlying Koren
    plate-current math is otherwise identical to schx_to_ngspice.py's own triode_subckt_koren/
    pentode_subckt (same softplus/pow calls, reused from NG verbatim).
  * Input/output I/O: ngspice reads an arbitrary-length PWL/XSPICE-filesource file and writes
    ascii `wrdata`; LTspice's own `wavefile=` input and `.wave` output are BOTH PCM-only and
    +/-1V-hard-clip-bounded (see ../ltspice_spicelib.py's docstring) -- this module receives an
    ALREADY-PCM-prepared wav (peak-safe, pre-divided by `in_scale`) and restores real volts via
    an E-source, exactly the pattern this fleet's hand-written gen_*_ltspice.py device modules
    already use (see e.g. parametric-devices/pedals/gen_ocd_ltspice.py).
  * `.tran`/`.options`/`.save`/`.wave` deck skeleton: LTspice has no separate `.control` block
    to append commands to after the fact -- these lines are part of the deck text itself, same
    constraint ../ltspice_spicelib.py's own docstring documents for every hand-written LTspice
    deck in this fleet.

One functional gap, ported faithfully rather than "fixed" here: schx_to_ngspice.py's own
Triode/Pentode subckts do not consume the .schx's Cgp/Cgk/Cpk interelectrode-capacitance
attributes (LiveSPICE's own solver honors them natively; the ngspice/LTspice B-source ports do
not) -- confirmed by reading schx_to_ngspice.py in full and grepping its git history for
Cgp/Cgk/Cpk (no hits). This translator matches that EXISTING ngspice behavior rather than adding
new physics ngspice itself doesn't have, per this module's own reuse-not-reinvent charter; fixing
that gap (if it turns out to matter for convergence) belongs in the shared math, not duplicated
here first.

VALIDATION STATUS (2026-09-24). Checked against schx_to_ngspice.py's own translation of the
SAME netlist (both derive from the identical .schx-parsing/pot-baking logic, so a close match
confirms the LTspice-specific retargeting itself, not the shared circuit math) on two real
devices from this fleet covering every component type Ampeg SVT Full.schx uses:

  * Tweed 5F6-A Preamp (Triode-only, DempwolfZolzer model, 4 tubes): ESR=0.0001, corr=0.999993
    against ngspice on a 0.3 s 440 Hz probe -- essentially exact.
  * Tweed 5F6-A Power Amp (Pentode + CenterTapTransformer + Inductor, no global NFB, 4 tubes):
    ESR=0.0023, corr=0.9997 against ngspice on a 0.3 s 100 Hz probe -- strong match.

Both also track the independent livespice_cli oracle's own render of the same circuit/input
reasonably well (correlation >0.99 on the Preamp once a short, already-documented onset-
settling transient is excluded -- see the OCD .ic-mismatch precedent in ltspice_spicelib.py --
though the two solvers converge to a different DC-bias-charging TRAJECTORY on that transient,
a pre-existing cross-backend difference this port inherits rather than introduces).

NOT YET TRUSTWORTHY ON THE ACTUAL PAYOFF CIRCUIT. A 2 s real-audio probe of the real, committed
Ampeg SVT Full.schx (521 components, 13 triodes + 6 pentodes + 1 CenterTapTransformer with a
global NFB loop) renders in LTspice FASTER than the livespice-cli oracle in raw wall-clock terms
(146-159 s vs 219 s on a machine also running unrelated background load -- roughly 1.4-1.5x, not
the order-of-magnitude win hoped for) but the OUTPUT ITSELF is not numerically trustworthy: it
develops large spurious voltage spikes (peaks 200-790+ V, correlation 0.01-0.22 against the
livespice-cli oracle's own render of the identical clip) that do not appear in LiveSPICE's
output. Tried and RULED OUT as the fix: a 3x finer --maxstep (1e-6 vs 3e-6 -- 2.8x slower, spikes
unchanged) and --koren triodes (eliminates the "Heightened Def Con" gmin-stepping thrash during
the initial .op search, same wall-clock, but the mid-transient spikes persist regardless). This
points at the same category of instability already documented for THIS circuit's global NFB
loop / undamped Mid Range LC resonance / ideal-transformer LF behavior in
Ampeg SVT Power Amp.backends.toml (ngspice's own translation of the same math is likewise only
"partial" there) rather than a defect specific to this LTspice port -- the B-source math is
shared. Do not trust an LTspice render of Ampeg SVT Full.schx for anything real until this is
root-caused; the smaller-circuit validation above supports trusting the TRANSLATOR (the syntax
retargeting this module exists to do), not yet this specific circuit's numerical behavior under
it.

Not a real-time backend, same as schx_to_ngspice.py: offline dataset/probe generation only.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Reuse schx_to_ngspice.py's dialect-independent parsing/translation logic directly, rather
# than re-deriving or copy-pasting it -- see module docstring. Mirrors the exact sys.path
# pattern gen_dataset_from_schx.py's own _run_ngspice() already uses to reach that module.
_NGDIR = str(Path(__file__).resolve().parent.parent / "ngspice")
if _NGDIR not in sys.path:
    sys.path.insert(0, _NGDIR)
import schx_to_ngspice as NG  # noqa: E402

qty, sp, node, terms, adjust_wipe, softplus = NG.qty, NG.sp, NG.node, NG.terms, NG.adjust_wipe, NG.softplus


def _bsrc_normalize(lines):
    """Reused ngspice B-source lines spell the assignment ' I = expr' / ' V = expr' (ngspice
    tolerates the spaces around '='); LTspice's own B-device reference always shows it
    unspaced (`I=<expression>`). Never seen LTspice reject the spaced form in practice, but
    there is no upside to relying on it -- normalize defensively rather than find out on a
    521-component netlist mid-render."""
    return [re.sub(r'\b([IV]) = ', r'\1=', ln) for ln in lines]


# --------------------------------------------------------------------------
# tube subckts needing LTspice-safe conditionals (ternary -> if(); see module docstring).
# Math identical to schx_to_ngspice.py's own triode_subckt_koren/pentode_subckt -- only the
# conditional syntax differs, so these stay side-by-side with NG's originals for an easy diff.
# --------------------------------------------------------------------------
def triode_subckt_koren_ltspice(name, p):
    Mu, Ex, Kg, Kp, Kvb = (qty(p['Mu']), qty(p['Ex']), qty(p['Kg']),
                           qty(p['Kp']), qty(p['Kvb']))
    vgk, vpk = 'V(G,K)', 'V(P,K)'
    E1 = '((%s)/%s*%s)' % (vpk, sp(Kp),
                           softplus('%s*(1/%s+(%s)/sqrt(%s+(%s)*(%s)))' % (
                               sp(Kp), sp(Mu), vgk, sp(Kvb), vpk, vpk)))
    return ['.subckt %s P G K' % name,
            'Bp P K I=if(%s>0,2*pow(%s,%s)/%s,0)' % (E1, E1, sp(Ex), sp(Kg)),
            'Dgk G K DGRID', 'CGP G P 2.4p', 'CGK G K 2.3p', '.ends']


def pentode_subckt_ltspice(name, p):
    Mu, Ex, Kg1, Kg2, Kp, Kvb = (qty(p['Mu']), qty(p['Ex']), qty(p['Kg1']),
                                 qty(p['Kg2']), qty(p['Kp']), qty(p['Kvb']))
    vpk, vgk, vg2k = 'V(P,K)', 'V(G,K)', 'V(G2,K)'
    E1 = '((%s)/%s*%s)' % (vpk, sp(Kp),
                           softplus('%s*(1/%s+(%s)/sqrt(%s+(%s)*(%s)))' % (
                               sp(Kp), sp(Mu), vgk, sp(Kvb), vg2k, vg2k)))
    ikoren = 'if(%s>0,pow(%s,%s),0)' % (E1, E1, sp(Ex))
    ip = 'Bp P K I=if(%s>0,%s/%s*atan((%s)/%s),0)' % (vpk, ikoren, sp(Kg1), vpk, sp(Kvb))
    ig2 = 'Bg2 G2 K I=%s/%s' % (ikoren, sp(Kg2))
    return ['.subckt %s P G2 G K' % name, ip, ig2, '.ends']


# --------------------------------------------------------------------------
# main translation
# --------------------------------------------------------------------------
def translate(netlist, pots=None, wav_path='input.wav', in_scale=1.0, dur=0.5, out_wav='out.wav',
              tap=None, method=None, koren=False, ot_damp='47k', ot_snub='10n', nfb_comp=None,
              out_scale=0.05, maxstep=3e-6, uic=False, extra=None, conv=None):
    """netlist: the JSON from `livespice_cli --circuit X.schx --netlist X.json` (identical input
    schx_to_ngspice.translate() consumes).

    wav_path/in_scale: an ALREADY-PCM-prepared excitation, as produced by
    ltspice_spicelib.load_input() -- the file's own samples are real-input-volts / in_scale (see
    that module's docstring for why: LTspice's wavefile= is +/-1V-PCM-hard-clip-bounded). Real
    volts are restored via an E-source, same pattern every gen_*_ltspice.py device module in
    this fleet already uses for its own hand-written deck.

    tap: probe node override. None (default) uses the .schx's own Speaker component, matching
    schx_to_ngspice.py's convention (including this fleet's non-loading Impedance="∞ Ω"
    internal-node-probing convention -- a Speaker tap IS just another schx node name here, same
    as it is for the livespice/ngspice backends)."""
    pots = pots or {}
    conv = conv or {}
    comps = netlist['components']
    lines = ['* schx -> LTspice (%s triodes, Koren pentodes, baked pots)'
             % ('Koren' if koren else 'DempwolfZolzer')]

    subckts, sub_defs = {}, []
    need_dgrid = [False]

    def tube_sub(kind, p):
        sig = (kind, tuple(sorted((k, p[k]) for k in p if k not in ('Model',))))
        if sig not in subckts:
            nm = '%s%d' % (kind, len(subckts))
            subckts[sig] = nm
            if kind == 'PEN':
                sub_defs.extend(pentode_subckt_ltspice(nm, p))
            elif koren:
                sub_defs.extend(triode_subckt_koren_ltspice(nm, p)); need_dgrid[0] = True
            else:
                # DempwolfZolzer: no ternary anywhere in NG.triode_subckt's output -- reused
                # verbatim (space-normalized for LTspice's unspaced I=/V= convention).
                sub_defs.extend(_bsrc_normalize(NG.triode_subckt(nm, p)))
        return subckts[sig]

    models, model_defs = {}, []

    def mdl(prefix, mstr):
        if mstr not in models:
            models[mstr] = '%s%d' % (prefix, len(models))
            model_defs.append('.model %s %s' % (models[mstr], mstr))
        return models[mstr]

    opamps, opamp_defs = {}, []

    def opamp_sub(p, rails):
        Aol = qty(p.get('Aol', '1e6')) or 1e6
        GBP = qty(p.get('GBP', '10e6')) or 10e6
        Rout = qty(p.get('Rout', '100')) or 100.0
        sig = (Aol, GBP, Rout, rails)
        if sig not in opamps:
            opamps[sig] = 'OA%d' % len(opamps)
            # opamp_subckt's only B-source line is 'Bo ob 0 V = min(max(...))' -- no ternary,
            # reused verbatim (space-normalized), same as the DempwolfZolzer triode above.
            opamp_defs.extend(_bsrc_normalize(NG.opamp_subckt(opamps[sig], Aol, GBP, Rout, rails)))
        return opamps[sig]

    body, out_node = [], None
    for c in comps:
        ty, p = c['type'], c['params']
        nm = re.sub(r'\s+', '_', c['name'])
        t = terms(c)
        if ty in ('Ground', 'Label', 'NamedWire'):
            continue
        elif ty == 'Resistor':
            n = list(t.values()); body.append('R%s %s %s %s' % (nm, n[0], n[1], sp(qty(c['value']))))
        elif ty == 'Capacitor':
            n = list(t.values()); body.append('C%s %s %s %s' % (nm, n[0], n[1], sp(qty(c['value']))))
        elif ty == 'Inductor':
            n = list(t.values())
            lval = c.get('value') or p.get('Inductance')
            body.append('L%s %s %s %s' % (nm, n[0], n[1], sp(qty(lval))))
        elif ty == 'Rail':
            body.append('V%s %s 0 DC %s' % (nm, list(t.values())[0], sp(qty(p['Voltage']))))
        elif ty == 'Input':
            # LTspice wavefile= input: `wav_path`'s own samples are real-input-volts/in_scale
            # (ltspice_spicelib.load_input's PCM-safety convention). Restore real volts via an
            # E-source (matches every gen_*_ltspice.py device module's own Einscale pattern),
            # then apply the .schx's own V0dBFS on top -- same two-stage gain schx_to_ngspice.py
            # applies (its WAV is likewise already at real-input-volts; V0dBFS is a further,
            # normally-unity, per-circuit calibration knob, not a substitute for it).
            src, cat = t['Anode'], t['Cathode']
            g = qty(p.get('V0dBFS', '1')) or 1.0
            body += ['V%s_raw n%s_raw 0 wavefile="%s" chan=0' % (nm, nm, os.path.abspath(wav_path)),
                     'E%s %s %s n%s_raw 0 %s' % (nm, src, cat, nm, sp(g * in_scale))]
        elif ty == 'Speaker':
            if out_node is None:  # first Speaker wins; --tap overrides below
                out_node = t['Anode']
        elif ty == 'Potentiometer' or c.get('isPot'):
            R = qty(p['Resistance']); wipe = pots.get(nm, qty(p.get('Wipe', '0.5')))
            A, W, K = t.get('Anode'), t.get('Wiper'), t.get('Cathode')
            P = adjust_wipe(wipe, p.get('Sweep', 'Linear'))
            raw, rwk = R * (1 - P), R * P
            if A != W: body.append('R%s_aw %s %s %s' % (nm, A, W, sp(max(raw, 1.0))))
            if W != K: body.append('R%s_wk %s %s %s' % (nm, W, K, sp(max(rwk, 1.0))))
        elif ty == 'Diode':
            body.append('D%s %s %s %s' % (nm, t['Anode'], t['Cathode'], mdl('DM', NG.diode_model(p, conv))))
        elif ty == 'BipolarJunctionTransistor':
            body.append('Q%s %s %s %s %s' % (nm, t['C'], t['B'], t['E'], mdl('QM', NG.bjt_model(p, conv))))
        elif ty == 'JunctionFieldEffectTransistor':
            body.append('J%s %s %s %s %s' % (nm, t['D'], t['G'], t['S'], mdl('JM', NG.jfet_model(p, conv))))
        elif ty in ('IdealOpAmp', 'OpAmp'):
            vp, vn = t.get('Vcc+'), t.get('Vcc-')
            rails = bool(vp and vn and not vp.startswith('_') and not vn.startswith('_'))
            s = opamp_sub(p, rails)
            pins = [t['+'], t['-'], t['Out']] + ([vp, vn] if rails else [])
            body.append('X%s %s %s' % (nm, ' '.join(pins), s))
        elif ty == 'Triode':
            s = tube_sub('TRI', p); body.append('X%s %s %s %s %s' % (nm, t['P'], t['G'], t['K'], s))
        elif ty == 'Pentode':
            s = tube_sub('PEN', p); body.append('X%s %s %s %s %s %s' % (nm, t['P'], t['G2'], t['G'], t['K'], s))
        elif ty == 'CenterTapTransformer':
            # Identical E/F (or opt-in mutual-inductance) SPICE syntax in both dialects --
            # reused directly, zero duplication. See NG.centertap_lines's own docstring/history.
            body += NG.centertap_lines(nm, c, t, comps, conv, ot_damp, ot_snub)
        else:
            print('WARN: unhandled component %s (%s)' % (nm, ty), file=sys.stderr)

    if tap:
        out_node = tap
    if out_node is None:
        sys.exit('no Speaker component (or --tap) found to probe')

    if need_dgrid[0]:
        sub_defs.insert(0, '.model DGRID D(IS=1e-12 RS=2000 N=1.0)')
    if nfb_comp and '=' in nfb_comp:
        n, cval_ = nfb_comp.split('=')
        body.append('Cnfbcomp %s 0 %s' % (node(n), cval_))
    if extra:
        for ln in extra.split(';'):
            if ln.strip():
                body.append(ln.strip())

    # LTspice's .wave output is +/-1V-PCM-bounded (a real >1V tap silently hard-clips at
    # exactly 1.0) -- scale it down before the .wave line, same convention as every
    # gen_*_ltspice.py device module's own Eoutscale (see ltspice_spicelib.py's docstring).
    body.append('Eoutscale spkout 0 %s 0 %s' % (out_node, sp(out_scale)))

    opts = '.options reltol=1e-3 abstol=5e-9 gmin=1e-9'
    if method:
        opts += ' method=%s' % method

    lines += sub_defs + opamp_defs + model_defs + [''] + body + [
        '', opts,
        # Without .save, LTspice records every node voltage/device current at every adaptive
        # timestep into a .raw beside the .wave -- measured 12-13 GB PER RENDER on a comparable
        # circuit (see ltspice_spicelib.ensure_save's own docstring). Saving only the tap
        # changes nothing about the simulation.
        '.save V(spkout)',
        '.tran 0 %s 0 %s%s' % (sp(dur), sp(maxstep), ' uic' if uic else ''),
        '.wave "%s" 24 48000 V(spkout)' % os.path.abspath(out_wav),
        '.end', '']
    return '\n'.join(lines)


def build_deck_factory(netlist, pots_base=None, koren=False, ot_damp='47k', ot_snub='10n',
                       nfb_comp=None, extra=None, conv=None, param_map=None, tap=None, uic=False):
    """Returns a `build_deck(wav_path, dur_s, maxstep, out_wav, knobs=None, tap=..., out_scale=...,
    in_scale=..., method=None) -> str` closure matching ltspice_spicelib.render_grid's own
    contract (see gen_ocd_ltspice.py for the hand-written-deck precedent this mirrors) --
    letting a generic .schx circuit reuse render_grid's existing parallelism/retry-escalation/
    timeout machinery instead of a third, one-off invocation path. `knobs` (from a render_grid
    job) is a dict of {pot Name: wipe 0..1} -- param_map, if given, maps a caller's own
    knob-name vocabulary to the .schx's actual Potentiometer Names, same convention
    render_backends.NgspiceSchxBackend already uses."""
    pmap = param_map or {}

    def build_deck(wav_path, dur_s, maxstep, out_wav, knobs=None, tap=tap, out_scale=0.05,
                   in_scale=1.0, method=None):
        pots = dict(pots_base or {})
        if knobs:
            pots.update({pmap.get(k, k): float(v) for k, v in knobs.items()})
        return translate(netlist, pots=pots, wav_path=wav_path, in_scale=in_scale, dur=dur_s,
                         out_wav=out_wav, tap=tap, method=method, koren=koren, ot_damp=ot_damp,
                         ot_snub=ot_snub, nfb_comp=nfb_comp, out_scale=out_scale, maxstep=maxstep,
                         uic=uic, extra=extra, conv=conv)
    return build_deck


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('netlist'); ap.add_argument('-o', '--out', default='circuit.net')
    ap.add_argument('--pots', default='', help='name=wipe,name=wipe')
    ap.add_argument('--wav', default='input.wav',
                    help='ALREADY-PCM-prepared excitation (see ltspice_spicelib.load_input) -- '
                         'its samples are real-input-volts / --in-scale')
    ap.add_argument('--in-scale', type=float, default=1.0,
                    help='divisor applied when --wav was written (1.0 if it is already real volts)')
    ap.add_argument('--dur', type=float, default=0.5)
    ap.add_argument('--out-wav', default='out.wav', help='where the .wave line writes rendered audio')
    ap.add_argument('--tap', default=None, help='probe node override (default: the .schx Speaker)')
    ap.add_argument('--maxstep', type=float, default=3e-6)
    ap.add_argument('--out-scale', type=float, default=0.05,
                    help='.wave output is +/-1V-PCM-bounded; scales the tap down before writing '
                         '(render_grid divides it back out) -- see ltspice_spicelib.py')
    ap.add_argument('--koren', action='store_true',
                    help='Koren triode model (softer/convergence mode) instead of exact DempwolfZolzer')
    ap.add_argument('--ot-damp', default='47k'); ap.add_argument('--ot-snub', default='10n')
    ap.add_argument('--nfb-comp', default=None, help='NFB compensation cap, NODE=value (e.g. nNFB=1n)')
    ap.add_argument('--method', default=None, help='LTspice integration method override (e.g. gear)')
    ap.add_argument('--uic', action='store_true',
                    help='skip the .op search, start from 0V everywhere (rarely needed for a '
                         'self-biasing tube circuit; see gen_ocd_ltspice.py for when a circuit '
                         'genuinely needs this)')
    ap.add_argument('--extra', default=None, help='inject extra SPICE lines, ;-separated')
    ap.add_argument('--conv', default='', help='device convergence/model overrides, key=val,... '
                    '(same keys as schx_to_ngspice.py --conv, plus xfmr_model=mutual)')
    a = ap.parse_args()
    pots = dict((kv.split('=')[0], float(kv.split('=')[1])) for kv in a.pots.split(',') if '=' in kv)
    conv = dict(kv.split('=', 1) for kv in a.conv.split(',') if '=' in kv)
    nl = json.load(open(a.netlist))
    open(a.out, 'w').write(translate(nl, pots, a.wav, a.in_scale, a.dur, a.out_wav, tap=a.tap,
                                     method=a.method, koren=a.koren, ot_damp=a.ot_damp,
                                     ot_snub=a.ot_snub, nfb_comp=a.nfb_comp, out_scale=a.out_scale,
                                     maxstep=a.maxstep, uic=a.uic, extra=a.extra, conv=conv))
    print('wrote', a.out)
