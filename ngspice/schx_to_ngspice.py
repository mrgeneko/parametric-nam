#!/usr/bin/env python3
"""General .schx -> ngspice netlist translator.

Consumes the JSON from `livespice_cli --circuit X.schx --netlist X.json`
(authoritative components + node connectivity + full device params) and emits
an ngspice `.cir`. Device models are ported from LiveSPICE's own equations:

  * Triode  (DempwolfZolzer): ig = Gg*softplus(Cg*Vgk)^Xi + Ig0
                              ik = -G*softplus(C*(Vpk/Mu+Vgk))^Gamma ; ip=-(ik+ig)
  * Pentode (Koren):          E1=Vpk/Kp*softplus(Kp*(1/Mu+Vgk/sqrt(Kvb+Vg2k^2)))
                              ip=(Vpk>0)?E1^Ex/Kg1*atan(Vpk/Kvb):0 ; ig2=E1^Ex/Kg2
  * Potentiometer: baked to two resistors at the wiper position.
  * CenterTapTransformer: coupled-inductor OT (approx; LiveSPICE's is ideal —
    see ngspice/README fidelity caveats).

Not a real-time backend: ngspice adaptive-timestep offline generation, for the
stiff/high-gain circuits where LiveSPICE's fixed step diverges.
"""
import json
import math
import os
import re
import sys


def adjust_wipe(x, sweep='Linear'):
    """Dial position -> resistance fraction, matching LiveSPICE VariableResistor.AdjustWipe.
    For a Logarithmic (audio) pot, dial 0.5 -> ~0.27, not 0.5."""
    x = min(max(float(x), 1e-3), 1.0 - 1e-3)
    if sweep in ('ReverseLinear', 'ReverseSigmoid', 'ReverseLogarithmic', 'ReverseAntiLogarithmic'):
        x = 1.0 - x
    e = math.exp(2); k = 1.8
    if sweep in ('Logarithmic', 'ReverseLogarithmic'):
        return (math.pow(e, x) - 1.0) / (e - 1.0)
    if sweep in ('AntiLogarithmic', 'ReverseAntiLogarithmic'):
        return 1.0 - (math.pow(e, 1.0 - x) - 1.0) / (e - 1.0)
    if sweep in ('Sigmoid', 'ReverseSigmoid'):
        return 1.0 / (1.0 + math.pow(x / (1.0 - x), -k))
    if sweep in ('AntiSigmoid', 'ReverseAntiSigmoid'):
        return 1.0 / (1.0 + math.pow(x / (1.0 - x), -1.0 / k))
    return x  # Linear / ReverseLinear

# --- quantity parsing: "68 kΩ" -> 68000.0, "22 nF" -> 22e-9, "500 mV" -> 0.5 ---
# Unit chars (Ω, µ, F, V, ...) vary by codepoint across sources, so parse the
# leading number + the FIRST suffix letter as a possible SI prefix; ignore units.
# NOTE: LiveSPICE uses M = mega (e.g. "1 MΩ"); we emit plain floats, so there is
# no SPICE M=milli ambiguity downstream.
def qty(s):
    if s is None:
        return None
    m = re.match(r'^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(\S*)', s.strip())
    if not m:
        return None
    val = float(m.group(1))
    suf = m.group(2)
    if suf:
        low = suf.lower()
        p = suf[0]
        if low.startswith('meg'):
            val *= 1e6
        elif p == 'p':
            val *= 1e-12
        elif p == 'n':
            val *= 1e-9
        elif p in ('u', 'µ', 'μ'):
            val *= 1e-6
        elif p == 'm':          # milli (M=mega handled above via 'meg' or capital)
            val *= 1e-3
        elif p == 'k':
            val *= 1e3
        elif p == 'M':
            val *= 1e6
        elif p == 'G':
            val *= 1e9
        # otherwise p is a unit char (Ω/V/A/F/H/s...) -> no scaling
    return val


def sp(x):
    """Format a float for ngspice (plain, avoids case-sensitive SI prefixes)."""
    return repr(float(x))


def node(n):
    return '0' if n in (None, 'GND') else n


def softplus(x):
    # numerically-stable ln(1+exp(x)) = max(x,0)+ln(1+exp(-abs(x)))
    return '(max(%s,0)+ln(1+exp(-abs(%s))))' % (x, x)


def terms(comp):
    """dict terminalName -> node (0 for ground)."""
    return {t['name']: node(t['node']) for t in comp['terminals']}


# --------------------------------------------------------------------------
# tube subcircuits (one per unique param signature)
# --------------------------------------------------------------------------
def triode_subckt(name, p):
    Mu, G, Gg, C, Cg, Xi, Gamma = (qty(p['Mu']), qty(p['G']), qty(p['Gg']),
                                   qty(p['C']), qty(p['Cg']), qty(p['Xi']), qty(p['Gamma']))
    Ig0 = qty(p.get('Ig0', '0'))
    vgk, vpk = 'V(G,K)', 'V(P,K)'
    ig = 'Bg G K I = %s*pow(%s/%s,%s)+%s' % (
        sp(Gg), softplus('%s*%s' % (sp(Cg), vgk)), sp(Cg), sp(Xi), sp(Ig0))
    ik = '%s*pow(%s/%s,%s)' % (sp(G), softplus('%s*((%s)/%s+%s)' % (sp(C), vpk, sp(Mu), vgk)), sp(C), sp(Gamma))
    # ip = -(ik+ig) = |ik| - ig ; Bp sources P->K
    ip = 'Bp P K I = %s-(%s*pow(%s/%s,%s)+%s)' % (
        ik, sp(Gg), softplus('%s*%s' % (sp(Cg), vgk)), sp(Cg), sp(Xi), sp(Ig0))
    return ['.subckt %s P G K' % name, ig, ip, '.ends']


def triode_subckt_koren(name, p):
    """Koren triode (convergence mode) — matches the prototype's proven-stable
    12AX7 subckt: Koren plate current + a grid diode + inter-electrode caps.
    Numerically softer than the exact DempwolfZolzer port, at a fidelity cost."""
    Mu, Ex, Kg, Kp, Kvb = (qty(p['Mu']), qty(p['Ex']), qty(p['Kg']),
                           qty(p['Kp']), qty(p['Kvb']))
    vgk, vpk = 'V(G,K)', 'V(P,K)'
    E1 = '((%s)/%s*%s)' % (vpk, sp(Kp),
                           softplus('%s*(1/%s+(%s)/sqrt(%s+(%s)*(%s)))' % (
                               sp(Kp), sp(Mu), vgk, sp(Kvb), vpk, vpk)))
    return ['.subckt %s P G K' % name,
            'Bp P K I = (%s>0)?2*pow(%s,%s)/%s:0' % (E1, E1, sp(Ex), sp(Kg)),
            'Dgk G K DGRID', 'CGP G P 2.4p', 'CGK G K 2.3p', '.ends']


def pentode_subckt(name, p):
    Mu, Ex, Kg1, Kg2, Kp, Kvb = (qty(p['Mu']), qty(p['Ex']), qty(p['Kg1']),
                                 qty(p['Kg2']), qty(p['Kp']), qty(p['Kvb']))
    vpk, vgk, vg2k = 'V(P,K)', 'V(G,K)', 'V(G2,K)'
    E1 = '((%s)/%s*%s)' % (vpk, sp(Kp),
                           softplus('%s*(1/%s+(%s)/sqrt(%s+(%s)*(%s)))' % (
                               sp(Kp), sp(Mu), vgk, sp(Kvb), vg2k, vg2k)))
    ikoren = '((%s>0)?pow(%s,%s):0)' % (E1, E1, sp(Ex))
    ip = 'Bp P K I = (%s>0)?%s/%s*atan((%s)/%s):0' % (vpk, ikoren, sp(Kg1), vpk, sp(Kvb))
    ig2 = 'Bg2 G2 K I = %s/%s' % (ikoren, sp(Kg2))
    return ['.subckt %s P G2 G K' % name, ip, ig2, '.ends']


# --------------------------------------------------------------------------
# main translation
# --------------------------------------------------------------------------
def translate(netlist, pots=None, input_pwl='input.pwl', dur=0.5, csv='out.csv',
              method='trap', koren=False, ot_damp='47k', ot_snub='10n', nfb_comp=None,
              input_mode='filesource', oversample=1, extra=None):
    pots = pots or {}
    comps = netlist['components']
    lines = ['* schx -> ngspice (%s triodes, Koren pentodes, baked pots)'
             % ('Koren' if koren else 'DempwolfZolzer')]

    # collect unique tube param signatures -> subckt name
    subckts, sub_defs = {}, []
    need_dgrid = [False]

    def tube_sub(kind, p):
        sig = (kind, tuple(sorted((k, p[k]) for k in p if k not in ('Model',))))
        if sig not in subckts:
            nm = '%s%d' % (kind, len(subckts))
            subckts[sig] = nm
            if kind == 'PEN':
                sub_defs.extend(pentode_subckt(nm, p))
            elif koren:
                sub_defs.extend(triode_subckt_koren(nm, p)); need_dgrid[0] = True
            else:
                sub_defs.extend(triode_subckt(nm, p))
        return subckts[sig]

    body, out_node = [], None
    for c in comps:
        ty, nm, p = c['type'], c['name'], c['params']
        t = terms(c)
        if ty in ('Ground', 'Label', 'NamedWire'):
            continue  # NamedWire = single-terminal net label; node already resolved
        elif ty == 'Resistor':
            n = list(t.values()); body.append('R%s %s %s %s' % (nm, n[0], n[1], sp(qty(c['value']))))
        elif ty == 'Capacitor':
            n = list(t.values()); body.append('C%s %s %s %s' % (nm, n[0], n[1], sp(qty(c['value']))))
        elif ty == 'Rail':
            body.append('V%s %s 0 DC %s' % (nm, list(t.values())[0], sp(qty(p['Voltage']))))
        elif ty == 'Input':
            # Input file is time,raw-sample pairs (sample in [-1,1]); the Input's
            # V0dBFS scales it to volts. filesource (default) reads the file at
            # sim time -> scales to any length. Inline PWL is a portable fallback
            # for short inputs.
            src, cat = t['Anode'], t['Cathode']
            g = qty(p.get('V0dBFS', '1')) or 1.0
            if input_mode == 'filesource':
                body += [
                    'Ain %%vd([%s %s]) fsrc_in' % (src, cat),
                    '.model fsrc_in filesource (file="%s" amploffset=[0] amplscale=[%s] '
                    'timeoffset=0 timescale=1 timerelative=false)' % (os.path.abspath(input_pwl), sp(g))]
            else:
                pairs = []
                with open(input_pwl) as f:
                    for ln in f:
                        c = ln.split()
                        if len(c) >= 2:
                            pairs.append('%s %s' % (c[0], sp(float(c[1]) * g)))
                out = ['Vin %s %s PWL(' % (src, cat)]
                for i in range(0, len(pairs), 8):
                    out.append('+ ' + ' '.join(pairs[i:i + 8]))
                out.append('+ )')
                body.extend(out)
        elif ty == 'Speaker':
            out_node = t['Anode']  # probe here
        elif ty == 'Potentiometer' or c.get('isPot'):
            R = qty(p['Resistance']); wipe = pots.get(nm, qty(p.get('Wipe', '0.5')))
            A, W, K = t.get('Anode'), t.get('Wiper'), t.get('Cathode')
            # Apply the pot taper (LiveSPICE AdjustWipe). The dial position 'wipe'
            # is NOT the resistance fraction for log/audio pots — e.g. a Logarithmic
            # pot at dial 0.5 sits at ~0.27, not 0.5. Baking linearly overdrives the
            # low/mid dial on gain pots (LeadPre/LeadPost) -> always-clipped.
            P = adjust_wipe(wipe, p.get('Sweep', 'Linear'))
            raw, rwk = R * (1 - P), R * P        # P=0 -> wiper at cathode (per LiveSPICE)
            # Floor at ~1 ohm, not 1m ohm: a near-zero pot section is a near-short
            # in the signal path that makes ngspice numerically stiff (a wipe=0/1
            # extreme could hang the solver). 1 ohm is tonally negligible.
            if A != W: body.append('R%s_aw %s %s %s' % (nm, A, W, sp(max(raw, 1.0))))
            if W != K: body.append('R%s_wk %s %s %s' % (nm, W, K, sp(max(rwk, 1.0))))
        elif ty == 'Triode':
            s = tube_sub('TRI', p); body.append('X%s %s %s %s %s' % (nm, t['P'], t['G'], t['K'], s))
        elif ty == 'Pentode':
            s = tube_sub('PEN', p); body.append('X%s %s %s %s %s %s' % (nm, t['P'], t['G2'], t['G'], t['K'], s))
        elif ty == 'CenterTapTransformer':
            # IDEAL center-tapped transformer via controlled sources — matches
            # LiveSPICE's exact model (no inductance to guess). LiveSPICE:
            #   Vp = 2*turns*V(SA,ST) = 2*turns*V(ST,SC)   (turns = Np/Ns_full)
            #   Ip*turns = Isa + Isc                        (ampere-turns)
            # Realized as: two VCVS set the half-secondary voltages from Vp, two
            # CCCS reflect each half-secondary current back into the primary.
            r = str(c['value']).split(':')
            turns = float(r[0]) / float(r[1]) if len(r) == 2 else 1.0
            SA, ST, SC, PA, PC = t['SA'], t['ST'], t['SC'], t['PA'], t['PC']
            gv = sp(1.0 / (2 * turns))   # V(SA,ST) = V(PA,PC)/(2*turns)
            gi = sp(1.0 / turns)         # Ip = (Isa+Isc)/turns
            body += ['E%s_a %s %s %s %s %s' % (nm, SA, ST, PA, PC, gv),
                     'E%s_c %s %s %s %s %s' % (nm, ST, SC, PA, PC, gv),
                     'F%s_a %s %s E%s_a %s' % (nm, PA, PC, nm, gi),
                     'F%s_c %s %s E%s_c %s' % (nm, PA, PC, nm, gi),
                     # bleeders: give the (otherwise ideal) windings a DC path so the
                     # matrix isn't singular / floating.
                     'R%s_a %s %s 1meg' % (nm, SA, ST),
                     'R%s_c %s %s 1meg' % (nm, ST, SC),
                     # OT damping: a pure-ideal OT + high loop-gain global NFB (e.g.
                     # the 5150) oscillates instantly. A plate-to-plate resistive damper
                     # + a snubber lower the OT Q to converge, at minor HF cost. Light
                     # defaults suit moderate amps; stiff amps need heavier (--ot-damp
                     # 3k --ot-snub 220n --nfb-comp <node>=1n for the 5150 full).
                     'R%s_ppd %s %s %s' % (nm, SA, SC, ot_damp),
                     'R%s_sn %s n%s_sn 100' % (nm, PA, nm),
                     'C%s_sn n%s_sn %s %s' % (nm, nm, PC, ot_snub)]
        else:
            print('WARN: unhandled component %s (%s)' % (nm, ty), file=sys.stderr)

    if need_dgrid[0]:
        sub_defs.insert(0, '.model DGRID D(IS=1e-12 RS=2000 N=1.0)')
    if nfb_comp and '=' in nfb_comp:      # HF compensation cap at the feedback node
        n, cval_ = nfb_comp.split('=')
        body.append('Cnfbcomp %s 0 %s' % (node(n), cval_))
    if extra:                             # inject extra SPICE lines (e.g. missing bright caps)
        for ln in extra.split(';'):
            if ln.strip():
                body.append(ln.strip())
    lines += sub_defs + [''] + body + [
        '', '.options method=%s reltol=1e-3 abstol=5e-9 vntol=1e-6 itl1=500 itl4=500 '
        'gmin=1e-9 gminsteps=50 rshunt=1e8' % method,
        # tran step = (1/48kHz)/oversample; finer step -> ngspice resolves more
        # bandwidth so the anti-alias decimation downstream has real HF to filter.
        # `save` only the output node — ngspice otherwise keeps EVERY node's full
        # transient in RAM (~100 nodes × millions of points), which OOMs on long
        # inputs / high oversample / many parallel workers. Storing one node cuts
        # memory ~100×. (The full circuit is still solved; only storage is limited.)
        '.control', 'set filetype=ascii', 'save v(%s)' % out_node,
        'tran %.6fu %s' % (20.8333 / oversample, sp(dur)),
        'wrdata %s v(%s)' % (csv, out_node), 'quit', '.endc', '.end', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('netlist'); ap.add_argument('-o', '--out', default='circuit.cir')
    ap.add_argument('--pots', default='', help='name=wipe,name=wipe')
    ap.add_argument('--pwl', default='input.pwl'); ap.add_argument('--dur', type=float, default=0.5)
    ap.add_argument('--csv', default='out.csv')
    ap.add_argument('--koren', action='store_true',
                    help='Koren triode model (softer/convergence mode) instead of exact DempwolfZolzer')
    ap.add_argument('--ot-damp', default='47k', help='OT plate-to-plate damper R (heavier=more stable)')
    ap.add_argument('--ot-snub', default='10n', help='OT snubber C')
    ap.add_argument('--nfb-comp', default=None, help='NFB compensation cap, NODE=value (e.g. nNFB=1n)')
    ap.add_argument('--input-mode', default='filesource', choices=['filesource', 'pwl'],
                    help='how to feed the input file (time,raw-sample pairs)')
    ap.add_argument('--oversample', type=int, default=1,
                    help='solve at N*48kHz (finer tran step); pair with anti-alias decimation')
    ap.add_argument('--extra', default=None,
                    help='inject extra SPICE lines, ;-separated (e.g. missing bright caps)')
    a = ap.parse_args()
    pots = dict((kv.split('=')[0], float(kv.split('=')[1])) for kv in a.pots.split(',') if '=' in kv)
    nl = json.load(open(a.netlist))
    open(a.out, 'w').write(translate(nl, pots, a.pwl, a.dur, a.csv, koren=a.koren,
                                     ot_damp=a.ot_damp, ot_snub=a.ot_snub, nfb_comp=a.nfb_comp,
                                     input_mode=a.input_mode, oversample=a.oversample, extra=a.extra))
    print('wrote', a.out)
