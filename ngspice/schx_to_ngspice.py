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
  * CenterTapTransformer: IDEAL, via VCVS/CCCS -- deliberately identical to
    LiveSPICE's model, so the two backends cannot disagree about the OT. It has
    no magnetizing inductance and therefore PASSES DC, which real iron cannot.
    Circuits needing that physics model it explicitly, as a shunt Inductor "Lm"
    across the secondary in the .schx, which both backends emit identically.
    Do not "fix" it here alone -- that would split the backends.

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
# semiconductor models (diode / BJT / JFET) + op-amp macromodel
# --------------------------------------------------------------------------
# Default parasitic caps / transit times / series R below are ESSENTIAL for transient
# convergence — a device with zero junction cap or channel resistance switches
# infinitely fast and drives ngspice's timestep to 0 ("timestep too small"). They're
# small (audio-band negligible). Resolution order per value:
#   real .schx param (p)  >  per-run convergence override (conv)  >  hardcoded default.
# High-gain hard-clip stages (e.g. the MT-2) need bigger `diode_cjo` to soften the
# clip edges; JFET-heavy circuits may need bigger `jfet_rd`/`jfet_rs`.
def _cv(p, pkey, conv, ckey, default):
    return p.get(pkey) or conv.get(ckey) or default


def diode_model(p, conv):
    return 'D(IS=%s N=%s CJO=%s TT=%s)' % (
        sp(qty(p.get('IS', '1e-14'))), sp(qty(p.get('n', '1'))),
        sp(qty(_cv(p, 'CJO', conv, 'diode_cjo', '4p'))),
        sp(qty(_cv(p, 'TT', conv, 'diode_tt', '5n'))))


def bjt_model(p, conv):
    typ = 'PNP' if str(p.get('Type', 'NPN')).upper() == 'PNP' else 'NPN'
    return '%s(IS=%s BF=%s BR=%s CJE=%s CJC=%s TF=%s)' % (
        typ, sp(qty(p.get('IS', '1e-16'))), sp(qty(p.get('BF', '100'))),
        sp(qty(p.get('BR', '1'))), sp(qty(_cv(p, 'CJE', conv, 'bjt_cje', '8p'))),
        sp(qty(_cv(p, 'CJC', conv, 'bjt_cjc', '4p'))),
        sp(qty(_cv(p, 'TF', conv, 'bjt_tf', '0.5n'))))


def jfet_model(p, conv):
    typ = 'PJF' if str(p.get('Type', 'N')).upper().startswith('P') else 'NJF'
    return '%s(VTO=%s BETA=%s LAMBDA=%s IS=%s CGS=%s CGD=%s RD=%s RS=%s)' % (
        typ, sp(qty(p.get('Vt0', '-2'))), sp(qty(p.get('Beta', '1e-4'))),
        sp(qty(p.get('Lambda', '0'))), sp(qty(p.get('IS', '1e-14'))),
        sp(qty(_cv(p, 'CGS', conv, 'jfet_cgs', '5p'))),
        sp(qty(_cv(p, 'CGD', conv, 'jfet_cgd', '5p'))),
        sp(qty(_cv(p, 'RD', conv, 'jfet_rd', '10'))),
        sp(qty(_cv(p, 'RS', conv, 'jfet_rs', '10'))))


def opamp_subckt(name, Aol, GBP, Rout, rails):
    """Single-pole op-amp macromodel. The gain node is ground-referenced (the
    external feedback loop sets the DC output level, incl. single-supply mid-bias);
    open-loop pole at GBP/Aol. When Vcc+/Vcc- are connected, the output buffer is
    clamped to the rails (captures op-amp clipping). ngspice's adaptive solver
    keeps the high-gain stage bounded (unlike LiveSPICE's fixed step)."""
    Rp = 1e6
    gm, Cp = Aol / Rp, Aol / (2 * math.pi * Rp * GBP)
    hdr = '.subckt %s inp inn out%s' % (name, ' vp vn' if rails else '')
    core = ['Ga 0 g inp inn %s' % sp(gm), 'Ra g 0 %s' % sp(Rp), 'Ca g 0 %s' % sp(Cp)]
    if rails:
        core += ['Bo ob 0 V = min(max(V(g),V(vn)+0.5),V(vp)-0.5)', 'Ro ob out %s' % sp(Rout)]
    else:
        core += ['Ro g out %s' % sp(Rout)]
    return [hdr] + core + ['.ends']


# --------------------------------------------------------------------------
# main translation
# --------------------------------------------------------------------------
def translate(netlist, pots=None, input_pwl='input.pwl', dur=0.5, csv='out.csv',
              method='trap', koren=False, ot_damp='47k', ot_snub='10n', nfb_comp=None,
              input_mode='filesource', oversample=1, extra=None, conv=None):
    pots = pots or {}
    conv = conv or {}   # per-run device convergence overrides (diode_cjo, jfet_rd, ...)
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

    # semiconductor .model dedup (diode/BJT/JFET) + op-amp macromodel dedup
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
            opamp_defs.extend(opamp_subckt(opamps[sig], Aol, GBP, Rout, rails))
        return opamps[sig]

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
        elif ty == 'Inductor':
            n = list(t.values())
            lval = c.get('value') or p.get('Inductance')   # value field may be null; fall back to params
            body.append('L%s %s %s %s' % (nm, n[0], n[1], sp(qty(lval))))
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
        elif ty == 'Diode':
            body.append('D%s %s %s %s' % (nm, t['Anode'], t['Cathode'], mdl('DM', diode_model(p, conv))))
        elif ty == 'BipolarJunctionTransistor':
            body.append('Q%s %s %s %s %s' % (nm, t['C'], t['B'], t['E'], mdl('QM', bjt_model(p, conv))))
        elif ty == 'JunctionFieldEffectTransistor':
            body.append('J%s %s %s %s %s' % (nm, t['D'], t['G'], t['S'], mdl('JM', jfet_model(p, conv))))
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
    # KLU direct solver: ngspice-46+ ships it but silently defaults to Sparse 1.3
    # (the 1980s solver) — the banner says "Compiled with KLU" yet every run logs
    # "Using SPARSE 1.3" until `.options klu` selects it. This workload is millions
    # of factorizations per render, and KLU is typically 1.3-2x faster on the
    # 100+-node amps. Harmless on non-KLU builds (unknown option -> warning,
    # Sparse fallback). Opt out via conv={'klu': '0'} if an A/B ever regresses.
    use_klu = str(conv.get('klu', '1')).lower() not in ('0', 'false', 'off', 'no')

    # tmax IS the internal step ceiling, and without an explicit one ngspice uses
    # min(tstep, span/50) = tstep — so the "adaptive" solver was forced to take
    # (1/48kHz)/oversample steps EVERYWHERE, including passages where LTE control
    # would happily stride far larger. Measured: an explicit tmax at the audio
    # period cut internal step count 2.5x with ESR 4.2e-6 vs a 2us reference.
    # (The old comment here claimed tstep was "just a print-resolution hint" and
    # tmax an opt-in tightener. It was backwards: tstep was the cap, and the
    # emitted tmax could only LOOSEN it at oversample>=2.)
    # Default the ceiling to ONE AUDIO PERIOD, decoupled from oversample: the
    # solver always resolves at least 48kHz, LTE takes it finer through clip
    # edges, and oversample keeps its real job — output density for the
    # anti-alias decimation downstream. conv={'tmax': '...'} overrides (smaller
    # = forced finer everywhere = slower; the DS-1's 20.8333u is now the default).
    tmax = conv.get('tmax') or '20.8333u'

    lines += sub_defs + opamp_defs + model_defs + [''] + body + [
        '', '.options method=%s reltol=1e-3 abstol=5e-9 vntol=1e-6 itl1=500 itl4=500 '
        'gmin=1e-9 gminsteps=50 rshunt=1e8%s' % (method, ' klu' if use_klu else ''),
        # `save` only the output node — ngspice otherwise keeps EVERY node's full
        # transient in RAM (~100 nodes × millions of points), which OOMs on long
        # inputs / high oversample / many parallel workers. Storing one node cuts
        # memory ~100×. (The full circuit is still solved; only storage is limited.)
        '.control', 'set filetype=ascii', 'save v(%s)' % out_node,
        # tstep is kept at (1/48kHz)/oversample for wrdata/print bookkeeping, but
        # the solve density is governed by tmax + LTE (see above), not tstep.
        'tran %.6fu %s 0 %s' % (20.8333 / oversample, sp(dur), tmax),
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
    ap.add_argument('--conv', default='',
                    help='device convergence overrides, key=val,...  keys: diode_cjo diode_tt '
                         'bjt_cje bjt_cjc bjt_tf jfet_cgs jfet_cgd jfet_rd jfet_rs '
                         '(e.g. diode_cjo=100p for high-gain hard-clip stages like the MT-2)')
    a = ap.parse_args()
    pots = dict((kv.split('=')[0], float(kv.split('=')[1])) for kv in a.pots.split(',') if '=' in kv)
    conv = dict(kv.split('=', 1) for kv in a.conv.split(',') if '=' in kv)
    nl = json.load(open(a.netlist))
    open(a.out, 'w').write(translate(nl, pots, a.pwl, a.dur, a.csv, koren=a.koren,
                                     ot_damp=a.ot_damp, ot_snub=a.ot_snub, nfb_comp=a.nfb_comp,
                                     input_mode=a.input_mode, oversample=a.oversample, extra=a.extra,
                                     conv=conv))
    print('wrote', a.out)
