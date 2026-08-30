#!/usr/bin/env python3
"""Prototype: translate the stiff high-gain amp head's Lead FULL circuit to an ngspice netlist.

Proof-of-concept for an *offline* ngspice dataset-generation backend, as an
alternative to livespice_cli for the stiff / high-gain / combined circuits where
LiveSPICE's fixed-timestep solver diverges and needs extreme oversampling.

The stiff amp head's Lead *full* circuit is the motivating case: in LiveSPICE it diverges to
~+95 dB at the default 2x oversample and needs --oversample 32 (~158 s per 5 s
of audio) to stay bounded. ngspice, with adaptive timestepping + trapezoidal
integration, converges cleanly on the same circuit with NO oversample hack.

Result (this prototype, hard 80->6000 Hz chirp input, pots baked to noon-ish):
  * DC operating point solves (all tubes bias correctly)
  * FULL amp completes the whole transient, BOUNDED (speaker +/-4.3 V; preamp
    internally slams +/-124 V -- a real cranked-amp sim), ~2 s wall for 0.3 s
    audio (~6x faster than LiveSPICE's 32x workaround, and correct)
  * never diverges -- always bounded; fails safe (aborts) rather than emitting
    +95 dB garbage the way a fixed-step solver does

Key findings:
  * The tube cascade was NOT the hard part for ngspice (DC + preamp trivial).
    The hard part is the coupled-inductor OT + global NFB loop.
  * Integration method matters most: method=gear ABORTS at the NFB node (~45 ms);
    method=trap COMPLETES. That is the critical knob (default here = trap).
  * Realistic OT winding DCR + damping is needed (near-ideal coupled inductors
    + global NFB oscillate and force the timestep to zero).

CAVEATS (this proves CONVERGENCE, not fidelity):
  * Tubes are Koren models with params ported from LiveSPICE's Dempwolf-Zolzer/
    Koren values -- close but not bit-identical; validate before trusting a
    dataset's tone.
  * Pots baked LINEAR (no taper); OT is an approximate coupled-inductor model
    (turns ratio + guessed L/DCR/snubber); output level is not calibrated.
  * Adaptive timesteps are non-uniform -> a real pipeline must resample to the
    audio grid (here tran uses a 48 kHz nominal step so output is near-uniform).

Env flags:
  METHOD=trap|gear   integration method (default trap -- the one that completes)
  PREAMP_ONLY=1      stop at the preamp output (drop the power amp)
  NO_NFB=1           remove R4 (open the PI-grid feedback injection)
  FLIP_SEC=1         reverse the OT secondary sense (diagnostic: makes NFB +ve)

Usage:
  python3 gen_evh5150_ngspice.py            # writes evh5150.cir next to this file
  ngspice -b evh5150.cir                     # -> evh5150_out.csv (wrdata, ascii)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CIR  = os.path.join(HERE, 'evh5150.cir')
CSV  = os.path.join(HERE, 'evh5150_out.csv')
METHOD = os.environ.get('METHOD', 'trap')
NO_NFB = os.environ.get('NO_NFB') == '1'
PREAMP_ONLY = os.environ.get('PREAMP_ONLY') == '1'
FLIP_SEC = os.environ.get('FLIP_SEC') == '1'

# ---- component list mirrors this amp's own generator script in the parametric-devices repo ----
def gstage(name, gin, gres, gleak, pr, ck, cc, bp, pout, cpl):
    g = 'n%s_g' % name; p = 'n%s_p' % name; k = 'n%s_k' % name
    e = [('Rg%s' % name, 'R', {'Resistance': gres}, {'A': gin, 'B': g}),
         ('Rl%s' % name, 'R', {'Resistance': gleak}, {'A': g, 'B': 'GND'}),
         ('V%s' % name, 'T', {}, {'P': p, 'G': g, 'K': k}),
         ('Rp%s' % name, 'R', {'Resistance': pr}, {'A': bp, 'B': p}),
         ('Rk%s' % name, 'R', {'Resistance': ck}, {'A': k, 'B': 'GND'}),
         ('Cc%s' % name, 'C', {'Capacitance': cpl}, {'A': p, 'B': pout})]
    if cc:
        e.append(('Ck%s' % name, 'C', {'Capacitance': cc}, {'A': k, 'B': 'GND'}))
    return e

NET = [
    ('J_IN', 'IN', {'V0dBFS': '0.05 V'}, {'S': 'IN', 'G': 'GND'}),
    ('BP1', 'RAIL', {'Voltage': '410 V'}, {'T': 'BP1'}),
    ('BP2', 'RAIL', {'Voltage': '410 V'}, {'T': 'BP2'}),
    ('BP', 'RAIL', {'Voltage': '450 V'}, {'T': 'BP'}),
    ('VBIAS', 'RAIL', {'Voltage': '-50 V'}, {'T': 'VBIAS'}),
    ('Rin', 'R', {'Resistance': '68 kΩ'}, {'A': 'IN', 'B': 'nV1A_g'}),
    ('Rleak', 'R', {'Resistance': '1 MΩ'}, {'A': 'nV1A_g', 'B': 'GND'}),
    ('V1A', 'T', {}, {'P': 'nV1A_p', 'G': 'nV1A_g', 'K': 'nV1A_k'}),
    ('Rp1a', 'R', {'Resistance': '220 kΩ'}, {'A': 'BP1', 'B': 'nV1A_p'}),
    ('Rk1a', 'R', {'Resistance': '1.5 kΩ'}, {'A': 'nV1A_k', 'B': 'GND'}),
    ('Ck1a', 'C', {'Capacitance': '1 µF'}, {'A': 'nV1A_k', 'B': 'GND'}),
    ('C6', 'C', {'Capacitance': '22 nF'}, {'A': 'nV1A_p', 'B': 'nc1'}),
    ('LeadPre', 'POT', {'Resistance': '1 MΩ', 'Wipe': '0.6', 'Sweep': 'Logarithmic'}, {'A': 'nc1', 'W': 'npre', 'K': 'GND'}),
]
NET += gstage('V1B', 'npre', '470 kΩ', '1 MΩ', '100 kΩ', '1.5 kΩ', '1 µF', 'BP1', 'nc2', '22 nF')
NET += gstage('V2A', 'nc2', '220 kΩ', '1 MΩ', '100 kΩ', '1.5 kΩ', '1 µF', 'BP1', 'nc3', '22 nF')
NET += gstage('V2B', 'nc3', '220 kΩ', '1 MΩ', '220 kΩ', '1.5 kΩ', '1 µF', 'BP2', 'nc4', '22 nF')
NET += gstage('V5B', 'nc4', '1 MΩ', '1 MΩ', '220 kΩ', '2.2 kΩ', '1 µF', 'BP2', 'nc5', '22 nF')
NET += gstage('V5A', 'nc5', '1 MΩ', '1 MΩ', '100 kΩ', '2.2 kΩ', None, 'BP2', 'nTSin', '22 nF')
NET += [
    ('RtsIn', 'R', {'Resistance': '1 MΩ'}, {'A': 'nTSin', 'B': 'GND'}),
    ('Ct', 'C', {'Capacitance': '250 pF'}, {'A': 'nTSin', 'B': 'nTA'}),
    ('Rslope', 'R', {'Resistance': '100 kΩ'}, {'A': 'nTSin', 'B': 'nSL'}),
    ('High', 'POT', {'Resistance': '250 kΩ', 'Wipe': '0.5', 'Sweep': 'Linear'}, {'A': 'nTA', 'W': 'nTSout', 'K': 'nSL'}),
    ('Cbass', 'C', {'Capacitance': '22 nF'}, {'A': 'nSL', 'B': 'nB3'}),
    ('Low', 'POT', {'Resistance': '1 MΩ', 'Wipe': '0.5', 'Sweep': 'Linear'}, {'A': 'nB3', 'W': 'nM3', 'K': 'nM3'}),
    ('Cmid', 'C', {'Capacitance': '22 nF'}, {'A': 'nSL', 'B': 'nM3'}),
    ('Mid', 'POT', {'Resistance': '25 kΩ', 'Wipe': '0.5', 'Sweep': 'Linear'}, {'A': 'nM3', 'W': 'GND', 'K': 'GND'}),
    ('Rgr', 'R', {'Resistance': '1 MΩ'}, {'A': 'nTSout', 'B': 'nrec_g'}),
    ('V3', 'T', {}, {'P': 'nrec_p', 'G': 'nrec_g', 'K': 'nrec_k'}),
    ('Rpr', 'R', {'Resistance': '100 kΩ'}, {'A': 'BP2', 'B': 'nrec_p'}),
    ('Rkr', 'R', {'Resistance': '1.5 kΩ'}, {'A': 'nrec_k', 'B': 'GND'}),
    ('Ckr', 'C', {'Capacitance': '1 µF'}, {'A': 'nrec_k', 'B': 'GND'}),
    ('Ccr', 'C', {'Capacitance': '22 nF'}, {'A': 'nrec_p', 'B': 'npost'}),
    ('LeadPost', 'POT', {'Resistance': '1 MΩ', 'Wipe': '0.6', 'Sweep': 'Logarithmic'}, {'A': 'npost', 'W': 'nPAout', 'K': 'GND'}),
    ('Ratt', 'R', {'Resistance': '1 MΩ'}, {'A': 'nPAout', 'B': 'nV3in'}),
    ('Rattsh', 'R', {'Resistance': '47 kΩ'}, {'A': 'nV3in', 'B': 'GND'}),
    ('R1', 'R', {'Resistance': '10 kΩ'}, {'A': 'BP', 'B': 'nB7'}),
    ('Cpi', 'C', {'Capacitance': '50 µF'}, {'A': 'nB7', 'B': 'GND'}),
    ('C1', 'C', {'Capacitance': '22 nF'}, {'A': 'nV3in', 'B': 'nV3ag'}),
    ('V3PI', 'T', {}, {'P': 'nV3ap', 'G': 'nV3ag', 'K': 'nTail'}),
    ('R2', 'R', {'Resistance': '82 kΩ'}, {'A': 'nB7', 'B': 'nV3ap'}),
    ('V3bPI', 'T', {}, {'P': 'nV3bp', 'G': 'GND', 'K': 'nTail'}),
    ('R3', 'R', {'Resistance': '100 kΩ'}, {'A': 'nB7', 'B': 'nV3bp'}),
    ('R5', 'R', {'Resistance': '470 Ω'}, {'A': 'nTail', 'B': 'nNFB'}),
    ('R4', 'R', {'Resistance': '1 MΩ'}, {'A': 'nV3ag', 'B': 'nNFB'}),
    ('R6', 'R', {'Resistance': '1 MΩ'}, {'A': 'nNFB', 'B': 'n11'}),
    ('R7', 'R', {'Resistance': '10 kΩ'}, {'A': 'nNFB', 'B': 'n12'}),
    ('C2p', 'C', {'Capacitance': '0.1 µF'}, {'A': 'n11', 'B': 'n12'}),
    ('R8', 'R', {'Resistance': '22 kΩ'}, {'A': 'n12', 'B': 'nSpk'}),
    ('Presence', 'POT', {'Resistance': '10 kΩ', 'Wipe': '0.5', 'Sweep': 'Linear'}, {'A': 'nPres', 'W': 'nPres', 'K': 'nSpk'}),
    ('C3p', 'C', {'Capacitance': '0.1 µF'}, {'A': 'nPres', 'B': 'n12'}),
    ('C4p', 'C', {'Capacitance': '22 nF'}, {'A': 'nV3ap', 'B': 'nGA'}),
    ('C5p', 'C', {'Capacitance': '22 nF'}, {'A': 'nV3bp', 'B': 'nGB'}),
    ('R13', 'R', {'Resistance': '220 kΩ'}, {'A': 'VBIAS', 'B': 'nGA'}),
    ('R14', 'R', {'Resistance': '220 kΩ'}, {'A': 'VBIAS', 'B': 'nGB'}),
    ('Rg5', 'R', {'Resistance': '2.2 kΩ'}, {'A': 'nGA', 'B': 'nV5g'}),
    ('V5', 'PENT', {}, {'P': 'nPUp', 'G': 'nV5g', 'G2': 'nSc5', 'K': 'GND'}),
    ('Rsc5', 'R', {'Resistance': '100 Ω'}, {'A': 'BP', 'B': 'nSc5'}),
    ('Rg7', 'R', {'Resistance': '2.2 kΩ'}, {'A': 'nGA', 'B': 'nV7g'}),
    ('V7', 'PENT', {}, {'P': 'nPUp', 'G': 'nV7g', 'G2': 'nSc7', 'K': 'GND'}),
    ('Rsc7', 'R', {'Resistance': '100 Ω'}, {'A': 'BP', 'B': 'nSc7'}),
    ('Rg6', 'R', {'Resistance': '2.2 kΩ'}, {'A': 'nGB', 'B': 'nV6g'}),
    ('V6', 'PENT', {}, {'P': 'nPLo', 'G': 'nV6g', 'G2': 'nSc6', 'K': 'GND'}),
    ('Rsc6', 'R', {'Resistance': '100 Ω'}, {'A': 'BP', 'B': 'nSc6'}),
    ('Rg8', 'R', {'Resistance': '2.2 kΩ'}, {'A': 'nGB', 'B': 'nV8g'}),
    ('V8', 'PENT', {}, {'P': 'nPLo', 'G': 'nV8g', 'G2': 'nSc8', 'K': 'GND'}),
    ('Rsc8', 'R', {'Resistance': '100 Ω'}, {'A': 'BP', 'B': 'nSc8'}),
    ('TX1', 'CTT', {'Turns': '7:108'}, {'SA': 'nPUp', 'SC': 'nPLo', 'ST': 'BP', 'PA': 'nSpk', 'PC': 'GND'}),
    ('Rspk', 'R', {'Resistance': '8 Ω'}, {'A': 'nSpk', 'B': 'GND'}),
]

def rval(s):  # '68 kΩ'->'68k', '1 MΩ'->'1meg', '470 Ω'->'470'
    s = s.replace('Ω', '').replace(' ', '')
    if s.endswith('k'): return s
    if s.endswith('M'): return s[:-1] + 'meg'
    return s

def cval(s):  # '22 nF'->'22n', '1 µF'->'1u', '250 pF'->'250p'
    return s.replace(' ', '').replace('F', '').replace('µ', 'u')

def node(n): return '0' if n == 'GND' else n

lines = ['* Stiff high-gain amp head, Lead Full -> ngspice (Koren tubes, baked pots, coupled-inductor OT)', '']
# --- Koren tube subcircuit library (params = LiveSPICE Koren-family values) ---
lines += [
 '* --- 12AX7 triode (Koren): Mu=83.5 Ex=1.4 Kg1=1060 Kp=600 Kvb=300 ---',
 '.subckt TRI12AX7 P G K',
 'B7 7 0 V = V(P,K)/600*ln(1+exp(600*(1/83.5+V(G,K)/sqrt(300+V(P,K)*V(P,K)))))',
 'BP P K I = 2*pow(uramp(V(7)),1.4)/1060',   # = (pwr+pwrs)/Kg1, ngspice-native (no PSPICE pwrs)
 'Dgk G K DGRID',
 'CGP G P 2.4p',
 'CGK G K 2.3p',
 '.ends', '',
 '* --- 6L6GC pentode (Koren): Mu=8.7 Ex=1.35 Kg1=1460 Kg2=4500 Kp=48 Kvb=12 ---',
 '.subckt PEN6L6 P G2 G K',
 'B7 7 0 V = V(G2,K)/48*ln(1+exp(48*(1/8.7+V(G,K)/sqrt(12+V(G2,K)*V(G2,K)))))',
 'BP P K I = 2*pow(uramp(V(7)),1.35)/1460*atan(V(P,K)/24)',
 'BG2 G2 K I = 2*pow(uramp(V(7)),1.35)/4500',
 'Dgk G K DGRID',
 '.ends', '',
 '.model DGRID D(IS=1e-12 RS=2000 N=1.0)', '',
 '* --- input: hard linear chirp 80->6000 Hz, 0.3V (torture signal, like the sweeps',
 '*     that hang LiveSPICE). Real pipeline would use a PWL from the guitar wav. ---',
 'Bin IN 0 V = 0.3*sin(6.28318530718*(80*time+0.5*(6000-80)/0.3*time*time))', '',
]

tri = pen = 0; stop = False
for cid, kind, attrs, pins in NET:
    if PREAMP_ONLY and cid == 'Ratt': stop = True  # everything from the interstage atten on = power amp
    if stop: continue
    if kind == 'IN': continue
    if NO_NFB and cid == 'R4': continue            # open the global NFB (PI-grid injection)
    if kind == 'RAIL':
        v = attrs['Voltage'].replace(' V', '').replace(' ', '')
        lines.append('V%s %s 0 DC %s' % (cid, node(list(pins.values())[0]), v)); continue
    if kind == 'R':
        lines.append('R%s %s %s %s' % (cid, node(pins['A']), node(pins['B']), rval(attrs['Resistance']))); continue
    if kind == 'C':
        lines.append('C%s %s %s %s' % (cid, node(pins['A']), node(pins['B']), cval(attrs['Capacitance']))); continue
    if kind == 'POT':
        R = float(rval(attrs['Resistance']).replace('k', 'e3').replace('meg', 'e6'))
        P = float(attrs['Wipe'])   # linear bake (tone accuracy is not the point of this convergence test)
        A, W, K = node(pins['A']), node(pins['W']), node(pins['K'])
        if A != W: lines.append('R%s_aw %s %s %.4g' % (cid, A, W, max(R * (1 - P), 1e-3)))
        if W != K: lines.append('R%s_wk %s %s %.4g' % (cid, W, K, max(R * P, 1e-3)))
        continue
    if kind == 'T':
        tri += 1; lines.append('X%s %s %s %s TRI12AX7' % (cid, node(pins['P']), node(pins['G']), node(pins['K']))); continue
    if kind == 'PENT':
        pen += 1; lines.append('X%s %s %s %s %s PEN6L6' % (cid, node(pins['P']), node(pins['G2']), node(pins['G']), node(pins['K']))); continue
    if kind == 'CTT':
        # reversed OT: plates->SA/SC, B+ center tap->ST, speaker->PA/PC. Turns 108(pp):7.
        # center-tapped primary halves Lp1/Lp2 + secondary Ls; realistic winding DCR +
        # damping (near-ideal coupled inductors + global NFB oscillate -> timestep->0).
        SA, SC, ST, PA, PC = [node(pins[x]) for x in ('SA', 'SC', 'ST', 'PA', 'PC')]
        ls = 'Ls %s nLsa 0.336' % PC if FLIP_SEC else 'Ls nLsa %s 0.336' % PC
        lines += ['Rp1dc %s nLp1a 60' % SA, 'Lp1 nLp1a %s 20' % ST,
                  'Rp2dc %s nLp2a 60' % SC, 'Lp2 nLp2a %s 20' % ST,
                  'Rsdc %s nLsa 0.5' % PA, ls,
                  'K1 Lp1 Lp2 0.999', 'K2 Lp1 Ls 0.999', 'K3 Lp2 Ls 0.999',
                  'Rsnub %s nsnub 10' % PA, 'Csnub nsnub %s 47n' % PC,
                  'Rppd %s %s 22k' % (SA, SC)]  # pure resistive plate-to-plate damper (lowers OT Q)
        continue

if PREAMP_ONLY:
    lines.append('RpaLoad nPAout 0 1meg')
    probe = 'v(nPAout) v(nV1A_p) v(nTSout) v(nrec_p)'
else:
    probe = 'v(nSpk) v(nV1A_p) v(nPAout) v(nV3in)'

lines += ['',
 '* method=trap COMPLETES the full amp; method=gear aborts at the NFB node (~45 ms).',
 # NO `.options klu` here, and do not lower itl4 -- both measured (2026-07-21):
 # with klu this transient ABORTS at 10.7ms (grid-diode node, "timestep too
 # small"); Sparse completes it. And itl4 < ~1000 aborts at 32.5ms -- the step
 # cut-and-retry does NOT substitute for Newton iterations at the hard moments,
 # so the high limit is load-bearing on this amp (it costs nothing where it
 # does not bind: the reverse-linear-drive pedal renders bit-identically at itl4=10 vs 500).
 '.options method=%s reltol=1e-3 abstol=1e-9 vntol=1e-6 itl1=1000 itl4=1000 gmin=1e-12' % METHOD,
 '.control',
 'set filetype=ascii',
 'tran 20.8333u 0.3',            # 48 kHz nominal output step; solver adapts finer internally
 'wrdata %s %s' % (CSV, probe),
 'quit',
 '.endc', '.end', '']

open(CIR, 'w').write('\n'.join(lines))
print('wrote %s: %d triodes, %d pentodes, method=%s%s%s'
      % (CIR, tri, pen, METHOD,
         '  [PREAMP_ONLY]' if PREAMP_ONLY else '', '  [NO_NFB]' if NO_NFB else ''))
print('run:  ngspice -b %s   (writes %s)' % (CIR, CSV))
