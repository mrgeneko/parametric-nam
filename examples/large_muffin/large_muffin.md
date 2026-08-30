# Large Muffin — a classic silicon fuzz/sustain pedal

Our own transcription of a classic, heavily-nonlinear silicon fuzz/sustain circuit from the
early-70s "triangle" era of a well-known fuzz pedal lineage — an excellent NAM target.
Topology and component values verified node-for-node against a published, independent circuit
analysis of this exact revision. Not modeling or affiliated with any specific manufacturer's
product; this is a generic worked example of the topology, with the pedal renamed throughout
this repo (including its shipped `.schx` file path) to keep any real product's name out of it.

Bundled here as `large_muffin.schx`. **3 knobs: SUSTAIN, TONE, VOLUME**
(all 100 k, **linear taper** per the source schematic's own component notes). +9 V rail.
Reference designators match the schematic.

## Topology (verified against the published circuit analysis)
`IN → R2 → C1 → Q4 (input booster) → C4 → SUSTAIN → C5 → Q3 (clip 1) → C13 →
Q2 (clip 2) → TONE → C3 → Q1 (recovery) → C2 → VOLUME → OUT`. The first three
transistors (Q4/Q3/Q2) are **2N5133 NPN** shunt-feedback common-emitter stages
(470 k collector→base feedback + 500 pF Miller cap + 120 Ω emitter). The **output
recovery Q1 is the exception**: it is **voltage-divider biased** (R7 470 k from
+9 V to base, R3 100 k base to ground) with **no shunt feedback and no Miller
cap** — a flat ~13 dB gain stage that recovers the passive-tone loss.

- **Clipping (Q3, Q2):** `collector → 1 µF (C6/C7) → anti-parallel 1N914 pair →
  base`. The **1 µF cap is in *series* with the diodes** so they clip only the AC
  (blocking the DC bias) — this is the actual clipping mechanism for this circuit, and it
  is *not* diodes placed straight across collector-base.
- **SUSTAIN**: passive divider (C4 in, C5 out); **R23 (1 k) is the ground-leg
  floor** so the signal isn't fully cut at minimum sustain.
- **TONE**: passive mid-scoop — treble cap C9 (.004 µF) → R5 (27 k) to gnd, bass
  leg R8 (27 k) → C8 (.01 µF) to gnd, blended by the 100 k pot; fed directly from
  Q2's collector, wiper → C3 → Q1.
- **Input**: R2 (36 k) in series *before* C1 (1 µF), then base; R14 (100 k) base
  bias to ground.

## Validation (livespice_cli)
- Loads; **3 controls**; **converges at 2×** (no NaN).
- **Internal clipping confirmed**: Q2 collector ≈ hard square (crest ≈ 1.17).
- **Knobs respond correctly**: SUSTAIN sets clipping amount (output crest 2.54 at
  low → 1.39 at high — dynamic vs squashed); TONE strong (rms 0.038↔0.334);
  VOLUME monotonic. Output level healthy.

## Lessons learned (5 rounds of comparison vs the published reference analysis)
Built iteratively, re-comparing against the **authoritative reference analysis
node-by-node each round** (not from memory). Four rounds found real bugs; the
fifth confirmed convergence. The traps, in order found:

1. **Clipping is AC-coupled, not diodes-across-C-B.** This circuit clips via
   `collector → 1 µF series cap (C6/C7) → anti-parallel 1N914 → base`. The series
   cap blocks the DC bias so the diodes clip only the AC swing. My first build put
   the diodes straight across collector↔base (DC-inclusive) and omitted/misplaced
   the series cap → output ~15× too quiet and wrong clipping shape. **The single
   most important detail in the circuit.**
2. **The sustain pot has a ground-leg floor (R23, 1 k).** Without it the signal is
   fully cut at minimum sustain; R23 sets a floor so it never fully mutes.
3. **The output recovery Q1 does NOT follow the other stages' pattern.** Q4/Q3/Q2
   are shunt-feedback (470 k C→B + 500 pF Miller); Q1 is **voltage-divider biased**
   (R7 from +9 V to base, R3 to ground) with **no shunt feedback and no Miller cap**.
   Copy-pasting the shunt-fb pattern onto Q1 is wrong. A **DC operating-point probe**
   caught/confirmed this: Q1's collector biases to ~5 V (mid-rail = clean recovery),
   exactly what the reference analysis describes.
4. **All three pots are linear taper.** Don't default Volume to logarithmic.

**Verification methodology that worked** (reusable for any pedal/amp build):
- Fetch the authoritative reference (a published analysis, schematic) and diff
  **node-by-node**, stage-by-stage — resist transcribing from memory.
- **Parts-list completeness check**: enumerate every reference designator (R/C/Q/D/
  pot) and confirm each is present, none missing, none extra. Caught nothing here
  (good) but cheaply rules out a whole class of omissions (pulldowns, filters).
- **DC bias probe**: feed silence, `PROBE` each collector/base, confirm every
  transistor sits in its active region (not railed, not saturated). Validates
  biasing independently of the audio-path behavior.
- When the user says *"do another comparison"*, treat it as a real audit — a clean
  pass (no fix) is a valid, informative outcome, not a failure.

## This revision vs a later revision of the same lineage (same topology, different values)
- R13/R11 = **39 k** (later revision: 10 k) · emitters R21/R10/R22 = **120 Ω** (later
  revision: 150/100 Ω) · Miller caps C10/C11/C12 = **500 pF** (later revision: 470 pF) ·
  R14 = **100 k** (later revision: 47 k) · several couplings **.68 µF** (later revision:
  some 1 µF / 100 nF) · R5/R8 = **27 k** (later revision: 22 k / 39 k). These are the
  documented values for this specific early revision.

## Model params
2N5133 → silicon NPN (IS = 1 pA, BF = 300 — high hFE for sustain); 1N914 →
IS = 2.52 nA / n = 1.752 (≈ 1N4148). Reuses the same generic BJT (`Q`) + Diode (`D`)
support this toolchain uses for every other circuit. `Rout` (1 MΩ) is an output-reference
resistor, not a schematic part.

## Not modeled
True-bypass switching, transistor/diode unit variance, power-supply filtering
(ideal +9 V rail).
