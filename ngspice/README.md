# ngspice backend (prototype)

An **offline** ngspice dataset-generation backend, as an alternative to
`livespice_cli` for the **stiff / high-gain / combined** circuits where
LiveSPICE's fixed-timestep solver diverges and needs extreme oversampling.

**Now wired into `batch_harness`** as `--backend ngspice` (see "Status" below).
`gen_evh5150_ngspice.py` remains as the original standalone prototype; the
general path is `schx_to_ngspice.py` (any `.schx`) driven by the harness.

## Why

LiveSPICE uses a fixed timestep. On very high-gain circuits, Newton-Raphson
overshoots the tube saturation knee and, with no adaptive step, runs away. The
**EVH 5150 Lead full amp** is the worst case: it diverges to ~+95 dB at the
default 2× oversample and needs **`--oversample 32`** (~158 s per 5 s of audio)
to stay bounded.

Real SPICE (ngspice) uses **adaptive timestepping + damped Newton**, which is
exactly the setting that offline dataset generation can afford (no real-time
deadline). It converges on the same circuit with **no oversample hack**.

## Result (5150 Lead full)

`gen_evh5150_ngspice.py` translates the LiveSPICE 5150 Lead full circuit
(`spice-circuits/amps/gen_evh5150_full.py`) to an ngspice netlist and runs it:

| | LiveSPICE (fixed step) | ngspice (adaptive) |
|---|---|---|
| at 2× oversample | **+95 dB runaway** | n/a |
| to get a stable render | **32× oversample** (~158 s / 5 s audio) | **just works** (~2 s / 0.3 s audio) |
| when it can't step | diverges to garbage | **fails safe** (bounded + aborts) |

The full amp completes the whole transient, **bounded** (speaker ±4.3 V; the
preamp internally slams ±124 V — a genuine cranked-amp sim), roughly **~5–6×
faster** than the LiveSPICE 32× workaround, and correct.

## What's in the netlist

- **9× 12AX7** (Koren triode subckt) + **4× 6L6GC** (Koren pentode subckt),
  params ported from LiveSPICE's Dempwolf-Zölzer/Koren values.
- FMV tone stack, baked pots (two resistors each), interstage ÷22 attenuator,
  12AX7 LTP phase inverter, global NFB, coupled-inductor output transformer.
- Input: a hard 80→6000 Hz chirp at 0.3 V (a torture signal, like the sweeps
  that hang LiveSPICE).

## Key findings

1. **The tube cascade was never the hard part for ngspice** — DC operating point
   solves cleanly, the 6-stage preamp runs trivially. The hard part is the
   **coupled-inductor OT + global NFB loop**.
2. **Integration method matters most.** With `method=gear` it aborts at the NFB
   node (~45 ms); with **`method=trap`** it completes. Trap is the default here.
3. **Realistic OT damping is required** — near-ideal coupled inductors + global
   NFB oscillate and force the timestep to zero. The model adds winding DCR, a
   secondary snubber, and a plate-to-plate resistive damper.
4. **It never diverges** — always bounded, even when it can't finish. That is the
   qualitative win over a fixed-step solver (no +95 dB garbage).

## Caveats — this proves CONVERGENCE, not fidelity

- **Tube models** are Koren with ported params — close but not bit-identical to
  LiveSPICE's. Validate against a LiveSPICE reference render before trusting a
  dataset's tone.
- ~~Pots baked linear~~ **FIXED**: the translator now applies the `.schx` `Sweep`
  taper (log/anti-log/sigmoid) via a faithful port of LiveSPICE's `AdjustWipe` —
  a log gain pot at dial 0.5 correctly bakes to ~0.27, not 0.5.
- **µF values**: LiveSPICE's parser misread U+00B5 micro signs as no-prefix
  (`"1 µF"` → 1 FARAD). Fixed by the vendored patch
  (`patches/livespice-microsign-prefix.patch`) + base-unit numeric netlist
  emission. **Apply the patch before building** or every µ-valued circuit is wrong.
- **OT** is an approximate coupled-inductor
  model (turns ratio + guessed L/DCR/snubber); **output level not calibrated**.
- **Adaptive timesteps are non-uniform** → a real pipeline must **resample** the
  output to the audio grid. Here `tran` uses a 48 kHz nominal step so output is
  near-uniform, but interpolation is still needed for exact alignment.
- **ALIASING (fixed via `--oversample`).** The transient output carries real
  ultrasonic content (a 4× solve resolves energy past 100 kHz); a plain `np.interp`
  to 48 kHz folds everything above 24 kHz back into the audband — **~20% RMS**,
  piled up near Nyquist. `--oversample N` now solves at N×48 kHz (finer `tran` step)
  and `_run_ngspice` FIR-decimates back to 48 kHz, filtering >24 kHz before it folds.
  **Default is 1× (naive) — pass `--oversample 2` (or more) to enable anti-aliasing.**
  Verified 2× on the 5150: near-Nyquist (18–24 kHz) 12.3%→9.2%. Note the aliasing is
  mostly above a cab's rolloff (largely inaudible cab'd) but inflates the full-band
  ESR. See [`../docs/evh5150_training_notes.md`](../docs/evh5150_training_notes.md).
- **MEMORY — long inputs need `save` (now automatic) + a sane `--workers`.** ngspice
  keeps *every node's* full transient in RAM until it writes. The translator now
  emits `save v(<out>)` so only the probed node is stored (~150 MB for 120 s @ 2×
  vs ~13 GB for all nodes). Even so, the Python side loads a large ASCII CSV per
  perm (~400 MB / 15 M rows for 120 s @ 2×, ~2 GB peak/worker), so cap `--workers`
  to `RAM / ~2 GB` on long + oversampled runs (e.g. 4 on a 30 GB box). 120 s @ 2× ×
  5 workers **without** the `save` fix OOMed a 30 GB machine.

## Usage

```bash
brew install ngspice                      # one-time
python3 gen_evh5150_ngspice.py            # writes evh5150.cir next to the script
ngspice -b evh5150.cir                    # -> evh5150_out.csv (ascii, via wrdata)
```

Env flags: `METHOD=trap|gear` (default trap), `PREAMP_ONLY=1` (drop the power
amp), `NO_NFB=1` (open the PI-grid feedback), `FLIP_SEC=1` (reverse OT secondary
— diagnostic).

## Status

Wired end to end. Usage:

```bash
# moderate amp (e.g. JCM800 power amp) — exact DempwolfZolzer tubes:
python batch_harness.py --backend ngspice --schx "<amp>.schx" \
    --knobs presence --values 0.2,0.5,0.8 --input guitar.wav --output ds

# stiff amp (EVH 5150 Lead full) — Koren tubes + OT damping + NFB comp.
# NOTE: bound leadpost>=0.05 (master at exactly 0 grounds the PI -> degenerate DC
# operating point -> the solver hangs). presence excluded (subtle + NFB-comp alters it).
# (config "L" — the re-tuned minimal recipe: no NFB-comp cap; it is stable
#  without one and the cap is a tone-shaper, not just a stabilizer)
python batch_harness.py --backend ngspice --koren --ot-damp 3k --ot-snub 100n \
    --oversample 2 --schx "EVH 5150 Lead Full.schx" \
    --knobs leadpre,leadpost,high,low,mid --bounds leadpost=0.05,1.0 \
    --fixed-params "Presence=0.5" --random 200 --input guitar.wav --output ds
```

Full write-up of the 5150 run (convergence recipe, training, aliasing, the
generalization gap, knob analysis): **[`../docs/evh5150_training_notes.md`](../docs/evh5150_training_notes.md)**.

Done:
1. ✅ **General translator** — `livespice_cli --netlist` (authoritative parse) →
   `schx_to_ngspice.py` (exact DempwolfZolzer triodes / Koren pentodes, baked
   pots, **E+F ideal transformer**, tunable OT damping + NFB comp, `--koren`).
2. ✅ **Resampling, with optional anti-aliasing** — non-uniform transient output →
   48 kHz grid; `--oversample N` solves at N×48 kHz and FIR-decimates back (removes
   the ~20% aliasing; default 1× is naive).
3. ✅ **XSPICE filesource** input — feeds full-length WAVs (no giant inline PWL).
4. ✅ **`batch_harness --backend ngspice`** — per-perm translate → run → resample
   → `.npy`, with the crest-factor divergence check.
5. ✅ **Bounded memory** — `save v(<out>)` stores only the probed node, so long +
   oversampled runs don't OOM (see Caveats for the `--workers` guidance).

Result: the **EVH 5150 Lead full converges** (Koren mode) where LiveSPICE
diverges even at 32× oversample. The JCM800 power amp matches a LiveSPICE render
at **~0.99 real-guitar correlation** with the default DempwolfZolzer tubes.

Open:
- **Anti-aliasing must be opted into** — default `--oversample 1` still aliases; use
  `2`+ for real datasets. Its ~1.5–2× solve cost + memory (see Caveats) is the trade.
- **Fidelity is signal- and circuit-dependent**: ~0.99 on the JCM800 (validated vs
  LiveSPICE). The **5150 full has now been trained + listened to** (no LiveSPICE
  reference exists): a 5-knob model reached validation ESR ~0.19 but **~0.43 on a
  new DI** — "sounds like the amp, not production-close." Root causes: aliasing +
  training-input under-coverage (a 30 s sweep slice) + 5-knob capacity spread. Full
  analysis + improvement plan in [`../docs/evh5150_training_notes.md`](../docs/evh5150_training_notes.md).
- **Per-circuit tuning is manual** — `--ot-damp`/`--nfb-comp` values and the NFB
  node name are hand-set for the 5150; not auto-derived.
- Tube **inter-electrode capacitances** and **pentode grid current** are omitted
  (small; would tighten the JCM800's last ~1% and level offset).
- **The Koren-tubes + heavy-OT-damping compromise is not an ngspice limitation** —
  evaluated [Xyce](https://xyce.sandia.gov/) (more nonlinear continuation/homotopy
  methods) as a potential way to use exact tubes + light damping instead. Verdict:
  **negative, structurally** — the 5150's failure is a *transient* integration
  instability, and Xyce's continuation methods only apply to *DC operating point*
  solving (confirmed in Xyce's own source). Full test + build notes:
  [`../docs/xyce_build_notes.md`](../docs/xyce_build_notes.md).

## When to prefer which backend

Use **LiveSPICE** by default — it's real-time-capable, `.schx`-native, its tube
models are the reference, it emits uniform audio-rate output, and it never
aborts (deterministic for batch automation). Reach for **ngspice** only for the
handful of stiff/high-gain/combined circuits (like the 5150 Lead full) where
LiveSPICE diverges or needs extreme oversampling.
