# ngspice backend (prototype)

An **offline** ngspice dataset-generation backend, as an alternative to
`livespice_cli` for the **stiff / high-gain / combined** circuits where
LiveSPICE's fixed-timestep solver diverges and needs extreme oversampling.

This is a **proof of concept**, not yet wired into `batch_harness`. See the
"Status / next steps" section.

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
- **Pots baked linear** (no taper); **OT** is an approximate coupled-inductor
  model (turns ratio + guessed L/DCR/snubber); **output level not calibrated**.
- **Adaptive timesteps are non-uniform** → a real pipeline must **resample** the
  output to the audio grid. Here `tran` uses a 48 kHz nominal step so output is
  near-uniform, but interpolation is still needed for exact alignment.

## Usage

```bash
brew install ngspice                      # one-time
python3 gen_evh5150_ngspice.py            # writes evh5150.cir next to the script
ngspice -b evh5150.cir                    # -> evh5150_out.csv (ascii, via wrdata)
```

Env flags: `METHOD=trap|gear` (default trap), `PREAMP_ONLY=1` (drop the power
amp), `NO_NFB=1` (open the PI-grid feedback), `FLIP_SEC=1` (reverse OT secondary
— diagnostic).

## Status / next steps

Prototype only. To become a real backend:

1. **Generalize the translator**: `.schx` → ngspice netlist for any circuit
   (this script hardcodes the 5150). Reuse the component/pot/tube mapping here.
2. **Validate tube-model fidelity** against LiveSPICE reference renders (the
   dataset's whole value is matching the target circuit's tone).
3. **Uniform resampling** of the non-uniform transient output to 48 kHz.
4. **Wire into `batch_harness`** as an alternate backend, selected per circuit
   (LiveSPICE default; ngspice for the stiff/combined cases).

## When to prefer which backend

Use **LiveSPICE** by default — it's real-time-capable, `.schx`-native, its tube
models are the reference, it emits uniform audio-rate output, and it never
aborts (deterministic for batch automation). Reach for **ngspice** only for the
handful of stiff/high-gain/combined circuits (like the 5150 Lead full) where
LiveSPICE diverges or needs extreme oversampling.
