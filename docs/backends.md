[← back to README](../README.md)

Full backend comparison for parametric-nam (`livespice` vs `ngspice` vs `ltspice-deck`).

# Backends

## Choosing one

| symptom | backend |
|---|---|
| nothing wrong — it converges and the output is physically plausible | **livespice** |
| diverges, or needs extreme `oversample` | **ngspice** (`.schx`-native, no deck needed) |
| converges but the output is **impossible** (bigger than the supply rails allow) | **ltspice-deck** / **ngspice** — see below |
| ngspice can't converge on real playing content at any `maxstep` | **ltspice-deck** |

The third row is the one that costs you a training run, because it is **silent**: divergence
announces itself, wrongness does not. Two questions catch it, and neither is a knob sweep —
knob sweeps are *relative* (dB vs centre) and cannot see a circuit that is uniformly too loud:

1. **Is the absolute output physically possible?** A 9 V pedal cannot output 200 V. Probe absolute
   node voltages at the hot corners, not transfer ratios.
2. **Does the model include the nonlinearity that actually dominates there?** If the real circuit
   spends its time clipping against its rails, a model that cannot saturate is not approximately
   right — it is unbounded.

**LiveSPICE** (`--backend livespice`, default) — real-time-capable, `.schx`-native,
reference tube models, uniform audio-rate output, never aborts. The right default for
essentially everything. But read "never aborts" as a *risk*, not only a feature: its
`IdealOpAmp` has no supply rails and cannot saturate, so a circuit whose dominant nonlinearity
is op-amp clipping renders happily and wrongly. Measured on one real pedal: **228x full
scale** at ordinary knob settings, converging cleanly, passing preflight, with every knob
responding in the correct direction. `measure_truncation.py` reported *textbook* convergence for
it — truncation error tells you how well you solved the equations you wrote, never whether they
were the right equations. See `parametric-devices/backends.toml`, which exists to record exactly
these cases.

**ngspice** (`--backend ngspice`, experimental) — an offline real-SPICE backend with
adaptive timestepping, for the handful of **stiff / very-high-gain** circuits (e.g. a
high-gain amp head) where LiveSPICE's fixed-step solver diverges or needs extreme
oversampling. It translates any `.schx` to an ngspice netlist (exact Dempwolf–Zölzer /
Koren tube models, E+F ideal transformer) and feeds the input via an XSPICE filesource.
Tuning knobs for stiff amps: `--koren`, `--ot-damp`, `--ot-snub`, `--nfb-comp`.
See **[`ngspice/README.md`](ngspice/README.md)** for usage, the convergence findings,
and the important fidelity caveats.

**This safety comes at a real, sometimes severe, speed cost** — adaptive timestepping
means ngspice's solve time grows with the circuit's actual stiffness rather than staying
fixed, and on a genuinely stiff circuit that growth can be dramatic: measured ~2.7x slower
than LTspice on a tweed-style low-gain amp at hard drive (step count exploding to 45492 vs
LTspice's 12543 for the same clip), and on a very-high-gain amp head, ngspice **failed to
converge on a hard-drive render at all** — aborting within microseconds regardless of
timestep, integration method, or input upsampling — while a hand-converted LTspice netlist of
the same circuit (in the separate `ltspice-batch` repo) rendered the same drive/pot position in
~32s. (There is no *generic* schx-to-LTspice translator here, but a deck can still be derived
from a `.schx` rather than hand-written — see ["Two ways to build a deck"](scripts.md#two-ways-to-build-a-deck).) Don't assume a
slow or stuck ngspice render will eventually finish; time-box it and compare against
`livespice` (or, for a hand-written-deck device, `ltspice-deck` below) before spending a
long timeout budget on it.

**Don't confuse this with `preflight.py`/`prepare_excitation.py --backend ngspice-deck`** — a
different flag on different tools, for a different situation. The `--backend ngspice` above
still describes the circuit as a `.schx` file, just solves it with ngspice instead of
LiveSPICE. `ngspice-deck` is for a device that has **no `.schx` at all** — a hand-written
ngspice netlist module (kept in a private devices repo), typically because the circuit needs
something `.schx` has no component for (a real MOSFET) or a feedback loop LiveSPICE's
fixed-timestep solver can't hold at all, not even with `--backend ngspice`'s translation.

**LTspice** (`--backend ltspice-deck` on `preflight.py`/`prepare_excitation.py`/
`grid_adequacy.py`/`check_transient_coverage.py`, or `tools/render_ltspice_deck.py`
directly) — the same deck-module situation as `ngspice-deck`. Two circumstances call for it:
a device whose ngspice deck can't converge on real playing content **at all**, independent of
timestep; and a device where every backend converges but LTspice is the one whose answer is
*stable* (see below).
Found on one real pedal: its ideal tanh-bounded op-amp B-source is a genuine
Newton-solver dead end in ngspice (70/70 renders across the full knob grid timed out, at
every `maxstep` from 3e-6 down to 3e-8), while LTspice gets past it with a real op-amp
macromodel and explicit `.ic`/`uic` initial-condition hints unavailable through ngspice's
B-source style — see `tools/ltspice_spicelib.py`'s module docstring for the full
investigation.

**Also worth reaching for when ngspice *does* converge.** On another real pedal both SPICE
backends bound the output correctly and agree to 1.3% at the clipping corners, and LTspice was
still chosen: ngspice's answers kept moving with settle time where LTspice's did not (centre
0.858 -> 1.137 V going from 0.8 s to 2.5 s of settle, versus 0.9597 -> 0.9590), and it was 2-3x
slower at the hard corners (22.5 s vs 7.1 s at all-max). *A result that changes when you lengthen
the settle has not converged*, whatever the solver reports.

Needs the native LTspice XVII app (`~/Applications/LTspice.app` or `/Applications/LTspice.app`),
not a SPICE binary — and on macOS **which build you install decides whether batch mode works at
all**: see [LTspice on macOS](../README.md#ltspice-on-macos-only-for---backend-ltspice-deck).

`render_backends.py` is the adapter layer `preflight.py` and `prepare_excitation.py` share
across all three hand-deck/schx splits (`NgspiceBackend`, `LtspiceBackend`) — see its module
docstring for the two-method contract a backend implements.
