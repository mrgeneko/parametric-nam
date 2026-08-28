# parametric-nam

Toolchain for training **parametric** Neural Amp Modeler (NAM) files — `.nam` models whose
knobs respond live to real knob settings, instead of being baked in at training time — from
either a SPICE circuit schematic or a set of real hardware captures. The resulting
`.param.nam` captures a circuit or device's behavior across its **full knob range** —
enabling real-time parametric inference in a compatible real-time host at a fraction of the
CPU cost of live simulation.

Two ways to build the same underlying dataset:

- **From a `.schx`** (e.g. an amp or pedal from `LiveSPICE-Amp-Collection`) — simulates the
  circuit across a sweep of knob settings, no real hardware needed.
- **From real hardware, no SPICE model at all** (`gen_dataset_from_captures.py`) — a folder of
  already-captured, fixed-setting files, one per knob setting: either existing `.nam` files,
  or your own recorded wet/dry `.wav` pairs (a shared sweep played through the real device and
  captured back via an audio interface). Each file's **name** encodes its knob settings, e.g.
  `"MyAmp G2, B5, M5, T5.nam"` → `Gain=0.2, Bass=0.5, Mids=0.5, Treble=0.5` (comma-separated
  `PrefixDigit` tokens, digit → value/10) — see `gen_dataset_from_captures.py` in
  [`docs/scripts.md`](docs/scripts.md) for the full naming convention and how to override it.

Either path feeds the same pipeline: train a FiLM-conditioned WaveNet on the paired audio, and
export a single `.param.nam` whose knobs match the real controls.

---

## Quick Start

### Try it now

`examples/muff/` holds a real, complete recipe: the `.schx` circuit and an annotated
`config.toml`. Two things are not in the repo:

1. **The oracle** (`livespice-cli`) — a small public sibling repo that simulates the circuit.
   Needs the .NET SDK to build.
2. **The sweep file** — download `T3K-sweep-v3.wav` from
   **<https://www.tone3000.com/capture>** into `examples/`. Feel free to substitute your own
   favorite sweep or DI recording instead — just point `input` in the config at it. A good
   input covers the top of the band (real playing alone rarely does) and includes real
   transient attacks, not just steady tones.

```bash
git clone https://github.com/mrgeneko/parametric-nam
git clone --recurse-submodules https://github.com/mrgeneko/livespice-cli   # as a SIBLING of parametric-nam
cd livespice-cli && ./build.sh && cd ..
# .NET 10 SDK required for the build: `apt install dotnet-sdk-10.0` (Linux) / `brew install dotnet` (macOS)
# / see https://dotnet.microsoft.com/download for Windows or other install methods

cd parametric-nam && ./setup.sh && . .venv/bin/activate
# download T3K-sweep-v3.wav (see above) into examples/ before running this:

python run_pipeline.py --config examples/muff/config.toml \
    --dataset-dir /tmp/muff_ds --nam-output /tmp/muff.param.nam \
    --checkpoint-dir /tmp/muff_ckpt
```

This trains an actual muff model end to end. See `examples/muff/muff.md` for the circuit notes.

### Bring your own circuit

The bundled example above needs nothing beyond what it already cloned. To train your own
circuit, all you need on top of that is your own `.schx` (e.g. from
`LiveSPICE-Amp-Collection`, or one you built yourself) and a `config.toml` recipe for it.
`scaffold_config.py --schx yours.schx` discovers its real controls and measures a
starting `oversample` for you (see [`docs/scripts.md`](docs/scripts.md)) — or copy
`examples/template.config.toml`
by hand (see **Per-circuit configs** below for the format). Either way, finish with
`grid_adequacy.py --config ... --apply` to turn the placeholder knob grid into a
measured one. No other repos are required.

`setup.sh` builds the oracle from `../livespice-cli` (needs the .NET SDK; `--no-cli` to
skip). If your checkout isn't a sibling of this repo, point at it explicitly:

```bash
export LIVESPICE_CLI=/path/to/livespice-cli/publish/livespice_cli
```

### Train a device

```bash
python run_pipeline.py --config path/to/your-device/config.toml \
    --dataset-dir    /tmp/ds \
    --nam-output     /tmp/model.param.nam \
    --checkpoint-dir /tmp/ckpt
```

One command runs **four** steps:

| step | what it does | why it exists |
|---|---|---|
| **0 — Grid adequacy** | renders each knob cell's midpoint and checks the grid can represent the target ESR | a too-coarse cell puts a floor under the model that **no training can lift**. Cheap either way — render count scales with knob-axis count, not the full permutation count (~15–200 across the real device fleet) — but not a fixed time: cost per render follows `--oversample` and backend (`ngspice` is markedly slower than `livespice`). |
| 1 — Generate | renders the dataset across the knob grid | |
| 2 — Combine | per-permutation WAVs → `outputs.npy` | |
| 3 — Train | FiLM-conditioned WaveNet → `.param.nam` + release folder | prints the **training budget** in gradient steps |

**Adding a *new* device to the private fleet?** That workflow (and the toolchain's other
methodology docs — grid adequacy, convergence, training budget, retraining, architecture,
lessons learned) lives in a private internal notes repo, since it's specific to maintaining
that fleet rather than to using this toolchain standalone.

### Listen

```bash
python param_infer.py --checkpoint /tmp/ckpt/best.pt \
    --input dry.wav --output-dir /tmp/out/ --params "Sustain=0.7,Tone=0.4"
```

For real-time, interactive listening — turning the knobs live instead of re-rendering a file
per setting — load the exported `.param.nam` into
[**NAMix**](https://github.com/mrgeneko/NAMix), a free, cross-platform (Linux/macOS/Windows)
VST3/AU/standalone plugin built specifically to host parametric models like the ones this
toolchain produces: a model's own knobs surface directly in the plugin UI, driven by
[our NeuralAmpModelerCore fork](https://github.com/mrgeneko/NeuralAmpModelerCore)'s parametric
(FiLM-conditioning) support. NAMix's releases also ship several demo `.param.nam` files if you
just want to try the knob experience before training your own.

## How it works

```
.schx schematic                             fixed-setting captures (.nam export or raw .wav)
    ↓  gen_dataset_from_schx.py                  ↓  gen_dataset_from_captures.py
    (simulate: sweep × N permutations)           (align each capture to a shared sweep, 1/file)
                        ╲                        ╱
                         Paired audio dataset (config.json, sweep.wav, outputs.npy, params.csv)
                             ↓  param_train.py  (train A2 Lite 4ch + Full 8ch jointly, FiLM knob conditioning)
                         .param.nam  (SlimmableContainer)
                             ↓  param_infer.py  (PyTorch inference at arbitrary knob positions — no C++)
                             ↓  NeuralAmpModelerCore  (ParametricWaveNet factory, C++ inference)
                         Real-time inference in a compatible host app
```

Two ways to reach the same dataset contract: simulate a circuit, or point at a folder of
real captures — see `gen_dataset_from_captures.py` in [`docs/scripts.md`](docs/scripts.md).
`run_pipeline.py` orchestrates the `.schx` path's three steps in one command; you can also run
each script directly (see `docs/scripts.md`).

---

## `run_pipeline.py` — the primary entry point

Runs **generate → combine → train** as one command, with per-step timing, a durable
**release folder**, and macOS/Linux **sleep-inhibition** for long runs. Steps auto-skip
when their outputs exist (`--skip-generate/-combine/-train`, `--force-generate`).

```bash
python run_pipeline.py \
    --dataset-dir <ds> --nam-output <model.param.nam> --checkpoint-dir <ckpt> \
    --backend livespice --schx "<amp>.schx" \
    --knobs gain,tone,volume --random 200 --bounds tone=0.15,0.85 \
    --input guitar.wav --mmap \
    --crop-len 24000 --epochs 0 --restart-period 50 --batch-size 64
```

- `--epochs 0` = **open-ended training** (see param_train) — runs until you
  `touch <ckpt>/STOP`, exporting the best model continuously.
- **Monitor progress**: every step's output streams, timestamped, to both the terminal
  and `<dataset-dir>/pipeline.log` (`--log` to put it elsewhere) — `tail -f
  <dataset-dir>/pipeline.log` from another terminal to follow a long run, including
  live training epochs once it reaches that step. `<checkpoint-dir>/metrics.csv` has
  the same training progress in structured, per-epoch form (val ESR per tier).
- On completion it builds `<nam-output>_release/`: the best-full & best-lite
  `.param.nam`, the `.schx`, `metrics.csv`, a provenance `MANIFEST.md` (params +
  timing + hardware + git revs), and a runnable `reproduce.sh`. `--no-release` skips it.
- Generation flags (`--random`, `--bounds`, `--oversample`, ngspice `--koren` etc.)
  are forwarded to `gen_dataset_from_schx`; training flags to `param_train`.

### Per-circuit configs (`--config`)

A circuit's *recipe* — schematic, input, knob grid, fixed params, widths, and
hyperparameters — lives in a declarative TOML, so the pipeline stays generic (one
`run_pipeline.py`, one small reviewable file per circuit) instead of a script per
`.schx`. Only the per-run output paths stay on the CLI:

```bash
python run_pipeline.py --config /path/to/device-config.toml \
    --dataset-dir /tmp/ds --nam-output /tmp/model.param.nam \
    --checkpoint-dir /tmp/ckpt
```

Any CLI flag overrides the config (config is loaded as argparse defaults). The `[knobs]`
table (`NAME = [v1, v2, …]`) expands to `--knobs`/`--range`, `[fixed]` to
`--fixed-params`, and `widths = [3,4,8]` to `--widths`.

**This repo carries no per-device configs of its own** beyond the bundled example — a
`config.toml` is a *living* recipe you maintain alongside your own circuits, distinct from
a specific run's *frozen* `reproduce.sh` output. `examples/muff/config.toml` is a fully
worked, annotated example of the format; `examples/template.config.toml` is a minimal
blank one to copy for a new device.

## Scripts

Full per-script reference (usage, flags, design rationale) lives in
[`docs/scripts.md`](docs/scripts.md). Quick index:

| Script | Purpose |
|---|---|
| `scaffold_config.py` | Generate a starting `config.toml` for a new circuit |
| `grid_adequacy.py` | Measure whether a knob grid is dense enough |
| `measure_truncation.py` | Measure BDF2 truncation error, pick `oversample` |
| `build_excitation.py` | Build a training excitation that covers the full input range |
| `gen_dataset_from_schx.py` | Generate the dataset from a `.schx` |
| `render_ngspice_deck.py` | Render a hand-written ngspice deck's knob sweep |
| `render_ltspice_deck.py` | Render an LTspice deck's knob sweep |
| `gen_dataset_from_captures.py` | Build a dataset from real hardware captures, no `.schx` needed |
| `param_train.py` | Train a parametric NAM |
| `param_infer.py` | Inference (Python, no C++) |
| `coverage_report.py` | Find holes in a sampled dataset |
| `sweep_report.py` | Visualize a set of WAVs at different knob settings |
| `export_checkpoint.py` | Checkpoint → `.param.nam` |
| `bake_nam.py` | Freeze a knob setting for stock/upstream NAM plugins |
| `plot_tone_response.py` | Frequency-response chart for an exported `.nam` |
| `release_run.sh` | Verify + stage a finished run |

---

## Backends

Three simulation backends: **`livespice`** (default, real-time-capable, `.schx`-native),
**`ngspice`** (offline, adaptive-timestep, for stiff/high-gain circuits), and **`ltspice-deck`**
(for circuits ngspice can't converge on, or where LTspice's answer is the stable one). Full
comparison, when to reach for each, and the real measured tradeoffs are in
[`docs/backends.md`](docs/backends.md).

---

## Known Issues

Four documented failure modes and their investigations/fixes live in
[`docs/known-issues.md`](docs/known-issues.md):

- **Rare knob-corner blowup (FiLM/LeakyReLU runaway)** — a trained model spikes to tens or
  hundreds of times its normal peak at a specific, narrow knob-grid corner.
- **Excitation needs a silent lead-in** — a cold-start settling transient can corrupt renders
  for circuits with a slow-charging DC-blocking network.
- **`preflight.py`'s EQ-knob checks can be swamped by the circuit's own gain control** — a
  high-gain device's default probe settings can produce a false REVERSED/DEAD knob-direction
  reading.
- **Audible knob-move transient noise** (inference-side, not a training/dataset bug) — root
  cause is understood (network ring-buffer state across a conditioning change), not yet fixed.

---

## What a run produces

A **dataset** directory (`gen_dataset_from_schx --combine` format):
```
config.json   # {"knobs": [...], "bounds": {...}, "param_map": {...}, ...}
sweep.wav     # the dry input used
outputs.npy   # [N_perms, N_samples] float32
params.csv    # idx, knob1, ..., rms, peak, ok, error
```

A **release folder** (`<nam-output>_release/`, built by `run_pipeline`): every
per-tier `.best_<tier>.param.nam` plus the composite `.optimal.param.nam`, the
`.schx`, `metrics.csv`, `MANIFEST.md` (provenance: params, per-step timing, hardware,
git revisions), and a runnable `reproduce.sh`.

---

## Reference

The `.param.nam` file format, architecture details (weight layout, A2 layer structure), the
now-removed LoRA-style knob conditioning's history, and NAM ecosystem compatibility notes are
all in [`docs/reference.md`](docs/reference.md).

## System Requirements

Training parametric NAM models is GPU- and memory-intensive. Real device grids run into
hundreds or thousands of permutations (e.g. a real 5-knob amp's 1944), and real training
budgets run into the tens of thousands of gradient steps (see `--target-steps` above) —
CPU-only training isn't realistically feasible for a real run, not just slow. `--device auto`
(the default) refuses to start if no GPU/MPS device is detected, rather than silently queuing
a run that will never finish — pass `--device cpu` explicitly if you genuinely want to (e.g. a
short debugging run).

**This repo has mostly been developed and tested on Apple Silicon Macs with 24 GB+ unified
memory** (`--device auto` picks `mps` there). It has not been systematically benchmarked
across the NVIDIA/AMD VRAM spectrum, so treat any specific VRAM number below as a rough,
anecdotal data point, not a validated requirement. If memory is tight on your hardware,
training a single, narrower `--widths` value (instead of the default multi-tier slimmable
config) uses meaningfully less memory — `SlimmableParametricA2.forward()` runs every trained
tier's forward pass jointly, so VRAM scales with how many widths you train at once, and a
single width skips that entirely.

### Notes for AMD GPU users (anecdotal — not the main tested path)

PyTorch's ROCm build may lack compiled kernels for certain GPU architectures.
If you encounter `rocBLAS` errors or `illegal memory access` on an AMD GPU using the
gfx1034 architecture, two workarounds are needed:

1. **Missing `TensileLibrary.dat`** — create a symlink to the fallback database:
   ```bash
   ln -s TensileLibrary_Type_4xi8I_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_fallback.dat \
         /path/to/torch/lib/rocblas/library/TensileLibrary.dat
   ```

2. **Override GPU architecture** — set this environment variable before running
   training (maps gfx1034 → gfx1030, which has precompiled kernels):
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0
   ```

3. **Symlink missing architecture kernels** — create gfx1034 → gfx1030 symlinks
   so rocBLAS and MIOpen find suitable code objects for the actual GPU:
   ```bash
   LIBDIR="/path/to/torch/lib/rocblas/library"
   for f in "$LIBDIR"/*gfx1030*; do
     base=$(basename "$f")
     target="${base/gfx1030/gfx1034}"
     [ ! -e "$LIBDIR/$target" ] && ln -s "$base" "$LIBDIR/$target"
   done
   ```

---

## Build & Dependencies

**The easy path is `./setup.sh`** (see Quick Start) — it does everything below
idempotently. The manual steps are documented here for reference / troubleshooting.

### Python packages
Pinned in [`requirements.txt`](requirements.txt) (torch, numpy, scipy, soundfile):
```bash
pip install -r requirements.txt
```
For NVIDIA CUDA, install the matching `torch` build from the PyTorch index first
(see the note at the top of `requirements.txt`), then `pip install -r requirements.txt`.

### .NET SDK + the oracle (for dataset generation)

`livespice_cli` — **the oracle** — lives in
[**`mrgeneko/livespice-cli`**](https://github.com/mrgeneko/livespice-cli), a small standalone
public repo, not here. This repo used to carry its own near-copy, and the two drifted, as
duplicates do: a fix making an unknown `--params` name a hard error (instead of silently
rendering at the defaults, so every "swept" permutation comes out **identical** and the trainer
learns the knob does nothing) landed in one copy and not the other. Two tools built from one
schematic, disagreeing about what the device's knobs *are*. There is now exactly one. (It was
originally extracted from `hotspice/oracle/` — same reasoning, promoted to its own repo since it
has no other functional connection to this project or to hotspice's emitter.)

```bash
git clone --recurse-submodules https://github.com/mrgeneko/livespice-cli   # as a SIBLING of this repo
cd livespice-cli && ./build.sh
```
(.NET 10 SDK: `apt install dotnet-sdk-10.0` on Linux, `brew install dotnet` on macOS, or see
Microsoft's own installer/instructions at <https://dotnet.microsoft.com/download> for Windows or
any other install method. `--recurse-submodules` matters — LiveSPICE has its own nested
submodule; a plain clone leaves it empty and the build fails.)

`gen_dataset_from_schx.py` resolves the binary in this order: **`$LIVESPICE_CLI`** → a sibling
`../livespice-cli/publish/livespice_cli`. Set the env var if your checkout lives
elsewhere.

The oracle builds against **pristine upstream LiveSPICE**, never a patched fork — an oracle built
from the thing under test is not an oracle — and `build.sh` warns if it finds fork markers in the
submodule (a sign the pin points somewhere other than pristine upstream).

> **The micro-sign patch is gone, and deliberately.** LiveSPICE's `Quantity` parser knows
> `U+03BC GREEK MU` (and ASCII `u`) but *not* `U+00B5 MICRO SIGN`, so upstream reads `4.7µF`
> written with U+00B5 as **4.7 farads** — no throw, no warning, every capacitor a dead short, and
> the simulation confidently wrong. We used to patch a vendored LiveSPICE fork, which is
> incompatible with keeping the oracle pristine. `livespice_cli` now **normalises U+00B5 → U+03BC
> when it loads the schematic**, so both encodings render bit-identically and no fork is needed.
> (Our library currently uses U+03BC in 6 of 23 files and U+00B5 in none — but U+00B5 is what most
> editors emit, so this was one save-from-the-wrong-tool away from a corrupted dataset.)

### ngspice (only for `--backend ngspice`)
```bash
apt install ngspice        # Linux;  macOS: brew install ngspice
```

### LTspice on macOS (only for `--backend ltspice-deck`)

**Install the native build from analog.com — NOT `brew install --cask ltspice`, and NOT the
current `LTspice_26.pkg`.** Both of those are the *Windows* binary wrapped in CrossOver/Wine.
The GUI launches and looks fine, which is what makes this worth writing down: `-b` batch mode
then produces **no `.raw`, no `.log`, no output of any kind, and exit status 0** — even on a
five-line RC netlist. There is nothing to debug against, and it looks exactly like a circuit
that failed to converge. Verified 2026-08-27 against LTspice 26.0.2.1 (`NSPrincipalClass =
CXApplication` in its `Info.plist` is the giveaway; a Wine-backed bottle also shows up as
`~/Library/Application Support/LTspice/Bottles/`).

The working install is **LTspice XVII 17.2.4**, ADI's last native macOS release ("Download for
MacOS 10.15 and forward"), a universal arm64/x86_64 binary. It runs correctly on macOS 26.

Three requirements, each of which silently breaks it if missed:

1. **The bundle must be named exactly `LTspice.app`.** It hardcodes its own name internally.
   Renamed — even to something as close as `LTspice17.app` — every launch dies at startup with
   `NSInvalidArgumentException ... 'data parameter is nil'` from `newJSONValue`, before it
   reads your netlist. The crash names JSON, so it reads like a corrupt install or a failed
   update check; it is neither.
2. **`~/Library/Application Support/LTspice/lib/sub/` must exist**, holding
   `UniversalOpAmp2.lib` and friends. The installer's `postinstall` unpacks `lib.zip` there.
   `gen_ocd_ltspice.py` and `gen_joyo_ltspice.py` reference that path directly.
3. **If a newer `LTspice.app` is already in `/Applications`, the 17.2.4 installer will not
   replace it** — macOS Installer refuses to overwrite a bundle carrying a higher
   `CFBundleVersion`, so 26.0.2.1 blocks it. It fails *silently*: the receipt registers
   (`pkgutil --pkg-info com.analog.LTspice` reports 17.2.4) and `lib.zip` lands, but the app
   is untouched. Check what you actually have before trusting the receipt:

```bash
plutil -p /Applications/LTspice.app/Contents/Info.plist | grep -E "ShortVersion|PrincipalClass"
#   want: "17.2.4"     and  NSPrincipalClass => "LTapplication"
#   wrong: "26.0.2.1"  and  NSPrincipalClass => "CXApplication"   <- Wine, batch mode is dead
```

If 26 is in the way and you want to keep it, install 17.2.4 to `~/Applications/LTspice.app`
instead — the name is what matters, not the directory, and
`ltspice_spicelib._find_ltspice_bin()` searches `~/Applications` **first**, then
`/Applications`, then honours an explicit `LTSPICE_BIN` env var override. Extract the app from
the pkg without running the installer:

```bash
pkgutil --expand-full ~/Downloads/LTspice.pkg /tmp/ltpkg17
ditto /tmp/ltpkg17/LTspice.pkg/Payload/Applications/LTspice.app ~/Applications/LTspice.app
xattr -dr com.apple.quarantine ~/Applications/LTspice.app
```

Smoke-test batch mode before blaming a circuit — this must write `rc.raw` and `rc.log`:

```bash
printf '* rc\nV1 in 0 SINE(0 1 1k)\nR1 in out 1k\nC1 out 0 100n\n.tran 0 5m 0 10u\n.end\n' > /tmp/rc.net
~/Applications/LTspice.app/Contents/MacOS/LTspice -b /tmp/rc.net && ls /tmp/rc.raw /tmp/rc.log
```

If you have a real `gen_<device>_ltspice.py` deck module to test against, render it
end-to-end the same way as a final check:

```bash
python render_ltspice_deck.py --pedal-dir path/to/your/devices \
    --module gen_mydevice_ltspice --tap spk --absolute --out-scale 0.1 tone.wav out.wav \
    --knob Drive=1 --knob Tone=1 --knob Level=1
# Compare the reported peak against what the circuit should produce at max drive -- a peak
# in the hundreds (rather than a plausible few volts) means LiveSPICE, not LTspice, rendered
# it (see docs/backends.md): IdealOpAmp has no supply rails and cannot saturate.
```

One gotcha that is not LTspice's fault: `.net` (or `.cir`) files must be plain SPICE netlists,
and LTspice **requires braces** around B-source expressions (`B1 a b V={min(...)}`), unlike
ngspice where they are optional.

### NeuralAmpModelerCore (C++ inference only)
Not required for training or Python inference — only for C++ inference validation and
host-app integration.

### `neural-amp-modeler` (optional — `.nam`-sourced captures and real `.wav` delay calibration)
Not required for the `.schx` path, or for a `.wav`-sourced dataset without real calibration
(it just falls back to a disclosed `delay=0` — see `gen_dataset_from_captures.py`'s
`detect_delay`). Needed only for **loading an existing `.nam` file as a capture source**
and for **real NAM blip-based latency calibration** on a `.wav` capture. Build as a sibling
of `livespice-cli`, not inside this repo:

```bash
git clone https://github.com/sdatkinson/neural-amp-modeler ../neural-amp-modeler
cd ../neural-amp-modeler && python3 -m venv venv && ./venv/bin/pip install -e .
```

Auto-discovered from `~/work/neural-amp-modeler/venv` (same convention
`capture_static.py` already uses); override with `$NEURAL_AMP_MODELER_HOME` if yours lives
elsewhere. The delay-calibration half runs in a **subprocess** under that venv's own
interpreter, not a lazy in-process import — `nam.train.core` depends on `numba`, which
checks `numpy`'s version at import time, and importing it in-process would hand it
whatever `numpy` *this* repo's own venv already loaded, not the sibling venv's compatible
one. A subprocess avoids that entirely; see `nam_delay_helper.py`.

> **Input formats**: any WAV (16/24/32-bit) and any content read correctly — an earlier
> O(N²) bug in `livespice_cli`'s WAV reader (fixed in `fa90d57`) once made non-24-bit
> inputs stall.

## Related Repos

- `LiveSPICE-Amp-Collection` — `.schx` circuit library
- [`livespice-cli`](https://github.com/mrgeneko/livespice-cli) — the oracle (`livespice_cli`), public
- [`NeuralAmpModelerCore`](https://github.com/mrgeneko/NeuralAmpModelerCore) — our fork,
  public. Adds the `"ParametricWaveNet"` architecture + live knob support on top of Steven
  Atkinson's original DSP core.
- Sample host apps with working parametric-knob support, both public, both built against
  that fork: [`NAMix`](https://github.com/mrgeneko/NAMix) (Linux VST3/standalone) and
  [`NeuralAmpModelerPlugin`](https://github.com/mrgeneko/NeuralAmpModelerPlugin) — rebranded
  "Anti-Static" — (macOS/Windows VST3/AU/standalone)

## Credits & Attribution

This toolchain builds on several open-source projects and published models:

- **[LiveSPICE](https://github.com/dsharlet/LiveSPICE)** (Dillon Sharlet, MIT) — the
  circuit simulator `livespice_cli` builds against (pristine, pinned in
  [`livespice-cli`](https://github.com/mrgeneko/livespice-cli)'s `extern/LiveSPICE` submodule),
  and the reference for the tube-model equations. The ngspice backend's tube
  subcircuits are **ported from LiveSPICE's `Triode.cs` / `Pentode.cs`**.
- **[ngspice](https://ngspice.sourceforge.io/)** (BSD) — the adaptive-timestep SPICE
  engine used by the experimental `--backend ngspice`, including the **XSPICE** code
  models (originally Georgia Tech Research Institute) whose `filesource` feeds the input.
- **Tube models** (as implemented by LiveSPICE and ported here):
  - **Norman Koren**'s SPICE triode/pentode model — the `--koren` mode and all pentodes.
  - **Dempwolf–Zölzer** triode model — K. Dempwolf & U. Zölzer, *"A physically-motivated
    triode model for circuit simulations,"* Proc. DAFx-11 (2011) — the default triode model.
- **[Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)** /
  **[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)**
  (Steven Atkinson, MIT) — the `.nam` format, the WaveNet ("A2") architecture, and the
  `SlimmableContainer` that this project targets.
- **[auraloss](https://github.com/csteinmetz1/auraloss)** (Christian Steinmetz, Apache-2.0)
  — not a dependency, but `param_train.py`'s multi-resolution STFT loss is a from-scratch
  reimplementation ported to match its `MultiResolutionSTFTLoss` formulas and default
  weights (themselves vendored into the official NAM trainer).
- **Ideal-transformer technique** — the ngspice output transformer uses the standard
  controlled-source (E+F) ideal-transformer method; the specific center-tapped
  equations are ported from LiveSPICE (above).

Underlying scientific-Python stack: PyTorch, NumPy, SciPy, soundfile.
