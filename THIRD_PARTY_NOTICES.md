# Third-party notices

parametric-nam is MIT-licensed (see `LICENSE`). It contains or is derived from the work of the projects below. Their copyright and
permission notices are reproduced here as their licenses require. Tools the repository only *runs* are listed at the end; none of them
is included in it.

## LiveSPICE

Portions of this repository are ported from, or follow, LiveSPICE (<https://github.com/dsharlet/LiveSPICE>) by Dillon Sharlet:

- `ngspice/schx_to_ngspice.py`: the device models (the Dempwolf-Zolzer triode and the Koren pentode, from `Triode.cs` and `Pentode.cs`) and
  the ideal centre-tapped transformer are ported from LiveSPICE's equations.
- `ngspice/gen_evh5150_ngspice.py`: component values and tube parameters taken from LiveSPICE's models of that amplifier.
- `ltspice/schx_to_ltspice.py`: derived from `ngspice/schx_to_ngspice.py`.
- The `.schx` circuit format that most of the tools read is LiveSPICE's.

The LiveSPICE license:

```
The MIT License (MIT)

Copyright (c) 2013 Dillon Sharlet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Neural Amp Modeler

The `.nam` file format, the WaveNet ("A2") architecture, the slimmable container and the parametric-knob metadata that `param_train.py`,
`export_checkpoint.py`, `nam_infer.py`, `nam_standard.py` and related files implement and target follow Steven Atkinson's
neural-amp-modeler (<https://github.com/sdatkinson/neural-amp-modeler>) and NeuralAmpModelerCore
(<https://github.com/sdatkinson/NeuralAmpModelerCore>).

neural-amp-modeler:

```
MIT License

Copyright (c) 2019-2025 Steven Atkinson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

NeuralAmpModelerCore:

```
MIT License

Copyright (c) 2023 Steven Atkinson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Credited, no code included

- **auraloss** (Christian Steinmetz, Apache-2.0, <https://github.com/csteinmetz1/auraloss>): `param_train.py`'s multi-resolution STFT
  loss is a from-scratch reimplementation of the formulas of its `MultiResolutionSTFTLoss`; no auraloss code is included.
- **Tube models:** Norman Koren's SPICE triode and pentode model, and K. Dempwolf and U. Zolzer, "A physically-motivated triode model for
  circuit simulations," Proc. DAFx-11 (2011), published equations as implemented in LiveSPICE above.

## Tools this repository runs but does not include

None of the following is bundled; install them yourself. If you redistribute them together with this repository (a container image, a
release archive), their own license terms apply to that distribution.

- **ngspice** (<https://ngspice.sourceforge.io/>, modified BSD licence, with its XSPICE code models): run as a subprocess by the ngspice
  backends; the decks use its `filesource` code model.
- **LTspice** (Analog Devices, proprietary freeware, <https://www.analog.com/ltspice>): run as a subprocess by the `ltspice-deck`
  backend. Generated decks refer to the op-amp library in *your* installation (`UniversalOpAmp2.lib`); that library is not copied here.
- **spicelib** (GPL-3.0, <https://github.com/nunobrum/spicelib>): an optional dependency of the ngspice and LTspice backends only
  (`requirements-ngspice.txt`), installed from PyPI and kept out of `requirements.txt` for that reason.
- **livespice-cli** (<https://github.com/mrgeneko/livespice-cli>, MIT) and the LiveSPICE it builds against: a separate repository, built
  by `setup.sh` into a sibling directory.
- **Xyce**: `patches/xyce-superbuild-fixes.patch` is a patch for Xyce's build script, whose context lines come from Xyce (a separate
  project under its own licence). It is not part of the toolchain.
