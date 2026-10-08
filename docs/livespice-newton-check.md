# Reading livespice_cli's Newton report

`livespice_cli` prints one line on stderr after every render:

    newton: solves=9331200 unconverged=101 (0.0011%) severe=100 first_sample=294792 last_sample=441570 ...

An *unconverged* solve ran out of Newton iterations; a *severe* one was still taking a step of more than a volt when it did.
Before this check nothing read the line: it ended up in the `warnings` column of `params.csv` and a render with a hundred of
them was kept. `gen_dataset_from_schx.py` now acts on it (`--newton-check`, default `fail`):

* a render with any severe unconverged solve, or an unconverged fraction above `--newton-max-fraction` (default 1e-6),
  **fails** with an error starting `newton:`, and the retry ladder escalates it like any other convergence failure;
* if the unconverged count does not at least halve on the next rung, the ladder stops and the combination stays a visible
  failure (`[not escalating: ...]`): that failure class does not depend on the timestep, and climbing the ladder only costs hours;
* `--newton-check warn` records the report in `warnings` and keeps the render (the old behaviour, plus the note); `off` ignores it.

## `--trust-region V` (opt-in per device, `trust_region` in a config)

Passed to `livespice_cli --trust-region`: a Newton step whose norm exceeds V volts is scaled down, direction kept. It is
**circuit-specific**. Measured with livespice_cli at oversample 4 on the 9.7 s stress signal:

| circuit | unconverged, off | at 30 V |
|---|---|---|
| Ampeg SVT preamp, Volume 0.8 | 101 | **0** (also 0 at 10 and 60 V; output unchanged, ESR 1.03e-3 against a 16x render either way) |
| Ampeg SVT preamp, all knobs up | 1593 | 755 (oversample does not help: 1143 at 8x, 816 at 16x) |
| Ampeg SVT Full, defaults and Volume 0.8 | 0 | 0 (also at 100 V) |
| Fender Deluxe Reverb Full (tubes2), defaults | 12 | **1740** (worse: a power amp's plate steps are legitimately large) |
| Soldano SLO-100 Crunch Full, defaults | 0 | **81** (worse) |
| JCM800 preamp, Tweed preamp, AC30 Top Boost | 0 | 0, output identical to rounding |

So it is off by default; switch it on for a device only after reading the `newton:` report with and without it.

End to end on the SVT preamp at Volume 0.8, `--oversample 4`: without it the combination fails at 4x (101), fails at 8x
(2 severe) and passes at 16x (190 s); with `--trust-region 30` it passes on the first rung at 4x (36 s). At the all-knobs-up corner
the ladder stops after 8x with `not escalating` instead of running to the 128x ceiling.
