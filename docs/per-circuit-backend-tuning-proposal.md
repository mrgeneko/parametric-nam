# Proposal: per-circuit backend render-tuning defaults

> **STATUS: PROPOSAL.** Nothing here is implemented. `render_backends.py`'s `--conv` channel
> exists and works exactly as described below; what's missing is a durable, automatically-applied
> per-circuit default for it.

## The gap

`NgspiceSchxBackend` hardcoded `ot_damp="47k"`, `ot_snub="10n"` for every circuit until
2026-09-21, tuned for the EVH 5150's OT (238:1 into 8 Ω). Fixing the Ampeg SVT power amp's
`ngspice` LF bug required a different pair of values (`ot_damp=220k, ot_snub=10n`, for its 400:1
into 4 Ω OT) — see `parametric-devices/amps/Ampeg SVT Power Amp.backends.toml`. That value is
real, measured, and committed. It is also **inert**: nothing reads it back out.

`render_backends.py`'s `self.conv` dict is a raw passthrough — `preflight.py`, `grid_adequacy.py`,
`gen_dataset_from_schx.py`, and `run_pipeline.py` all forward whatever `--conv key=val,...` string
their own caller happened to type, with a global hardcoded fallback if nobody typed anything. There
is no per-circuit source of truth in between. Today, rendering the SVT power amp through ngspice
without manually typing `--conv "ot_damp=220k,ot_snub=10n"` silently uses the EVH's tuning instead
— the fix that got committed only takes effect if a human remembers to ask for it by hand, every
time, at every call site.

## Why the `.schx` isn't the right home

`ot_damp`/`ot_snub` are not circuit components — they don't exist in the real amp. They're a pair
of fictitious elements `schx_to_ngspice.py`'s `CenterTapTransformer` translation injects purely to
numerically stabilize its own ideal-ampere-turns (VCVS/CCCS) transformer model. LiveSPICE never
uses them at all; they are `ngspice`-translation-specific. Storing them in the `.schx` would
misrepresent them as part of the amp. This data is **per-circuit AND per-backend** — exactly the
shape `<Device>.backends.toml` already has (`backend.ngspice` / `backend.livespice` as separate
tables), and exactly the shape the `.schx` does not.

## Proposed shape

Add a `conv` key to each backend's existing table in the sidecar, same `key=val,...` string
`--conv` already parses — no new grammar:

```toml
backend.ngspice = { valid = "partial", reason = "...", conv = "ot_damp=220k,ot_snub=10n" }
```

Backend-scoped by construction: an `ot_damp` override under `backend.ngspice` has no bearing on
`backend.livespice`, which doesn't have that knob at all (its own per-circuit tuning, e.g.
`OT_LEAK_MH`, would live under `backend.livespice.conv` the same way). One mechanism for both
backends, not two.

### Propagation

`gen_registry.py` already copies `valid`/`reason` from each sidecar into `devices.toml` per
backend (the same code path last session's `"partial"`-collapse bug lived in — being touched
carefully already). Add `conv` to that same copy. Mechanical, no new logic shape.

### Lookup and merge, at render time

`preflight.py`, `grid_adequacy.py`, `gen_dataset_from_schx.py`, and `run_pipeline.py` each already
know the `.schx` path and backend name when they construct a `render_backends.py` backend object.
Locating the sidecar needs no new plumbing — same basename, `.backends.toml` suffix, already sits
next to the `.schx`.

Precedence, merged **key by key**, not all-or-nothing:

1. `--conv` key explicitly passed by the caller — wins, always.
2. Otherwise, that key's value from the circuit's own sidecar `conv` string, if present.
3. Otherwise, the backend's hardcoded global default (`47k`/`10n` today).

So `--conv "ot_snub=22n"` against the SVT would still pick up its sidecar's `ot_damp=220k`, while
overriding just `ot_snub` for that one run. The merge belongs in one place —
`render_backends.py`, where `self.conv`/`self.ng_base` are already assembled — so all four call
sites inherit it for free rather than reimplementing the lookup four times.

### Backward compatibility

A circuit with no `conv` key in its sidecar renders exactly as it does today. This is additive:
existing behavior for every circuit other than the SVT (which currently has no automatic
`ngspice` render-fix at all) is unchanged.

## Non-goals

This does not attempt to fix the SVT's remaining `ngspice` LF issue — a ~60–64 V floor at 40 Hz
against a 49.1 V physical ceiling, unmoved by `ot_damp`/`ot_snub`/`Lm` tuning, suspected to be an
LF-specific limitation of the ideal-transformer (VCVS/CCCS) translation itself rather than
anything a tuning default can reach (see the same `backends.toml` entry). That's a separate,
harder, not-yet-root-caused problem. This proposal only makes the tuning that *does* exist
actually take effect automatically instead of living as inert prose.

## Open questions

- Should `gen_registry.py` warn (or refuse to regenerate) if a sidecar's `conv` string fails to
  parse as `key=val,...`, rather than silently propagating garbage into `devices.toml`?
- Worth a `--no-sidecar-conv` escape hatch for a caller who wants the global default even when a
  sidecar override exists (e.g. while re-diagnosing whether the override is actually responsible
  for an observed change)? Leaning yes, cheap insurance, not spec'd above — add if it turns out to
  matter in practice.
