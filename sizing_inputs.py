"""What an excitation was sized AGAINST -- recorded in its recipe.json, compared later.

build_excitation.py's recipe records what it built (args, source and output hashes) and
prepare_excitation.py's `sizing` block records the onsets that justified those numbers. Neither
recorded the circuit or the knob grid the onsets were measured on, so a recipe on disk could not
say whether it still matched the config next to it. Mesa ORANGE and RED trained against exactly
that kind of stale excitation (they failed 6/43 and 4/43 corners); the only warning was a comment
in the config.

`recipe["sizing"]["inputs"]` closes that. It records, at sizing time:

  circuit         sha256 of the .schx (or deck-module) bytes -- the same `identity` the onset
                  cache is keyed on
  grid, fixed     the knob grid VALUES and pinned params. The whole grid matters, not just its
                  min/max: prepare_excitation probes interior points drawn (seeded) from every
                  knob's full value list, so editing an interior value can change what was probed
  sample_grid     how many interior points that was
  conditions      the onset-cache key's own `extra` string (oversample, iterations, capture chain,
                  solver source revision, ...) -- recorded verbatim for forensics, not compared

`compare()` reports how a recorded block sits against the current config. The verdicts differ in
strength on purpose:

  stale           the GRID or FIXED params differ. The probed points would differ, so the sizing
                  is not evidence about this grid. Act on it.
  circuit-differs the circuit bytes differ but the grid is the same. ADVISORY only: any .schx
                  edit changes a content hash, including ones that do not move saturation onset,
                  so this says "re-check", not "re-size". check_transient_coverage.py is the
                  arbiter of whether the excitation still covers.
  match           nothing differs
  unrecorded      the recipe predates this block (or has none); no signal either way
"""
import hashlib

SCHEMA = 1


def parse_ranges(entries) -> "dict[str, list[float]]":
    """['Gain=0.1,0.5', ...] -> {'Gain': [0.1, 0.5]}. Same rule as prepare_excitation._parse_ranges."""
    out: dict = {}
    for entry in entries or []:
        name, vals = entry.split("=", 1)
        out[name.strip()] = [float(v) for v in vals.split(",")]
    return out


def parse_fixed(s: "str | None") -> "dict[str, float]":
    """'A=0.5,B=1' -> {'A': 0.5, 'B': 1.0}. Same rule as prepare_excitation._parse_fixed."""
    out: dict = {}
    for kv in filter(None, (p.strip() for p in (s or "").split(","))):
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def inputs_record(*, identity: bytes, circuit_kind: str, knob_ranges: dict, fixed: dict,
                  sample_grid: int, conditions: str) -> dict:
    return {
        "schema": SCHEMA,
        "circuit": {"kind": circuit_kind, "sha256": hashlib.sha256(identity).hexdigest()},
        "grid": {k: [float(x) for x in v] for k, v in knob_ranges.items()},
        "fixed": {k: float(v) for k, v in (fixed or {}).items()},
        "sample_grid": int(sample_grid),
        "conditions": conditions,
    }


def _diff_grid(recorded: dict, current: dict) -> "list[str]":
    out = []
    for k in sorted(set(recorded) | set(current)):
        if k not in current:
            out.append(f"knob {k} is no longer in the grid")
        elif k not in recorded:
            out.append(f"knob {k} was added to the grid")
        elif [float(x) for x in recorded[k]] != [float(x) for x in current[k]]:
            out.append(f"{k}: sized on {recorded[k]}, config now has {current[k]}")
    return out


def compare(recorded: "dict | None", *, circuit_sha256: "str | None",
            knob_ranges: dict, fixed: dict) -> dict:
    """How a recorded `inputs` block sits against the current config. See the module docstring
    for what each status means and how strongly to act on it."""
    if not recorded or recorded.get("schema") != SCHEMA:
        return {"status": "unrecorded", "reasons": []}
    reasons = _diff_grid(recorded.get("grid", {}), knob_ranges)
    rf, cf = recorded.get("fixed", {}), {k: float(v) for k, v in (fixed or {}).items()}
    for k in sorted(set(rf) | set(cf)):
        if rf.get(k) != cf.get(k):
            reasons.append(f"fixed {k}: sized with {rf.get(k)}, config now has {cf.get(k)}")
    if reasons:
        return {"status": "stale", "reasons": reasons}
    rc = (recorded.get("circuit") or {}).get("sha256")
    if circuit_sha256 and rc and rc != circuit_sha256:
        return {"status": "circuit-differs",
                "reasons": ["the circuit bytes differ from what the excitation was sized against"]}
    return {"status": "match", "reasons": []}
