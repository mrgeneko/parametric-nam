"""Where a circuit's backend-validity verdicts are read from.

A verdict records what a simulator backend can and cannot render faithfully, e.g.
`{"livespice": {"valid": false, "reason": "..."}}`. `valid` is true, false or "partial".

Two homes, in this order:

  1. `<stem>.backends.toml` beside the .schx (unchanged: existing users keep working).
  2. the circuit file `<stem>.acmod` beside it: the member `backend_validity` (same shape,
     what tomllib.loads of the sidecar returns). A version 2 file carries it inside its
     top-level `extra` object; a version 1 (flat) file carries it at the top level.

If neither exists the caller announces that nothing is checking.
"""
from __future__ import annotations

import json
from pathlib import Path

MEMBER = "backend_validity"


def acmod_path(schx: Path) -> Path:
    return schx.with_suffix(".acmod")


def sidecar_path(schx: Path) -> Path:
    return schx.with_suffix(".backends.toml")


def member_from_doc(doc) -> tuple[bool, object]:
    """(present, value) of the verdict member in a parsed circuit file of either version."""
    if not isinstance(doc, dict):
        return False, None
    if doc.get("version") == 2:
        extra = doc.get("extra")
        if isinstance(extra, dict) and MEMBER in extra:
            return True, extra[MEMBER]
        return False, None
    if MEMBER in doc:
        return True, doc[MEMBER]
    return False, None


def _well_formed(spec) -> str | None:
    """None if `spec` is a {backend: {..}} mapping, else what is wrong with it."""
    if not isinstance(spec, dict):
        return f"must be an object mapping backend names to verdicts, got {type(spec).__name__}"
    for k, v in spec.items():
        if not isinstance(v, dict):
            return f"entry {k!r} must be an object, got {type(v).__name__}"
    return None


def read_backend_verdicts(schx: Path) -> tuple[dict | None, str | None, str | None]:
    """(spec, source_name, error). source_name is None when neither home has verdicts;
    spec is None with an error when a home exists but cannot be used."""
    sidecar = sidecar_path(schx)
    if sidecar.exists():
        try:
            import tomllib
            return tomllib.loads(sidecar.read_text()), sidecar.name, None
        except Exception as e:
            return None, sidecar.name, f"failed to parse ({e})"
    cj = acmod_path(schx)
    if not cj.exists():
        return None, None, None
    try:
        doc = json.loads(cj.read_text())
    except Exception:
        return None, None, None  # an unreadable circuit file is not a verdict source
    present, spec = member_from_doc(doc)
    if not present:
        return None, None, None
    name = f"{cj.name} ({MEMBER})"
    bad = _well_formed(spec)
    if bad:
        return None, name, f"is malformed ({bad})"
    return spec, name, None


def load_backend_verdicts(schx: Path) -> tuple[dict | None, str | None]:
    """(spec | None, source_name): the verdicts for this circuit and where they came from."""
    spec, name, _err = read_backend_verdicts(schx)
    return spec, name
