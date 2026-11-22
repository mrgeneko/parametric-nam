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


# ---- recording a verdict in the circuit file (targeted text edit, nothing else touched) ------

_dec = json.JSONDecoder()
_WS = " \t\r\n"


def _skip_ws(t: str, i: int) -> int:
    while i < len(t) and t[i] in _WS:
        i += 1
    return i


def _members(t: str, start: int) -> list[tuple[str, int, int, int]]:
    """For the JSON object whose '{' is at t[start]: [(key, key_start, value_start, value_end)]."""
    from json.decoder import scanstring
    assert t[start] == "{"
    i, out = _skip_ws(t, start + 1), []
    while t[i] != "}":
        k0 = i
        key, i = scanstring(t, i + 1, True)
        i = _skip_ws(t, i)
        assert t[i] == ":"
        v0 = _skip_ws(t, i + 1)
        _, v1 = _dec.raw_decode(t, v0)
        out.append((key, k0, v0, v1))
        i = _skip_ws(t, v1)
        if t[i] == ",":
            i = _skip_ws(t, i + 1)
    return out


def _end_of_object(t: str, start: int) -> int:
    return _dec.raw_decode(t, start)[1] - 1  # index of the closing '}'


def _set_path(t: str, obj_start: int, path: list[str], value) -> str:
    key, rest = path[0], path[1:]
    mem = _members(t, obj_start)
    new_val = json.dumps(value if not rest else _nest(rest, value), ensure_ascii=False)
    for k, k0, v0, v1 in mem:
        if k == key:
            if rest and t[v0] == "{":
                return _set_path(t, v0, rest, value)
            return t[:v0] + new_val + t[v1:]
    close = _end_of_object(t, obj_start)
    item = f"{json.dumps(key)}: {new_val}"
    if not mem:
        return t[:close] + item + t[close:]
    last_k0, last_v1 = mem[-1][1], mem[-1][3]
    j = last_k0
    while j > 0 and t[j - 1] in _WS:  # the whitespace the last member sits after
        j -= 1
    lead = t[j:last_k0]
    return t[:last_v1] + "," + lead + item + t[last_v1:]


def _nest(path: list[str], value):
    for k in reversed(path):
        value = {k: value}
    return value


def set_backend_validity_text(text: str, value: dict) -> str:
    """`text` (a v1 or v2 circuit file) with its backend_validity member set to `value`, every
    other byte unchanged. Raises ValueError rather than return a file that differs in any other
    way (checked by re-parsing)."""
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError("circuit file is not a JSON object")
    path = ["extra", MEMBER] if doc.get("version") == 2 else [MEMBER]
    if doc.get("version") == 2 and "extra" in doc and not isinstance(doc["extra"], dict):
        raise ValueError("`extra` is not an object")
    start = _skip_ws(text, 0)
    new = _set_path(text, start, path, value)
    expect = json.loads(text)
    if len(path) == 2:
        expect.setdefault("extra", {})[MEMBER] = value
    else:
        expect[MEMBER] = value
    if json.loads(new) != expect:
        raise ValueError("targeted edit did not reproduce the intended document")
    return new


# ---- reading a circuit file of either layout ------------------------------------------------

_FLAT_SECTIONS = ("circuit", "derived", "playback", "provenance", "extra")


def flat_view(doc):
    """The flat (version 1 style) dict of a parsed circuit file. A version 1 file passes through.
    For a version 2 file the members of its `circuit`, `derived` (quality, prepared, ...),
    `playback`, `provenance` (source, backend) and `extra` sections are lifted to the top level,
    beside the header members; `controls` is not expanded (nothing here reads it)."""
    if not isinstance(doc, dict) or doc.get("version") != 2:
        return doc
    flat = {k: v for k, v in doc.items() if k not in _FLAT_SECTIONS and k != "controls"}
    for sec in _FLAT_SECTIONS:
        body = doc.get(sec)
        if isinstance(body, dict):
            for k, v in body.items():
                flat.setdefault(k, v)
    flat["version"] = 1
    return flat
