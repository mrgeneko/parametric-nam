"""Shared helpers for building a parametric dataset from real captures (not a .schx render) --
a SET OF EXISTING FILES, each a fixed/captured knob-setting of a real device, where every
permutation's filename encodes its knob settings as comma-separated tokens (see
parse_filename_tokens). Used by gen_dataset_from_captures.py for both its source kinds (.nam
exports and raw .wav captures) -- they share the same filename convention and the same collision
hazard, so this module holds ONLY what's genuinely source-agnostic -- filename parsing, the
digit-scale collision check, and the argparse wiring for --knob-map/--knob-scale/--mapping-csv.
It does NOT assemble config.json (per-source fields differ too much to fake a shared schema
prematurely -- see docs, "sketch it out properly").
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# Default prefix -> knob-name map, matching the amp-community-style "MyAmp DST" captures this was built for.
# A token's ALPHA PREFIX (not just its first letter) is the dict key, so "Rvb"/"Rsn"/"Prsn" never
# collide with "R" or with each other -- filename splitting isolates the whole prefix run before
# lookup, see parse_filename_tokens.
DEFAULT_PREFIX_MAP = {
    "G": "Gain", "B": "Bass", "M": "Mids", "T": "Treble",
    "Rvb": "Reverb", "Rsn": "Resonance", "Prsn": "Presence",
}

_TOKEN_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def parse_filename_tokens(stem: str, prefix_map: dict, scale_overrides: dict) -> tuple:
    """Extract every "PREFIXdigits" token from a filename stem (e.g. "G2, B5, M5, T5, Rvb0" or
    "Xyz DST T3 Drv2") into {knob_name: float_value}, skipping any token whose prefix isn't in
    prefix_map (unknown/device-name tokens like "Xyz"/"DST"/"MyAmp" are expected and silently
    ignored, not an error).

    Scans EVERY comma-or-whitespace-separated word in the whole stem, not just the last word of
    each comma segment -- a filename can carry more than one relevant token per segment with no
    comma between them (e.g. "T3 Drv2" has two: some dual-drive pedal captures use "Xyz DST T3
    Drv2", Tone and Drive back to back with only a space). Treating commas as just another word-separator
    (rather than iterating comma-segments and taking each one's last word) handles both
    conventions with the same code path.

    Also returns {knob_name: digit_string} (the raw token digits, e.g. "10") so callers can catch
    the DEFAULT SCALE'S AMBIGUITY: 10**(-len(digits)) assumes every extra digit is another decimal
    place (a PERCENTAGE convention: "50" -> 0.50), which silently collides with a DIAL-POSITION
    convention (1..10 written out, where "10" means the max, 1.0, not 0.10) -- "G1" and "G10" both
    scale to 0.1 by default. There is no way to tell which convention a filename uses from the
    digit string alone, so this can only be caught after the fact by comparing outputs across the
    whole batch (see check_scale_collisions), not fixed inside this function.
    """
    out, raw_digits = {}, {}
    for tok in stem.replace(",", " ").split():
        m = _TOKEN_RE.match(tok)
        if not m:
            continue
        prefix, digits = m.group(1), m.group(2)
        name = prefix_map.get(prefix)
        if name is None:
            continue
        scale = scale_overrides.get(prefix, scale_overrides.get(name))
        if scale is None:
            scale = 10 ** (-len(digits))   # single digit -> /10, matches the user's spec exactly
        out[name] = round(int(digits) * scale, 6)
        raw_digits[name] = digits
    return out, raw_digits


def load_mapping_csv(path: Path) -> dict:
    """filename,<Knob1>,<Knob2>,... escape hatch -- bypasses regex/token parsing entirely for
    conventions that don't fit the "comma-separated PREFIXdigits tokens" shape."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fname = row.pop("filename")
            out[fname] = {k: float(v) for k, v in row.items() if v not in (None, "")}
    return out


def add_knob_parsing_args(ap: argparse.ArgumentParser) -> None:
    """Wires --knob-map / --knob-scale / --mapping-csv onto a capture-source tool's parser."""
    ap.add_argument("--knob-map", action="append", default=[],
                     help="PREFIX=Name, overrides/extends the default filename-token prefix map "
                          "(e.g. --knob-map D=Drive). Repeatable.")
    ap.add_argument("--knob-scale", action="append", default=[],
                     help="PREFIX_OR_NAME=FACTOR, overrides the default digit->value scale "
                          "(default: 1/10^len(digits), i.e. a single digit -> value/10). Repeatable.")
    ap.add_argument("--mapping-csv", type=Path,
                     help="filename,<Knob1>,<Knob2>,... CSV; bypasses filename-token parsing "
                          "entirely for a batch that doesn't follow the token convention")


def resolve_knob_maps(args) -> tuple:
    """--knob-map/--knob-scale/--mapping-csv CLI args -> (prefix_map, scale_overrides, mapping)."""
    prefix_map = dict(DEFAULT_PREFIX_MAP)
    for kv in args.knob_map:
        k, v = kv.split("=", 1)
        prefix_map[k.strip()] = v.strip()
    scale_overrides = {}
    for kv in args.knob_scale:
        k, v = kv.split("=", 1)
        scale_overrides[k.strip()] = float(v)
    mapping = load_mapping_csv(args.mapping_csv) if args.mapping_csv else {}
    return prefix_map, scale_overrides, mapping


def parse_all_filenames(files, prefix_map: dict, scale_overrides: dict, mapping: dict) -> tuple:
    """Parse every file's knob settings, preferring an exact --mapping-csv row over filename-token
    parsing. Returns (per_file, per_file_raw) -- per_file_raw is {} for mapping-csv rows (an
    explicit CSV has no digit-scale ambiguity to check, see check_scale_collisions)."""
    per_file, per_file_raw = {}, {}
    for f in files:
        if f.name in mapping:
            per_file[f], per_file_raw[f] = mapping[f.name], {}
        else:
            per_file[f], per_file_raw[f] = parse_filename_tokens(f.stem, prefix_map, scale_overrides)
    empty = [f.name for f, p in per_file.items() if not p]
    if empty:
        print(f"WARNING: no knobs parsed from filename for: {empty}\n"
              f"         (use --knob-map / --mapping-csv if this batch uses a different convention)",
              file=sys.stderr)
    return per_file, per_file_raw


def check_scale_collisions(per_file: dict, per_file_raw: dict, error) -> None:
    """Catch the default scale's DIAL-POSITION-vs-PERCENTAGE ambiguity (see
    parse_filename_tokens' docstring): if two files carry DIFFERENT raw digit strings for the
    same knob (e.g. "1" and "10") but the default 10**(-len(digits)) scale collapsed them to the
    SAME float value, that's not a legitimate duplicate capture -- it's the scale guessing wrong.
    Fail loudly with the exact colliding files rather than silently training on a corrupted grid.

    `error` is called with the message on collision (e.g. argparse.ArgumentParser.error, or
    sys.exit) -- the caller decides how a collision should terminate the program.
    """
    for name in sorted({k for p in per_file_raw.values() for k in p}):
        by_value: dict = {}
        for f, raw in per_file_raw.items():
            if name in raw:
                by_value.setdefault(per_file[f][name], []).append((f.name, raw[name]))
        for value, hits in by_value.items():
            distinct_digits = {digits for _, digits in hits}
            if len(hits) >= 2 and len(distinct_digits) >= 2:
                error(
                    f"'{name}' scale collision: {[h[0] for h in hits]} carry DIFFERENT filename "
                    f"tokens ({sorted(distinct_digits)}) but all scaled to the SAME value "
                    f"({value}) under the default 10**(-len(digits)) rule. This is the classic "
                    f"dial-position-vs-percentage ambiguity (\"1\" and \"10\" both -> 0.1) -- pass "
                    f"--knob-scale {name}=FACTOR (a fixed factor applied to the raw digits, e.g. "
                    f"0.1 for a 1..10 dial) to disambiguate.")
