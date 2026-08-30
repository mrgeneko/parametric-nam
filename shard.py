#!/usr/bin/env python3
"""ONE parser for the --shard LOW-HIGH/TOTAL spec, shared by every renderer that supports it.

WHY THIS IS ITS OWN MODULE. A shard spec is a distributed-rendering foot-gun: a bad or
divergent parse silently renders the WRONG slice on a machine, or renders the whole grid
redundantly on every machine, and neither shows up as an error -- only later as a dataset
with holes or with duplicated work. Two copies of this logic is exactly the setup that lets
that drift happen unnoticed, and this repo has already paid for that once (see
render_backends.py's docstring: preflight.py and the ngspice hand-deck family each carried
their own ~300-line copy of the knob-check logic until one silently diverged).

gen_dataset_from_schx.py (livespice) and the deck renderers (render_ltspice_deck.py,
render_ngspice_deck.py) share no other module -- they are separate entry points over
different backends -- so this is deliberately a standalone file with no imports beyond `re`,
importable from any of them without dragging in a backend.
"""
import re


def parse_shard(spec: str) -> tuple:
    """Parse a --shard 'LOW-HIGH/TOTAL' spec into (low, high, total), both ends inclusive.

    Raises ValueError with an actionable message on anything malformed -- fail loudly rather
    than clamping or guessing, for the reason in this module's docstring."""
    m = re.match(r"^(\d+)-(\d+)/(\d+)$", spec.strip())
    if not m:
        raise ValueError(f"--shard must look like LOW-HIGH/TOTAL, e.g. 0-15/48 (got {spec!r})")
    low, high, total = (int(g) for g in m.groups())
    if not (0 <= low <= high < total):
        raise ValueError(f"--shard {spec!r}: need 0 <= LOW <= HIGH < TOTAL (got low={low}, "
                         f"high={high}, total={total})")
    return low, high, total


def select(indexed, spec: str):
    """Filter [(global_index, item), ...] down to this shard's slice.

    Filters on the GLOBAL index, never a re-numbered position: every renderer names its output
    by that index (cap_NNNN.wav, sig/NNNN.npy), which is what makes merging shards from several
    machines filename-safe. Returns (kept, low, high, total)."""
    low, high, total = parse_shard(spec)
    return [(i, x) for i, x in indexed if low <= (i % total) <= high], low, high, total
