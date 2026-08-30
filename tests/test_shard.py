"""Shared --shard spec parser and selector (shard.py).

The parser cases below came from tests/test_gen_dataset_from_schx.py, moved here when
_parse_shard was extracted into shard.py so gen_dataset_from_schx.py and both deck renderers
could share ONE implementation -- see shard.py's docstring for why two copies is a hazard.
"""
import itertools

import pytest

from shard import parse_shard, select


# --------------------------------------------------------------------------- _parse_shard

def test_parse_shard_valid_spec():
    assert parse_shard("0-15/48") == (0, 15, 48)


def test_parse_shard_single_remainder_is_low_equals_high():
    assert parse_shard("3-3/8") == (3, 3, 8)


def test_parse_shard_strips_whitespace():
    assert parse_shard(" 0-15/48 ") == (0, 15, 48)


@pytest.mark.parametrize("spec", [
    "0/48",             # missing the LOW-HIGH range entirely
    "0-15",             # missing /TOTAL
    "a-b/48",           # non-numeric
    "0-15/48/2",        # extra segment
    "",
    "0..15/48",         # wrong range separator
])
def test_parse_shard_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_shard(spec)


@pytest.mark.parametrize("spec", [
    "5-3/8",    # LOW > HIGH
    "0-8/8",    # HIGH == TOTAL (must be < TOTAL, moduli are 0..TOTAL-1)
    "0-15/8",   # HIGH >= TOTAL
])
def test_parse_shard_rejects_out_of_range_bounds(spec):
    with pytest.raises(ValueError):
        parse_shard(spec)


def test_shard_filter_selects_the_right_permutations_by_index_modulo_total():
    """The actual filter render_grid's caller applies -- exercised directly here since it's
    inlined in main() rather than its own function, to pin the exact selection semantics an
    orchestration script sharding across machines depends on."""
    low, high, total = parse_shard("0-1/4")
    to_run = [(i, f"perm{i}") for i in range(10)]
    selected = [(i, p) for i, p in to_run if low <= (i % total) <= high]
    assert [i for i, _ in selected] == [0, 1, 4, 5, 8, 9]


def test_shard_filter_partitions_are_disjoint_and_exhaustive_across_a_full_split():
    """Every permutation must land in EXACTLY one shard when the [LOW,HIGH] ranges tile
    [0, TOTAL) with no gaps or overlaps -- the property an equal or capacity-weighted split
    both rely on for merged results to be complete without duplicate renders."""
    total = 12
    shards = ["0-3/12", "4-7/12", "8-11/12"]  # equal 3-way split, weights 4/4/4
    to_run = [(i, f"perm{i}") for i in range(50)]
    seen = []
    for spec in shards:
        low, high, t = parse_shard(spec)
        assert t == total
        seen += [i for i, _ in to_run if low <= (i % t) <= high]
    assert sorted(seen) == list(range(50))
    assert len(seen) == len(set(seen))  # no permutation claimed by two shards


def test_shard_filter_supports_capacity_weighted_uneven_ranges():
    """A wider [LOW,HIGH] range gives a faster machine proportionally more of the grid --
    the mechanism --shard's own help text describes for capacity-weighted (e.g. core-count-
    proportional) splits, not just an equal N-way split."""
    total = 8
    fast = parse_shard("0-5/8")   # 6/8 of the grid -- the faster machine
    slow = parse_shard("6-7/8")   # 2/8 of the grid -- the slower machine
    to_run = [(i, f"perm{i}") for i in range(40)]
    fast_n = len([i for i, _ in to_run if fast[0] <= (i % fast[2]) <= fast[1]])
    slow_n = len([i for i, _ in to_run if slow[0] <= (i % slow[2]) <= slow[1]])
    assert fast_n + slow_n == 40
    assert fast_n == 3 * slow_n  # 6:2 weight ratio reflected exactly in a multiple-of-8 grid

# --------------------------------------------------------------------------- select()

def test_select_keeps_global_indices_not_renumbered_positions():
    combos = list(itertools.product([0.2, 0.8], [0.25, 0.75], [0.25, 0.75]))   # 8 perms
    kept, low, high, total = select(list(enumerate(combos)), "1-1/2")
    assert [i for i, _ in kept] == [1, 3, 5, 7]
    assert (low, high, total) == (1, 1, 2)
    # the payload travels with its global index
    assert all(c == combos[i] for i, c in kept)


def test_select_shards_tile_a_grid_that_is_not_a_multiple_of_total():
    idx = list(enumerate(range(37)))
    got = []
    for spec in ("0-0/3", "1-1/3", "2-2/3"):
        got += [i for i, _ in select(idx, spec)[0]]
    assert sorted(got) == list(range(37))


def test_select_full_range_is_the_identity():
    idx = list(enumerate(range(50)))
    kept, *_ = select(idx, "0-7/8")
    assert [i for i, _ in kept] == list(range(50))


def test_select_propagates_a_bad_spec():
    with pytest.raises(ValueError):
        select(list(enumerate(range(4))), "9-9/2")
