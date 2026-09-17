"""Sharded onset measurement: the slice/merge contract for distributing prepare_excitation.py.

Every assertion here guards a way a DISTRIBUTED sizing run goes wrong silently. The sizing
step picks ONE number -- the worst-case onset across every corner -- and that number sets the
excitation peak for the whole device, so a hole, a duplicate, or a corner measured by a
divergent solver build mis-sizes everything downstream with nothing failing.
"""
import json

import pytest

import prepare_excitation as pe


def _shard_file(tmp_path, name, rows, total=8, solver="livespice:abc123", shard="0-0/1"):
    p = tmp_path / name
    p.write_text(json.dumps({"corner_total": total, "solver": solver,
                             "shard": shard, "rows": rows}))
    return str(p)


def _rows(idxs, onset=0.1):
    return [{"index": i, "corner": f"c{i}", "params": {}, "onset_v": onset,
             "knee_v": None, "method": "knee+sat95-v1"} for i in idxs]


def test_shard_corners_stripes_and_partitions_exactly():
    """Modulo striping, not contiguous blocks -- corner cost is wildly uneven, so contiguous
    blocks hand one machine a run of slow corners. Every corner must land in exactly one shard."""
    corners = [(f"c{i}", {}) for i in range(20)]
    a = pe.shard_corners(corners, "0-1/4")
    b = pe.shard_corners(corners, "2-3/4")
    ia = [i for i, _ in a]
    ib = [i for i, _ in b]
    assert set(ia).isdisjoint(ib)
    assert sorted(ia + ib) == list(range(20))
    assert ia[:4] == [0, 1, 4, 5]        # striped, not [0,1,2,3]


def test_merge_rejects_a_missing_corner(tmp_path):
    f = _shard_file(tmp_path, "a.json", _rows([0, 1, 2]), total=8)
    with pytest.raises(SystemExit, match="missing from the merge"):
        pe.merge_onset_shards([f])


def test_merge_rejects_a_duplicated_corner(tmp_path):
    f1 = _shard_file(tmp_path, "a.json", _rows([0, 1]), total=4)
    f2 = _shard_file(tmp_path, "b.json", _rows([1, 2, 3]), total=4)
    with pytest.raises(SystemExit, match="more than one shard"):
        pe.merge_onset_shards([f1, f2])


def test_merge_rejects_mismatched_solver_builds(tmp_path):
    """The motivating case is real: enabling SimulateCapacitances was unsolvable until
    livespice-cli's submodule moved to 134d5c0, and two of five fleet machines had not
    rebuilt. Onsets from different builds are not comparable."""
    f1 = _shard_file(tmp_path, "a.json", _rows([0, 1]), total=4, solver="livespice:aaa")
    f2 = _shard_file(tmp_path, "b.json", _rows([2, 3]), total=4, solver="livespice:bbb")
    with pytest.raises(SystemExit, match="DIFFERENT renderer builds"):
        pe.merge_onset_shards([f1, f2])


def test_merge_rejects_an_unfingerprintable_renderer(tmp_path):
    f = _shard_file(tmp_path, "a.json", _rows([0, 1]), total=2, solver="livespice:UNKNOWN")
    with pytest.raises(SystemExit, match="cannot be proven comparable"):
        pe.merge_onset_shards([f])


def test_merge_rejects_shards_from_different_grids(tmp_path):
    """A config edited between dispatches -- the shards describe different devices."""
    f1 = _shard_file(tmp_path, "a.json", _rows([0, 1]), total=4)
    f2 = _shard_file(tmp_path, "b.json", _rows([2, 3]), total=6)
    with pytest.raises(SystemExit, match="disagree on the corner count"):
        pe.merge_onset_shards([f1, f2])


def test_merge_accepts_a_complete_set_and_orders_it(tmp_path):
    f1 = _shard_file(tmp_path, "a.json", _rows([2, 0]), total=4)
    f2 = _shard_file(tmp_path, "b.json", _rows([3, 1]), total=4)
    rows = pe.merge_onset_shards([f1, f2])
    assert [r["index"] for r in rows] == [0, 1, 2, 3]


def test_worst_case_is_the_max_across_all_shards(tmp_path):
    """The whole point of merging before sizing: the peak comes from the global max, which
    may live in any shard."""
    f1 = _shard_file(tmp_path, "a.json", _rows([0, 1], onset=0.05), total=4)
    f2 = _shard_file(tmp_path, "b.json", [{"index": 2, "corner": "c2", "params": {},
                                           "onset_v": 0.9, "knee_v": None, "method": "m"},
                                          {"index": 3, "corner": "c3", "params": {},
                                           "onset_v": 0.1, "knee_v": None, "method": "m"}],
                     total=4)
    rows = pe.merge_onset_shards([f1, f2])
    assert max(r["onset_v"] for r in rows) == 0.9
