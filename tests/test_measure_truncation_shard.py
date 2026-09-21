"""Sharded oversample measurement: the slice/merge contract for distributing
measure_truncation.py -- same shape as tests/test_prepare_excitation_shard.py, applied to
knob settings instead of corners.

Every assertion here guards a way a distributed truncation measurement goes wrong silently.
The measurement picks the WORST setting per candidate oversample, and that number decides
whether a full training run gets a contaminated target, so a hole, a duplicate, or a shard
measured by a divergent solver build has to be refused loudly, not averaged over.
"""
import json

import pytest

from shard import select
import measure_truncation as mt


def _row(idx, params, esr):
    return {"index": idx, "params": params, "esr": esr}


def _shard_file(tmp_path, name, rows, total=6, candidates=(2, 4, 8), ref_os=32,
                solver="livespice:abc123", shard="0-0/1"):
    p = tmp_path / name
    p.write_text(json.dumps({"setting_total": total, "candidates": list(candidates),
                             "ref_os": ref_os, "solver": solver, "shard": shard, "rows": rows}))
    return str(p)


# --------------------------------------------------------------------------- shard selection

def test_settings_shard_stripes_and_partitions_exactly():
    """Modulo striping, not contiguous blocks -- same reason as every other shardable step
    here: settings are not equal render cost (a candidate near a knob's stiff extreme can cost
    far more than one at the midpoint), so contiguous blocks would hand one machine a run of
    slow settings."""
    knobs = ["Bass", "Mid", "Treble", "Volume"]
    settings = mt.probe_settings(knobs)
    indexed = list(enumerate(settings))
    a, *_ = select(indexed, "0-1/4")
    b, *_ = select(indexed, "2-3/4")
    ia = [i for i, _ in a]
    ib = [i for i, _ in b]
    assert set(ia).isdisjoint(ib)
    assert sorted(ia + ib) == list(range(len(settings)))


def test_probe_settings_is_deterministic():
    """Sharding by index only works if every worker enumerates settings identically -- no
    dependence on dict iteration, filesystem order, or anything host-specific."""
    knobs = ["Bass", "Mid", "Treble", "Volume", "Brilliance", "Master"]
    assert mt.probe_settings(knobs) == mt.probe_settings(knobs)


# --------------------------------------------------------------------------- score_rows

def test_score_rows_picks_the_worst_setting_per_candidate():
    rows = [
        _row(0, {"Gain": 0.0}, {"2": {"num": 1.0, "den": 100.0}, "4": {"num": 0.1, "den": 100.0}}),
        _row(1, {"Gain": 1.0}, {"2": {"num": 9.0, "den": 100.0}, "4": {"num": 0.5, "den": 100.0}}),
    ]
    res = mt.score_rows(rows, (2, 4))
    worst2, at2 = res[2]
    worst4, at4 = res[4]
    assert worst2 == pytest.approx(0.09)
    assert at2 == {"Gain": 1.0}
    assert worst4 == pytest.approx(0.005)
    assert at4 == {"Gain": 1.0}


def test_score_rows_matches_between_one_shard_and_two_merged_shards():
    """The actual equivalence guarantee: scoring a sharded-then-merged run must reproduce
    exactly what scoring the same rows unsharded would -- same function, same reduction,
    just a different route to the same row list."""
    all_rows = [
        _row(0, {"Gain": 0.0}, {"2": {"num": 1.0, "den": 50.0}}),
        _row(1, {"Gain": 0.25}, {"2": {"num": 4.0, "den": 50.0}}),
        _row(2, {"Gain": 0.5}, {"2": {"num": 2.0, "den": 50.0}}),
        _row(3, {"Gain": 0.75}, {"2": {"num": 6.0, "den": 50.0}}),
    ]
    unsharded = mt.score_rows(all_rows, (2,))
    merged = mt.score_rows(all_rows[:2] + all_rows[2:], (2,))  # same rows, split point varies
    assert unsharded == merged


def test_score_rows_excludes_a_near_silent_setting_from_the_worst_pick():
    """Real failure mode: probe_settings() always tests every knob at its own 0/1 extreme,
    and a Volume/Master-shaped knob at its 'off' extreme renders near-silent (confirmed on the
    Ceriatone Muchless Captain Reverb's Master=0.0: rms=0.000000). The reference energy (den)
    there is a tiny positive noise-floor number, not exactly zero, so a bare `den > 0` guard
    lets num/den blow up into a confidently wrong 'worst setting' -- this asserts that setting
    is excluded instead, even though its raw num/den ratio would otherwise dominate."""
    rows = [
        _row(0, {"Master": 0.5}, {"2": {"num": 0.02, "den": 100.0}}),   # normal setting
        _row(1, {"Master": 1.0}, {"2": {"num": 0.05, "den": 90.0}}),    # normal setting
        # near-silent: den is 1e-9x the loudest row's den, well under the exclusion floor --
        # its own ratio (1e-10/1e-12 = 100) would otherwise swamp every real setting above.
        _row(2, {"Master": 0.0}, {"2": {"num": 1e-10, "den": 1e-12}}),
    ]
    res = mt.score_rows(rows, (2,))
    worst, at = res[2]
    assert at == {"Master": 1.0}          # the genuine worst setting, not the silent one
    assert worst == pytest.approx(0.05 / 90.0)
    assert res["excluded_near_silent"][2] == [{"Master": 0.0}]


def test_score_rows_does_not_flag_the_only_setting_as_near_silent():
    """Exclusion is RELATIVE to the loudest setting measured for that candidate -- with only
    one row, there is nothing louder to judge it against, so it is not near-silent by this
    definition and must still be scored (there being nothing else to pick as 'worst')."""
    rows = [_row(0, {"Master": 0.0}, {"2": {"num": 1e-10, "den": 1e-12}})]
    res = mt.score_rows(rows, (2,))
    worst, at = res[2]
    assert at == {"Master": 0.0}
    assert "excluded_near_silent" not in res


def test_score_rows_does_not_confuse_a_zero_den_render_failure_with_near_silence():
    """den == 0.0 EXACTLY is measure()'s own separate 'this render produced nothing usable'
    sentinel (see esr_terms: it returns (0.0, 0.0) when too few samples survive the lead-in
    skip). That must fall through the existing `den > 0` NaN guard as before, not get counted
    in `excluded_near_silent` -- the two are different failure modes and conflating them would
    hide a genuine render failure inside a list meant for numerically-unstable-but-real ones."""
    rows = [
        _row(0, {"Master": 0.5}, {"2": {"num": 0.02, "den": 100.0}}),
        _row(1, {"Master": 1.0}, {"2": {"num": 0.0, "den": 0.0}}),   # render failure, not near-silence
    ]
    res = mt.score_rows(rows, (2,))
    worst, at = res[2]
    assert at == {"Master": 0.5}
    assert "excluded_near_silent" not in res


# --------------------------------------------------------------------------- merge_truncation_shards

def test_merge_rejects_a_missing_setting(tmp_path):
    f = _shard_file(tmp_path, "a.json", [_row(0, {}, {}), _row(1, {}, {})], total=4)
    with pytest.raises(SystemExit, match="missing from the merge"):
        mt.merge_truncation_shards([f])


def test_merge_rejects_a_duplicated_setting(tmp_path):
    f1 = _shard_file(tmp_path, "a.json", [_row(0, {}, {}), _row(1, {}, {})], total=3)
    f2 = _shard_file(tmp_path, "b.json", [_row(1, {}, {}), _row(2, {}, {})], total=3)
    with pytest.raises(SystemExit, match="more than one shard"):
        mt.merge_truncation_shards([f1, f2])


def test_merge_rejects_mismatched_solver_builds(tmp_path):
    f1 = _shard_file(tmp_path, "a.json", [_row(0, {}, {})], total=2, solver="livespice:aaa")
    f2 = _shard_file(tmp_path, "b.json", [_row(1, {}, {})], total=2, solver="livespice:bbb")
    with pytest.raises(SystemExit, match="DIFFERENT renderer builds"):
        mt.merge_truncation_shards([f1, f2])


def test_merge_rejects_an_unfingerprintable_renderer(tmp_path):
    f = _shard_file(tmp_path, "a.json", [_row(0, {}, {})], total=1, solver="livespice:UNKNOWN")
    with pytest.raises(SystemExit, match="cannot be proven comparable"):
        mt.merge_truncation_shards([f])


def test_merge_rejects_shards_from_different_grids(tmp_path):
    f1 = _shard_file(tmp_path, "a.json", [_row(0, {}, {})], total=2)
    f2 = _shard_file(tmp_path, "b.json", [_row(1, {}, {})], total=4)
    with pytest.raises(SystemExit, match="disagree on the setting count"):
        mt.merge_truncation_shards([f1, f2])


def test_merge_rejects_mismatched_candidates_or_ref_os(tmp_path):
    f1 = _shard_file(tmp_path, "a.json", [_row(0, {}, {})], total=2, candidates=(2, 4))
    f2 = _shard_file(tmp_path, "b.json", [_row(1, {}, {})], total=2, candidates=(2, 4, 8))
    with pytest.raises(SystemExit, match="disagree on candidates/ref_os"):
        mt.merge_truncation_shards([f1, f2])

    f3 = _shard_file(tmp_path, "c.json", [_row(0, {}, {})], total=2, ref_os=32)
    f4 = _shard_file(tmp_path, "d.json", [_row(1, {}, {})], total=2, ref_os=16)
    with pytest.raises(SystemExit, match="disagree on candidates/ref_os"):
        mt.merge_truncation_shards([f3, f4])


def test_merge_accepts_a_complete_set_and_scores_identically_to_one_shard():
    rows = [
        _row(0, {"Gain": 0.0}, {"2": {"num": 1.0, "den": 50.0}}),
        _row(1, {"Gain": 0.5}, {"2": {"num": 4.0, "den": 50.0}}),
        _row(2, {"Gain": 1.0}, {"2": {"num": 2.0, "den": 50.0}}),
    ]

    def write(tmp_path, name, subset):
        return _shard_file(tmp_path, name, subset, total=3, candidates=(2,))

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        f1 = write(tp, "a.json", [rows[0], rows[2]])
        f2 = write(tp, "b.json", [rows[1]])
        merged_rows, cands, ref_os = mt.merge_truncation_shards([f1, f2])
        assert cands == (2,)
        assert ref_os == 32
        assert merged_rows == rows
        assert mt.score_rows(merged_rows, cands) == mt.score_rows(rows, cands)
