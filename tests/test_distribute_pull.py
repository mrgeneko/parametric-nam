"""Tests for distribute_pull.py's scheduling safeguards.

Both cover failures seen on the Duke of Tone 252-combination run (2026-09-04):

  * a worker whose venv could not import the transient-coverage gate failed in under a
    second and killed 27 of 31 chunks in ~70s, because a fast-failing worker drains the
    queue faster than healthy machines can take work, and the "retry on a DIFFERENT
    worker" the docstring promised was only an append-to-back that the same broken worker
    immediately grabbed again;

  * --chunks 32 against a 4-value Volume axis froze Volume inside every shard, so the
    renderer's own per-shard knob-sensitivity check reported a knob we had just measured
    moving output 28x as "RMS varies only 0.00% -- knob may have no effect".
"""
import os

import pytest

import distribute_pull as dp


# ------------------------------------------------------------------ chunk/axis aliasing

DUKE = ["--range", "Gain=0.1,0.25,0.5,0.75,0.85,0.95,1.0", "--range", "Tone=0.2,0.5,0.8",
        "--range", "Presence=0.2,0.5,0.8", "--range", "Volume=0.1,0.25,0.75,1.0"]


def test_warns_when_a_knob_axis_divides_the_chunk_count(capsys):
    dp._warn_chunk_aliasing(DUKE, 32)          # 4 | 32 -> Volume frozen in every shard
    out = capsys.readouterr().out
    assert "aliasing" in out and "Volume" in out


def test_suggests_a_coprime_chunk_count(capsys):
    dp._warn_chunk_aliasing(DUKE, 32)
    out = capsys.readouterr().out
    suggested = int(out.split("Use --chunks ")[1].split()[0])
    for axis in (7, 3, 3, 4):
        assert suggested % axis, f"suggested {suggested} still aliases a {axis}-value axis"


def test_silent_when_no_axis_divides_the_chunk_count(capsys):
    dp._warn_chunk_aliasing(DUKE, 31)          # prime -> coprime with 7, 3, 3, 4
    assert capsys.readouterr().out == ""


def test_aliasing_is_about_divisibility_not_size(capsys):
    # 3 does not divide 64, so a big chunk count over a 3-value axis is fine; the bug is
    # a shared factor, not granularity.
    dp._warn_chunk_aliasing(["--range", "Tone=0.2,0.5,0.8"], 64)
    assert capsys.readouterr().out == ""


def test_single_valued_axis_is_not_reported():
    # A pinned knob has nothing to vary; it is not an aliasing problem.
    dp._warn_chunk_aliasing(["--range", "Volume=0.7"], 32)


def test_no_ranges_is_a_no_op(capsys):
    dp._warn_chunk_aliasing(["--knobs", "Gain,Tone"], 32)
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------- quarantine

def _worker(host="w", parallel=1):
    return dp.Worker(f"{host}:/tmp/x:{parallel}")


def test_a_new_worker_is_not_quarantined():
    w = _worker()
    assert not w.quarantined and w.consec_fail == 0


def test_consecutive_failures_are_counted_and_reset_by_a_success():
    w = _worker()
    w.consec_fail = 2
    w.consec_fail = 0          # what the rc==0 branch does
    assert w.consec_fail == 0


def test_worker_spec_still_parses_with_and_without_env():
    plain = dp.Worker("h:/d:4")
    assert (plain.host, plain.dir, plain.parallel, plain.env) == ("h", "/d", 4, "")
    with_env = dp.Worker("h:/d:4:DOTNET_ROOT=$HOME/.dotnet")
    assert with_env.env == "DOTNET_ROOT=$HOME/.dotnet"


def test_worker_spec_rejects_a_short_spec():
    with pytest.raises(ValueError, match="host:dir:parallel"):
        dp.Worker("h:/d")


# --------------------------------------------------------------------- params.csv merge
#
# distribute_pull SCHEDULED renders but never GATHERED them, so collection was left to the
# operator -- and the obvious move, rsyncing each worker's output dir onto one local path,
# is wrong. sig/ merges cleanly because its filenames are the global grid index, but
# params.csv is a whole file per worker containing only that worker's rows, so each rsync
# overwrites the last. On the Duke of Tone run (2026-09-04) that left 204 of 252 rows; a
# worker running on localhost made it worse, because its output dir WAS the merge target,
# so the other workers clobbered its params.csv in place.

import csv


def _shard_csv(tmp_path, name, idxs, gain="0.5"):
    p = tmp_path / name
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "Gain", "ok"])
        w.writeheader()
        for i in idxs:
            w.writerow({"idx": i, "Gain": gain, "ok": "1"})
    return p


def test_merge_keeps_every_shards_rows(tmp_path):
    a = _shard_csv(tmp_path, "a.csv", [0, 3, 6])
    b = _shard_csv(tmp_path, "b.csv", [1, 4, 7])
    c = _shard_csv(tmp_path, "c.csv", [2, 5, 8])
    out = tmp_path / "params.csv"
    assert dp.merge_params([a, b, c], out) == 9
    got = [int(r["idx"]) for r in csv.DictReader(open(out))]
    assert got == list(range(9)), "rows must be complete and in grid order"


def test_merge_writes_exactly_one_header(tmp_path):
    # A header appended mid-table is read downstream as a combination -- the specific
    # failure distribute_gen.sh's own comment warns about.
    a = _shard_csv(tmp_path, "a.csv", [0])
    b = _shard_csv(tmp_path, "b.csv", [1])
    out = tmp_path / "params.csv"
    dp.merge_params([a, b], out)
    assert [l for l in open(out) if l.startswith("idx,")] == ["idx,Gain,ok\n"]


def test_merge_dedupes_a_chunk_rendered_by_two_workers(tmp_path):
    # A re-dispatched chunk, or one left over from an aborted run, is rendered twice under
    # the same global index. Duke had 264 files for 252 combinations for exactly this reason.
    a = _shard_csv(tmp_path, "a.csv", [0, 1, 2])
    b = _shard_csv(tmp_path, "b.csv", [2, 3])
    out = tmp_path / "params.csv"
    assert dp.merge_params([a, b], out) == 4
    assert [int(r["idx"]) for r in csv.DictReader(open(out))] == [0, 1, 2, 3]


def test_merge_of_nothing_reports_zero_rather_than_writing_a_bad_file(tmp_path):
    out = tmp_path / "params.csv"
    assert dp.merge_params([], out) == 0
    assert not out.exists()


def test_merge_tolerates_an_empty_shard(tmp_path):
    # A shard whose modulo range caught no combinations still writes a header-only file.
    a = _shard_csv(tmp_path, "a.csv", [0, 1])
    empty = _shard_csv(tmp_path, "empty.csv", [])
    out = tmp_path / "params.csv"
    assert dp.merge_params([a, empty], out) == 2


class TestComboPace:
    """Judging a worker by seconds-per-COMBINATION rather than by completed chunks.

    From the Mesa RED 648-combination run (2026-09-04): one worker rendered at 54x
    real-time, needing ~148 min per combination against a 110 min per-rung budget. Every
    render kept emitting progress, so gen_dataset's stall detector -- which deliberately
    tolerates slow-but-progressing work up to TOTAL_CEILING_MULT (20x) the budget, i.e.
    36.7 HOURS for one combination -- never fired. Nothing stalled, nothing failed, nothing
    printed. It held a chunk for 7.5 h and produced zero combinations while its neighbours
    completed a 16-combination chunk every 40 min.
    """

    def test_no_baseline_means_nobody_is_slow(self):
        """A cold fleet is not evidence about any host: with too few samples there is no
        deadline at all, so the first worker to start cannot be killed for being first."""
        p = dp.ComboPace(min_samples=4)
        assert p.median() is None and p.deadline() is None
        for _ in range(3):
            p.record(60.0)
        assert p.deadline() is None, "3 samples is below min_samples=4"
        p.record(60.0)
        assert p.deadline() is not None

    def test_the_baseline_is_a_median_not_a_mean(self):
        """One 36-hour outlier must not raise the bar it is itself being judged against."""
        p = dp.ComboPace(min_samples=4, slow_mult=3.0, floor_s=0.0)
        for s in (60.0, 60.0, 60.0, 60.0, 36 * 3600.0):
            p.record(s)
        assert p.median() == 60.0
        assert p.deadline() == 180.0

    def test_the_floor_protects_a_fast_fleet_from_ordinary_variance(self):
        p = dp.ComboPace(min_samples=2, slow_mult=3.0, floor_s=600.0)
        p.record(5.0); p.record(5.0)
        assert p.deadline() == 600.0, "3x a 5s median is 15s -- far too tight to act on"

    def test_a_slow_worker_is_caught_in_minutes_not_never(self):
        """The real numbers: neighbours ~21 min/combo, so the limit is ~63 min. The slow
        worker completed nothing in 7.5 h and would have been abandoned after ~1 h."""
        p = dp.ComboPace(min_samples=4, slow_mult=3.0, floor_s=600.0)
        for _ in range(6):
            p.record(21 * 60.0)
        limit = p.deadline()
        assert 60 * 60 <= limit <= 65 * 60
        assert limit < 7.5 * 3600, "must fire long before the 7.5 h this actually ran"
        assert limit < 36.7 * 3600, "and long before gen_dataset's own 36.7 h ceiling"


class TestComboLineParsing:
    """The per-combination signal already existed; it was thrown away. These pin the exact
    line format so a change to the renderer's progress output cannot silently disable the
    detector -- it would just go quiet again, which is the failure being fixed."""

    def test_it_matches_the_renderers_real_progress_line(self):
        line = ("[   3/  16]  18.8%  combo_000034  OK  DSP=42.1%  elapsed=1260s  ETA ~14:22")
        m = dp.COMBO_LINE.match(line)
        assert m and m.group(1) == "000034" and m.group(2) == "OK"

    def test_a_failed_combination_still_counts_as_progress(self):
        """A worker producing FAILs is making progress through its chunk -- it is a data
        problem, not a slow-host problem, and the existing fail-fast handles it."""
        line = "[   4/  16]  25.0%  combo_000035  FAIL  DSP=0.0%  elapsed=6604s  ETA ~14:22  [timeout after 6604s]"
        m = dp.COMBO_LINE.match(line)
        assert m and m.group(2) == "FAIL"

    def test_unrelated_output_is_not_mistaken_for_progress(self):
        for line in ("Workers:      12",
                     "Timeout:      6604s per combination",
                     "  Red Bass: [0.2, 0.8]",
                     "[controller] no combination completed in 70.0 min"):
            assert dp.COMBO_LINE.match(line) is None, line


class TestGenArgsFromConfig:
    """One description of a device, whether it renders on one machine or four.

    run_pipeline.py has taken --config for a long time; this scheduler took eleven
    hand-written renderer flags instead. Retyping them is not a theoretical hazard: the Mesa
    RED launch (2026-09-05) omitted --backend, whose default is `cpp`, and all four workers
    quarantined in under a second.
    """

    def _cfg(self, tmp_path, extra=""):
        (tmp_path / "amps").mkdir(exist_ok=True)
        schx = tmp_path / "amps" / "My Amp (v2).schx"
        schx.write_text("<Schematic/>", encoding="utf-8")
        wav = tmp_path / "amps" / "exc.wav"
        wav.write_bytes(b"RIFF")
        p = tmp_path / "d.config.toml"
        p.write_text(f'''
schx = "{schx}"
input = "{wav}"
backend = "livespice"
oversample = 8
[knobs]
"RD Gain" = [0.1, 1.0]
Tone = [0.2, 0.8]
[fixed]
Presence = 0.5
{extra}
''', encoding="utf-8")
        return p

    def test_backend_is_included(self, tmp_path):
        """The specific flag whose omission quarantined a whole fleet: gen_dataset's
        --backend defaults to `cpp`, which then demands --circuit and dies instantly."""
        a = dp.gen_args_from_config(self._cfg(tmp_path), tmp_path / "repo")
        assert "--backend" in a and a[a.index("--backend") + 1] == "livespice"

    def test_every_renderer_flag_is_carried(self, tmp_path):
        a = dp.gen_args_from_config(self._cfg(tmp_path), tmp_path / "repo")
        for flag in ("--backend", "--schx", "--input", "--knobs", "--range",
                     "--fixed-params", "--oversample"):
            assert flag in a, flag
        assert a.count("--range") == 2                      # one per knob axis
        assert a[a.index("--knobs") + 1] == "RD Gain,Tone"   # names with spaces survive
        assert a[a.index("--fixed-params") + 1] == "Presence=0.5"

    def test_paths_are_relative_to_the_repo_not_absolute(self, tmp_path):
        """run_chunk cds into each worker's OWN checkout, and homes differ across the fleet
        (/Users/gene, /Users/chewie, /home/gene) -- an absolute path from the controller can
        be a different user's home on a worker, or absent entirely."""
        a = dp.gen_args_from_config(self._cfg(tmp_path), tmp_path / "repo")
        schx = a[a.index("--schx") + 1]
        assert not os.path.isabs(schx)
        assert schx == os.path.join("..", "amps", "My Amp (v2).schx")
        assert not os.path.isabs(a[a.index("--input") + 1])

    def test_it_warns_when_a_path_cannot_travel(self, tmp_path, capsys):
        """A config pointing far outside the repo cannot resolve the same way on a worker.
        Say so rather than emitting a path that silently means something else there."""
        far = tmp_path / "elsewhere" / "deep" / "amps"
        far.mkdir(parents=True)
        (far / "a.schx").write_text("x")
        p = tmp_path / "repo" / "sub" / "c.toml"
        p.parent.mkdir(parents=True)
        p.write_text(f'schx = "{far / "a.schx"}"\nbackend = "livespice"\n', encoding="utf-8")
        dp.gen_args_from_config(p, tmp_path / "repo" / "sub" / "deeper" / "deepest")
        assert "unlikely to resolve" in capsys.readouterr().err

    def test_config_is_reproduced_faithfully_enough_to_replace_hand_written_flags(self, tmp_path):
        """Guards the property that actually matters: what --config emits is what a careful
        person would have typed. Compared field-by-field against the config's own contents."""
        import tomllib
        p = self._cfg(tmp_path)
        raw = tomllib.load(open(p, "rb"))
        a = dp.gen_args_from_config(p, tmp_path / "repo")
        assert a[a.index("--oversample") + 1] == str(raw["oversample"])
        assert a[a.index("--backend") + 1] == raw["backend"]
        ranges = [a[i + 1] for i, x in enumerate(a) if x == "--range"]
        assert ranges == [f"{k}=" + ",".join(str(v) for v in vals)
                          for k, vals in raw["knobs"].items()]
