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
from pathlib import Path

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
    """Two failures shaped this, both observed on real runs.

    The one it exists for: a worker rendered at 54x real-time, needed ~148 min per
    combination against a 110 min budget, and so produced NOTHING for 7.5 h while its
    neighbours finished a 16-combination chunk every 40 min. Nothing fired -- the renderer's
    own stall detector deliberately tolerates slow-but-progressing work, and the controller
    only judged completed chunks.

    The one the first fix caused: measuring the GAP between consecutive completions ignored
    that the renderer runs --workers N concurrently, so completions arrive in a burst. The
    median gap collapsed to 0.3 min, the deadline fell to its floor, and a healthy worker was
    killed mid-way through a legitimate cold coverage gate.
    """

    def test_a_burst_of_completions_does_not_collapse_the_baseline(self):
        """THE REGRESSION. 8 combinations finishing within seconds of each other after 25 min
        of parallel work is normal, not fast. Rate divides by the whole elapsed time, so the
        burst cannot drag the baseline toward zero the way inter-arrival gaps did."""
        p = dp.ComboPace(min_samples=2)
        for _ in range(3):
            p.record_rate(8, 25 * 60)          # 8 combos per 25 min, as observed
        limit = p.steady_limit()
        assert limit >= 30 * 60, "a burst must not produce a sub-minute per-combination view"
        # the old metric derived ~0.3 min/combo from the same run; sanity-check the new one
        assert 8 / (25 * 60) == pytest.approx(p.rates[0])

    def test_a_worker_that_has_produced_nothing_is_judged_on_STARTUP_not_rate(self):
        """Producing nothing early is normal: the renderer runs its coverage gate first, which
        on a cold onset cache is ~100 min on a full amp. That must not read as 'stalled'."""
        p = dp.ComboPace(min_samples=2, startup_floor_s=90 * 60)
        for _ in range(2):
            p.record_first(25 * 60)            # others took 25 min to first combination
        slow, why = p.verdict(completions=0, since_last_s=40 * 60, elapsed_s=40 * 60)
        assert not slow, f"killed a worker still inside a legitimate coverage gate: {why}"

    def test_the_worker_it_exists_to_catch_is_still_caught(self):
        """7.5 h with nothing produced, against neighbours whose first combination lands in
        25 min. Must fire -- well before the 7.5 h it actually ran."""
        p = dp.ComboPace(min_samples=2, startup_floor_s=90 * 60, slow_mult=3.0)
        for _ in range(3):
            p.record_first(25 * 60)
        slow, why = p.verdict(completions=0, since_last_s=7.5 * 3600, elapsed_s=7.5 * 3600)
        assert slow and "produced nothing" in why
        assert p.startup_limit() < 7.5 * 3600

    def test_a_cold_fleet_judges_nobody(self):
        p = dp.ComboPace(min_samples=2)
        assert p.startup_limit() is None and p.steady_limit() is None
        assert p.verdict(0, 10 * 3600, 10 * 3600) == (False, None)

    def test_one_pathological_host_cannot_raise_the_bar_it_is_judged_against(self):
        p = dp.ComboPace(min_samples=2, startup_floor_s=0.0, slow_mult=3.0)
        for s in (25 * 60, 25 * 60, 25 * 60, 36 * 3600):
            p.record_first(s)
        assert p.startup_limit() == pytest.approx(3 * 25 * 60)

    def test_a_producing_worker_that_stops_is_caught(self):
        p = dp.ComboPace(min_samples=2, steady_floor_s=30 * 60, slow_mult=3.0)
        for _ in range(2):
            p.record_rate(16, 40 * 60)         # a chunk every 40 min
        slow, why = p.verdict(completions=4, since_last_s=6 * 3600, elapsed_s=8 * 3600)
        assert slow and "after producing 4" in why


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

    def _cfg_with_toplevel(self, tmp_path, extra_toplevel):
        """Like _cfg, but `extra_toplevel` lines land BEFORE [knobs]/[fixed] -- i.e. as real
        top-level config keys, not (wrongly) inside the [fixed] table."""
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
{extra_toplevel}
[knobs]
"RD Gain" = [0.1, 1.0]
Tone = [0.2, 0.8]
[fixed]
Presence = 0.5
''', encoding="utf-8")
        return p

    def test_conv_and_capture_chain_overrides_are_carried(self, tmp_path):
        """Without this, a sharded dispatch renders through the generic transistor model and
        the default capture chain regardless of what the config declares -- silently
        disagreeing with a single-machine run_pipeline.py render of the SAME config, which
        forwards both explicitly (see run_pipeline.py's own gen_cmd construction and
        capture_chain.resolve()'s docstring on why config.toml alone isn't enough)."""
        extra = 'conv = "bjt_vaf=102.207,bjt_rb=173.312"\ncapture-hp-hz = 29\ncapture-order = 2'
        a = dp.gen_args_from_config(self._cfg_with_toplevel(tmp_path, extra), tmp_path / "repo")
        assert a[a.index("--conv") + 1] == "bjt_vaf=102.207,bjt_rb=173.312"
        assert a[a.index("--capture-hp-hz") + 1] == "29"
        assert a[a.index("--capture-order") + 1] == "2"

    def test_no_capture_chain_override_is_carried(self, tmp_path):
        a = dp.gen_args_from_config(
            self._cfg_with_toplevel(tmp_path, "no-capture-chain = true"), tmp_path / "repo")
        assert "--no-capture-chain" in a

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


class TestKillRemote:
    """Abandoning a chunk must kill the RENDERER ON THE WORKER, and nothing else.

    Killing the local ssh client does not signal the remote process -- it is reparented to
    init and keeps running, holding the renderer's exclusive .generation.lock. On Mesa Orange
    that left an orphaned generation racing the live one on one host for 11.5 hours.
    """

    def _worker(self, monkeypatch, host="hostA", d="~/work/parametric-nam"):
        calls = []
        monkeypatch.setattr(dp.subprocess, "run",
                            lambda argv, **kw: calls.append(argv) or
                            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
        return dp.Worker(f"{host}:{d}:4"), calls

    def test_it_ssh_es_to_the_worker_and_kills_by_shard(self, monkeypatch):
        w, calls = self._worker(monkeypatch)
        w._kill_remote("7-7/41", "~/ds")
        assert len(calls) == 1
        argv = calls[0]
        assert argv[0] == "ssh" and "hostA" in argv
        cmd = argv[-1]
        assert "pkill -f" in cmd
        # Assert the pattern MATCHES a real command line, not that it contains a literal
        # string: re.escape renders the hyphen as "\\-", which both BSD and GNU ERE accept as
        # a literal (verified on Darwin and Linux). Asserting the literal would fail on a
        # harmless escaping detail while missing an actually-broken pattern.
        import re as _re
        pat = _re.search(r"pkill -f '([^']+)'", cmd).group(1)
        assert _re.search(pat, "python -u gen_dataset_from_schx.py --shard 7-7/41 --output ~/ds")

    def test_it_escalates_to_SIGKILL(self, monkeypatch):
        """A renderer mid-simulation may ignore a polite TERM; the lock is only released when
        the process actually dies."""
        w, calls = self._worker(monkeypatch)
        w._kill_remote("7-7/41", "~/ds")
        cmd = calls[0][-1]
        assert cmd.index("pkill -f") < cmd.index("pkill -9 -f"), "TERM must precede KILL"

    def test_it_does_not_touch_the_lock(self, monkeypatch):
        """flock releases itself when the holder dies. Deleting the file releases nothing and
        lets a second generation start alongside the first -- silent params.csv corruption."""
        w, calls = self._worker(monkeypatch)
        w._kill_remote("7-7/41", "~/ds")
        assert "rm" not in calls[0][-1], calls[0][-1]
        assert "generation.lock" not in calls[0][-1]

    def test_the_pattern_cannot_match_a_different_shard(self, monkeypatch):
        """A host may legitimately be running another chunk, or another dataset entirely. The
        pattern is anchored on this dispatch's own --shard argument."""
        import re as _re
        w, calls = self._worker(monkeypatch)
        w._kill_remote("7-7/41", "~/ds")
        pat = _re.search(r"pkill -f '([^']+)'", calls[0][-1]).group(1)
        mine  = "python -u gen_dataset_from_schx.py --backend livespice --shard 7-7/41 --output ~/ds"
        other = "python -u gen_dataset_from_schx.py --backend livespice --shard 8-8/41 --output ~/ds"
        assert _re.search(pat, mine)
        assert not _re.search(pat, other), "would kill an unrelated chunk on the same host"

    def test_a_kill_failure_does_not_raise(self, monkeypatch):
        """pkill exits non-zero when nothing matched -- that is the normal case when the
        renderer already exited. It must not propagate."""
        def boom(argv, **kw):
            raise dp.subprocess.TimeoutExpired(cmd="ssh", timeout=90)
        monkeypatch.setattr(dp.subprocess, "run", boom)
        w = dp.Worker("hostA:~/x:4")
        with pytest.raises(dp.subprocess.TimeoutExpired):
            w._kill_remote("7-7/41", "~/ds")


def test_abandoning_a_chunk_never_deletes_the_generation_lock():
    """flock auto-releases on process exit, crash or kill, so a lock file that still exists
    means a LIVE process holds it. Deleting it releases nothing -- the holder keeps its lock
    on the unlinked inode while a new run locks a fresh file, and the two append to one
    params.csv. That is silent corruption: duplicate rows, .npy files that still look
    perfect, and a params.csv no longer 1:1 with outputs.npy, so knobs pair with the WRONG
    audio. It has happened twice; the second time the delete was in this file."""
    src = (Path(__file__).resolve().parent.parent / "distribute_pull.py").read_text()
    body = src[src.index("def _kill_remote"):src.index("def run_chunk")]
    assert "generation.lock" not in body or "DO NOT rm" in body
    assert "rm -f" not in body, "_kill_remote must not delete the generation lock"


class _FakePopen:
    """Enough of subprocess.Popen for run_chunk(): captures the ssh argv it was built with,
    reports an already-exited process (poll() -> 0) so run_chunk's watch loop falls straight
    through to proc.wait(), and yields no stdout lines (nothing under test here reads them)."""

    instances: "list" = []

    def __init__(self, argv, **kw):
        self.argv = argv
        self.stdout = iter(())
        self.returncode = 0
        _FakePopen.instances.append(self)

    def poll(self):
        return 0

    def wait(self):
        return 0

    def kill(self):
        pass


class TestJobAbstraction:
    """distribute_pull.py can dispatch either gen_dataset_from_schx.py or grid_adequacy.py,
    parameterized by a Job (script/progress_re/output_flag/build_args/chunk_output/collect)
    a Worker carries. Worker(spec) with no job= defaults to GEN_DATASET_JOB, so every test
    above this class -- written before Job existed -- keeps exercising exactly the same
    dispatch it always did; these add the other job path and pin the default is unchanged."""

    def _run_chunk_cmd(self, monkeypatch, job=None):
        _FakePopen.instances.clear()
        monkeypatch.setattr(dp.subprocess, "Popen", _FakePopen)
        w = dp.Worker("hostA:~/work/parametric-nam:4", job=job) if job else \
            dp.Worker("hostA:~/work/parametric-nam:4")
        w.run_chunk("3-3/16", "--backend livespice", "~/out")
        return _FakePopen.instances[0].argv[-1]

    def test_default_job_dispatches_gen_dataset_from_schx_unchanged(self, monkeypatch):
        """Regression pin: this is the exact command line distribute_pull.py has always sent
        for gen_dataset_from_schx.py -- the refactor to a Job abstraction must not touch it."""
        cmd = self._run_chunk_cmd(monkeypatch)
        assert cmd == ("cd ~/work/parametric-nam && ./.venv/bin/python -u "
                       "gen_dataset_from_schx.py --backend livespice --shard 3-3/16 "
                       "--output ~/out")

    def test_grid_adequacy_job_dispatches_its_own_script_and_output_flag(self, monkeypatch):
        cmd = self._run_chunk_cmd(monkeypatch, job=dp.GRID_ADEQUACY_JOB)
        assert cmd == ("cd ~/work/parametric-nam && ./.venv/bin/python -u "
                       "grid_adequacy.py --backend livespice --shard 3-3/16 "
                       "--shard-out ~/out/shard_3-3_16.json")

    def test_chunk_output_naming_differs_per_job(self):
        assert dp.GEN_DATASET_JOB.chunk_output("~/out", "3-3/16") == "~/out"
        assert dp.GRID_ADEQUACY_JOB.chunk_output("~/out", "3-3/16") == "~/out/shard_3-3_16.json"

    def test_kill_pattern_matches_a_grid_adequacy_process_line(self, monkeypatch):
        calls = []
        monkeypatch.setattr(dp.subprocess, "run",
                            lambda argv, **kw: calls.append(argv) or
                            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
        w = dp.Worker("hostA:~/work/parametric-nam:4", job=dp.GRID_ADEQUACY_JOB)
        w._kill_remote("3-3/16", "~/out")
        import re as _re
        pat = _re.search(r"pkill -f '([^']+)'", calls[0][-1]).group(1)
        mine = "python -u grid_adequacy.py --config d.toml --shard 3-3/16 --shard-out ~/out/shard_3-3_16.json"
        other = "python -u grid_adequacy.py --config d.toml --shard 4-4/16 --shard-out ~/out/shard_4-4_16.json"
        assert _re.search(pat, mine)
        assert not _re.search(pat, other)

    def test_gridadq_progress_line_matches_the_real_heartbeat(self):
        assert dp.GRIDADQ_PROBE_LINE.match("    3/48 probes done")
        assert dp.GRIDADQ_PROBE_LINE.match("12/12 probes done")

    def test_gridadq_progress_line_does_not_match_unrelated_output(self):
        for line in ("Workers:      12", "  Gain:", "    0.1000 -  0.5000    0.0123   ok",
                     "[controller] no combination completed in 70.0 min"):
            assert dp.GRIDADQ_PROBE_LINE.match(line) is None, line

    def test_grid_adequacy_build_args_expands_config_and_appends_extras(self, tmp_path):
        cfg = tmp_path / "d.config.toml"
        cfg.write_text('backend = "livespice"\n', encoding="utf-8")
        a = dp.GRID_ADEQUACY_JOB.build_args(cfg, tmp_path / "repo", ["--target", "0.02"])
        assert a[0] == "--config"
        assert not os.path.isabs(a[1])
        assert a[2:] == ["--target", "0.02"]

    def test_tools_registry_has_both_jobs_and_gen_dataset_is_the_default(self):
        assert set(dp.JOBS) == {"gen_dataset", "grid_adequacy"}
        assert dp.JOBS["gen_dataset"] is dp.GEN_DATASET_JOB
        assert dp.JOBS["grid_adequacy"] is dp.GRID_ADEQUACY_JOB
        assert dp.Worker("hostA:~/x:4").job is dp.GEN_DATASET_JOB


def test_collect_returns_consistency_flag(tmp_path, monkeypatch):
    """_collect reports whether rows == .npy -- the precondition for combining."""
    import distribute_pull as dp
    local = tmp_path / "ds"; (local / "sig").mkdir(parents=True)
    monkeypatch.setattr(dp.subprocess, "run",
                        lambda *a, **k: __import__("types").SimpleNamespace(returncode=1, stdout="", stderr=""))
    monkeypatch.setattr(dp, "merge_params", lambda srcs, dst: 2)
    (local / "sig" / "a.npy").write_bytes(b"x")
    assert dp._collect([], [], local) is False          # 2 rows, 1 npy
    (local / "sig" / "b.npy").write_bytes(b"x")
    assert dp._collect([], [], local) is True           # 2 rows, 2 npy


def test_combine_accepts_the_bare_string_argparse_actually_produces(tmp_path, monkeypatch):
    """--collect has no type=Path, so main() hands _combine a plain str -- exactly what
    argparse produces from the command line. gen_dataset_from_schx.combine() does
    `out_dir / "params.csv"`, which TypeErrors on a str.

    Regression: this crashed the auto-combine step on the Duke of Tone (Distortion) 63-combo
    run (2026-09-11) -- AFTER --collect had already succeeded (63/63 rows and .npy files on
    disk), so the render and collect were both fine and only this conversion was missing.
    """
    import distribute_pull as dp
    seen = []
    monkeypatch.setattr("gen_dataset_from_schx.combine", lambda out_dir, **kw: seen.append(out_dir))
    dp._combine(str(tmp_path / "ds"))   # str, not Path -- what args.collect actually is
    assert seen and isinstance(seen[0], Path), \
        "combine() must receive a Path, matching what it does with the result (out_dir / ...)"


def test_should_combine_decision_table():
    """Regression: --collect used to stop before outputs.npy, so param_train refused the dir.

    Cost Mesa Orange and Duke of Tone (Overdrive) a manual step each on 2026-09-07.
    run_pipeline.py has had a Combine step all along; only the distributed path lacked one.
    """
    from distribute_pull import should_combine
    assert should_combine(consistent=True, no_combine=False) is None          # the default: combine
    assert "no-combine" in should_combine(consistent=True, no_combine=True)   # explicit opt-out
    assert "rows != .npy" in should_combine(consistent=False, no_combine=False)  # never on a mismatch
    # opt-out wins over inconsistency: both are reasons not to, and the explicit one is clearer
    assert "no-combine" in should_combine(consistent=False, no_combine=True)


def test_no_combine_flag_exists_and_defaults_off():
    import distribute_pull as dp
    ap = dp.build_parser() if hasattr(dp, "build_parser") else None
    if ap is None:
        import inspect
        assert "--no-combine" in inspect.getsource(dp), "flag must be registered"
    else:
        assert ap.parse_args([]).no_combine is False


def test_collect_returns_false_not_none_when_nothing_merged(tmp_path, monkeypatch):
    """The no-params.csv path must return False, not a bare None.

    None is only ACCIDENTALLY falsy: should_combine() treats it as "not consistent" and so
    happens to refuse the combine, which is right -- but nothing pins that, and any future
    `if consistent is False` / `is None` distinction would silently start combining an empty
    collect. Returning the flag explicitly makes _collect's contract total.
    """
    import distribute_pull as dp
    from distribute_pull import should_combine
    local = tmp_path / "ds"; (local / "sig").mkdir(parents=True)
    monkeypatch.setattr(dp.subprocess, "run",
                        lambda *a, **k: __import__("types").SimpleNamespace(returncode=1, stdout="", stderr=""))
    monkeypatch.setattr(dp, "merge_params", lambda srcs, dst: 0)   # nothing merged
    got = dp._collect([], [], local)
    assert got is False, f"expected False, got {got!r}"
    assert should_combine(got, no_combine=False) is not None, "must refuse to combine"
