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
import subprocess
import sys
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


def test_worker_spec_rejects_a_spec_with_no_dir():
    with pytest.raises(ValueError, match="host:dir"):
        dp.Worker("h")


def test_worker_spec_auto_detects_parallel_when_omitted(monkeypatch):
    # PARALLEL became optional 2026-09-26 -- omitting it (or leaving it empty/"auto") queries
    # cpu_topology.physical_cpu_count() instead of requiring an explicit int. Mocked here so
    # the test is fast/deterministic rather than actually SSHing anywhere.
    seen = []
    monkeypatch.setattr(dp, "physical_cpu_count", lambda host=None: seen.append(host) or 7)

    omitted = dp.Worker("h:/d")
    assert (omitted.host, omitted.dir, omitted.parallel, omitted.env) == ("h", "/d", 7, "")
    assert seen[-1] == "h"

    empty_with_env = dp.Worker("h:/d::VAR=1")
    assert (empty_with_env.parallel, empty_with_env.env) == (7, "VAR=1")

    auto_keyword = dp.Worker("h:/d:auto")
    assert auto_keyword.parallel == 7

    # localhost/127.0.0.1 probe with host=None (local detection), not host="localhost" --
    # SSHing to yourself to ask your own core count would be silly when the tools this
    # dispatches to already run local detection for free.
    local = dp.Worker("localhost:/d")
    assert seen[-1] is None


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
    n, idx = dp.merge_params([a, b, c], out)
    assert n == 9
    assert idx == list(range(9))
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
    n, idx = dp.merge_params([a, b], out)
    assert n == 4
    assert idx == [0, 1, 2, 3]
    assert [int(r["idx"]) for r in csv.DictReader(open(out))] == [0, 1, 2, 3]


def test_merge_of_nothing_reports_zero_rather_than_writing_a_bad_file(tmp_path):
    out = tmp_path / "params.csv"
    assert dp.merge_params([], out) == (0, [])
    assert not out.exists()


def test_merge_tolerates_an_empty_shard(tmp_path):
    # A shard whose modulo range caught no combinations still writes a header-only file.
    a = _shard_csv(tmp_path, "a.csv", [0, 1])
    empty = _shard_csv(tmp_path, "empty.csv", [])
    out = tmp_path / "params.csv"
    n, idx = dp.merge_params([a, empty], out)
    assert n == 2
    assert idx == [0, 1]


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

    def test_ngspice_deck_fields_are_carried(self, tmp_path):
        """This scheduler had never dispatched an ngspice-deck device before JC-120's
        sag/reactive-speaker render (2026-09-26) -- every chunk quarantined in seconds with
        "--pedal-dir and --module are required for --backend ngspice-deck", the exact "Mesa
        RED omitted --backend" failure mode test_backend_is_included guards, just for a
        newer field set. pedal-dir is a PATH (needs repo-relative treatment like schx/input);
        module/probe-node/maxstep are plain values."""
        (tmp_path / "amps").mkdir(exist_ok=True)
        pedal_dir = tmp_path / "amps"
        wav = pedal_dir / "exc.wav"
        wav.write_bytes(b"RIFF")
        p = tmp_path / "d.config.toml"
        p.write_text(f'''
input = "{wav}"
backend = "ngspice-deck"
pedal-dir = "{pedal_dir}"
module = "gen_jc120_ch1_ngspice"
probe-node = "nspout"
maxstep = 1e-05
[knobs]
Volume = [0.1, 1.0]
[fixed]
Bright = 0.0
''', encoding="utf-8")
        a = dp.gen_args_from_config(p, tmp_path / "repo")
        assert a[a.index("--pedal-dir") + 1] == os.path.join("..", "amps")
        assert not os.path.isabs(a[a.index("--pedal-dir") + 1])
        assert a[a.index("--module") + 1] == "gen_jc120_ch1_ngspice"
        assert a[a.index("--probe-node") + 1] == "nspout"
        assert a[a.index("--maxstep") + 1] == "1e-05"

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


class TestCheckTransientCoverageArgsFromConfig:
    """check_transient_coverage.py takes the same one-flag --config interface as
    grid_adequacy.py, so this builder is trivially identical to
    grid_adequacy_args_from_config -- closes the "transient-coverage isn't sharded yet" gap
    config-gate-proposal.md and per-item-sharding-proposal.md both flag."""

    def test_repo_relative_config_path(self, tmp_path):
        cfg = tmp_path / "amps" / "d.config.toml"
        cfg.parent.mkdir()
        cfg.write_text("x")
        repo = tmp_path / "repo"
        repo.mkdir()
        args = dp.check_transient_coverage_args_from_config(cfg, repo)
        assert args == ["--config", "../amps/d.config.toml"]

    def test_absolute_path_kept_when_not_a_repo_sibling(self, tmp_path):
        cfg = Path("/some/other/place/d.config.toml")
        args = dp.check_transient_coverage_args_from_config(cfg, tmp_path)
        assert args[0] == "--config"


class TestTcovCornerLine:
    """The stall-detector/pacing signal for check_transient_coverage.py's per-corner
    completion line (_measure's own print, both sharded and unsharded). No running N/M count
    the way GRIDADQ_PROBE_LINE has, but this fires exactly once per completed corner."""

    def test_matches_a_passing_corner(self):
        line = f'{"all-min":16} onset={"     1.234 V":>10}  OK'
        assert dp.TCOV_CORNER_LINE.match(line)

    def test_matches_a_skipped_corner_with_a_long_trailing_message(self):
        line = (f'{"Gain=lo-solo":16} onset={"NONE":>10}  '
               f'SKIP (every render in the sweep failed -- see stderr for why)')
        assert dp.TCOV_CORNER_LINE.match(line)

    def test_matches_a_failing_corner(self):
        line = f'{"Volume=hi-solo":16} onset={"0.512 V":>10}  FAIL -- transient never reaches saturation here'
        assert dp.TCOV_CORNER_LINE.match(line)

    def test_does_not_match_an_unrelated_line(self):
        assert not dp.TCOV_CORNER_LINE.match("Transient saturation coverage: Some Amp")

    def test_does_not_match_the_merge_reports_summary_lines(self):
        assert not dp.TCOV_CORNER_LINE.match("PASSED: transient content reaches saturation at every checked corner.")
        assert not dp.TCOV_CORNER_LINE.match("FAILED: 2/25 corners never see a transient past their own onset.")


class TestCheckTransientCoverageJob:
    def test_registered_under_its_own_name(self):
        assert dp.JOBS["check_transient_coverage"] is dp.CHECK_TRANSIENT_COVERAGE_JOB

    def test_script_and_output_flag(self):
        j = dp.CHECK_TRANSIENT_COVERAGE_JOB
        assert j.script == "check_transient_coverage.py"
        assert j.output_flag == "--emit-onsets"

    def test_chunk_output_uses_a_distinct_prefix_from_grid_adequacy(self):
        gj = dp.GRID_ADEQUACY_JOB.chunk_output("/out", "3-3/16")
        tj = dp.CHECK_TRANSIENT_COVERAGE_JOB.chunk_output("/out", "3-3/16")
        assert gj != tj
        assert tj == "/out/tcov_shard_3-3_16.json"
        assert gj == "/out/shard_3-3_16.json"

    def test_collect_ignores_no_combine_and_repair_missing_like_its_siblings(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(dp, "_collect_check_transient_coverage",
                            lambda *a: seen.setdefault("called", a))
        dp.CHECK_TRANSIENT_COVERAGE_JOB.collect(["w"], ["/out"], "/local", "cfg.toml", [],
                                                no_combine=True, repair_missing=True)
        assert seen["called"] == (["w"], ["/out"], "/local", "cfg.toml", [])


class TestCollectCheckTransientCoverage:
    """Mirrors TestCollectLabelsAndExpectedCount's fake-rsync technique (a real local file
    copy standing in for the network hop) so the actual merge invocation runs for real."""

    class FakeW:
        def __init__(self, host):
            self.host = host

    def _fake_rsync(self, monkeypatch):
        import shutil
        merge_calls = []
        def run(cmd, **kw):
            if cmd[0] == "rsync":
                src_spec, dst = cmd[2], cmd[3]
                _, _, src = src_spec.partition(":")
                src_path = Path(src)
                if not src_path.exists():
                    return subprocess.CompletedProcess(cmd, 1, "", "no such file")
                shutil.copytree(src_path, dst, dirs_exist_ok=True)
                return subprocess.CompletedProcess(cmd, 0, "", "")
            merge_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "merged ok\n", "")
        monkeypatch.setattr(dp.subprocess, "run", run)
        return merge_calls

    def test_merges_tcov_shard_files_not_grid_adequacys(self, tmp_path, monkeypatch):
        merge_calls = self._fake_rsync(monkeypatch)
        shard_dir = tmp_path / "worker_out"
        shard_dir.mkdir()
        (shard_dir / "tcov_shard_0-0_2.json").write_text("{}")
        (shard_dir / "shard_0-0_2.json").write_text("{}")   # a grid_adequacy file -- must be ignored
        local = tmp_path / "merged"
        logged = []
        monkeypatch.setattr(dp, "log", logged.append)
        dp._collect_check_transient_coverage([self.FakeW("h1")], [str(shard_dir)], local,
                                              Path("cfg.toml"), [])
        assert any("merging 1 shard" in m for m in logged)
        # The actual invocation, not just the log line: check_transient_coverage.py's flag is
        # --merge-onsets, NOT --merge (grid_adequacy.py's own flag name) -- a real, distinct
        # mistake to guard against given how closely these two collectors mirror each other.
        assert len(merge_calls) == 1
        cmd = merge_calls[0]
        assert cmd[1].endswith("check_transient_coverage.py")
        assert "--merge-onsets" in cmd and "--merge" not in cmd   # exact-token check: distinct flags
        assert str(local / "tcov_shard_0-0_2.json") in cmd
        assert str(local / "shard_0-0_2.json") not in cmd   # the grid_adequacy file, excluded

    def test_no_shards_found_logs_and_does_not_crash(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        empty = tmp_path / "empty"
        empty.mkdir()
        local = tmp_path / "merged"
        logged = []
        monkeypatch.setattr(dp, "log", logged.append)
        dp._collect_check_transient_coverage([self.FakeW("h1")], [str(empty)], local,
                                              Path("cfg.toml"), [])
        assert any("nothing to merge" in m for m in logged)


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

    def test_measure_truncation_job_dispatches_its_own_script_and_output_flag(self, monkeypatch):
        cmd = self._run_chunk_cmd(monkeypatch, job=dp.MEASURE_TRUNCATION_JOB)
        assert cmd == ("cd ~/work/parametric-nam && ./.venv/bin/python -u "
                       "measure_truncation.py --backend livespice --shard 3-3/16 "
                       "--emit ~/out/shard_3-3_16.json")

    def test_chunk_output_naming_for_measure_truncation_matches_grid_adequacy_convention(self):
        assert (dp.MEASURE_TRUNCATION_JOB.chunk_output("~/out", "3-3/16")
                == "~/out/shard_3-3_16.json")

    def test_measure_trunc_progress_line_matches_the_real_heartbeat(self):
        assert dp.MEASURE_TRUNC_LINE.match("  3/15 settings done")
        assert dp.MEASURE_TRUNC_LINE.match("12/12 settings done")

    def test_measure_trunc_progress_line_does_not_match_unrelated_output(self):
        for line in ("Workers:      12", "    3/48 probes done",
                      "[controller] no combination completed in 70.0 min"):
            assert dp.MEASURE_TRUNC_LINE.match(line) is None, line

    def test_measure_truncation_build_args_expands_config_and_input_and_appends_extras(self, tmp_path):
        cfg = tmp_path / "d.config.toml"
        cfg.write_text('backend = "livespice"\ninput = "%s"\n' % (tmp_path / "sweep.wav"),
                       encoding="utf-8")
        a = dp.measure_truncation_args_from_config(cfg, tmp_path / "repo")
        assert a == ["--config", os.path.relpath(cfg, tmp_path / "repo"),
                     "--input", os.path.relpath(tmp_path / "sweep.wav", tmp_path / "repo")]
        a2 = dp.MEASURE_TRUNCATION_JOB.build_args(cfg, tmp_path / "repo", ["--ref-os", "16"])
        assert a2[-2:] == ["--ref-os", "16"]

    def test_measure_truncation_build_args_omits_input_when_config_has_none(self, tmp_path):
        cfg = tmp_path / "d.config.toml"
        cfg.write_text('backend = "livespice"\n', encoding="utf-8")
        a = dp.measure_truncation_args_from_config(cfg, tmp_path / "repo")
        assert a == ["--config", os.path.relpath(cfg, tmp_path / "repo")]

    def test_tools_registry_has_all_jobs_and_gen_dataset_is_the_default(self):
        assert set(dp.JOBS) == {"gen_dataset", "grid_adequacy", "measure_truncation",
                                "check_transient_coverage"}
        assert dp.JOBS["gen_dataset"] is dp.GEN_DATASET_JOB
        assert dp.JOBS["grid_adequacy"] is dp.GRID_ADEQUACY_JOB
        assert dp.JOBS["measure_truncation"] is dp.MEASURE_TRUNCATION_JOB
        assert dp.JOBS["check_transient_coverage"] is dp.CHECK_TRANSIENT_COVERAGE_JOB
        assert dp.Worker("hostA:~/x:4").job is dp.GEN_DATASET_JOB


class _FakeSSHDropPopen:
    """Simulates the SSH-drop shape of run_chunk's watch loop: stdout hits EOF immediately
    (broken pipe, nothing read), so the pump thread dies with `done == 0` and the loop exits
    via `if not t.is_alive(): break` -- never through the pace/killed_slow branch, since pace
    is None here just like a plain (non-slow-watching) dispatch. proc.wait() then surfaces
    ssh's own exit code for a dropped connection."""

    instances: "list" = []

    def __init__(self, argv, **kw):
        self.argv = argv
        self.stdout = iter(())
        _FakeSSHDropPopen.instances.append(self)

    def poll(self):
        return None

    def wait(self):
        self.returncode = 255
        return 255

    def kill(self):
        pass


class TestKillRemoteOnFailure:
    """Regression for the 2026-09-24 AC30 Top Boost incident: one dropped SSH connection
    (rc=255) left the remote gen_dataset process orphaned and still holding
    .generation.lock, which then failed the next 30 chunks dispatched to that host --
    _kill_remote() existed and was documented for exactly this, but was only wired to the
    explicit pace-based killed_slow branch, never to a plain connection drop."""

    def test_ssh_drop_calls_kill_remote_even_without_pace(self, monkeypatch):
        monkeypatch.setattr(dp.subprocess, "Popen", _FakeSSHDropPopen)
        calls = []
        monkeypatch.setattr(dp.Worker, "_kill_remote",
                            lambda self, chunk, output: calls.append((chunk, output)))
        w = dp.Worker("hostA:~/work/parametric-nam:4")
        rc, dt, out = w.run_chunk("3-3/16", "--backend livespice", "~/out")
        assert rc == 255
        assert calls == [("3-3/16", "~/out")]

    def test_successful_chunk_does_not_call_kill_remote(self, monkeypatch):
        """The new unconditional-on-failure call must stay off the ordinary success path --
        every chunk paying an extra ssh round-trip would be real, needless overhead at
        fleet scale."""
        monkeypatch.setattr(dp.subprocess, "Popen", _FakePopen)
        calls = []
        monkeypatch.setattr(dp.Worker, "_kill_remote",
                            lambda self, chunk, output: calls.append((chunk, output)))
        w = dp.Worker("hostA:~/work/parametric-nam:4")
        rc, dt, out = w.run_chunk("3-3/16", "--backend livespice", "~/out")
        assert rc == 0
        assert calls == []


def test_collect_returns_consistency_flag(tmp_path, monkeypatch):
    """_collect reports whether rows == .npy -- the precondition for combining."""
    import distribute_pull as dp
    local = tmp_path / "ds"; (local / "sig").mkdir(parents=True)
    monkeypatch.setattr(dp.subprocess, "run",
                        lambda *a, **k: __import__("types").SimpleNamespace(returncode=1, stdout="", stderr=""))
    monkeypatch.setattr(dp, "merge_params", lambda srcs, dst: (2, [0, 1]))
    (local / "sig" / "000000.npy").write_bytes(b"x")
    assert dp._collect([], [], local) is False          # 2 rows, 1 npy
    (local / "sig" / "000001.npy").write_bytes(b"x")
    assert dp._collect([], [], local) is True           # 2 rows, 2 npy


def test_collect_names_the_exact_orphaned_indices(tmp_path, monkeypatch, capsys):
    """A rows/.npy mismatch must name WHICH indices are affected, not just report two counts.

    Regression: the Ceriatone Captain Reverb (sag) run (2026-09-21) had 880 rows / 882 .npy
    files, and finding the two culprits (801, 842) took a manual npy-vs-csv diff because the
    log only ever printed the two counts. Without --repair-missing this must still surface the
    exact indices so a human (or a future caller) doesn't have to re-derive them by hand.
    """
    import distribute_pull as dp
    local = tmp_path / "ds"; (local / "sig").mkdir(parents=True)
    monkeypatch.setattr(dp.subprocess, "run",
                        lambda *a, **k: __import__("types").SimpleNamespace(returncode=1, stdout="", stderr=""))
    monkeypatch.setattr(dp, "merge_params", lambda srcs, dst: (2, [0, 1]))
    for name in ("000000.npy", "000001.npy", "000005.npy"):
        (local / "sig" / name).write_bytes(b"x")
    logged = []
    monkeypatch.setattr(dp, "log", logged.append)
    assert dp._collect([], [], local) is False
    text = "\n".join(logged)
    assert "5" in text, f"the orphaned index (5) must be named in the log, got: {logged!r}"


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


def test_repair_missing_flag_exists_and_defaults_off():
    import distribute_pull as dp
    ap = dp.build_parser() if hasattr(dp, "build_parser") else None
    if ap is None:
        import inspect
        assert "--repair-missing" in inspect.getsource(dp), "flag must be registered"
    else:
        assert ap.parse_args([]).repair_missing is False


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
    monkeypatch.setattr(dp, "merge_params", lambda srcs, dst: (0, []))   # nothing merged
    got = dp._collect([], [], local)
    assert got is False, f"expected False, got {got!r}"
    assert should_combine(got, no_combine=False) is not None, "must refuse to combine"


class TestGateCheckIntegration:
    """distribute_pull.py's own wiring of the gate check (the WARN-vs-ABORT decision itself is
    gate_check_outcome, tested in test_run_pipeline.py). Runs main() for real against a minimal
    config, faking only verify_gate and blocking Worker construction so nothing is actually
    dispatched -- proves --require-gate aborts BEFORE any worker is touched, and that the other
    flags/tool gate it correctly."""

    def _cfg(self, tmp_path):
        (tmp_path / "amps").mkdir(exist_ok=True)
        schx = tmp_path / "amps" / "Amp.schx"
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
Gain = [0.1, 1.0]
''', encoding="utf-8")
        return p

    def _no_dispatch(self, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("a Worker was constructed -- dispatch was not supposed to happen")
        monkeypatch.setattr(dp, "Worker", boom)

    def _argv(self, cfg, tmp_path, *extra):
        return ["distribute_pull.py", "--worker", "host:/tmp/x", "--config", str(cfg),
                "--output", str(tmp_path / "out"), *extra]

    def test_require_gate_aborts_before_any_worker_is_touched(self, tmp_path, monkeypatch):
        import gate_config
        monkeypatch.setattr(gate_config, "verify_gate", lambda *a, **kw: (False, "no gate sidecar"))
        self._no_dispatch(monkeypatch)
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path, "--require-gate"))
        with pytest.raises(SystemExit) as exc:
            dp.main()
        assert exc.value.code == 2

    def test_default_warns_and_still_reaches_dispatch(self, tmp_path, monkeypatch):
        import gate_config
        monkeypatch.setattr(gate_config, "verify_gate", lambda *a, **kw: (False, "no gate sidecar"))
        reached = {}
        monkeypatch.setattr(dp, "Worker", lambda *a, **kw: reached.setdefault("hit", True) or None)
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path))
        with pytest.raises(Exception):   # goes on to fail elsewhere (no real host) -- fine
            dp.main()
        assert reached.get("hit") is True

    def test_skip_gate_check_never_calls_verify_gate(self, tmp_path, monkeypatch):
        import gate_config
        def boom(*a, **kw):
            raise AssertionError("verify_gate should not be called under --skip-gate-check")
        monkeypatch.setattr(gate_config, "verify_gate", boom)
        monkeypatch.setattr(dp, "Worker", lambda *a, **kw: None)
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path, "--skip-gate-check"))
        with pytest.raises(Exception):
            dp.main()

    def test_non_gen_dataset_tool_never_calls_verify_gate(self, tmp_path, monkeypatch):
        import gate_config
        def boom(*a, **kw):
            raise AssertionError("verify_gate should not be called for --tool grid_adequacy")
        monkeypatch.setattr(gate_config, "verify_gate", boom)
        monkeypatch.setattr(dp, "Worker", lambda *a, **kw: None)
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path, "--tool", "grid_adequacy"))
        with pytest.raises(Exception):
            dp.main()

    def test_passing_gate_does_not_abort(self, tmp_path, monkeypatch):
        import gate_config
        monkeypatch.setattr(gate_config, "verify_gate", lambda *a, **kw: (True, "matches"))
        reached = {}
        monkeypatch.setattr(dp, "Worker", lambda *a, **kw: reached.setdefault("hit", True) or None)
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path, "--require-gate"))
        with pytest.raises(Exception):
            dp.main()
        assert reached.get("hit") is True


def _cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr=err)


class TestExtractBackend:
    def test_finds_backend_value(self):
        assert dp.extract_backend(["--schx", "x.schx", "--backend", "livespice", "--input", "y"]) == "livespice"

    def test_missing_flag_is_none(self):
        assert dp.extract_backend(["--schx", "x.schx"]) is None

    def test_flag_with_no_value_is_none_not_a_crash(self):
        assert dp.extract_backend(["--backend"]) is None


class TestVersionCheckCommand:
    def test_cds_into_the_worker_dir(self):
        assert dp.version_check_command("/remote/repo", "livespice").startswith("cd /remote/repo && ")

    def test_reads_the_sha_then_self_invokes_solver_identity(self):
        cmd = dp.version_check_command("/r", "livespice")
        assert "git rev-parse HEAD" in cmd
        assert "solver_identity('livespice')" in cmd

    def test_missing_backend_defaults_to_cpp(self):
        assert "solver_identity('cpp')" in dp.version_check_command("/r", None)


class TestParseVersionCheckOutput:
    def test_two_lines(self):
        assert dp.parse_version_check_output("abc123\nlivespice:def456+789abc\n") == \
            ("abc123", "livespice:def456+789abc")

    def test_blank_lines_are_ignored(self):
        assert dp.parse_version_check_output("\nabc123\n\nlivespice:x\n") == ("abc123", "livespice:x")

    def test_one_line_is_unparseable(self):
        assert dp.parse_version_check_output("abc123\n") == (None, None)

    def test_empty_is_unparseable(self):
        assert dp.parse_version_check_output("") == (None, None)


class TestCompareVersions:
    def test_matching_sha_and_solver_passes(self):
        ok, reason = dp.compare_versions("abc123def456", "abc123def456", "livespice:x+y", "livespice:x+y")
        assert ok and "matches" in reason

    def test_worker_sha_none_refuses(self):
        ok, reason = dp.compare_versions(None, "abc123", None, "livespice:x")
        assert not ok and "commit SHA" in reason

    def test_sha_mismatch_refuses(self):
        ok, reason = dp.compare_versions("aaa000000000", "bbb111111111", None, None)
        assert not ok and "commit mismatch" in reason
        assert "aaa000000000" in reason and "bbb111111111" in reason

    def test_solver_mismatch_with_matching_sha_refuses(self):
        ok, reason = dp.compare_versions("abc123def456", "abc123def456",
                                         "livespice:aaa+bbb", "livespice:ccc+ddd")
        assert not ok and "solver mismatch" in reason

    def test_both_unknown_solver_does_not_refuse(self):
        # "UNKNOWN" means couldn't determine, not "definitely different" -- see compare_versions'
        # own docstring. The 472-commit-stale incident this item targets is caught by the SHA
        # check regardless; the solver half must not manufacture a false refusal here.
        ok, _ = dp.compare_versions("abc123def456", "abc123def456", "livespice:UNKNOWN", "livespice:x+y")
        assert ok

    def test_non_livespice_backends_compare_equal_with_no_special_casing(self):
        ok, _ = dp.compare_versions("abc123def456", "abc123def456",
                                    "ngspice-deck:unidentified", "ngspice-deck:unidentified")
        assert ok

    def test_reason_names_both_shas_short_form(self):
        _, reason = dp.compare_versions("aaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbb", None, None)
        assert "aaaaaaaaaaaa" in reason and "bbbbbbbbbbbb" in reason   # 12-char, not the full sha


class TestProbeWorkerVersion:
    def test_success_parses_both_lines(self, monkeypatch):
        monkeypatch.setattr(dp.subprocess, "run", lambda *a, **kw: _cp(0, "abc123\nlivespice:x+y\n"))
        assert dp.probe_worker_version("host", "/r", "livespice") == ("abc123", "livespice:x+y")

    def test_nonzero_exit_is_none_none(self, monkeypatch):
        monkeypatch.setattr(dp.subprocess, "run", lambda *a, **kw: _cp(1, "", "not found"))
        assert dp.probe_worker_version("host", "/r", "livespice") == (None, None)

    def test_timeout_is_none_none_not_raised(self, monkeypatch):
        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=20)
        monkeypatch.setattr(dp.subprocess, "run", boom)
        assert dp.probe_worker_version("host", "/r", "livespice") == (None, None)

    def test_ssh_binary_missing_is_none_none_not_raised(self, monkeypatch):
        def boom(*a, **kw):
            raise OSError("no such file")
        monkeypatch.setattr(dp.subprocess, "run", boom)
        assert dp.probe_worker_version("host", "/r", "livespice") == (None, None)


class TestVerifyWorkers:
    class FakeWorker:
        def __init__(self, host, d):
            self.host, self.dir = host, d

    def test_matching_workers_are_kept(self, monkeypatch):
        import prepare_excitation
        monkeypatch.setattr(dp, "local_commit_sha", lambda: "abc123def456")
        monkeypatch.setattr(dp, "probe_worker_version", lambda host, d, b: ("abc123def456", "livespice:x"))
        monkeypatch.setattr(prepare_excitation, "solver_identity", lambda backend: "livespice:x")
        w = self.FakeWorker("h1", "/r")
        assert dp.verify_workers([w], "livespice") == [w]

    def test_mismatched_worker_is_excluded_not_fatal_here(self, monkeypatch):
        monkeypatch.setattr(dp, "local_commit_sha", lambda: "abc123def456")
        monkeypatch.setattr(dp, "probe_worker_version", lambda host, d, b: ("zzz999999999", None))
        w = self.FakeWorker("h1", "/r")
        kept = dp.verify_workers([w], None)
        assert kept == []   # verify_workers only filters; main() decides whether that's fatal

    def test_mixed_fleet_keeps_only_the_matching_one(self, monkeypatch):
        monkeypatch.setattr(dp, "local_commit_sha", lambda: "abc123def456")
        def fake_probe(host, d, b):
            return ("abc123def456", None) if host == "good" else ("zzz999999999", None)
        monkeypatch.setattr(dp, "probe_worker_version", fake_probe)
        good, bad = self.FakeWorker("good", "/r"), self.FakeWorker("bad", "/r")
        kept = dp.verify_workers([good, bad], None)
        assert kept == [good]

    def test_controller_not_a_git_checkout_skips_the_whole_check(self, monkeypatch):
        monkeypatch.setattr(dp, "local_commit_sha", lambda: None)
        def boom(*a, **kw):
            raise AssertionError("probe_worker_version should not run when the controller "
                                 "has no determinable SHA")
        monkeypatch.setattr(dp, "probe_worker_version", boom)
        w = self.FakeWorker("h1", "/r")
        assert dp.verify_workers([w], None) == [w]   # unchanged, not filtered to empty


class TestVersionCheckCliIntegration:
    """Real wiring in main(): --skip-version-check bypasses entirely; a mismatched worker is
    excluded and (if it was the only one) the run errors out before any dispatch."""

    def _cfg(self, tmp_path):
        (tmp_path / "amps").mkdir(exist_ok=True)
        schx = tmp_path / "amps" / "Amp.schx"
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
Gain = [0.1, 1.0]
''', encoding="utf-8")
        return p

    def _argv(self, cfg, tmp_path, *extra):
        return ["distribute_pull.py", "--worker", "host:/tmp/x", "--config", str(cfg),
                "--output", str(tmp_path / "out"), "--skip-gate-check", *extra]

    def test_skip_version_check_never_calls_verify_workers(self, tmp_path, monkeypatch):
        # A bare `with pytest.raises(Exception): dp.main()` would swallow an AssertionError
        # raised FROM INSIDE verify_workers just as happily as the real "no such host" failure
        # main() goes on to hit -- indistinguishable, so it can't be the signal. Record the call
        # instead and assert on that AFTER the broad exception block, matching this file's own
        # pattern elsewhere (e.g. test_passing_verification_still_reaches_dispatch's `reached`).
        called = {}
        monkeypatch.setattr(dp, "verify_workers", lambda *a, **kw: called.setdefault("hit", True))
        monkeypatch.setattr(dp, "Worker", lambda *a, **kw: object())
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path, "--skip-version-check"))
        with pytest.raises(Exception):
            dp.main()
        assert "hit" not in called

    def test_default_calls_verify_workers_and_a_mismatch_aborts_before_dispatch(self, tmp_path, monkeypatch):
        # verify_workers excludes everyone -- main() must stop at ap.error() (SystemExit) and
        # never reach the queue/thread-dispatch code that follows.
        monkeypatch.setattr(dp, "verify_workers", lambda workers, backend: [])
        monkeypatch.setattr(dp, "Worker", lambda *a, **kw: object())   # avoid a real ssh core-probe
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path))
        with pytest.raises(SystemExit) as exc:
            dp.main()
        assert exc.value.code == 2

    def test_passing_verification_still_reaches_dispatch(self, tmp_path, monkeypatch):
        reached = {}
        def fake_verify(workers, backend):
            reached["backend"] = backend
            return workers
        monkeypatch.setattr(dp, "verify_workers", fake_verify)
        monkeypatch.setattr(dp, "Worker", lambda *a, **kw: object())
        monkeypatch.setattr("sys.argv", self._argv(self._cfg(tmp_path), tmp_path))
        with pytest.raises(Exception):   # goes on to fail elsewhere (no real host) -- fine
            dp.main()
        assert reached["backend"] == "livespice"


# ---------------------------------------------------------------------------
# Per-item sharding (per-item-sharding-proposal.md Phases 1-3,
# docs/implementation-roadmap.md item 6)
# ---------------------------------------------------------------------------
class TestParseRangeAxes:
    def test_space_form(self):
        assert dp._parse_range_axes(["--range", "Gain=0.1,0.5,1.0"]) == [("Gain", 3)]

    def test_equals_form(self):
        assert dp._parse_range_axes(["--range=Gain=0.1,0.5,1.0"]) == [("Gain", 3)]

    def test_multiple_ranges(self):
        axes = dp._parse_range_axes(["--range", "Gain=0.1,0.5,1.0", "--range", "Tone=0.2,0.8"])
        assert axes == [("Gain", 3), ("Tone", 2)]

    def test_single_value_range_is_cardinality_one_not_dropped(self):
        assert dp._parse_range_axes(["--range", "Fixed=0.5"]) == [("Fixed", 1)]

    def test_non_range_args_ignored(self):
        assert dp._parse_range_axes(["--backend", "livespice", "--oversample", "8"]) == []

    def test_malformed_range_with_no_equals_is_skipped(self):
        assert dp._parse_range_axes(["--range", "garbage"]) == []

    def test_dangling_range_flag_with_no_value_is_ignored(self):
        assert dp._parse_range_axes(["--range"]) == []

    def test_knob_names_with_spaces_survive(self):
        assert dp._parse_range_axes(["--range", "RD Gain=0.1,1.0"]) == [("RD Gain", 2)]

    def test_repeated_knob_overrides_rather_than_multiplies(self):
        """A config's own --range Gain=... followed by an extra --range Gain=... (an override
        passed after `--`, e.g. `fleet_ctl.py submit --config ... -- --range Gain=0.5,0.9`)
        must behave like gen_dataset_from_schx.py's own last-one-wins parsing, not add a
        second Gain axis -- multiplying gave a chunk count that didn't match the real grid."""
        axes = dp._parse_range_axes(["--range", "Gain=0.1,0.15,0.25,0.5,0.75,0.9,1.0",
                                     "--range", "Gain=0.5,0.9"])
        assert axes == [("Gain", 2)]

    def test_repeated_knob_keeps_its_original_position(self):
        axes = dp._parse_range_axes(["--range", "Gain=0.1,0.5,1.0", "--range", "Tone=0.2,0.8",
                                     "--range", "Gain=0.5,0.9"])
        assert axes == [("Gain", 2), ("Tone", 2)]


class TestDeriveItemCount:
    def test_single_axis(self):
        assert dp.derive_item_count(["--range", "Gain=0.1,0.5,1.0"]) == 3

    def test_multiple_axes_multiply(self):
        gen_args = ["--range", "Gain=0.1,0.5,1.0", "--range", "Tone=0.2,0.8"]
        assert dp.derive_item_count(gen_args) == 6

    def test_items_override_wins_over_range(self):
        gen_args = ["--range", "Gain=0.1,0.5,1.0"]
        assert dp.derive_item_count(gen_args, items_override=99) == 99

    def test_items_override_used_when_no_range_present(self):
        assert dp.derive_item_count(["--backend", "cpp"], items_override=10) == 10

    def test_no_range_and_no_override_raises_loudly_rather_than_guessing(self):
        with pytest.raises(ValueError, match="could not derive"):
            dp.derive_item_count(["--backend", "livespice"])

    def test_zero_items_override_raises(self):
        with pytest.raises(ValueError, match="positive"):
            dp.derive_item_count([], items_override=0)

    def test_negative_items_override_raises(self):
        with pytest.raises(ValueError, match="positive"):
            dp.derive_item_count([], items_override=-5)

    def test_single_value_ranges_do_not_change_the_product(self):
        gen_args = ["--range", "Gain=0.1,0.5,1.0", "--range", "Fixed=0.5"]
        assert dp.derive_item_count(gen_args) == 3

    def test_overriding_a_configs_range_shrinks_the_count_not_multiplies_it(self):
        gen_args = ["--range", "Fuzz=0.1,0.15,0.25,0.5,0.75,0.95,1.0", "--range", "Fuzz=0.5,0.9"]
        assert dp.derive_item_count(gen_args) == 2


class TestWarnChunkAliasingStillWorksAfterRefactor:
    """_warn_chunk_aliasing now delegates to _parse_range_axes (shared with
    derive_item_count) -- confirms the refactor preserved its n_vals>=2 filtering
    (a fixed/single-value knob can never alias, so it must stay excluded here even
    though derive_item_count needs it counted)."""

    def test_single_value_range_cannot_alias(self, monkeypatch):
        logged = []
        monkeypatch.setattr(dp, "log", logged.append)
        dp._warn_chunk_aliasing(["--range", "Fixed=0.5"], chunks=8)
        assert logged == []

    def test_still_warns_on_a_real_aliasing_axis(self, monkeypatch):
        logged = []
        monkeypatch.setattr(dp, "log", logged.append)
        dp._warn_chunk_aliasing(["--range", "Volume=0.1,0.2,0.3,0.4"], chunks=32)
        assert any("aliasing" in m for m in logged)


class TestCollectLabelsAndExpectedCount:
    """_collect with `labels`/`expected_count` (Phase 3). Fakes rsync with a real local file
    copy (dp.subprocess.run patched) instead of mocking file contents away entirely, so the
    actual merge/consistency logic in _collect runs for real against real files -- only the
    network transport is faked."""

    class FakeW:
        def __init__(self, host):
            self.host = host

    def _fake_rsync(self, monkeypatch):
        import shutil
        def run(cmd, **kw):
            # cmd[2] is "host:/local/path" or "host:/local/path/" (rsync host:path syntax);
            # the tests below always use a REAL local path after the colon, so this "fakes"
            # only the network hop, not the filesystem semantics.
            src_spec, dst = cmd[2], cmd[3]
            _, _, src = src_spec.partition(":")
            src_path = Path(src)
            if not src_path.exists():
                return subprocess.CompletedProcess(cmd, 1, "", "rsync: no such file")
            if src.endswith("/"):
                shutil.copytree(src_path, dst, dirs_exist_ok=True)
            else:
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(dp.subprocess, "run", run)

    def _make_shard(self, base, idx, gain, npy_bytes=b"X" * 100):
        d = base / f"shard{idx}"
        (d / "sig").mkdir(parents=True)
        (d / "sig" / f"{idx}.npy").write_bytes(npy_bytes)
        with open(d / "params.csv", "w", newline="") as f:
            f.write(f"idx,Gain\n{idx},{gain}\n")
        return d

    def test_default_labels_are_host_backward_compatible(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        s0 = self._make_shard(tmp_path, 0, 0.1)
        local = tmp_path / "merged"
        ok = dp._collect([self.FakeW("h1")], [str(s0)], local)
        assert ok is True
        assert (local / "params.csv").read_text() == "idx,Gain\n0,0.1\n"

    def test_distinct_labels_prevent_same_host_slots_from_clobbering(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        s0 = self._make_shard(tmp_path, 0, 0.1)
        s1 = self._make_shard(tmp_path, 1, 0.9)
        w = self.FakeW("localhost")   # SAME worker/host for both slots, on purpose
        local = tmp_path / "merged"
        ok = dp._collect([w, w], [str(s0), str(s1)], local,
                         labels=["localhost-slot0", "localhost-slot1"])
        assert ok is True
        rows = sorted(local.glob("sig/*.npy"))
        assert [p.stem for p in rows] == ["0", "1"]
        assert "0,0.1" in (local / "params.csv").read_text()
        assert "1,0.9" in (local / "params.csv").read_text()

    def test_expected_count_matching_stays_consistent(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        s0 = self._make_shard(tmp_path, 0, 0.1)
        local = tmp_path / "merged"
        assert dp._collect([self.FakeW("h1")], [str(s0)], local, expected_count=1) is True

    def test_expected_count_short_is_inconsistent(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        s0 = self._make_shard(tmp_path, 0, 0.1)
        local = tmp_path / "merged"
        # only 1 of 4 combinations ever rendered -- orphan-free (nothing MISFILED), but
        # incomplete, which is exactly what expected_count catches and 1:1 checking alone can't.
        assert dp._collect([self.FakeW("h1")], [str(s0)], local, expected_count=4) is False

    def test_expected_count_none_skips_the_check_entirely(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        s0 = self._make_shard(tmp_path, 0, 0.1)
        local = tmp_path / "merged"
        assert dp._collect([self.FakeW("h1")], [str(s0)], local, expected_count=None) is True

    def test_mismatched_npy_sizes_are_flagged(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        s0 = self._make_shard(tmp_path, 0, 0.1, npy_bytes=b"X" * 100)
        s1 = self._make_shard(tmp_path, 1, 0.9, npy_bytes=b"X" * 50)   # different size -- truncated?
        w = self.FakeW("localhost")
        local = tmp_path / "merged"
        ok = dp._collect([w, w], [str(s0), str(s1)], local,
                         labels=["localhost-slot0", "localhost-slot1"], expected_count=2)
        assert ok is False

    def test_uniform_npy_sizes_do_not_trip_the_size_check(self, tmp_path, monkeypatch):
        self._fake_rsync(monkeypatch)
        s0 = self._make_shard(tmp_path, 0, 0.1, npy_bytes=b"X" * 100)
        s1 = self._make_shard(tmp_path, 1, 0.9, npy_bytes=b"Y" * 100)
        w = self.FakeW("localhost")
        local = tmp_path / "merged"
        ok = dp._collect([w, w], [str(s0), str(s1)], local,
                         labels=["localhost-slot0", "localhost-slot1"], expected_count=2)
        assert ok is True


class TestPerItemCliWiring:
    """The dispatch-mode branch in main(): thread args are captured by patching
    threading.Thread itself (target/args recorded, start/join are no-ops) so this tests the
    EXACT wiring (output dirs, --workers N, thread count) without any real ssh or a full mocked
    dispatch cycle -- the real over-the-wire mechanism this wiring drives is verified
    separately (see the module's own real end-to-end smoke test, not part of this suite)."""

    class FakeThread:
        calls = []
        def __init__(self, target=None, args=(), daemon=None):
            FakeThread.calls.append((target, args))
        def start(self): pass
        def join(self): pass

    class FakeWorker:
        _next_host = 0
        def __init__(self, spec, job=None):
            parts = spec.split(":")
            self.host, self.dir = parts[0], parts[1]
            self.parallel = int(parts[2]) if len(parts) > 2 and parts[2] else 4
            self.job = job
            self.done = self.failed = 0
            self.secs = 0.0
            self.rate = 0.0   # real Worker.rate is a property; a plain 0.0 is enough here --
                              # nothing in these tests exercises real dispatch, only the wiring
                              # up to thread-spawn, but main()'s final report reads it either way

    def _cfg(self, tmp_path, ranges="Gain = [0.1, 0.5, 1.0]\nTone = [0.2, 0.8]"):
        (tmp_path / "amps").mkdir(exist_ok=True)
        schx = tmp_path / "amps" / "Amp.schx"
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
{ranges}
''', encoding="utf-8")
        return p

    def _run(self, monkeypatch, tmp_path, *extra, worker="host:/repo:4"):
        FakeThread = type("FakeThread", (), {"calls": []})
        def make_thread(target=None, args=(), daemon=None):
            FakeThread.calls.append((target, args))
            class T:
                def start(self_s): pass
                def join(self_s): pass
            return T()
        monkeypatch.setattr(dp.threading, "Thread", make_thread)
        monkeypatch.setattr(dp, "Worker", self.FakeWorker)
        monkeypatch.setattr(dp, "verify_workers", lambda workers, backend: workers)
        ssh_calls = []
        def fake_run(cmd, **kw):
            ssh_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(dp.subprocess, "run", fake_run)
        argv = ["distribute_pull.py", "--worker", worker, "--config", str(self._cfg(tmp_path)),
                "--output", "/out", "--skip-gate-check", *extra]
        monkeypatch.setattr("sys.argv", argv)
        dp.main()
        return FakeThread.calls, ssh_calls

    def test_legacy_mode_is_unchanged_one_thread_per_worker(self, tmp_path, monkeypatch):
        calls, _ = self._run(monkeypatch, tmp_path, "--chunks", "6")
        assert len(calls) == 1
        target, args = calls[0]
        w, output_dir, workers_flag = args
        assert output_dir == "/out" and workers_flag == 4   # worker's own .parallel

    def test_per_item_mode_spawns_slots_worth_of_threads(self, tmp_path, monkeypatch):
        calls, _ = self._run(monkeypatch, tmp_path, "--chunk-size", "1", worker="host:/repo:3")
        assert len(calls) == 3   # defaults to w.parallel when --slots is not given
        for _, (w, output_dir, workers_flag) in calls:
            assert workers_flag == 1   # always 1 in per-item mode
        dirs = sorted(output_dir for _, (w, output_dir, workers_flag) in calls)
        assert dirs == ["/out/slot-0", "/out/slot-1", "/out/slot-2"]

    def test_slots_flag_overrides_worker_parallel(self, tmp_path, monkeypatch):
        calls, _ = self._run(monkeypatch, tmp_path, "--chunk-size", "1", "--slots", "2",
                          worker="host:/repo:8")
        assert len(calls) == 2   # NOT 8 -- --slots overrides the worker's own parallel

    def test_chunk_size_other_than_one_is_a_hard_error(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, tmp_path, "--chunk-size", "5")

    def test_slot_dirs_are_mkdir_ped_before_dispatch(self, tmp_path, monkeypatch):
        # Real bug this guards against, found by an actual end-to-end render (not a mock):
        # gen_dataset_from_schx.py's own disk-space check only falls back ONE level when
        # --output is missing (shutil.disk_usage(args.output.parent if ... else args.output)),
        # so a brand-new --output path (the common case for a device's first-ever render)
        # leaves BOTH <output>/slot-K and its parent <output> missing, and disk_usage() raised
        # FileNotFoundError outright. Every slot dir must be created before ANY thread starts.
        calls, ssh_calls = self._run(monkeypatch, tmp_path, "--chunk-size", "1",
                                     worker="host:/repo:3")
        mkdirs = [c for c in ssh_calls if "mkdir" in " ".join(str(x) for x in c)]
        assert len(mkdirs) == 3
        made = sorted(" ".join(str(x) for x in c) for c in mkdirs)
        assert any("mkdir -p /out/slot-0" in c for c in made)
        assert any("mkdir -p /out/slot-1" in c for c in made)
        assert any("mkdir -p /out/slot-2" in c for c in made)

    def test_legacy_mode_mkdirs_output_but_never_a_slot_dir(self, tmp_path, monkeypatch):
        # Same gap as per-item's own mkdir, one level shallower: a brand-new --output with no
        # PARENT either leaves shutil.disk_usage()'s one-level fallback missing too. Legacy
        # mode gets ONE mkdir per worker for the plain --output -- never a "slot-N" path,
        # which is per-item-only.
        calls, ssh_calls = self._run(monkeypatch, tmp_path, "--chunks", "6")
        mkdirs = [c for c in ssh_calls if "mkdir" in " ".join(str(x) for x in c)]
        assert len(mkdirs) == 1
        assert "mkdir -p /out" in " ".join(str(x) for x in mkdirs[0])
        assert not any("slot-" in str(x) for x in mkdirs[0])

    def test_items_override_is_used_instead_of_deriving_from_range(self, tmp_path, monkeypatch):
        # Without --items this config derives 3*2=6 -- the real assertion is that main() does
        # NOT error out deriving that, i.e. --items successfully overrode it (derive_item_count
        # itself, and its --items-wins-over-range behavior, are covered directly in
        # TestDeriveItemCount; thread count alone can't distinguish "derived 6" from
        # "overridden to 10" since both give 1 thread for 1 slot).
        calls, _ = self._run(monkeypatch, tmp_path, "--chunk-size", "1", "--items", "10",
                          worker="host:/repo:1")
        assert len(calls) == 1

    def test_no_range_and_no_items_is_a_clear_error_not_a_crash(self, tmp_path, monkeypatch):
        # No --config at all (so no --range is ever built) and no --items: derive_item_count
        # has nothing to derive from and must raise -> main() must turn that into ap.error
        # (SystemExit), not let the ValueError propagate raw.
        monkeypatch.setattr(dp.threading, "Thread", lambda **kw: None)
        monkeypatch.setattr(dp, "Worker", self.FakeWorker)
        monkeypatch.setattr(dp, "verify_workers", lambda workers, backend: workers)
        argv = ["distribute_pull.py", "--worker", "host:/repo:1", "--output", "/out",
                "--skip-gate-check", "--chunk-size", "1", "--", "--backend", "livespice"]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            dp.main()
