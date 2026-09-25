"""measure_truncation.py --checkpoint: resume after an interruption.

A full-amp truncation measurement runs for hours (an SVT Full run was 4/11 settings in after
2.5 h), so a kill or crash must cost at most the one setting in flight -- and, just as
important, a resume must never quietly pool settings measured under a different circuit,
input or setting grid. That second failure is real: the first SVT Full measurement was
contaminated when its .schx was regenerated mid-run. Every test here guards one way a resumable
measurement goes wrong silently.

The renderer is faked (_render_batch) so these run in milliseconds: candidate renders are the
reference plus a deterministic error that shrinks with oversample, so the ESR terms are real
numbers and a resumed run can be compared for EXACT equality with an uninterrupted one.
"""
import json
import os
import re
import sys
import zlib

import numpy as np
import pytest

import measure_truncation as mt

SR = 48000
CANDS = (2, 4, 8)
REF = 32
KNOBS = ["Gain", "Tone"]          # probe_settings -> 7 settings
N_SETTINGS = len(mt.probe_settings(KNOBS))


class Interrupted(Exception):
    """Stands in for a kill/crash: raised out of the fake renderer, not caught by measure()."""


class FakeRenderer:
    """Deterministic stand-in for _render_batch. Records every call so a test can assert
    exactly which settings were (re)rendered."""

    def __init__(self, schx_path=None):
        self.calls = []                 # one entry per _render_batch call: list of triples
        self.interrupt_on_call = None   # 1-based call number that raises Interrupted
        self.fail_call = None           # 1-based call number that returns a None render
        self.touch_schx_on_call = None  # 1-based call number that edits the .schx mid-render
        self.schx_path = schx_path

    def __call__(self, schx, clips, iterations, speaker, td, workers, triples):
        triples = list(triples)
        self.calls.append(triples)
        n = len(self.calls)
        if self.interrupt_on_call == n:
            raise Interrupted()
        if self.touch_schx_on_call == n:
            with open(self.schx_path, "ab") as f:
                f.write(b"<!-- edited mid-measurement -->")
        out = {}
        for p, os_, wi in triples:
            k = (tuple(sorted(p.items())), os_, wi)
            seed = zlib.crc32(repr(k[0]).encode())     # NOT hash(): str hashing is randomised per process
            t = np.arange(SR) / SR
            gain = 1.0 + sum(p.values())
            ref = gain * np.sin(2 * np.pi * (200 + 50 * wi) * t)
            if os_ >= REF:
                out[k] = ref
            else:
                noise = np.random.default_rng(seed + os_).standard_normal(SR)
                out[k] = ref + noise * 0.05 * gain / os_ ** 2
        if self.fail_call == n:
            first = next(iter(out))
            out[first] = None
        return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    schx = tmp_path / "amp.schx"
    schx.write_text("<Schematic>original circuit</Schematic>")
    fake = FakeRenderer(schx)
    monkeypatch.setattr(mt, "_render_batch", fake)
    return {"schx": schx, "fake": fake, "td": tmp_path, "ckpt": str(tmp_path / "run.ckpt.json")}


def fingerprint(**over):
    fp = {"input_sha256": "aaa", "knobs": KNOBS, "ref_os": REF, "candidates": list(CANDS),
          "iterations": 256, "speaker": None, "probe_s": 10.0, "n_windows": 4,
          "lead_silence_s": None, "solver": "livespice:test+sub"}
    fp.update(over)
    return fp


def run(env, checkpoint=True, fp=None, **kw):
    return mt.measure(env["schx"], KNOBS, [env["td"] / "w0.wav", env["td"] / "w1.wav"], 0, SR,
                      REF, CANDS, 256, None, env["td"], 2,
                      checkpoint=env["ckpt"] if checkpoint else None,
                      fingerprint=(fp or fingerprint()) if checkpoint else None, **kw)


def saved(env):
    return json.loads(open(env["ckpt"]).read())


def settings_rendered(fake):
    """Distinct settings across the MEASUREMENT renders. The reference-convergence check is one
    extra call at (ref_os, 2*ref_os) on the worst setting -- which may be a setting that was
    already measured, so it must not be counted as one being (re)measured."""
    seen = []
    for call in fake.calls:
        if {os_ for _, os_, _ in call} == {REF, 2 * REF}:
            continue
        for p, _, _ in call:
            if p not in seen:
                seen.append(p)
    return seen


# --------------------------------------------------------------------------- happy path

def test_checkpointed_run_matches_an_uncheckpointed_run_exactly(env):
    """The checkpoint must be pure bookkeeping: same numbers with or without it."""
    plain = run(env, checkpoint=False)
    env["fake"].calls.clear()
    ck = run(env)
    assert ck == plain


def test_every_completed_setting_is_persisted(env):
    run(env)
    d = saved(env)
    assert d["version"] == mt.CHECKPOINT_VERSION
    assert [r["index"] for r in d["rows"]] == list(range(N_SETTINGS))
    assert d["fingerprint"]["schx_sha256"] == mt.file_sha256(env["schx"])
    assert d["fingerprint"]["setting_total"] == N_SETTINGS


# --------------------------------------------------------------------------- resume

def test_interrupted_run_resumes_and_renders_only_the_remainder(env):
    reference = run(env, checkpoint=False)
    env["fake"].calls.clear()

    env["fake"].interrupt_on_call = 4           # dies during the 4th setting
    with pytest.raises(Interrupted):
        run(env)
    assert [r["index"] for r in saved(env)["rows"]] == [0, 1, 2]   # 3 done, in-flight one lost

    env["fake"].calls.clear()
    env["fake"].interrupt_on_call = None
    resumed = run(env)
    rendered = settings_rendered(env["fake"])
    assert len(rendered) == N_SETTINGS - 3       # only the unfinished settings were rendered
    assert resumed == reference                  # and the result is EXACTLY the uninterrupted one


def test_resume_prints_a_heartbeat_the_distributor_recognises(env, capsys):
    """distribute_pull.py decides a shard is alive from lines matching MEASURE_TRUNC_LINE;
    a resumed setting must still advance that count."""
    env["fake"].interrupt_on_call = 3
    with pytest.raises(Interrupted):
        run(env)
    env["fake"].interrupt_on_call = None
    capsys.readouterr()
    run(env)
    err = capsys.readouterr().err
    pat = re.compile(r"^\s*\d+/\d+\s+settings\s+done")     # == distribute_pull.MEASURE_TRUNC_LINE
    beats = [ln for ln in err.splitlines() if pat.match(ln)]
    assert len(beats) == N_SETTINGS
    assert sum("from checkpoint" in b for b in beats) == 2
    assert "resuming from" in err


def test_fully_measured_checkpoint_rerenders_nothing_not_even_the_reference_check(env):
    first = run(env)
    assert len(env["fake"].calls) == N_SETTINGS + 1        # every setting + the ref-error render
    env["fake"].calls.clear()
    again = run(env)
    assert env["fake"].calls == []
    assert again == first


def test_a_stale_reference_cache_is_recomputed_not_trusted(env):
    """The cached ref-error is only valid for the worst setting it was computed at."""
    run(env)
    d = saved(env)
    d["ref_error"]["at"] = {"Gain": 123.0, "Tone": 123.0}
    open(env["ckpt"], "w").write(json.dumps(d))
    env["fake"].calls.clear()
    run(env)
    assert len(env["fake"].calls) == 1                     # just the ref-error render


# --------------------------------------------------------------------------- refusals

def test_resume_refuses_a_changed_schx(env):
    run(env)
    env["schx"].write_text("<Schematic>C2 moved to ground</Schematic>")
    with pytest.raises(SystemExit) as e:
        run(env)
    assert "schx_sha256" in str(e.value)
    assert "DIFFERENT configuration" in str(e.value)


@pytest.mark.parametrize("field,val", [
    ("input_sha256", "bbb"), ("ref_os", 64), ("candidates", [2, 4]), ("iterations", 64),
    ("speaker", "V30"), ("probe_s", 20.0), ("n_windows", 8), ("lead_silence_s", 2.0),
    ("solver", "livespice:other+sub"), ("knobs", ["Gain", "Tone", "Extra"]),
])
def test_resume_refuses_any_changed_measurement_parameter(env, field, val):
    run(env)
    with pytest.raises(SystemExit) as e:
        run(env, fp=fingerprint(**{field: val}))
    assert field in str(e.value)


def test_refusal_leaves_the_checkpoint_untouched(env):
    run(env)
    before = open(env["ckpt"]).read()
    with pytest.raises(SystemExit):
        run(env, fp=fingerprint(iterations=1))
    assert open(env["ckpt"]).read() == before


def test_corrupt_checkpoint_is_refused_not_overwritten(env):
    open(env["ckpt"], "w").write('{"version": 1, "rows": [')      # truncated JSON
    with pytest.raises(SystemExit) as e:
        run(env)
    assert "unreadable" in str(e.value)
    assert open(env["ckpt"]).read() == '{"version": 1, "rows": ['   # hours of work not clobbered


def test_unknown_checkpoint_version_is_refused(env):
    run(env)
    d = saved(env)
    d["version"] = 99
    open(env["ckpt"], "w").write(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        run(env)
    assert "version" in str(e.value)


def test_changed_setting_grid_at_a_checkpointed_index_is_refused(env):
    run(env)
    d = saved(env)
    d["rows"][2]["params"] = {"Gain": 0.123, "Tone": 0.5}
    open(env["ckpt"], "w").write(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        run(env)
    assert "setting grid changed" in str(e.value)


def test_checkpoint_without_fingerprint_is_a_programming_error(env):
    with pytest.raises(ValueError):
        mt.measure(env["schx"], KNOBS, [env["td"] / "w0.wav"], 0, SR, REF, CANDS, 256, None,
                   env["td"], 2, checkpoint=env["ckpt"])


# --------------------------------------------------------------------------- circuit edited mid-run

def test_schx_edited_mid_measurement_aborts_and_does_not_persist_the_tainted_setting(env):
    """The exact contamination that hit the first SVT Full run."""
    env["fake"].touch_schx_on_call = 3
    with pytest.raises(RuntimeError, match="changed on disk"):
        run(env)
    assert [r["index"] for r in saved(env)["rows"]] == [0, 1]      # setting 2 (the tainted one) is not
    # and the checkpoint now describes the OLD circuit, so a resume against the new one is refused
    env["fake"].touch_schx_on_call = None
    with pytest.raises(SystemExit):
        run(env)


def test_schx_edited_mid_measurement_aborts_even_without_a_checkpoint(env):
    env["fake"].touch_schx_on_call = 2
    with pytest.raises(RuntimeError, match="changed on disk"):
        run(env, checkpoint=False)


# --------------------------------------------------------------------------- failed renders

def test_a_setting_with_a_failed_render_is_not_checkpointed_and_is_remeasured(env):
    reference = run(env, checkpoint=False)
    env["fake"].calls.clear()

    env["fake"].fail_call = 2                       # one render of setting 1 comes back None
    run(env)
    assert [r["index"] for r in saved(env)["rows"]] == [i for i in range(N_SETTINGS) if i != 1]

    env["fake"].calls.clear()
    env["fake"].fail_call = None
    healed = run(env)
    assert len(settings_rendered(env["fake"])) == 1  # only the failed one is redone
    assert healed == reference


# --------------------------------------------------------------------------- atomic writes

def test_a_failed_write_leaves_the_previous_checkpoint_intact(env, monkeypatch):
    run(env)
    before = open(env["ckpt"]).read()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(mt._os, "replace", boom)
    with pytest.raises(OSError):
        mt.write_checkpoint(env["ckpt"], fingerprint(), [])
    assert open(env["ckpt"]).read() == before
    json.loads(before)                                # and it is still valid JSON


def test_no_temp_files_are_left_after_a_normal_run(env):
    run(env)
    leftovers = [p.name for p in env["td"].iterdir() if ".tmp" in p.name]
    assert leftovers == []


# --------------------------------------------------------------------------- sharded runs

def test_sharded_run_resumes_with_global_indices_and_emits_identically(env, monkeypatch):
    import prepare_excitation
    monkeypatch.setattr(prepare_excitation, "solver_identity", lambda _b: "livespice:test+sub")
    emit_a = str(env["td"] / "a.json")
    emit_b = str(env["td"] / "b.json")

    run(env, shard="0-0/2", emit=emit_a)                   # uninterrupted reference
    os.remove(env["ckpt"])
    env["fake"].calls.clear()

    env["fake"].interrupt_on_call = 3
    with pytest.raises(Interrupted):
        run(env, shard="0-0/2", emit=emit_b)
    idx = [r["index"] for r in saved(env)["rows"]]
    assert idx == [0, 2]                                   # GLOBAL indices, striped by modulo 2

    env["fake"].interrupt_on_call = None
    run(env, shard="0-0/2", emit=emit_b)
    assert json.loads(open(emit_a).read())["rows"] == json.loads(open(emit_b).read())["rows"]


def test_resuming_with_a_different_shard_spec_is_refused(env):
    env["fake"].interrupt_on_call = 2
    with pytest.raises(Interrupted):
        run(env, shard="0-0/2", emit=str(env["td"] / "x.json"))
    env["fake"].interrupt_on_call = None
    with pytest.raises(SystemExit) as e:
        run(env, shard="1-1/2", emit=str(env["td"] / "y.json"))
    assert "shard" in str(e.value)


# --------------------------------------------------------------------------- fingerprint + CLI

def test_run_fingerprint_tracks_the_input_content_not_its_name(tmp_path, monkeypatch):
    import prepare_excitation
    monkeypatch.setattr(prepare_excitation, "solver_identity", lambda _b: "livespice:t+s")
    wav = tmp_path / "sweep.wav"
    wav.write_bytes(b"one")
    args = (KNOBS, REF, CANDS, 256, None, 10.0, 4, None)
    a = mt.run_fingerprint(wav, *args)
    wav.write_bytes(b"two")                                # same path, new content
    b = mt.run_fingerprint(wav, *args)
    assert a["input_sha256"] != b["input_sha256"]
    assert a["solver"] == "livespice:t+s"


def test_file_sha256_streams_large_files(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(3 * (1 << 20) + 17))         # spans several read chunks
    import hashlib
    assert mt.file_sha256(big) == hashlib.sha256(big.read_bytes()).hexdigest()


def test_checkpoint_flag_is_rejected_with_merge(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["measure_truncation.py", "--input", "x.wav",
                                      "--config", "c.toml", "--merge", "a.json",
                                      "--checkpoint", str(tmp_path / "c.json")])
    with pytest.raises(SystemExit) as e:
        mt.main()
    assert e.value.code == 2
