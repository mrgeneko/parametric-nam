"""Tests for gen_dataset_from_schx.py's pure decision logic: the convergence-failure
classifier, the per-backend retry-escalation ladder, the backend-validity gate against an
external devices.toml, and process_one()'s escalate-on-failure orchestration.

None of these invoke a real livespice_cli/ngspice/LTspice binary -- process_one()'s tests
monkeypatch _render_once (the actual subprocess boundary) so the escalation LOGIC is tested
independently of any oracle being installed.

See gen_dataset_from_schx.py.
"""
from types import SimpleNamespace

import pytest

import gen_dataset_from_schx as g


# --------------------------------------------------------------------------- _is_convergence_failure

@pytest.mark.parametrize("err", [
    "Circuit.SimulationDiverged near t=2.65s",
    "output contains NaN",
    "value is Inf",
    "timestep too small",
    "matrix is singular",
    "no convergence after 50 iterations",
    "solver unstable at high gain",
    "crest factor exceeds max",
    "input truncated unexpectedly",
    "timeout after 1200s",
    "too many iterations",
    "single-sample spike detected",
    "Newton overshoot at t=0.4s",
])
def test_convergence_keywords_are_recognized_case_insensitively(err):
    assert g._is_convergence_failure(err)
    assert g._is_convergence_failure(err.upper())


@pytest.mark.parametrize("err", [
    "",
    "unknown backend: foo",
    "--bounds knob 'gain' not in --knobs list",
    "input WAV not found: /tmp/missing.wav",
    "unknown circuit 'foo'. known: bar, baz",
    "permission denied",
])
def test_non_convergence_errors_are_not_retried(err):
    assert not g._is_convergence_failure(err)


def test_empty_or_none_error_is_not_a_convergence_failure():
    assert not g._is_convergence_failure("")
    assert not g._is_convergence_failure(None)


# --------------------------------------------------------------------------- _rung_str

def test_rung_str_skips_falsy_values_and_formats_the_rest():
    assert g._rung_str({"oversample": 8, "iterations": 256}) == "iterations=256 oversample=8"


def test_rung_str_skips_none_empty_and_zero():
    assert g._rung_str({"oversample": 0, "conv": "", "method": None, "koren": 4}) == "koren=4"


def test_rung_str_empty_dict_is_empty_string():
    assert g._rung_str({}) == ""


# --------------------------------------------------------------------------- _rungs: livespice

def test_livespice_rungs_double_oversample_up_to_256_ceiling():
    rungs = g._rungs("livespice", oversample=2, ng=None)
    oversamples = [r["oversample"] for r in rungs]
    assert oversamples == [2, 4, 8, 16, 32, 64, 128, 256]


def test_livespice_rungs_always_carry_256_iterations():
    rungs = g._rungs("livespice", oversample=8, ng=None)
    assert all(r["iterations"] == 256 for r in rungs)


def test_livespice_rungs_default_oversample_when_falsy():
    # oversample=0/None both fall back to the documented default of 2.
    assert g._rungs("livespice", oversample=0, ng=None)[0]["oversample"] == 2
    assert g._rungs("livespice", oversample=None, ng=None)[0]["oversample"] == 2


def test_livespice_rungs_starting_above_ceiling_yields_one_rung():
    rungs = g._rungs("livespice", oversample=256, ng=None)
    assert rungs == [{"oversample": 256, "iterations": 256}]


# --------------------------------------------------------------------------- _rungs: ngspice

def test_ngspice_rungs_never_repeat_a_prior_rung():
    # The exact regression the docstring describes: a caller-supplied base that already has
    # input_upsample/method/diode_cjo set used to make every rung identical, so a failing
    # combination was retried with the SAME settings every time.
    base = {"input_upsample": 4, "method": "gear", "conv": {"diode_cjo": "100p"}}
    rungs = g._rungs("ngspice", oversample=2, ng=base)
    frozen = [_freeze(r) for r in rungs]
    assert len(frozen) == len(set(frozen))


def _freeze(rung):
    """Make an ngspice rung dict (whose 'conv' value is itself a dict) hashable for a
    no-duplicates check."""
    out = dict(rung)
    if "conv" in out:
        out["conv"] = tuple(sorted(out["conv"].items()))
    return tuple(sorted(out.items()))


def test_ngspice_rungs_escalate_from_an_empty_base():
    rungs = g._rungs("ngspice", oversample=2, ng={})
    assert len(rungs) > 1, "an empty base must still escalate through tmax/upsample/method/conv"
    # first rung is the (empty) base, unmodified
    assert rungs[0] == {}
    # by the end, gear + diode_cjo have both been tried
    assert rungs[-1].get("method") == "gear"
    assert rungs[-1].get("conv", {}).get("diode_cjo") == "100p"


def test_ngspice_rungs_respects_an_already_set_conv_dict_without_dropping_keys():
    base = {"conv": {"klu": "1"}}
    rungs = g._rungs("ngspice", oversample=2, ng=base)
    # every rung's conv dict must still carry the caller's klu=1 -- escalation adds to the
    # base, it must not silently drop an unrelated setting the caller already made.
    assert all(r.get("conv", {}).get("klu") == "1" for r in rungs)


def test_ngspice_rungs_none_base_treated_as_empty():
    assert g._rungs("ngspice", oversample=2, ng=None) == g._rungs("ngspice", oversample=2, ng={})


# --------------------------------------------------------------------------- _rungs: cpp / unknown

def test_cpp_rungs_escalate_oversample_only():
    rungs = g._rungs("cpp", oversample=2, ng=None)
    assert rungs == [{"oversample": 2}, {"oversample": 4}, {"oversample": 8}]


def test_cpp_rungs_default_oversample_when_falsy():
    assert g._rungs("cpp", oversample=0, ng=None)[0]["oversample"] == 2


def test_unknown_backend_yields_a_single_empty_rung():
    assert g._rungs("nonexistent-backend", oversample=2, ng=None) == [{}]


# --------------------------------------------------------------------------- check_backend

class _FakeAp:
    def __init__(self):
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)
        raise SystemExit(2)  # mirror argparse's own ArgumentParser.error() behavior


def test_check_backend_announces_when_no_sidecar_exists(tmp_path):
    # Was: "noops when registry file is absent". A gate that says nothing when it is doing
    # nothing is worse than no gate -- see the module comment in check_backend.
    schx = tmp_path / "Some Circuit.schx"
    schx.write_text("<Schematic/>")
    err = io.StringIO()
    ap = _FakeAp()
    with contextlib.redirect_stderr(err):
        g.check_backend(schx, "livespice", ap)
    assert ap.errors == []
    assert "nothing is checking" in err.getvalue().lower()




# --------------------------------------------------------------------------- process_one escalation

def _result(idx, ok=False, error=""):
    return g.Result(idx, ok=ok, error=error)


def test_process_one_returns_immediately_on_first_rung_success(tmp_path, monkeypatch):
    calls = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        calls.append(kw.get("iterations"))
        return _result(idx, ok=True)

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "livespice")
    assert r.ok
    assert r.rung == 0
    assert len(calls) == 1


def test_process_one_escalates_through_convergence_failures_and_records_winning_rung(tmp_path, monkeypatch):
    attempts = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        attempts.append(kw.get("iterations"))
        if len(attempts) < 3:
            return _result(idx, error="Circuit.SimulationDiverged near t=1.0s")
        return _result(idx, ok=True)

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "livespice")
    assert r.ok
    assert r.rung == 2  # 0-indexed: third attempt won
    assert len(attempts) == 3


def test_process_one_does_not_retry_a_non_convergence_error(tmp_path, monkeypatch):
    attempts = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        attempts.append(1)
        return _result(idx, error="unknown circuit 'foo'")

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "livespice")
    assert not r.ok
    assert len(attempts) == 1, "a config error must fail once, not burn the whole rung ladder"


def test_process_one_does_not_escalate_a_timeout_on_livespice(tmp_path, monkeypatch):
    attempts = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        attempts.append(1)
        return _result(idx, error="timeout after 1200s")

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "livespice")
    assert not r.ok
    assert len(attempts) == 1, "livespice/cpp timeouts must not escalate -- higher rungs are strictly slower"
    assert "not escalating" in r.error


def test_process_one_does_not_escalate_a_timeout_on_cpp_either(tmp_path, monkeypatch):
    attempts = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        attempts.append(1)
        return _result(idx, error="timeout after 1200s")

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "cpp")
    assert not r.ok
    assert len(attempts) == 1


def test_process_one_DOES_escalate_a_timeout_on_ngspice(tmp_path, monkeypatch):
    # ngspice is explicitly exempted from the timeout-does-not-escalate rule: its rungs change
    # method/damping at roughly equal solver cost, so a retry there can genuinely win.
    attempts = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        attempts.append(1)
        if len(attempts) < 2:
            return _result(idx, error="timeout after 1200s")
        return _result(idx, ok=True)

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "ngspice")
    assert r.ok
    assert len(attempts) == 2


def test_process_one_exhausts_all_rungs_and_annotates_the_final_error(tmp_path, monkeypatch):
    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        return _result(idx, error="Circuit.SimulationDiverged")

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "cpp")  # cpp: 3 rungs, cheap to exhaust
    assert not r.ok
    assert "exhausted 3 convergence rungs" in r.error


def test_process_one_no_retry_flag_uses_a_single_bare_rung(tmp_path, monkeypatch):
    attempts = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        attempts.append(kw.get("iterations"))
        return _result(idx, error="Circuit.SimulationDiverged")

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "livespice", no_retry=True)
    assert not r.ok
    assert len(attempts) == 1
    assert attempts[0] is None  # the bare {} rung has no "iterations" key


def test_process_one_returns_ok_immediately_if_output_already_exists(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(g, "_render_once", lambda *a, **kw: calls.append(1) or _result(0))
    path = g.sig_path(tmp_path, 0)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"already rendered")
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "livespice")
    assert r.ok
    assert not calls, "an existing .npy must short-circuit before any render is attempted"


def test_process_one_start_rung_resumes_past_previously_failed_rungs(tmp_path, monkeypatch):
    # start_rung: a previous run's winning rung is recorded per-row precisely so a re-render
    # does not pay for the whole failed-rung ladder again.
    attempts = []

    def fake_render_once(idx, params, out_dir, input_wav, backend, *a, **kw):
        attempts.append(kw.get("iterations"))
        return _result(idx, ok=True)

    monkeypatch.setattr(g, "_render_once", fake_render_once)
    r = g.process_one(0, {}, tmp_path, tmp_path / "in.wav", "livespice", start_rung=3)
    assert r.ok
    assert r.rung == 3
    assert len(attempts) == 1  # jumped straight to rung 3, did not retry rungs 0-2


# ------------------------------------------------------- input_provenance build-recipe status
#
# The recipe sidecar rides into dataset_config.json -> the published model dir. It used to be
# embedded when present and SILENTLY omitted otherwise, which made two very different states
# indistinguishable after the fact: a legitimately raw source (no recipe should exist) versus a
# built excitation whose sidecar was left behind when the wav moved. Worse, nothing checked that
# the sidecar described THIS wav -- a stale one attached confidently wrong provenance, and
# check_transient_coverage._transient_peak_from_recipe reads that same sidecar to decide whether
# the coverage gate passes, so a stale --realistic-peak silently mis-gates the run too.

import json
import numpy as np
import soundfile as sf


def _wav(tmp_path, name="x.wav", freq=440.0, n=4800):
    p = tmp_path / name
    sf.write(str(p), (0.3 * np.sin(2 * np.pi * freq * np.arange(n) / 48000)).astype("float32"), 48000)
    return p


def test_provenance_records_absent_status_instead_of_omitting_silently(tmp_path):
    prov = g.input_provenance(_wav(tmp_path))
    assert "build_recipe" not in prov
    assert prov["build_recipe_status"].startswith("absent")
    assert "x.recipe.json" in prov["build_recipe_status"]


def test_provenance_embeds_and_marks_verified_when_sidecar_hash_matches(tmp_path):
    w = _wav(tmp_path)
    real_sha = g.input_provenance(w)["audio_sha1"]
    (tmp_path / "x.recipe.json").write_text(json.dumps(
        {"tool": "build_excitation.py", "args": {"realistic_peak": 9.9},
         "output": {"audio_sha1": real_sha}}))
    prov = g.input_provenance(w)
    assert prov["build_recipe"]["args"]["realistic_peak"] == 9.9
    assert prov["build_recipe_status"].startswith("verified")


def test_provenance_refuses_a_stale_sidecar_rather_than_attaching_wrong_provenance(tmp_path):
    w = _wav(tmp_path)
    (tmp_path / "x.recipe.json").write_text(json.dumps(
        {"tool": "build_excitation.py", "args": {"realistic_peak": 9.9},
         "output": {"audio_sha1": "deadbeef" * 5}}))
    prov = g.input_provenance(w)
    assert "build_recipe" not in prov, "a recipe for a DIFFERENT wav must not be embedded"
    assert prov["build_recipe_status"].startswith("STALE")
    assert prov["build_recipe_error"] == prov["build_recipe_status"]


def test_provenance_embeds_but_flags_a_sidecar_with_no_hash_to_verify_against(tmp_path):
    w = _wav(tmp_path)
    (tmp_path / "x.recipe.json").write_text(json.dumps(
        {"tool": "build_excitation.py", "args": {"realistic_peak": 9.9}, "output": {}}))
    prov = g.input_provenance(w)
    assert prov["build_recipe"]["args"]["realistic_peak"] == 9.9
    assert prov["build_recipe_status"].startswith("embedded")


def test_provenance_records_parse_error_status_for_a_corrupt_sidecar(tmp_path):
    w = _wav(tmp_path)
    (tmp_path / "x.recipe.json").write_text("{not valid json")
    prov = g.input_provenance(w)
    assert "build_recipe" not in prov
    assert prov["build_recipe_status"].startswith("parse_error")
    assert "build_recipe_error" in prov

# ============================================================ backend-validity gate, sidecar
#
# Verdicts live in <stem>.backends.toml beside the .schx, found by deriving the path from the
# schematic itself. The devices.toml fallback was removed 2026-09-04: it reached a verdict
# through two name indirections (slug -> registry entry -> .schx filename) that a
# bring-your-own-circuit user could never satisfy and a rename broke silently, and it needed
# PARAMETRIC_DEVICES set, which nobody would.
#
# The recurring theme in these tests: a safety check that says nothing while doing nothing is
# worse than no check, because someone who believes they are protected gets output identical
# to someone who is not.

import io
import contextlib
import argparse
from pathlib import Path


def _sidecar(tmp_path, toml_text, schx="Mine.schx"):
    p = tmp_path / schx
    p.write_text("<Schematic/>")
    p.with_suffix(".backends.toml").write_text(toml_text)
    return p


def _run(schx, backend="livespice", ap=None):
    err = io.StringIO()
    ap = ap or argparse.ArgumentParser()
    with contextlib.redirect_stderr(err):
        g.check_backend(schx, backend, ap)
    return err.getvalue()


# ---- inactive paths all announce themselves -------------------------------------------------

def test_no_sidecar_says_nothing_is_checking(tmp_path):
    schx = tmp_path / "Unknown.schx"
    schx.write_text("<Schematic/>")
    assert "nothing is checking" in _run(schx).lower()


def test_unparseable_sidecar_says_inactive(tmp_path):
    assert "INACTIVE" in _run(_sidecar(tmp_path, "not [valid toml"))


def test_backend_with_no_entry_is_assumed_valid_but_says_so(tmp_path):
    out = _run(_sidecar(tmp_path, 'ngspice = { valid = true }\n'), backend="livespice")
    assert "no verdict" in out and "absence of evidence" in out


# ---- the three verdict states ---------------------------------------------------------------

def test_valid_passes_with_a_note(tmp_path):
    out = _run(_sidecar(tmp_path, 'livespice = { valid = true, reason = "converges" }\n'))
    assert out.startswith("note") and "converges" in out


def test_invalid_blocks(tmp_path):
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        _run(_sidecar(tmp_path, 'livespice = { valid = false, reason = "diverges" }\n'), ap=ap)
    assert "diverges" in ap.errors[0]


def test_partial_warns_but_proceeds(tmp_path):
    # "partial" means usable WITH KNOWN LIMITS. Which limits matter depends on the grid and
    # the excitation, which this gate cannot judge -- so quote them and let the operator
    # decide. It passed as plain `true` until 2026-09-04 because the check was a truthiness
    # test and "partial" is a truthy string.
    out = _run(_sidecar(tmp_path, 'livespice = { valid = "partial", reason = "unphysical <100Hz" }\n'))
    assert out.startswith("WARNING") and "PARTIAL" in out and "unphysical" in out


@pytest.mark.parametrize("spelling", ["partial", "PARTIAL", " Partial "])
def test_partial_recognised_regardless_of_case_or_whitespace(tmp_path, spelling):
    out = _run(_sidecar(tmp_path, f'livespice = {{ valid = "{spelling}", reason = "x" }}\n'))
    assert "PARTIAL" in out


def test_partial_is_not_recommended_as_valid_when_another_backend_blocks(tmp_path):
    # The real ampeg-svt case: livespice invalid, ngspice partial. The refusal used to list
    # ngspice under "Valid backend(s)", steering the operator onto the backend its own reason
    # describes as returning 2297 W at 40 Hz.
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        _run(_sidecar(tmp_path,
                      'livespice = { valid = false, reason = "diverges above 400 Hz" }\n'
                      'ngspice = { valid = "partial", reason = "NOT YET USABLE below 100 Hz" }\n'),
             ap=ap)
    assert "NONE RECORDED" in ap.errors[0]
    assert "PARTIAL" in ap.errors[0]


# ---- the sidecar's whole point --------------------------------------------------------------

def test_works_with_no_registry_and_a_circuit_nobody_has_seen(tmp_path, monkeypatch):
    monkeypatch.delenv("PARAMETRIC_DEVICES", raising=False)
    monkeypatch.delenv("SPICE_CIRCUITS", raising=False)
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        _run(_sidecar(tmp_path, 'livespice = { valid = false, reason = "my own pedal" }\n',
                      schx="Nobody Has Ever Seen This.schx"), ap=ap)
    assert "my own pedal" in ap.errors[0]


def test_refusal_names_the_sidecar_not_a_private_repo(tmp_path):
    # "fix it in parametric-devices/backends.toml" is useless advice to someone who does not
    # have that repo -- and after the migration they should not need it.
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        _run(_sidecar(tmp_path, 'livespice = { valid = false, reason = "no" }\n'), ap=ap)
    assert "Mine.backends.toml" in ap.errors[0]
    assert "parametric-devices" not in ap.errors[0]


# ---- the sidecar is HAND-WRITTEN, so lint it -------------------------------------------------
#
# Nothing generates these files and nothing can: a verdict records what a SIMULATOR can and
# cannot do, which is not derivable from a .schx. So the realistic failure is a typo, and both
# typos are silent in the worst way -- a misspelt BACKEND key reads as "no verdict, assumed
# valid" (indistinguishable from an uncharacterised device), and a misspelt VALID key makes the
# author's own "this backend is invalid" text print as an ordinary approving note while the
# backend is allowed through.

def test_misspelt_backend_key_is_flagged_as_a_dead_verdict(tmp_path):
    out = _run(_sidecar(tmp_path, 'livespcie = { valid = false, reason = "x" }\n'))
    assert "typo for 'livespice'" in out and "DEAD" in out


def test_misspelt_valid_key_is_flagged(tmp_path):
    # The nastier one: without this the reason prints as an approval.
    out = _run(_sidecar(tmp_path, 'livespice = { vaild = false, reason = "x" }\n'))
    assert "unknown key 'vaild'" in out and "assumed VALID" in out


def test_a_valid_value_that_is_neither_bool_nor_partial_is_flagged(tmp_path):
    out = _run(_sidecar(tmp_path, 'livespice = { valid = "no", reason = "x" }\n'))
    assert "neither true, false nor" in out


def test_a_backend_this_script_cannot_render_is_not_a_typo(tmp_path):
    # ltspice verdicts are recorded for the deck tooling; --backend here is cpp/livespice/
    # ngspice. An unknown key must not be flagged unless it LOOKS like a known one.
    out = _run(_sidecar(tmp_path, 'ltspice = { valid = true, reason = "shipping backend" }\n'))
    assert "WARNING" not in out


def test_a_correct_sidecar_lints_clean(tmp_path):
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        _run(_sidecar(tmp_path, 'livespice = { valid = false, reason = "diverges" }\n'), ap=ap)
    # blocked on the verdict, with no lint noise


# ------------------------------------------ what the oracle said while SUCCEEDING
#
# livespice_cli runs LiveSPICE's own ConsoleLog at MessageType.Warning, and the warnings that
# matter arrive on renders that exit 0:
#     "Failed to find partition initial conditions, simulation may be unstable."
#     "Warning: Unconnected terminal '<node>'"
# _render_once captured stderr and then used it ONLY on a non-zero exit, so every one of these
# was dropped. The first is the solver saying it is not confident in the answer it is handing
# you -- Duke of Tone's three real wiring bugs were each found through that message. The second
# is a schematic defect, and a far worse thing to learn about from a trained model.

_BANNER_STDERR = ("Input:    h1k.wav\n"
                  "Format:   48000 Hz, 1 ch, 32-bit float\n"
                  "Circuit:  Some Pedal  (oversample 4x)\n"
                  "Output:   /tmp/x.wav  (144000 samples)\n")


def test_a_clean_render_reports_no_warnings():
    # The banner is printed on every run; treating it as a warning would make the signal
    # useless by firing on everything.
    assert g._oracle_warnings(_BANNER_STDERR) == ""


def test_the_micro_sign_note_is_not_a_warning():
    # livespice_cli's own "note:" about normalising U+00B5 is informational.
    err = _BANNER_STDERR + "note: normalising U+00B5 MICRO SIGN -> U+03BC GREEK MU\n"
    assert g._oracle_warnings(err) == ""


def test_the_solver_saying_it_may_be_unstable_is_kept():
    err = _BANNER_STDERR + "Failed to find partition initial conditions, simulation may be unstable.\n"
    assert "may be unstable" in g._oracle_warnings(err)


def test_an_unconnected_terminal_is_kept():
    err = _BANNER_STDERR + "Warning: Unconnected terminal 'n_bout'\n"
    assert "Unconnected terminal" in g._oracle_warnings(err)


def test_repeated_warnings_are_collapsed():
    err = _BANNER_STDERR + ("Warning: Unconnected terminal 'n_x'\n" * 5)
    assert g._oracle_warnings(err).count("Unconnected") == 1


def test_a_kept_warning_would_be_classed_as_a_convergence_concern():
    # The existing classifier already matches "unstable" -- it just never saw the text, because
    # nothing on the success path passed it along.
    err = _BANNER_STDERR + "Failed to find partition initial conditions, simulation may be unstable.\n"
    assert g._is_convergence_failure(g._oracle_warnings(err))


def test_params_header_is_defined_once_for_both_writers():
    # Two literals existed: the fresh-file writer and the repair path that prepends a header to
    # a headerless CSV. Adding a column to one would give a repaired file a header narrower
    # than its own rows, and a reader would silently mis-associate every field after it.
    h = g._params_header(["Gain", "Tone"])
    assert h[:3] == ["idx", "Gain", "Tone"]
    assert h[-1] == "warnings"
