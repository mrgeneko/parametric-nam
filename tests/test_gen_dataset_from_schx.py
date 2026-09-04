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


def test_check_backend_noops_when_registry_file_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("PARAMETRIC_DEVICES", str(tmp_path / "does-not-exist"))
    ap = _FakeAp()
    g.check_backend(tmp_path / "Some Circuit.schx", "livespice", ap)
    assert ap.errors == []


def test_check_backend_noops_when_registry_is_malformed_toml(tmp_path, monkeypatch):
    reg = tmp_path / "devices.toml"
    reg.write_text("this is not [valid toml")
    monkeypatch.setenv("PARAMETRIC_DEVICES", str(tmp_path))
    ap = _FakeAp()
    g.check_backend(tmp_path / "Some Circuit.schx", "livespice", ap)
    assert ap.errors == []


def test_check_backend_noops_when_device_not_in_registry(tmp_path, monkeypatch):
    reg = tmp_path / "devices.toml"
    reg.write_text('[some-device]\nname = "Some Device"\nschx = "pedals/Some Device.schx"\n')
    monkeypatch.setenv("PARAMETRIC_DEVICES", str(tmp_path))
    ap = _FakeAp()
    g.check_backend(tmp_path / "A Totally Different Circuit.schx", "livespice", ap)
    assert ap.errors == []


def test_check_backend_noops_when_backend_has_no_entry_for_this_device(tmp_path, monkeypatch):
    reg = tmp_path / "devices.toml"
    reg.write_text(
        '[some-device]\n'
        'name = "Some Device"\n'
        'schx = "pedals/Some Device.schx"\n'
    )
    monkeypatch.setenv("PARAMETRIC_DEVICES", str(tmp_path))
    ap = _FakeAp()
    # no [some-device.backend] table at all -> absence of evidence, not a claim
    g.check_backend(tmp_path / "Some Device.schx", "ngspice", ap)
    assert ap.errors == []


def test_check_backend_prints_a_note_and_does_not_error_when_backend_is_valid(tmp_path, monkeypatch, capsys):
    reg = tmp_path / "devices.toml"
    reg.write_text(
        '[some-device]\n'
        'name = "Some Device"\n'
        'schx = "pedals/Some Device.schx"\n'
        '[some-device.backend.ngspice]\n'
        'valid = true\n'
        'reason = "converges cleanly"\n'
    )
    monkeypatch.setenv("PARAMETRIC_DEVICES", str(tmp_path))
    ap = _FakeAp()
    g.check_backend(tmp_path / "Some Device.schx", "ngspice", ap)
    assert ap.errors == []
    assert "converges cleanly" in capsys.readouterr().err


def test_check_backend_errors_when_backend_is_declared_invalid(tmp_path, monkeypatch):
    reg = tmp_path / "devices.toml"
    reg.write_text(
        '[some-device]\n'
        'name = "Some Device"\n'
        'schx = "pedals/Some Device.schx"\n'
        '[some-device.backend.livespice]\n'
        'valid = false\n'
        'reason = "collapses transistor gyrators to a flat response"\n'
        '[some-device.backend.ngspice]\n'
        'valid = true\n'
    )
    monkeypatch.setenv("PARAMETRIC_DEVICES", str(tmp_path))
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        g.check_backend(tmp_path / "Some Device.schx", "livespice", ap)
    assert len(ap.errors) == 1
    msg = ap.errors[0]
    assert "CANNOT be rendered faithfully" in msg
    assert "collapses transistor gyrators" in msg
    assert "ngspice" in msg  # the valid alternative must be surfaced


def test_check_backend_matches_by_schx_basename_not_full_path(tmp_path, monkeypatch):
    # The registry's own schx field is a repo-relative path ("pedals/Some Device.schx"); the
    # match is against the caller's schx basename, so an arbitrary caller-side directory must
    # still resolve correctly.
    reg = tmp_path / "devices.toml"
    reg.write_text(
        '[some-device]\n'
        'name = "Some Device"\n'
        'schx = "pedals/Some Device.schx"\n'
        '[some-device.backend.livespice]\n'
        'valid = false\n'
        'reason = "nope"\n'
    )
    monkeypatch.setenv("PARAMETRIC_DEVICES", str(tmp_path))
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        g.check_backend(tmp_path / "some" / "unrelated" / "dir" / "Some Device.schx",
                        "livespice", ap)
    assert ap.errors


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


# ------------------------------------------------------ backend-validity gate visibility
#
# check_backend used to `return` silently in FOUR situations -- registry missing, registry
# unparseable, device absent, backend unlisted. That is the worst behaviour a safety check
# can have: someone who believes they are protected gets output identical to someone who
# is not. A device that genuinely cannot be rendered on the chosen backend (the
# metal-distortion pedal's transistor gyrators under livespice) would render
# "successfully" into a wrong dataset. Every inactive path now says so.

import argparse
import io
import contextlib
from pathlib import Path


def _gate(tmp_registry, schx="Mine.schx", backend="livespice"):
    err = io.StringIO()
    ap = argparse.ArgumentParser()
    with contextlib.redirect_stderr(err):
        g.check_backend(Path(schx), backend, ap, registry=tmp_registry)
    return err.getvalue()


def test_env_pointing_at_a_missing_registry_warns_loudly_but_does_not_abort(tmp_path):
    # The loudest inactive case -- this user configured the gate, so they believe it ran.
    # WARNING not error: the gate is advisory (an unlisted device is already assumed valid),
    # and someone with PARAMETRIC_DEVICES set in a shell profile on a machine where that repo
    # is not cloned should not be blocked from rendering by an informational check. See
    # test_check_backend_noops_when_registry_file_is_absent, which pins the no-abort contract.
    out = _gate(tmp_path / "nope" / "devices.toml")
    assert "DID NOT RUN" in out


def test_unparseable_registry_says_so(tmp_path):
    reg = tmp_path / "devices.toml"
    reg.write_text("this is not valid toml {{{")
    assert "INACTIVE" in _gate(reg) and "parse" in _gate(reg)


def test_device_absent_from_the_registry_says_so(tmp_path):
    reg = tmp_path / "devices.toml"
    reg.write_text('[some-other]\nname = "Other"\nschx = "amps/Other.schx"\n')
    out = _gate(reg, schx="Mine.schx")
    assert "not in devices.toml" in out


def test_registered_device_with_no_verdict_for_this_backend_says_so(tmp_path):
    reg = tmp_path / "devices.toml"
    reg.write_text('[mine]\nname = "Mine"\nschx = "pedals/Mine.schx"\n')
    assert "no verdict" in _gate(reg, schx="Mine.schx")


def test_an_invalid_backend_still_hard_fails(tmp_path):
    reg = tmp_path / "devices.toml"
    reg.write_text('[mine]\nname = "Mine"\nschx = "pedals/Mine.schx"\n'
                   '[mine.backend.livespice]\nvalid = false\nreason = "gyrators"\n'
                   '[mine.backend.ngspice]\nvalid = true\n')
    with pytest.raises(SystemExit):
        _gate(reg, schx="Mine.schx", backend="livespice")


def test_a_valid_backend_passes_without_noise(tmp_path):
    reg = tmp_path / "devices.toml"
    reg.write_text('[mine]\nname = "Mine"\nschx = "pedals/Mine.schx"\n'
                   '[mine.backend.livespice]\nvalid = true\n')
    assert _gate(reg, schx="Mine.schx") == ""


# ---------------------------------------------------- the third verdict state: "partial"
#
# backends.toml already used three states -- ampeg-svt-power-amp declares
# ngspice = { valid = "partial" } -- but check_backend only ever did a truthiness test, so
# "partial" passed exactly like `true`, printing its own reason as an ordinary note. That
# reason reads "BUT NOT YET USABLE: below ~100 Hz the output is UNPHYSICAL ... 2297 W at
# 40 Hz ... That is ringing, not output." Worse, when the OTHER backend was invalid, the
# refusal message listed the partial one under "Valid backend(s) for this device" -- so the
# gate blocked you and then recommended a backend the registry says is unusable.

_PARTIAL_REG = ('[amp]\nname = "Amp"\nschx = "amps/Amp.schx"\n'
                '[amp.backend.livespice]\nvalid = false\nreason = "diverges above 400 Hz"\n'
                '[amp.backend.ngspice]\nvalid = "partial"\nreason = "NOT YET USABLE below 100 Hz"\n')


def _reg(tmp_path, text):
    p = tmp_path / "devices.toml"
    p.write_text(text)
    return p


def test_partial_warns_but_does_not_block(tmp_path):
    # "partial" means usable WITH KNOWN LIMITS. Whether those limits matter depends on the
    # grid and the excitation, which this gate cannot judge -- so warn and let the operator
    # decide, rather than either blocking or staying quiet.
    out = _gate(_reg(tmp_path, _PARTIAL_REG), schx="Amp.schx", backend="ngspice")
    assert "PARTIAL" in out and "NOT YET USABLE" in out


def test_partial_is_louder_than_an_ordinary_valid_note(tmp_path):
    partial = _gate(_reg(tmp_path, _PARTIAL_REG), schx="Amp.schx", backend="ngspice")
    ok = _gate(_reg(tmp_path, '[amp]\nname = "Amp"\nschx = "amps/Amp.schx"\n'
                              '[amp.backend.ngspice]\nvalid = true\nreason = "fine"\n'),
               schx="Amp.schx", backend="ngspice")
    assert partial.startswith("WARNING") and ok.startswith("note")


def test_a_partial_backend_is_not_recommended_as_valid_when_blocking(tmp_path):
    ap = _FakeAp()
    with pytest.raises(SystemExit):
        g.check_backend(Path("Amp.schx"), "livespice", ap,
                        registry=_reg(tmp_path, _PARTIAL_REG))
    msg = ap.errors[0]
    assert "NONE RECORDED" in msg, "a partial backend must not be listed as plainly valid"
    assert "PARTIAL" in msg, "...but it should still be mentioned, with its status"


@pytest.mark.parametrize("spelling", ["partial", "PARTIAL", " Partial "])
def test_partial_recognised_regardless_of_case_or_whitespace(tmp_path, spelling):
    reg = _reg(tmp_path, f'[amp]\nname = "Amp"\nschx = "amps/Amp.schx"\n'
                         f'[amp.backend.ngspice]\nvalid = "{spelling}"\nreason = "limits"\n')
    assert "PARTIAL" in _gate(reg, schx="Amp.schx", backend="ngspice")
