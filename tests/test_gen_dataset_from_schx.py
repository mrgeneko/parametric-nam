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
