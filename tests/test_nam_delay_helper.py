"""Properties tools/nam_delay_helper.py must have. It runs as a subprocess under a SIBLING
venv's interpreter specifically to dodge a numpy/numba version contamination bug (see module
docstring) -- its whole contract with the caller (gen_dataset_from_captures.py's detect_delay())
is "exactly one JSON line on stdout, exit 0, even for an expected failure." A raised exception
or extra stdout output breaks that contract silently. `nam` is not installed in THIS venv
(confirmed -- ModuleNotFoundError), which is itself the real, always-exercised "nam_unavailable"
path; the other branches are exercised by injecting a fake nam.train.core module.

See tools/nam_delay_helper.py.
"""
import json
import sys
import types

import pytest

from tools import nam_delay_helper as helper


class TestEmit:
    def test_delay_only(self, capsys):
        helper._emit(128)
        assert json.loads(capsys.readouterr().out) == {"delay": 128}

    def test_includes_source_when_given(self, capsys):
        helper._emit(None, source="nam_unavailable")
        assert json.loads(capsys.readouterr().out) == {"delay": None, "source": "nam_unavailable"}

    def test_includes_detail_when_given(self, capsys):
        helper._emit(None, source="delay_zero_fallback", detail="ValueError: bad file")
        assert json.loads(capsys.readouterr().out) == {
            "delay": None, "source": "delay_zero_fallback", "detail": "ValueError: bad file"}

    def test_prints_exactly_one_line(self, capsys):
        helper._emit(128, source="nam_standard_calibration")
        out = capsys.readouterr().out
        assert out.count("\n") == 1


def install_fake_nam(monkeypatch, detect_fn=None, analyze_fn=None, validation_error=None):
    nam_pkg = types.ModuleType("nam")
    train_pkg = types.ModuleType("nam.train")
    core_mod = types.ModuleType("nam.train.core")
    core_mod._InputValidationError = validation_error or type("FakeInputValidationError", (Exception,), {})
    core_mod._detect_input_version = detect_fn
    core_mod._analyze_latency = analyze_fn
    nam_pkg.train = train_pkg
    train_pkg.core = core_mod
    monkeypatch.setitem(sys.modules, "nam", nam_pkg)
    monkeypatch.setitem(sys.modules, "nam.train", train_pkg)
    monkeypatch.setitem(sys.modules, "nam.train.core", core_mod)
    return core_mod._InputValidationError


def run_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["nam_delay_helper.py", "dry.wav", "wet.wav"])
    helper.main()
    return json.loads(capsys.readouterr().out)


class TestMainWithoutNamInstalled:
    def test_import_failure_reports_nam_unavailable(self, monkeypatch, capsys):
        # `nam` isn't installed in THIS venv on its own, but capture_static.py (imported by
        # another test module in the same session) inserts a real sibling
        # neural-amp-modeler/venv's site-packages onto sys.path at import time if one happens
        # to exist on the machine running the tests -- that mutation isn't test-scoped, so
        # whether `nam` resolves for real here otherwise depends on run order and the local
        # machine, not this test. Force the ImportError deterministically instead of relying
        # on ambient environment state: setting a sys.modules entry to None makes the next
        # import of that name raise ImportError unconditionally, regardless of sys.path.
        for name in ("nam", "nam.train", "nam.train.core"):
            monkeypatch.setitem(sys.modules, name, None)
        out = run_main(monkeypatch, capsys)
        assert out["delay"] is None
        assert out["source"] == "nam_unavailable"
        assert "detail" in out


class TestMainWithFakeNam:
    def test_input_validation_error_falls_back_without_calling_analyze_latency(self, monkeypatch, capsys):
        analyze_calls = []
        my_validation_error = type("FakeInputValidationError", (Exception,), {})
        install_fake_nam(
            monkeypatch,
            detect_fn=lambda path: (_ for _ in ()).throw(my_validation_error("no match")),
            analyze_fn=lambda **kw: analyze_calls.append(kw),
            validation_error=my_validation_error,
        )
        out = run_main(monkeypatch, capsys)
        assert out == {"delay": None, "source": "delay_zero_fallback"}
        assert analyze_calls == []

    def test_unrelated_detect_exception_reports_delay_zero_fallback_with_detail(self, monkeypatch, capsys):
        install_fake_nam(monkeypatch, detect_fn=lambda path: (_ for _ in ()).throw(ValueError("bad wav")))
        out = run_main(monkeypatch, capsys)
        assert out["delay"] is None
        assert out["source"] == "delay_zero_fallback"
        assert "ValueError" in out["detail"]

    def test_no_version_match_reports_delay_zero_fallback_without_detail(self, monkeypatch, capsys):
        install_fake_nam(monkeypatch, detect_fn=lambda path: (None, False))
        out = run_main(monkeypatch, capsys)
        assert out == {"delay": None, "source": "delay_zero_fallback"}

    def test_successful_calibration_reports_the_recommended_delay(self, monkeypatch, capsys):
        result = types.SimpleNamespace(calibration=types.SimpleNamespace(recommended=128))
        install_fake_nam(monkeypatch, detect_fn=lambda path: ("v1", True),
                          analyze_fn=lambda **kw: result)
        out = run_main(monkeypatch, capsys)
        assert out == {"delay": 128, "source": "nam_standard_calibration"}

    def test_analyze_latency_exception_reports_delay_zero_fallback_with_detail(self, monkeypatch, capsys):
        def boom(**kw):
            raise RuntimeError("no impulse response found")
        install_fake_nam(monkeypatch, detect_fn=lambda path: ("v1", True), analyze_fn=boom)
        out = run_main(monkeypatch, capsys)
        assert out["delay"] is None
        assert out["source"] == "delay_zero_fallback"
        assert "no impulse response found" in out["detail"]

    def test_recommended_delay_of_none_reports_delay_zero_fallback_without_detail(self, monkeypatch, capsys):
        result = types.SimpleNamespace(calibration=types.SimpleNamespace(recommended=None))
        install_fake_nam(monkeypatch, detect_fn=lambda path: ("v1", True),
                          analyze_fn=lambda **kw: result)
        out = run_main(monkeypatch, capsys)
        assert out == {"delay": None, "source": "delay_zero_fallback"}
