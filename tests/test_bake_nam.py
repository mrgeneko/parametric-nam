"""Tests for bake_nam.py — declared-default baking + the dual-payload
(--embed-parametric) file. The stock-load assertion needs neural-amp-modeler
(see conftest.py) and skips if absent."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import bake_nam
import param_train
from param_train import ParametricA2

ROOT = Path(__file__).resolve().parent.parent


def _make_parametric_file(path: Path, defaults=None):
    """Write a small parametric .param.nam. Models are skip-only."""
    m = ParametricA2(3, 2)
    cfg = {"param_names": ["SUSTAIN", "TONE"],
           "bounds": {"SUSTAIN": [0.0, 1.0], "TONE": [0.0, 1.0]}}
    if defaults:
        cfg["defaults"] = defaults
    d = m.export_nam(cfg, {"version": "0.7.0"}, 48000)
    path.write_text(json.dumps(d))
    return path


def _bake_raw(src, out, *extra):
    return subprocess.run(
        [sys.executable, "bake_nam.py", "--in", str(src), "-o", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True)


def _bake(src, out, *extra):
    r = _bake_raw(src, out, *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(out.read_text())


def test_export_nam_propagates_steps_for_discrete_knobs():
    """A knob marked discrete via config["steps"] (from --steps at dataset-generation
    time) must carry a "steps" field in the exported parameter def, so readers (e.g.
    NeuralAmpModelerCore's DSPParamDef.steps) can render an N-position selector instead
    of a continuous knob. Continuous knobs must NOT get a spurious "steps" key."""
    m = ParametricA2(3, 2)
    cfg = {"param_names": ["SUSTAIN", "TONE"],
           "bounds": {"SUSTAIN": [0.0, 1.0], "TONE": [0.0, 1.0]},
           "steps": {"TONE": 3}}
    d = m.export_nam(cfg, {"version": "0.7.0"}, 48000)
    params = {p["name"]: p for p in d["config"]["parametric"]["parameters"]}
    assert params["TONE"]["steps"] == 3
    assert "steps" not in params["SUSTAIN"]


def test_knob_default_prefers_declared_then_midpoint():
    assert bake_nam._knob_default({"name": "x", "min": 0, "max": 1, "default": 0.3}) == 0.3
    assert bake_nam._knob_default({"name": "x", "min": 0.2, "max": 0.8}) == 0.5  # midpoint
    assert bake_nam._knob_default({"name": "x"}) == 0.5                          # last resort


def test_bake_uses_declared_default_not_half(tmp_path):
    """A no-params bake must use the circuit's declared default (the Timmy case)."""
    src = _make_parametric_file(tmp_path / "m.param.nam", {"SUSTAIN": 0.3})
    out = _bake(src, tmp_path / "o.nam")
    assert out["architecture"] == "WaveNet"
    assert out["metadata"]["baked_setting"]["SUSTAIN"] == 0.3   # declared, not 0.5
    assert out["metadata"]["baked_setting"]["TONE"] == 0.5      # midpoint fallback


def test_override_beats_declared_default(tmp_path):
    src = _make_parametric_file(tmp_path / "m.param.nam", {"SUSTAIN": 0.3})
    out = _bake(src, tmp_path / "o.nam", "--params", "SUSTAIN=0.85")
    assert out["metadata"]["baked_setting"]["SUSTAIN"] == 0.85


def test_dual_is_the_default(tmp_path):
    """A plain bake must be safe to hand anyone: a stock plugin plays the baked tone, a
    parametric host unlocks knobs. A raw .param.nam would THROW in stock, so the dual
    shape is the safe default rather than an opt-in."""
    src = _make_parametric_file(tmp_path / "m.param.nam", {"SUSTAIN": 0.3})
    out = _bake(src, tmp_path / "o.nam")            # no flags
    assert out["architecture"] == "WaveNet"
    assert bake_nam.EMBED_KEY in out
    assert out["metadata"]["parametric_embedded"] is True


def test_no_embed_produces_static_only(tmp_path):
    """Capture-pack members opt out — embedding the master in every tone multiplies size."""
    src = _make_parametric_file(tmp_path / "m.param.nam", {"SUSTAIN": 0.3})
    out = _bake(src, tmp_path / "o.nam", "--no-embed-parametric")
    assert out["architecture"] == "WaveNet"
    assert bake_nam.EMBED_KEY not in out
    assert "parametric_embedded" not in out["metadata"]


def test_embed_parametric_is_dual_payload(tmp_path):
    src = _make_parametric_file(tmp_path / "m.param.nam", {"SUSTAIN": 0.3})
    original = json.loads(src.read_text())
    out = _bake(src, tmp_path / "o.nam", "--embed-parametric")
    # top-level is a stock-legible baked WaveNet; parametric rides along verbatim
    assert out["architecture"] == "WaveNet"
    assert out[bake_nam.EMBED_KEY]["architecture"] == "ParametricWaveNet"
    assert out[bake_nam.EMBED_KEY] == original
    assert out["metadata"]["parametric_embedded"] is True


def test_export_stamps_skip_and_schema_version(tmp_path):
    """Every model self-describes so readers can REJECT legacy files rather than guess."""
    src = _make_parametric_file(tmp_path / "m.param.nam")
    par = json.loads(src.read_text())["config"]["parametric"]
    assert par["head_mode"] == "skip"
    assert par["schema_version"] == param_train.K_PARAM_SCHEMA_VERSION


def test_legacy_residual_model_is_rejected(tmp_path):
    """SKIP-ONLY. A legacy residual (or untagged) model must be REFUSED, not baked: its
    weights under a skip head produce garbage (corr ~0.3) with no error."""
    src = _make_parametric_file(tmp_path / "m.param.nam")
    d = json.loads(src.read_text())
    d["config"]["parametric"]["head_mode"] = "residual"     # legacy model
    src.write_text(json.dumps(d))
    r = _bake_raw(src, tmp_path / "o.nam")
    assert r.returncode != 0
    assert "no longer supported" in (r.stdout + r.stderr)
    assert not (tmp_path / "o.nam").exists()


def test_untagged_model_is_rejected(tmp_path):
    """No head_mode at all == a pre-tagging residual model. Also refused."""
    src = _make_parametric_file(tmp_path / "m.param.nam")
    d = json.loads(src.read_text())
    del d["config"]["parametric"]["head_mode"]
    src.write_text(json.dumps(d))
    r = _bake_raw(src, tmp_path / "o.nam")
    assert r.returncode != 0
    assert not (tmp_path / "o.nam").exists()


def test_schema_guard_rejects_a_newer_schema(tmp_path):
    """A file from a future build must fail loudly, not be silently misread."""
    src = _make_parametric_file(tmp_path / "m.param.nam")
    d = json.loads(src.read_text())
    d["config"]["parametric"]["schema_version"] = param_train.K_PARAM_SCHEMA_VERSION + 1
    src.write_text(json.dumps(d))
    r = _bake_raw(src, tmp_path / "o.nam")
    assert r.returncode != 0
    assert "schema_version" in (r.stdout + r.stderr)


def test_stock_loads_dual_file_ignoring_embed(tmp_path):
    """The official NAM loader reads the top-level static model and ignores the
    embedded parametric payload."""
    pytest.importorskip("nam", reason="neural-amp-modeler not installed (dev dep)")
    from nam.models import _from_nam
    src = _make_parametric_file(tmp_path / "m.param.nam", {"SUSTAIN": 0.3})
    out = _bake(src, tmp_path / "o.nam", "--embed-parametric")
    official = _from_nam.init_from_nam(out)
    official.eval()
    with torch.no_grad():
        y = official(torch.randn(1, 4096))
    assert np.isfinite(np.asarray(y)).all()
