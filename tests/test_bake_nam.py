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
from param_train import ParametricA2

ROOT = Path(__file__).resolve().parent.parent


def _make_parametric_file(path: Path, defaults=None):
    """Write a small parametric .param.nam with declared knob metadata."""
    m = ParametricA2(3, 2)
    cfg = {"param_names": ["SUSTAIN", "TONE"],
           "bounds": {"SUSTAIN": [0.0, 1.0], "TONE": [0.0, 1.0]}}
    if defaults:
        cfg["defaults"] = defaults
    d = m.export_nam(cfg, {"version": "0.7.0"}, 48000)
    path.write_text(json.dumps(d))
    return path


def _bake(src, out, *extra):
    subprocess.run([sys.executable, "bake_nam.py", "--in", str(src), "-o", str(out), *extra],
                   check=True, cwd=ROOT, capture_output=True, text=True)
    return json.loads(out.read_text())


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


def test_embed_parametric_is_dual_payload(tmp_path):
    src = _make_parametric_file(tmp_path / "m.param.nam", {"SUSTAIN": 0.3})
    original = json.loads(src.read_text())
    out = _bake(src, tmp_path / "o.nam", "--embed-parametric")
    # top-level is a stock-legible baked WaveNet; parametric rides along verbatim
    assert out["architecture"] == "WaveNet"
    assert out[bake_nam.EMBED_KEY]["architecture"] == "ParametricWaveNet"
    assert out[bake_nam.EMBED_KEY] == original
    assert out["metadata"]["parametric_embedded"] is True


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
