"""Tests for bake_nam.py — declared-default baking + the dual-payload
(--embed-parametric) file. The stock-load assertion needs neural-amp-modeler
(see conftest.py) and skips if absent."""
import json
import subprocess
import sys
from pathlib import Path

from bake_nam import _model_stem, auto_name

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


def test_export_nam_propagates_gear_and_attribution_metadata():
    """gear_make/gear_model/gear_type (from gen_dataset_from_schx.py's --gear-make/--gear-model/
    --gear-type, baked into the dataset's config.json) and modeled_by (from param_train.py's
    --modeled-by, merged into dataset.config at train time) must all reach the exported .nam's
    metadata -- these were previously readable by export_nam() but had no CLI path to set them
    anywhere in the pipeline, so a real model (bigmuff_v4_optimal.param.nam) shipped with
    modeled_by missing entirely and gear_type silently defaulted to the wrong value ("amp" on
    an actual pedal)."""
    m = ParametricA2(3, 2)
    cfg = {"param_names": ["SUSTAIN", "TONE"],
           "bounds": {"SUSTAIN": [0.0, 1.0], "TONE": [0.0, 1.0]},
           "gear_make": "Electro-Harmonix", "gear_model": "Big Muff Pi V1", "gear_type": "pedal",
           "modeled_by": "Gene Ko"}
    d = m.export_nam(cfg, {"version": "0.7.0"}, 48000)
    meta = d["metadata"]
    assert meta["gear_make"] == "Electro-Harmonix"
    assert meta["gear_model"] == "Big Muff Pi V1"
    assert meta["gear_type"] == "pedal"
    assert meta["modeled_by"] == "Gene Ko"


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

class TestAutoName:
    """auto_name(): the filename must encode the setting that was actually baked.

    -o used to be required, so every caller invented its own scheme -- which is how one capture
    pack ends up with several conventions, and how a filename drifts from the setting it claims
    to describe. The caller has already stated the knob values; deriving the name from them is
    strictly less error-prone than restating them by hand.
    """

    BAKED = {"Gain": 0.55, "Bass": 0.5, "Treble": 0.2}

    def test_strips_the_double_param_nam_extension(self):
        assert _model_stem(Path("timmy_v4_optimal.param.nam")) == "timmy_v4_optimal"

    def test_strips_a_plain_nam_extension(self):
        assert _model_stem(Path("thing.nam")) == "thing"

    def test_encodes_every_knob_and_value(self):
        assert auto_name(Path("m.param.nam"), self.BAKED) == "m_Gain0.55_Bass0.5_Treble0.2.nam"

    def test_includes_defaulted_knobs_not_just_the_ones_passed(self):
        """The RESOLVED setting, not the --params string. A knob left out is still baked in at
        its declared default, so omitting it from the name would describe a different file --
        and would let two different settings collide on one filename."""
        full = auto_name(Path("m.param.nam"), {"Gain": 0.7, "Bass": 0.5, "Treble": 0.5})
        assert "Bass0.5" in full and "Treble0.5" in full

    def test_two_spellings_of_one_value_give_one_name(self):
        """%g formatting: 0.50 and 0.5 are the same setting and must not make two files."""
        assert auto_name(Path("m.param.nam"), {"Gain": 0.50}) == \
               auto_name(Path("m.param.nam"), {"Gain": 0.5})

    def test_width_included_only_when_pinned(self):
        """Tiers are genuinely different models, so a pack mixing them would collide -- but an
        unpinned width must not clutter the common case."""
        assert "w8" in auto_name(Path("m.param.nam"), self.BAKED, channels=8)
        assert "w" not in auto_name(Path("m.param.nam"), {"Gain": 0.5}).replace("m_", "")

    def test_distinct_settings_give_distinct_names(self):
        a = auto_name(Path("m.param.nam"), {"Gain": 1.0, "Bass": 0.35, "Treble": 0.8})
        b = auto_name(Path("m.param.nam"), {"Gain": 1.0, "Bass": 0.8, "Treble": 0.35})
        assert a != b
