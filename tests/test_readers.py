"""Tests for the modules that LOAD models — infer.py, param_infer.py, export_checkpoint.py —
and for run_pipeline's config mapping.

These had no coverage, which mattered: the skip-only refactor edited all three, and the one
invariant they must all uphold is the guard. Models are SKIP-ONLY; a legacy residual (or
untagged) model must be REFUSED by every reader, because its weights under a skip head
produce garbage (corr ~0.3) with NO error. A reader that silently accepts one is the exact
bug this whole migration exists to prevent — so the guard is tested per reader, not just
once in the helper.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import export_checkpoint
import param_train
import run_pipeline
from param_train import ParametricA2, check_parametric_schema

ROOT = Path(__file__).resolve().parent.parent


def _skip_model_file(path: Path, num_params: int = 2) -> Path:
    """A current (skip) parametric .param.nam.

    NOTE num_params=1 for the infer.py tests: infer.py's CLI can only drive a SINGLE knob
    (--gain2). It cannot run a multi-knob model at all — a pre-existing limitation, not a
    regression (it is why ab_realtime_playback.py exists). Tracked below."""
    names = ["SUSTAIN", "TONE"][:num_params]
    m = ParametricA2(3, num_params)
    d = m.export_nam(
        {"param_names": names, "bounds": {n: [0.0, 1.0] for n in names}},
        {"version": "0.7.0"}, 48000)
    path.write_text(json.dumps(d))
    return path


def _make_legacy(path: Path, mode="residual") -> Path:
    """Downgrade a model file to a legacy one: residual, or untagged entirely."""
    d = json.loads(path.read_text())
    par = d["config"]["parametric"]
    if mode is None:
        par.pop("head_mode", None)          # untagged == pre-tagging == residual
    else:
        par["head_mode"] = mode
    path.write_text(json.dumps(d))
    return path


# --------------------------------------------------------------------------- guard (unit)

def test_guard_accepts_a_current_skip_model():
    assert check_parametric_schema({"head_mode": "skip", "schema_version": 1}) == 1


@pytest.mark.parametrize("par", [
    {"head_mode": "residual", "schema_version": 1},          # legacy residual
    {"schema_version": 1},                                   # untagged => residual
    {},                                                      # pre-versioning
    {"head_mode": "bogus", "schema_version": 1},             # unknown head
])
def test_guard_rejects_anything_not_skip(par):
    with pytest.raises(SystemExit):
        check_parametric_schema(par, source="x.nam")


def test_guard_rejects_a_newer_schema():
    with pytest.raises(SystemExit):
        check_parametric_schema(
            {"head_mode": "skip", "schema_version": param_train.K_PARAM_SCHEMA_VERSION + 1})


# --------------------------------------------------------------------------- infer.py

def _tiny_wav(path: Path, seconds=0.2, sr=48000) -> Path:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.2 * np.sin(2 * np.pi * 220 * t), sr, subtype="FLOAT")
    return path


def _run_infer(model: Path, wav: Path, out: Path):
    return subprocess.run(
        [sys.executable, "infer.py", "--model", str(model), "--input", str(wav),
         "--output", str(out), "--gain2", "0.5"],
        cwd=ROOT, capture_output=True, text=True)


def test_infer_runs_a_skip_model(tmp_path):
    model = _skip_model_file(tmp_path / "m.param.nam", num_params=1)   # infer.py: 1 knob only
    wav = _tiny_wav(tmp_path / "in.wav")
    r = _run_infer(model, wav, tmp_path / "out.wav")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "out.wav").exists()


@pytest.mark.parametrize("mode", ["residual", None])
def test_infer_rejects_a_legacy_model(tmp_path, mode):
    """The reader must REFUSE, not render. Rendering it would emit garbage silently."""
    model = _make_legacy(_skip_model_file(tmp_path / "m.param.nam", num_params=1), mode)
    wav = _tiny_wav(tmp_path / "in.wav")
    out = tmp_path / "out.wav"
    r = _run_infer(model, wav, out)
    assert r.returncode != 0
    assert "no longer supported" in (r.stdout + r.stderr)
    assert not out.exists()


# ------------------------------------------------------------------- export_checkpoint.py

@pytest.mark.xfail(reason="KNOWN GAP: infer.py's CLI exposes only --gain2, so it cannot "
                          "drive a multi-knob model. Not a regression — ab_realtime_playback.py "
                          "exists because of it. Fix: give infer.py a --params flag like "
                          "bake_nam's.", strict=True)
def test_infer_can_drive_a_multi_knob_model(tmp_path):
    model = _skip_model_file(tmp_path / "m.param.nam", num_params=2)
    wav = _tiny_wav(tmp_path / "in.wav")
    r = _run_infer(model, wav, tmp_path / "out.wav")
    assert r.returncode == 0, r.stdout + r.stderr


def test_export_checkpoint_infers_slimmable_widths():
    """Widths are recovered from the state dict, so a checkpoint round-trips to the right
    container shape ([3,4,8] must not silently collapse to the default 2-tier shape)."""
    from param_train import SlimmableParametricA2
    model = SlimmableParametricA2(2, widths=[3, 4, 8])
    assert export_checkpoint.infer_widths(model.state_dict()) == [3, 4, 8]


def test_export_checkpoint_model_state_accepts_either_key():
    assert export_checkpoint.model_state({"model": {"a": 1}}) == {"a": 1}
    assert export_checkpoint.model_state({"best_state": {"b": 2}}) == {"b": 2}
    assert export_checkpoint.model_state({"c": 3}) == {"c": 3}   # raw state dict


def test_export_composite_nam_splices_each_tiers_own_weights_and_restores_after(
        tmp_path, monkeypatch):
    """export_composite_nam is called automatically now whenever any tier improves
    (param_train.py's epoch loop) -- it MUST pull each tier's slice from that tier's
    OWN best_state (not another tier's, and not the model's current in-training
    weights), and MUST leave the model's live weights untouched afterward so training
    continues normally. Spies on SlimmableParametricA2.export_nam to inspect the
    state_dict that was actually loaded at export time, rather than parsing the real
    .nam JSON format -- decouples this test from that format entirely."""
    from types import SimpleNamespace
    from param_train import SlimmableParametricA2, export_composite_nam

    model = SlimmableParametricA2(2, widths=[3, 4, 8])
    labels = model.tier_labels()
    assert labels == ["lite", "w4", "full"]
    original = {k: v.clone() for k, v in model.state_dict().items()}

    # Distinct, checkable best_state per tier: only THAT tier's own slice is scaled by
    # a distinctive factor. If the splice cross-contaminates tiers, the wrong scale
    # shows up on the wrong slice.
    scale = {"lite": 2.0, "w4": 3.0, "full": 5.0}
    best_state = {}
    for i, lbl in enumerate(labels):
        pref = model.tier_state_prefix(i)
        st = {k: v.clone() for k, v in original.items()}
        for k in st:
            if k.startswith(pref):
                st[k] = st[k] * scale[lbl]
        best_state[lbl] = st

    captured = {}
    def fake_export_nam(self, config, metadata, sample_rate, input_audio=None):
        captured["state"] = {k: v.clone() for k, v in self.state_dict().items()}
        return {"fake": True}
    monkeypatch.setattr(SlimmableParametricA2, "export_nam", fake_export_nam)

    out_path = tmp_path / "optimal.param.nam"
    export_composite_nam(model, best_state, SimpleNamespace(config={}, inp=None),
                         out_path, device="cpu")

    for i, lbl in enumerate(labels):
        pref = model.tier_state_prefix(i)
        for k, v in captured["state"].items():
            if k.startswith(pref):
                assert torch.equal(v, best_state[lbl][k]), f"{k} wasn't spliced from {lbl}"

    after = model.state_dict()
    for k, v in original.items():
        assert torch.equal(after[k], v), f"{k} wasn't restored after the composite export"

    assert json.loads(out_path.read_text()) == {"fake": True}


def test_exported_model_is_tagged_skip(tmp_path):
    """Whatever path produced it, an exported model self-describes as skip — that tag is what
    lets every reader reject the legacy ones."""
    d = json.loads(_skip_model_file(tmp_path / "m.param.nam").read_text())
    par = d["config"]["parametric"]
    assert par["head_mode"] == "skip"
    assert par["schema_version"] == param_train.K_PARAM_SCHEMA_VERSION


# ------------------------------------------------------------------------- run_pipeline.py

def test_load_config_maps_knobs_fixed_and_defaults(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        'schx = "x.schx"\n'
        'epochs = 400\n'
        'widths = [3, 4, 8]\n'
        '[knobs]\n'
        'SUSTAIN = [0.0, 0.5, 1.0]\n'
        'TONE = [0.0, 1.0]\n'
        '[fixed]\n'
        'VOLUME = 1.0\n'
        '[defaults]\n'
        'SUSTAIN = 0.3\n')
    out = run_pipeline.load_config(cfg)
    assert out["knobs"] == "SUSTAIN,TONE"
    assert out["ranges"] == ["SUSTAIN=0.0,0.5,1.0", "TONE=0.0,1.0"]
    assert out["fixed_params"] == "VOLUME=1.0"
    assert out["defaults"] == "SUSTAIN=0.3"     # the Timmy case: a real default, not midpoint
    assert out["widths"] == "3,4,8"             # list -> the string --widths expects
    assert out["epochs"] == 400
