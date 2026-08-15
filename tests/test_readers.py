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
import param_infer
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


def test_export_checkpoint_detects_lora_rank():
    """LoRA rank is recovered from the state dict alone (net_A.weight's shape, divided by
    that tier's own channels from rechannel.weight) -- no args_dict needed, the same
    state-dict-only recovery detect_spectral_norm already relies on for --compose mode."""
    from param_train import SlimmableParametricA2, ParametricA2
    assert export_checkpoint.detect_lora_rank(SlimmableParametricA2(2, widths=[3, 8]).state_dict()) == 0
    assert export_checkpoint.detect_lora_rank(
        SlimmableParametricA2(2, widths=[3, 8], lora_rank=5).state_dict()) == 5
    assert export_checkpoint.detect_lora_rank(
        SlimmableParametricA2(2, widths=[3, 4, 8], lora_rank=2).state_dict()) == 2
    assert export_checkpoint.detect_lora_rank(ParametricA2(3, 2).state_dict()) == 0
    assert export_checkpoint.detect_lora_rank(ParametricA2(3, 2, lora_rank=4).state_dict()) == 4


def test_require_tier_agreement_passes_when_unanimous():
    assert export_checkpoint.require_tier_agreement({"lite": 4, "full": 4}, "LoRA rank") == 4
    assert export_checkpoint.require_tier_agreement({"lite": False, "full": False},
                                                     "spectral_norm") is False


def test_require_tier_agreement_rejects_lora_rank_mismatch():
    """--compose must refuse to splice a rank-4 tier with a rank-0 (no LoRA) or
    differently-ranked tier into one container -- this is the actual check
    export_checkpoint.py's --compose mode relies on to reject that combination."""
    with pytest.raises(SystemExit, match="LoRA rank"):
        export_checkpoint.require_tier_agreement({"lite": 0, "full": 4}, "LoRA rank")
    with pytest.raises(SystemExit, match="LoRA rank"):
        export_checkpoint.require_tier_agreement({"lite": 2, "full": 4}, "LoRA rank")


def test_param_infer_load_model_reads_back_lora_rank(tmp_path):
    """param_infer.py's load_model() threads lora_rank out of a checkpoint's args_dict the
    same way spectral_norm already was ('free recovery' -- the whole argparse namespace is
    saved verbatim at training time, see param_infer.py's own comment on this). Without this,
    a LoRA checkpoint would silently reconstruct as lora_rank=0 -- load_state_dict would then
    fail loudly (missing lora.* keys) rather than misload, but the model would simply be
    unloadable, defeating the point of inspecting a LoRA checkpoint with this tool at all."""
    from param_train import SlimmableParametricA2

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    (dataset_dir / "config.json").write_text(json.dumps({"knobs": ["a", "b"]}))

    torch.manual_seed(0)
    original = SlimmableParametricA2(num_params=2, widths=[3, 8], lora_rank=4).eval()

    ckpt_path = tmp_path / "best.pt"
    torch.save({
        "model": original.state_dict(),
        "args_dict": {"dataset": str(dataset_dir), "widths": [3, 8], "lora_rank": 4},
    }, str(ckpt_path))

    model, args_dict = param_infer.load_model(str(ckpt_path))
    assert args_dict["lora_rank"] == 4
    for sub in model.submodels:
        assert sub.lora_rank == 4
        assert all(layer.lora is not None for layer in sub.layers)

    x = torch.randn(1, 1, 4096)
    cond = torch.tensor([[0.3, 0.7]])
    with torch.no_grad():
        y_original = original(x, cond)
        y_loaded = model(x, cond)
    for a, b in zip(y_original, y_loaded):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def _minimal_train_cmd_args(**overrides):
    """A Namespace covering every attribute build_train_cmd() reads, with the same
    defaults argparse would assign (see run_pipeline.py's own --xxx default= values).
    overrides lets a test vary just the flag(s) it cares about."""
    import argparse
    defaults = dict(
        nam_output="out.param.nam", checkpoint_dir="ckpt", restart_period=50, restart_mult=1,
        stale_cycles=3, batch_size=16, lr=3e-4, crop_len=48000, mrstft_weight=0.0,
        val_split=0.05, val_passes=1, device="cpu", seed=42, widths=None, mmap=True,
        resume=None, amp="off", init_from=None, param_sensitivity=False, knob_boost=None,
        per_tier_clip=False, clip_norm=1.0, spectral_norm=False, lora_rank=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_train_cmd_omits_lora_rank_when_zero():
    args = _minimal_train_cmd_args(lora_rank=0)
    cmd = run_pipeline.build_train_cmd(args, "ds", 100, 4)
    assert "--lora-rank" not in cmd


def test_build_train_cmd_forwards_lora_rank():
    """The actual gap this test closes: --lora-rank was added to run_pipeline.py's CLI and
    threaded into train_cmd (mirroring the existing --spectral-norm pattern) with no test
    confirming the flag actually reaches param_train.py's command line."""
    args = _minimal_train_cmd_args(lora_rank=4)
    cmd = run_pipeline.build_train_cmd(args, "ds", 100, 4)
    assert "--lora-rank" in cmd
    assert cmd[cmd.index("--lora-rank") + 1] == "4"


def test_lora_checkpoint_round_trips_through_state_dict_reconstruction(tmp_path):
    """The actual thing Phase 2 validates: build a small LoRA-enabled slimmable model
    (simulating a saved checkpoint's 'model' state), detect its widths/lora_rank purely
    from the state dict (as export_checkpoint.py's single-checkpoint path does), rebuild a
    FRESH model from those detected values, load the state, and confirm the reconstruction's
    forward pass is bit-identical to the original -- proves detect_lora_rank's output is
    actually sufficient to reconstruct a working model, not just a plausible-looking number.
    """
    from param_train import SlimmableParametricA2

    torch.manual_seed(0)
    original = SlimmableParametricA2(num_params=2, widths=[3, 8], lora_rank=4).eval()
    for p in original.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    state = original.state_dict()

    widths = export_checkpoint.infer_widths(state)
    lora_rank = export_checkpoint.detect_lora_rank(state)
    assert widths == [3, 8]
    assert lora_rank == 4

    rebuilt = SlimmableParametricA2(num_params=2, widths=widths, lora_rank=lora_rank).eval()
    rebuilt.load_state_dict(state)

    x = torch.randn(1, 1, 4096)
    cond = torch.tensor([[0.2, 0.9]])
    with torch.no_grad():
        y_original = original(x, cond)
        y_rebuilt = rebuilt(x, cond)
    for a, b in zip(y_original, y_rebuilt):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_export_nam_lora_round_trip_matches_live_model():
    """Regression test for a real bug found in param_train.py's own export-time round-trip
    check (main()'s `m2 = ParametricA2(ch, num_params)` -- missing lora_rank): reconstructing
    a LoRA-trained submodel from its exported `weights` list without threading lora_rank
    through doesn't crash (load_weights doesn't validate weight count) -- it silently
    consumes each layer's lora.net_A/net_B weights as if they belonged to the NEXT layer's
    conv/mixin/l1x1, corrupting every downstream layer. Caught as a large-but-finite
    max_diff, not an error.

    Calls param_train.verify_export_round_trip() directly -- the ACTUAL function main() now
    calls for this check (extracted specifically so this test exercises real production code,
    not an independent reimplementation of the same pattern that could silently drift from
    it) -- for a multi-tier slimmable model (the scenario that actually surfaced the bug --
    two different channel counts sharing one lora_rank)."""
    from param_train import SlimmableParametricA2, verify_export_round_trip

    torch.manual_seed(0)
    model = SlimmableParametricA2(num_params=2, widths=[4, 8], lora_rank=3).eval()
    for p in model.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)

    nam = model.export_nam({"param_names": ["a", "b"]}, {"version": "0.7.0"},
                           sample_rate=48000, input_audio=None)

    x = torch.randn(1, 1, 4096)
    cond = torch.tensor([[0.2, 0.9]])
    results = verify_export_round_trip(nam, model, num_params=2, device="cpu",
                                       test_inp=x, test_params=cond)
    assert [lbl for lbl, _ in results] == model.tier_labels()
    for lbl, max_diff in results:
        assert max_diff <= 1e-6, f"tier {lbl!r}: round-trip max_diff={max_diff:.3e} (expected ~0)"


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


# --------------------------------------------------------------------------- resolve_device

def test_resolve_device_passes_through_an_explicit_choice_unchanged():
    # Explicit "cpu" is the escape hatch -- honored as asked, never refused, even
    # though --device auto refuses in the exact same (no GPU) situation below.
    assert param_train.resolve_device("cpu") == "cpu"


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert param_train.resolve_device("auto") == "cuda"


def test_resolve_device_auto_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert param_train.resolve_device("auto") == "mps"


def test_resolve_device_auto_hard_fails_when_no_gpu_is_available(monkeypatch):
    # Not a warn-and-proceed: CPU training is realistically infeasible for a real
    # run, not just slow, so the auto-detected case refuses outright rather than
    # silently queuing a run that will never finish.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="--device cpu"):
        param_train.resolve_device("auto")
