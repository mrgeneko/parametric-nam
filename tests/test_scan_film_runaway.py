"""Tests for scan_film_runaway.py's model-reconstruction path.

Had zero test coverage before -- not even for its pre-existing (non-LoRA) behavior. The gap
that actually mattered: load_all_submodels() reconstructed every submodel with lora_rank=0
regardless of what the export declared, so a LoRA-tagged model would hit the weight_count()
mismatch guard and refuse to load at all -- the mandatory safety sweep (see internal
engineering notes on the LoRA-conditioning plan) couldn't run against a LoRA model.
"""
import json

import torch

from scan_film_runaway import load_all_submodels
from param_train import SlimmableParametricA2


def _write_nam(tmp_path, lora_rank, widths=(4, 8), num_params=2, seed=0):
    torch.manual_seed(seed)
    model = SlimmableParametricA2(num_params=num_params, widths=list(widths),
                                  lora_rank=lora_rank).eval()
    for p in model.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    config = {"param_names": [f"knob{i}" for i in range(num_params)]}
    nam = model.export_nam(config, {"version": "0.7.0"}, sample_rate=48000, input_audio=None)
    path = tmp_path / "model.param.nam"
    path.write_text(json.dumps(nam))
    return path, model


def test_load_all_submodels_film_only(tmp_path):
    path, _ = _write_nam(tmp_path, lora_rank=0)
    submodels = load_all_submodels(str(path))
    assert [ch for _, _, ch in submodels] == [4, 8]
    for model, param_names, _ in submodels:
        assert param_names == ["knob0", "knob1"]
        assert all(layer.lora is None for layer in model.layers)


def test_load_all_submodels_lora(tmp_path):
    """The actual regression: before the fix, this raised SystemExit (weight count
    mismatch) for any LoRA-tagged export -- the safety sweep couldn't run at all."""
    path, _ = _write_nam(tmp_path, lora_rank=3)
    submodels = load_all_submodels(str(path))
    assert [ch for _, _, ch in submodels] == [4, 8]
    for model, _, _ in submodels:
        assert all(layer.lora is not None and layer.lora.rank == 3 for layer in model.layers)


def test_load_all_submodels_lora_matches_live_model(tmp_path):
    """Not just 'doesn't crash' -- the reconstructed model must be numerically identical
    to the model that was actually exported, same bar as the export/round-trip tests."""
    path, original = _write_nam(tmp_path, lora_rank=3)
    submodels = load_all_submodels(str(path))

    x = torch.randn(1, 1, 4096)
    cond = torch.tensor([[0.2, 0.9]])
    with torch.no_grad():
        y_original = original(x, cond)
    for (model, _, _), y_src in zip(submodels, y_original):
        with torch.no_grad():
            y_loaded = model(x, cond)
        torch.testing.assert_close(y_loaded, y_src, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# --config auto-discovery (main()'s corner-set selection)
#
# Motivated by a real miss: scanning a published Tweed model with no --config at all
# (the natural way to invoke this tool by hand) silently used the reduced hypercube corner
# set and came back clean, while the FULL trained grid (--config) found a real 8-9x FiLM/
# LeakyReLU runaway at an interior grid point no hypercube vertex reaches. Auto-discovering
# a config.toml next to --nam (every release bundle ships one) makes the full grid the
# default without requiring the caller to remember --config.
# ---------------------------------------------------------------------------
import sys

import numpy as np
import soundfile as sf

import scan_film_runaway as sfr


def _write_reference(tmp_path, seconds=1.0, sr=48000):
    n = int(seconds * sr)
    x = (0.5 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    path = tmp_path / "reference.wav"
    sf.write(str(path), x, sr)
    return path


def _run_main(monkeypatch, argv, full_corners, hyper_corners):
    calls = {"full": 0, "hyper": 0}

    def fake_full(config_path, param_names):
        calls["full"] += 1
        return full_corners

    def fake_hyper(param_names):
        calls["hyper"] += 1
        return hyper_corners

    monkeypatch.setattr(sfr, "full_grid_corners", fake_full)
    monkeypatch.setattr(sfr, "hypercube_corners", fake_hyper)
    monkeypatch.setattr(sys, "argv", argv)
    sfr.main()
    return calls


def test_auto_discovers_config_toml_next_to_nam(tmp_path, monkeypatch, capsys):
    path, _ = _write_nam(tmp_path, lora_rank=0)
    ref = _write_reference(tmp_path)
    (tmp_path / "config.toml").write_text("# stub, never actually parsed (mocked)\n")
    corner = ("knob0=0.5,knob1=0.5", [0.5, 0.5])

    calls = _run_main(monkeypatch, ["scan_film_runaway.py", "--nam", str(path),
                                    "--reference", str(ref), "--chunk-s", "0.5"],
                      full_corners=[corner], hyper_corners=[corner])

    assert calls == {"full": 1, "hyper": 0}
    out = capsys.readouterr().out
    assert "auto-discovered" in out
    assert "[full grid]" in out
    assert "[reduced hypercube]" not in out


def test_no_auto_config_forces_reduced_set_even_with_config_toml_present(tmp_path, monkeypatch, capsys):
    path, _ = _write_nam(tmp_path, lora_rank=0)
    ref = _write_reference(tmp_path)
    (tmp_path / "config.toml").write_text("# stub\n")
    corner = ("knob0=0.5,knob1=0.5", [0.5, 0.5])

    calls = _run_main(monkeypatch, ["scan_film_runaway.py", "--nam", str(path),
                                    "--reference", str(ref), "--no-auto-config", "--chunk-s", "0.5"],
                      full_corners=[corner], hyper_corners=[corner])

    assert calls == {"full": 0, "hyper": 1}
    out = capsys.readouterr().out
    assert "auto-discovered" not in out
    assert "[reduced hypercube]" in out


def test_no_config_anywhere_falls_back_to_reduced_set_with_warning(tmp_path, monkeypatch, capsys):
    """No trained grid discoverable at all -- no dataset_params.csv, no config.toml next to
    --nam. Must still work (the reduced set is a legitimate last resort), but must warn
    loudly: this is exactly the path that silently missed a real defect before, and it is
    also the path that reports EXTRAPOLATION beyond the trained range as if it were
    instability (2026-09-17: 47,461x and 43x readings on two models that are 0/576 and
    0/16 across their real grids)."""
    path, _ = _write_nam(tmp_path, lora_rank=0)
    ref = _write_reference(tmp_path)
    corner = ("knob0=0.5,knob1=0.5", [0.5, 0.5])

    calls = _run_main(monkeypatch, ["scan_film_runaway.py", "--nam", str(path),
                                    "--reference", str(ref), "--chunk-s", "0.5"],
                      full_corners=[corner], hyper_corners=[corner])

    assert calls == {"full": 0, "hyper": 1}
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "hypercube" in out           # wording covers the 3-source discovery chain
    assert "EXTRAPOLATION" in out       # the fallback must say findings are not a verdict
    assert "[reduced hypercube]" in out


def test_explicit_config_overrides_auto_discovery_path(tmp_path, monkeypatch, capsys):
    """--config wins even when a DIFFERENT config.toml also happens to sit next to --nam --
    explicit should never be silently shadowed by auto-discovery."""
    path, _ = _write_nam(tmp_path, lora_rank=0)
    ref = _write_reference(tmp_path)
    (tmp_path / "config.toml").write_text("# the one that should NOT be used\n")
    explicit_dir = tmp_path / "elsewhere"
    explicit_dir.mkdir()
    explicit_cfg = explicit_dir / "other.toml"
    explicit_cfg.write_text("# the explicit one\n")
    corner = ("knob0=0.5,knob1=0.5", [0.5, 0.5])

    seen_paths = []
    def fake_full(config_path, param_names):
        seen_paths.append(config_path)
        return [corner]
    monkeypatch.setattr(sfr, "full_grid_corners", fake_full)
    monkeypatch.setattr(sfr, "hypercube_corners", lambda param_names: [corner])
    monkeypatch.setattr(sys, "argv", ["scan_film_runaway.py", "--nam", str(path),
                                      "--reference", str(ref), "--config", str(explicit_cfg), "--chunk-s", "0.5"])
    sfr.main()

    assert seen_paths == [str(explicit_cfg)]
    out = capsys.readouterr().out
    assert "auto-discovered" not in out
