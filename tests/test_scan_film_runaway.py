"""Tests for tools/scan_film_runaway.py's model-reconstruction path.

Had zero test coverage before -- not even for its pre-existing (non-LoRA) behavior. The gap
that actually mattered: load_all_submodels() reconstructed every submodel with lora_rank=0
regardless of what the export declared, so a LoRA-tagged model would hit the weight_count()
mismatch guard and refuse to load at all -- the mandatory safety sweep (see internal
engineering notes on the LoRA-conditioning plan) couldn't run against a LoRA model.
"""
import json

import torch

from tools.scan_film_runaway import load_all_submodels
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
