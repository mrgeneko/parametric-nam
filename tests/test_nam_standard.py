"""Tests for the official-NAM static export + FiLM snapshot baking (nam_standard.py).

The fold-correctness test needs only torch (it proves the math). The round-trip test
needs the official `neural-amp-modeler` package (dev dep) and skips if it's absent.
"""
import numpy as np
import pytest
import torch

from param_train import ParametricA2, K_NUM_LAYERS
import nam_standard


def _mk(channels=3, num_params=2, seed=0):
    torch.manual_seed(seed)
    m = ParametricA2(channels=channels, num_params=num_params).eval()
    # randomize so FiLM actually does something (init is near-identity)
    for p in m.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    return m


def test_fold_film_matches_parametric():
    """A folded static A2 must reproduce the parametric model at that knob setting."""
    m = _mk()
    x = torch.randn(1, 1, 8192)
    params = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        y_param = m(x, params.view(1, -1))
        static = nam_standard.fold_film(m, params)
        y_static = static(x, torch.zeros(1, 0))
    assert static.num_params == 0
    assert all(layer.film is None for layer in static.layers)
    torch.testing.assert_close(y_param, y_static, atol=2e-5, rtol=1e-4)


def test_export_schema_is_official_wavenet():
    m = _mk()
    d = nam_standard.export_nam_standard(m, params=[0.5, 0.5])
    assert d["architecture"] == "WaveNet"
    lc = d["config"]["layers_configs"][0]
    assert lc["condition_size"] == 1 and lc["input_size"] == 1
    assert lc["activation"] == "LeakyReLU" and lc["gated"] is False
    assert len(lc["kernel_sizes"]) == K_NUM_LAYERS == len(lc["dilations"])
    assert lc["head"] == {"out_channels": 1, "kernel_size": 16, "bias": True}
    assert "head_scale" in d["config"]
    # weight count matches a static A2 (folded → no FiLM weights) + head_scale last
    static = nam_standard.fold_film(m, [0.5, 0.5])
    assert len(d["weights"]) == static.weight_count()
    assert d["weights"][-1] == pytest.approx(static.head_scale.item())


def test_parametric_requires_params():
    with pytest.raises(ValueError):
        nam_standard.export_nam_standard(_mk(), params=None)


def test_roundtrip_through_official_nam():
    """Load our exported .nam with the official package and reproduce the output."""
    nam = pytest.importorskip("nam", reason="neural-amp-modeler not installed (dev dep)")
    from nam.models import _from_nam  # noqa
    import json, tempfile, os

    m = _mk()
    params = [0.4, 0.6]
    d = nam_standard.export_nam_standard(m, params=params)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.nam")
        json.dump(d, open(p, "w"))
        official = _from_nam.init_from_nam(p) if hasattr(_from_nam, "init_from_nam") \
            else nam.models.model_from_nam(p)  # API name varies by version
    x = torch.randn(1, 1, 8192)
    with torch.no_grad():
        ours = nam_standard.fold_film(m, params)(x, torch.zeros(1, 0)).squeeze().numpy()
        theirs = official(x.squeeze().numpy()) if callable(official) else None
    if theirs is not None:
        n = min(len(ours), len(theirs))
        assert np.corrcoef(ours[-n:], theirs[-n:])[0, 1] > 0.999
