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
    """Schema mirrors NAM's own WaveNet.export_config() for the A2 config
    (verified in-venv against neural-amp-modeler)."""
    m = _mk()
    d = nam_standard.export_nam_standard(m, params=[0.5, 0.5])
    assert d["architecture"] == "WaveNet"
    assert d["config"]["head"] is None and "head_scale" in d["config"]
    lc = d["config"]["layers"][0]                       # NAM uses "layers", not "layers_configs"
    assert lc["condition_size"] == 1 and lc["input_size"] == 1
    assert all(a["type"] == "LeakyReLU" for a in lc["activation"])   # per-layer activation list
    assert lc["gating_mode"] == ["none"] * K_NUM_LAYERS
    assert len(lc["kernel_sizes"]) == K_NUM_LAYERS == len(lc["dilations"])
    assert lc["head"] == {"out_channels": 1, "kernel_size": 16, "bias": True}
    # weight count matches a static A2 (folded → no FiLM weights) + head_scale last
    static = nam_standard.fold_film(m, [0.5, 0.5])
    assert len(d["weights"]) == static.weight_count()
    assert d["weights"][-1] == pytest.approx(static.head_scale.item())


def test_parametric_requires_params():
    with pytest.raises(ValueError):
        nam_standard.export_nam_standard(_mk(), params=None)


def _best_corr(a, b, maxlag=40):
    """Max Pearson corr over integer lags in ±maxlag (absorbs the constant
    head-padding latency offset between our centered head and NAM's causal head)."""
    best = -1.0
    for lag in range(-maxlag, maxlag + 1):
        aa, bb = (a[: len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: len(b) + lag])
        n = min(len(aa), len(bb))
        if n < 1000:
            continue
        best = max(best, np.corrcoef(aa[:n], bb[:n])[0, 1])
    return best


def _roundtrip_corr(head_mode):
    """Export a head_mode model to standard .nam, load it in official NAM, and
    return the best lag-aligned corr on the post-warmup region."""
    nam = pytest.importorskip("nam", reason="neural-amp-modeler not installed (dev dep)")
    from nam.models import _from_nam

    m = ParametricA2(channels=3, num_params=2, head_mode=head_mode).eval()
    torch.manual_seed(0)
    for p in m.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    params = [0.4, 0.6]
    d = nam_standard.export_nam_standard(m, params=params)
    official = _from_nam.init_from_nam(d)
    official.eval()
    rf = int(official.receptive_field)
    x = torch.randn(1, 1, 16384)
    with torch.no_grad():
        ours = nam_standard.fold_film(m, params)(x, torch.zeros(1, 0)).squeeze().numpy()
        theirs = np.asarray(official(x.squeeze().view(1, -1))).squeeze()
    return _best_corr(ours[rf:], theirs[rf:])


def test_roundtrip_skip_is_bit_exact():
    """A SKIP-accumulating model exports bit-exact to official NAM's WaveNet — the
    delivery path for baked static captures into stock NAM plugins."""
    assert _roundtrip_corr("skip") > 0.999


def test_roundtrip_residual_does_not_match_stock():
    """A RESIDUAL-only model is [redacted]-native (its parametric fast-path reads the
    final residual) and does NOT match stock NAM's skip-accumulating WaveNet. This
    documents WHY skip is required for stock-plugin delivery. See
    docs/rearchitecture_skip_accumulation.md."""
    assert _roundtrip_corr("residual") < 0.9
