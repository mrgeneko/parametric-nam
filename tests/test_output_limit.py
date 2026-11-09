"""Tests for the output limiter (param_train.soft_limit, --output-limit, export schema 5).

Claims: identity (value and slope) up to the knee, strictly bounded by the ceiling above it,
bounded whatever the network does, and export -> reload agrees with the live model.
"""
import warnings

import pytest
import torch

import nam_standard
from param_train import (ParametricA2, SlimmableParametricA2, K_PARAM_SCHEMA_VERSION_BOUND,
                         K_PARAM_SCHEMA_VERSION_LIMIT, parametric_ctor_kwargs, soft_limit)

KNEE, CEIL = 0.9, 0.99


def test_identity_up_to_the_knee_both_signs():
    y = torch.linspace(-KNEE, KNEE, 101)
    assert torch.equal(soft_limit(y, KNEE, CEIL), y)


def test_saturates_below_the_ceiling_and_is_monotonic():
    y = torch.linspace(-1e6, 1e6, 20001)
    out = soft_limit(y, KNEE, CEIL)
    assert out.abs().max() <= CEIL
    assert out.abs().max() > CEIL - 1e-4          # actually reaches it
    assert (out[1:] >= out[:-1]).all()


def test_slope_is_one_at_the_knee():
    y = torch.tensor([KNEE - 1e-3, KNEE + 1e-3], requires_grad=True)
    soft_limit(y, KNEE, CEIL).sum().backward()
    assert torch.allclose(y.grad, torch.ones(2), atol=2e-3)


def test_disabled_is_a_no_op():
    y = torch.randn(50) * 10
    assert soft_limit(y, 0.0, 0.0) is y


def test_bad_limiter_args_are_refused():
    with pytest.raises(ValueError):
        ParametricA2(3, 2, out_knee=1.0, out_ceiling=0.5)
    with pytest.raises(ValueError):
        ParametricA2(3, 2, out_knee=0.0, out_ceiling=0.5)


def _mk(**kw):
    torch.manual_seed(5)
    m = ParametricA2(3, 2, **kw).eval()
    for p in m.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    return m


def test_runaway_network_stays_bounded():
    m = _mk(out_knee=KNEE, out_ceiling=CEIL)
    with torch.no_grad():
        m.head_scale.mul_(1e6)
        out = m(torch.randn(1, 1, 3000), torch.tensor([[0.3, 0.8]]))
    assert out.abs().max() <= CEIL


def test_in_range_audio_is_untouched():
    plain = _mk()
    lim = _mk(out_knee=1e3, out_ceiling=1.1e3)        # knee far above any output
    x = torch.randn(1, 1, 3000) * 0.3
    p = torch.tensor([[0.3, 0.8]])
    with torch.no_grad():
        assert torch.equal(plain(x, p), lim(x, p))


def test_export_declares_schema_5_only_with_a_limiter():
    cfg = {"param_names": ["A", "B"]}
    off = _mk().export_nam(cfg, {"version": "0.7.0"}, 48000)["config"]["parametric"]
    bound = _mk(gamma_bound=4.0).export_nam(cfg, {"version": "0.7.0"}, 48000)["config"]["parametric"]
    on = _mk(gamma_bound=4.0, out_knee=KNEE, out_ceiling=CEIL).export_nam(
        cfg, {"version": "0.7.0"}, 48000)["config"]["parametric"]
    assert "output_limit" not in off and "output_limit" not in bound
    assert bound["schema_version"] == K_PARAM_SCHEMA_VERSION_BOUND
    assert on["schema_version"] == K_PARAM_SCHEMA_VERSION_LIMIT
    assert on["output_limit"] == {"knee": KNEE, "ceiling": CEIL}
    assert on["type"].endswith("+bound+limit")


def test_limited_model_round_trips_through_export():
    m = _mk(cond_hidden=4, gamma_bound=4.0, out_knee=0.05, out_ceiling=0.08)  # knee inside range
    d = m.export_nam({"param_names": ["a", "b"], "bounds": {"a": [0, 1], "b": [0, 1]}},
                     {"version": "0.7.0"}, 48000)
    kw = parametric_ctor_kwargs(d["config"]["parametric"])
    assert kw["out_knee"] == 0.05 and kw["out_ceiling"] == 0.08
    m2 = ParametricA2(3, 2, **kw)
    m2.load_weights(d["weights"]); m2.eval()
    x = torch.randn(1, 1, 2000) * 3.0
    p = torch.tensor([[0.3, 0.8]])
    with torch.no_grad():
        a, b = m(x, p), m2(x, p)
    assert a.abs().max() > 0.05                        # the limiter was actually exercised
    assert torch.allclose(a, b, atol=1e-5)


def test_slimmable_passes_the_limiter_to_every_tier():
    s = SlimmableParametricA2(2, widths=[3, 4], out_knee=KNEE, out_ceiling=CEIL)
    assert all(t.out_knee == KNEE and t.out_ceiling == CEIL for t in s.submodels)


def test_standard_nam_export_warns_that_the_limiter_cannot_be_carried():
    m = _mk(out_knee=KNEE, out_ceiling=CEIL)
    static = nam_standard.fold_film(m, [0.3, 0.8])
    with pytest.warns(UserWarning, match="output limiter"):
        nam_standard._wavenet_dict(static)


def test_training_gradient_survives_a_far_saturated_output():
    """Run F's full tier died because a tier whose raw output starts far above the knee got ~0
    gradient from the tanh. In training mode the limiter is straight-through: same forward value,
    identity backward."""
    y = torch.full((4,), 5.0, requires_grad=True)
    soft_limit(y, KNEE, CEIL).sum().backward()
    assert y.grad.abs().max() < 1e-6                           # the plain limiter really is dead there
    y2 = torch.full((4,), 5.0, requires_grad=True)
    out = soft_limit(y2, KNEE, CEIL, ste=True)
    assert torch.allclose(out, soft_limit(torch.full((4,), 5.0), KNEE, CEIL), atol=1e-6)   # forward unchanged
    out.sum().backward()
    assert torch.equal(y2.grad, torch.ones(4))


def test_model_uses_the_straight_through_gradient_only_while_training():
    m = _mk(out_knee=KNEE, out_ceiling=CEIL)
    with torch.no_grad():
        m.head_scale.mul_(1e4)                                 # raw output far above the knee
    x, p = torch.randn(1, 1, 500), torch.tensor([[0.3, 0.8]])
    m.train(); m(x, p).sum().backward()
    assert m.head.weight.grad.abs().max() > 0                  # trainable despite saturation
    m.eval()
    with torch.no_grad():
        assert m(x, p).abs().max() <= CEIL
