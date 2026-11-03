"""Tests for the knob-conditioning encoder (param_train.CondEncoder, --cond-hidden).

The claim it validates: with an encoder, every FiLM layer consumes cond = [knobs,
tanh(mlp(knobs))] instead of the raw knobs, and every path that hands FiLM a cond vector
(forward, flat-weight export/reload, fold_film) agrees bit-for-bit on what that vector is.
"""
import torch

from param_train import (ParametricA2, SlimmableParametricA2, CondEncoder,
                         K_PARAM_SCHEMA_VERSION, K_PARAM_SCHEMA_VERSION_COND,
                         verify_export_round_trip)
import nam_standard


def _mk(channels=3, num_params=2, cond_hidden=0, seed=0):
    torch.manual_seed(seed)
    m = ParametricA2(channels=channels, num_params=num_params, cond_hidden=cond_hidden).eval()
    for p in m.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    return m


def test_cond_off_is_absent():
    m = _mk(cond_hidden=0)
    assert m.cond_enc is None
    assert all(layer.film.net.in_features == 2 for layer in m.layers)
    params = torch.rand(4, 2)
    assert torch.equal(m.cond(params), params)


def test_cond_on_widens_every_film_layer():
    m = _mk(cond_hidden=8)
    assert isinstance(m.cond_enc, CondEncoder)
    assert m.cond_enc.out_dim == 2 + 8
    assert all(layer.film.net.in_features == 10 for layer in m.layers)


def test_cond_concat_keeps_raw_knobs_first():
    """CONCAT, not replace: columns 0..num_params-1 of cond ARE the knobs (what
    --knob-boost's column hooks rely on), and the added features are tanh-bounded."""
    m = _mk(cond_hidden=8)
    params = torch.rand(5, 2)
    cond = m.cond(params)
    assert cond.shape == (5, 10)
    assert torch.equal(cond[:, :2], params)
    assert cond[:, 2:].abs().max() <= 1.0


def test_cond_is_nonlinear_in_knob_position():
    """The whole point: gamma/beta are no longer affine in the knobs. Check on the encoder
    output directly -- the midpoint of two knob settings must NOT map to the midpoint of
    their encodings (it would for any affine map)."""
    m = _mk(cond_hidden=8, seed=3)
    a = torch.tensor([[0.1, 0.1]])
    b = torch.tensor([[0.9, 0.9]])
    mid = (a + b) / 2
    enc_mid = m.cond(mid)[:, 2:]
    enc_avg = (m.cond(a)[:, 2:] + m.cond(b)[:, 2:]) / 2
    assert not torch.allclose(enc_mid, enc_avg, atol=1e-4)


def test_export_reload_round_trip():
    """Flat weight list (encoder first, control path before audio path) reloads into a
    fresh model built with the same cond_hidden and reproduces forward() exactly."""
    m = _mk(cond_hidden=8, seed=1)
    w = m.export_weights()
    assert len(w) == m.weight_count()
    m2 = ParametricA2(channels=3, num_params=2, cond_hidden=8).eval()
    m2.load_weights(w)
    x = torch.randn(1, 1, 4096)
    p = torch.tensor([[0.3, 0.8]])
    with torch.no_grad():
        torch.testing.assert_close(m(x, p), m2(x, p), atol=0.0, rtol=0.0)


def test_round_trip_without_cond_hidden_is_wrong():
    """Guards the failure mode cond_hidden threading prevents: a reader that forgets the
    encoder consumes its weights as rechannel/conv weights and plays a different model.
    verify_export_round_trip threads it from the source submodel, so it must pass."""
    torch.manual_seed(2)
    sm = SlimmableParametricA2(num_params=2, widths=[3, 4], cond_hidden=8).eval()
    for p in sm.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    nam = sm.export_nam({"param_names": ["A", "B"]}, {"version": "0.7.0"}, 48000)
    for lbl, diff in verify_export_round_trip(nam, sm, 2, "cpu"):
        assert diff == 0.0, (lbl, diff)
    # And a naive reader (no cond_hidden) must NOT silently reproduce it.
    bad = ParametricA2(3, 2).eval()
    try:
        bad.load_weights(nam["config"]["submodels"][0]["model"]["weights"])
    except (StopIteration, RuntimeError):
        return   # desync caught loudly -- acceptable
    x = torch.randn(1, 1, 4096)
    p = torch.tensor([[0.3, 0.8]])
    with torch.no_grad():
        assert not torch.allclose(bad(x, p), sm.submodels[0](x, p))


def test_export_declares_schema_3_only_with_encoder():
    plain = _mk(cond_hidden=0)
    enc = _mk(cond_hidden=8)
    cfg = {"param_names": ["A", "B"]}
    par_plain = plain.export_nam(cfg, {"version": "0.7.0"}, 48000)["config"]["parametric"]
    par_enc = enc.export_nam(cfg, {"version": "0.7.0"}, 48000)["config"]["parametric"]
    assert par_plain["schema_version"] == K_PARAM_SCHEMA_VERSION
    assert par_plain["type"] == "film" and "cond_encoder" not in par_plain
    assert par_enc["schema_version"] == K_PARAM_SCHEMA_VERSION_COND
    assert par_enc["type"] == "film+cond"
    ce = par_enc["cond_encoder"]
    assert ce["hidden"] == 8 and ce["depth"] == 2 and ce["cond_size"] == 10
    assert ce["output"] == "concat"


def test_fold_film_goes_through_encoder():
    """fold_film at a knob setting must equal the live parametric model at that setting --
    which only holds if the fold encodes the knobs the same way forward() does."""
    m = _mk(cond_hidden=8, seed=4)
    x = torch.randn(1, 1, 4096)
    knobs = [0.25, 0.9]
    static = nam_standard.fold_film(m, knobs)
    assert static.cond_enc is None and static.num_params == 0
    with torch.no_grad():
        live = m(x, torch.tensor([knobs]))
        baked = static(x, torch.zeros(1, 0))
    torch.testing.assert_close(live, baked, atol=1e-5, rtol=1e-5)


def test_spectral_norm_bake_keeps_encoder():
    m = ParametricA2(3, 2, spectral_norm=True, cond_hidden=8).eval()
    for n, p in m.named_parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    baked = m._bake_spectral_norm()
    assert baked.cond_enc is not None
    x = torch.randn(1, 1, 4096)
    p = torch.tensor([[0.6, 0.2]])
    with torch.no_grad():
        torch.testing.assert_close(m(x, p), baked(x, p), atol=1e-6, rtol=1e-6)
