"""Tests for LoRA-style low-rank knob conditioning (param_train.LoRA, A2Layer.lora).

See internal engineering notes ("LoRA-style knob conditioning") for the motivation and
the locked formula this validates against: delta_x = A(cond) @ (B(cond) @ x), l1x1 only,
zero-init net_A so LoRA-on at init is bit-identical to LoRA-off.
"""
import torch

from param_train import ParametricA2, LoRA


def _mk(channels=3, num_params=2, lora_rank=0, seed=0):
    torch.manual_seed(seed)
    m = ParametricA2(channels=channels, num_params=num_params, lora_rank=lora_rank).eval()
    # Randomize conv/mixin/l1x1/film (not lora -- see test_lora_zero_init_is_noop, which
    # depends on net_A staying exactly zero) so FiLM/LoRA actually have something to act on
    # (init is near-identity for both).
    for name, p in m.named_parameters():
        if ".lora." in name:
            continue
        p.data = p.data + 0.05 * torch.randn_like(p)
    return m


def test_lora_off_is_absent():
    """--lora-rank 0 (the default): no LoRA module exists anywhere, not just inactive."""
    m = _mk(lora_rank=0)
    assert all(layer.lora is None for layer in m.layers)


def test_lora_on_adds_module_per_layer():
    m = _mk(lora_rank=4)
    assert all(isinstance(layer.lora, LoRA) for layer in m.layers)
    assert all(layer.lora.rank == 4 for layer in m.layers)


def test_lora_zero_init_is_noop():
    """At construction, net_A is exactly zero, so delta() is exactly zero for ANY cond/x --
    enabling LoRA must never perturb a model that would otherwise train identically to
    LoRA-off. Checked directly on delta() (not by comparing two separate model
    constructions), since building a second model with lora_rank=0 would consume a
    different number of RNG draws (net_B's random init) and shift every later layer's
    conv/mixin/l1x1/film initialization relative to the lora_rank>0 model -- exactly the
    initialization-lottery confound spectral_norm's own deferred-wrapping fix (A2Layer's
    docstring) exists to avoid. Comparing delta() to an exact zero tensor sidesteps that
    entirely: it's a property of ONE model, not a diff between two.
    """
    m = _mk(lora_rank=4)
    x = torch.randn(2, 3, 128)
    cond = torch.rand(2, 2)
    for layer in m.layers:
        delta = layer.lora.delta(x, cond)
        assert torch.equal(delta, torch.zeros_like(delta))


def test_lora_on_matches_lora_off_forward_at_init():
    """End-to-end: a LoRA-enabled model's forward() output at init is bit-identical to
    the same model instance with LoRA disabled afterward (not a second construction --
    same weights, isolating LoRA's own init contribution from any RNG-order difference)."""
    m = _mk(lora_rank=4)
    x = torch.randn(1, 1, 8192)
    params = torch.tensor([[0.3, 0.7]])
    with torch.no_grad():
        y_with_lora = m(x, params)
    saved = [layer.lora for layer in m.layers]
    for layer in m.layers:
        layer.lora = None
    with torch.no_grad():
        y_without_lora = m(x, params)
    for layer, lora in zip(m.layers, saved):
        layer.lora = lora  # restore, don't leave the model mutated for other tests
    torch.testing.assert_close(y_with_lora, y_without_lora, atol=0.0, rtol=0.0)


def test_lora_effective_weight_varies_with_knob_after_training():
    """The actual property LoRA is meant to buy over FiLM: after training, l1x1's
    EFFECTIVE per-knob behavior genuinely differs with the knob (not just rescaled) --
    verify this directly, not just that the code runs.

    Trains a tiny model briefly on synthetic data whose target output depends on the
    knob in a way a fixed backbone + affine-only FiLM would need many layers to
    approximate (here: the sign of channel 0 is flipped for cond[0] > 0.5), then checks
    that LoRA's own delta() at two different knob settings is NOT just a scaled copy of
    itself -- i.e. cosine similarity between the two deltas (flattened over channels) is
    measurably less than 1, confirming a real DIRECTIONAL change, not merely a bigger/
    smaller version of the same direction (which is all a diagonal FiLM rescale could
    ever produce).
    """
    torch.manual_seed(1)
    m = ParametricA2(channels=4, num_params=1, lora_rank=4)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    T = 2048
    for step in range(60):
        x = torch.randn(8, 1, T)
        cond = torch.rand(8, 1)
        with torch.no_grad():
            target = torch.where(cond.view(-1, 1, 1) > 0.5, x, -x)
        y = m(x, cond)
        loss = torch.nn.functional.mse_loss(y, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    m.eval()
    x = torch.randn(1, 4, 256)
    cond_lo = torch.tensor([[0.1]])
    cond_hi = torch.tensor([[0.9]])
    layer = m.layers[0]
    with torch.no_grad():
        d_lo = layer.lora.delta(x, cond_lo).flatten()
        d_hi = layer.lora.delta(x, cond_hi).flatten()
    assert d_lo.abs().sum() > 1e-6, "LoRA delta didn't move at all during training"
    cos_sim = torch.nn.functional.cosine_similarity(d_lo, d_hi, dim=0).item()
    assert cos_sim < 0.999, (
        f"LoRA delta at two different knob settings is essentially the same direction "
        f"(cos_sim={cos_sim:.6f}) -- expected a real directional change, not just scaling."
    )
