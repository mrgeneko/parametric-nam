"""Wires verify_lora_cpp_parity.py into the test suite (LoRA plan Phase 5).

This is the check that actually proves Python and C++ agree on the LoRA math, not just that
each side's own test suite passes in isolation -- see the module docstring on
verify_lora_cpp_parity.py for the full rationale and the two C++-vs-Python behavioral
differences (knob smoothing, DC blocker) it compensates for.

Needs NeuralAmpModelerCore's run_tests target already built (produces render_parametric) --
skips cleanly (not a failure) if that's not present in this environment, the same way
test_nam_standard.py's official-NAM round-trip test skips when the optional `nam` package
isn't installed. Run manually to build the prerequisite:
    cd ~/work/chainsmith/NeuralAmpModelerCore/build && cmake --build . --target run_tests
"""
from pathlib import Path

import pytest

from verify_lora_cpp_parity import find_render_parametric, run_parity_check

NAM_CORE = Path.home() / "work/chainsmith/NeuralAmpModelerCore"


def _require_render_parametric():
    render_bin, _ = find_render_parametric(NAM_CORE)
    if render_bin is None:
        pytest.skip(f"render_parametric not built under {NAM_CORE} -- run "
                   f"'cmake --build . --target run_tests' in its build/ dir to enable "
                   f"this cross-repo check")


def test_parity_lora_disabled_matches_fast_path():
    """rank=0 (FiLM-only, takes the fast path per Phase 4's gate) -- confirms the harness
    itself and the pre-existing (non-LoRA) path still agree, i.e. this isn't a check that
    only ever exercises the generic path."""
    _require_render_parametric()
    run_parity_check(NAM_CORE, rank=0, channels=3)


def test_parity_lora_enabled_matches_generic_path():
    """rank=2 (FiLM+LoRA, excluded from the fast path -- takes the generic path)."""
    _require_render_parametric()
    run_parity_check(NAM_CORE, rank=2, channels=3)


def test_parity_lora_enabled_full_width():
    """A second rank/width combination (rank=4, channels=8 -- the 'full' tier shape) so
    this isn't only ever validated at the smallest A2 topology."""
    _require_render_parametric()
    run_parity_check(NAM_CORE, rank=4, channels=8)
