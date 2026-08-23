"""Properties tools/preflight.py's _build_backend dispatch must have. A wrong knob list here
(e.g. forgetting --exclude-knob) means preflight silently checks the WRONG set of knobs against
a real device -- the exact class of gap this whole tool exists to close. The livespice path
(parse_schx_controls/resolve_knobs against a real .schx) is exercised functionally elsewhere in
this repo's own manual verification; this file covers the ngspice-deck path and the shared
unknown-backend guard, which need no .schx fixture.

See tools/preflight.py.
"""
import types
from pathlib import Path

import pytest

from tools.preflight import _build_backend
from tools.render_backends import NgspiceBackend


def write_fake_pedal_module(tmp_path, name, knob_names=("Gain", "Tone", "Volume")):
    mod_path = tmp_path / f"{name}.py"
    mod_path.write_text(
        f"KNOB_NAMES = {list(knob_names)!r}\n"
        "def build_deck(input_src, knobs):\n"
        "    return ''\n"
    )
    return mod_path


def ngspice_args(pedal_dir=None, module=None, exclude_knob=(), probe_node="OUT",
                  maxstep=3e-6, parallel_sims=8, peak_max_v=40.0):
    return types.SimpleNamespace(backend="ngspice-deck", pedal_dir=pedal_dir, module=module,
                                  exclude_knob=list(exclude_knob), probe_node=probe_node,
                                  maxstep=maxstep, parallel_sims=parallel_sims, peak_max_v=peak_max_v)


class TestNgspiceDeckBackend:
    def test_builds_an_ngspice_backend_with_all_knobs(self, tmp_path):
        mod_path = write_fake_pedal_module(tmp_path, "pedal_x")
        backend, knobs, identity, cache_extra = _build_backend(
            ngspice_args(pedal_dir=str(tmp_path), module="pedal_x"))
        assert isinstance(backend, NgspiceBackend)
        assert knobs == ["Gain", "Tone", "Volume"]
        assert identity == mod_path.read_bytes()

    def test_exclude_knob_removes_it_from_the_swept_set(self, tmp_path):
        write_fake_pedal_module(tmp_path, "pedal_y")
        _, knobs, _, _ = _build_backend(
            ngspice_args(pedal_dir=str(tmp_path), module="pedal_y", exclude_knob=["Volume"]))
        assert knobs == ["Gain", "Tone"]
        assert "Volume" not in knobs

    def test_probe_node_is_passed_through_to_the_backend(self, tmp_path):
        write_fake_pedal_module(tmp_path, "pedal_z")
        backend, _, _, _ = _build_backend(
            ngspice_args(pedal_dir=str(tmp_path), module="pedal_z", probe_node="spk"))
        assert backend.probe_node == "spk"

    def test_cache_extra_reflects_peak_max_v(self, tmp_path):
        write_fake_pedal_module(tmp_path, "pedal_w")
        _, _, _, cache_extra = _build_backend(
            ngspice_args(pedal_dir=str(tmp_path), module="pedal_w", peak_max_v=5.0))
        assert "maxv=5.0" in cache_extra

    def test_missing_pedal_dir_or_module_exits(self):
        with pytest.raises(SystemExit):
            _build_backend(ngspice_args(pedal_dir=None, module=None))


class TestUnknownBackend:
    def test_unknown_backend_exits(self):
        args = types.SimpleNamespace(backend="totally-not-a-backend")
        with pytest.raises(SystemExit):
            _build_backend(args)
