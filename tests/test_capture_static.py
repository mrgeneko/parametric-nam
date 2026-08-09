"""Tests for capture_static.py's patch_input_level_dbu -- the static-capture side of
input_level_dbu population. nam-full's own export (nam/train/full.py's main(), calling
model.net.export()/.export_container() with no metadata argument at all) has no
mechanism to set this, unlike param_train.py's export_nam() which derives and writes it
directly -- so this has to be patched into the already-exported .nam after the fact,
using the identical derivation (_schx_input_v0dbfs + _input_level_dbu).
"""
import json
from pathlib import Path

import capture_static

# A minimal schx with a Circuit.Input carrying V0dBFS -- just enough structure for
# _schx_input_v0dbfs (iterates <Element><Component _Type=... V0dBFS=.../></Element>).
_SCHX_WITH_V0DBFS = """<?xml version="1.0"?>
<Schematic>
  <Element>
    <Component _Type="Circuit.Input, Circuit" Name="Input" V0dBFS="1 V" />
  </Element>
</Schematic>
"""

_SCHX_NO_INPUT = """<?xml version="1.0"?>
<Schematic>
  <Element>
    <Component _Type="Circuit.Resistor, Circuit" Name="R1" Resistance="1 kOhm" />
  </Element>
</Schematic>
"""


def _write_schx(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.schx"
    p.write_text(content)
    return str(p)


def _container_nam(tmp_path: Path) -> Path:
    """A 2-submodel SlimmableContainer .nam, matching what nam-full's export_container
    produces -- each submodel carries its OWN metadata block (date/loudness/gain), same
    as a real static capture, and none of them have input_level_dbu yet."""
    d = {
        "version": "0.7.0",
        "architecture": "SlimmableContainer",
        "config": {"submodels": [
            {"max_value": 0.5, "model": {"architecture": "WaveNet",
             "metadata": {"date": {"year": 2026}, "loudness": -20.0, "gain": 0.5}}},
            {"max_value": 1.0, "model": {"architecture": "WaveNet",
             "metadata": {"date": {"year": 2026}, "loudness": -19.5, "gain": 0.4}}},
        ]},
        "weights": [1, 2, 3],
        "metadata": {"date": {"year": 2026}, "loudness": -20.0, "gain": 0.5},
        "sample_rate": 48000,
    }
    p = tmp_path / "model.nam"
    p.write_text(json.dumps(d))
    return p


def test_patch_input_level_dbu_derives_from_schx_v0dbfs(tmp_path):
    schx = _write_schx(tmp_path, _SCHX_WITH_V0DBFS)
    nam_path = _container_nam(tmp_path)

    result = capture_static.patch_input_level_dbu(nam_path, schx)

    # V0dBFS=1V is the ecosystem's standard convention -- matches the ~-0.79 dBu figure
    # documented in param_train.py's own module comment (_DBU_0_RMS_VOLTS derivation).
    assert result == capture_static._input_level_dbu(1.0)
    assert -1.0 < result < -0.5


def test_patch_input_level_dbu_sets_every_metadata_block_not_just_top_level(tmp_path):
    """A container's per-submodel metadata must ALSO get input_level_dbu -- it's a
    property of the schx's input stage, not of any one tier, and param_train.py's own
    export_composite_nam writes the identical value into both (verified against a real
    published .nam earlier in this work). Missing this was the exact gap this test
    guards against."""
    schx = _write_schx(tmp_path, _SCHX_WITH_V0DBFS)
    nam_path = _container_nam(tmp_path)

    capture_static.patch_input_level_dbu(nam_path, schx)

    d = json.loads(nam_path.read_text())
    assert d["metadata"]["input_level_dbu"] is not None
    for sub in d["config"]["submodels"]:
        assert sub["model"]["metadata"]["input_level_dbu"] == d["metadata"]["input_level_dbu"]


def test_patch_input_level_dbu_omits_when_schx_has_no_v0dbfs(tmp_path):
    schx = _write_schx(tmp_path, _SCHX_NO_INPUT)
    nam_path = _container_nam(tmp_path)

    result = capture_static.patch_input_level_dbu(nam_path, schx)

    assert result is None
    d = json.loads(nam_path.read_text())
    assert d["metadata"]["input_level_dbu"] is None
    for sub in d["config"]["submodels"]:
        assert sub["model"]["metadata"]["input_level_dbu"] is None


def test_patch_input_level_dbu_leaves_weights_and_architecture_untouched(tmp_path):
    schx = _write_schx(tmp_path, _SCHX_WITH_V0DBFS)
    nam_path = _container_nam(tmp_path)
    before = json.loads(nam_path.read_text())

    capture_static.patch_input_level_dbu(nam_path, schx)

    after = json.loads(nam_path.read_text())
    assert after["weights"] == before["weights"]
    assert after["architecture"] == before["architecture"]
    assert after["sample_rate"] == before["sample_rate"]
