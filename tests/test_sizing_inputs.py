"""sizing_inputs: what an excitation was sized against, and how a recorded block compares."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sizing_inputs as si  # noqa: E402

GRID = {"Gain": [0.1, 0.5, 1.0], "Tone": [0.2, 0.8]}
FIXED = {"Level": 0.5}


def rec(**over):
    kw = dict(identity=b"<schx v1/>", circuit_kind="schx", knob_ranges=GRID, fixed=FIXED,
              sample_grid=9, conditions="os=8|it=256|solver=x")
    kw.update(over)
    return si.inputs_record(**kw)


def cmp(recorded, **over):
    kw = dict(circuit_sha256=si.inputs_record(identity=b"<schx v1/>", circuit_kind="schx", knob_ranges={},
                                              fixed={}, sample_grid=0, conditions="")["circuit"]["sha256"],
              knob_ranges=GRID, fixed=FIXED)
    kw.update(over)
    return si.compare(recorded, **kw)


class TestRecord:
    def test_shape(self):
        r = rec()
        assert r["schema"] == si.SCHEMA and r["circuit"]["kind"] == "schx"
        assert len(r["circuit"]["sha256"]) == 64
        assert r["grid"] == GRID and r["fixed"] == FIXED and r["sample_grid"] == 9
        assert r["conditions"] == "os=8|it=256|solver=x"

    def test_circuit_hash_follows_content_not_a_path(self):
        assert rec(identity=b"a")["circuit"]["sha256"] != rec(identity=b"b")["circuit"]["sha256"]
        assert rec(identity=b"a")["circuit"]["sha256"] == rec(identity=b"a")["circuit"]["sha256"]

    def test_is_json_serialisable(self):
        import json
        json.dumps(rec())


class TestCompare:
    def test_match(self):
        assert cmp(rec())["status"] == "match"

    def test_no_recorded_block_is_unrecorded_not_stale(self):
        assert cmp(None)["status"] == "unrecorded"
        assert cmp({})["status"] == "unrecorded"

    def test_unknown_schema_is_unrecorded(self):
        r = rec(); r["schema"] = 999
        assert cmp(r)["status"] == "unrecorded"

    def test_interior_grid_edit_is_stale_because_interior_points_are_probed(self):
        g = {**GRID, "Gain": [0.1, 0.25, 0.5, 1.0]}
        c = cmp(rec(), knob_ranges=g)
        assert c["status"] == "stale" and "Gain" in c["reasons"][0]

    def test_extreme_grid_edit_is_stale(self):
        assert cmp(rec(), knob_ranges={**GRID, "Tone": [0.2, 0.9]})["status"] == "stale"

    def test_added_and_removed_knobs_are_stale(self):
        added = cmp(rec(), knob_ranges={**GRID, "Bass": [0.0, 1.0]})
        removed = cmp(rec(), knob_ranges={"Gain": GRID["Gain"]})
        assert added["status"] == "stale" and "added" in added["reasons"][0]
        assert removed["status"] == "stale" and "no longer" in removed["reasons"][0]

    def test_grid_value_order_matters_because_mid_is_an_index(self):
        assert cmp(rec(), knob_ranges={**GRID, "Gain": [1.0, 0.5, 0.1]})["status"] == "stale"

    def test_fixed_param_change_is_stale(self):
        c = cmp(rec(), fixed={"Level": 0.6})
        assert c["status"] == "stale" and "Level" in c["reasons"][0]

    def test_circuit_change_alone_is_advisory_not_stale(self):
        assert cmp(rec(), circuit_sha256="0" * 64)["status"] == "circuit-differs"

    def test_grid_change_outranks_circuit_change(self):
        assert cmp(rec(), circuit_sha256="0" * 64, knob_ranges={"Gain": [0.0, 1.0]})["status"] == "stale"

    def test_unknown_current_circuit_hash_is_not_a_difference(self):
        assert cmp(rec(), circuit_sha256=None)["status"] == "match"


class TestParsers:
    def test_ranges(self):
        assert si.parse_ranges(["Gain=0.1,0.5", "Tone = 0.2,0.8"]) == {"Gain": [0.1, 0.5], "Tone": [0.2, 0.8]}
        assert si.parse_ranges(None) == {}

    def test_fixed(self):
        assert si.parse_fixed("A=0.5, B=1") == {"A": 0.5, "B": 1.0}
        assert si.parse_fixed(None) == {} and si.parse_fixed("") == {}
