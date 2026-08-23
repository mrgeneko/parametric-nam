"""Properties tools/merge_tiers.py must have. Assembling a multi-tier SlimmableContainer is
pure JSON surgery specifically BECAUSE tiers share no weights -- these tests lock in the parts
that make that surgery safe: refusing to silently merge two products that differ in anything
but channel width, refusing a silent width collision, and recomputing max_value breakpoints
exactly the way param_train's max_values() would.

See tools/merge_tiers.py.
"""
import json

import pytest

from tools.merge_tiers import main, par_of, signature, submodels_of, width_of


def make_bare_nam(width, version="0.7.0", head_mode="skip", schema_version=1,
                   parameters=None, sample_rate=48000, weights=(0.1, 0.2), metadata=None):
    return {
        "version": version,
        "architecture": "ParametricWaveNet",
        "sample_rate": sample_rate,
        "config": {
            "layers": width,
            "parametric": {"head_mode": head_mode, "schema_version": schema_version,
                            "parameters": parameters if parameters is not None else {"Gain": {}}},
        },
        "weights": list(weights),
        "metadata": metadata if metadata is not None else {"tier": f"w{width}"},
    }


def write_nam(path, nam):
    path.write_text(json.dumps(nam))


class TestSubmodelsOf:
    def test_bare_wavenet_becomes_one_submodel_at_max_value_1(self):
        nam = make_bare_nam(3)
        subs = submodels_of(nam, "x.nam")
        assert len(subs) == 1
        assert subs[0]["max_value"] == 1.0
        assert subs[0]["model"] is nam

    def test_container_returns_its_own_submodels(self):
        sub = {"max_value": 0.5, "model": make_bare_nam(3)}
        nam = {"architecture": "SlimmableContainer", "config": {"submodels": [sub]}}
        assert submodels_of(nam, "x.nam") == [sub]

    def test_container_without_submodels_errors(self):
        nam = {"architecture": "SlimmableContainer", "config": {}}
        with pytest.raises(SystemExit):
            submodels_of(nam, "x.nam")

    def test_unsupported_architecture_errors(self):
        with pytest.raises(SystemExit):
            submodels_of({"architecture": "Something"}, "x.nam")


class TestWidthAndParOf:
    def test_width_of_reads_the_layer_count(self):
        assert width_of({"model": make_bare_nam(5)}) == 5

    def test_par_of_returns_the_parametric_config(self):
        assert par_of({"model": make_bare_nam(5)})["head_mode"] == "skip"

    def test_par_of_defaults_to_empty_dict_when_absent(self):
        nam = make_bare_nam(5)
        del nam["config"]["parametric"]
        assert par_of({"model": nam}) == {}


class TestSignature:
    def test_matches_across_tiers_differing_only_in_width(self):
        assert signature({"model": make_bare_nam(3)}) == signature({"model": make_bare_nam(8)})

    def test_differs_when_parameter_set_differs(self):
        a = signature({"model": make_bare_nam(3, parameters={"Gain": {}})})
        b = signature({"model": make_bare_nam(3, parameters={"Tone": {}})})
        assert a != b

    def test_differs_when_version_differs(self):
        a = signature({"model": make_bare_nam(3, version="0.7.0")})
        b = signature({"model": make_bare_nam(3, version="0.6.0")})
        assert a != b

    def test_differs_when_sample_rate_differs(self):
        a = signature({"model": make_bare_nam(3, sample_rate=48000)})
        b = signature({"model": make_bare_nam(3, sample_rate=44100)})
        assert a != b

    def test_differs_when_head_mode_differs(self):
        a = signature({"model": make_bare_nam(3, head_mode="skip")})
        b = signature({"model": make_bare_nam(3, head_mode="linear")})
        assert a != b


class TestMainEndToEnd:
    def test_merges_two_tiers_with_even_breakpoints(self, tmp_path, monkeypatch):
        w3, w8, out = tmp_path / "w3.nam", tmp_path / "w8.nam", tmp_path / "out.nam"
        write_nam(w3, make_bare_nam(3))
        write_nam(w8, make_bare_nam(8, metadata={"tier": "w8"}))
        monkeypatch.setattr("sys.argv", ["merge_tiers.py", str(w3), str(w8), "--out", str(out)])
        assert main() == 0

        container = json.loads(out.read_text())
        assert container["architecture"] == "SlimmableContainer"
        subs = container["config"]["submodels"]
        assert [s["model"]["config"]["layers"] for s in subs] == [3, 8]
        assert [s["max_value"] for s in subs] == [0.5, 1.0]
        assert container["metadata"] == {"tier": "w8"}  # carried from the widest tier

    def test_three_tier_breakpoints_match_param_trains_even_spacing(self, tmp_path, monkeypatch):
        paths = [tmp_path / f"w{w}.nam" for w in (3, 5, 8)]
        for p, w in zip(paths, (3, 5, 8)):
            write_nam(p, make_bare_nam(w))
        out = tmp_path / "out.nam"
        monkeypatch.setattr("sys.argv", ["merge_tiers.py", *map(str, paths), "--out", str(out)])
        assert main() == 0
        subs = json.loads(out.read_text())["config"]["submodels"]
        expected = [round((i + 1) / 3, 6) for i in range(3)]
        assert [s["max_value"] for s in subs] == expected

    def test_duplicate_width_without_replace_is_refused(self, tmp_path, monkeypatch):
        a, b = tmp_path / "a.nam", tmp_path / "b.nam"
        write_nam(a, make_bare_nam(3))
        write_nam(b, make_bare_nam(3))
        monkeypatch.setattr("sys.argv", ["merge_tiers.py", str(a), str(b), "--out", str(tmp_path / "out.nam")])
        with pytest.raises(SystemExit):
            main()

    def test_replace_lets_the_later_input_win(self, tmp_path, monkeypatch):
        a, b, out = tmp_path / "a.nam", tmp_path / "b.nam", tmp_path / "out.nam"
        write_nam(a, make_bare_nam(3, metadata={"tier": "old"}))
        write_nam(b, make_bare_nam(3, metadata={"tier": "new"}))
        monkeypatch.setattr("sys.argv", ["merge_tiers.py", str(a), str(b), "--out", str(out), "--replace"])
        assert main() == 0
        subs = json.loads(out.read_text())["config"]["submodels"]
        assert subs[0]["model"]["metadata"]["tier"] == "new"

    def test_incompatible_tiers_are_refused(self, tmp_path, monkeypatch):
        w3, w8 = tmp_path / "w3.nam", tmp_path / "w8.nam"
        write_nam(w3, make_bare_nam(3, parameters={"Gain": {}}))
        write_nam(w8, make_bare_nam(8, parameters={"Tone": {}}))
        monkeypatch.setattr("sys.argv", ["merge_tiers.py", str(w3), str(w8), "--out", str(tmp_path / "out.nam")])
        with pytest.raises(SystemExit):
            main()

    def test_non_skip_head_mode_is_refused(self, tmp_path, monkeypatch):
        w3 = tmp_path / "w3.nam"
        write_nam(w3, make_bare_nam(3, head_mode="linear"))
        monkeypatch.setattr("sys.argv", ["merge_tiers.py", str(w3), "--out", str(tmp_path / "out.nam")])
        with pytest.raises(SystemExit):
            main()

    def test_missing_weights_is_refused(self, tmp_path, monkeypatch):
        w3 = tmp_path / "w3.nam"
        write_nam(w3, make_bare_nam(3, weights=[]))
        monkeypatch.setattr("sys.argv", ["merge_tiers.py", str(w3), "--out", str(tmp_path / "out.nam")])
        with pytest.raises(SystemExit):
            main()
