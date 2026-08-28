"""Properties render_ngspice_deck.py's CLI orchestration must have. It replaced 6
near-identical per-device render scripts (see module docstring); the only genuinely
device-specific logic left is knob-name validation and the grid/manifest/mapping.csv
bookkeeping around ngspice_spicelib's load_input/render_grid/render_one, which are stubbed
out here so this file's own argument handling and output-writing logic is tested directly.

See render_ngspice_deck.py.
"""
import json
import sys

import pytest

from render_ngspice_deck import main


def write_fake_pedal_module(tmp_path, name, knob_names=("Gain", "Tone")):
    mod_path = tmp_path / f"{name}.py"
    mod_path.write_text(
        f"KNOB_NAMES = {list(knob_names)!r}\n"
        "def build_deck(input_src, knobs):\n"
        "    return ''\n"
    )
    return tmp_path


def common_args(tmp_path, module, infile="in.wav", extra=()):
    return ["render_ngspice_deck.py", "--pedal-dir", str(tmp_path), "--module", module,
            "--probe-node", "OUT", infile, *extra]


class TestKnobValidation:
    def test_unknown_knob_name_exits(self, tmp_path, monkeypatch):
        write_fake_pedal_module(tmp_path, "pedal_a")
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_a", extra=[
            "out.wav", "--knob", "Bogus=0.5"]))
        with pytest.raises(SystemExit):
            main()


class TestSingleRender:
    def test_reports_ok_when_peak_under_threshold(self, tmp_path, monkeypatch, capsys):
        write_fake_pedal_module(tmp_path, "pedal_b")
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))
        monkeypatch.setattr("render_ngspice_deck.render_one", lambda *a, **kw: 1.2)
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_b", extra=[
            "out.wav", "--knob", "Gain=0.5"]))
        main()
        assert "OK" in capsys.readouterr().out

    def test_reports_failed_when_render_does_not_converge(self, tmp_path, monkeypatch, capsys):
        write_fake_pedal_module(tmp_path, "pedal_c")
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))
        monkeypatch.setattr("render_ngspice_deck.render_one", lambda *a, **kw: None)
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_c", extra=["out.wav"]))
        main()
        assert "FAILED" in capsys.readouterr().out

    def test_peak_above_ok_max_peak_is_reported_failed(self, tmp_path, monkeypatch, capsys):
        write_fake_pedal_module(tmp_path, "pedal_d")
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))
        monkeypatch.setattr("render_ngspice_deck.render_one", lambda *a, **kw: 999.0)
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_d", extra=["out.wav"]))
        main()
        assert "FAILED" in capsys.readouterr().out


class TestGridRequiresOutdir:
    def test_grid_without_outdir_raises(self, tmp_path, monkeypatch):
        write_fake_pedal_module(tmp_path, "pedal_e")
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_e", extra=[
            "--grid", "Gain=0.1,0.5,0.9"]))
        with pytest.raises(AssertionError):
            main()


class TestGridManifestAndMapping:
    def test_writes_a_manifest_row_per_combo_and_a_cartesian_product_count(self, tmp_path, monkeypatch):
        write_fake_pedal_module(tmp_path, "pedal_f")
        outdir = tmp_path / "caps"
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))

        def fake_render_grid(build_deck, jobs, probe_node, sr, t, input_src, tmp, maxstep, parallel_sims):
            return {outfile: 1.0 for _knobs, outfile in jobs}
        monkeypatch.setattr("render_ngspice_deck.render_grid", fake_render_grid)
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_f", extra=[
            "--grid", "Gain=0.1,0.5,0.9", "Tone=0.2,0.8", "--outdir", str(outdir)]))
        main()

        manifest = [json.loads(l) for l in (outdir / "manifest.jsonl").read_text().splitlines()]
        assert len(manifest) == 3 * 2  # cartesian product of the two grids
        assert all(row["ok"] for row in manifest)

    def test_mapping_csv_excludes_failed_renders(self, tmp_path, monkeypatch):
        write_fake_pedal_module(tmp_path, "pedal_g")
        outdir = tmp_path / "caps"

        def fake_render_grid(build_deck, jobs, probe_node, sr, t, input_src, tmp, maxstep, parallel_sims):
            # first job "fails" (None peak), second "converges"
            return {jobs[0][1]: None, jobs[1][1]: 1.0}
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))
        monkeypatch.setattr("render_ngspice_deck.render_grid", fake_render_grid)
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_g", extra=[
            "--grid", "Gain=0.1,0.9", "--outdir", str(outdir)]))
        main()

        manifest = [json.loads(l) for l in (outdir / "manifest.jsonl").read_text().splitlines()]
        assert sum(1 for r in manifest if not r["ok"]) == 1

        mapping_lines = (outdir / "mapping.csv").read_text().strip().splitlines()
        assert len(mapping_lines) == 1 + 1  # header + only the converged row

    def test_peak_at_or_above_ok_max_peak_is_marked_not_ok(self, tmp_path, monkeypatch):
        write_fake_pedal_module(tmp_path, "pedal_h")
        outdir = tmp_path / "caps"

        def fake_render_grid(build_deck, jobs, probe_node, sr, t, input_src, tmp, maxstep, parallel_sims):
            return {jobs[0][1]: 999.0}
        monkeypatch.setattr("render_ngspice_deck.load_input",
                             lambda *a, **kw: (1000, [0] * 10, "src"))
        monkeypatch.setattr("render_ngspice_deck.render_grid", fake_render_grid)
        monkeypatch.setattr(sys, "argv", common_args(tmp_path, "pedal_h", extra=[
            "--grid", "Gain=0.5", "--outdir", str(outdir)]))
        main()
        manifest = [json.loads(l) for l in (outdir / "manifest.jsonl").read_text().splitlines()]
        assert manifest[0]["ok"] is False
