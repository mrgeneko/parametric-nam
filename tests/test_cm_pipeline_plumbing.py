"""--backend cm through the pipeline around gen_dataset_from_schx.py: the gate commands, the fleet dispatcher's config expansion,
the sizing and coverage probes' backend, and the choices every tool offers. No renderer is run."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import gate_config  # noqa: E402


def test_gate_runs_transient_and_preflight_for_cm():
    assert "cm" in gate_config.TRANSIENT_BACKENDS and "cm" in gate_config.PREFLIGHT_BACKENDS
    cfg = {"backend": "cm", "schx": "/x/amp.schx", "input": "/x/in.wav", "knobs": "Gain,Tone"}
    cmd, why = gate_config.preflight_command(cfg)
    assert why == "" and cmd[cmd.index("--backend") + 1] == "cm" and cmd[cmd.index("--schx") + 1] == "/x/amp.schx"
    assert cmd[cmd.index("--knobs") + 1] == "Gain,Tone"
    cmd, why = gate_config.transient_command(cfg, Path("/x/amp.config.toml"))
    assert cmd is not None and why == ""


def test_gate_needs_schx_for_cm_preflight():
    cmd, why = gate_config.preflight_command({"backend": "cm", "input": "/x/in.wav"})
    assert cmd is None and "schx" in why


def test_cm_settings_change_the_gate_fingerprint_keys():
    assert "cm_tables" in gate_config.SIZING_KEYS and "cm_lead_in" in gate_config.SIZING_KEYS


def test_fleet_config_expansion_forwards_cm_settings_but_not_the_controller_path(tmp_path):
    import distribute_pull
    cfg = tmp_path / "amp.config.toml"
    cfg.write_text('backend = "cm"\nschx = "amp.schx"\ninput = "in.wav"\noversample = "auto"\ncm_lead_in = 3.0\ncm_tables = "off"\n'
                   'trunc_target = 0.006\ncm_run = "/controller/only/cm_run"\n')
    (tmp_path / "amp.schx").write_text("<x/>"); (tmp_path / "in.wav").write_bytes(b"x")
    out = distribute_pull.gen_args_from_config(cfg, tmp_path)
    assert out[out.index("--backend") + 1] == "cm"
    assert out[out.index("--cm-lead-in") + 1] == "3.0" and out[out.index("--cm-tables") + 1] == "off"
    assert out[out.index("--trunc-target") + 1] == "0.006" and out[out.index("--oversample") + 1] == "auto"
    assert "--cm-run" not in out and not any("/controller/only" in a for a in out)


@pytest.mark.parametrize("module,attr", [("render_backends", "CmBackend"), ("render_backends", "cm_solver_identity")])
def test_probe_backend_exists(module, attr):
    assert hasattr(__import__(module), attr)


def test_solver_identity_for_cm_uses_the_renderer(tmp_path, monkeypatch):
    import prepare_excitation
    script = tmp_path / "cm_run"
    script.write_text("#!/bin/sh\necho 'libcm abc1234'\n"); script.chmod(0o755)
    monkeypatch.setenv("CM_RUN", str(script))
    assert prepare_excitation.solver_identity("cm") == "cm:libcm abc1234"
    monkeypatch.setenv("CM_RUN", str(tmp_path / "missing"))
    assert prepare_excitation.solver_identity("cm") == "cm:unidentified"


def test_every_tool_that_picks_a_backend_offers_cm():
    for f in ("run_pipeline.py", "prepare_excitation.py", "preflight.py", "scaffold_config.py"):
        src = (HERE / f).read_text()
        assert '"cm"' in src, f"{f} does not offer the cm backend"


def test_oracle_check_picks_both_ends_and_the_centre():
    import oracle_check
    assert oracle_check.pick_rows(list(range(10)), 3) == [0, 5, 9]
    assert oracle_check.pick_rows(list(range(2)), 3) == [0, 1]
    assert oracle_check.pick_rows(list(range(100)), 5) == [0, 25, 50, 74, 99]


def test_oracle_check_esr_and_gain_fit():
    import numpy as np, oracle_check
    rng = np.random.default_rng(0)
    b = rng.standard_normal(1000)
    e, eg, g = oracle_check.esr_pair(2.0 * b, b, 0)       # a pure level difference: ESR 1, gone after the gain fit
    assert abs(e - 1.0) < 1e-9 and eg < 1e-12 and abs(g - 0.5) < 1e-9
    e, eg, g = oracle_check.esr_pair(b + 0.01 * rng.standard_normal(1000), b, 0)
    assert 5e-5 < e < 2e-4 and abs(g - 1.0) < 0.01


def test_oracle_check_records_into_the_manifest_and_refuses_the_same_renderer(tmp_path, monkeypatch, capsys):
    import json, numpy as np, soundfile as sf, oracle_check
    sr = 48000
    t = np.arange(sr * 3) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(tmp_path / "in.wav", x, sr, subtype="FLOAT")
    (tmp_path / "amp.schx").write_text("<x/>")
    rows = [dict(idx=i, ok=1, Gain=g) for i, g in enumerate((0.1, 0.5, 0.9))]
    (tmp_path / "params.csv").write_text("idx,Gain,ok\n" + "\n".join(f"{r['idx']},{r['Gain']},1" for r in rows) + "\n")
    outputs = np.stack([x * 1.01 * (r["Gain"] + 0.5) * 0.5 for r in rows])   # the dataset: 'scaled' by 0.5 (output_scale)
    np.save(tmp_path / "outputs.npy", outputs)
    cfg = {"backend": "cm", "schx": str(tmp_path / "amp.schx"), "knobs": ["Gain"], "param_map": {"Gain": "Gain"}, "input_wav": str(tmp_path / "in.wav"),
           "output_scale": 0.5, "capture_chain": None, "renderer": {"name": "cm", "esr_vs_oracle": None}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    def fake_oracle(oracle, cfg_, in_wav, params, out_wav, oversample):
        y, _ = sf.read(str(in_wav), dtype="float32")
        sf.write(str(out_wav), (y * (params["Gain"] + 0.5)).astype("float32"), sr, subtype="FLOAT")

    monkeypatch.setattr(oracle_check, "render_oracle", fake_oracle)
    monkeypatch.setattr(oracle_check, "oracle_identity", lambda o: "livespice:test")
    monkeypatch.setattr(sys, "argv", ["oracle_check.py", "--dataset", str(tmp_path), "--n", "3", "--seconds", "3"])
    assert oracle_check.main() == 0
    rec = json.loads((tmp_path / "config.json").read_text())["renderer"]["esr_vs_oracle"]
    assert rec["oracle"] == "livespice" and rec["oracle_version"] == "livespice:test" and rec["n"] == 3
    assert abs(rec["gain_median"] - 1.0 / 1.01) < 1e-3 and rec["esr_median"] < 1e-3      # a 1 % level error: ESR 1e-4
    cfg["backend"] = "livespice"
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        oracle_check.main()
