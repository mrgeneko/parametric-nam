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
