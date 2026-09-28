"""gate_config.py: fingerprint semantics, step sequencing, sidecar, and refusal on staleness.

No renderer runs: gate_config._run is replaced by a fake that records each command and returns
a scripted exit code, so these test the sequencer's own logic (what runs, in what order, what is
recorded, what invalidates a pass) and nothing about the tools it shells out to.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gate_config as gc  # noqa: E402


def make_cfg(tmp_path, *, backend="livespice", extra="", exc=True, knobs="Gain = [0.1, 0.5, 1.0]"):
    schx = tmp_path / "c.schx"
    schx.write_text("<schx v1/>")
    wav = tmp_path / "exc.wav"
    if exc:
        wav.write_bytes(b"RIFF-excitation-v1")
    cfg = tmp_path / "dev.config.toml"
    cfg.write_text(f'''schx = "{schx}"
input = "{wav}"
backend = "{backend}"
oversample = 8
epochs = 0
lr = 3e-4
{extra}
[knobs]
{knobs}
[knob-kind]
Gain = "drive"
[fixed]
Level = 0.5
''')
    return cfg, schx, wav


class Runner:
    """Fake gc._run: records commands; `fail` maps a script basename to a nonzero exit."""
    def __init__(self, fail=None, on_run=None):
        self.cmds, self.fail, self.on_run = [], fail or {}, on_run

    def __call__(self, cmd):
        self.cmds.append([str(c) for c in cmd])
        script = Path(str(cmd[1])).name
        if self.on_run:
            self.on_run(script)
        return self.fail.get(script, 0)

    @property
    def scripts(self):
        return [Path(c[1]).name for c in self.cmds]


@pytest.fixture
def runner(monkeypatch):
    r = Runner()
    monkeypatch.setattr(gc, "_run", r)
    return r


def run_main(cfg, *extra):
    return gc.main(["--config", str(cfg), "--no-solver-id", *extra])


def sidecar(cfg):
    return json.loads(gc.sidecar_path(cfg).read_text())


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
class TestFingerprint:
    def fp(self, cfg):
        return gc.fingerprints(gc.load_config(cfg), with_solver=False)

    def test_stable(self, tmp_path):
        cfg, *_ = make_cfg(tmp_path)
        assert self.fp(cfg) == self.fp(cfg)

    def test_schx_content_changes_both(self, tmp_path):
        cfg, schx, _ = make_cfg(tmp_path)
        a = self.fp(cfg)
        schx.write_text("<schx v2/>")
        b = self.fp(cfg)
        assert a["fingerprint"] != b["fingerprint"]
        assert a["sizing_fingerprint"] != b["sizing_fingerprint"]

    def test_schx_path_is_not_part_of_it(self, tmp_path):
        cfg, schx, _ = make_cfg(tmp_path)
        a = self.fp(cfg)
        other = tmp_path / "elsewhere.schx"
        other.write_text(schx.read_text())
        cfg.write_text(cfg.read_text().replace(str(schx), str(other)))
        assert self.fp(cfg)["fingerprint"] == a["fingerprint"]

    def test_knob_range_changes_sizing(self, tmp_path):
        cfg, *_ = make_cfg(tmp_path)
        a = self.fp(cfg)
        cfg.write_text(cfg.read_text().replace("[0.1, 0.5, 1.0]", "[0.1, 0.5, 0.9, 1.0]"))
        assert self.fp(cfg)["sizing_fingerprint"] != a["sizing_fingerprint"]

    def test_training_hyperparams_do_not_invalidate(self, tmp_path):
        cfg, *_ = make_cfg(tmp_path)
        a = self.fp(cfg)
        cfg.write_text(cfg.read_text().replace("epochs = 0", "epochs = 500").replace("lr = 3e-4", "lr = 1e-3"))
        assert self.fp(cfg)["fingerprint"] == a["fingerprint"]

    def test_excitation_changes_full_but_not_sizing(self, tmp_path):
        cfg, _, wav = make_cfg(tmp_path)
        a = self.fp(cfg)
        wav.write_bytes(b"RIFF-excitation-v2")
        b = self.fp(cfg)
        assert a["fingerprint"] != b["fingerprint"]
        assert a["sizing_fingerprint"] == b["sizing_fingerprint"]

    def test_knob_kind_changes_full_but_not_sizing(self, tmp_path):
        cfg, *_ = make_cfg(tmp_path)
        a = self.fp(cfg)
        cfg.write_text(cfg.read_text().replace('Gain = "drive"', 'Gain = "rms"'))
        b = self.fp(cfg)
        assert a["fingerprint"] != b["fingerprint"]
        assert a["sizing_fingerprint"] == b["sizing_fingerprint"]

    def test_missing_schx_is_a_gate_error(self, tmp_path):
        cfg, schx, _ = make_cfg(tmp_path)
        schx.unlink()
        with pytest.raises(gc.GateError, match="schx not found"):
            self.fp(cfg)

    def test_diff_components_names_what_changed(self):
        assert gc.diff_components({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == ["b", "c"]


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------
class TestDecideExcitation:
    def test_missing_wav_sizes_it(self):
        assert gc.decide_excitation("auto", False, None, "x")[0] is True

    def test_missing_wav_with_never_is_an_error(self):
        with pytest.raises(gc.GateError):
            gc.decide_excitation("never", False, None, "x")

    def test_always(self):
        assert gc.decide_excitation("always", True, {"sizing_fingerprint": "x"}, "x")[0] is True

    def test_no_prior_trusts_existing(self):
        resize, why = gc.decide_excitation("auto", True, None, "x")
        assert resize is False and "trusting" in why

    def test_prior_with_changed_sizing_resizes(self):
        resize, why = gc.decide_excitation("auto", True, {"sizing_fingerprint": "old"}, "new")
        assert resize is True and "stale" in why

    def test_prior_unchanged_keeps(self):
        assert gc.decide_excitation("auto", True, {"sizing_fingerprint": "x"}, "x")[0] is False

    def test_never_keeps_even_when_stale(self):
        assert gc.decide_excitation("never", True, {"sizing_fingerprint": "old"}, "new")[0] is False


class TestCommands:
    def test_livespice_preflight_mirrors_run_pipeline(self, tmp_path):
        cfg, schx, wav = make_cfg(tmp_path)
        cmd, _ = gc.preflight_command(gc.load_config(cfg))
        assert cmd[2:8] == ["--backend", "livespice", "--schx", str(schx), "--input", str(wav)]
        assert "--knobs" in cmd and "--knob-kind" in cmd and "--fixed-params" in cmd

    def test_ngspice_deck_preflight_defaults(self, tmp_path):
        cfg, *_ = make_cfg(tmp_path, backend="ngspice-deck",
                           extra=f'pedal-dir = "{tmp_path}"\nmodule = "gen_x"')
        cmd, _ = gc.preflight_command(gc.load_config(cfg))
        assert cmd[cmd.index("--probe-node") + 1] == "OUT"
        assert cmd[cmd.index("--maxstep") + 1] == str(3e-6)

    @pytest.mark.parametrize("backend", ["ngspice", "cpp", "ltspice-deck"])
    def test_backends_without_a_mode_skip_with_reason(self, tmp_path, backend):
        cfg, *_ = make_cfg(tmp_path, backend=backend)
        c = gc.load_config(cfg)
        assert gc.preflight_command(c)[0] is None
        assert "no mode" in gc.preflight_command(c)[1]
        assert gc.transient_command(c, cfg)[0] is None

    def test_grid_command_is_check_only(self, tmp_path):
        assert "--apply" not in gc.grid_command(tmp_path / "x.toml", 0.03)

    def test_prepare_command_leaves_config_alone_when_wav_exists(self, tmp_path):
        cfg, _, wav = make_cfg(tmp_path)
        c = gc.load_config(cfg)
        assert "--no-update-config" in gc.prepare_command(c, cfg, "s.wav", [])
        wav.unlink()
        assert "--no-update-config" not in gc.prepare_command(c, cfg, "s.wav", [])

    def test_resolve_sweep_file_prefers_arg_then_recipe(self, tmp_path):
        cfg, _, wav = make_cfg(tmp_path)
        c = gc.load_config(cfg)
        assert gc.resolve_sweep_file("given.wav", c) == "given.wav"
        src = tmp_path / "sweep.wav"
        src.write_bytes(b"x")
        wav.with_suffix(".recipe.json").write_text(json.dumps({"source": {"path": str(src)}}))
        assert gc.resolve_sweep_file(None, c) == str(src)
        src.unlink()
        with pytest.raises(gc.GateError, match="--sweep-file"):
            gc.resolve_sweep_file(None, c)


# ---------------------------------------------------------------------------
# the gate end to end (tools faked)
# ---------------------------------------------------------------------------
class TestRunGate:
    def test_pass_runs_transient_then_preflight_and_writes_sidecar(self, tmp_path, runner):
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg) == 0
        assert runner.scripts == ["check_transient_coverage.py", "preflight.py"]
        s = sidecar(cfg)
        assert s["status"] == "pass" and s["schema"] == gc.SCHEMA
        assert [x["step"] for x in s["steps"]] == ["excitation", "transient", "preflight"]
        assert s["steps"][0]["status"] == "not-needed"
        assert s["fingerprint"] and s["components"]

    def test_missing_excitation_is_sized_first(self, tmp_path, runner):
        cfg, _, wav = make_cfg(tmp_path, exc=False)
        # the fake prepare_excitation "builds" the wav so later steps and the fingerprint see it
        runner.on_run = lambda script: wav.write_bytes(b"built") if script == "prepare_excitation.py" else None
        assert run_main(cfg, "--sweep-file", "sweep.wav") == 0
        assert runner.scripts == ["prepare_excitation.py", "check_transient_coverage.py", "preflight.py"]
        cmd = runner.cmds[0]
        assert cmd[cmd.index("--sweep-file") + 1] == "sweep.wav"
        assert "--no-update-config" not in cmd

    def test_missing_excitation_without_sweep_file_is_exit_2(self, tmp_path, runner, capsys):
        cfg, *_ = make_cfg(tmp_path, exc=False)
        assert run_main(cfg) == 2
        assert "sweep file" in capsys.readouterr().err
        assert runner.cmds == [] and not gc.sidecar_path(cfg).exists()

    def test_transient_failure_stops_before_preflight_and_is_recorded(self, tmp_path, monkeypatch):
        r = Runner(fail={"check_transient_coverage.py": 1})
        monkeypatch.setattr(gc, "_run", r)
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg) == 1
        assert r.scripts == ["check_transient_coverage.py"]
        s = sidecar(cfg)
        assert s["status"] == "fail" and s["failed_step"] == "transient"
        ok, msg = gc.verify_gate(cfg, with_solver=False)
        assert not ok and "FAILED" in msg and "transient" in msg

    def test_preflight_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gc, "_run", Runner(fail={"preflight.py": 1}))
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg) == 1
        assert sidecar(cfg)["failed_step"] == "preflight"

    def test_skipped_steps_are_recorded_not_dropped(self, tmp_path, runner):
        cfg, *_ = make_cfg(tmp_path, backend="ngspice")
        assert run_main(cfg) == 0
        assert runner.cmds == []
        by = {x["step"]: x for x in sidecar(cfg)["steps"]}
        assert by["transient"]["status"] == "skipped" and by["preflight"]["status"] == "skipped"
        assert "no mode" in by["preflight"]["detail"]

    def test_check_grid_runs_first_and_never_applies(self, tmp_path, runner):
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg, "--check-grid") == 0
        assert runner.scripts[0] == "grid_adequacy.py"
        assert "--apply" not in runner.cmds[0]

    def test_grid_failure_stops_everything(self, tmp_path, monkeypatch):
        r = Runner(fail={"grid_adequacy.py": 1})
        monkeypatch.setattr(gc, "_run", r)
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg, "--check-grid") == 1
        assert r.scripts == ["grid_adequacy.py"]
        assert sidecar(cfg)["failed_step"] == "grid"

    def test_grid_step_is_absent_by_default(self, tmp_path, runner):
        cfg, *_ = make_cfg(tmp_path)
        run_main(cfg)
        assert "grid_adequacy.py" not in runner.scripts

    def test_stale_prior_triggers_resize(self, tmp_path, runner):
        cfg, schx, wav = make_cfg(tmp_path)
        assert run_main(cfg) == 0                       # first pass, no re-size
        schx.write_text("<schx v2/>")                   # circuit edited after the gate passed
        runner.cmds.clear()
        runner.on_run = lambda script: wav.write_bytes(b"resized") if script == "prepare_excitation.py" else None
        assert run_main(cfg, "--sweep-file", "sweep.wav") == 0
        assert runner.scripts[0] == "prepare_excitation.py"
        assert "--no-update-config" in runner.cmds[0]   # wav existed, so the config text is left alone

    def test_unchanged_prior_does_not_resize(self, tmp_path, runner):
        cfg, *_ = make_cfg(tmp_path)
        run_main(cfg)
        runner.cmds.clear()
        run_main(cfg)
        assert "prepare_excitation.py" not in runner.scripts

    def test_circuit_edited_mid_flight_fails_the_gate(self, tmp_path, monkeypatch):
        cfg, schx, _ = make_cfg(tmp_path)
        r = Runner(on_run=lambda script: schx.write_text("<edited mid-flight/>")
                   if script == "check_transient_coverage.py" else None)
        monkeypatch.setattr(gc, "_run", r)
        assert run_main(cfg) == 1
        s = sidecar(cfg)
        assert s["status"] == "fail" and s["failed_step"] == "consistency"
        assert "sizing.schx_sha256" in s["reason"]

    def test_dry_run_runs_and_writes_nothing(self, tmp_path, runner, capsys):
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg, "--dry-run") == 0
        assert runner.cmds == [] and not gc.sidecar_path(cfg).exists()
        assert "excitation:" in capsys.readouterr().out

    def test_dry_run_shows_fleet_dispatch_not_the_single_machine_commands(self, tmp_path, runner,
                                                                          capsys, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory", lambda p: {})
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg, "--dry-run", "--workers", "h1,h2", "--check-grid") == 0
        out = capsys.readouterr().out
        assert "fleet:       h1, h2" in out
        assert "distribute_pull.py" in out and "grid_adequacy" in out
        assert "check_transient_coverage" in out
        assert "fleet-wav-sync:" in out
        assert runner.cmds == [] and not gc.sidecar_path(cfg).exists()

    def test_dry_run_without_fleet_flags_shows_no_fleet_lines(self, tmp_path, runner, capsys):
        cfg, *_ = make_cfg(tmp_path)
        run_main(cfg, "--dry-run")
        out = capsys.readouterr().out
        assert "fleet:" not in out and "distribute_pull.py" not in out


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
class TestVerify:
    def passed(self, tmp_path, runner):
        cfg, schx, wav = make_cfg(tmp_path)
        assert run_main(cfg) == 0
        return cfg, schx, wav

    def test_no_sidecar_refuses(self, tmp_path):
        cfg, *_ = make_cfg(tmp_path)
        ok, msg = gc.verify_gate(cfg, with_solver=False)
        assert not ok and "no gate sidecar" in msg

    def test_fresh_pass_verifies(self, tmp_path, runner):
        cfg, *_ = self.passed(tmp_path, runner)
        assert gc.verify_gate(cfg, with_solver=False)[0]
        assert run_main(cfg, "--verify") == 0

    def test_editing_the_circuit_makes_it_stale_and_names_the_component(self, tmp_path, runner):
        cfg, schx, _ = self.passed(tmp_path, runner)
        schx.write_text("<schx v2/>")
        ok, msg = gc.verify_gate(cfg, with_solver=False)
        assert not ok and "STALE" in msg and "sizing.schx_sha256" in msg
        assert run_main(cfg, "--verify") == 1

    def test_replacing_the_excitation_makes_it_stale(self, tmp_path, runner):
        cfg, _, wav = self.passed(tmp_path, runner)
        wav.write_bytes(b"a different excitation")
        ok, msg = gc.verify_gate(cfg, with_solver=False)
        assert not ok and "excitation_sha256" in msg

    def test_retuning_training_does_not_stale_it(self, tmp_path, runner):
        cfg, *_ = self.passed(tmp_path, runner)
        cfg.write_text(cfg.read_text().replace("epochs = 0", "epochs = 900"))
        assert gc.verify_gate(cfg, with_solver=False)[0]

    def test_wrong_schema_is_refused(self, tmp_path, runner):
        cfg, *_ = self.passed(tmp_path, runner)
        s = sidecar(cfg)
        s["schema"] = 999
        gc.sidecar_path(cfg).write_text(json.dumps(s))
        assert not gc.verify_gate(cfg, with_solver=False)[0]

    def test_corrupt_sidecar_is_refused_not_crashed(self, tmp_path, runner):
        cfg, *_ = self.passed(tmp_path, runner)
        gc.sidecar_path(cfg).write_text("{not json")
        ok, msg = gc.verify_gate(cfg, with_solver=False)
        assert not ok and "unreadable" in msg

    def test_missing_config_is_exit_2(self, tmp_path, capsys):
        assert gc.main(["--config", str(tmp_path / "nope.toml"), "--no-solver-id"]) == 2


# ---------------------------------------------------------------------------
# the recipe's own record of what it was sized against
# ---------------------------------------------------------------------------
import sizing_inputs  # noqa: E402


def write_recipe(cfg, schx, wav, *, grid=None, fixed=None, schx_bytes=None):
    """A recipe.json whose sizing.inputs describes the given circuit/grid (defaults: make_cfg's)."""
    inputs = sizing_inputs.inputs_record(
        identity=schx_bytes if schx_bytes is not None else schx.read_bytes(),
        circuit_kind="schx",
        knob_ranges=grid if grid is not None else {"Gain": [0.1, 0.5, 1.0]},
        fixed=fixed if fixed is not None else {"Level": 0.5}, sample_grid=4, conditions="os=8")
    wav.with_suffix(".recipe.json").write_text(json.dumps({"sizing": {"inputs": inputs}}))


class TestRecipeInputs:
    def test_recipe_status_match(self, tmp_path):
        cfg, schx, wav = make_cfg(tmp_path)
        write_recipe(cfg, schx, wav)
        assert gc.recipe_status(gc.load_config(cfg))["status"] == "match"

    def test_recipe_status_no_recipe_and_unrecorded(self, tmp_path):
        cfg, schx, wav = make_cfg(tmp_path)
        assert gc.recipe_status(gc.load_config(cfg))["status"] == "no-recipe"
        wav.with_suffix(".recipe.json").write_text(json.dumps({"sizing": {}}))
        assert gc.recipe_status(gc.load_config(cfg))["status"] == "unrecorded"

    def test_recipe_status_unreadable_recipe_does_not_crash(self, tmp_path):
        cfg, schx, wav = make_cfg(tmp_path)
        wav.with_suffix(".recipe.json").write_text("{nope")
        assert gc.recipe_status(gc.load_config(cfg))["status"] == "unreadable"

    def test_grid_mismatch_in_recipe_forces_a_resize_without_a_prior_gate(self, tmp_path, runner):
        cfg, schx, wav = make_cfg(tmp_path)
        write_recipe(cfg, schx, wav, grid={"Gain": [0.1, 0.9]})      # sized on a different grid
        runner.on_run = lambda s: wav.write_bytes(b"resized") if s == "prepare_excitation.py" else None
        assert run_main(cfg, "--sweep-file", "sweep.wav") == 0
        assert runner.scripts[0] == "prepare_excitation.py"
        step = sidecar(cfg)["steps"][0]
        assert step["recipe_inputs"]["status"] == "stale" and "Gain" in step["detail"]

    def test_circuit_only_mismatch_keeps_the_excitation_and_says_so(self, tmp_path, runner):
        cfg, schx, wav = make_cfg(tmp_path)
        write_recipe(cfg, schx, wav, schx_bytes=b"<an earlier circuit revision/>")
        assert run_main(cfg) == 0
        assert "prepare_excitation.py" not in runner.scripts
        step = sidecar(cfg)["steps"][0]
        assert step["status"] == "not-needed" and "advisory" in step["detail"]
        assert step["recipe_inputs"]["status"] == "circuit-differs"

    def test_matching_recipe_keeps_the_excitation(self, tmp_path, runner):
        cfg, schx, wav = make_cfg(tmp_path)
        write_recipe(cfg, schx, wav)
        run_main(cfg)
        assert "prepare_excitation.py" not in runner.scripts
        assert "matches" in sidecar(cfg)["steps"][0]["detail"]

    def test_a_prior_gate_outranks_the_recipe(self, tmp_path, runner):
        cfg, schx, wav = make_cfg(tmp_path)
        run_main(cfg)                                          # passes, writes a sidecar
        write_recipe(cfg, schx, wav, grid={"Gain": [0.1, 0.9]})  # recipe now disagrees
        runner.cmds.clear()
        run_main(cfg)
        assert "prepare_excitation.py" not in runner.scripts   # unchanged sizing fingerprint wins

    def test_resize_never_ignores_the_recipe(self, tmp_path, runner):
        cfg, schx, wav = make_cfg(tmp_path)
        write_recipe(cfg, schx, wav, grid={"Gain": [0.1, 0.9]})
        run_main(cfg, "--resize", "never")
        assert "prepare_excitation.py" not in runner.scripts


# ---------------------------------------------------------------------------
# fleet mode (config-gate-proposal.md's "Where fleet/worker concerns fit",
# docs/implementation-roadmap.md item 8)
# ---------------------------------------------------------------------------
import subprocess as _subprocess  # noqa: E402


class TestNormalizeInventoryArg:
    def test_none_stays_none(self):
        assert gc._normalize_inventory_arg(None) is None

    def test_bare_flag_sentinel_becomes_none(self):
        assert gc._normalize_inventory_arg("__DEFAULT__") is None

    def test_explicit_path_string_becomes_a_path(self):
        assert gc._normalize_inventory_arg("/tmp/fleet.toml") == Path("/tmp/fleet.toml")

    def test_a_path_object_passes_through(self, tmp_path):
        assert gc._normalize_inventory_arg(tmp_path) == tmp_path


class TestResolveFleetHosts:
    def test_workers_csv_wins(self, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory", lambda p: {"should": {}, "not": {}, "be": {}, "used": {}})
        assert gc.resolve_fleet_hosts("h1, h2 ,h3", None) == ["h1", "h2", "h3"]

    def test_empty_workers_string_raises(self):
        with pytest.raises(gc.GateError, match="empty"):
            gc.resolve_fleet_hosts("  , ,", None)

    def test_falls_back_to_inventory_sorted(self, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory",
                            lambda p: {"zeta": {}, "alpha": {}})
        assert gc.resolve_fleet_hosts(None, None) == ["alpha", "zeta"]

    def test_empty_inventory_raises(self, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory", lambda p: {})
        with pytest.raises(gc.GateError, match="no hosts"):
            gc.resolve_fleet_hosts(None, None)

    def test_bare_inventory_sentinel_is_normalized_before_loading(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory",
                            lambda p: seen.setdefault("path", p) or {"h1": {}})
        gc.resolve_fleet_hosts(None, "__DEFAULT__")
        assert seen["path"] is None


class TestRepoDirForHost:
    def test_uses_inventorys_own_repo_field(self, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory",
                            lambda p: {"mac-1": {"repo": "~/render/parametric-nam"}})
        assert gc.repo_dir_for_host("mac-1", None) == "~/render/parametric-nam"

    def test_falls_back_to_default_when_host_unknown(self, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory", lambda p: {})
        assert gc.repo_dir_for_host("mystery-host", None) == "~/work/parametric-nam"

    def test_falls_back_to_default_when_repo_field_absent(self, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory", lambda p: {"h1": {"cores": 8}})
        assert gc.repo_dir_for_host("h1", None) == "~/work/parametric-nam"

    def test_custom_default_is_honored(self, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "load_inventory", lambda p: {})
        assert gc.repo_dir_for_host("h1", None, default="/opt/nam") == "/opt/nam"


class TestSyncFindpeakCache:
    def test_builds_the_right_command(self, monkeypatch):
        seen = {}
        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return _subprocess.CompletedProcess(cmd, 0, "ok\n", "")
        monkeypatch.setattr(gc.subprocess, "run", fake_run)
        rc, out = gc.sync_findpeak_cache(["h1", "h2"])
        assert rc == 0 and "ok" in out
        assert seen["cmd"][0].endswith("sync_findpeak_cache.sh")
        assert seen["cmd"][1:] == ["--workers", "h1,h2"]

    def test_never_raises_on_failure(self, monkeypatch):
        def boom(cmd, **kw):
            raise OSError("no such file")
        monkeypatch.setattr(gc.subprocess, "run", boom)
        rc, out = gc.sync_findpeak_cache(["h1"])
        assert rc != 0 and "OSError" in out

    def test_never_raises_on_timeout(self, monkeypatch):
        def boom(cmd, **kw):
            raise _subprocess.TimeoutExpired(cmd="x", timeout=1)
        monkeypatch.setattr(gc.subprocess, "run", boom)
        rc, out = gc.sync_findpeak_cache(["h1"])
        assert rc != 0


class TestSyncExcitationWav:
    def test_missing_wav_raises(self, tmp_path):
        cfg = {"input": str(tmp_path / "nope.wav")}
        with pytest.raises(gc.GateError, match="not found"):
            gc.sync_excitation_wav(cfg, ["h1"], {"h1": "~/work/parametric-nam"})

    def test_destination_is_repo_relative_on_each_worker(self, tmp_path, monkeypatch):
        wav = tmp_path / "amps" / "exc.wav"
        wav.parent.mkdir()
        wav.write_bytes(b"RIFF")
        monkeypatch.setattr(gc, "HERE", tmp_path)   # "this repo root" for repo_relpath
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(gc.subprocess, "run", fake_run)
        results = gc.sync_excitation_wav({"input": str(wav)}, ["h1"], {"h1": "/remote/repo"})
        assert results == [("h1", True, "ok")]
        rsync_cmd = calls[1]
        assert rsync_cmd[0] == "rsync"
        assert rsync_cmd[-1] == "h1:/remote/repo/amps/exc.wav"

    def test_per_host_failure_is_reported_not_raised(self, tmp_path, monkeypatch):
        wav = tmp_path / "exc.wav"
        wav.write_bytes(b"RIFF")
        monkeypatch.setattr(gc, "HERE", tmp_path)
        def fake_run(cmd, **kw):
            if cmd[0] == "ssh":
                return _subprocess.CompletedProcess(cmd, 0, "", "")
            return _subprocess.CompletedProcess(cmd, 1, "", "connection refused")
        monkeypatch.setattr(gc.subprocess, "run", fake_run)
        results = gc.sync_excitation_wav({"input": str(wav)}, ["bad-host"], {"bad-host": "/r"})
        host, ok, detail = results[0]
        assert host == "bad-host" and ok is False and "connection refused" in detail

    def test_one_bad_host_does_not_stop_the_rest(self, tmp_path, monkeypatch):
        wav = tmp_path / "exc.wav"
        wav.write_bytes(b"RIFF")
        monkeypatch.setattr(gc, "HERE", tmp_path)
        def fake_run(cmd, **kw):
            if cmd[0] == "rsync" and "bad:" in cmd[-1]:
                return _subprocess.CompletedProcess(cmd, 1, "", "unreachable")
            return _subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(gc.subprocess, "run", fake_run)
        results = gc.sync_excitation_wav({"input": str(wav)}, ["good", "bad"],
                                         {"good": "/r", "bad": "/r"})
        by_host = {h: ok for h, ok, _ in results}
        assert by_host == {"good": True, "bad": False}


class TestDispatchCommand:
    def test_grid_and_transient_use_distinct_tools_and_work_subdirs(self, tmp_path):
        cfg = tmp_path / "d.config.toml"
        gcmd, gdir = gc.grid_command_fleet(cfg, ["h1"], {"h1": "/r"})
        tcmd, tdir = gc.transient_command_fleet(cfg, ["h1"], {"h1": "/r"})
        assert "--tool" in gcmd and gcmd[gcmd.index("--tool") + 1] == "grid_adequacy"
        assert "--tool" in tcmd and tcmd[tcmd.index("--tool") + 1] == "check_transient_coverage"
        assert gdir != tdir
        assert gdir.name == "grid" and tdir.name == "transient"

    def test_worker_flags_built_per_host(self, tmp_path):
        cfg = tmp_path / "d.config.toml"
        cmd, _ = gc.grid_command_fleet(cfg, ["h1", "h2"], {"h1": "/repo1", "h2": "/repo2"})
        assert cmd.count("--worker") == 2
        assert "h1:/repo1" in cmd and "h2:/repo2" in cmd

    def test_skip_gate_check_is_always_forwarded(self, tmp_path):
        # gate_config.py dispatching itself via distribute_pull.py must not re-trigger
        # distribute_pull's OWN gate-check wiring (item 3) -- that would be circular.
        cfg = tmp_path / "d.config.toml"
        cmd, _ = gc.grid_command_fleet(cfg, ["h1"], {"h1": "/r"})
        assert "--skip-gate-check" in cmd

    def test_collect_dir_is_namespaced_by_config_stem(self, tmp_path):
        cfg = tmp_path / "my_amp.config.toml"
        _, collect_dir = gc.grid_command_fleet(cfg, ["h1"], {"h1": "/r"})
        assert "my_amp" in str(collect_dir)


class TestMergeVerdictCommand:
    def test_none_when_no_shards_present(self, tmp_path):
        assert gc._merge_verdict_command("grid_adequacy.py", "--merge", "shard_*.json",
                                         tmp_path, Path("cfg.toml")) is None

    def test_grid_verdict_uses_correct_flag_and_extra_target(self, tmp_path):
        (tmp_path / "shard_0-0_2.json").write_text("{}")
        cmd = gc._merge_verdict_command("grid_adequacy.py", "--merge", "shard_*.json",
                                        tmp_path, Path("cfg.toml"), extra=["--target", "0.05"])
        assert cmd[1].endswith("grid_adequacy.py")
        assert "--merge" in cmd and "--target" in cmd and "0.05" in cmd

    def test_transient_verdict_uses_merge_onsets_not_merge(self, tmp_path):
        (tmp_path / "tcov_shard_0-0_2.json").write_text("{}")
        cmd = gc._merge_verdict_command("check_transient_coverage.py", "--merge-onsets",
                                        "tcov_shard_*.json", tmp_path, Path("cfg.toml"))
        assert "--merge-onsets" in cmd and "--merge" not in cmd

    def test_only_matching_glob_is_picked_up(self, tmp_path):
        (tmp_path / "shard_0-0_2.json").write_text("{}")          # grid_adequacy's own file
        (tmp_path / "tcov_shard_0-0_2.json").write_text("{}")     # must be excluded here
        cmd = gc._merge_verdict_command("grid_adequacy.py", "--merge", "shard_*.json",
                                        tmp_path, Path("cfg.toml"))
        assert str(tmp_path / "tcov_shard_0-0_2.json") not in cmd
        assert str(tmp_path / "shard_0-0_2.json") in cmd


class TestRunGateFleetIntegration:
    """run_gate's fleet branch, driven through main() -- Runner fakes _run (the sequenced
    dispatch/merge/preflight/excitation steps); subprocess.run is faked separately for the
    cache-sync and wav-sync calls, which don't go through _run at all."""

    def _fleet_subprocess(self, monkeypatch, dispatch_calls=None):
        calls = dispatch_calls if dispatch_calls is not None else []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _subprocess.CompletedProcess(cmd, 0, "ok\n", "")
        monkeypatch.setattr(gc.subprocess, "run", fake_run)
        return calls

    def test_no_fleet_flags_runs_single_machine_as_before(self, tmp_path, runner, monkeypatch):
        sub_calls = self._fleet_subprocess(monkeypatch)
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg) == 0
        # git_rev(HERE) always runs (unrelated to fleet mode, stamps tool_git_rev in the
        # sidecar) -- the real assertion is that no ssh/rsync/sync_findpeak_cache call happens.
        assert not any(c[0] in ("ssh", "rsync") or "sync_findpeak_cache" in str(c[0])
                      for c in sub_calls)
        s = sidecar(cfg)
        assert not any("fleet" in x["step"] for x in s["steps"])

    def test_workers_flag_triggers_fleet_steps_in_order(self, tmp_path, runner, monkeypatch):
        self._fleet_subprocess(monkeypatch)
        cfg, schx, wav = make_cfg(tmp_path)

        def on_run(script):
            # make the fleet dispatch + merge steps look like they produced real shard files,
            # and preflight look like a real pass, exactly as run_main's callers already need
            # for the existing (non-fleet) preflight/transient tests to pass.
            if script == "distribute_pull.py":
                pass   # collect dirs are created lazily by _merge_verdict_command's own glob
        runner.on_run = on_run
        # Fleet mode needs the collected shard dirs to exist for the merge step to find
        # anything -- create them ahead of the run so "transient"/"grid" don't skip.
        import gate_config as _gc
        grid_dir = Path.home() / ".cache" / "parametric-nam" / "gate-fleet" / cfg.stem / "grid"
        tcov_dir = Path.home() / ".cache" / "parametric-nam" / "gate-fleet" / cfg.stem / "transient"
        grid_dir.mkdir(parents=True, exist_ok=True)
        tcov_dir.mkdir(parents=True, exist_ok=True)
        (grid_dir / "shard_0-0_1.json").write_text("{}")
        (tcov_dir / "tcov_shard_0-0_1.json").write_text("{}")
        try:
            assert run_main(cfg, "--workers", "localhost", "--check-grid") == 0
        finally:
            import shutil as _shutil
            _shutil.rmtree(grid_dir.parent, ignore_errors=True)
        steps = [s["step"] for s in sidecar(cfg)["steps"]]
        assert steps.index("fleet-cache-sync") < steps.index("grid-dispatch")
        assert steps.index("grid-dispatch") < steps.index("grid")
        assert steps.index("grid") < steps.index("excitation")
        assert steps.index("excitation") < steps.index("fleet-wav-sync")
        assert steps.index("fleet-wav-sync") < steps.index("transient-dispatch")
        assert steps.index("transient-dispatch") < steps.index("transient")
        assert steps.index("transient") < steps.index("fleet-cache-sync-after")
        assert steps.index("fleet-cache-sync-after") < steps.index("preflight")

    def test_dispatch_failure_stops_before_the_merge_step(self, tmp_path, monkeypatch):
        r = Runner(fail={"distribute_pull.py": 1})
        monkeypatch.setattr(gc, "_run", r)
        self._fleet_subprocess(monkeypatch)
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg, "--workers", "localhost", "--check-grid") == 1
        s = sidecar(cfg)
        assert s["failed_step"] == "grid-dispatch"
        assert r.scripts == ["distribute_pull.py"]   # never reached the merge command

    def test_no_shards_collected_fails_the_gate_not_a_silent_pass(self, tmp_path, monkeypatch):
        # dispatch "succeeds" (rc 0) but produces no shard files at all -- must not be
        # silently treated as passing.
        monkeypatch.setattr(gc, "_run", Runner())
        self._fleet_subprocess(monkeypatch)
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg, "--workers", "localhost", "--check-grid") == 1
        assert sidecar(cfg)["failed_step"] == "grid"

    def test_wav_sync_failure_fails_the_gate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gc, "_run", Runner())
        def fake_run(cmd, **kw):
            if cmd[0] == "rsync":
                return _subprocess.CompletedProcess(cmd, 1, "", "no route to host")
            return _subprocess.CompletedProcess(cmd, 0, "ok\n", "")
        monkeypatch.setattr(gc.subprocess, "run", fake_run)
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg, "--workers", "localhost") == 1
        assert sidecar(cfg)["failed_step"] == "fleet-wav-sync"

    def test_cache_sync_failure_does_not_fail_the_gate(self, tmp_path, monkeypatch):
        # sync_findpeak_cache is an optimization, not a correctness requirement -- see its
        # own docstring.
        monkeypatch.setattr(gc, "_run", Runner())
        def fake_run(cmd, **kw):
            if "sync_findpeak_cache.sh" in str(cmd[0]):
                return _subprocess.CompletedProcess(cmd, 1, "", "boom")
            return _subprocess.CompletedProcess(cmd, 0, "ok\n", "")
        monkeypatch.setattr(gc.subprocess, "run", fake_run)
        cfg, *_ = make_cfg(tmp_path)
        grid_dir = Path.home() / ".cache" / "parametric-nam" / "gate-fleet" / cfg.stem / "transient"
        grid_dir.mkdir(parents=True, exist_ok=True)
        (grid_dir / "tcov_shard_0-0_1.json").write_text("{}")
        try:
            run_main(cfg, "--workers", "localhost")
        finally:
            import shutil as _shutil
            _shutil.rmtree(grid_dir.parent, ignore_errors=True)
        by_step = {s["step"]: s for s in sidecar(cfg)["steps"]}
        assert by_step["fleet-cache-sync"]["status"] == "warn"

    def test_workers_and_inventory_both_absent_never_calls_fleet_functions(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("fleet resolution should not run without --workers/--inventory")
        monkeypatch.setattr(gc, "resolve_fleet_hosts", boom)
        monkeypatch.setattr(gc, "_run", Runner())
        cfg, *_ = make_cfg(tmp_path)
        assert run_main(cfg) == 0
