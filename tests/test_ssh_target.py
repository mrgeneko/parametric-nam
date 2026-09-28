"""ssh_target.py: the inventory's address/user/port/identity_file made real for every fleet
ssh/rsync call. Two layers: pure rendering, and REAL OpenSSH resolution via `ssh -G -F <file>`
(no connection is made) so the generated config is checked by the program that will read it."""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ssh_target as st  # noqa: E402
import cpu_topology  # noqa: E402
import distribute_pull as dp  # noqa: E402
import gate_config as gc  # noqa: E402
import run_pipeline as rp  # noqa: E402

HAVE_SSH = shutil.which("ssh") is not None
HERE = Path(__file__).resolve().parent.parent

INV = {
    "linux-1": {"address": "linux-1", "user": "gene", "identity_file": "~/.ssh/id_fleet"},
    "linux-2": {"address": "10.0.0.7", "user": "chewie", "port": 2222},
    "plain":   {"address": "plain", "cores": 4},          # nothing ssh-relevant to say
}


def _resolve(cfg: Path, host: str) -> dict:
    """What OpenSSH itself would use for `host` under `cfg`. Lists collapse to a list of values."""
    r = subprocess.run(["ssh", "-G", "-F", str(cfg), host], capture_output=True, text=True,
                       timeout=10)
    assert r.returncode == 0, r.stderr
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        out.setdefault(k, []).append(v)
    return out


class TestHostBlock:
    def test_user_only(self):
        assert st.host_block("h", {"address": "h", "user": "u"}) == ["Host h", "    User u"]

    def test_hostname_only_when_address_differs(self):
        assert "    HostName 1.2.3.4" in st.host_block("h", {"address": "1.2.3.4"})
        assert st.host_block("h", {"address": "h"}) is None

    def test_default_port_omitted_nondefault_kept(self):
        assert st.host_block("h", {"port": 22}) is None
        assert "    Port 2222" in st.host_block("h", {"port": 2222})

    def test_identity_file_pins_identities_only(self):
        b = st.host_block("h", {"identity_file": "~/.ssh/k"})
        assert "    IdentityFile ~/.ssh/k" in b and "    IdentitiesOnly yes" in b

    def test_no_identities_only_without_a_key(self):
        assert "IdentitiesOnly" not in "\n".join(st.host_block("h", {"user": "u"}))

    def test_value_with_space_is_quoted(self):
        assert '    IdentityFile "/Users/a b/key"' in st.host_block("h", {"identity_file": "/Users/a b/key"})

    @pytest.mark.parametrize("bad", ["a b", "a*", "a?", "!a", "a,b"])
    def test_pattern_or_space_in_name_refused(self, bad):
        with pytest.raises(ValueError):
            st.host_block(bad, {"user": "u"})

    def test_quote_or_newline_in_value_refused(self):
        with pytest.raises(ValueError):
            st.host_block("h", {"user": 'a"b'})
        with pytest.raises(ValueError):
            st.host_block("h", {"user": "a\nProxyCommand x"})


class TestRender:
    def test_none_when_no_host_has_anything_to_say(self):
        assert st.render_ssh_config({"a": {"address": "a"}, "b": {}}) is None
        assert st.render_ssh_config({}) is None

    def test_only_hosts_with_fields_get_blocks(self):
        text = st.render_ssh_config(INV)
        assert "Host linux-1" in text and "Host linux-2" in text and "Host plain" not in text

    def test_user_config_included_after_a_match_all_reset(self):
        text = st.render_ssh_config(INV)
        assert text.rstrip().endswith("Match all\nInclude ~/.ssh/config")

    def test_deterministic_regardless_of_dict_order(self):
        assert st.render_ssh_config(INV) == st.render_ssh_config(dict(reversed(list(INV.items()))))


class TestWriteConfig:
    def test_content_addressed_and_private(self, tmp_path):
        p = st.write_config("Host a\n    User u\n", tmp_path)
        assert p.parent == tmp_path and p.suffix == ".conf"
        assert p.read_text() == "Host a\n    User u\n"
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
        assert st.write_config("Host a\n    User u\n", tmp_path) == p
        assert st.write_config("Host b\n    User u\n", tmp_path) != p

    def test_rewrite_of_existing_does_not_touch_it(self, tmp_path):
        p = st.write_config("x\n", tmp_path)
        mtime = p.stat().st_mtime_ns
        st.write_config("x\n", tmp_path)
        assert p.stat().st_mtime_ns == mtime

    def test_no_tmp_files_left_behind(self, tmp_path):
        st.write_config("x\n", tmp_path)
        assert [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")] == []


class TestConfigureAndArgv:
    def _inv_file(self, tmp_path, body):
        p = tmp_path / "fleet.toml"
        p.write_text(body)
        return p

    def test_unconfigured_is_plain_ssh_and_plain_rsync(self):
        assert st.active_config() is None
        assert st.ssh_argv("h", "-o", "BatchMode=yes") == ["ssh", "-o", "BatchMode=yes", "h"]
        assert st.rsync_e() == []

    def test_missing_inventory_configures_nothing(self, tmp_path):
        assert st.configure(tmp_path / "nope.toml") is None
        assert st.ssh_argv("h") == ["ssh", "h"]

    def test_inventory_without_ssh_fields_configures_nothing(self, tmp_path):
        inv = self._inv_file(tmp_path, '[hosts."a"]\naddress = "a"\ncores = 4\n')
        assert st.configure(inv) is None

    def test_configure_activates_F_on_ssh_and_rsync(self, tmp_path):
        inv = self._inv_file(tmp_path, '[hosts."a"]\naddress = "a"\nuser = "u"\n')
        cfg = st.configure(inv)
        assert cfg is not None and cfg.is_file()
        assert st.ssh_argv("a", "-o", "X=1") == ["ssh", "-F", str(cfg), "-o", "X=1", "a"]
        assert st.rsync_e() == ["-e", f"ssh -F {cfg}"]

    def test_rsync_e_quotes_a_path_with_spaces(self, tmp_path):
        st._active = tmp_path / "a b" / "c.conf"
        assert st.rsync_e() == ["-e", f"ssh -F '{tmp_path / 'a b' / 'c.conf'}'"]

    def test_reconfigure_with_nothing_deactivates(self, tmp_path):
        st.configure(self._inv_file(tmp_path, '[hosts."a"]\nuser = "u"\n'))
        assert st.active_config() is not None
        st.configure(tmp_path / "nope.toml")
        assert st.active_config() is None


@pytest.mark.skipif(not HAVE_SSH, reason="needs the ssh client for -G")
class TestRealOpenSshResolution:
    """The point of the module: what OpenSSH actually does with the file we generate."""

    @pytest.fixture
    def cfg(self, tmp_path):
        user_cfg = tmp_path / "user_ssh_config"
        # The operator's own config: a stale User for linux-1 (inventory must beat it) and a
        # ProxyJump the inventory says nothing about (must survive via the Include).
        user_cfg.write_text("Host linux-1\n    User stale\n    ProxyJump bastion\n"
                            "Host plain\n    User plainuser\n")
        return st.write_config(st.render_ssh_config(INV, str(user_cfg)), tmp_path)

    def test_per_host_users_differ(self, cfg):
        assert _resolve(cfg, "linux-1")["user"] == ["gene"]
        assert _resolve(cfg, "linux-2")["user"] == ["chewie"]

    def test_address_becomes_hostname_and_port_applies(self, cfg):
        r = _resolve(cfg, "linux-2")
        assert r["hostname"] == ["10.0.0.7"] and r["port"] == ["2222"]

    def test_identity_file_pinned_with_identities_only(self, cfg):
        r = _resolve(cfg, "linux-1")
        assert r["identityfile"] == ["~/.ssh/id_fleet"] and r["identitiesonly"] == ["yes"]

    def test_inventory_wins_over_operators_own_config(self, cfg):
        assert _resolve(cfg, "linux-1")["user"] == ["gene"]          # not "stale"

    def test_operators_other_settings_still_apply_via_include(self, cfg):
        assert _resolve(cfg, "linux-1")["proxyjump"] == ["bastion"]

    def test_host_without_a_block_falls_through_to_operators_config(self, cfg):
        assert _resolve(cfg, "plain")["user"] == ["plainuser"]

    def test_include_is_global_not_conditional_on_the_last_host_block(self, cfg):
        # a host that matches NO generated block must still see the operator's config
        assert _resolve(cfg, "plain")["user"] == ["plainuser"]
        # ...and so must the LAST generated block's own host (the trap: a bare Include after
        # the final Host block is scoped to that block only)
        assert _resolve(cfg, "linux-2")["user"] == ["chewie"]


def _cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, out, err)


class TestEverySiteUsesTheConfig:
    """Each ssh/rsync call site must go through ssh_target -- a site that forgets is a host that
    silently gets the wrong login. One test per call site, asserting the real argv."""

    @pytest.fixture
    def active(self, tmp_path):
        p = tmp_path / "x.conf"
        p.write_text("")
        st._active = p
        return p

    def _capture(self, monkeypatch, module, stdout=""):
        calls = []
        def fake(argv, *a, **kw):
            calls.append(list(argv))
            return _cp(0, stdout)
        monkeypatch.setattr(module.subprocess, "run", fake)
        return calls

    def test_version_probe(self, monkeypatch, active):
        calls = self._capture(monkeypatch, dp, "abc\nlivespice:x\n")
        dp.probe_worker_version("h", "/r", "livespice")
        assert calls[0][:3] == ["ssh", "-F", str(active)] and calls[0][-2] == "h"

    def test_cpu_topology_remote(self, monkeypatch, active):
        calls = self._capture(monkeypatch, cpu_topology, "LINUX\nprocessor : 0\n")
        cpu_topology.physical_cpu_count("h")
        assert calls[0][:3] == ["ssh", "-F", str(active)]

    def test_cpu_topology_nproc_fallback(self, monkeypatch, active):
        calls = []
        def fake(argv, *a, **kw):
            calls.append(list(argv))
            if "nproc" in argv:
                return _cp(0, "6\n")
            raise RuntimeError("primary probe fails")
        monkeypatch.setattr(cpu_topology.subprocess, "run", fake)
        assert cpu_topology.physical_cpu_count("h") == 6
        assert calls[-1][:3] == ["ssh", "-F", str(active)] and calls[-1][-1] == "nproc"

    def test_gate_wav_sync_ssh_and_rsync(self, monkeypatch, tmp_path, active):
        wav = tmp_path / "amps" / "e.wav"
        wav.parent.mkdir()
        wav.write_bytes(b"RIFF")
        monkeypatch.setattr(gc, "HERE", tmp_path)
        calls = self._capture(monkeypatch, gc)
        gc.sync_excitation_wav({"input": str(wav)}, ["h"], {"h": "/r"})
        ssh_calls = [c for c in calls if c[0] == "ssh"]
        assert len(ssh_calls) >= 2 and all(c[:3] == ["ssh", "-F", str(active)] for c in ssh_calls)
        rsync = next(c for c in calls if c[0] == "rsync")
        assert rsync[:4] == ["rsync", "-a", "-e", f"ssh -F {active}"]

    @pytest.mark.parametrize("fn", ["_collect", "_collect_grid_adequacy",
                                    "_collect_check_transient_coverage", "_collect_measure_truncation"])
    def test_collect_rsync(self, monkeypatch, tmp_path, active, fn):
        calls = self._capture(monkeypatch, dp)
        w = type("W", (), {"host": "h"})()
        try:
            getattr(dp, fn)([w], ["/out"], tmp_path / "d", None, ())
        except BaseException:
            pass        # nothing real to merge -- only the rsync argv matters here
        rs = [c for c in calls if c[0] == "rsync"]
        assert rs and all(c[2:4] == ["-e", f"ssh -F {active}"] for c in rs)

    def test_no_bare_ssh_or_rsync_left_in_fleet_modules(self):
        # a NEW call site added without ssh_target would silently ignore the inventory
        import re
        bare = re.compile(r'\[\s*"(ssh|rsync)"\s*,\s*(?!.*ssh_target)')
        for name in ("distribute_pull.py", "gate_config.py", "cpu_topology.py"):
            for i, line in enumerate((HERE / name).read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if 'subprocess.run(["ssh"' in line or 'Popen(["ssh"' in line or "[\"ssh\", \"-o\"" in line:
                    pytest.fail(f"{name}:{i} builds a bare ssh argv: {line.strip()}")
                if '["rsync", "-a",' in line and "rsync_e" not in line:
                    pytest.fail(f"{name}:{i} builds an rsync argv without rsync_e(): {line.strip()}")


class TestDispatchPassesInventory:
    def test_gate_dispatch_command_includes_inventory_when_given(self, tmp_path):
        inv = tmp_path / "fleet.toml"
        cmd, _ = gc.grid_command_fleet(Path("c.toml"), ["h"], {"h": "/r"}, inventory=inv)
        assert cmd[cmd.index("--inventory") + 1] == str(inv)

    def test_gate_dispatch_command_omits_it_without_one(self):
        cmd, _ = gc.transient_command_fleet(Path("c.toml"), ["h"], {"h": "/r"})
        assert "--inventory" not in cmd

    def test_pipeline_command_includes_inventory(self, tmp_path):
        inv = tmp_path / "fleet.toml"
        cmd = rp.fleet_generate_command(Path("c.toml"), tmp_path, ["h"], {"h": "/r"}, None, None, inv)
        assert cmd[cmd.index("--inventory") + 1] == str(inv)

    def test_pipeline_command_omits_it_without_one(self, tmp_path):
        cmd = rp.fleet_generate_command(Path("c.toml"), tmp_path, ["h"], {"h": "/r"}, None, None)
        assert "--inventory" not in cmd


class TestFleetContext:
    def test_existing_inventory_is_configured_and_returned(self, tmp_path, monkeypatch):
        inv = tmp_path / "fleet.toml"
        inv.write_text('[hosts."h"]\nrepo = "~/r"\nuser = "u"\n')
        ctx = gc.fleet_context(["h"], str(inv))
        assert ctx["inventory"] == inv and ctx["repo_dirs"] == {"h": "~/r"}
        assert st.active_config() is not None

    def test_dry_run_computes_the_same_answer_without_configuring(self, tmp_path):
        inv = tmp_path / "fleet.toml"
        inv.write_text('[hosts."h"]\nuser = "u"\n')
        ctx = gc.fleet_context(["h"], str(inv), configure=False)
        assert ctx["inventory"] == inv and st.active_config() is None

    def test_absent_inventory_file_gives_none_and_stays_plain_ssh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gc.fleet_inventory, "default_inventory_path",
                            lambda: tmp_path / "nope.toml")
        ctx = gc.fleet_context(["h"], None)
        assert ctx["inventory"] is None and st.active_config() is None
        assert ctx["repo_dirs"] == {"h": "~/work/parametric-nam"}


class TestDistributePullInventoryFlag:
    def test_flag_exists_and_bare_form_is_accepted(self):
        r = subprocess.run([sys.executable, str(HERE / "distribute_pull.py"), "--help"],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and "--inventory [INVENTORY]" in r.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestSyncScriptSshConfig:
    """sync_findpeak_cache.sh is a shell script: run it for real with fake ssh/rsync on PATH
    that record their argv and env."""

    def _run(self, tmp_path, *extra):
        bin_ = tmp_path / "bin"
        bin_.mkdir()
        log = tmp_path / "calls.log"
        for tool in ("ssh", "rsync"):
            f = bin_ / tool
            f.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{log}"\n'
                         f'echo "RSYNC_RSH=$RSYNC_RSH" >> "{log}"\n'
                         '[ "$1" = "-F" ] || case "$*" in *echo*HOME*) echo /home/x;; esac\n'
                         'case "$*" in *"echo \\$HOME"*|*\'echo $HOME\'*) echo /home/x;; esac\n')
            f.chmod(0o755)
        env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", HOME=str(tmp_path))
        r = subprocess.run(["bash", str(HERE / "sync_findpeak_cache.sh"), "--workers", "h1",
                            *extra], capture_output=True, text=True, env=env, timeout=60)
        return r, log.read_text() if log.exists() else ""

    def test_with_ssh_config_every_ssh_call_and_rsync_rsh_carry_it(self, tmp_path):
        cfg = tmp_path / "my config.conf"
        r, log = self._run(tmp_path, "--ssh-config", str(cfg))
        assert r.returncode == 0, r.stderr
        ssh_lines = [l for l in log.splitlines() if l.startswith("ssh ")]
        assert ssh_lines and all(l.startswith(f"ssh -F {cfg} ") for l in ssh_lines)
        rsh = [l for l in log.splitlines() if l.startswith("RSYNC_RSH=")]
        assert rsh and all(f"-F '{cfg}'" in l for l in rsh)

    def test_without_it_nothing_changes(self, tmp_path):
        r, log = self._run(tmp_path)
        assert r.returncode == 0, r.stderr
        assert "-F" not in log


class TestConfigureSsh:
    def test_flag_absent_is_a_noop_leaving_plain_ssh(self):
        assert dp.configure_ssh(None) is None and st.active_config() is None

    def test_explicit_path_activates_config(self, tmp_path):
        inv = tmp_path / "f.toml"
        inv.write_text('[hosts."a"]\nuser = "u"\n')
        cfg = dp.configure_ssh(str(inv))
        assert cfg is not None and st.active_config() == cfg

    def test_bare_flag_reads_the_default_inventory(self, tmp_path, monkeypatch):
        inv = tmp_path / "default.toml"
        inv.write_text('[hosts."a"]\nuser = "u"\n')
        import fleet_inventory
        monkeypatch.setattr(fleet_inventory, "default_inventory_path", lambda: inv)
        assert dp.configure_ssh("__DEFAULT__") is not None

    def test_inventory_with_nothing_to_say_stays_plain(self, tmp_path):
        inv = tmp_path / "f.toml"
        inv.write_text('[hosts."a"]\ncores = 4\n')
        assert dp.configure_ssh(str(inv)) is None
