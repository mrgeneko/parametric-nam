"""fleet_inventory.py: probing (with a faked ssh/local runner, no network), TOML rendering,
and the load/write round-trip. See docs/fleet-deployment-proposal.md \u00a72 and
docs/implementation-roadmap.md item 4.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fleet_inventory as fi  # noqa: E402


def cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr=err)


class FakeRun:
    """Fake `run(host, argv) -> CompletedProcess`. `answers` maps a predicate over argv to a
    result; the first match wins. Records every call for assertions."""
    def __init__(self, rules=None, default=None):
        self.rules = rules or []      # [(match_fn, CompletedProcess)]
        self.default = default if default is not None else cp(1, "", "no such command")
        self.calls = []

    def __call__(self, host, argv):
        self.calls.append((host, list(argv)))
        for match, result in self.rules:
            if match(argv):
                return result
        return self.default

    def when(self, needle, result):
        self.rules.append((lambda argv, n=needle: n in " ".join(argv), result))
        return self


# ---------------------------------------------------------------------------
# quoting for the ssh boundary -- see default_ssh's own docstring for the two real regressions
# (ngspice-deck missing remotely, `~` no longer expanding remotely) this section guards against
# ---------------------------------------------------------------------------
class TestSshQuote:
    def test_plain_word_is_untouched(self):
        assert fi._ssh_quote("ngspice") == "ngspice"

    def test_string_with_a_space_is_quoted(self):
        assert fi._ssh_quote("command -v ngspice") == "'command -v ngspice'"

    def test_leading_tilde_is_left_bare_so_the_remote_shell_expands_it(self):
        assert fi._ssh_quote("~/work/parametric-nam") == "~/work/parametric-nam"

    def test_tilde_user_form_is_also_left_bare(self):
        assert fi._ssh_quote("~gene/work") == "~gene/work"

    def test_tilde_path_with_unsafe_characters_after_it_still_gets_the_rest_quoted(self):
        out = fi._ssh_quote("~/My Work/repo")
        assert out.startswith("~") and "My Work" in out and "'" in out

    def test_bare_tilde_alone(self):
        assert fi._ssh_quote("~") == "~"

    def test_a_tilde_not_at_the_start_is_not_treated_specially(self):
        # Only a LEADING ~ is shell tilde-expansion syntax; mid-string it's an ordinary char
        # shlex.quote is free to wrap however it likes (it still does, since `~` isn't in its
        # own "safe" set) -- the only contract is that a shell parses the result back to the
        # original literal string, not that it comes back unquoted.
        import shlex as _shlex
        assert _shlex.split(fi._ssh_quote("a~b")) == ["a~b"]

    def test_parens_and_quotes_are_safely_wrapped(self):
        out = fi._ssh_quote("getattr(x, 'y')")
        assert out.count("(") <= out.count("'") // 2 + 1   # wrapped, not left bare


class TestDefaultSsh:
    def test_local_argv_is_passed_through_unquoted(self, monkeypatch):
        seen = {}
        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(fi.subprocess, "run", fake_run)
        fi.default_ssh(None, ["sh", "-lc", "command -v ngspice"])
        assert seen["cmd"] == ["sh", "-lc", "command -v ngspice"]

    def test_remote_argv_elements_are_individually_quoted(self, monkeypatch):
        seen = {}
        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(fi.subprocess, "run", fake_run)
        fi.default_ssh("mac-1", ["sh", "-lc", "command -v ngspice"])
        assert seen["cmd"][-3:] == ["sh", "-lc", "'command -v ngspice'"]

    def test_remote_tilde_paths_stay_expandable(self, monkeypatch):
        seen = {}
        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(fi.subprocess, "run", fake_run)
        fi.default_ssh("mac-1", ["test", "-e", "~/work/parametric-nam/.git"])
        assert seen["cmd"][-1] == "~/work/parametric-nam/.git"

    def test_unreachable_host_is_wrapped_not_raised_as_a_raw_oserror(self, monkeypatch):
        def boom(cmd, **kw):
            raise OSError("no route to host")
        monkeypatch.setattr(fi.subprocess, "run", boom)
        with pytest.raises(fi.Unreachable):
            fi.default_ssh("gone", ["test", "-e", "x"])


# ---------------------------------------------------------------------------
# individual probes
# ---------------------------------------------------------------------------
class TestProbeRepo:
    def test_hint_tried_first(self, tmp_path):
        run = FakeRun().when("test -e /custom/repo/.git", cp(0))
        assert fi.probe_repo(run, "h", "/custom/repo") == "/custom/repo"

    def test_falls_back_to_default_candidates(self):
        run = FakeRun().when(f"test -e {fi.DEFAULT_REPO_CANDIDATES[1]}/.git", cp(0))
        assert fi.probe_repo(run, "h", None) == fi.DEFAULT_REPO_CANDIDATES[1]

    def test_fleet_inventory_py_itself_counts_as_a_marker_with_no_git_dir(self):
        c = fi.DEFAULT_REPO_CANDIDATES[0]
        run = FakeRun().when(f"test -e {c}/fleet_inventory.py", cp(0))
        assert fi.probe_repo(run, "h", None) == c

    def test_nothing_found_is_none(self):
        assert fi.probe_repo(FakeRun(), "h", None) is None

    def test_local_uses_real_filesystem_not_the_runner(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        run = FakeRun()   # never answers "yes" -- local path must bypass it entirely
        assert fi.probe_repo(run, None, str(tmp_path)) == str(tmp_path)
        assert run.calls == []


class TestProbeBackends:
    def test_livespice_from_sibling_checkout(self):
        run = FakeRun().when("../livespice-cli/publish/livespice_cli", cp(0))
        b = fi.probe_backends(run, "h", "/repo")
        assert "livespice" in b and "ngspice-schx" in b

    def test_livespice_from_legacy_checkout(self):
        run = FakeRun().when("livespice_cli/publish/livespice_cli", cp(0))
        assert "livespice" in fi.probe_backends(run, "h", "/repo")

    def test_no_repo_no_livespice(self):
        run = FakeRun().when("livespice_cli", cp(0))   # would match if repo were used anyway
        assert "livespice" not in fi.probe_backends(run, "h", None)

    def test_ngspice_deck_from_path(self):
        run = FakeRun().when("command -v ngspice", cp(0, "/usr/bin/ngspice\n"))
        assert "ngspice-deck" in fi.probe_backends(run, "h", None)

    def test_which_empty_stdout_is_not_found(self):
        run = FakeRun().when("command -v ngspice", cp(0, ""))   # rc 0 but empty -- treat as absent
        assert "ngspice-deck" not in fi.probe_backends(run, "h", None)

    def test_ltspice_deck_from_either_candidate(self):
        for cand in fi.LTSPICE_BIN_CANDIDATES:
            run = FakeRun().when(cand, cp(0))
            assert "ltspice-deck" in fi.probe_backends(run, "h", None)

    def test_nothing_found_is_empty_not_a_crash(self):
        assert fi.probe_backends(FakeRun(), "h", "/repo") == []


class TestProbeAccelerator:
    def test_no_repo_is_undetermined_with_a_reason(self):
        out = fi.probe_accelerator(FakeRun(), "h", None)
        assert out["accelerator"] is None and "repo" in out["note"]

    def test_no_venv_is_undetermined_with_a_reason(self):
        run = FakeRun()   # every `test -e` fails
        out = fi.probe_accelerator(run, "h", "/repo")
        assert out["accelerator"] is None and "not found" in out["note"]

    def _venv_run(self, exec_result):
        """A venv exists (`test -e ...python3` -> rc 0); the exec call (argv contains "-c")
        gets `exec_result`. Rule order matters: the exec call's argv0 still CONTAINS
        ".venv/bin/python3" as a substring, so the "-c" rule must be checked FIRST or it would
        never be reached."""
        return FakeRun().when("-c", exec_result).when(".venv/bin/python3", cp(0))

    def test_cuda_with_vram_and_gpu_count(self):
        run = self._venv_run(cp(0, '{"accelerator": "cuda", "gpus": 1, "vram_gb": 16}\n'))
        out = fi.probe_accelerator(run, "h", "/repo")
        assert out == {"accelerator": "cuda", "gpus": 1, "vram_gb": 16}

    def test_rocm_distinguished_from_cuda(self):
        run = self._venv_run(cp(0, '{"accelerator": "rocm", "gpus": 1}\n'))
        assert fi.probe_accelerator(run, "h", "/repo")["accelerator"] == "rocm"

    def test_mps_has_no_vram(self):
        run = self._venv_run(cp(0, '{"accelerator": "mps"}\n'))
        out = fi.probe_accelerator(run, "h", "/repo")
        assert out["accelerator"] == "mps" and "vram_gb" not in out

    def test_cpu_only_is_the_real_answer_none_not_undetermined(self):
        run = self._venv_run(cp(0, '{"accelerator": "none"}\n'))
        assert fi.probe_accelerator(run, "h", "/repo")["accelerator"] == "none"

    def test_nonzero_exit_is_undetermined(self):
        run = self._venv_run(cp(1, "", "ImportError: no module named torch"))
        out = fi.probe_accelerator(run, "h", "/repo")
        assert out["accelerator"] is None and "probe failed" in out["note"]

    def test_torch_missing_is_reported_not_crashed(self):
        run = self._venv_run(cp(0, '{"accelerator": "unavailable", "error": "ImportError: torch"}\n'))
        out = fi.probe_accelerator(run, "h", "/repo")
        assert out["accelerator"] is None and "ImportError" in out["note"]

    def test_garbage_output_does_not_crash(self):
        out = fi.probe_accelerator(self._venv_run(cp(0, "not json")), "h", "/repo")
        assert out["accelerator"] is None

    def test_local_expands_tilde_before_exec(self, monkeypatch):
        """The real bug this caught: subprocess.run does not shell-expand `~`, so the local
        (host=None) path must expand it itself before using it as argv[0]. Isolated from
        _test()'s own filesystem check (stubbed True) so this tests only the expansion."""
        monkeypatch.setattr(fi, "_test", lambda run, host, path: True)
        seen = {}
        def run(host, argv):
            seen["argv0"] = argv[0]
            return cp(0, '{"accelerator": "none"}\n')
        fi.probe_accelerator(run, None, "~/repo")
        expected = str(Path("~/repo/.venv/bin/python3").expanduser())
        assert "~" not in seen["argv0"] and seen["argv0"] == expected

    def test_remote_leaves_tilde_for_the_remote_shell_to_expand(self):
        seen = {}
        def run(host, argv):
            seen["argv0"] = argv[0]
            return cp(0, '{"accelerator": "cuda"}\n') if "-c" in argv else cp(0)
        fi.probe_accelerator(run, "remote-host", "~/repo")
        assert seen["argv0"] == "~/repo/.venv/bin/python3"


class TestResolveSshConfig:
    """user/identity_file: resolved from the LOCAL machine's own `ssh -G`, never over the
    remote `run` channel the other probes use -- see the function's own docstring for why."""

    def _local_run(self, stdout, rc=0):
        calls = []
        def run(argv):
            calls.append(argv)
            return cp(rc, stdout)
        run.calls = calls
        return run

    def test_single_identity_file_is_recorded(self):
        run = self._local_run("user gene\nhostname 1.2.3.4\nidentityfile ~/.ssh/id_ed25519_fleet\n")
        assert fi.resolve_ssh_config("mbp", run) == {"user": "gene",
                                                      "identity_file": "~/.ssh/id_ed25519_fleet"}

    def test_multiple_identity_files_are_ambiguous_and_omitted(self):
        # OpenSSH's own built-in fallback list (nothing explicitly configured for this alias) --
        # this Air's real `ssh -G localhost` output, five defaults, no single answer.
        stdout = "user gene\n" + "".join(f"identityfile ~/.ssh/{k}\n"
                                         for k in ("id_rsa", "id_ecdsa", "id_ecdsa_sk",
                                                  "id_ed25519", "id_ed25519_sk"))
        out = fi.resolve_ssh_config("localhost", self._local_run(stdout))
        assert out == {"user": "gene"}   # user is unambiguous even when identity isn't

    def test_no_identityfile_line_at_all_is_just_the_user(self):
        run = self._local_run("user gene\nhostname h\n")
        assert fi.resolve_ssh_config("h", run) == {"user": "gene"}

    def test_ssh_failing_is_empty_not_a_crash(self):
        assert fi.resolve_ssh_config("h", self._local_run("", rc=255)) == {}

    def test_ssh_binary_missing_is_empty_not_a_crash(self):
        def boom(argv):
            raise FileNotFoundError("ssh")
        assert fi.resolve_ssh_config("h", boom) == {}

    def test_garbled_output_is_empty_not_a_crash(self):
        assert fi.resolve_ssh_config("h", self._local_run("not key-value\n")) == {}

    def test_always_runs_ssh_dash_g_not_a_real_connection(self):
        run = self._local_run("user gene\n")
        fi.resolve_ssh_config("mbp", run)
        assert run.calls == [["ssh", "-G", "mbp"]]

    def test_real_localhost_lookup_does_not_crash(self):
        """Exercises the real subprocess path (no fake), against a target every machine has."""
        out = fi.resolve_ssh_config("localhost")
        assert isinstance(out, dict)   # whatever this machine's own ssh -G says is fine either way


class TestProbeHostSshConfig:
    def test_ssh_config_is_merged_for_a_remote_target(self, monkeypatch):
        monkeypatch.setattr(fi, "physical_cpu_count", lambda host: 4)
        monkeypatch.setattr(fi, "resolve_ssh_config", lambda host, local_run=None: {"user": "gene"})
        out = fi.probe_host(FakeRun(), "mbp", address="mbp")
        assert out["user"] == "gene"

    def test_local_self_entry_never_calls_resolve_ssh_config(self, monkeypatch):
        monkeypatch.setattr(fi, "physical_cpu_count", lambda host: 4)
        def boom(host, local_run=None):
            raise AssertionError("resolve_ssh_config should not be called for the local host")
        monkeypatch.setattr(fi, "resolve_ssh_config", boom)
        fi.probe_host(FakeRun(), None, address="me")   # must not raise


class TestEnvHint:
    def test_dotnet_on_path_needs_no_hint(self):
        run = FakeRun().when("command -v dotnet", cp(0, "/usr/bin/dotnet\n"))
        assert fi.env_hint(run, "h") == {}

    def test_missing_dotnet_found_at_conventional_path(self):
        run = FakeRun().when("~/.dotnet/dotnet", cp(0))   # command -v dotnet falls to default (rc 1)
        assert fi.env_hint(run, "h") == {"DOTNET_ROOT": "~/.dotnet"}

    def test_missing_everywhere_is_no_hint(self):
        assert fi.env_hint(FakeRun(), "h") == {}


# ---------------------------------------------------------------------------
# probe_host: composition
# ---------------------------------------------------------------------------
class TestProbeHost:
    def test_train_is_always_false(self, monkeypatch):
        monkeypatch.setattr(fi, "physical_cpu_count", lambda host: 8)
        out = fi.probe_host(FakeRun(), "h", address="h")
        assert out["train"] is False

    def test_cores_comes_from_cpu_topology_not_reimplemented(self, monkeypatch):
        calls = []
        monkeypatch.setattr(fi, "physical_cpu_count", lambda host: calls.append(host) or 12)
        out = fi.probe_host(FakeRun(), "some-host", address="some-host")
        assert out["cores"] == 12 and calls == ["some-host"]

    def test_missing_repo_is_noted_not_silently_dropped(self, monkeypatch):
        monkeypatch.setattr(fi, "physical_cpu_count", lambda host: 4)
        out = fi.probe_host(FakeRun(), "h", address="h")
        assert out["repo"] is None and "not found" in out["note"]

    def test_address_is_recorded_verbatim(self, monkeypatch):
        monkeypatch.setattr(fi, "physical_cpu_count", lambda host: 4)
        assert fi.probe_host(FakeRun(), "h", address="mac-1.tailnet.ts.net")["address"] == "mac-1.tailnet.ts.net"


# ---------------------------------------------------------------------------
# TOML rendering + round trip
# ---------------------------------------------------------------------------
class TestRenderToml:
    def full_host(self, **over):
        h = {"address": "mac-1", "repo": "~/work/parametric-nam", "cores": 8,
             "backends": ["livespice", "ngspice-deck"], "accelerator": "mps", "train": False}
        h.update(over)
        return h

    def test_round_trips_through_load(self, tmp_path):
        text = fi.render_toml({"mac-1": self.full_host()})
        p = tmp_path / "fleet.toml"
        p.write_text(text)
        loaded = fi.load_inventory(p)
        assert loaded["mac-1"]["cores"] == 8
        assert loaded["mac-1"]["backends"] == ["livespice", "ngspice-deck"]
        assert loaded["mac-1"]["accelerator"] == "mps"
        assert loaded["mac-1"]["train"] is False

    def test_undetermined_accelerator_is_commented_not_written_as_a_value(self, tmp_path):
        text = fi.render_toml({"h": self.full_host(accelerator=None, note="no venv")})
        assert "accelerator  = " not in text
        assert "not determined" in text
        p = tmp_path / "fleet.toml"; p.write_text(text)
        assert "accelerator" not in fi.load_inventory(p)["h"]

    def test_real_none_accelerator_is_written_as_the_string_none(self, tmp_path):
        text = fi.render_toml({"h": self.full_host(accelerator="none")})
        assert 'accelerator  = "none"' in text
        p = tmp_path / "fleet.toml"; p.write_text(text)
        assert fi.load_inventory(p)["h"]["accelerator"] == "none"

    def test_missing_repo_has_no_repo_key_but_keeps_the_note(self, tmp_path):
        h = self.full_host(repo=None, note="repo not found -- set by hand")
        text = fi.render_toml({"h": h})
        p = tmp_path / "fleet.toml"; p.write_text(text)
        assert "repo" not in fi.load_inventory(p)["h"]
        assert "not found" in text

    def test_vram_and_gpus_only_appear_when_known(self, tmp_path):
        text = fi.render_toml({"h": self.full_host(vram_gb=16, gpus=1)})
        p = tmp_path / "fleet.toml"; p.write_text(text)
        loaded = fi.load_inventory(p)["h"]
        assert loaded["vram_gb"] == 16 and loaded["gpus"] == 1
        text2 = fi.render_toml({"h": self.full_host()})
        assert "vram_gb" not in text2 and "gpus " not in text2

    def test_env_only_appears_when_present(self, tmp_path):
        text = fi.render_toml({"h": self.full_host(env={"DOTNET_ROOT": "~/.dotnet"})})
        p = tmp_path / "fleet.toml"; p.write_text(text)
        assert fi.load_inventory(p)["h"]["env"] == {"DOTNET_ROOT": "~/.dotnet"}
        assert "env" not in fi.render_toml({"h": self.full_host()})

    def test_user_and_identity_file_only_appear_when_present(self, tmp_path):
        text = fi.render_toml({"h": self.full_host(user="gene",
                                                    identity_file="~/.ssh/id_ed25519_fleet")})
        p = tmp_path / "fleet.toml"; p.write_text(text)
        loaded = fi.load_inventory(p)["h"]
        assert loaded["user"] == "gene" and loaded["identity_file"] == "~/.ssh/id_ed25519_fleet"
        text2 = fi.render_toml({"h": self.full_host()})
        assert "user " not in text2 and "identity_file" not in text2

    def test_names_with_hyphens_and_dots_are_valid_toml_keys(self, tmp_path):
        text = fi.render_toml({"mac-1.tailnet": self.full_host()})
        p = tmp_path / "fleet.toml"; p.write_text(text)
        assert "mac-1.tailnet" in fi.load_inventory(p)

    def test_quotes_and_backslashes_in_strings_are_escaped(self, tmp_path):
        text = fi.render_toml({"h": self.full_host(repo='C:\\path\\with"quote')})
        p = tmp_path / "fleet.toml"; p.write_text(text)
        assert fi.load_inventory(p)["h"]["repo"] == 'C:\\path\\with"quote'

    def test_hosts_are_sorted_for_stable_diffs(self):
        text = fi.render_toml({"zeta": self.full_host(), "alpha": self.full_host()})
        assert text.index('[hosts."alpha"]') < text.index('[hosts."zeta"]')


class TestLoadInventory:
    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert fi.load_inventory(tmp_path / "nope.toml") == {}

    def test_default_path_matches_the_findpeak_cache_convention(self):
        p = fi.default_inventory_path()
        assert p == Path.home() / ".config" / "parametric-nam" / "fleet.toml"


# ---------------------------------------------------------------------------
# CLI (main())
# ---------------------------------------------------------------------------
class TestMain:
    def _patch(self, monkeypatch, hosts_by_name):
        def fake_probe_host(run, ssh_host, *, address, repo_hint=None):
            name = address
            if name in hosts_by_name:
                return hosts_by_name[name]
            raise fi.Unreachable(f"{name}: simulated failure")
        monkeypatch.setattr(fi, "probe_host", fake_probe_host)

    def host(self):
        return {"address": "h", "repo": "/r", "cores": 4, "backends": [], "train": False}

    def test_writes_to_given_inventory_path(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, {"me": self.host()})
        out = tmp_path / "out.toml"
        monkeypatch.setattr(sys, "argv", ["fleet_inventory.py", "--probe-hosts", "--self", "me",
                                          "--inventory", str(out)])
        assert fi.main() == 0
        assert "me" in fi.load_inventory(out)

    def test_no_self_and_no_worker_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["fleet_inventory.py", "--probe-hosts", "--no-self"])
        assert fi.main() == 2

    def test_worker_spec_splits_host_and_repo_hint(self, tmp_path, monkeypatch):
        seen = {}
        def fake_probe_host(run, ssh_host, *, address, repo_hint=None):
            seen["args"] = (ssh_host, address, repo_hint)
            return self.host()
        monkeypatch.setattr(fi, "probe_host", fake_probe_host)
        out = tmp_path / "out.toml"
        monkeypatch.setattr(sys, "argv", ["fleet_inventory.py", "--probe-hosts", "--no-self",
                                          "--worker", "linux-1:~/render/parametric-nam",
                                          "--inventory", str(out)])
        assert fi.main() == 0
        assert seen["args"] == ("linux-1", "linux-1", "~/render/parametric-nam")

    def test_unreachable_worker_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, {"me": self.host()})   # "bad" is not in the dict -> Unreachable
        out = tmp_path / "out.toml"
        monkeypatch.setattr(sys, "argv", ["fleet_inventory.py", "--probe-hosts", "--self", "me",
                                          "--worker", "bad", "--inventory", str(out)])
        assert fi.main() == 0
        assert "me" in fi.load_inventory(out) and "bad" not in fi.load_inventory(out)

    def test_all_unreachable_is_exit_1(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, {})
        out = tmp_path / "out.toml"
        monkeypatch.setattr(sys, "argv", ["fleet_inventory.py", "--probe-hosts", "--worker", "bad",
                                          "--no-self", "--inventory", str(out)])
        assert fi.main() == 1

    def test_rewriting_overwrites_stale_hosts(self, tmp_path, monkeypatch):
        out = tmp_path / "out.toml"
        self._patch(monkeypatch, {"a": self.host()})
        monkeypatch.setattr(sys, "argv", ["fleet_inventory.py", "--probe-hosts", "--self", "a",
                                          "--inventory", str(out)])
        fi.main()
        self._patch(monkeypatch, {"b": self.host()})
        monkeypatch.setattr(sys, "argv", ["fleet_inventory.py", "--probe-hosts", "--self", "b",
                                          "--inventory", str(out)])
        fi.main()
        loaded = fi.load_inventory(out)
        assert "b" in loaded and "a" not in loaded
