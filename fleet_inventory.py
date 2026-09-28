#!/usr/bin/env python3
"""Generate and load a per-host fleet inventory -- roadmap item 4
(docs/implementation-roadmap.md), fleet-deployment-proposal.md's §2.

    python fleet_inventory.py --probe-hosts --worker mac-1 --worker linux-1:~/render/parametric-nam
    python fleet_inventory.py --probe-hosts --no-self --inventory ~/my-fleet.toml

Probes each host over SSH (or in-process for the local machine) for the facts the fleet doc
says were "rediscovered by hand" during a real sharded render: physical core count, which
backends the oracle/simulators support, accelerator and VRAM, where the repo checkout lives,
and (for a `--worker` target, resolved from the LOCAL machine's own `ssh -G`, not probed
remotely) the login user and identity file this machine would actually use to reach it --
closing the gap roadmap item 1 left open, that address/key/user setup stayed hand-maintained
in each machine's own ~/.ssh/config. Writes a reviewable TOML file, the way scaffold_config.py
emits a device config: measured and annotated, not asserted -- a fact this tool could not
determine is recorded as such (absent, or a comment), never silently guessed.

WHAT IS NOT PROBED, on purpose (annotated in the file instead, for the operator to fill in):
  * `train` -- always written `false`. The proposal is explicit that this is deliberately NOT
    implied by having a GPU (a laptop that sleeps, or the controller itself, is a poor training
    host despite the hardware). A human decision, every time.
  * `max_render_s` -- needs an actual timed render of a real circuit; omitted here.
  * `$LIVESPICE_CLI`/`$DOTNET_ROOT` as they'd appear in an interactive shell -- a non-interactive
    SSH command does not reliably source the same profile a login shell would. DOTNET_ROOT gets
    one filesystem-based heuristic (see env_hint below); LIVESPICE_CLI does not, since unlike
    dotnet's install location there is no single conventional path to guess.

Design: every remote fact is gathered through one injected `ssh(host, argv) -> CompletedProcess`
call per probe -- raw shell commands, the same convention cpu_topology.physical_cpu_count already
uses for its own remote probe (reused here directly, not re-derived), rather than shipping and
self-invoking this script remotely: at inventory-generation time the repo may not even be cloned
on the target yet (see the onboarding-cost table in fleet-deployment-proposal.md), so a probe
that assumes its own presence there begs the question it exists to answer. The one exception is
the accelerator/VRAM probe, which genuinely needs Python (torch) and so runs through the
worker's own venv -- the SAME assumption gen_dataset_from_schx.py already makes about every
worker (a built venv is an onboarding prerequisite, not something this tool re-verifies).
"""
import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cpu_topology import physical_cpu_count  # noqa: E402

SCHEMA = 1
DEFAULT_REPO_CANDIDATES = ("~/work/parametric-nam", "~/parametric-nam")
# Mirrors gen_dataset_from_schx._find_livespice_cli's own two filesystem locations (the
# $LIVESPICE_CLI env-var branch is skipped remotely -- see the module docstring).
LIVESPICE_CLI_SUFFIXES = ("../livespice-cli/publish/livespice_cli", "livespice_cli/publish/livespice_cli")
# Mirrors ltspice_spicelib._find_ltspice_bin's own two candidates (env var skipped, same reason).
LTSPICE_BIN_CANDIDATES = ("~/Applications/LTspice.app/Contents/MacOS/LTspice",
                          "/Applications/LTspice.app/Contents/MacOS/LTspice")

ACCEL_PROBE = (
    "import json,sys\n"
    "try:\n"
    "    import torch\n"
    "    cuda = torch.cuda.is_available()\n"
    "    rocm = cuda and getattr(torch.version, 'hip', None) is not None\n"
    "    mps = (not cuda) and bool(getattr(getattr(torch.backends, 'mps', None), "
    "'is_available', lambda: False)())\n"
    "    out = {'accelerator': 'rocm' if rocm else 'cuda' if cuda else 'mps' if mps else 'none'}\n"
    "    if cuda:\n"
    "        out['gpus'] = torch.cuda.device_count()\n"
    "        try:\n"
    "            out['vram_gb'] = int(torch.cuda.get_device_properties(0).total_memory // (1024**3))\n"
    "        except Exception:\n"
    "            pass\n"
    "    print(json.dumps(out))\n"
    "except Exception as e:\n"
    "    print(json.dumps({'accelerator': 'unavailable', 'error': f'{type(e).__name__}: {e}'}))\n"
)


class Unreachable(Exception):
    """The host could not be probed at all (SSH failed outright)."""


_TILDE_PREFIX = re.compile(r"^~[\w.-]*")   # `~` or `~user`, POSIX tilde-prefix syntax


def _ssh_quote(s: str) -> str:
    """shlex.quote(), except a LEADING `~`/`~user` is left OUTSIDE the quotes so the remote
    shell still tilde-expands it -- plain shlex.quote() wraps the whole string (tilde included)
    in literal single quotes, which suppresses every shell expansion, tilde included. Found via
    a second real regression from the fix below: DEFAULT_REPO_CANDIDATES ("~/work/parametric-
    nam", ...) started reporting "not found" against a real remote target the moment argv
    elements were quoted uniformly, because `test -e '~/work/parametric-nam/.git'` tests for a
    file literally named with a tilde character, not the home directory. For any string that
    does not start with `~`, this is identical to shlex.quote().
    """
    m = _TILDE_PREFIX.match(s)
    if not m:
        return shlex.quote(s)
    prefix, rest = m.group(0), s[m.end():]
    return prefix + shlex.quote(rest) if rest else prefix


def default_ssh(host: "str | None", argv: "list[str]", timeout: float = 15.0):
    """The real runner: local subprocess, or `ssh host <argv>` -- mirrors
    cpu_topology.physical_cpu_count's own host=None-means-local convention.

    Remote argv elements are individually quoted with _ssh_quote (shlex.quote, tilde-preserving
    -- see its own docstring). ssh does not preserve argv-element boundaries onto the remote
    host: everything after `host` is joined with plain spaces into ONE string and handed to the
    remote login shell to re-parse from scratch (see ssh(1)) -- without per-element quoting, an
    argv element containing whitespace or shell metacharacters silently loses its grouping.
    Confirmed two ways this actually broke, not just in theory: probing `localhost` as a remote
    target (`--worker localhost`) found ngspice-deck missing where the LOCAL self-probe of the
    identical machine found it present -- `_which`'s ["sh", "-lc", "command -v ngspice"]
    rejoined into `sh -lc command -v ngspice` remotely, so `-lc`'s one required argument became
    just "command", not "command -v ngspice"; and the accelerator probe's raw Python source
    (parens, quotes) was parsed as shell syntax outright ("zsh: parse error near ')'"). A local
    subprocess.run(list) never goes through a shell at all, so this only bites the remote
    branch -- local argv elements are passed through as-is.
    """
    cmd = (list(argv) if host is None else
          ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host,
           *[_ssh_quote(a) for a in argv]])
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise Unreachable(f"{host or 'localhost'}: timed out ({e})") from e
    except OSError as e:
        raise Unreachable(f"{host or 'localhost'}: {e}") from e


def _test(run, host, path: str) -> bool:
    """`test -e path` (or -e a locally-expanded Path when host is None, since `~` needs the
    REMOTE shell to expand it but Python can expand it directly and more reliably for local)."""
    if host is None:
        return Path(path).expanduser().exists()
    r = run(host, ["test", "-e", path])
    return r.returncode == 0


def _which(run, host, prog: str) -> bool:
    r = run(host, ["sh", "-lc", f"command -v {shlex.quote(prog)}"])
    return r.returncode == 0 and r.stdout.strip() != ""


def probe_repo(run, host, hint: "str | None") -> "str | None":
    """The repo checkout path, or None if nothing at any candidate looked like this repo (a
    `.git` dir, or -- since a worker's checkout might be a subtree/export without one -- this
    very file, fleet_inventory.py, present). hint is tried first."""
    for cand in ([hint] if hint else []) + list(DEFAULT_REPO_CANDIDATES):
        if _test(run, host, f"{cand}/.git") or _test(run, host, f"{cand}/fleet_inventory.py"):
            return cand
    return None


def probe_cores(host) -> int:
    return physical_cpu_count(host)   # never raises; degrades to a usable number on its own


def probe_backends(run, host, repo: "str | None") -> "list[str]":
    backends = []
    if repo and any(_test(run, host, f"{repo}/{suf}") for suf in LIVESPICE_CLI_SUFFIXES):
        backends += ["livespice", "ngspice-schx"]   # one oracle serves both, see prepare_excitation.py
    if _which(run, host, "ngspice"):
        backends.append("ngspice-deck")
    if any(_test(run, host, c) for c in LTSPICE_BIN_CANDIDATES):
        backends.append("ltspice-deck")
    return backends


def probe_accelerator(run, host, repo: "str | None") -> dict:
    """Returns {'accelerator': ..., 'gpus': ..., 'vram_gb': ...} (only the keys that were
    determined). Needs the worker's own venv -- see module docstring for why this is the one
    probe that runs Python remotely rather than a raw shell command."""
    if not repo:
        return {"accelerator": None, "note": "repo not found; accelerator not probed"}
    py = f"{repo}/.venv/bin/python3"
    if not _test(run, host, py):
        return {"accelerator": None, "note": f"{py} not found; accelerator not probed"}
    # ssh runs argv through the remote user's shell, which expands `~`; a local subprocess.run
    # does not (no shell involved), so `py` must be expanded by hand for the host=None case --
    # the exact asymmetry _test() already hides for its own existence check, above.
    # ssh runs argv through the remote user's shell, which expands `~`; a local subprocess.run
    # does not (no shell involved), so `py` must be expanded by hand for the host=None case --
    # the exact asymmetry _test() already hides for its own existence check, above. Passing
    # ACCEL_PROBE's raw Python source as one argv element is safe on BOTH paths: locally it
    # reaches -c untouched (no shell involved at all), and remotely `run` (default_ssh, unless
    # overridden) is responsible for quoting each argv element before it crosses the ssh
    # boundary -- see default_ssh's own docstring for why that quoting has to live there, not
    # per call site.
    py_run = str(Path(py).expanduser()) if host is None else py
    r = run(host, [py_run, "-c", ACCEL_PROBE])
    if r.returncode != 0 or not r.stdout.strip():
        return {"accelerator": None, "note": f"accelerator probe failed: {(r.stderr or '').strip()[:200]}"}
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"accelerator": None, "note": "accelerator probe returned unparseable output"}
    if out.get("accelerator") == "unavailable":
        return {"accelerator": None, "note": out.get("error", "torch unavailable")}
    return out


def resolve_ssh_config(alias: str, local_run=None) -> dict:
    """`user`/`identity_file` for `alias`, resolved from the LOCAL machine's own SSH client
    config (`ssh -G`, no connection made -- it's OpenSSH itself reporting what it would use,
    the same convention the mini fleet session already relies on by hand via `~/.ssh/config`
    Host aliases). Always LOCAL, unlike every other probe in this module: it answers "how
    would *I* reach `alias`", not a fact about the remote host itself, so it is never routed
    through the remote `run` callable those probes use.

    `identity_file` is recorded only when `ssh -G` resolves to EXACTLY ONE IdentityFile. More
    than one means nothing was explicitly configured for this alias -- OpenSSH's own built-in
    fallback list (id_rsa, id_ecdsa, id_ed25519, ...) it tries in order -- which is not a fact
    about this host worth writing down (a fleet that has not pinned a key here gains nothing
    from an inventory line that will be wrong the day the operator's default keys change).
    `user` is always recorded when resolvable -- ssh -G never leaves it ambiguous, it is either
    explicitly configured or the local login name.
    """
    run = local_run or (lambda argv: subprocess.run(argv, capture_output=True, text=True,
                                                     timeout=8))
    try:
        r = run(["ssh", "-G", alias])
    except Exception:
        return {}
    if r.returncode != 0:
        return {}
    user = None
    identity_files = []
    for line in r.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts
        if key == "user":
            user = val
        elif key == "identityfile":
            identity_files.append(val)
    out = {}
    if user:
        out["user"] = user
    if len(identity_files) == 1:
        out["identity_file"] = identity_files[0]
    return out


def env_hint(run, host) -> dict:
    """The one env heuristic this tool ventures: dotnet missing from a non-interactive PATH but
    present at its conventional install location. Everything else in `env` is left to the
    operator -- see module docstring."""
    if _which(run, host, "dotnet"):
        return {}
    if _test(run, host, "~/.dotnet/dotnet"):
        return {"DOTNET_ROOT": "~/.dotnet"}
    return {}


def probe_host(run, host: "str | None", *, address: str, repo_hint: "str | None" = None) -> dict:
    """One host's full fact set. `host` is the ssh target (None = local). Never raises for a
    single failed sub-probe -- an Unreachable host-level failure from `run` DOES propagate, since
    at that point nothing about the host can be recorded at all."""
    repo = probe_repo(run, host, repo_hint)
    facts = {
        "address": address,
        "repo": repo,
        "cores": probe_cores(host),
        "backends": probe_backends(run, host, repo),
        "train": False,   # always -- see module docstring
    }
    facts.update(probe_accelerator(run, host, repo))
    if host is not None:   # nothing to resolve for the local self-entry -- see resolve_ssh_config
        facts.update(resolve_ssh_config(host))
    env = env_hint(run, host)
    if env:
        facts["env"] = env
    if repo is None:
        facts["note"] = ("repo not found at " + ", ".join([repo_hint] if repo_hint else []) +
                         (", " if repo_hint else "") + ", ".join(DEFAULT_REPO_CANDIDATES) +
                         " -- set `repo` by hand")
    return facts


# ---------------------------------------------------------------------------
# TOML rendering (hand-built, not a round-trip of an existing file -- this always writes a
# FRESH inventory; re-running --probe-hosts overwrites it, same as scaffold_config.py's own
# "measured, annotated, and yours to correct" convention: review the diff, don't hand-edit
# and then re-probe over your own edits)
# ---------------------------------------------------------------------------
def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_toml(hosts: "dict[str, dict]") -> str:
    lines = [
        "# Generated by fleet_inventory.py --probe-hosts. Reviewable, not gospel -- see",
        "# docs/fleet-deployment-proposal.md \u00a72 for what each field means and what this",
        "# tool deliberately does NOT infer (train, max_render_s). Re-running --probe-hosts",
        "# OVERWRITES this file; move hand edits elsewhere first if you need to keep them.",
        "",
    ]
    for name in sorted(hosts):
        h = hosts[name]
        # Quoted key segment: an unquoted TOML bare key may only contain [A-Za-z0-9_-], and a
        # literal `.` in particular would otherwise parse as a NESTED table (hosts.mac-1.tailnet
        # -> hosts.'mac-1'.'tailnet', two levels), silently splitting one host into two keys.
        lines.append(f"[hosts.{_toml_str(name)}]")
        lines.append(f"address      = {_toml_str(h['address'])}")
        if h.get("repo"):
            lines.append(f"repo         = {_toml_str(h['repo'])}")
        else:
            lines.append('# repo not found -- set by hand, e.g.: repo = "~/work/parametric-nam"'
                         ' (see note below)')
        lines.append(f"cores        = {h['cores']}            # physical, not logical/SMT")
        backends = ", ".join(_toml_str(b) for b in h.get("backends", []))
        lines.append(f"backends     = [{backends}]")
        # accel is `None` when the probe COULDN'T determine it (no repo/venv, ssh failure, ...);
        # the string "none" is a real, determined answer (a genuine CPU-only host) -- these are
        # not the same thing and must not render the same way.
        accel = h.get("accelerator")
        if accel is not None:
            lines.append(f"accelerator  = {_toml_str(accel)}")
        else:
            lines.append("# accelerator not determined -- see note below")
        if h.get("gpus") is not None:
            lines.append(f"gpus         = {h['gpus']}")
        if h.get("vram_gb") is not None:
            lines.append(f"vram_gb      = {h['vram_gb']}")
        lines.append(f"train        = false           # review: set true to allow training jobs here")
        if h.get("env"):
            kv = ", ".join(f"{k} = {_toml_str(v)}" for k, v in h["env"].items())
            lines.append(f"env          = {{ {kv} }}")
        if h.get("user"):
            lines.append(f"user         = {_toml_str(h['user'])}   # from this machine's own ssh -G")
        if h.get("identity_file"):
            lines.append(f"identity_file = {_toml_str(h['identity_file'])}")
        if h.get("note"):
            lines.append(f"# {h['note']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# loading (for future consumers -- distribute_pull.py --inventory, gate_config.py fleet mode,
# neither of which exist yet; this is the format they will read, per the roadmap)
# ---------------------------------------------------------------------------
def default_inventory_path() -> Path:
    """Mirrors ~/.cache/parametric-nam/findpeak's own convention: YOUR inventory is per-machine
    state, not project content."""
    return Path.home() / ".config" / "parametric-nam" / "fleet.toml"


def load_inventory(path: "Path | None" = None) -> "dict[str, dict]":
    """{name: facts}. Empty dict (not an error) if the file doesn't exist -- callers decide
    whether that's fatal; this loader only reports what's there."""
    import tomllib
    p = Path(path) if path else default_inventory_path()
    if not p.is_file():
        return {}
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    return raw.get("hosts", {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-hosts", action="store_true", required=True,
                    help="the only mode today -- probe hosts and write the inventory")
    ap.add_argument("--worker", action="append", default=[], metavar="HOST[:REPO]",
                    help="repeatable. HOST is an ssh-reachable name (mesh DNS or ~/.ssh/config "
                        "alias). REPO, if given, is tried before the default candidate paths.")
    ap.add_argument("--self", metavar="NAME", default=None,
                    help="include the local machine under this name (default: its hostname). "
                        "See --no-self.")
    ap.add_argument("--no-self", action="store_true", help="do not probe the local machine")
    ap.add_argument("--inventory", type=Path, default=None,
                    help=f"where to write the TOML (default {default_inventory_path()})")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    targets: "list[tuple[str | None, str, str | None]]" = []   # (ssh_host, name, repo_hint)
    if not args.no_self:
        targets.append((None, args.self or socket.gethostname().split(".")[0], None))
    for spec in args.worker:
        host, _, repo = spec.partition(":")
        targets.append((host, host, repo or None))
    if not targets:
        print("nothing to probe: pass --worker, or drop --no-self", file=sys.stderr)
        return 2

    hosts = {}
    failed = []
    for ssh_host, name, repo_hint in targets:
        print(f"probing {name} ({ssh_host or 'local'})...", file=sys.stderr)
        try:
            hosts[name] = probe_host(default_ssh, ssh_host, address=(ssh_host or name),
                                     repo_hint=repo_hint)
        except Unreachable as e:
            print(f"  UNREACHABLE: {e}", file=sys.stderr)
            failed.append(name)

    out = args.inventory or default_inventory_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(render_toml(hosts))
    os.replace(tmp, out)
    print(f"\nwrote {len(hosts)} host(s) to {out}" + (f" ({len(failed)} unreachable, skipped: "
         f"{', '.join(failed)})" if failed else ""))
    return 1 if failed and not hosts else 0


if __name__ == "__main__":
    sys.exit(main())
