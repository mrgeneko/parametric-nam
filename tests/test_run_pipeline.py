"""Tests for run_pipeline.py's pure, already-extracted-for-testability logic: the
reproducible-path rewriter (portable()), the training-command flag-forwarding
(build_train_cmd()), the per-circuit TOML config loader (load_config()), and the
missing-permutations fail-fast gate (check_missing_permutations()).

None of these invoke gen_dataset_from_schx.py/param_train.py as subprocesses -- they test
the command/decision construction directly.

See run_pipeline.py.
"""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_pipeline as rp


# --------------------------------------------------------------------------- portable()

def test_portable_leaves_an_unrelated_path_unchanged():
    assert rp.portable("/some/other/place/file.wav") == "/some/other/place/file.wav"


def test_portable_rewrites_a_path_under_parametric_nam_itself():
    p = rp.HERE / "gen_dataset_from_schx.py"
    assert rp.portable(p) == "${PARAMETRIC_NAM:-$HOME/work/parametric-nam}/gen_dataset_from_schx.py"


def test_portable_rewrites_a_path_under_each_known_sibling_repo():
    cases = {
        rp.HERE.parent / "parametric-devices" / "pedals" / "X.schx":
            "${PARAMETRIC_DEVICES:-$HOME/work/parametric-devices}/pedals/X.schx",
        rp.HERE.parent / "sweep-files" / "sweep120s.wav":
            "${SWEEP_FILES:-$HOME/work/sweep-files}/sweep120s.wav",
        rp.HERE.parent / "livespice-cli" / "publish" / "livespice_cli":
            "${LIVESPICE_CLI_REPO:-$HOME/work/livespice-cli}/publish/livespice_cli",
        rp.HERE.parent / "hotspice" / "oracle":
            "${HOTSPICE:-$HOME/work/hotspice}/oracle",
    }
    for path, expected in cases.items():
        assert rp.portable(path) == expected


def test_portable_matches_a_sibling_repo_before_falling_back_to_home():
    # A sibling repo living under $HOME must match the MORE SPECIFIC sibling-repo entry
    # (its own ${VAR:-...} template), not the bare "$HOME/..." fallback string -- sibling
    # entries are checked first.
    p = rp.HERE.parent / "parametric-devices" / "devices.toml"
    result = rp.portable(p)
    assert result == "${PARAMETRIC_DEVICES:-$HOME/work/parametric-devices}/devices.toml"
    assert not result.startswith("$HOME/")


def test_portable_matches_exact_base_path_with_no_trailing_content():
    assert rp.portable(rp.HERE) == "${PARAMETRIC_NAM:-$HOME/work/parametric-nam}"


def test_portable_falls_back_to_home_for_an_unmatched_home_subpath():
    p = Path.home() / "Documents" / "notes.txt"
    assert rp.portable(p) == "$HOME/Documents/notes.txt"


def test_portable_accepts_a_path_object_or_a_string_identically():
    p = rp.HERE / "README.md"
    assert rp.portable(p) == rp.portable(str(p))


# --------------------------------------------------------------------------- build_train_cmd()

def _base_args(**overrides):
    defaults = dict(
        nam_output=Path("/tmp/out.param.nam"), checkpoint_dir=Path("/tmp/ckpt"),
        restart_period=2000, restart_mult=2, stale_cycles=3, batch_size=32, lr=1e-3,
        crop_len=48000, mrstft_weight=0.5, val_split=0.1, val_passes=1, device="cpu",
        seed=0, widths=None, mmap=True, resume=None, amp="fp16", init_from=None,
        param_sensitivity=False, knob_boost=None, per_tier_clip=False, clip_norm=1.0,
        spectral_norm=False, lora_rank=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_train_cmd_always_forwards_amp_even_when_off():
    # The 2026-07-31 incident this guards against: `if args.amp != "off"` used to silently
    # drop the flag when the caller explicitly asked for --amp off, since param_train.py's
    # own default is "fp16", not "off" -- so the omission was NOT equivalent to "off".
    cmd = rp.build_train_cmd(_base_args(amp="off"), Path("/tmp/ds"), epochs=100, repeats=1)
    assert "--amp" in cmd
    assert cmd[cmd.index("--amp") + 1] == "off"


def test_build_train_cmd_forwards_amp_for_every_value_not_just_off():
    for value in ("off", "fp16", "bf16"):
        cmd = rp.build_train_cmd(_base_args(amp=value), Path("/tmp/ds"), epochs=100, repeats=1)
        assert cmd[cmd.index("--amp") + 1] == value


def test_build_train_cmd_omits_optional_flags_when_falsy():
    cmd = rp.build_train_cmd(_base_args(), Path("/tmp/ds"), epochs=100, repeats=1)
    for flag in ("--widths", "--no-mmap", "--resume", "--init-from", "--param-sensitivity",
                 "--knob-boost", "--per-tier-clip", "--spectral-norm", "--lora-rank"):
        assert flag not in cmd


def test_build_train_cmd_includes_optional_flags_when_set():
    args = _base_args(widths="3,5,8", mmap=False, resume=Path("/tmp/ckpt/latest.pt"),
                       init_from=Path("/tmp/base.pt"), param_sensitivity=True,
                       knob_boost="drive=2.0", per_tier_clip=True, spectral_norm=True,
                       lora_rank=4)
    cmd = rp.build_train_cmd(args, Path("/tmp/ds"), epochs=100, repeats=1)
    assert cmd[cmd.index("--widths") + 1] == "3,5,8"
    assert "--no-mmap" in cmd
    assert cmd[cmd.index("--resume") + 1] == Path("/tmp/ckpt/latest.pt")
    assert cmd[cmd.index("--init-from") + 1] == Path("/tmp/base.pt")
    assert "--param-sensitivity" in cmd
    assert cmd[cmd.index("--knob-boost") + 1] == "drive=2.0"
    assert "--per-tier-clip" in cmd
    assert "--spectral-norm" in cmd
    assert cmd[cmd.index("--lora-rank") + 1] == "4"


def test_build_train_cmd_uses_the_passed_epochs_and_repeats_not_args_own():
    # epochs/repeats are DERIVED (from --target-steps) and passed in separately -- they must
    # win over anything of the same name on args, since args may not even have them for every
    # call site.
    cmd = rp.build_train_cmd(_base_args(), Path("/tmp/ds"), epochs=12345, repeats=7)
    assert cmd[cmd.index("--epochs") + 1] == 12345
    assert cmd[cmd.index("--repeats") + 1] == 7


def test_build_train_cmd_clip_norm_only_forwarded_when_non_default():
    cmd = rp.build_train_cmd(_base_args(clip_norm=1.0), Path("/tmp/ds"), epochs=1, repeats=1)
    assert "--clip-norm" not in cmd
    cmd = rp.build_train_cmd(_base_args(clip_norm=0.5), Path("/tmp/ds"), epochs=1, repeats=1)
    assert cmd[cmd.index("--clip-norm") + 1] == 0.5


# --------------------------------------------------------------------------- load_config()

def test_load_config_scalars_map_by_dest_hyphen_or_underscore(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('backend = "ngspice"\ncrop-len = 96000\nlr = 0.0005\n')
    out = rp.load_config(cfg)
    assert out["backend"] == "ngspice"
    assert out["crop_len"] == 96000
    assert out["lr"] == 0.0005


def test_load_config_widths_list_becomes_a_comma_joined_string(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("widths = [3, 4, 8]\n")
    assert rp.load_config(cfg)["widths"] == "3,4,8"


def test_load_config_path_dests_are_expanded_to_path_objects(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('dataset-dir = "~/work/tmp/ds"\nschx = "~/work/parametric-devices/x.schx"\n')
    out = rp.load_config(cfg)
    assert isinstance(out["dataset_dir"], Path)
    assert str(out["dataset_dir"]).startswith(str(Path.home()))
    assert isinstance(out["schx"], Path)


def test_load_config_knobs_table_expands_to_knobs_and_ranges(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        "[knobs]\n"
        "gain = [0.1, 0.5, 0.9]\n"
        "tone = [0.0, 1.0]\n"
    )
    out = rp.load_config(cfg)
    assert out["knobs"] == "gain,tone"
    assert "gain=0.1,0.5,0.9" in out["ranges"]
    assert "tone=0.0,1.0" in out["ranges"]


def test_load_config_fixed_table_expands_to_a_kv_string(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[fixed]\nRock = 0\nSag = 1\n')
    out = rp.load_config(cfg)
    assert out["fixed_params"] == "Rock=0,Sag=1"


def test_load_config_knob_boost_and_knob_kind_tables_expand(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        '[knob-boost]\ndrive = 2.0\n'
        '[knob-kind]\ntone = "hi"\n'
    )
    out = rp.load_config(cfg)
    assert out["knob_boost"] == "drive=2.0"
    assert out["knob_kind"] == "tone=hi"


def test_load_config_absent_tables_are_simply_not_present(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('backend = "livespice"\n')
    out = rp.load_config(cfg)
    for key in ("knobs", "ranges", "fixed_params", "defaults", "knob_boost", "knob_kind"):
        assert key not in out


# --------------------------------------------------------------------------- check_missing_permutations()

def _write_dataset(tmp_path, expected, rows):
    (tmp_path / "config.json").write_text(json.dumps({"permutation_count": expected}))
    fieldnames = sorted({k for r in rows for k in r})
    import csv
    with open(tmp_path / "params.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_check_missing_permutations_noop_when_nothing_generated_yet(tmp_path):
    fh = io.StringIO()
    rp.check_missing_permutations(tmp_path, fh, allow_missing=False)  # no config.json/params.csv
    assert fh.getvalue() == ""


def test_check_missing_permutations_noop_when_all_succeeded(tmp_path):
    _write_dataset(tmp_path, expected=2,
                    rows=[{"idx": 0, "ok": "1", "gain": 0.1},
                          {"idx": 1, "ok": "1", "gain": 0.5}])
    fh = io.StringIO()
    rp.check_missing_permutations(tmp_path, fh, allow_missing=False)
    assert fh.getvalue() == ""


def test_check_missing_permutations_exits_by_default_when_some_failed(tmp_path):
    _write_dataset(tmp_path, expected=2,
                    rows=[{"idx": 0, "ok": "1", "gain": 0.1},
                          {"idx": 1, "ok": "0", "gain": 0.5, "error": "diverged"}])
    fh = io.StringIO()
    with pytest.raises(SystemExit):
        rp.check_missing_permutations(tmp_path, fh, allow_missing=False)
    assert "1 of 2 permutations failed" in fh.getvalue()
    assert "diverged" in fh.getvalue()


def test_check_missing_permutations_warns_but_proceeds_with_allow_missing(tmp_path):
    _write_dataset(tmp_path, expected=2,
                    rows=[{"idx": 0, "ok": "1", "gain": 0.1},
                          {"idx": 1, "ok": "0", "gain": 0.5, "error": "diverged"}])
    fh = io.StringIO()
    rp.check_missing_permutations(tmp_path, fh, allow_missing=True)  # must not raise
    assert "WARNING" in fh.getvalue()


def test_check_missing_permutations_noop_when_permutation_count_is_absent(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({}))
    (tmp_path / "params.csv").write_text("idx,ok\n0,1\n")
    fh = io.StringIO()
    rp.check_missing_permutations(tmp_path, fh, allow_missing=False)
    assert fh.getvalue() == ""
