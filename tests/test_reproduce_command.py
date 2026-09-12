"""reproduce.sh must actually reproduce the run.

Both defects below shipped in every published bundle and were invisible until someone tried
to run the script -- which is exactly when you most need it to work. Found by inspecting
Mesa Orange's published bundle, 2026-09-12.
"""
import sys, types, shlex
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_pipeline import reproduce_command


def _args(**kw):
    d = dict(dataset_dir="/tmp/ds", nam_output="/tmp/m.nam", checkpoint_dir="/tmp/ck",
             backend="livespice", schx="/tmp/a.schx", circuit=None,
             knobs="OR Gain,Or Master,Or Bass", oversample="8", trunc_target=1e-3,
             random=None, no_anchors=False, max_crest=50.0, values=None, ranges=[],
             bounds=[], gang=[], steps=[], fixed_params="Orange Presence=0.5",
             speaker=None, input="/tmp/in.wav", widths="5,9", mmap=True,
             repeats=1, epochs=0, restart_period=50, restart_mult=2, crop_len=48000,
             batch_size=64, lr=0.0003, target_steps=25000, config=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


class TestKnobsAreQuoted:
    """Knob names routinely contain spaces ("OR Gain"). Unquoted, bash splits one flag
    into several and the run cannot be reproduced at all."""

    def test_knobs_flag_survives_shell_splitting(self):
        cmd = reproduce_command(_args())
        line = next(l for l in cmd.splitlines() if "--knobs" in l)
        toks = shlex.split(line.rstrip("\\"))
        i = toks.index("--knobs")
        assert toks[i + 1] == "OR Gain,Or Master,Or Bass", \
            f"--knobs split into pieces: {toks[i:i+4]}"

    def test_whole_command_round_trips_through_shlex(self):
        """Any unquoted spaced value anywhere would show up here."""
        cmd = reproduce_command(_args(speaker="S 1", values="0.1,0.5"))
        toks = shlex.split(cmd.replace("\\\n", " "))
        for flag, want in (("--knobs", "OR Gain,Or Master,Or Bass"),
                           ("--speaker", "S 1"),
                           ("--fixed-params", "Orange Presence=0.5")):
            assert toks[toks.index(flag) + 1] == want, flag


class TestTheBudgetSurvives:
    """`--target-steps` had no branch here at all, and `--repeats` emitted args.repeats --
    the raw CLI default of 1. Mesa Orange's bundle records `--repeats 1` for a run that
    trained at 20; the config's own comment warns repeats=1 on a small grid gives 1-2
    gradient steps per epoch and a false plateau. The script ran, and reproduced nothing."""

    def test_target_steps_is_emitted(self):
        assert "--target-steps 25000" in reproduce_command(_args())

    def test_derived_repeats_wins_over_the_cli_default(self):
        cmd = reproduce_command(_args(repeats=1), repeats=20)
        assert "--repeats 20" in cmd
        assert "--repeats 1 " not in cmd

    def test_falls_back_to_args_when_no_derived_value(self):
        assert "--repeats 7" in reproduce_command(_args(repeats=7), repeats=None)


class TestConfigFormIsPreferred:
    """A config-driven run should POINT at the config copied beside the script rather than
    re-expand it -- re-expansion is where the budget got lost."""

    def test_config_form_is_short_and_self_locating(self):
        cmd = reproduce_command(_args(config=Path("/x/c.toml")), have_config=True)
        assert "--config" in cmd and "dirname" in cmd
        assert "--knobs" not in cmd and "--range" not in cmd

    def test_expanded_form_used_when_no_config_present(self):
        cmd = reproduce_command(_args(config=None), have_config=False)
        assert "--knobs" in cmd and "--config" not in cmd
