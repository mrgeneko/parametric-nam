"""Properties measure_ngspice_timestep.py must have. Built directly from a real
production incident: NgspiceBackend's hardcoded maxstep=3e-6 default turned out to be genuinely
too coarse for the MOSFET-clipping pedal's real 2N7000 MOSFET clipping (re-rendering the identical knob
setting at 3e-6/1e-6/3e-7 gave RMS 1.210/1.614/1.593 -- a real, unconverged bias, not noise),
which is exactly what made grid_adequacy.py --apply explode instead of converge.

A second, subtler bug surfaced immediately after: this tool originally rendered through
render_backends.NgspiceBackend, whose render_many always uses render_grid's default
(maxstep, maxstep/3, maxstep/10) escalation ladder -- so a job that failed to converge at the
"requested" maxstep was silently retried at a DIFFERENT, finer one, meaning two jobs nominally
at the same maxstep could have actually been rendered at different timesteps. That contaminated
the very comparison this tool exists to make (confirmed: a real run's numbers were non-monotonic
and enormous). Fixed by calling render_grid() directly with rungs=(maxstep,) -- a genuine
single-shot attempt, no escalation -- which is exactly what these tests pin down: the mocked
render_grid records which maxstep it was actually called with, so a regression back to going
through NgspiceBackend's escalating path would show up as a wrong recorded maxstep.

See measure_ngspice_timestep.py.
"""
import numpy as np
import pytest
import soundfile as sf
from scipy.io import wavfile

from measure_ngspice_timestep import load_config, load_device, measure


def write_pedal_module(tmp_path, name="gen_fake_ngspice", knob_names=("Gain",)):
    pedal_dir = tmp_path / "pedals"
    pedal_dir.mkdir(exist_ok=True)
    (pedal_dir / f"{name}.py").write_text(
        f"KNOB_NAMES = {list(knob_names)!r}\ndef build_deck(**kw): return ''\n")
    return pedal_dir


def write_config(tmp_path, pedal_dir, module="gen_fake_ngspice", knobs=("Gain",),
                 fixed=None, backend="ngspice-deck"):
    cfg = tmp_path / "config.toml"
    knob_lines = "\n".join(f'{k} = [0.0, 1.0]' for k in knobs)
    fixed_block = ""
    if fixed:
        fixed_lines = "\n".join(f"{k} = {v}" for k, v in fixed.items())
        fixed_block = f"\n[fixed]\n{fixed_lines}\n"
    cfg.write_text(
        f'backend    = "{backend}"\n'
        f'pedal-dir  = "{pedal_dir}"\n'
        f'module     = "{module}"\n'
        f'probe-node = "OUT"\n\n'
        f"[knobs]\n{knob_lines}\n"
        f"{fixed_block}"
    )
    return cfg


class TestLoadDevice:
    def test_reads_name_knobs_fixed_and_build_deck(self, tmp_path):
        pedal_dir = write_pedal_module(tmp_path, knob_names=("Gain", "Tone"))
        cfg = write_config(tmp_path, pedal_dir, knobs=("Gain", "Tone"), fixed={"Volume": 1.0})
        name, build_deck, probe_node, knobs, fixed = load_device(cfg)
        assert name == tmp_path.name
        assert sorted(knobs) == ["Gain", "Tone"]
        assert fixed == {"Volume": 1.0}
        assert probe_node == "OUT"
        assert callable(build_deck)

    def test_rejects_a_non_ngspice_deck_config(self, tmp_path):
        pedal_dir = write_pedal_module(tmp_path)
        cfg = write_config(tmp_path, pedal_dir, backend="livespice")
        with pytest.raises(ValueError, match="ngspice-deck"):
            load_device(cfg)

    def test_rejects_a_config_with_no_knobs(self, tmp_path):
        pedal_dir = write_pedal_module(tmp_path)
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f'backend    = "ngspice-deck"\n'
            f'pedal-dir  = "{pedal_dir}"\n'
            f'module     = "gen_fake_ngspice"\n'
        )
        with pytest.raises(ValueError, match="knobs"):
            load_device(cfg)


def make_fake_render_grid(value_fn, calls=None):
    """A stand-in for ngspice_spicelib.render_grid: writes an int16 wav whose value encodes
    value_fn(knobs, maxstep), and returns {outfile: peak} -- matching the real function's
    contract closely enough for measure()'s own un-normalization to round-trip it. Records
    every (rungs, maxstep) it was called with so a test can assert single-shot (rungs=(ms,)),
    not escalating, behavior."""

    def fake_render_grid(build_deck, jobs, probe_node, sr, t, input_src, tmp,
                         maxstep=3e-6, parallel_sims=8, rungs=None):
        if calls is not None:
            calls.append({"maxstep": maxstep, "rungs": rungs})
        peaks = {}
        for knobs, outfile in jobs:
            val = value_fn(knobs, maxstep)
            if val is None:
                peaks[outfile] = None
                continue
            pk = abs(val) + 1e-9
            y = np.full(len(t), val, dtype=np.float64)
            int16 = np.clip(y / pk * 0.9 * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(outfile, sr, int16)
            peaks[outfile] = pk
        return peaks

    return fake_render_grid


def make_clips(tmp_path, sr=1000, dur_s=20, n=2):
    clips = []
    for i in range(n):
        p = tmp_path / f"clip{i}.wav"
        sf.write(str(p), np.zeros(sr * dur_s, dtype=np.float32), sr)
        clips.append(p)
    return clips


class TestMeasure:
    def test_finer_maxstep_closer_to_reference_gives_lower_esr(self, tmp_path, monkeypatch):
        # Simulate a real bias that shrinks toward a "true" value (1.0) as maxstep shrinks:
        # 3e-6 -> 1.5 (biased), 1e-6 -> 1.05 (closer), ref (tiny) -> 1.0 (truth).
        bias = {3e-6: 1.5, 1e-6: 1.05, 1e-8: 1.0, 5e-9: 1.0}
        monkeypatch.setattr("measure_ngspice_timestep.render_grid",
                            make_fake_render_grid(lambda knobs, ms: bias[ms]))
        clips = make_clips(tmp_path)
        res = measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                      clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                      candidates=(3e-6, 1e-6), parallel_sims=2, td=tmp_path)
        e_coarse, _ = res[3e-6]
        e_fine, _ = res[1e-6]
        assert e_coarse > e_fine > 0
        # closed form for a constant-array ESR against a constant reference (int16 round-trip
        # introduces a small quantization error, hence rel tolerance not an exact match)
        assert e_coarse == pytest.approx(((1.5 - 1.0) / 1.0) ** 2, rel=1e-3)
        assert e_fine == pytest.approx(((1.05 - 1.0) / 1.0) ** 2, rel=1e-3)

    def test_zero_error_when_every_maxstep_agrees(self, tmp_path, monkeypatch):
        monkeypatch.setattr("measure_ngspice_timestep.render_grid",
                            make_fake_render_grid(lambda knobs, ms: 1.0))
        clips = make_clips(tmp_path)
        res = measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                      clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                      candidates=(3e-6,), parallel_sims=2, td=tmp_path)
        e, _ = res[3e-6]
        assert e == pytest.approx(0.0, abs=1e-6)

    def test_fixed_params_are_forwarded_to_every_render(self, tmp_path, monkeypatch):
        seen = []

        def value_fn(knobs, ms):
            seen.append(dict(knobs))
            return 1.0

        monkeypatch.setattr("measure_ngspice_timestep.render_grid",
                            make_fake_render_grid(value_fn))
        clips = make_clips(tmp_path)
        measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"],
               fixed={"Volume": 1.0}, clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
               candidates=(3e-6,), parallel_sims=2, td=tmp_path)
        assert seen and all(p.get("Volume") == 1.0 for p in seen)

    def test_all_renders_failing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("measure_ngspice_timestep.render_grid",
                            make_fake_render_grid(lambda knobs, ms: None))
        clips = make_clips(tmp_path)
        with pytest.raises(RuntimeError, match="EVERY render failed"):
            measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                   clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                   candidates=(3e-6,), parallel_sims=2, td=tmp_path)

    def test_reports_a_ref_error_against_half_the_reference_maxstep(self, tmp_path, monkeypatch):
        monkeypatch.setattr("measure_ngspice_timestep.render_grid",
                            make_fake_render_grid(lambda knobs, ms: 1.0))
        clips = make_clips(tmp_path)
        res = measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                      clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                      candidates=(3e-6,), parallel_sims=2, td=tmp_path)
        assert "ref_error" in res
        assert res["ref_error"] == pytest.approx(0.0, abs=1e-6)  # fake agrees everywhere

    def test_every_render_is_single_shot_never_escalating(self, tmp_path, monkeypatch):
        """The exact regression this tool's second bug was: going through NgspiceBackend meant
        a nominal maxstep could silently resolve to a finer one via render_grid's own
        (maxstep, maxstep/3, maxstep/10) escalation. render_grid must always be called with
        rungs=(maxstep,) -- a single, fixed attempt -- for the comparison to mean anything."""
        calls = []
        monkeypatch.setattr("measure_ngspice_timestep.render_grid",
                            make_fake_render_grid(lambda knobs, ms: 1.0, calls=calls))
        clips = make_clips(tmp_path)
        measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
               clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
               candidates=(3e-6, 1e-6), parallel_sims=2, td=tmp_path)
        assert calls
        assert all(c["rungs"] == (c["maxstep"],) for c in calls)

    def test_reference_recheck_only_renders_the_worst_setting(self, tmp_path, monkeypatch):
        """The exact regression this tool's first bug was: the reference-convergence recheck
        re-rendered EVERY knob setting (7x the necessary cost at the most expensive, finest
        timestep) instead of just the single worst-setting corner it needs."""
        job_counts = []

        def fake_render_grid(build_deck, jobs, probe_node, sr, t, input_src, tmp,
                             maxstep=3e-6, parallel_sims=8, rungs=None):
            job_counts.append((maxstep, len(jobs)))
            peaks = {}
            for knobs, outfile in jobs:
                pk = 1.0
                wavfile.write(outfile, sr, np.full(len(t), 1000, dtype=np.int16))  # non-zero
                peaks[outfile] = pk
            return peaks

        monkeypatch.setattr("measure_ngspice_timestep.render_grid", fake_render_grid)
        clips = make_clips(tmp_path)
        ref_maxstep = 1e-8
        measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain", "Tone"], fixed={},
               clips=clips, lead_n=0, sr=1000, ref_maxstep=ref_maxstep,
               candidates=(3e-6,), parallel_sims=2, td=tmp_path)
        recheck_calls = [n for ms, n in job_counts if ms == pytest.approx(ref_maxstep / 2)]
        assert recheck_calls, "expected a batch at ref_maxstep/2"
        assert all(n == 1 for n in recheck_calls), \
            f"reference recheck must render exactly 1 setting per window, got {recheck_calls}"
