"""Properties tools/measure_ngspice_timestep.py must have. Built directly from a real
production incident: NgspiceBackend's hardcoded maxstep=3e-6 default turned out to be genuinely
too coarse for the Fulltone OCD's real 2N7000 MOSFET clipping (re-rendering the identical knob
setting at 3e-6/1e-6/3e-7 gave RMS 1.210/1.614/1.593 -- a real, unconverged bias, not noise),
which is exactly what made tools/grid_adequacy.py --apply explode instead of converge. These
tests exercise measure()'s ESR-against-a-finer-reference math and load_device()'s config
validation against a FakeNgspiceBackend, independent of any real ngspice render.

See tools/measure_ngspice_timestep.py.
"""
import numpy as np
import pytest
import soundfile as sf

from tools.measure_ngspice_timestep import load_config, load_device, measure


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


class FakeNgspiceBackend:
    """render_many returns a constant array per job (one value per maxstep, from `value_fn`) --
    makes measure()'s pooled ESR an exact, hand-checkable closed form."""

    def __init__(self, build_deck, probe_node="OUT", parallel_sims=8, maxstep=3e-6):
        self.maxstep = maxstep

    value_fn = staticmethod(lambda params, maxstep: 1.0)

    def render_many(self, jobs, handle, scratch):
        _sr, t_, _input_src = handle
        out = {}
        for job in jobs:
            val = self.value_fn(job["params"], self.maxstep)
            out[job["tag"]] = np.full(len(t_), val, dtype=np.float64)
        return out


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

        class Backend(FakeNgspiceBackend):
            value_fn = staticmethod(lambda params, maxstep: bias[maxstep])

        monkeypatch.setattr("tools.measure_ngspice_timestep.NgspiceBackend", Backend)
        clips = make_clips(tmp_path)
        res = measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                      clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                      candidates=(3e-6, 1e-6), parallel_sims=2, td=tmp_path)
        e_coarse, _ = res[3e-6]
        e_fine, _ = res[1e-6]
        assert e_coarse > e_fine > 0
        # exact closed form for a constant-array ESR against a constant reference
        assert e_coarse == pytest.approx(((1.5 - 1.0) / 1.0) ** 2, rel=1e-6)
        assert e_fine == pytest.approx(((1.05 - 1.0) / 1.0) ** 2, rel=1e-6)

    def test_zero_error_when_every_maxstep_agrees(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.measure_ngspice_timestep.NgspiceBackend", FakeNgspiceBackend)
        clips = make_clips(tmp_path)
        res = measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                      clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                      candidates=(3e-6,), parallel_sims=2, td=tmp_path)
        e, _ = res[3e-6]
        assert e == pytest.approx(0.0, abs=1e-9)

    def test_fixed_params_are_forwarded_to_every_render(self, tmp_path, monkeypatch):
        seen = []

        class RecordingBackend(FakeNgspiceBackend):
            def render_many(self, jobs, handle, scratch):
                seen.extend(dict(j["params"]) for j in jobs)
                return super().render_many(jobs, handle, scratch)

        monkeypatch.setattr("tools.measure_ngspice_timestep.NgspiceBackend", RecordingBackend)
        clips = make_clips(tmp_path)
        measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"],
               fixed={"Volume": 1.0}, clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
               candidates=(3e-6,), parallel_sims=2, td=tmp_path)
        assert seen and all(p.get("Volume") == 1.0 for p in seen)

    def test_all_renders_failing_raises(self, tmp_path, monkeypatch):
        class FailingBackend(FakeNgspiceBackend):
            def render_many(self, jobs, handle, scratch):
                return {j["tag"]: None for j in jobs}

        monkeypatch.setattr("tools.measure_ngspice_timestep.NgspiceBackend", FailingBackend)
        clips = make_clips(tmp_path)
        with pytest.raises(RuntimeError, match="EVERY render failed"):
            measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                   clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                   candidates=(3e-6,), parallel_sims=2, td=tmp_path)

    def test_reports_a_ref_error_against_half_the_reference_maxstep(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.measure_ngspice_timestep.NgspiceBackend", FakeNgspiceBackend)
        clips = make_clips(tmp_path)
        res = measure(build_deck=lambda **kw: None, probe_node="OUT", knobs=["Gain"], fixed={},
                      clips=clips, lead_n=0, sr=1000, ref_maxstep=1e-8,
                      candidates=(3e-6,), parallel_sims=2, td=tmp_path)
        assert "ref_error" in res
        assert res["ref_error"] == pytest.approx(0.0, abs=1e-9)  # FakeNgspiceBackend agrees everywhere
