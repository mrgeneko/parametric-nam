"""Properties tools/grid_adequacy.py's pure grid-density math and config-splice helper must
have. This tool exists because training-based grid comparisons were confounded by optimizer
noise (see module docstring); its whole value is that `esr`/`cell_error`/`suggest_axis` are
correct WITHOUT training anything. A fake render() stands in for a real circuit so the
cell-error/suggest/write-back logic is tested independently of ngspice/livespice.

See tools/grid_adequacy.py.
"""
import numpy as np
import pytest
import soundfile as sf

from tools.grid_adequacy import Renderer, cell_error, esr, measure_grid, suggest_axis, write_knobs


class FakeRender:
    """render(params) -> one constant-filled array per probe window, value = fn(params). A
    constant array makes cell_error's pooled-window ESR an exact, hand-checkable closed form
    (see each test), instead of depending on real circuit output."""

    def __init__(self, fn, n_windows=1, win_len=5000, lead_n=0, fail=frozenset()):
        self.fn = fn
        self.n_windows = n_windows
        self.win_len = win_len
        self.lead_n = lead_n
        self.fail = fail

    def __call__(self, params):
        key = tuple(sorted(params.items()))
        if key in self.fail:
            return [None] * self.n_windows
        val = self.fn(params)
        vals = val if isinstance(val, (list, tuple)) else [val] * self.n_windows
        return [np.full(self.win_len, v, dtype=np.float64) for v in vals]

    def print_fail_summary(self):
        pass


class TestEsr:
    def test_identical_arrays_zero_error(self):
        a = np.array([1.0, 2.0, 3.0])
        assert esr(a, a) == pytest.approx(0.0)

    def test_truncates_to_shorter_array(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0, 999.0])
        assert esr(a, b) == pytest.approx(0.0)

    def test_zero_reference_gives_nan_not_a_crash(self):
        assert np.isnan(esr(np.array([1.0]), np.array([0.0])))


class TestCellError:
    def test_zero_for_a_perfectly_linear_response(self):
        """The whole point of the tool: a circuit whose response is linear in the knob is
        exactly recoverable by interpolation, so the grid imposes zero error regardless of
        how coarse it is."""
        render = FakeRender(lambda p: p["Gain"])
        e = cell_error(render, {"Gain": [0.0, 1.0]}, "Gain", 0.0, 1.0, {})
        assert e == pytest.approx(0.0, abs=1e-9)

    def test_matches_the_closed_form_for_a_nonlinear_response(self):
        lo, hi = 0.0, 1.0
        render = FakeRender(lambda p: p["Gain"] ** 2)
        e = cell_error(render, {"Gain": [lo, hi]}, "Gain", lo, hi, {})
        mid = 0.5 * (lo + hi)
        interp = 0.5 * (lo ** 2 + hi ** 2)
        expected = ((interp - mid ** 2) / mid ** 2) ** 2
        assert e == pytest.approx(expected, rel=1e-6)

    def test_pools_error_across_multiple_windows_not_just_the_first(self):
        # Window 0 always interpolates exactly (linear); window 1 never does (quadratic). If
        # cell_error only looked at one window's error it would report zero.
        render = FakeRender(lambda p: [p["Gain"], p["Gain"] ** 2], n_windows=2)
        e = cell_error(render, {"Gain": [0.0, 1.0]}, "Gain", 0.0, 1.0, {})
        assert e > 0.0  # the quadratic window's error must show up, not be washed out

    def test_all_windows_failing_gives_nan(self):
        render = FakeRender(lambda p: p["Gain"], fail={(("Gain", 0.0),), (("Gain", 0.5),), (("Gain", 1.0),)})
        e = cell_error(render, {"Gain": [0.0, 1.0]}, "Gain", 0.0, 1.0, {})
        assert np.isnan(e)


class TestMeasureGridFailureGuard:
    def test_every_probe_failing_raises_instead_of_printing_a_blank_table(self):
        """A config typo (bad knob name, wrong circuit) fails every render identically -- this
        must surface as one clear error, not a per-cell table full of unexplained '?'."""
        render = FakeRender(lambda p: 0.0, fail={(("Gain", 0.0),), (("Gain", 0.5),), (("Gain", 1.0),)})
        with pytest.raises(RuntimeError, match="EVERY probe render failed"):
            measure_grid(render, {"Gain": [0.0, 0.5, 1.0]}, target=0.03, workers=2)

    def test_successful_probes_return_a_worst_by_axis_table(self):
        render = FakeRender(lambda p: p["Gain"])
        worst_by_axis, n_coarse, n_over = measure_grid(render, {"Gain": [0.0, 0.5, 1.0]},
                                                         target=0.03, workers=2)
        assert set(worst_by_axis["Gain"]) == {(0.0, 0.5), (0.5, 1.0)}
        assert n_coarse == 0  # perfectly linear -> every cell interpolates exactly


class TestSuggestAxis:
    def test_bisects_a_cell_over_target(self):
        out = suggest_axis([0.0, 1.0], worst={(0.0, 1.0): 0.5}, target=0.1)
        assert out == [0.0, 0.5, 1.0]

    def test_leaves_an_ok_cell_untouched(self):
        out = suggest_axis([0.0, 1.0, 2.0], worst={(0.0, 1.0): 0.05, (1.0, 2.0): 0.05}, target=0.1)
        assert out == [0.0, 1.0, 2.0]

    def test_drops_a_point_between_two_deeply_oversampled_cells(self):
        values = [0.0, 1.0, 2.0, 3.0]
        worst = {(0.0, 1.0): 0.01, (1.0, 2.0): 0.01, (2.0, 3.0): 0.05}
        out = suggest_axis(values, worst, target=1.0)
        assert 1.0 not in out
        assert out == [0.0, 2.0, 3.0]

    def test_missing_cell_data_is_left_untouched_not_treated_as_coarse_or_oversampled(self):
        values = [0.0, 1.0, 2.0]
        out = suggest_axis(values, worst={}, target=0.1)
        assert out == values

    def test_endpoints_are_always_preserved(self):
        values = [0.0, 1.0, 2.0, 3.0]
        worst = {(0.0, 1.0): 0.5, (1.0, 2.0): 0.5, (2.0, 3.0): 0.5}
        out = suggest_axis(values, worst, target=0.1)
        assert out[0] == 0.0
        assert out[-1] == 3.0


class TestWriteKnobs:
    TEMPLATE = (
        'schx = "device.schx"\n'
        "\n"
        "[knobs]\n"
        "Gain = [0.0, 0.5, 1.0]\n"
        "Tone = [0.0, 1.0]\n"
        "# Volume is a passive output divider, not swept.\n"
        "\n"
        "[fixed]\n"
        "Volume = 1.0\n"
    )

    def test_replaces_only_the_value_lines(self, tmp_path):
        cfg = tmp_path / "device.toml"
        cfg.write_text(self.TEMPLATE)
        write_knobs(cfg, {"Gain": [0.1, 0.9], "Tone": [0.2, 0.5, 0.8]})
        out = cfg.read_text()
        assert "Gain = [0.1, 0.9]" in out
        assert "Tone = [0.2, 0.5, 0.8]" in out
        assert "0.0, 0.5, 1.0" not in out

    def test_preserves_the_trailing_comment_paragraph_and_later_sections(self, tmp_path):
        cfg = tmp_path / "device.toml"
        cfg.write_text(self.TEMPLATE)
        write_knobs(cfg, {"Gain": [0.1, 0.9], "Tone": [0.2, 0.8]})
        out = cfg.read_text()
        assert "# Volume is a passive output divider, not swept." in out
        assert "[fixed]" in out
        assert "Volume = 1.0" in out

    def test_preserves_content_before_the_knobs_table(self, tmp_path):
        cfg = tmp_path / "device.toml"
        cfg.write_text(self.TEMPLATE)
        write_knobs(cfg, {"Gain": [0.1, 0.9], "Tone": [0.2, 0.8]})
        assert cfg.read_text().startswith('schx = "device.schx"\n')

    def test_missing_knobs_table_exits(self, tmp_path):
        cfg = tmp_path / "device.toml"
        cfg.write_text('schx = "device.schx"\n[fixed]\nVolume = 1.0\n')
        with pytest.raises(SystemExit):
            write_knobs(cfg, {"Gain": [0.1, 0.9]})


class TestRendererNgspiceDeck:
    """The Renderer's hand-deck twin of its schx-translated "ngspice" mode -- same stratified-
    window/lead-in machinery, but rendered through render_backends.NgspiceBackend against a
    gen_*_ngspice.py module directly, since there's no .schx to netlist-dump for a device whose
    clipping needs a real component (a MOSFET, a real BJT) .schx has no model for."""

    def _write_pedal_module(self, tmp_path, name="gen_fake_ngspice", knob_names=("Gain",)):
        pedal_dir = tmp_path / "pedals"
        pedal_dir.mkdir(exist_ok=True)
        (pedal_dir / f"{name}.py").write_text(
            f"KNOB_NAMES = {list(knob_names)!r}\ndef build_deck(**kw): return ''\n")
        return pedal_dir

    def _write_input(self, tmp_path, sr=1000, dur_s=20):
        inp = tmp_path / "input.wav"
        sf.write(str(inp), np.zeros(sr * dur_s, dtype=np.float32), sr)
        return inp

    class FakeNgspiceBackend:
        """render_many returns a constant array (one per window) equal to the Gain knob's
        value -- lets a test assert on the exact params a Renderer call forwards."""

        def __init__(self, build_deck, probe_node="OUT"):
            self.probe_node = probe_node

        def render_many(self, jobs, handle, scratch):
            tag = jobs[0]["tag"]
            val = jobs[0]["params"].get("Gain", 0.0)
            _sr, t_, _input_src = handle
            return {tag: np.full(len(t_), val, dtype=np.float64)}

    def test_uses_two_windows_and_the_ngspice_backends_output(self, tmp_path, monkeypatch):
        pedal_dir = self._write_pedal_module(tmp_path)
        inp = self._write_input(tmp_path)
        monkeypatch.setattr("tools.grid_adequacy.NgspiceBackend", self.FakeNgspiceBackend)
        td = tmp_path / "scratch"; td.mkdir()

        r = Renderer(schx=None, inp=inp, oversample=2, iterations=256, fixed="", td=td,
                    probe_s=8.0, backend="ngspice-deck", pedal_dir=str(pedal_dir),
                    module="gen_fake_ngspice", probe_node="OUT")
        out = r({"Gain": 0.7})
        assert len(out) == 2  # n_windows bumped to 2, same segfault-avoidance as "ngspice"
        assert all(np.allclose(w, 0.7) for w in out)

    def test_fixed_params_are_merged_into_the_rendered_knobs(self, tmp_path, monkeypatch):
        pedal_dir = self._write_pedal_module(tmp_path, knob_names=("Gain", "Tone"))
        inp = self._write_input(tmp_path)
        seen = []

        class RecordingBackend(self.FakeNgspiceBackend):
            def render_many(self, jobs, handle, scratch):
                seen.append(dict(jobs[0]["params"]))
                return super().render_many(jobs, handle, scratch)

        monkeypatch.setattr("tools.grid_adequacy.NgspiceBackend", RecordingBackend)
        td = tmp_path / "scratch"; td.mkdir()
        r = Renderer(schx=None, inp=inp, oversample=2, iterations=256, fixed="Tone=0.3",
                    td=td, probe_s=8.0, backend="ngspice-deck", pedal_dir=str(pedal_dir),
                    module="gen_fake_ngspice", probe_node="OUT")
        r({"Gain": 0.5})
        assert all(p == {"Tone": 0.3, "Gain": 0.5} for p in seen)

    def test_non_convergence_is_recorded_as_none_not_a_crash(self, tmp_path, monkeypatch):
        pedal_dir = self._write_pedal_module(tmp_path)
        inp = self._write_input(tmp_path)

        class FailingBackend:
            def __init__(self, build_deck, probe_node="OUT"):
                pass

            def render_many(self, jobs, handle, scratch):
                return {jobs[0]["tag"]: None}

        monkeypatch.setattr("tools.grid_adequacy.NgspiceBackend", FailingBackend)
        td = tmp_path / "scratch"; td.mkdir()
        r = Renderer(schx=None, inp=inp, oversample=2, iterations=256, fixed="", td=td,
                    probe_s=8.0, backend="ngspice-deck", pedal_dir=str(pedal_dir),
                    module="gen_fake_ngspice", probe_node="OUT")
        out = r({"Gain": 0.5})
        assert out == [None, None]

    def test_results_are_cached_by_params(self, tmp_path, monkeypatch):
        pedal_dir = self._write_pedal_module(tmp_path)
        inp = self._write_input(tmp_path)
        calls = []

        class CountingBackend(self.FakeNgspiceBackend):
            def render_many(self, jobs, handle, scratch):
                calls.append(1)
                return super().render_many(jobs, handle, scratch)

        monkeypatch.setattr("tools.grid_adequacy.NgspiceBackend", CountingBackend)
        td = tmp_path / "scratch"; td.mkdir()
        r = Renderer(schx=None, inp=inp, oversample=2, iterations=256, fixed="", td=td,
                    probe_s=8.0, backend="ngspice-deck", pedal_dir=str(pedal_dir),
                    module="gen_fake_ngspice", probe_node="OUT")
        r({"Gain": 0.5})
        n_first = len(calls)
        r({"Gain": 0.5})
        assert len(calls) == n_first, "identical params must be a cache hit, not a re-render"

    def test_lead_silence_s_override_reaches_write_probe_clip(self, tmp_path, monkeypatch):
        """The exact regression this override exists for: a circuit whose real settling time
        exceeds write_probe_clip's 1.0s default (the Fulltone OCD's ~5s C10/RVOL2 network) needs
        this to actually reach the probe clips grid_adequacy builds -- not just be accepted and
        silently ignored."""
        pedal_dir = self._write_pedal_module(tmp_path)
        inp = self._write_input(tmp_path)
        seen_lead_s = []

        import tools.grid_adequacy as ga
        real_write_probe_clip = ga.write_probe_clip

        def spy(sig, sr, path, lead_s=None):
            seen_lead_s.append(lead_s)
            return real_write_probe_clip(sig, sr, path, lead_s=lead_s)

        monkeypatch.setattr("tools.grid_adequacy.write_probe_clip", spy)
        monkeypatch.setattr("tools.grid_adequacy.NgspiceBackend", self.FakeNgspiceBackend)
        td = tmp_path / "scratch"; td.mkdir()
        Renderer(schx=None, inp=inp, oversample=2, iterations=256, fixed="", td=td,
                probe_s=8.0, backend="ngspice-deck", pedal_dir=str(pedal_dir),
                module="gen_fake_ngspice", probe_node="OUT", lead_silence_s=5.0)
        assert seen_lead_s and all(v == 5.0 for v in seen_lead_s)
