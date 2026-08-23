"""Properties tools/grid_adequacy.py's pure grid-density math and config-splice helper must
have. This tool exists because training-based grid comparisons were confounded by optimizer
noise (see module docstring); its whole value is that `esr`/`cell_error`/`suggest_axis` are
correct WITHOUT training anything. A fake render() stands in for a real circuit so the
cell-error/suggest/write-back logic is tested independently of ngspice/livespice.

See tools/grid_adequacy.py.
"""
import numpy as np
import pytest

from tools.grid_adequacy import cell_error, esr, measure_grid, suggest_axis, write_knobs


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
