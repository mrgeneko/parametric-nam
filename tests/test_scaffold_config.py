"""Properties scaffold_config.py's placeholder-grid generation and TOML-splice helpers
must have. The grid math encodes real, empirically-found density needs (see README's
scaffold_config.py section): a gain/volume knob's audible character changes fastest near the
bottom of its range, and a second point just below max stabilizes that grid cell -- these
tests lock in the actual numbers, not just "some grid comes out."

See scaffold_config.py.
"""
import numpy as np
import pytest

from scaffold_config import (
    _format_knob_kind,
    _format_knobs,
    _grid_for_kind,
    _replace_line,
    _replace_table,
)


class TestToneKnobGrid:
    """hi/lo/mid: still evenly spaced (no observed need for endpoint density), but narrowed
    to [0.2, 0.8] -- the fully-CCW/CW extremes rarely hold a tone stack's interesting
    behavior."""

    @pytest.mark.parametrize("kind", ["hi", "lo", "mid"])
    def test_narrowed_range(self, kind):
        pts = _grid_for_kind(kind, 3)
        assert pts[0] == 0.2
        assert pts[-1] == 0.8

    @pytest.mark.parametrize("kind", ["hi", "lo", "mid"])
    def test_evenly_spaced(self, kind):
        pts = _grid_for_kind(kind, 5)
        gaps = np.diff(pts)
        assert np.allclose(gaps, gaps[0]), "tone knob grid must stay evenly spaced"

    def test_point_count_matches_grid_points(self):
        assert len(_grid_for_kind("hi", 4)) == 4

    def test_minimum_two_points(self):
        assert len(_grid_for_kind("hi", 1)) == 2, "n_points < 2 must still give a usable grid"


class TestGainVolumeKnobGrid:
    """drive/rms: NOT evenly spaced -- fixed density anchors at the bottom (where a gain
    knob's character changes fastest) and a stabilizing point just below the top, on top of
    a --grid-points evenly-spaced baseline across the full [0.1, 1.0] range."""

    @pytest.mark.parametrize("kind", ["drive", "rms"])
    def test_required_anchors_present(self, kind):
        pts = _grid_for_kind(kind, 3)
        for anchor in (0.1, 0.15, 0.25, 0.95, 1.0):
            assert anchor in pts, f"{kind} grid missing required anchor {anchor}: {pts}"

    @pytest.mark.parametrize("kind", ["drive", "rms"])
    def test_range_is_point_one_to_one(self, kind):
        pts = _grid_for_kind(kind, 3)
        assert pts[0] == 0.1, "gain/volume grid starts at 0.1, not 0.0"
        assert pts[-1] == 1.0

    @pytest.mark.parametrize("kind", ["drive", "rms"])
    def test_sorted_ascending_no_duplicates(self, kind):
        pts = _grid_for_kind(kind, 5)
        assert pts == sorted(pts)
        assert len(pts) == len(set(pts))

    def test_larger_grid_points_adds_real_middle_coverage(self):
        """A larger --grid-points should add genuine middle-of-range points, not just
        repeat the fixed anchors -- otherwise --grid-points would be a no-op for this kind."""
        small = _grid_for_kind("drive", 3)
        large = _grid_for_kind("drive", 9)
        assert len(large) > len(small)

    def test_default_three_points_matches_documented_example(self):
        """README/module docstring's own worked example: --grid-points=3 on [0.1,1.0] gives
        a linspace baseline of [0.1, 0.55, 1.0], merged with the anchors -> exactly 6 points."""
        assert _grid_for_kind("drive", 3) == [0.1, 0.15, 0.25, 0.55, 0.95, 1.0]


class TestUnclassifiedKnobGrid:
    """No role match: legacy behavior, evenly spaced full [0, 1] range."""

    def test_full_range(self):
        pts = _grid_for_kind(None, 3)
        assert pts[0] == 0.0
        assert pts[-1] == 1.0

    def test_evenly_spaced(self):
        pts = _grid_for_kind(None, 5)
        gaps = np.diff(pts)
        assert np.allclose(gaps, gaps[0])


class TestFormatKnobs:
    def test_one_line_per_knob_with_correct_grid(self):
        lines = _format_knobs(["Gain", "Treble"], 3)
        assert len(lines) == 2
        assert "Gain" in lines[0] and "[0.1, 0.15, 0.25, 0.55, 0.95, 1.0]" in lines[0]
        assert "Treble" in lines[1] and "[0.2, 0.5, 0.8]" in lines[1]

    def test_names_aligned_to_widest(self):
        lines = _format_knobs(["Gain", "ReallyLongKnobName"], 3)
        eq_positions = [ln.index("=") for ln in lines]
        assert eq_positions[0] == eq_positions[1], "TOML assignment columns should line up"


class TestFormatKnobKind:
    def test_known_name_writes_live_assignment(self):
        lines = _format_knob_kind(["Gain"])
        assert lines[0].startswith("Gain")
        assert '"drive"' in lines[0]
        assert "guessed from name" in lines[0]

    def test_unknown_name_is_commented_out(self):
        lines = _format_knob_kind(["Zorp"])
        assert lines[0].lstrip().startswith("#")
        assert "UNCONFIRMED" in lines[0]

    def test_mixed_known_and_unknown(self):
        lines = _format_knob_kind(["Gain", "Zorp", "Treble"])
        assert not lines[0].lstrip().startswith("#")
        assert lines[1].lstrip().startswith("#")
        assert not lines[2].lstrip().startswith("#")


class TestReplaceTable:
    """The splice helper that writes [knobs]/[knob-kind] into a config template. Its own
    boundary-finding regex only scans the EXISTING template content to know what to cut --
    the NEW body_lines are inserted verbatim, so a commented-out line among otherwise-live
    assignments (exactly what an UNCONFIRMED knob-kind entry looks like) must not corrupt
    the splice. This is the property that made mixing commented/live lines in
    _format_knob_kind's output safe -- tested directly, not just inferred."""

    TEMPLATE = (
        "intro line\n"
        "[knobs]\n"
        "Knob1 = [0.0, 0.5, 1.0]\n"
        "Knob2 = [0.0, 0.5, 1.0]\n"
        "\n"
        "[fixed]\n"
        "# comment\n"
    )

    def test_replaces_existing_table_body(self):
        out = _replace_table(self.TEMPLATE, "knobs", ["Gain = [0.1, 1.0]"])
        assert "Gain = [0.1, 1.0]" in out
        assert "Knob1" not in out
        assert "Knob2" not in out

    def test_preserves_surrounding_content(self):
        out = _replace_table(self.TEMPLATE, "knobs", ["Gain = [0.1, 1.0]"])
        assert out.startswith("intro line\n")
        assert "[fixed]" in out
        assert "# comment" in out

    def test_commented_line_in_new_body_does_not_corrupt_splice(self):
        body = ["Gain = [0.1, 1.0]", "# Zorp = \"UNCONFIRMED\"   # unrecognized", "Treble = [0.2, 0.8]"]
        out = _replace_table(self.TEMPLATE, "knobs", body)
        for line in body:
            assert line in out
        assert "[fixed]" in out, "content after the table must survive intact"

    def test_missing_header_raises(self):
        with pytest.raises(ValueError):
            _replace_table(self.TEMPLATE, "nonexistent-table", ["x = 1"])


class TestReplaceLine:
    TEMPLATE = "schx       = \"placeholder.schx\"\ninput      = \"sweep.wav\"\n"

    def test_replaces_matching_line(self):
        out = _replace_line(self.TEMPLATE, "schx", 'schx       = "real.schx"')
        assert 'schx       = "real.schx"' in out
        assert "placeholder.schx" not in out

    def test_leaves_other_lines_untouched(self):
        out = _replace_line(self.TEMPLATE, "schx", 'schx       = "real.schx"')
        assert 'input      = "sweep.wav"' in out

    def test_missing_key_raises(self):
        with pytest.raises(ValueError):
            _replace_line(self.TEMPLATE, "nonexistent", "x = 1")
