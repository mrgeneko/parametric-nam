"""plan_onset_refinement: which floor-limited corners still need the downward extension."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_excitation import plan_onset_refinement, knee_at_floor  # noqa: E402


def _curve(gains, start=0.005, step=2.0):
    """Curve of (in_v, out_rms) from a list of small-signal gains at geometric inputs."""
    return [(start * step ** i, g * start * step ** i) for i, g in enumerate(gains)]


RESOLVED = {"curve": _curve([100, 100, 100, 60, 30, 15])}      # flat, then falls: knee in range
FLOOR = {"curve": _curve([100, 80, 60, 40, 20, 10])}           # falling from the first point


def test_knee_at_floor_classification():
    assert not knee_at_floor(RESOLVED)
    assert knee_at_floor(FLOOR)
    assert not knee_at_floor({})              # no curve: never claims a floor
    assert not knee_at_floor({"curve": [(0.005, 1.0)]})


def test_skips_floor_limited_corner_far_below_resolved_max():
    p1 = {0: (0.9, RESOLVED), 1: (0.1, FLOOR), 2: (0.6, FLOOR)}
    floor, refine, m = plan_onset_refinement(p1, 0.5)
    assert floor == {1, 2} and m == 0.9
    assert refine == [2]                       # 0.6 >= 0.5*0.9 -> could still be the max
    # 0.1 < 0.45 -> skipped


def test_margin_one_refines_only_at_or_above_the_max():
    p1 = {0: (0.9, RESOLVED), 1: (0.89, FLOOR), 2: (0.95, FLOOR)}
    _, refine, _ = plan_onset_refinement(p1, 1.0)
    assert refine == [2]


def test_no_resolved_corner_refines_everything():
    """The Powerball case: nothing to compare against, so nothing may be pruned."""
    p1 = {0: (0.001, FLOOR), 1: (0.008, FLOOR), 2: (0.002, FLOOR)}
    floor, refine, m = plan_onset_refinement(p1, 0.5)
    assert m is None and refine == [0, 1, 2] and floor == {0, 1, 2}


def test_missing_onset_is_always_refined_never_pruned():
    p1 = {0: (0.9, RESOLVED), 1: (None, {"curve": []})}
    floor, refine, _ = plan_onset_refinement(p1, 0.5)
    assert 1 in floor and refine == [1]


def test_all_resolved_means_nothing_to_extend():
    p1 = {0: (0.9, RESOLVED), 1: (0.4, RESOLVED)}
    floor, refine, m = plan_onset_refinement(p1, 0.5)
    assert floor == set() and refine == [] and m == 0.9


# ---- end to end through worst_case_onset, with a stub that honours min_start_v ----
import pytest  # noqa: E402
from prepare_excitation import worst_case_onset  # noqa: E402

FLOOR_CURVE = [list(p) for p in FLOOR["curve"]]
RESOLVED_CURVE = [list(p) for p in RESOLVED["curve"]]


class TestWorstCaseOnsetPruning:
    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def _stub(self, monkeypatch, calls):
        """Corners (Gain, Tone): (0,0) resolved 5.0 | (0,1) floor-limited 0.1 | (1,0) floor-limited
        4.0 whose extended onset is 6.0 | (1,1) resolved 1.0."""
        def fake(backend, params, tmp, max_v=40.0, lead_silence_s=0.0, min_start_v=1e-9,
                 start_v=0.005, **kw):
            calls.append((params["Gain"], params["Tone"], min_start_v))
            key = (params["Gain"], params["Tone"])
            extends = min_start_v < start_v
            if key == (0.0, 0.0):
                return {"onset_99pct_input_v": 5.0, "curve": RESOLVED_CURVE}
            if key == (1.0, 1.0):
                return {"onset_99pct_input_v": 1.0, "curve": RESOLVED_CURVE}
            if key == (0.0, 1.0):
                return {"onset_99pct_input_v": 0.1, "curve": FLOOR_CURVE}
            return {"onset_99pct_input_v": 6.0 if extends else 4.0, "curve": FLOOR_CURVE}
        monkeypatch.setattr("prepare_excitation.find_saturation_point", fake)

    def _run(self, tmp_path, **kw):
        return worst_case_onset(backend=object(), identity=b"x", cache_extra="e",
                                knob_ranges={"Gain": [0.0, 1.0], "Tone": [0.0, 1.0]}, fixed={},
                                tmp=str(tmp_path), quiet=True, no_cache=True, **kw)

    def test_default_prunes_the_corner_that_cannot_be_the_max(self, monkeypatch, tmp_path):
        calls = []
        self._stub(monkeypatch, calls)
        worst, rows = self._run(tmp_path)
        pass1 = [c for c in calls if c[2] == 0.005]
        pass2 = [c for c in calls if c[2] != 0.005]
        assert len(pass1) == 4                                  # every corner, no extension
        assert pass2 == [(1.0, 0.0, 1e-9)]                      # only the one within the margin
        assert worst == pytest.approx(6.0)                      # the refined value, not 4.0
        pruned = [r for r in rows if r.get("pruned_unrefined")]
        assert len(pruned) == 1 and pruned[0]["onset_v"] == pytest.approx(0.1)

    def test_opt_out_extends_every_corner_once_and_marks_none(self, monkeypatch, tmp_path):
        calls = []
        self._stub(monkeypatch, calls)
        worst, rows = self._run(tmp_path, prune_onset_margin=None)
        assert len(calls) == 4 and all(c[2] == 1e-9 for c in calls)
        assert not any(r.get("pruned_unrefined") for r in rows)
        assert worst == pytest.approx(6.0)

    def test_pruning_does_not_change_the_worst_case(self, monkeypatch, tmp_path):
        a, b = [], []
        self._stub(monkeypatch, a)
        w_prune, _ = self._run(tmp_path)
        self._stub(monkeypatch, b)
        w_full, _ = self._run(tmp_path, prune_onset_margin=None)
        assert w_prune == pytest.approx(w_full)

    def test_no_extension_possible_means_no_second_pass(self, monkeypatch, tmp_path):
        calls = []
        self._stub(monkeypatch, calls)
        self._run(tmp_path, min_start_v=0.005)                  # floor == start: nothing to extend
        assert len(calls) == 4
