"""repair_dataset_glitches.py: median-of-three outlier patching of sporadic solver glitches.

The properties that matter, each guarded below:
  * a glitch that only the DATASET row has is found and removed;
  * a glitch that only one of the two EXTRA renders has (the dataset row is fine) touches nothing;
  * every sample outside a patched span is left bit-identical -- the repair must never degrade a
    row that was already right;
  * the original file is preserved, the repair is recorded, and a second run is refused.
"""
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

import repair_dataset_glitches as R
from capture_chain import capture_chain

SR = 48000
N = SR                      # 1 s rows keep the tests fast
SCALE = 0.5


def clean_signal(seed=0, n=N):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    return 0.4 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 3100 * t) \
        + 1e-4 * rng.standard_normal(n)


def noisy_refs(x0, noise=1e-3, seed=7):
    """Two independent extra renders: like real renders they differ from each other and from the
    dataset row by small noise, so the median is NOT equal to the row outside a glitch."""
    rng = np.random.default_rng(seed)
    return x0 + noise * rng.standard_normal(len(x0)), x0 + noise * rng.standard_normal(len(x0))


def spike(x, at, width=48, amp=0.3):
    y = x.copy()
    y[at:at + width] += amp * np.hanning(width)
    return y


# --------------------------------------------------------------------------- detect_spans

def test_a_glitch_only_the_dataset_row_has_is_found_and_removed():
    x0 = clean_signal()
    x = spike(x0, 20000)
    a, b = x0 + 1e-4, x0 - 1e-4
    spans, m = R.detect_spans(x, a, b, tol=0.01, pad=96, merge_gap=240)
    assert len(spans) == 1
    s, e = spans[0]
    assert s <= 20000 and e >= 20048                     # the whole glitch is inside the span
    fixed = R.apply_patch(x, m, spans, fade=24)
    assert np.abs(fixed - x0).max() < 5e-4               # back to the clean signal, to noise level


def test_a_glitch_in_only_one_extra_render_flags_nothing():
    x0 = clean_signal()
    good, bad = x0 + 1e-4, spike(x0, 30000, amp=5.0)
    for a, b in ((bad, good), (good, bad)):
        spans, _ = R.detect_spans(x0, a, b, tol=0.01, pad=96, merge_gap=240)
        assert spans == []


def test_samples_outside_every_span_are_bit_identical():
    x0 = clean_signal()
    x = spike(x0, 12000)
    a, b = noisy_refs(x0)
    spans, m = R.detect_spans(x, a, b, 0.01, 96, 240)
    assert not np.array_equal(m[:5000], x[:5000])          # the median genuinely differs from x
    fixed = R.apply_patch(x, m, spans, fade=24)
    keep = np.ones(N, bool)
    for s, e in spans:
        keep[s:e] = False
    assert np.array_equal(fixed[keep], x[keep])


def test_two_renders_agreeing_on_the_same_wrong_value_win_the_vote():
    """Documented limit of median-of-three: if two renders share a glitch, it is not fixed."""
    x0 = clean_signal()
    bad = spike(x0, 9000)
    spans, _ = R.detect_spans(bad, bad + 1e-5, x0, 0.01, 96, 240)
    assert spans == []


def test_noise_below_tol_is_never_flagged():
    x0 = clean_signal()
    rng = np.random.default_rng(1)
    x = x0 + 2e-3 * rng.standard_normal(N)
    spans, _ = R.detect_spans(x, x0, x0, tol=0.01, pad=96, merge_gap=240)
    assert spans == []


def test_nearby_glitches_merge_into_one_span_and_distant_ones_stay_separate():
    x0 = clean_signal()
    near = spike(spike(x0, 10000), 10200)                # 200 samples apart (~4 ms)
    far = spike(spike(x0, 10000), 30000)
    ref = (x0 + 1e-4, x0 - 1e-4)
    assert len(R.detect_spans(near, *ref, 0.01, 24, 240)[0]) == 1
    assert len(R.detect_spans(far, *ref, 0.01, 24, 240)[0]) == 2


def test_padded_spans_that_overlap_are_merged_into_one():
    """Runs further apart than merge_gap stay separate RUNS, but their padding can still overlap."""
    x0 = clean_signal()
    x = spike(spike(x0, 10000), 10300)                  # 300 samples apart > merge_gap of 100
    spans, _ = R.detect_spans(x, x0 + 1e-4, x0 - 1e-4, 0.01, pad=500, merge_gap=100)
    assert len(spans) == 1


def test_spans_are_clamped_to_the_array_at_both_ends():
    x0 = clean_signal()
    x = spike(spike(x0, 0), N - 48)
    spans, m = R.detect_spans(x, x0 + 1e-4, x0 - 1e-4, 0.01, pad=2000, merge_gap=240)
    assert spans[0][0] == 0 and spans[-1][1] == N
    assert np.isfinite(R.apply_patch(x, m, spans, fade=24)).all()


def test_patch_has_no_step_at_the_span_edges():
    x0 = clean_signal()
    x = spike(x0, 20000)
    spans, m = R.detect_spans(x, x0 + 1e-3, x0 - 1e-3, 0.01, 96, 240)
    fixed = R.apply_patch(x, m, spans, fade=24)
    assert np.abs(np.diff(fixed)).max() <= np.abs(np.diff(x0)).max() * 1.5 + 2e-3


def test_no_glitch_means_no_spans_and_an_unchanged_row():
    x0 = clean_signal()
    spans, m = R.detect_spans(x0, x0, x0, 0.01, 96, 240)
    assert spans == [] and np.array_equal(R.apply_patch(x0, m, spans, 24), x0)


# --------------------------------------------------------------------------- process_render

def test_process_render_applies_the_chain_then_the_scale():
    y = clean_signal()
    cap = {"corner_hz": 18.0, "order": 3}
    got = R.process_render(y, SR, cap, SCALE)
    want = capture_chain(y.astype(np.float32), SR, corner_hz=18.0, order=3) * SCALE
    assert np.allclose(got, want, atol=1e-7)
    assert np.allclose(R.process_render(y, SR, None, SCALE), y.astype(np.float32) * SCALE, atol=1e-7)


# --------------------------------------------------------------------------- repair() end to end

def build_dataset(tmp_path, capture=None, glitch_rows=(1,)):
    """3 combos. Dataset rows = processed clean renders, with a glitch in each glitch_rows row.
    Returns (dataset dir, raw clean signals, patterns for the two extra renders)."""
    ds = tmp_path / "ds"
    ds.mkdir()
    raws = [clean_signal(seed=i) for i in range(3)]
    rows = []
    for i, r in enumerate(raws):
        row = R.process_render(r, SR, capture, SCALE)
        rows.append(spike(row, 20000 + 100 * i) if i in glitch_rows else row)
    np.save(ds / "outputs.npy", np.stack(rows).astype(np.float32))
    (ds / "config.json").write_text(json.dumps(
        {"input": {"samplerate": SR}, "output_scale": SCALE, "capture_chain": capture}))
    for tag, eps in (("a", 1e-4), ("b", -1e-4)):
        for i, r in enumerate(raws):
            sf.write(str(tmp_path / f"c{i:02d}_{tag}.wav"), r + eps, SR, subtype="FLOAT")
    return ds, raws, str(tmp_path / "c{idx:02d}_a.wav"), str(tmp_path / "c{idx:02d}_b.wav")


def run(ds, pa, pb, **kw):
    args = dict(tol=0.01, pad_ms=2, merge_ms=5, fade_ms=0.5)
    args.update(kw)
    return R.repair(ds, pa, pb, log=lambda *_: None, **args)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.mark.parametrize("capture", [None, {"corner_hz": 18.0, "order": 3}])
def test_repair_patches_only_the_glitchy_row_and_keeps_the_original(tmp_path, capture):
    ds, raws, pa, pb = build_dataset(tmp_path, capture)
    before = np.load(ds / "outputs.npy").copy()
    orig_hash = sha(ds / "outputs.npy")
    summary = run(ds, pa, pb)
    after = np.load(ds / "outputs.npy")
    assert summary["repaired"] == [1]
    assert np.array_equal(after[0], before[0]) and np.array_equal(after[2], before[2])
    clean = R.process_render(raws[1], SR, capture, SCALE)
    assert np.abs(before[1] - clean).max() > 0.05        # it really was glitchy
    assert np.abs(after[1] - clean).max() < 1e-3         # and now is not
    assert np.array_equal(np.load(ds / "outputs.pre_repair.npy"), before)
    rec = json.loads((ds / "config.json").read_text())["glitch_repair"]
    assert rec["pre_repair_sha256"] == orig_hash and rec["combos"]["1"]["spans"]
    assert rec["tol"] == 0.01 and rec["pre_repair_file"] == "outputs.pre_repair.npy"


def test_dry_run_reports_but_writes_nothing(tmp_path):
    ds, _, pa, pb = build_dataset(tmp_path)
    h = sha(ds / "outputs.npy")
    cfg = (ds / "config.json").read_text()
    summary = run(ds, pa, pb, dry_run=True)
    assert summary["repaired"] == [1] and summary["combos"][1]["spans"]   # it FOUND the glitch
    assert sha(ds / "outputs.npy") == h and (ds / "config.json").read_text() == cfg
    assert not (ds / "outputs.pre_repair.npy").exists()


def test_a_second_repair_is_refused_and_force_never_clobbers_the_original(tmp_path):
    ds, _, pa, pb = build_dataset(tmp_path)
    run(ds, pa, pb)
    with pytest.raises(SystemExit, match="already repaired"):
        run(ds, pa, pb)
    # Re-introduce a glitch so --force has something to patch: it gets past the flag, but the
    # pre-repair copy is the only original and must never be overwritten.
    orig = np.load(ds / "outputs.pre_repair.npy").copy()
    arr = np.load(ds / "outputs.npy")
    arr[2] = spike(arr[2].astype(np.float64), 5000).astype(np.float32)
    np.save(ds / "outputs.npy", arr)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run(ds, pa, pb, force=True)
    assert np.array_equal(np.load(ds / "outputs.pre_repair.npy"), orig)


def test_clean_dataset_is_left_alone(tmp_path):
    ds, _, pa, pb = build_dataset(tmp_path, glitch_rows=())
    h = sha(ds / "outputs.npy")
    summary = run(ds, pa, pb)
    assert summary["repaired"] == [] and sha(ds / "outputs.npy") == h
    assert not (ds / "outputs.pre_repair.npy").exists()
    assert "glitch_repair" not in json.loads((ds / "config.json").read_text())


def test_a_combination_with_a_missing_render_is_skipped_and_reported(tmp_path):
    ds, _, pa, pb = build_dataset(tmp_path)
    (tmp_path / "c01_b.wav").unlink()
    summary = run(ds, pa, pb)
    assert summary["skipped"] == [1] and summary["repaired"] == []


def test_length_mismatch_between_a_render_and_the_dataset_is_an_error(tmp_path):
    ds, raws, pa, pb = build_dataset(tmp_path)
    sf.write(str(tmp_path / "c01_a.wav"), raws[1][:-10], SR, subtype="FLOAT")
    with pytest.raises(SystemExit, match="lengths differ"):
        run(ds, pa, pb)


def test_combos_filter_limits_what_is_considered(tmp_path):
    ds, _, pa, pb = build_dataset(tmp_path, glitch_rows=(0, 1))
    summary = run(ds, pa, pb, combos=[1])
    assert summary["repaired"] == [1] and 0 not in summary["combos"]


def test_cli_dry_run(tmp_path, monkeypatch, capsys):
    ds, _, pa, pb = build_dataset(tmp_path)
    monkeypatch.setattr("sys.argv", ["repair_dataset_glitches.py", "--dataset", str(ds),
                                     "--render-a", pa, "--render-b", pb, "--dry-run"])
    R.main()
    out = capsys.readouterr().out
    assert "combo   1" in out and "dry run" in out
    assert not (ds / "outputs.pre_repair.npy").exists()
