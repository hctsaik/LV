"""Unit tests for scripts/completeness.py — the coverage completeness
heatmap (attribute-grid binning, embedding diversity probe, five-state
classification, Coverage Health)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from completeness import (
    STATE_EMPTY,
    STATE_FAKE,
    STATE_HEALTHY,
    STATE_LOW,
    STATE_MISSING,
    STATE_OVER,
    bucketize,
    build_completeness,
    categorical_buckets,
    cell_diversity,
    classify_cell,
    compute_target,
    coverage_health,
    effective_rank,
    image_stats,
)


# ── module hygiene ──────────────────────────────────────────────────────

def test_completeness_no_streamlit_import():
    import completeness
    assert "import streamlit" not in Path(completeness.__file__).read_text(encoding="utf-8")


# ── bucketize / categorical ─────────────────────────────────────────────

def test_bucketize_quantile_equal_population():
    vals = list(range(100))
    idx, labels = bucketize(vals, 4, method="quantile")
    assert len(labels) == 4
    counts = np.bincount(idx, minlength=4)
    assert counts.min() >= 20  # roughly balanced

def test_bucketize_uniform_width():
    idx, labels = bucketize([0, 1, 2, 9, 10], 2, method="uniform")
    assert len(labels) == 2
    assert idx[0] == 0 and idx[-1] == 1

def test_bucketize_degenerate_all_equal():
    idx, labels = bucketize([5, 5, 5], 3)
    assert labels == ["全部"] and set(idx) == {0}

def test_bucketize_empty():
    assert bucketize([], 3) == ([], [])

def test_categorical_buckets_sorted():
    idx, labels = categorical_buckets(["b", "a", "b", "c"])
    assert labels == ["a", "b", "c"]
    assert idx == [1, 0, 1, 2]


# ── effective_rank / diversity ──────────────────────────────────────────

def test_effective_rank_duplicates_near_one():
    emb = np.ones((10, 8))  # identical rows → no spread
    assert effective_rank(emb) == pytest.approx(1.0, abs=1e-6)

def test_effective_rank_isotropic_high():
    emb = np.eye(8)  # spread across all axes
    assert effective_rank(emb) > 5.0

def test_cell_diversity_duplicates_low_spread_high_zero():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 16))
    g = 0.5
    dup = np.repeat(base[:1], 10, axis=0)  # 10 copies of one row
    assert cell_diversity(dup, list(range(10)), g) == pytest.approx(0.0, abs=1e-6)
    diverse = cell_diversity(base, list(range(20)), g)
    assert diverse > 0.0

def test_cell_diversity_single_sample_is_zero():
    assert cell_diversity(np.zeros((1, 4)), [0], 0.5) == 0.0


# ── classify_cell five states ───────────────────────────────────────────

def test_classify_cell_all_states():
    t, d_star = 10, 0.6
    assert classify_cell(0, 0.0, t, d_star) == STATE_EMPTY
    assert classify_cell(3, 0.0, t, d_star) == STATE_MISSING      # < 0.5t
    assert classify_cell(7, 0.0, t, d_star) == STATE_LOW          # 0.5t..t
    assert classify_cell(12, 0.8, t, d_star) == STATE_HEALTHY     # >= t, diverse
    assert classify_cell(12, 0.3, t, d_star) == STATE_FAKE        # >= t, dup
    assert classify_cell(20, 0.8, t, d_star) == STATE_OVER        # >= 1.5t, diverse

def test_classify_fake_beats_over_when_low_diversity():
    # plenty of samples but near-duplicates → FAKE, not OVER
    assert classify_cell(40, 0.2, 10, 0.6) == STATE_FAKE


# ── compute_target ──────────────────────────────────────────────────────

def test_compute_target_uncalibrated_falls_back_to_floor():
    assert compute_target(20, t_abs=30) == 30.0

def test_compute_target_frequency_raises_above_floor():
    # a high-frequency cell deserves more than the floor
    t = compute_target(10, t_abs=10, freq_weight=0.5, n_total=1000, p=1.0)
    assert t == 500.0

def test_compute_target_boundary_weight():
    assert compute_target(10, t_abs=10, boundary_w=1.3) == pytest.approx(13.0)


# ── coverage_health ─────────────────────────────────────────────────────

def _cell(n, d, t, state):
    return {"n": n, "d": d, "t": t, "state": state}

def test_coverage_health_all_healthy_is_100():
    cells = [_cell(15, 0.8, 10, STATE_HEALTHY) for _ in range(4)]
    h = coverage_health(cells)
    assert h["coverage_health"] == 100.0
    assert h["counts"][STATE_HEALTHY] == 4

def test_coverage_health_empty_cells_drag_score_down():
    cells = [_cell(15, 0.8, 10, STATE_HEALTHY), _cell(0, 0.0, 10, STATE_EMPTY)]
    h = coverage_health(cells)
    assert h["coverage_health"] == 50.0  # one full, one empty
    assert h["top_gaps"][0]["shortfall"] == 10.0

def test_coverage_health_fake_penalized():
    full = coverage_health([_cell(10, 0.8, 10, STATE_HEALTHY)])["coverage_health"]
    fake = coverage_health([_cell(10, 0.2, 10, STATE_FAKE)])["coverage_health"]
    assert fake < full and fake == pytest.approx(70.0)

def test_coverage_health_gini_balanced_vs_skewed():
    balanced = coverage_health([_cell(10, 0.8, 10, STATE_HEALTHY) for _ in range(4)])
    skewed = coverage_health(
        [_cell(40, 0.8, 10, STATE_OVER)] + [_cell(0, 0.0, 10, STATE_EMPTY)] * 3)
    assert skewed["gini"] > balanced["gini"]

def test_coverage_health_empty_input():
    h = coverage_health([])
    assert h["coverage_health"] == 0.0 and h["top_gaps"] == []


# ── build_completeness end to end ───────────────────────────────────────

def test_build_completeness_grid_shape_and_missing_cells():
    rng = np.random.default_rng(1)
    # 3 classes x 2 buckets, but deliberately leave one combo empty
    records = [{"label": "a"}] * 12 + [{"label": "b"}] * 12 + [{"label": "c"}] * 12
    emb = rng.normal(size=(36, 16))
    bx, lx = categorical_buckets([r["label"] for r in records])
    # class a entirely in bucket 0 → cell (a, hi) is a genuine missing combo
    by = [0] * 12 + [0, 1] * 6 + [0, 1] * 6
    out = build_completeness(records, emb, bx, by, lx, ["lo", "hi"], t_abs=5)
    assert len(out["cells"]) == 3 * 2  # full grid, including empties
    states = {(c["x"], c["y"]): c["state"] for c in out["cells"]}
    assert states[(0, 1)] == STATE_EMPTY  # the deliberately empty combo
    assert 0 <= out["health"]["coverage_health"] <= 100
    assert out["calibrated"] is False

def test_build_completeness_indices_partition_records():
    rng = np.random.default_rng(2)
    records = [{"label": "x"}] * 20
    emb = rng.normal(size=(20, 8))
    bx, lx = categorical_buckets([r["label"] for r in records])
    by, ly = bucketize(list(range(20)), 4)
    out = build_completeness(records, emb, bx, by, lx, ly, t_abs=3)
    all_idx = sorted(i for c in out["cells"] for i in c["indices"])
    assert all_idx == list(range(20))  # every record lands in exactly one cell


# ── image_stats ─────────────────────────────────────────────────────────

def test_image_stats_fields_and_ranges(tmp_path):
    p = tmp_path / "x.jpg"
    arr = np.random.default_rng(0).integers(0, 255, (80, 60, 3)).astype("uint8")
    Image.fromarray(arr).save(p)
    s = image_stats(p)
    assert set(s) == {"brightness", "contrast", "sharpness", "aspect"}
    assert 0.0 <= s["brightness"] <= 1.0 and 0.0 <= s["contrast"] <= 1.0
    assert s["sharpness"] >= 0.0
    assert s["aspect"] == pytest.approx(60 / 80, abs=1e-3)

def test_image_stats_bright_vs_dark(tmp_path):
    bright = tmp_path / "b.png"; dark = tmp_path / "d.png"
    Image.fromarray(np.full((32, 32, 3), 230, "uint8")).save(bright)
    Image.fromarray(np.full((32, 32, 3), 20, "uint8")).save(dark)
    assert image_stats(bright)["brightness"] > image_stats(dark)["brightness"]

def test_image_stats_missing_raises(tmp_path):
    with pytest.raises(OSError):
        image_stats(tmp_path / "nope.jpg")


# ── (a) frequency-prior calibration ─────────────────────────────────────

from completeness import STATE_NA, cell_centroid, mine_candidates  # noqa: E402


def test_freq_class_na_excluded_from_health():
    rng = np.random.default_rng(3)
    records = [{"label": "a"}] * 10 + [{"label": "b"}] * 10
    emb = rng.normal(size=(20, 8))
    bx, lx = categorical_buckets([r["label"] for r in records])
    by = [0] * 20  # single y bucket
    # mark cell (b,0) as 不適用 → must not drag coverage even if empty-ish
    fclasses = {(1, 0): STATE_NA}
    out = build_completeness(records, emb, bx, by, lx, ["all"], t_abs=5,
                             freq_classes=fclasses)
    states = {(c["x"], c["y"]): c["state"] for c in out["cells"]}
    assert states[(1, 0)] == STATE_NA
    # health denominator excludes the NA cell (only cell a is scored)
    assert out["health"]["counts"][STATE_NA] == 1
    assert out["calibrated"] is True


def test_freq_class_low_lowers_target_high_raises():
    rng = np.random.default_rng(4)
    records = [{"label": "a"}] * 6
    emb = rng.normal(size=(6, 8))
    bx, lx = categorical_buckets([r["label"] for r in records])
    by = [0] * 6
    low = build_completeness(records, emb, bx, by, lx, ["all"], t_abs=10,
                             freq_classes={(0, 0): "低"})
    high = build_completeness(records, emb, bx, by, lx, ["all"], t_abs=10,
                              freq_classes={(0, 0): "高"})
    assert low["cells"][0]["t"] == 4.0    # 10 * 0.4
    assert high["cells"][0]["t"] == 20.0  # 10 * 2.0
    # 6 samples: meets the 低 target (4) but misses the 高 target (20)
    assert low["cells"][0]["state"] != STATE_MISSING
    assert high["cells"][0]["n"] < high["cells"][0]["t"]


# ── (b) candidate mining ────────────────────────────────────────────────

def test_cell_centroid_and_empty():
    emb = np.array([[1.0, 0.0], [3.0, 0.0]])
    assert np.allclose(cell_centroid(emb, [0, 1]), [2.0, 0.0])
    assert cell_centroid(emb, []) is None


def test_mine_candidates_ranks_by_proximity():
    pool = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    q = np.array([1.0, 0.0])
    idx, dist = mine_candidates(pool, q, k=2)
    assert idx[:2] == [0, 1]            # the two aligned with the query
    assert dist == sorted(dist)

def test_mine_candidates_max_distance_drops_far():
    pool = np.array([[1.0, 0.0], [0.0, 1.0]])  # one near, one orthogonal
    idx, _ = mine_candidates(pool, np.array([1.0, 0.0]), k=5, max_distance=0.1)
    assert idx == [0]

def test_mine_candidates_empty_pool():
    assert mine_candidates(np.zeros((0, 4)), np.ones(4)) == ([], [])
