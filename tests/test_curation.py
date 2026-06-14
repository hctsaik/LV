"""Unit tests for the F4/F5 curation functions in scripts/interaction.py:
label disagreement (kNN label audit) and duplicate / leakage pair scans."""
from __future__ import annotations

import numpy as np
import pytest

from interaction import (
    compute_label_disagreement,
    find_duplicate_pairs_embedding,
    find_duplicate_pairs_phash,
    hamming_distance_hex,
)


# ── compute_label_disagreement (F5) ─────────────────────────────────────

def _two_clusters(n_per: int = 10, seed: int = 0):
    """Two tight, well-separated clusters in 4-D."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.01, size=(n_per, 4)) + np.array([1, 0, 0, 0])
    b = rng.normal(loc=0.0, scale=0.01, size=(n_per, 4)) + np.array([0, 1, 0, 0])
    return np.vstack([a, b])


def test_disagreement_flags_point_with_wrong_label():
    emb = _two_clusters()
    labels = ["A"] * 10 + ["B"] * 10
    labels[3] = "B"  # one point in cluster A carries cluster B's label
    scores = compute_label_disagreement(emb, labels, k=5)
    assert scores[3] == pytest.approx(1.0)  # all its neighbours say A
    clean = [s for i, s in enumerate(scores) if i != 3]
    assert max(clean) <= 0.4  # clean points stay low
    assert int(np.argmax(scores)) == 3


def test_disagreement_zero_when_single_class():
    emb = _two_clusters()
    scores = compute_label_disagreement(emb, ["same"] * 20, k=5)
    assert np.all(scores == 0.0)


def test_disagreement_scores_in_unit_range_and_aligned():
    emb = _two_clusters(n_per=6)
    labels = ["A"] * 6 + ["B"] * 6
    scores = compute_label_disagreement(emb, labels, k=3)
    assert scores.shape == (12,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_disagreement_k_clamped_and_tiny_inputs():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = compute_label_disagreement(emb, ["A", "B"], k=99)
    assert scores.shape == (2,)
    assert np.all(scores == 1.0)  # each point's only neighbour disagrees
    assert compute_label_disagreement(np.zeros((1, 2)), ["A"], k=5).tolist() == [0.0]
    assert compute_label_disagreement(np.zeros((0, 2)), [], k=5).shape == (0,)


def test_disagreement_self_excluded_with_duplicate_rows():
    # two byte-identical rows with different labels: each sees the OTHER
    # (not itself) as nearest neighbour → both must score 1.0 with k=1
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scores = compute_label_disagreement(emb, ["A", "B", "C"], k=1)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1.0)


# ── hamming + phash pair scan (F4) ──────────────────────────────────────

def test_hamming_distance_hex():
    assert hamming_distance_hex("00" * 8, "00" * 8) == 0
    assert hamming_distance_hex("0" * 16, "1" + "0" * 15) == 1
    assert hamming_distance_hex("f" * 16, "0" * 16) == 64


def test_phash_pairs_exact_and_near_duplicates():
    h = ["aa" * 8, "aa" * 8, "ab" * 8, "ff" * 8]  # 0↔1 identical
    pairs = find_duplicate_pairs_phash(h, max_hamming=0)
    assert pairs == [(0, 1, 0)]
    near = find_duplicate_pairs_phash(["0" * 16, "1" + "0" * 15], max_hamming=1)
    assert near == [(0, 1, 1)]


def test_phash_pairs_skip_none_and_sorted_capped():
    h = ["aa" * 8, None, "aa" * 8, "aa" * 8]
    pairs = find_duplicate_pairs_phash(h, max_hamming=0)
    assert (0, 2, 0) in pairs and (0, 3, 0) in pairs and (2, 3, 0) in pairs
    assert all(j != 1 and i != 1 for i, j, _ in pairs)
    capped = find_duplicate_pairs_phash(h, max_hamming=0, max_pairs=2)
    assert len(capped) == 2


def test_phash_pairs_cross_split_only_is_leakage_filter():
    h = ["aa" * 8] * 4
    splits = ["train", "train", "val", "val"]
    pairs = find_duplicate_pairs_phash(h, max_hamming=0, splits=splits,
                                       cross_split_only=True)
    assert pairs == [(0, 2, 0), (0, 3, 0), (1, 2, 0), (1, 3, 0)]


# ── embedding pair scan (F4) ────────────────────────────────────────────

def test_embedding_pairs_find_duplicate_rows():
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pairs = find_duplicate_pairs_embedding(emb, max_distance=0.01)
    assert len(pairs) == 1
    i, j, d = pairs[0]
    assert (i, j) == (0, 1) and d == pytest.approx(0.0, abs=1e-9)


def test_embedding_pairs_threshold_and_cross_split():
    emb = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0], [0.01, 0.999]])
    splits = ["train", "val", "train", "val"]
    all_pairs = find_duplicate_pairs_embedding(emb, max_distance=0.01)
    assert {(p[0], p[1]) for p in all_pairs} == {(0, 1), (2, 3)}
    cross = find_duplicate_pairs_embedding(emb, max_distance=0.01, splits=splits,
                                           cross_split_only=True)
    assert {(p[0], p[1]) for p in cross} == {(0, 1), (2, 3)}
    none = find_duplicate_pairs_embedding(emb, max_distance=1e-6)
    assert none == []


def test_embedding_pairs_tiny_input():
    assert find_duplicate_pairs_embedding(np.zeros((1, 4))) == []
    assert find_duplicate_pairs_embedding(np.zeros((0, 4))) == []


# ── escape report-card signals (defect-mechanisms decision tree) ────────

from interaction import (  # noqa: E402
    ESCAPE_A,
    ESCAPE_B,
    ESCAPE_D,
    ESCAPE_REVIEW,
    attribute_escape,
    neighbor_hit_density,
    neighbor_label_entropy,
)


def _two_clusters(n_per=10, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.01, (n_per, 4)) + np.array([1, 0, 0, 0])
    b = rng.normal(0.0, 0.01, (n_per, 4)) + np.array([0, 1, 0, 0])
    return np.vstack([a, b])


def test_hit_density_dense_vs_isolated():
    emb = _two_clusters(10)
    # a point inside a tight cluster has many neighbours within a small radius
    assert neighbor_hit_density(emb, 0, radius=0.05) >= 8
    isolated = np.vstack([emb, np.array([[0.0, 0.0, 5.0, 0.0]])])
    assert neighbor_hit_density(isolated, len(isolated) - 1, radius=0.05) == 0


def test_label_entropy_pure_vs_mixed():
    emb = _two_clusters(10)
    labels = ["A"] * 10 + ["B"] * 10
    assert neighbor_label_entropy(emb, labels, 0, k=5) == pytest.approx(0.0, abs=1e-9)
    # a point whose neighbourhood is a 50/50 label mix → entropy near 1
    mixed_labels = (["A", "B"] * 5) + (["A", "B"] * 5)
    assert neighbor_label_entropy(emb, mixed_labels, 0, k=8) > 0.8


def test_attribute_escape_branches():
    # A: high label entropy
    a = attribute_escape(hit_density=10, label_entropy=0.9, outlier_pct=0.5)
    assert a["class"] == ESCAPE_A
    # D: no neighbours + outlier
    d = attribute_escape(hit_density=0, label_entropy=0.1, outlier_pct=0.95)
    assert d["class"] == ESCAPE_D
    # B: sparse neighbours
    b = attribute_escape(hit_density=1, label_entropy=0.1, outlier_pct=0.5)
    assert b["class"] == ESCAPE_B
    # review: everything looks fine from embeddings → defer to human
    r = attribute_escape(hit_density=30, label_entropy=0.1, outlier_pct=0.4)
    assert r["class"] == ESCAPE_REVIEW
    assert all("confidence" in x and "reasons" in x for x in (a, d, b, r))


def test_attribute_escape_score_gives_C():
    from interaction import ESCAPE_C
    c = attribute_escape(hit_density=20, label_entropy=0.1, outlier_pct=0.4,
                         score=0.51, threshold=0.5)
    assert c["class"] == ESCAPE_C  # |0.51-0.5|/0.5 = 2% <= 10%


def test_load_scores_csv(tmp_path):
    from interaction import load_scores_csv
    csv = tmp_path / "scores.csv"
    csv.write_text("filename,score,threshold\na.jpg,0.42,0.5\nb.jpg,0.9,\n",
                   encoding="utf-8")
    out = load_scores_csv(csv)
    assert out["a.jpg"] == (0.42, 0.5)
    assert out["b.jpg"] == (0.9, None)  # blank threshold → None
    assert load_scores_csv(tmp_path / "nope.csv") == {}


# ── F6 diversity selection (farthest-point sampling) ────────────────────

from interaction import farthest_point_sampling  # noqa: E402


def test_fps_picks_one_from_each_cluster():
    emb = _two_clusters(10)  # cluster A rows 0-9, B rows 10-19
    picks = farthest_point_sampling(emb, 2)
    assert len(picks) == 2
    a_picked = any(i < 10 for i in picks)
    b_picked = any(i >= 10 for i in picks)
    assert a_picked and b_picked  # diverse → spans both clusters


def test_fps_seeds_cover_gaps():
    emb = _two_clusters(10)
    # seed the whole of cluster A → the first diverse pick must be from B
    picks = farthest_point_sampling(emb, 1, seed_indices=list(range(10)))
    assert picks and picks[0] >= 10
    assert all(i not in range(10) for i in picks)  # never returns a seed


def test_fps_clamps_and_degenerate():
    emb = _two_clusters(3)
    assert len(farthest_point_sampling(emb, 99)) == 6   # clamped to N
    assert farthest_point_sampling(emb, 0) == []
    assert farthest_point_sampling(np.zeros((0, 4)), 5) == []


def test_fps_deterministic():
    emb = _two_clusters(8, seed=3)
    assert farthest_point_sampling(emb, 4) == farthest_point_sampling(emb, 4)


# ── §3 gray-zone purgatory helpers ──────────────────────────────────────

from interaction import (  # noqa: E402
    gray_decision_csv, gray_zone_summary, nearest_anchor, select_gray_zone)


def test_gray_zone_summary_backlog_and_buckets():
    # 10 scores: 3 high(≥0.6), 2 mid(0.3–0.6), 2 low(>0–0.3), 3 clean(0)
    s = np.array([0.9, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0])
    out = gray_zone_summary(s, thr=0.5, hi=0.6, lo=0.3)
    assert out["n_total"] == 10
    assert out["n_gray"] == 4 and out["pct_gray"] == 40.0  # ≥0.5
    assert out["high"] == 3 and out["mid"] == 2
    assert out["low"] == 2 and out["clean"] == 3
    assert gray_zone_summary(np.array([]))["n_total"] == 0


def test_select_gray_zone_most_ambiguous_first():
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.0])
    assert select_gray_zone(scores, 3) == [1, 3, 2]  # highest disagreement first
    assert select_gray_zone(scores, 99) == [1, 3, 2, 0, 4]  # clamped
    assert select_gray_zone(np.zeros(0), 3) == []
    assert select_gray_zone(scores, 0) == []


def test_nearest_anchor_picks_closest():
    emb = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]])
    # item 0 is closest to anchor 1 (aligned), not anchor 2 (orthogonal)
    a, d = nearest_anchor(emb, 0, [1, 2])
    assert a == 1 and d < 0.05
    # self is excluded from anchors
    a2, _ = nearest_anchor(emb, 0, [0, 2])
    assert a2 == 2
    assert nearest_anchor(emb, 0, []) == (None, float("inf"))


def test_gray_decision_csv_contract():
    csv_text = gray_decision_csv([
        {"path": "x.jpg", "soft_label": "瑕疵", "confidence": "0.67",
         "anchor": "明確是#3", "reason": "邊緣有微小凹陷",
         "proposer": "標註A", "approver": "QA", "status": "approved"},
    ])
    lines = csv_text.splitlines()
    assert lines[0] == "path,soft_label,confidence,anchor,reason,proposer,approver,status"
    assert "邊緣有微小凹陷" in lines[1] and lines[1].endswith("approved")
    assert gray_decision_csv([]).strip() == \
        "path,soft_label,confidence,anchor,reason,proposer,approver,status"


# ── three-signal root-cause diagnosis (H1–H5) — BDD scenarios as tests ──

from interaction import (  # noqa: E402
    CAUSE_H1, CAUSE_H2, CAUSE_H3, CAUSE_H4, CAUSE_H5, diagnose_root_cause,
)


def test_diag_h1_sparse_consistent_high_entropy_is_coverage_gap():
    # 稀疏 + 人類一致 + 高熵 → H1 覆蓋缺口, 補資料有效
    r = diagnose_root_cause(s1_consistency=0.95, s2_density=1, s3_entropy=0.8)
    assert r["cause"] == CAUSE_H1
    assert r["add_data"].startswith("有效")
    assert r["s2_sparse"] and r["s3_high"] and not r["s1_low"]


def test_diag_h4_dense_consistent_high_entropy_is_capacity_limit():
    # 密集 + 人類一致 + 高熵 → H4 容量限制, 補資料無效（換架構）
    r = diagnose_root_cause(s1_consistency=0.95, s2_density=30, s3_entropy=0.8)
    assert r["cause"] == CAUSE_H4 and "有限" in r["add_data"]


def test_diag_h3_dense_confident_wrong_is_label_noise():
    # 密集 + 篤定卻錯（低熵）→ H3 標籤雜訊
    r = diagnose_root_cause(s1_consistency=0.95, s2_density=30, s3_entropy=0.1)
    assert r["cause"] == CAUSE_H3


def test_diag_h5_sparse_confident_wrong_is_ood():
    # 稀疏 + 篤定卻錯 → H5 OOD, 需收新樣態
    r = diagnose_root_cause(s1_consistency=0.95, s2_density=0, s3_entropy=0.1)
    assert r["cause"] == CAUSE_H5 and "新樣態" in r["add_data"]


def test_diag_h2_human_inconsistent_overrides_everything():
    # 人類不一致 → H2 定義歧義（不論 S2/S3）, 補資料不收斂
    for d, e in [(0, 0.9), (30, 0.1), (5, 0.5)]:
        r = diagnose_root_cause(s1_consistency=0.4, s2_density=d, s3_entropy=e)
        assert r["cause"] == CAUSE_H2 and r["s1_low"]
        assert "無效" in r["add_data"]


def test_diag_threshold_boundaries_configurable():
    # at exactly the thresholds: density==thr is "sparse", entropy==thr is "high"
    r = diagnose_root_cause(0.95, 3, 0.5, density_thr=3, entropy_thr=0.5)
    assert r["s2_sparse"] and r["s3_high"] and r["cause"] == CAUSE_H1
    # raise the consistency bar so a 0.8 score now counts as inconsistent
    r2 = diagnose_root_cause(0.8, 1, 0.8, consistency_thr=0.9)
    assert r2["cause"] == CAUSE_H2


# ── 桶① physical-detectability gate (H0) in front of H1–H5 ──────────────

def test_diag_h0_signal_none_overrides_to_physical_ceiling():
    from interaction import CAUSE_H0
    from signal_strength import SIGNAL_NONE
    # signals that would otherwise read H1 覆蓋缺口 (sparse + consistent +
    # high entropy) — but the defect's signal is not in the pixels at all.
    r = diagnose_root_cause(0.95, 1, 0.8, signal_level=SIGNAL_NONE)
    assert r["cause"] == CAUSE_H0
    assert "無效" in r["add_data"]  # adding data cannot help 桶①
    assert r["signal_level"] == SIGNAL_NONE


def test_diag_h1_caveat_flags_unverified_detectability():
    from signal_strength import SIGNAL_OBVIOUS, SIGNAL_SUSPECT
    # signal not measured → H1 still fires, but the verdict is flagged as
    # resting on an unverified 桶① assumption.
    r = diagnose_root_cause(0.95, 1, 0.8)
    assert r["cause"] == CAUSE_H1 and r["caveat"]
    # weak/borderline signal → still caveated.
    r_s = diagnose_root_cause(0.95, 1, 0.8, signal_level=SIGNAL_SUSPECT)
    assert r_s["cause"] == CAUSE_H1 and r_s["caveat"]
    # signal confirmed clearly present → advice trustworthy, no caveat.
    r_ok = diagnose_root_cause(0.95, 1, 0.8, signal_level=SIGNAL_OBVIOUS)
    assert r_ok["cause"] == CAUSE_H1 and not r_ok["caveat"]


# ── curation log (time dimension: selection + reason) ───────────────────

from interaction import curation_log_csv, match_shas_to_indices  # noqa: E402


def test_curation_log_csv_contract():
    entries = [
        {"ts": "2026-06-13T10:00", "reason": "可疑灰帶一批", "n": 2,
         "items": [{"filename": "a.jpg", "sha256": "aa"},
                   {"filename": "b.jpg", "sha256": "bb"}]},
    ]
    lines = curation_log_csv(entries).splitlines()
    assert lines[0] == "ts,reason,n,filenames,sha256s"
    assert "可疑灰帶一批" in lines[1]
    assert "a.jpg b.jpg" in lines[1] and "aa bb" in lines[1]
    assert curation_log_csv([]).strip() == "ts,reason,n,filenames,sha256s"


def test_match_shas_to_indices_reselect_and_degrade():
    sha_to_index = {"aa": 0, "bb": 1, "cc": 2}
    # all present → re-select in order
    assert match_shas_to_indices(["bb", "aa"], sha_to_index) == [1, 0]
    # a hash from another dataset is skipped, not an error
    assert match_shas_to_indices(["aa", "zz", "cc"], sha_to_index) == [0, 2]
    # de-duplicated
    assert match_shas_to_indices(["aa", "aa"], sha_to_index) == [0]
    assert match_shas_to_indices([], sha_to_index) == []
