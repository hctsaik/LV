"""Unit tests for the embedding-space coverage / gap-filling functions in
scripts/interaction.py (the 嵌入覆蓋圖 view): raw-space sparsity scoring,
new-folder gap-filler ranking, 1-NN provisional labelling, the quiz handoff
contract, and the H1–H5 honesty gate over sparse points."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from interaction import (
    CAUSE_H2,
    CAUSE_H5,
    bbox_to_pixels,
    candidates_to_quiz_records,
    crop_bbox,
    cross_class_nn_pairs,
    diagnose_sparse_points,
    discover_yolo_objects,
    nearest_labels,
    parse_yolo_boxes,
    rank_gap_fillers,
    reference_coverage,
    sparsity_scores,
)


def _two_clusters(n_per: int = 12, seed: int = 0) -> np.ndarray:
    """Two tight, well-separated clusters in 4-D (cosine-separable)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.01, size=(n_per, 4)) + np.array([1, 0, 0, 0])
    b = rng.normal(0.0, 0.01, size=(n_per, 4)) + np.array([0, 1, 0, 0])
    return np.vstack([a, b])


# ── sparsity_scores ──────────────────────────────────────────────────────

def test_sparsity_isolated_point_scores_highest():
    emb = _two_clusters(n_per=10)
    # an isolated point far from both clusters
    lonely = np.array([[0.0, 0.0, 1.0, 0.0]])
    full = np.vstack([emb, lonely])
    scores = sparsity_scores(full, k=5)
    assert int(np.argmax(scores)) == len(full) - 1  # the lonely row is sparsest
    assert scores.shape == (len(full),)


def test_sparsity_degenerate_sizes():
    assert sparsity_scores(np.zeros((0, 4))).shape == (0,)
    assert sparsity_scores(np.ones((1, 4))).tolist() == [0.0]


# ── rank_gap_fillers ─────────────────────────────────────────────────────

def test_gap_fillers_prefers_far_candidate():
    dataset = _two_clusters(n_per=12)
    # one candidate sits inside cluster A (covered), one in an empty region
    covered = np.array([1.0, 0.01, 0.0, 0.0])
    gap = np.array([0.0, 0.0, 1.0, 0.0])
    cand = np.vstack([covered, gap])
    idxs, scores = rank_gap_fillers(cand, dataset, k=5)
    assert idxs[0] == 1                      # the empty-region candidate ranks first
    assert scores[0] > scores[1]             # and scores descending
    assert scores == sorted(scores, reverse=True)


def test_gap_fillers_min_distance_filters_covered():
    dataset = _two_clusters(n_per=12)
    covered = np.array([1.0, 0.0, 0.0, 0.0])  # duplicate of cluster-A centre
    gap = np.array([0.0, 0.0, 1.0, 0.0])
    cand = np.vstack([covered, gap])
    idxs, scores = rank_gap_fillers(cand, dataset, k=5, min_distance=0.5)
    assert idxs == [1]                       # covered candidate dropped
    idxs_top, _ = rank_gap_fillers(cand, dataset, k=5, top=1)
    assert len(idxs_top) == 1


def test_gap_fillers_empty_inputs():
    assert rank_gap_fillers(np.zeros((0, 4)), _two_clusters()) == ([], [])
    assert rank_gap_fillers(_two_clusters(), np.zeros((0, 4))) == ([], [])


# ── nearest_labels ───────────────────────────────────────────────────────

def test_nearest_labels_transfers_cluster_label():
    ref = _two_clusters(n_per=10)
    labels = ["A"] * 10 + ["B"] * 10
    queries = np.array([[1.0, 0.02, 0.0, 0.0],   # near A
                        [0.0, 1.0, 0.02, 0.0]])  # near B
    assert nearest_labels(queries, ref, labels) == ["A", "B"]


def test_nearest_labels_empty_reference_blank():
    q = np.ones((3, 4))
    assert nearest_labels(q, np.zeros((0, 4)), []) == ["", "", ""]
    assert nearest_labels(np.zeros((0, 4)), q, ["x"]) == []


# ── candidates_to_quiz_records (quiz handoff contract) ───────────────────

def test_candidates_to_quiz_records_carries_provenance():
    recs = [{"path": "a.jpg"}, {"path": "b.jpg", "split": "pool"}]
    out = candidates_to_quiz_records(recs, ["dog", "cat"], gap_scores=[0.4, 0.2])
    assert [r["label"] for r in out] == ["dog", "cat"]
    assert all(r["provisional"] and r["source"] == "gap_filler" for r in out)
    assert out[0]["split"] == "candidate" and out[1]["split"] == "pool"
    assert out[0]["gap_score"] == pytest.approx(0.4)


def test_candidates_to_quiz_records_tolerates_missing_scores_labels():
    recs = [{"path": "a.jpg"}]
    out = candidates_to_quiz_records(recs, [])  # no labels, no scores
    assert out[0]["label"] == "" and out[0]["gap_score"] is None


def test_quiz_handoff_records_consumable_by_build_quiz():
    """The send-to-quiz contract: candidates_to_quiz_records output must be a
    valid build_quiz input (label present, indices in range)."""
    from quiz import build_quiz

    recs = [{"path": f"{i}.jpg"} for i in range(8)]
    labels = ["dog", "cat"] * 4               # provisional, ≥2 classes
    scores = list(np.linspace(1.0, 0.0, 8))
    qrecs = candidates_to_quiz_records(recs, labels, scores)
    quiz = build_quiz(qrecs, np.asarray(scores), n_questions=6, seed=0)
    assert quiz["questions"]
    assert all(0 <= q["record_idx"] < len(qrecs) for q in quiz["questions"])


# ── diagnose_sparse_points (H1–H5 honesty gate) ──────────────────────────

def test_sparse_confident_region_is_h5_collect_new_regime():
    # an isolated, label-clean point: model is confident (entropy 0) yet the
    # region is empty → a NEW regime (H5), for which collecting new data helps.
    emb = _two_clusters(n_per=10)
    lonely = np.array([[0.0, 0.0, 1.0, 0.0]])
    full = np.vstack([emb, lonely])
    labels = ["A"] * 20 + ["A"]  # single class → entropy 0 (no model hesitation)
    out = diagnose_sparse_points(full, labels, [len(full) - 1],
                                 s1_consistency=0.95, radius=0.05, k=10)
    assert out[0]["cause"] == CAUSE_H5
    assert "新樣態" in out[0]["add_data"]    # collect the new regime
    assert out[0]["s2_density"] == 0          # genuinely sparse


def test_low_human_consistency_is_h2_not_collect():
    emb = _two_clusters(n_per=10)
    labels = ["A"] * 10 + ["B"] * 10
    # low S1 → definition ambiguity dominates regardless of density
    out = diagnose_sparse_points(emb, labels, [0, 1],
                                 s1_consistency=0.3, radius=0.05, k=5)
    assert all(o["cause"] == CAUSE_H2 for o in out)
    assert all("無效" in o["add_data"] for o in out)


# ── object-level coverage: YOLO parsing / cropping / discovery ────────────

def _write_yolo_dataset(tmp_path, boxes_by_image):
    """Lay out a minimal <root>/images + <root>/labels YOLO dataset.
    boxes_by_image: {stem: [(cid, cx, cy, w, h), ...]}. Returns image paths."""
    img_dir = tmp_path / "images"; lbl_dir = tmp_path / "labels"
    img_dir.mkdir(); lbl_dir.mkdir()
    paths = []
    for stem, boxes in boxes_by_image.items():
        Image.new("RGB", (100, 100), (123, 50, 200)).save(img_dir / f"{stem}.jpg")
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in boxes))
        paths.append(img_dir / f"{stem}.jpg")
    return paths


def test_parse_yolo_boxes_skips_garbage(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\nbad line\n1 0.1 0.1 0.0 0.3\n2 0.9 0.9 0.1 0.1 0.99\n")
    boxes = parse_yolo_boxes(f)
    # the 0-width line is dropped; the trailing-confidence line still parses
    assert [b[0] for b in boxes] == [0, 2]
    assert parse_yolo_boxes(tmp_path / "missing.txt") == []


def test_bbox_to_pixels_clamps_and_pads():
    # centred half-size box in a 100×100 image → (25,25,75,75)
    assert bbox_to_pixels(0.5, 0.5, 0.5, 0.5, 100, 100) == (25, 25, 75, 75)
    # a box on the edge stays inside the image after padding
    x0, y0, x1, y1 = bbox_to_pixels(0.95, 0.95, 0.2, 0.2, 100, 100, pad=0.5)
    assert 0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 100


def test_crop_bbox_returns_expected_pixels():
    img = Image.new("RGB", (100, 100))
    crop = crop_bbox(img, 0.5, 0.5, 0.4, 0.6)
    assert crop.size == (40, 60)


def test_discover_yolo_objects_one_record_per_box(tmp_path):
    paths = _write_yolo_dataset(tmp_path, {
        "a": [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.1, 0.1)],
        "b": [(2, 0.5, 0.5, 0.5, 0.5)],
    })
    objs = discover_yolo_objects(paths, class_names=["cat", "dog", "bird"])
    assert len(objs) == 3                       # 2 boxes in a + 1 in b
    assert [o["label"] for o in objs] == ["cat", "dog", "bird"]
    assert objs[0]["obj_index"] == 0 and objs[1]["obj_index"] == 1
    assert objs[0]["image_path"] == paths[0]


def test_discover_yolo_objects_missing_classnames_and_labels(tmp_path):
    paths = _write_yolo_dataset(tmp_path, {"a": [(5, 0.5, 0.5, 0.2, 0.2)]})
    # unknown class id → fallback name; image with no label contributes nothing
    objs = discover_yolo_objects(paths)
    assert objs[0]["label"] == "class_5"
    Image.new("RGB", (10, 10)).save(tmp_path / "images" / "empty.jpg")
    objs2 = discover_yolo_objects([tmp_path / "images" / "empty.jpg"])
    assert objs2 == []


# ── cross_class_nn_pairs (label-disagreement lines on the embedding plot) ──

def test_cross_class_nn_pairs_finds_only_cross_label_neighbours():
    # two tight clusters that are well separated → nearest neighbour is same
    # class for everyone → no cross-class pairs
    emb = _two_clusters(n_per=8)
    labels = ["A"] * 8 + ["B"] * 8
    assert cross_class_nn_pairs(emb, labels, k=1) == []
    # now plant one B-labelled point INSIDE cluster A → its NN is an A point,
    # and that A point's NN may be the intruder → at least one cross pair
    intruder = np.array([[1.0, 0.001, 0.0, 0.0]])
    emb2 = np.vstack([emb, intruder])
    labels2 = labels + ["B"]
    pairs = cross_class_nn_pairs(emb2, labels2, k=1)
    assert pairs                                   # the intruder shows up
    assert all(labels2[i] != labels2[j] for i, j in pairs)


def test_cross_class_nn_pairs_deduped_capped_and_degenerate():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 5))
    labels = [("A" if i % 2 else "B") for i in range(30)]  # alternating → lots
    pairs = cross_class_nn_pairs(emb, labels, k=3, max_pairs=5)
    assert len(pairs) <= 5
    assert all(i < j for i, j in pairs)            # unordered, deduped
    assert len(set(pairs)) == len(pairs)
    assert cross_class_nn_pairs(np.zeros((1, 4)), ["A"]) == []


# ── reference_coverage (嵌入覆蓋圖 關係②：A 相對外部參照 B) ───────────────

def test_reference_coverage_flags_uncovered_region():
    # A = cluster around [1,0,0,0]; B has points near A (covered) + one far away
    a = _two_clusters(n_per=10)[:10]                # one tight A cluster
    covered = np.array([[1.0, 0.01, 0.0, 0.0]])     # sits on top of A
    faraway = np.array([[0.0, 0.0, 1.0, 0.0]])      # a region A never covers
    b = np.vstack([covered, faraway])
    uncovered, recall, d = reference_coverage(a, b, radius=0.05)
    assert uncovered == [1]                          # only the far B point
    assert recall == pytest.approx(0.5)              # 1 of 2 B points covered
    assert d.shape == (2,) and d[1] > d[0]


def test_reference_coverage_full_and_empty():
    a = _two_clusters(n_per=8)
    # B identical to A → fully covered → recall 1.0, none uncovered
    uncovered, recall, _ = reference_coverage(a, a, radius=0.2)
    assert uncovered == [] and recall == pytest.approx(1.0)
    # empty inputs are safe
    unc0, rec0, _ = reference_coverage(np.zeros((0, 4)), a, 0.1)
    assert unc0 == [] and rec0 == 1.0
    unc1, rec1, d1 = reference_coverage(a, np.zeros((0, 4)), 0.1)
    assert unc1 == [] and rec1 == 1.0 and len(d1) == 0
