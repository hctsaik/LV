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
